# SPDX-License-Identifier: MIT
"""Run a target's planned points and record measurements."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from agentx_ci.preflight import validate_hardware, verify_image
from agentx_ci.recipe_resolve import POLICY, ResolveError, SelectionError, fetch_inferencex, selected_points
from agentx_ci.utils import (
    REPO_ROOT,
    begin_attempt,
    pending_results,
    read_json,
    sha256_json,
    utc_now,
    valid_metrics,
    write_json,
)


def metrics_of(aggregate: dict) -> dict:
    request = aggregate["request_metrics"]
    metrics = {
        "tput_per_gpu": request["throughput"]["per_gpu"]["total_tput_tps"],
        "p90_e2e_norm_intvty": request["latency"]["e2e_norm_intvty"]["p90"],
        "num_requests_successful": aggregate.get("num_requests_successful"),
    }
    if not valid_metrics(metrics):
        raise ValueError("Invalid metrics: axes must be non-boolean finite positive numbers")
    return metrics


# Tuning variables come from the hashed plan or the verified image defaults.
ENVIRONMENT_POLICY = "paired-aklr-image-env-v2"
INFRA_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HF_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "XDG_CACHE_HOME",
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "AITER_JIT_DIR",
    "GPU_ARCHS",
    "FLYDSL_RUNTIME_CACHE_DIR",
    "FLYDSL_AUTOTUNE_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "AKLR_KERNEL_CACHE_DIR",
)


def execution_env(image_defaults: dict[str, str] | None = None) -> dict[str, str]:
    """Use verified image defaults only when explicitly requested by the server."""
    return {
        **(image_defaults or {}),
        **{name: os.environ[name] for name in INFRA_ENV if name in os.environ},
    }


def image_environment(inspection: dict) -> dict[str, str]:
    """Parse Config.Env after the inspection has passed image identity verification."""
    config = inspection.get("Config")
    if not isinstance(config, dict) or "Env" not in config:
        raise ValueError("Docker image inspection has no Config.Env")
    entries = config["Env"]
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise ValueError("Docker image Config.Env must be a list or null")
    defaults = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, str) or "=" not in entry:
            raise ValueError(f"Docker image Config.Env entry {index} must be NAME=VALUE")
        name, value = entry.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or "\0" in value:
            raise ValueError(f"Docker image Config.Env entry {index} has an invalid name or value")
        if name in defaults:
            raise ValueError(f"Docker image Config.Env entry {index} repeats a variable name")
        defaults[name] = value
    return defaults


def image_environment_evidence(defaults: dict[str, str]) -> dict:
    """Record image defaults without copying live infrastructure credentials."""
    protected_name = re.compile(r"token|secret|password|passwd|credential|auth|key|cookie|proxy", re.I)
    protected_url = re.compile(
        r"://[^/\s]*@|[?&][^=]*(?:token|secret|password|credential|signature|auth|key)[^=]*=", re.I
    )
    return {
        "policy": ENVIRONMENT_POLICY,
        "scope": "server",
        "source": "docker-image-inspect.Config.Env",
        "defaults_sha256": sha256_json(defaults),
        "defaults": {
            name: "<redacted>" if protected_name.search(name) or protected_url.search(value) else value
            for name, value in sorted(defaults.items())
        },
    }


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status < 400
    except (urllib.error.URLError, OSError):
        return False


def wait_ready(base_url: str, server: subprocess.Popen, timeout_s: float, interval_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with code {server.returncode} before ready")
        if http_ok(f"{base_url}/health") and http_ok(f"{base_url}/v1/models"):
            return
        time.sleep(min(interval_s, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"server not ready after {timeout_s:.0f} s")


def stop(server: subprocess.Popen, grace_s: float = 120) -> None:
    """Stop the process group even if its leader has exited."""
    deadline = time.monotonic() + grace_s
    try:
        os.killpg(server.pid, signal.SIGTERM)
        while time.monotonic() < deadline:
            server.poll()
            os.killpg(server.pid, 0)
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    server.wait()


def run_client(command: list[str], *, timeout: float, **kwargs) -> int:
    """Stop the client's process group on exit or timeout."""
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        return process.wait(timeout=timeout)
    finally:
        stop(process)


def validate_runtime(points: list[dict]) -> None:
    for point in points:
        runtime = point.get("runtime", {})
        for field in ("ready_timeout_s", "point_timeout_s", "poll_s"):
            value = runtime.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise RuntimeError(f"{point['arm']}: upstream runtime.{field} needs a finite positive value")
        port = runtime.get("port")
        if port is not None and (isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536):
            raise RuntimeError(f"invalid upstream server port {port!r}")
        dram = point["client_env"].get("TOTAL_CPU_DRAM_GB")
        if not isinstance(dram, str) or not dram.isdigit():
            raise RuntimeError(f"{point['arm']}: upstream TOTAL_CPU_DRAM_GB is missing or invalid")
        env = point["client_env"]
        for name in ("TP", "PP_SIZE", "PCP_SIZE", "DCP_SIZE", "EP_SIZE"):
            value = env.get(name)
            if not isinstance(value, str) or not value.isdigit() or int(value) <= 0:
                raise RuntimeError(f"{point['arm']}: invalid or missing {name}; regenerate the plan")
        if int(env["TP"]) * int(env["PP_SIZE"]) * int(env["PCP_SIZE"]) != point["gpus"]:
            raise RuntimeError(f"{point['arm']}: client and launch GPU counts differ")
        if env.get("DP_ATTENTION") not in ("true", "false"):
            raise RuntimeError(f"{point['arm']}: invalid or missing DP_ATTENTION; regenerate the plan")


def available_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def port_free(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


PREPARE_TRACES = """
import json, os, subprocess, sys
from pathlib import Path
from infx.bench.agentic.venv import Runtime, bootstrap
runtime_dir, dataset, revision = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
runtime = Runtime(runtime_dir)
if bootstrap(runtime):
    sys.exit("AgentX runtime bootstrap failed")
offline = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
env = {k: v for k, v in os.environ.items() if k not in offline}
subprocess.run([str(runtime.hf), "download", "--repo-type", "dataset", dataset, "--revision", revision], env=env, check=True)
ref = Path(env['HF_HUB_CACHE']) / ('datasets--' + dataset.replace('/', '--')) / 'refs' / 'main'
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(revision)
load = '''
import json, sys
from datasets import load_dataset
revision = {"revision": sys.argv[2]} if len(sys.argv) > 2 else {}
data = load_dataset(sys.argv[1], split="train", streaming=False, trust_remote_code=False, **revision)
print(json.dumps({"fingerprint": data._fingerprint, "files": data.cache_files, "rows": data.num_rows}))
'''
prepared = subprocess.check_output([str(runtime.python), "-c", load, dataset, revision], env=env)
offline_env = {**env, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
subprocess.run([str(runtime.hf), "download", "--repo-type", "dataset", dataset],
               env=offline_env, check=True)
replayed = subprocess.check_output([str(runtime.python), "-c", load, dataset], env=offline_env)
if json.loads(prepared) != json.loads(replayed):
    sys.exit("offline trace loading differs from the pinned preparation")
"""


def trace_datasets_cache(host: dict, dataset: str, revision: str) -> Path:
    """Isolate datasets' fallback cache by trace revision."""
    return host["datasets_cache"] / sha256_json([dataset, revision])


def prepare_traces(host: dict, points: list[dict], log_path: Path) -> None:
    """Prepare traces and verify offline loading."""
    sources = {(p["trace"]["repo"], p["trace"]["revision"]) for p in points}
    if len(sources) != len({repo for repo, _ in sources}):
        raise RuntimeError("a run cannot use different revisions of the same trace dataset")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for dataset, revision in sorted(sources):
        with log_path.open("a") as log:
            code = run_client(
                [sys.executable, "-c", PREPARE_TRACES, str(host["agentx_runtime_dir"]), dataset, revision],
                cwd=host["inferencex_e2e"],
                env={
                    **execution_env(),
                    "PYTHONPATH": str(host["inferencex_e2e"]),
                    "HF_HUB_CACHE": str(host["trace_hub"]),
                    "HF_DATASETS_CACHE": str(trace_datasets_cache(host, dataset, revision)),
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(p["runtime"]["point_timeout_s"] for p in points),
            )
        if code:
            raise RuntimeError(f"preparing the AgentX traces failed; see {log_path}")


def run_live(point: dict, out: Path, host: dict) -> dict:
    runtime = point["runtime"]
    port = runtime["port"] if runtime["port"] is not None else available_port()
    base_url = f"http://127.0.0.1:{port}"
    if not port_free(port):
        raise RuntimeError(f"port {port} is already in use")
    started = time.monotonic()
    budget = runtime["point_timeout_s"]
    placeholders = {
        "${AGENTX_MODEL_PATH}": host["model_path"],
        "${AGENTX_PORT}": str(port),
    }
    command = [placeholders.get(arg, arg) for arg in point["launch"]["command"]]
    with (out / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            env={**execution_env(host["server_image_env"]), **point["launch"]["env"]},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_ready(base_url, server, min(runtime["ready_timeout_s"], budget), runtime["poll_s"])
        result_name = point["exp_name"]
        client_env = {
            **execution_env(),
            **point["client_env"],
            **host["client_env_overrides"],
            "PORT": str(port),
            "SERVED_MODEL_NAME": point["served_model_name"],
            "TOKENIZER": host["tokenizer"],
            "AIPERF_SERVER_METRICS_URLS": f"{base_url}/metrics",
            "AIPERF_RUNTIME_DIR": str(host["agentx_runtime_dir"]),
            "HF_DATASETS_CACHE": str(
                trace_datasets_cache(host, point["trace"]["repo"], point["trace"]["revision"])
            ),
            "HF_HUB_CACHE": str(host["trace_hub"]),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "RESULT_DIR": str(out / "agentic"),
            "RESULT_FILENAME": result_name,
            "AGENTIC_OUTPUT_DIR": str(out),
            "PYTHONPATH": str(host["inferencex_e2e"]),
        }
        with (out / "client.log").open("w") as log:
            code = run_client(
                [str(host["agentx_runtime_dir"] / "venv/bin/python"), "-m", "infx.bench", "agentic"],
                cwd=host["inferencex_e2e"],
                env=client_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(0, budget - (time.monotonic() - started)),
            )
        if code:
            raise RuntimeError(f"AgentX client exited with code {code}")
        return read_json(out / f"{result_name}.json")
    finally:
        stop(server)


def run_point(point: dict, out: Path, host: dict) -> dict:
    record = {
        "arm": point["arm"],
        "conc": point["conc"],
        "exp_name": point["exp_name"],
        "serve_spec_sha256": point["hashes"]["serve_spec_sha256"],
        "status": "ok",
        "error": None,
        "started_at": utc_now(),
    }
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    try:
        aggregate = run_live(point, out, host)
        record.update(metrics_of(aggregate))
        record["raw_data"] = aggregate
    except (RuntimeError, OSError, KeyError, ValueError, TypeError, subprocess.TimeoutExpired) as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}"[:500])
    record["elapsed_s"] = round(time.monotonic() - began, 1)
    return record


def aklr_sha() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() or "unknown"


def source_dirty() -> bool:
    """Exclude unknown source states from baseline promotion."""
    if "AKLR_SOURCE_DIRTY" in os.environ:
        return os.environ["AKLR_SOURCE_DIRTY"] != "false"
    done = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode != 0 or bool(done.stdout.strip())


def observed_hardware() -> dict:
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPUs are required")
    count = torch.cuda.device_count()
    return {
        "devices": [
            {"name": p.name, "arch": p.gcnArchName.split(":", 1)[0], "total_memory": p.total_memory}
            for p in (torch.cuda.get_device_properties(i) for i in range(count))
        ],
        "peer_access": [
            [i == j or torch.cuda.can_device_access_peer(i, j) for j in range(count)] for i in range(count)
        ],
    }


def execution_identity(serving: dict) -> tuple[dict, dict[str, str]]:
    """Return public identity and server defaults from one verified inspection."""
    path = os.environ.get("AGENTX_EXECUTION_FILE")
    if not path:
        raise RuntimeError("execution image is unverified; start the run with the Docker sweep launcher")
    inspection = read_json(path)
    if not isinstance(inspection, dict):
        raise ValueError("Docker image inspection must be an object")
    identity = verify_image(serving["image"]["run"], inspection)
    defaults = image_environment(inspection)
    identity["server_environment"] = image_environment_evidence(defaults)
    return identity, defaults


def client_packages(host: dict) -> dict:
    script = "import json,importlib.metadata as m; print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"
    return json.loads(
        subprocess.check_output(
            [str(host["agentx_runtime_dir"] / "venv/bin/python"), "-c", script],
            text=True,
            timeout=60,
            env=execution_env(),
        )
    )


def server_packages() -> dict:
    try:
        return {
            name: importlib.metadata.version(name)
            for name in ("sglang", "torch", "triton", "amd-aiter", "flydsl")
        }
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"serving package identity is unavailable: {error}") from error


def host_settings(args: argparse.Namespace, serving: dict) -> dict:
    weights = serving["weights"]
    trace_hub = args.work / "hf-traces-hub"
    trace_hub.mkdir(parents=True, exist_ok=True)
    model_path, tokenizer = args.model_path, args.model_path
    if model_path is None:
        hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
        model_dir = hub / ("models--" + weights["repo"].replace("/", "--"))
        model_path = model_dir / "snapshots" / weights["revision"]
        # Offline tokenizers require refs/main at the pinned revision.
        tokenizer = weights["repo"]
        main = model_dir / "refs" / "main"
        if not main.is_file():
            raise RuntimeError(
                f"{main} is missing; prepare refs/main at the pinned {weights['revision']} "
                "for the AgentX client's offline tokenizer"
            )
        main_revision = main.read_text()
        if main_revision != weights["revision"]:
            raise RuntimeError(
                f"{main} must contain exactly the pinned revision {weights['revision']}, "
                "with no trailing newline or whitespace; the AgentX client's offline tokenizer "
                "uses the ref contents verbatim"
            )
        # Reuse pinned weights in the writable trace hub.
        model_link = trace_hub / model_dir.name
        if model_link.is_symlink() and model_link.resolve() != model_dir.resolve():
            model_link.unlink()
        if not model_link.exists():
            model_link.symlink_to(model_dir.resolve(), target_is_directory=True)
        if model_link.resolve() != model_dir.resolve():
            raise RuntimeError(f"trace hub contains a conflicting model cache: {model_link}")
    if not Path(model_path).is_dir():
        raise RuntimeError(f"model snapshot {model_path} not found; prefill the HF hub cache")
    inferencex = serving["provenance"]["inferencex"]
    inferencex_dir = fetch_inferencex(inferencex["repo"], inferencex["sha"], args.work / "inferencex")
    overrides = {"DURATION": str(args.duration)} if args.duration else {}
    return {
        "model_path": str(model_path),
        "tokenizer": str(tokenizer),
        "inferencex_e2e": inferencex_dir / "inferencex-e2e",
        "agentx_runtime_dir": args.work / "agentx-runtime",
        "datasets_cache": args.work / "hf-datasets",
        "trace_hub": trace_hub,
        "client_env_overrides": overrides,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m agentx_ci.runtime", description=__doc__)
    parser.add_argument("--serving", type=Path, required=True, help="serving.json for one model target")
    parser.add_argument("--out", type=Path, required=True, help="directory for results.json")
    parser.add_argument("--work", type=Path, default=Path("/tmp/agentx-work"))
    parser.add_argument("--arms", help="comma-separated arms, or all; defaults to the plan selection")
    parser.add_argument("--point-policy", choices=("all", POLICY), help="Must match the frozen serving plan")
    parser.add_argument(
        "--points",
        help="comma-separated ARM:CONC using the generated ids in serving.json (smoke runs only)",
    )
    parser.add_argument("--model-path", help="default: the pinned revision in $HF_HUB_CACHE")
    parser.add_argument("--duration", type=int, help="override DURATION (smoke runs only)")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    args = parser.parse_args(argv)
    args.out, args.work = args.out.resolve(), args.work.resolve()
    attempt_id = begin_attempt(args.out)
    if args.model_path:
        args.model_path = str(Path(args.model_path).resolve())
    serving = read_json(args.serving)
    selection = serving.get("selection", {})
    policy = selection.get("policy", "all")
    if args.point_policy is not None and args.point_policy != policy:
        parser.error("--point-policy must match the plan; regenerate it with plan to change it")
    if policy == POLICY and (args.points is not None or args.arms is not None):
        parser.error("Pareto selection is fixed by the plan; --arms and --points are not allowed")
    args.arms = args.arms if args.arms is not None else selection.get("arms", "all")
    args.points = (
        args.points if args.points is not None else (selection.get("points") if policy == "all" else None)
    )
    args.duration = args.duration if args.duration is not None else selection.get("duration")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    try:
        effective = (
            {**serving, "selection": {**selection, "arms": args.arms, "points": args.points}}
            if policy == "all"
            else serving
        )
        points = selected_points(effective)
    except SelectionError as error:
        parser.error(str(error))
    try:
        results = pending_results(
            serving,
            points,
            {
                "id": str(args.run_id),
                "attempt_id": attempt_id,
                "aklr_sha": aklr_sha(),
                "source_dirty": source_dirty(),
                "host": socket.gethostname(),
                "arms_requested": args.arms,
                "points_requested": args.points,
                "duration_override_s": args.duration,
                "model_path_override": args.model_path,
            },
        )
    except (ValueError, KeyError, TypeError) as error:
        results = read_json(args.out / "results.json")
        results["run"].update(error=str(error), finished_at=utc_now())
        write_json(args.out / "results.json", results)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    records = results["points"]
    write_json(args.out / "results.json", results)
    if policy == POLICY:
        write_json(args.out / "selection.json", selection)
    try:
        validate_runtime(points)
        execution, image_defaults = execution_identity(serving)
        hardware = observed_hardware()
        validate_hardware(hardware, serving["hardware"])
        if any(p["gpus"] > len(hardware["devices"]) for p in points):
            raise ValueError("planned point requires more GPUs than are visible")
        for target in (results, results["control"]):
            target["run"].update(execution=execution, hardware=hardware)
        host = host_settings(args, serving)
        host["server_image_env"] = image_defaults
        print(f"[{utc_now()}] preparing the AgentX traces", flush=True)
        prepare_traces(host, points, args.out / "prepare_traces.log")
        packages = client_packages(host)
        serving_packages = server_packages()
        for target in (results, results["control"]):
            target["run"]["client_packages"] = packages
            target["run"]["server_packages"] = serving_packages
    except (
        RuntimeError,
        ResolveError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        results["run"].update(error=str(error), finished_at=utc_now())
        write_json(args.out / "results.json", results)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for mode, target in (("disabled", results["control"]), ("enabled", results)):
        target["run"]["started_at"] = utc_now()
        measurements = {}
        for planned in points:
            point = copy.deepcopy(planned)
            point["launch"]["env"].update(
                AKLR_SGLANG_PLUGIN="1" if mode == "enabled" else "0",
                AKLR_DISABLE="0" if mode == "enabled" else "1",
            )
            measurement = {
                "environment_policy": ENVIRONMENT_POLICY,
                "server_image_env_sha256": execution["server_environment"]["defaults_sha256"],
                "weights": serving["weights"],
                "launch": point["launch"],
                "client_env": {**point["client_env"], **host["client_env_overrides"]},
                "runtime": point["runtime"],
                "model_path_override": args.model_path,
                "trace": point["trace"],
                "trace_policy": "revision-isolated-v1",
            }
            measurements[point["exp_name"]] = measurement
            print(f"[{utc_now()}] AKLR {mode} · {point['arm']} c{point['conc']}", flush=True)
            record = run_point(
                point, args.out / "attempts" / attempt_id / mode / "points" / point["exp_name"], host
            )
            record.update(measurement=measurement, measurement_sha256=sha256_json(measurement))
            print(f"  -> {record['status']} {record.get('error') or ''}", flush=True)
            target["points"].append(record)
            target["run"]["finished_at"] = utc_now()
            write_json(args.out / "results.json", results)
        for arm, spec in target["serving"]["arms"].items():
            spec["arm_measurement_sha256"] = sha256_json(
                {p["conc"]: measurements[p["exp_name"]] for p in points if p["arm"] == arm}
            )
        target["run"]["completed"] = True
        write_json(args.out / "results.json", results)
    all_records = results["control"]["points"] + records
    failed = sum(r["status"] != "ok" for r in all_records)
    print(
        f"{len(all_records) - failed}/{len(all_records)} paired points ok; wrote {args.out / 'results.json'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
