# SPDX-License-Identifier: MIT
"""Run a target's planned points and record measurements."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
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

from agentx_ci.inferencex_client import FetchError, fetch_inferencex
from agentx_ci.preflight import validate_hardware, verify_image
from agentx_ci.serving_config import (
    EXP_NAME,
    POLICY,
    RecipeError,
    SelectionError,
    check_point_runtime,
    point_id,
    selected_points,
)
from agentx_ci.utils import (
    INTERACTIVITY_METRIC,
    REPO_ROOT,
    begin_attempt,
    full_response_interactivity,
    pending_results,
    plugin_switch,
    points_framework,
    read_json,
    sha256_json,
    utc_now,
    valid_metrics,
    write_json,
)


def metrics_of(aggregate: dict, gpus: int) -> dict:
    if aggregate.get("num_gpus") != gpus:
        raise ValueError(
            f"AgentX client counted {aggregate.get('num_gpus')!r} GPUs, the point uses {gpus}; "
            "throughput per GPU would be wrong"
        )
    request = aggregate["request_metrics"]
    metrics = {
        "tput_per_gpu": request["throughput"]["per_gpu"]["total_tput_tps"],
        INTERACTIVITY_METRIC: full_response_interactivity(aggregate),
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
    """Publish image-default names and their complete digest without exposing values."""
    return {
        "policy": ENVIRONMENT_POLICY,
        "scope": "server",
        "source": "docker-image-inspect.Config.Env",
        "defaults_sha256": sha256_json(defaults),
        "defaults": {name: "<redacted>" for name in sorted(defaults)},
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


class CleanupError(RuntimeError):
    """A child process could not be stopped safely; the sweep must abort."""


def stop(server: subprocess.Popen, grace_s: float = 120) -> None:
    """Stop the process group even if its leader has exited."""
    deadline = time.monotonic() + grace_s
    try:
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
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise CleanupError(
            f"Could not stop process group {server.pid}: {type(error).__name__}: {error}"
        ) from error


def run_client(command: list[str], *, timeout: float, **kwargs) -> int:
    """Stop the client's process group on exit or timeout."""
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        return process.wait(timeout=timeout)
    finally:
        stop(process)


def validate_runtime(points: list[dict]) -> None:
    names = [point.get("exp_name") for point in points]
    if not all(isinstance(name, str) and EXP_NAME.fullmatch(name) for name in names):
        raise RuntimeError("every point needs an exp_name that is a single safe path component")
    if len(names) != len(set(names)):
        raise RuntimeError("point exp_name values must be unique")
    for point in points:
        try:
            check_point_runtime(point_id(point), point)
        except RecipeError as error:
            raise RuntimeError(str(error)) from error


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


RESOLVE_LOADERS = """
import json, sys
from infx.bench.agentic import traces
print(json.dumps({loader: traces.resolve("", loader)[1] for loader in json.loads(sys.argv[1])}))
"""


def verify_trace_loaders(host: dict, points: list[dict]) -> None:
    """Require the pinned client to replay, for each point's loader, the dataset its trace records."""
    loaders = sorted({p["client_env"]["WEKA_LOADER_OVERRIDE"] for p in points})
    done = subprocess.run(
        [sys.executable, "-c", RESOLVE_LOADERS, json.dumps(loaders)],
        cwd=host["inferencex_e2e"],
        env={**execution_env(), "PYTHONPATH": str(host["inferencex_e2e"])},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if done.returncode:
        raise RuntimeError(
            f"the AgentX client rejects the trace loaders {loaders}: {done.stderr.strip()[-400:]}"
        )
    datasets = json.loads(done.stdout)
    for point in points:
        loader, repo = point["client_env"]["WEKA_LOADER_OVERRIDE"], point["trace"]["repo"]
        if datasets.get(loader) != repo:
            raise RuntimeError(
                f"point {point_id(point)}: loader {loader} replays {datasets.get(loader)!r}, "
                f"but its trace is {repo!r}"
            )


def prepare_traces(host: dict, points: list[dict], log_path: Path) -> None:
    """Prepare traces and verify offline loading."""
    verify_trace_loaders(host, points)
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
        record.update(metrics_of(aggregate, point["gpus"]))
        record["raw_data"] = aggregate
    except (RuntimeError, OSError, KeyError, ValueError, TypeError, subprocess.TimeoutExpired) as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}"[:500])
        if isinstance(error, CleanupError):
            record["cleanup_failed"] = True
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


def server_packages(framework: str) -> dict:
    try:
        return {
            name: importlib.metadata.version(name)
            for name in (framework, "torch", "triton", "amd-aiter", "flydsl")
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
    client = serving["client"]
    inferencex_dir = fetch_inferencex(client["repo"], client["sha"], args.work / "inferencex")
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
    policy = selection.get("policy")
    if policy != POLICY:
        parser.error("serving.json has no Pareto selection; regenerate it with plan")
    args.duration = args.duration if args.duration is not None else selection.get("duration")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    try:
        points = selected_points(serving)
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
                "points_requested": selection.get("points_requested"),
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
    write_json(args.out / "selection.json", selection)
    try:
        validate_runtime(points)
        framework = points_framework(points)
        switch = plugin_switch(framework)
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
        serving_packages = server_packages(framework)
        for target in (results, results["control"]):
            target["run"]["client_packages"] = packages
            target["run"]["server_packages"] = serving_packages
    except (
        RuntimeError,
        FetchError,
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
                {switch: "1" if mode == "enabled" else "0", "AKLR_DISABLE": "0" if mode == "enabled" else "1"}
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
            if record.get("cleanup_failed"):
                error = f"AKLR {mode} {point['arm']} c{point['conc']}: {record['error']}"
                for interrupted in (target, results):
                    interrupted["run"].update(
                        completed=False, cleanup_failed=True, error=error, finished_at=utc_now()
                    )
                write_json(args.out / "results.json", results)
                print(f"ERROR: {error}", file=sys.stderr)
                return 1
            write_json(args.out / "results.json", results)
        for arm, spec in target["serving"]["arms"].items():
            spec["arm_measurement_sha256"] = sha256_json(
                {p["conc"]: measurements[p["exp_name"]] for p in points if p["arm"] == arm}
            )
        target["run"]["completed"] = True
        write_json(args.out / "results.json", results)
    all_records = results["control"]["points"] + records
    successful = sum(r["status"] == "ok" and valid_metrics(r) for r in all_records)
    failed = len(all_records) - successful
    status = "Failed" if not successful else "Partial success" if failed else "Success"
    print(
        f"{status}: {successful}/{len(all_records)} measurement points successful; "
        f"{failed} failed; wrote {args.out / 'results.json'}"
    )
    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
