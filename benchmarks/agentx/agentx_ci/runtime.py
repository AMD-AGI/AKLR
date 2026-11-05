# SPDX-License-Identifier: MIT
"""Execute planned points in the container and record validated measurements."""

from __future__ import annotations

import argparse
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from agentx_ci.recipe_resolve import POLICY, ResolveError, SelectionError, fetch_inferencex, selected_points
from agentx_ci.utils import (
    REPO_ROOT,
    begin_attempt,
    read_json,
    sha256_json,
    utc_now,
    valid_metrics,
    write_json,
)


def metrics_of(aggregate: dict) -> dict:
    """The two Pareto axes from an InferenceX AgentX aggregate."""
    request = aggregate["request_metrics"]
    metrics = {
        "tput_per_gpu": request["throughput"]["per_gpu"]["total_tput_tps"],
        "p90_e2e_norm_intvty": request["latency"]["e2e_norm_intvty"]["p90"],
        "num_requests_successful": aggregate.get("num_requests_successful"),
    }
    if not valid_metrics(metrics):
        raise ValueError("Invalid metrics: axes must be non-boolean finite positive numbers")
    return metrics


# Inherit infrastructure only; tunables belong in the hashed plan.
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
    "FLYDSL_RUNTIME_CACHE_DIR",
    "FLYDSL_AUTOTUNE_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "AKLR_KERNEL_CACHE_DIR",
)


def execution_env() -> dict[str, str]:
    return {name: os.environ[name] for name in INFRA_ENV if name in os.environ}


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
    """Stop the whole group created with start_new_session, even after its leader exits."""
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
    """Own the client's entire process group, including replay/bootstrap descendants."""
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        return process.wait(timeout=timeout)
    finally:
        stop(process)


def validate_runtime(points: list[dict]) -> None:
    """Validate upstream budgets and client topology."""
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
    """Allocate a port when the recipe leaves it unspecified."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def port_free(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


# Prepare both datasets and hub caches for offline replay.
PREPARE_TRACES = """
import os, subprocess, sys
from pathlib import Path
from infx.bench.agentic import traces
from infx.bench.agentic.venv import Runtime, bootstrap
runtime_dir, model_prefix = Path(sys.argv[1]), sys.argv[2]
runtime = Runtime(runtime_dir)
if bootstrap(runtime):
    sys.exit("AgentX runtime bootstrap failed")
_, dataset = traces.resolve(model_prefix, os.environ.get("WEKA_LOADER_OVERRIDE") or None)
offline = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
env = {k: v for k, v in os.environ.items() if k not in offline}
load = "import sys; from datasets import load_dataset; load_dataset(sys.argv[1])"
subprocess.run([str(runtime.python), "-c", load, dataset], env=env, check=True)
subprocess.run([str(runtime.hf), "download", "--repo-type", "dataset", dataset], env=env, check=True)
# Verify the client can resolve traces offline.
subprocess.run([str(runtime.hf), "download", "--repo-type", "dataset", dataset],
               env={**env, "HF_HUB_OFFLINE": "1"}, check=True)
"""


def prepare_traces(host: dict, points: list[dict], log_path: Path) -> None:
    """Fill ``host["datasets_cache"]`` with the AgentX traces the points replay."""
    sources = {
        (
            p["client_env"]["MODEL_PREFIX"],
            p["client_env"].get("WEKA_LOADER_OVERRIDE", "") or "",
        )
        for p in points
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for prefix, loader in sorted(sources):
        with log_path.open("a") as log:
            code = run_client(
                [sys.executable, "-c", PREPARE_TRACES, str(host["agentx_runtime_dir"]), prefix],
                cwd=host["inferencex_e2e"],
                env={
                    **execution_env(),
                    "PYTHONPATH": str(host["inferencex_e2e"]),
                    "HF_HUB_CACHE": str(host["trace_hub"]),
                    "HF_DATASETS_CACHE": str(host["datasets_cache"]),
                    "WEKA_LOADER_OVERRIDE": loader,
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(p["runtime"]["point_timeout_s"] for p in points),
            )
        if code:
            raise RuntimeError(f"preparing the AgentX traces failed; see {log_path}")


def run_live(point: dict, out: Path, host: dict) -> dict:
    """One point on this host; returns the client's aggregate."""
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
            env={**execution_env(), **point["launch"]["env"]},
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
            "HF_DATASETS_CACHE": str(host["datasets_cache"]),
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
                [sys.executable, "-m", "infx.bench", "agentic"],
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
    """Unknown source state is not eligible for promotion to a formal baseline."""
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


def host_settings(args: argparse.Namespace, serving: dict) -> dict:
    weights = serving["weights"]
    trace_hub = args.work / "hf-traces-hub"
    trace_hub.mkdir(parents=True, exist_ok=True)
    model_path, tokenizer = args.model_path, args.model_path
    if model_path is None:
        hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
        model_dir = hub / ("models--" + weights["repo"].replace("/", "--"))
        model_path = model_dir / "snapshots" / weights["revision"]
        # Offline tokenizer lookup uses refs/main, which must match the pinned revision.
        tokenizer = weights["repo"]
        main = model_dir / "refs" / "main"
        if not main.is_file():
            raise RuntimeError(
                f"{main} is missing; prepare refs/main at the pinned {weights['revision']} "
                "for the AgentX client's offline tokenizer"
            )
        main_revision = main.read_text().strip()
        if main_revision != weights["revision"]:
            raise RuntimeError(
                f"{main} is {main_revision}, not the pinned {weights['revision']}; "
                "the AgentX client's tokenizer would come from another revision"
            )
        # Link pinned weights into the writable trace hub for offline tokenizer lookup.
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
    parser.add_argument("--serving", type=Path, required=True, help="serving.json from resolve")
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
    records = []
    results = {
        "model": serving["model"],
        "run": {
            "id": str(args.run_id),
            "attempt_id": attempt_id,
            "serving_sha256": sha256_json(serving),
            "mode": "live",
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "completed": False,
            "aklr_sha": aklr_sha(),
            "source_dirty": source_dirty(),
            "host": socket.gethostname(),
            "image": serving["image"],
            "inferencex": serving["provenance"]["inferencex"],
            "arms_requested": args.arms,
            "points_requested": args.points,
            "duration_override_s": args.duration,
            "weights": serving["weights"],
            "model_path_override": args.model_path,
            "points_expected": len(points),
            "point_policy": policy,
            **({"selection": selection} if policy == POLICY else {}),
        },
        "serving": {
            "grid_sha256": serving["grid_sha256"],
            "arms": {
                arm: dict(spec)
                for arm, spec in serving["arms"].items()
                if any(p["arm"] == arm for p in points)
            },
        },
        "points": records,
    }
    measurements = {
        p["exp_name"]: {
            "environment_policy": "planned-tunables-v1",
            "weights": serving["weights"],
            "launch": p["launch"],
            "client_env": {**p["client_env"], **({"DURATION": str(args.duration)} if args.duration else {})},
            "runtime": p["runtime"],
            "model_path_override": args.model_path,
        }
        for p in points
    }
    for arm, spec in results["serving"]["arms"].items():
        spec["arm_measurement_sha256"] = sha256_json(
            {p["conc"]: measurements[p["exp_name"]] for p in points if p["arm"] == arm}
        )
    write_json(args.out / "results.json", results)
    if policy == POLICY:
        write_json(args.out / "selection.json", selection)
    try:
        validate_runtime(points)
        host = host_settings(args, serving)
        print(f"[{utc_now()}] preparing the AgentX traces", flush=True)
        prepare_traces(host, points, args.out / "prepare_traces.log")
    except (RuntimeError, ResolveError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        results["run"].update(error=str(error), finished_at=utc_now())
        write_json(args.out / "results.json", results)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for point in points:
        print(f"[{utc_now()}] {point['arm']} c{point['conc']}: {point['exp_name']}", flush=True)
        record = run_point(point, args.out / "attempts" / attempt_id / "points" / point["exp_name"], host)
        record["measurement"] = measurements[point["exp_name"]]
        record["measurement_sha256"] = sha256_json(record["measurement"])
        print(f"  -> {record['status']} {record.get('error') or ''}", flush=True)
        records.append(record)
        results["run"]["finished_at"] = utc_now()
        write_json(args.out / "results.json", results)
    results["run"]["completed"] = True
    write_json(args.out / "results.json", results)
    failed = sum(r["status"] != "ok" for r in records)
    print(f"{len(records) - failed}/{len(records)} points ok; wrote {args.out / 'results.json'}")
    return 1 if failed == len(records) else 0


if __name__ == "__main__":
    sys.exit(main())
