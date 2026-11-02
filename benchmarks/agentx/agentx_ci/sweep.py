# SPDX-License-Identifier: MIT
"""Run every point of serving.json on this host and write results.json.

Per point: start sglang with the point's launch spec, wait for ``/health`` and ``/v1/models``,
run InferenceX's AgentX client (``python3 -m infx.bench agentic``) from the same InferenceX
commit, read its aggregate JSON, stop the server's process group. Points run one at a time;
a failed point is recorded and the sweep moves on.

``--mock`` skips the server and the client and writes a fake aggregate per point, so the
rest of the pipeline runs without GPUs.
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from agentx_ci.common import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    load_config,
    read_json,
    utc_now,
    write_json,
)
from agentx_ci.recipe_resolve import fetch_inferencex

GPUS_PER_NODE = 8
POLL_S = 10
# Fake curves for --mock: tokens/s per GPU grows with concurrency, interactivity falls; TP8
# wins at low concurrency and TP4 at high, so the two arms cross near c4 as on real hardware.
MOCK_CURVES = {
    "tp8": {"tput": (22.0, 0.85), "intvty": (140.0, 0.18)},
    "tp4": {"tput": (30.0, 0.80), "intvty": (110.0, 0.22)},
}


def metrics_of(aggregate: dict) -> dict:
    """The two Pareto axes from an InferenceX AgentX aggregate."""
    request = aggregate["request_metrics"]
    return {
        "tput_per_gpu": request["throughput"]["per_gpu"]["total_tput_tps"],
        "p90_e2e_norm_intvty": request["latency"]["e2e_norm_intvty"]["p90"],
        "num_requests_successful": aggregate.get("num_requests_successful"),
    }


def mock_aggregate(point: dict, seed: int, scale: float = 1.0) -> dict:
    """An aggregate with the fields ``metrics_of`` reads, ±4 % deterministic noise per (seed, point)."""
    curve = MOCK_CURVES.get(point["arm"], MOCK_CURVES["tp4"])
    rng = random.Random(f"{seed}:{point['arm']}:{point['conc']}")
    (tput_a, tput_b), (intvty_a, intvty_b) = curve["tput"], curve["intvty"]
    tput = tput_a * point["conc"] ** tput_b * scale * rng.uniform(0.96, 1.04)
    intvty = intvty_a / point["conc"] ** intvty_b * rng.uniform(0.96, 1.04)
    return {
        "mock": True,
        "num_requests_successful": 40 * point["conc"],
        "request_metrics": {
            "throughput": {"per_gpu": {"total_tput_tps": round(tput, 2)}},
            "latency": {"e2e_norm_intvty": {"p90": round(intvty, 2)}},
        },
    }


def host_memory_gb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // (1024 * 1024)
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def total_cpu_dram_gb(point: dict, utilization: float, memory_gb: int) -> str:
    """InferenceX's rule: the point's GPU share of host memory × dram-utilization; 0 without offload."""
    if point["kv_offloading"] != "dram":
        return "0"
    return str(int(memory_gb * point["gpus"] / GPUS_PER_NODE * utilization))


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status < 400
    except (urllib.error.URLError, OSError):
        return False


def wait_ready(base_url: str, server: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"server exited with code {server.returncode} before ready"
            )
        if http_ok(f"{base_url}/health") and http_ok(f"{base_url}/v1/models"):
            return
        time.sleep(POLL_S)
    raise RuntimeError(f"server not ready after {timeout_s:.0f} s")


def stop(server: subprocess.Popen, grace_s: int = 120) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def port_free(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def run_live(point: dict, out: Path, host: dict) -> dict:
    """One point on this host; returns the client's aggregate."""
    port, base_url = host["port"], f"http://127.0.0.1:{host['port']}"
    if not port_free(port):
        raise RuntimeError(f"port {port} is already in use")
    started = time.monotonic()
    budget = host["point_timeout_s"][point["arm"]]
    placeholders = {
        "${AGENTX_MODEL_PATH}": host["model_path"],
        "${AGENTX_PORT}": str(port),
    }
    command = [placeholders.get(arg, arg) for arg in point["launch"]["command"]]
    (out / "server_command.txt").write_text(" ".join(command) + "\n")
    with (out / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            env={**os.environ, **point["launch"]["env"]},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_ready(base_url, server, min(host["ready_timeout_s"], budget))
        result_name = point["exp_name"]
        client_env = {
            **os.environ,
            **point["client_env"],
            **host["client_env_overrides"],
            "TOTAL_CPU_DRAM_GB": total_cpu_dram_gb(
                point, host["dram_utilization"], host["memory_gb"]
            ),
            "PORT": str(port),
            "SERVED_MODEL_NAME": point["served_model_name"],
            "TOKENIZER": host["model_path"],
            "AIPERF_SERVER_METRICS_URLS": f"{base_url}/metrics",
            "AIPERF_RUNTIME_DIR": str(host["agentx_runtime_dir"]),
            "RESULT_DIR": str(out / "agentic"),
            "RESULT_FILENAME": result_name,
            "AGENTIC_OUTPUT_DIR": str(out),
            "PYTHONPATH": str(host["ix_e2e"]),
        }
        with (out / "client.log").open("w") as log:
            done = subprocess.run(
                [sys.executable, "-m", "infx.bench", "agentic"],
                cwd=host["ix_e2e"],
                env=client_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(60, budget - (time.monotonic() - started)),
                check=False,
            )
        if done.returncode:
            raise RuntimeError(f"AgentX client exited with code {done.returncode}")
        return read_json(out / f"{result_name}.json")
    finally:
        stop(server)


def run_point(point: dict, out: Path, host: dict | None, mock: dict | None) -> dict:
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
        if mock is not None:
            aggregate = mock_aggregate(
                point, mock["seed"], mock["scale"].get(point["arm"], 1.0)
            )
            write_json(out / f"{point['exp_name']}.json", aggregate)
        else:
            aggregate = run_live(point, out, host)
        record.update(metrics_of(aggregate))
    except (RuntimeError, OSError, KeyError, subprocess.TimeoutExpired) as error:
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


def parse_scale(items: list[str]) -> dict[str, float]:
    return {
        arm: float(value) for arm, _, value in (item.partition("=") for item in items)
    }


def host_settings(args: argparse.Namespace, config: dict, serving: dict) -> dict:
    sweep = config["sweep"]
    weights = serving["weights"]
    model_path = args.model_path
    if model_path is None:
        hub = Path(
            os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")
        )
        model_dir = "models--" + weights["repo"].replace("/", "--")
        model_path = hub / model_dir / "snapshots" / weights["revision"]
    if not Path(model_path).is_dir():
        raise RuntimeError(
            f"model snapshot {model_path} not found; prefill the HF hub cache"
        )
    ix = serving["provenance"]["inferencex"]
    ix_dir = fetch_inferencex(ix["repo"], ix["sha"], args.work / "inferencex")
    overrides = {"DURATION": str(args.duration)} if args.duration else {}
    return {
        "port": sweep["port"],
        "ready_timeout_s": sweep["ready_timeout_s"],
        "point_timeout_s": sweep["point_timeout_s"],
        "dram_utilization": sweep["dram_utilization"],
        "memory_gb": host_memory_gb(),
        "model_path": str(model_path),
        "ix_e2e": ix_dir / "inferencex-e2e",
        "agentx_runtime_dir": args.work / "agentx-runtime",
        "client_env_overrides": overrides,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m agentx_ci sweep", description=__doc__
    )
    parser.add_argument(
        "--serving", type=Path, required=True, help="serving.json from resolve"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out", type=Path, required=True, help="directory for results.json"
    )
    parser.add_argument("--work", type=Path, default=Path("/tmp/agentx-work"))
    parser.add_argument("--arms", default="all", help="comma-separated arms, or all")
    parser.add_argument(
        "--points",
        help="comma-separated ARM:CONC, e.g. tp8:1,tp4:4 (smoke runs only)",
    )
    parser.add_argument(
        "--model-path", help="default: the pinned revision in $HF_HUB_CACHE"
    )
    parser.add_argument(
        "--duration", type=int, help="override DURATION (smoke runs only)"
    )
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument(
        "--mock", action="store_true", help="fake results, no GPU or server"
    )
    parser.add_argument("--mock-seed", type=int, default=0)
    parser.add_argument(
        "--mock-scale",
        action="append",
        default=[],
        metavar="ARM=FACTOR",
        help="scale an arm's fake throughput",
    )
    args = parser.parse_args(argv)

    config, serving = load_config(args.config), read_json(args.serving)
    points = serving["points"]
    if args.arms != "all":
        wanted = set(args.arms.split(","))
        points = [p for p in points if p["arm"] in wanted]
    if args.points:
        wanted = set(args.points.split(","))
        points = [p for p in points if f"{p['arm']}:{p['conc']}" in wanted]
        missing = wanted - {f"{p['arm']}:{p['conc']}" for p in points}
        if missing:
            print(f"ERROR: no such points: {sorted(missing)}", file=sys.stderr)
            return 1
    mock = (
        {"seed": args.mock_seed, "scale": parse_scale(args.mock_scale)}
        if args.mock
        else None
    )
    try:
        host = None if args.mock else host_settings(args, config, serving)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    started = utc_now()
    records = []
    for point in points:
        print(
            f"[{utc_now()}] {point['arm']} c{point['conc']}: {point['exp_name']}",
            flush=True,
        )
        record = run_point(point, args.out / "points" / point["exp_name"], host, mock)
        print(f"  -> {record['status']} {record.get('error') or ''}", flush=True)
        records.append(record)
    results = {
        "schema": 1,
        "run": {
            "id": str(args.run_id),
            "mode": "mock" if args.mock else "live",
            "started_at": started,
            "finished_at": utc_now(),
            "aklr_sha": aklr_sha(),
            "host": socket.gethostname(),
            "image": serving["image"],
            "inferencex": serving["provenance"]["inferencex"],
            "arms_requested": args.arms,
        },
        "serving": {
            "grid_sha256": serving["grid_sha256"],
            "arms": {
                arm: spec
                for arm, spec in serving["arms"].items()
                if any(p["arm"] == arm for p in points)
            },
            "warnings": serving["warnings"],
        },
        "points": records,
    }
    write_json(args.out / "results.json", results)
    failed = sum(r["status"] != "ok" for r in records)
    print(
        f"{len(records) - failed}/{len(records)} points ok; wrote {args.out / 'results.json'}"
    )
    return 1 if failed == len(records) else 0
