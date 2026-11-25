# SPDX-License-Identifier: MIT
"""Measure one group in its container: AKLR off, then on, each mode one server running every workload point."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from perf_utils.identity import aklr_sha, observed_hardware, server_packages, source_dirty
from perf_utils.image import execution_identity, validate_hardware
from perf_utils.inferencex import FetchError, fetch
from perf_utils.server import (
    CleanupError,
    available_port,
    execution_env,
    port_free,
    stop,
    wait_gpus_idle,
    wait_ready,
)

from e2e_perf import client
from e2e_perf.model_config import PLUGIN_SWITCHES
from e2e_perf.utils import SCHEMA, begin_attempt, metrics_of, read_json, snapshot, utc_now, write_json

MODES = ("disabled", "enabled")


def switches(framework: str, mode: str) -> dict[str, str]:
    """AKLR's environment for a mode: disabled turns off AKLR and the framework's plugin."""
    enabled = mode == "enabled"
    return {"AKLR_DISABLE": "0" if enabled else "1", PLUGIN_SWITCHES[framework]: "1" if enabled else "0"}


def pending_results(plan: dict, run_fields: dict) -> dict:
    """Results with the plan's identity and no points, so a failure before measuring still names its group."""
    framework = plan["framework"]["id"]
    return {
        "schema": SCHEMA,
        "model": plan["model"],
        "target": plan["target"],
        "framework": plan["framework"],
        "run": {
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "completed": False,
            "error": None,
            "run_key": plan["run_key"],
            "image": plan["image"],
            "expected_hardware": plan["hardware"],
            "weights": plan["weights"],
            "client": plan["client"],
            "workloads": plan["workloads"],
            "recipe_sha256": plan["recipe_sha256"],
            "gpus": plan["recipe"]["gpus"],
            "launch": plan["recipe"]["launch"],
            **run_fields,
        },
        "modes": {
            mode: {
                "switches": switches(framework, mode),
                "points": [],
                "error": None,
                "started_at": None,
                "finished_at": None,
            }
            for mode in MODES
        },
    }


def scalars(result: dict) -> dict:
    """The result's summary values; per-request lists stay in the result file."""
    return {key: value for key, value in result.items() if not isinstance(value, (list, dict))}


def start_server(command: list[str], env: dict[str, str], log: Path) -> subprocess.Popen:
    with log.open("w") as output:
        return subprocess.Popen(
            command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )


def run_mode(plan: dict, mode: str, host: dict, out: Path, record: dict, save: Callable[[], None]) -> None:
    """Start one server and run every workload point against it; a dead server leaves the rest not completed."""
    recipe = plan["recipe"]
    points = [(workload, conc) for workload in plan["workloads"] for conc in workload["concurrency"]]
    runtime = recipe["runtime"]
    record["started_at"] = utc_now()
    out.mkdir(parents=True, exist_ok=True)
    waited = wait_gpus_idle()
    if waited >= 1:
        print(f"  waited {waited:.0f} s for GPU VRAM to reach 0%", flush=True)
    port = available_port()
    if not port_free(port):
        raise RuntimeError(f"port {port} is already in use")
    base_url = f"http://127.0.0.1:{port}"
    placeholders = {"${E2E_MODEL_PATH}": host["model_path"], "${E2E_PORT}": str(port)}
    command = [placeholders.get(arg, arg) for arg in recipe["launch"]["command"]]
    env = {
        **execution_env(host["image_env"]),
        **recipe["launch"]["env"],
        **switches(plan["framework"]["id"], mode),
    }
    server = start_server(command, env, out / "server.log")
    try:
        down = None
        try:
            wait_ready(base_url, server, runtime["ready_timeout_s"], runtime["poll_s"])
        except RuntimeError as error:
            down = f"server not ready: {error}"
        for workload, conc in points:
            name = f"{workload['id']}/c{conc}"
            point = {
                "workload": workload["id"],
                "conc": conc,
                "status": "ok",
                "error": None,
                "started_at": utc_now(),
            }
            began = time.monotonic()
            if down is None and server.poll() is not None:
                down = f"server exited with code {server.returncode}"
            if down is not None:
                point.update(status="not-completed", error=down)
            else:
                print(f"[{utc_now()}] AKLR {mode} · {name}", flush=True)
                result = out / workload["id"] / f"c{conc}" / "result.json"
                result.parent.mkdir(parents=True, exist_ok=True)
                argv = client.client_argv(
                    base_url, recipe["served_model_name"], host["model_path"], workload, conc, result
                )
                try:
                    code = client.run(
                        argv, host["inferencex"], result.parent / "client.log", runtime["point_timeout_s"]
                    )
                    if code:
                        raise RuntimeError(f"client exited with code {code}")
                    raw = read_json(result)
                    requests = workload["num_prompts_per_conc"] * conc
                    point.update(metrics_of(raw, recipe["gpus"], requests), raw=scalars(raw))
                except CleanupError:
                    raise
                except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
                    point.update(status="failed", error=f"{type(error).__name__}: {error}"[:500])
            point["elapsed_s"] = round(time.monotonic() - began, 1)
            print(f"  -> {name} {point['status']} {point['error'] or ''}", flush=True)
            record["points"].append(point)
            save()
        record["error"] = down
    finally:
        stop(server)
        record["finished_at"] = utc_now()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m e2e_perf.runtime", description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="plan.json of one group")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--hub", type=Path, default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    args = parser.parse_args(argv)
    out, work = args.out.resolve(), args.work.resolve()
    attempt_id = begin_attempt(out)
    plan = read_json(args.plan)
    results = pending_results(
        plan,
        {
            "id": str(args.run_id),
            "attempt_id": attempt_id,
            "aklr_sha": aklr_sha(),
            "source_dirty": source_dirty(),
            "host": socket.gethostname(),
        },
    )
    run = results["run"]

    def save() -> None:
        run["finished_at"] = utc_now()
        write_json(out / "results.json", results)

    def fail(error: str) -> int:
        run["error"] = error
        save()
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    save()
    try:
        if args.hub is None:
            raise RuntimeError("--hub or HF_HUB_CACHE must name the HF hub cache")
        execution, image_env = execution_identity(plan)
        hardware = observed_hardware()
        validate_hardware(hardware, plan["hardware"])
        host = {
            "model_path": str(snapshot(plan["weights"], Path(args.hub))),
            "inferencex": fetch(
                plan["client"]["repo"],
                plan["client"]["sha"],
                work / "inferencex",
                ("/inferencex-e2e/infx/",),
                "inferencex-e2e/infx",
            )
            / "inferencex-e2e",
            "image_env": image_env,
        }
        run.update(
            execution=execution, hardware=hardware, server_packages=server_packages(plan["framework"]["id"])
        )
    except (RuntimeError, FetchError, OSError, ValueError, KeyError, TypeError) as error:
        return fail(f"{type(error).__name__}: {error}")
    for mode in MODES:
        try:
            run_mode(plan, mode, host, out / "attempts" / attempt_id / mode, results["modes"][mode], save)
        except (CleanupError, RuntimeError, OSError) as error:
            return fail(f"AKLR {mode}: {type(error).__name__}: {error}")
        save()
    run["completed"] = True
    down = [
        f"AKLR {mode}: {results['modes'][mode]['error']}" for mode in MODES if results["modes"][mode]["error"]
    ]
    run["error"] = "; ".join(down) or None
    save()
    points = [point for mode in MODES for point in results["modes"][mode]["points"]]
    ok = sum(point["status"] == "ok" for point in points)
    print(f"{ok}/{len(points)} measurement points ok; wrote {out / 'results.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
