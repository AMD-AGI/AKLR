# SPDX-License-Identifier: MIT
"""Run one planned model target in its verified Docker image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agentx_ci.utils import complete_measurement, read_json, validated_results


def summarize(out: Path, exit_code: int) -> int:
    """Expose partial measurements without hiding launcher or cancellation failures."""
    try:
        results = validated_results(read_json(out / "results.json"))
        rows, successful, total = [], 0, 0
        for mode, target in (("disabled", results["control"]), ("enabled", results)):
            expected = target["run"]["points_expected"]
            points = target["points"]
            identities = [(point["arm"], point["conc"]) for point in points]
            if (
                type(expected) is not int
                or expected <= 0
                or len(points) > expected
                or len(identities) != len(set(identities))
            ):
                raise ValueError("invalid measurement point counts")
            ok = sum(point.get("status") == "ok" for point in points)
            successful += ok
            total += expected
            rows.append(f"| {mode} | {ok} | {len(points) - ok} | {expected - len(points)} | {expected} |")
        status = "Success" if not exit_code and complete_measurement(results) else "Partial success"
        if exit_code or not successful:
            status, exit_code = "Failed", exit_code or 1
        lines = [
            f"## AgentX sweep: {status}",
            "",
            f"**{successful}/{total} measurement points successful.**",
            "",
            "| AKLR | Successful | Failed | Not completed | Planned |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "Point failures and their logs remain in the result artifact and report.",
        ]
    except (OSError, ValueError, KeyError, TypeError):
        exit_code = exit_code or 1
        lines = ["## AgentX sweep: Failed", "", "No usable measurement results were saved."]
    if exit_code:
        lines.extend(["", f"Sweep process exited with status {exit_code}; see the sweep and launcher logs."])
    summary = "\n".join(lines) + "\n"
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(summary)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentx_bench sweep", description=__doc__)
    parser.add_argument(
        "--serving", type=Path, required=True, help="serving.json for one model target from plan"
    )
    parser.add_argument("--hub", type=Path, help="Host HF hub cache; required unless --in-container")
    parser.add_argument(
        "--in-container",
        action="store_true",
        help="Execute inside the prepared container using the launcher's verified execution metadata",
    )
    parser.add_argument("--out", type=Path, required=True, help="Host directory for results and logs")
    parser.add_argument(
        "--image", help="Must match the recipe image; update the recipe serving.json to change it"
    )
    parser.add_argument("--cache", type=Path, help="Host kernel cache directory; defaults to <out>/cache")
    parser.add_argument("--work", type=Path, help="Host runtime/trace workspace; defaults to <out>/work")
    parser.add_argument("--run-id", help="Run identifier; must match the plan for report --plan")
    parser.add_argument("--duration", type=int, help="Override per-point duration in seconds (smoke run)")
    parser.add_argument("--model-path", help="Model path visible inside the container")
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.in_container:
        if any(value is not None for value in (args.hub, args.image, args.cache)):
            parser.error("--hub, --image and --cache configure Docker and cannot be used with --in-container")
        args.work = args.work or args.out / "work"
    command = []
    for name in (
        "run_id",
        "duration",
        "model_path",
    ):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if args.in_container:
        from agentx_ci import runtime

        command.extend(["--serving", str(args.serving), "--out", str(args.out), "--work", str(args.work)])
        return runtime.main(command)
    from agentx_ci import docker_launch

    return summarize(args.out, docker_launch.run(args, command))
