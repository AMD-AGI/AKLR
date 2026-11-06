# SPDX-License-Identifier: MIT
"""Run one planned model target in its verified Docker image."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentx_bench sweep", description=__doc__)
    parser.add_argument(
        "--serving", type=Path, required=True, help="serving.json for one model target from plan or resolve"
    )
    parser.add_argument("--hub", type=Path, help="Host HF hub cache; required unless --in-container")
    parser.add_argument(
        "--in-container",
        action="store_true",
        help="Execute inside the prepared container using the launcher's verified execution metadata",
    )
    parser.add_argument("--out", type=Path, required=True, help="Host directory for results and logs")
    parser.add_argument("--image", help="Must match the plan's image; use plan --run-image to change it")
    parser.add_argument("--cache", type=Path, help="Host kernel cache directory; defaults to <out>/cache")
    parser.add_argument("--work", type=Path, help="Host runtime/trace workspace; defaults to <out>/work")
    parser.add_argument("--run-id", help="Run identifier; must match the plan for report --plan")
    parser.add_argument("--duration", type=int, help="Override per-point duration in seconds (smoke run)")
    parser.add_argument("--arms", help="Comma-separated arms for a full-grid plan")
    parser.add_argument("--points", help="Comma-separated ARM:CONC filters for a full-grid plan")
    parser.add_argument("--point-policy", choices=("all", "inferencex-pareto"), help="Must match the plan")
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
        "arms",
        "points",
        "point_policy",
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

    return docker_launch.run(args, command)
