# SPDX-License-Identifier: MIT
"""Measure one planned group in its verified Docker image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from e2e_perf.runtime import MODES
from e2e_perf.utils import expected_points, read_json


def summarize(out: Path, exit_code: int) -> int:
    """Write the per-mode point counts to the job summary."""
    try:
        results = read_json(out / "results.json")
        expected = len(expected_points(results))
        rows, ok_total = [], 0
        for mode in MODES:
            points = results["modes"][mode]["points"]
            counts = {status: sum(p["status"] == status for p in points) for status in ("ok", "failed")}
            ok_total += counts["ok"]
            rows.append(
                f"| {mode} | {counts['ok']} | {counts['failed']} | {expected - counts['ok'] - counts['failed']} |"
            )
        status = "Success" if ok_total == 2 * expected and not exit_code else "Partial success"
        if exit_code or not ok_total:
            status, exit_code = "Failed", exit_code or 1
        lines = [
            f"## Daily end-to-end performance sweep: {status}",
            "",
            "| AKLR | OK | Failed | Not completed |",
            "| --- | ---: | ---: | ---: |",
            *rows,
        ]
        if results["run"].get("error"):
            lines += ["", f"Error: {results['run']['error']}"]
    except (OSError, ValueError, KeyError, TypeError):
        exit_code = exit_code or 1
        lines = ["## Daily end-to-end performance sweep: Failed", "", "No usable results were saved."]
    summary = "\n".join(lines) + "\n"
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(summary)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m e2e_perf sweep", description=__doc__)
    parser.add_argument("--plan", type=Path, help="plan.json of one group; required unless --summarize")
    parser.add_argument("--hub", type=Path, help="host HF hub cache; required unless --in-container")
    parser.add_argument("--in-container", action="store_true", help="measure inside the prepared container")
    parser.add_argument(
        "--summarize",
        type=int,
        metavar="EXIT_CODE",
        help="only summarize the results in --out, measured elsewhere (on SaFE) with this exit code",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="host kernel cache; defaults to <out>/cache")
    parser.add_argument("--work", type=Path, help="host work directory; defaults to <out>/work")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.summarize is not None:
        return summarize(args.out, args.summarize)
    if args.plan is None:
        parser.error("--plan is required")
    options = ["--run-id", args.run_id] if args.run_id is not None else []
    if args.in_container:
        if args.hub is not None or args.cache is not None:
            parser.error("--hub and --cache configure Docker and cannot be used with --in-container")
        from e2e_perf import runtime

        work = args.work or args.out / "work"
        return runtime.main([*options, "--plan", str(args.plan), "--out", str(args.out), "--work", str(work)])
    from e2e_perf import launcher

    return summarize(args.out, launcher.run(args, options))
