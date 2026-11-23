# SPDX-License-Identifier: MIT
"""Rebuild site-data's kernels/ from the checkout's committed schemas.

Run from the repository root: PYTHONPATH=benchmarks/kernels python3 -m kernels_report ... SECTION
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from aklr.core.platform.env import RuntimeEnv
from kernels_report.data_collect import Context, SchemaErrors, collect, read_measurements
from kernels_report.report_render import write_section


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernels_report", description=__doc__.splitlines()[0])
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY", "AMD-AGI/AKLR")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument("--run-url", default="")
    parser.add_argument("--repo-url", default=f"{server}/{repository}")
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    # dispatch's threshold: AKLR_MIN_SPEEDUP, else its default
    parser.add_argument("--min-speedup", type=float, default=None)
    parser.add_argument("section", type=Path, help="kernels/ in a site-data checkout; replaced")
    args = parser.parse_args(argv)
    min_speedup = args.min_speedup if args.min_speedup is not None else RuntimeEnv().min_speedup

    root = Path.cwd()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    context = Context(
        commit=commit,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        run_id=args.run_id,
        run_url=args.run_url,
        repo_url=args.repo_url,
        ref=args.ref,
        event=args.event,
    )
    try:
        reports = collect(root, min_speedup=min_speedup, measurements=read_measurements(root))
    except SchemaErrors as e:
        print(f"main's schemas do not load:\n{e}", file=sys.stderr)
        return 1
    summary = write_section(args.section, reports, context)
    points = sum(len(r.points) for r in reports)
    served = sum(r.served for r in reports)
    print(
        f"{len(reports)} contracts, {served}/{points} points served at {commit[:12]}: {summary['status_counts']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
