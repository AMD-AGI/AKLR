# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The ``python -m aklr validate`` command."""

from __future__ import annotations

import argparse
import math
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from aklr.core.validation.pipeline import Pipeline, PlatformRecord
from aklr.core.validation.stages import (
    STAGE_ORDER,
    AuditStage,
    BenchStage,
    E2eStage,
    PipelineStage,
    Progress,
    ValidationScope,
)

__all__ = ["DEFAULT_REPORT_DIR", "main"]

DEFAULT_REPORT_DIR: Final = Path("aklr_validation")


def main(argv: Sequence[str]) -> int:
    """Validate the change against its base and return the pipeline exit code."""
    parser = argparse.ArgumentParser(prog="python -m aklr validate")
    parser.add_argument("paths", nargs="*", type=Path, metavar="PATH")
    parser.add_argument("--base", help="a git ref; enables the change rules")
    parser.add_argument(
        "--stages",
        type=lambda text: tuple(text.split(",")),
        default=STAGE_ORDER,
        help=f"a part of {','.join(STAGE_ORDER)} that starts with audit (default: all)",
    )
    parser.add_argument("--devices", help="0-7 or 0,2,5")
    parser.add_argument(
        "--under", metavar="DIR", help="only the schemas under DIR, such as aklr/plugins/aiter"
    )
    parser.add_argument("--exclude", metavar="DIR", help="not the schemas under DIR")
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="build and check the kernels only, on the workloads not recorded failed: nothing is timed or written back",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--e2e-deadline",
        type=float,
        help="Unix timestamp by which e2e must finish, including cleanup (CI job budget)",
    )
    parser.add_argument(
        "--e2e-all-scenarios",
        action="store_true",
        help="serve every match of scenarios.yaml, not only the first of each model",
    )
    parser.add_argument(
        "--e2e-baseline-dir",
        type=Path,
        help="keep each scenario's AKLR-off result here and reuse it while its recipe and sglang are"
        " the same; whoever can write here decides the baseline, so give untrusted runs a read-only one",
    )
    parser.add_argument(
        "--allow-measured-edits",
        action="store_true",
        help="skip the PERF_EDITED rule (CI: the bot's measurement PR only)",
    )
    args = parser.parse_args(argv)
    if args.e2e_deadline is not None and not math.isfinite(args.e2e_deadline):
        parser.error("--e2e-deadline must be a finite Unix timestamp")
    deadline = None if args.e2e_deadline is None else time.monotonic() + args.e2e_deadline - time.time()
    names = args.stages
    if names[:1] != ("audit",) or names != tuple(n for n in STAGE_ORDER if n in names):
        parser.error(
            f"--stages {','.join(names)!r} is not a part of {','.join(STAGE_ORDER)} that starts with audit"
        )
    Progress().write("collecting schemas and platform")
    try:
        scope = ValidationScope.of_working_tree(args.base, args.paths or None)
        if args.under is not None:
            scope = scope.under(args.under)
        if args.exclude is not None:
            scope = scope.under(args.exclude, inside=False)
        stages: list[PipelineStage] = [AuditStage(allow_measured_edits=args.allow_measured_edits)]
        device_stages = {stage.name: stage for stage in (BenchStage, E2eStage)}
        selected = [device_stages[name] for name in names[1:]]
        if selected:
            if args.devices is None:
                indices = range(PlatformRecord.current().n_visible_devices)
            else:
                indices = [
                    index
                    for part in args.devices.split(",")
                    for first, _, last in [part.partition("-")]
                    for index in range(int(first), int(last or first) + 1)
                ]
            devices = [f"cuda:{i}" for i in indices]
            stages.extend(
                stage(
                    devices,
                    deadline=deadline,
                    all_scenarios=args.e2e_all_scenarios,
                    baseline_dir=args.e2e_baseline_dir,
                )
                if stage is E2eStage
                else stage(devices, timed=not args.no_timing)
                for stage in selected
            )
        pipeline = Pipeline(stages)
    except ValueError as error:
        parser.error(str(error))
    report = pipeline.run(scope, out=args.out)
    if args.no_timing:  # nothing measured: no earlier run's rewrites or baselines stay beside this report
        for export in ("schemas", "baseline"):
            shutil.rmtree(args.out / export, ignore_errors=True)
    print(f"validation {report.conclusion}: {args.out / 'report.md'}")
    return report.exit_code
