# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The ``python -m aklr validate`` command."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from aklr.core.validation.pipeline import Pipeline, PlatformRecord
from aklr.core.validation.stages import (
    STAGE_ORDER,
    AuditStage,
    BenchStage,
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
        default=("audit", "bench"),
        help=f"a part of {','.join(STAGE_ORDER)} that starts with audit",
    )
    parser.add_argument("--devices", help="0-7 or 0,2,5")
    parser.add_argument("--under", metavar="DIR", help="only the schemas under DIR, such as aklr/ops")
    parser.add_argument("--exclude", metavar="DIR", help="not the schemas under DIR")
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="build and check the kernels only, on the workloads not recorded failed: nothing is timed or written back",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--allow-measured-edits",
        action="store_true",
        help="skip the PERF_EDITED rule (CI: the bot's measurement PR only)",
    )
    args = parser.parse_args(argv)
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
        if "bench" in names:
            if args.devices is None:
                indices = range(PlatformRecord.current().n_visible_devices)
            else:
                indices = [
                    index
                    for part in args.devices.split(",")
                    for first, _, last in [part.partition("-")]
                    for index in range(int(first), int(last or first) + 1)
                ]
            if not indices:
                raise ValueError("no visible device and no --devices")
            stages.append(BenchStage([f"cuda:{i}" for i in indices], timed=not args.no_timing))
        pipeline = Pipeline(stages)
    except ValueError as error:
        parser.error(str(error))
    report = pipeline.run(scope)
    report.write(args.out)
    if args.no_timing:  # nothing measured: no earlier run's rewrites stay beside this report
        shutil.rmtree(args.out / "schemas", ignore_errors=True)
    else:
        report.write_schemas(scope.root, args.out / "schemas")
    print(f"validation {report.conclusion}: {args.out / 'report.md'}")
    return report.exit_code
