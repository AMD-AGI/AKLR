###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import argparse
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
    """Validate the change against its base and return the pipeline's exit code."""
    parser = argparse.ArgumentParser(prog="python -m aklr validate")
    parser.add_argument("paths", nargs="*", type=Path, metavar="PATH")
    parser.add_argument("--base", help="a git ref; enables the change rules")
    parser.add_argument(
        "--stages",
        type=lambda text: tuple(text.split(",")),
        default=(AuditStage.name, BenchStage.name),
        help=f"a part of {','.join(STAGE_ORDER)} that starts with {AuditStage.name}",
    )
    parser.add_argument("--devices", help="0-7 or 0,2,5")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--allow-measured-edits",
        action="store_true",
        help="skip the PERF_EDITED rule (CI: the bot's measurement PR only)",
    )
    args = parser.parse_args(argv)
    names = args.stages
    if names[:1] != (AuditStage.name,) or names != tuple(n for n in STAGE_ORDER if n in names):
        parser.error(
            f"--stages {','.join(names)!r} is not a part of {','.join(STAGE_ORDER)}"
            f" that starts with {AuditStage.name}"
        )
    Progress().write("collecting schemas and platform")
    try:
        scope = ValidationScope.of_working_tree(args.base, args.paths or None)
        stages: list[PipelineStage] = [AuditStage(allow_measured_edits=args.allow_measured_edits)]
        if BenchStage.name in names:
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
            stages.append(BenchStage([f"cuda:{i}" for i in indices]))
        pipeline = Pipeline(stages)
    except ValueError as error:
        parser.error(str(error))
    report = pipeline.run(scope)
    report.write(args.out)
    report.write_schemas(scope.root, args.out / "schemas")
    print(f"validation {report.conclusion}: {args.out / 'report.md'}")
    return report.exit_code
