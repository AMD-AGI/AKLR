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

from aklr.core.validation.report import PlatformRecord, validate
from aklr.core.validation.validators import (
    AuditValidator,
    BenchValidator,
    Progress,
    ValidationScope,
    Validator,
)

__all__ = ["DEFAULT_REPORT_DIR", "main"]

DEFAULT_REPORT_DIR: Final = Path("aklr_validation")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m aklr validate")
    parser.add_argument("paths", nargs="*", type=Path, metavar="PATH")
    parser.add_argument("--base", help="a git ref; enables the change rules")
    parser.add_argument(
        "--validators",
        type=lambda text: tuple(text.split(",")),
        default=Validator.names(),
        help=f"which of {','.join(Validator.names())} to run, comma-separated, each on its own (default: all)",
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
    parser.add_argument("--num-trials", type=int, default=3, help="accuracy trials per workload")
    parser.add_argument(
        "--isolate",
        action="append",
        default=[],
        metavar="OP_TYPE",
        help="run each of this op type's bench tasks in a fresh process (repeatable)",
    )
    parser.add_argument(
        "--allow-measured-edits",
        action="store_true",
        help="skip the PERF_EDITED rule (CI: the bot's measurement PR only)",
    )
    args = parser.parse_args(argv)
    names = args.validators
    try:
        validator_types = []
        for position, name in enumerate(names):
            validator_types.append(Validator.resolve(name))
            if name in names[:position]:
                parser.error(f"--validators: validator {name!r} is named twice")
        Progress().write("collecting schemas and platform")
        scope = ValidationScope.of_working_tree(args.base, args.paths or None)
        if args.under is not None:
            scope = scope.under(args.under)
        if args.exclude is not None:
            scope = scope.under(args.exclude, inside=False)
        validators: list[Validator] = []
        for validator_type in validator_types:
            if validator_type is AuditValidator:
                validators.append(AuditValidator(allow_measured_edits=args.allow_measured_edits))
                continue
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
            validators.append(
                BenchValidator(
                    [f"cuda:{i}" for i in indices],
                    isolate=args.isolate,
                    num_trials=args.num_trials,
                    timed=not args.no_timing,
                )
            )
    except ValueError as error:
        parser.error(str(error))
    report = validate(scope, validators)
    report.write(args.out)
    report.write_schemas(scope.root, args.out / "schemas")
    print(f"validation {report.conclusion}: {args.out / 'report.md'}")
    return report.exit_code
