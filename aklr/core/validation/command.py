# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The ``python -m aklr validate`` command."""

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
    E2eStage,
    PipelineStage,
    Progress,
    ValidationScope,
)
from aklr.core.validation.stages._e2e._config import E2eConfig

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
    parser.add_argument(
        "--e2e-scenario",
        dest="scenarios",
        action="append",
        default=[],
        metavar="NAME",
        help="an e2e scenario, or a model's every scenario, of scenarios.yaml (repeatable; default: every enabled one)",
    )
    parser.add_argument(
        "--e2e-cookbook-source",
        metavar="REF_OR_DIR",
        help="the sglang cookbook's branch, tag or SHA, or a local sglang checkout (default: scenarios.yaml's ref)",
    )
    parser.add_argument(
        "--e2e-hosts",
        type=lambda text: tuple(text.split(",")),
        default=("127.0.0.1",),
        metavar="H0,H1,...",
        help="the nodes e2e serves on, node 0 first (default: this one)",
    )
    parser.add_argument(
        "--e2e-launch",
        metavar="PREFIX",
        help="how to start a command on another node, {cmd} its one shell-quoted word:"
        " 'ssh {host} {cmd}' or 'srun -w {host} --overlap sh -c {cmd}'",
    )
    parser.add_argument(
        "--e2e-work-dir",
        type=Path,
        metavar="DIR",
        help="where e2e keeps its measurements, logs and call files; a shared filesystem for several nodes",
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
    if names[:1] != ("audit",) or names != tuple(n for n in STAGE_ORDER if n in names):
        parser.error(
            f"--stages {','.join(names)!r} is not a part of {','.join(STAGE_ORDER)} that starts with audit"
        )
    Progress().write("collecting schemas and platform")
    try:
        scope = ValidationScope.of_working_tree(args.base, args.paths or None)
        stages: list[PipelineStage] = [AuditStage(allow_measured_edits=args.allow_measured_edits)]
        config = E2eConfig.read() if "e2e" in names else None
        scenarios = config.selected(args.scenarios) if config is not None else ()
        devices: list[str] = []
        # e2e with no enabled scenario is skipped, and needs no device
        if "bench" in names or scenarios:
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
            devices = [f"cuda:{i}" for i in indices]
        if "bench" in names:
            stages.append(BenchStage(devices))
        if config is not None:
            stages.append(
                E2eStage(
                    scenarios,
                    devices,
                    cookbook_repo=config.cookbook_repo,
                    cookbook_source=args.e2e_cookbook_source or config.cookbook_ref,
                    hosts=args.e2e_hosts,
                    launch=args.e2e_launch,
                    work_dir=args.e2e_work_dir,
                )
            )
        pipeline = Pipeline(stages)
    except ValueError as error:
        parser.error(str(error))
    report = pipeline.run(scope)
    report.write(args.out)
    report.write_schemas(scope.root, args.out / "schemas")
    report.write_baselines(scope, args.out / "baseline")
    print(f"validation {report.conclusion}: {args.out / 'report.md'}")
    return report.exit_code
