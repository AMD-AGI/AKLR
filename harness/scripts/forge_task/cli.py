# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Turn one kernel-less AKLR schema into a KernelForge task, and put the kernel a task ends with back into AKLR.
KernelForge runs the task itself; see README.md."""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from . import workspace
from .schema import SchemaError, find, load

# The checkout this file is in: <root>/harness/scripts/forge_task/cli.py.
DEFAULT_AKLR_ROOT = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python harness/scripts/forge_task")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="write the task: workspace, driver, brief, anchor, KernelForge's commands",
    )
    which = prepare.add_mutually_exclusive_group(required=True)
    which.add_argument("--schema", help="name of a kernel-less schema in --aklr-root")
    which.add_argument("--schema-file", type=Path, help="a kernel-less schema file")
    prepare.add_argument(
        "--aklr-root",
        type=Path,
        default=DEFAULT_AKLR_ROOT,
        help="the AKLR checkout the driver validates in; nobody else may work in it",
    )
    prepare.add_argument(
        "--run-dir",
        type=Path,
        help="an empty directory for the task (default: ./aklr-runs/<schema>-<time>)",
    )
    prepare.add_argument(
        "--starting-kernel",
        type=Path,
        help="kernel.py to start from, or a directory whose *.py files, kernel.py among them, are the kernel's "
        "modules; gemm schemas default to torch",
    )
    prepare.add_argument("--kernel-backend", choices=("flydsl", "triton", "gluon"), default="flydsl")
    prepare.add_argument("--gpu-target", default="gfx950")
    prepare.add_argument(
        "--anchor",
        choices=("baseline", "start"),
        default="baseline",
        help="score against AKLR's baseline (default) or the starting kernel",
    )

    publish = commands.add_parser("publish", help="write the task's kernel.py into the AKLR checkout")
    publish.add_argument("run_dir", type=Path)
    return parser


def _prepare(args: argparse.Namespace) -> int:
    aklr_root = args.aklr_root.resolve()
    try:
        schema = load(args.schema_file or find(aklr_root, args.schema))
        starting_kernel = (
            workspace.read_starting_kernel(args.starting_kernel)
            if args.starting_kernel
            else workspace.builtin_starting_kernel(schema, aklr_root)
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run = workspace.materialize(
            schema,
            args.run_dir or Path.cwd() / "aklr-runs" / f"{schema.name}-{stamp}",
            aklr_root=aklr_root,
            language=args.kernel_backend,
            starting_kernel=starting_kernel,
            anchor=args.anchor,
        )
        workspace.check_starting_kernel(run)
        anchored = args.anchor == "baseline"
        if anchored:
            workspace.write_baseline(run)
    except (
        SchemaError,
        FileExistsError,
        workspace.StartingKernelError,
        workspace.AnchorError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    workspace.commit(run)
    start, resume = workspace.forge_loop_commands(
        run,
        aklr_root=aklr_root,
        language=args.kernel_backend,
        gpu_target=args.gpu_target,
        anchored=anchored,
    )
    workspace.write_script(run.start_script, start)
    workspace.write_script(run.resume_script, resume)
    print(f"task: {run.root} ({len(schema.workloads)} workloads)")
    print(f"KernelForge's command, also in {run.start_script}:")
    print(f"  {shlex.join(start)}")
    print(f"to resume it after an interruption: {run.resume_script}")
    return 0


def _publish(args: argparse.Namespace) -> int:
    run = workspace.Run.at(args.run_dir)
    if not run.driver.is_file():
        print(f"error: {run.root} holds no task", file=sys.stderr)
        return 2
    completed = workspace.run_driver(run, "--publish")
    if completed.returncode:
        sys.stderr.write(completed.stdout + completed.stderr)
        return completed.returncode
    path = completed.stdout.strip()
    print(f"published: {path}")
    print(f"next: python -m aklr export check && python -m aklr validate --base origin/main {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return _prepare(args) if args.command == "prepare" else _publish(args)
