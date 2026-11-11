# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Write the KernelForge task for one AKLR schema. The schema copy and AKLR's validation reports live beside the
workspace, not in it: the driver writes them on every measurement, which KernelForge's workspace guard would read
as tampering."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .schema import Schema

ASSETS = Path(__file__).resolve().parent / "assets"

# KernelForge keeps its state in forge_experiments/ and stops at the next round when .stop appears; neither may
# count as a change to the workspace.
GITIGNORE = "__pycache__/\n*.pyc\n*.log\nbuild/\nforge_experiments/\n.stop\n"

# What the agents read, per kernel language: aklr-kb, and its folder for the language (each has a SKILL.md).
WIKI = Path("harness/skills/aklr-kb")
SKILLS = {
    "flydsl": (WIKI, WIKI / "flydsl"),
    "triton": (WIKI, WIKI / "triton"),
    "gluon": (WIKI, WIKI / "triton"),
}

# A torch statement that computes a plain gemm-contract schema into its preallocated output, by the operand rank.
_TORCH_GEMMS = {
    2: "torch.matmul(a, b.t() if trans_b else b, out=out)",
    3: "torch.bmm(a, b.transpose(1, 2) if trans_b else b, out=out)",
}

_ANCHORS = {
    "baseline": "A score is AKLR's baseline time over the candidate's, per workload, averaged over the workloads: "
    "above 1.0 is faster than the baseline on average, not on every workload.",
    "start": "A score is the starting kernel's time over the candidate's, per workload, averaged over the "
    "workloads: above 1.0 is faster than `kernel.py` was when the task began, not faster than AKLR's baseline.",
}


class StartingKernelError(RuntimeError):
    """The task has no starting kernel, or AKLR does not pass it on every workload."""


class AnchorError(RuntimeError):
    """AKLR's baseline times could not be written as the scoring anchor."""


@dataclass(frozen=True)
class Run:
    """The directory one task owns: KernelForge's workspace and, beside it, what the driver writes."""

    root: Path

    @classmethod
    def at(cls, root: Path) -> Run:
        return cls(root=Path(root).resolve())

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def kernel(self) -> Path:
        return self.workspace / "kernel.py"

    @property
    def driver(self) -> Path:
        return self.workspace / "driver.py"

    @property
    def program_md(self) -> Path:
        return self.workspace / "program.md"

    @property
    def schema_json(self) -> Path:
        return self.root / "schema.json"

    @property
    def validate_dir(self) -> Path:
        return self.root / "aklr_validate"

    @property
    def baseline_json(self) -> Path:
        return self.root / "baseline.json"

    @property
    def start_script(self) -> Path:
        return self.root / "forge_loop.sh"

    @property
    def resume_script(self) -> Path:
        return self.root / "forge_loop_resume.sh"


def read_starting_kernel(path: Path) -> dict[str, str]:
    """The starting kernel's files by name: a file is ``kernel.py``; a directory gives every ``*.py`` directly in it,
    ``kernel.py`` among them, which AKLR imports as the modules of one package."""
    if path.is_file():
        return {"kernel.py": path.read_text(encoding="utf-8")}
    files = {file.name: file.read_text(encoding="utf-8") for file in sorted(path.glob("*.py"))}
    if "kernel.py" not in files or "driver.py" in files:
        raise StartingKernelError(
            f"{path}: a starting kernel's directory holds kernel.py, the file whose run AKLR calls, "
            "and no driver.py, the task's driver"
        )
    return files


def builtin_starting_kernel(schema: Schema, aklr_root: Path) -> dict[str, str]:
    """A gemm-contract schema's starting kernel, one file writing into the preallocated output: torch's product for
    a plain gemm, and for a scaled or quantized one, which torch cannot compute, its contract's AKLR baseline."""
    if schema.op_type != "gemm":
        raise StartingKernelError(
            f"no built-in starting kernel for op {schema.op_type!r}; pass --starting-kernel"
        )
    # The keyword arguments AKLR calls the entry with, and the contract's baseline as [module, name] or null.
    completed = subprocess.run(
        [sys.executable, str(ASSETS / "aklr_api_params.py"), str(aklr_root), str(schema.path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise StartingKernelError(f"AKLR could not read {schema.path}:\n{completed.stderr[-2000:]}")
    entry = json.loads(completed.stdout)
    params = entry["params"]
    signature = f"def run(*, {', '.join(params)}):\n"
    if set(params) <= {"a", "b", "trans_b", "bias", "out_dtype", "out"}:
        rank = len(schema.inputs["a"]["shape"])
        return {
            "kernel.py": (
                f'"""Starting point for {schema.name}: torch writes the product into the preallocated output."""\n\n'
                f"import torch\n\n\n{signature}    {_TORCH_GEMMS[rank]}\n"
            )
        }
    if entry["baseline"] is None:
        raise StartingKernelError(
            f"{schema.name}: torch cannot compute its contract and AKLR has no baseline for it; pass --starting-kernel"
        )
    module, name = entry["baseline"]
    call = ", ".join(f"{param}={param}" for param in params)
    return {
        "kernel.py": (
            f'"""Starting point for {schema.name}: AKLR\'s baseline for its contract, its result in the preallocated '
            'output."""\n\n'
            f"from {module} import {name}\n\n\n{signature}"
            f"    result = {name}({call})\n"
            "    if result is not out:\n"
            "        out.copy_(result)\n"
        )
    }


def _fill(template: str, values: dict[str, str]) -> str:
    for placeholder, value in values.items():
        template = template.replace(f"@{placeholder}@", value)
    unfilled = sorted(set(re.findall(r"@[A-Z_]+@", template)))
    if unfilled:
        raise ValueError(f"template still carries unfilled placeholders: {unfilled}")
    return template


def _workload_table(schema: Schema) -> str:
    names = list(schema.variable_axes)
    rows = [
        "| case | " + " | ".join(f"`{name}`" for name in names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for workload in schema.workloads:
        values = " | ".join(str(workload.axes.get(name, "")) for name in names)
        rows.append(f"| `{workload.case_id}` | {values} |")
    return "\n".join(rows)


def program_md(schema: Schema, *, aklr_root: Path, language: str, anchor: str, schema_json: Path) -> str:
    """The implementer's brief: the task, AKLR's contract for it, and its workloads."""
    constants = ", ".join(f"{k}={v}" for k, v in schema.constant_axes.items())
    return _fill(
        (ASSETS / "program.md").read_text(encoding="utf-8"),
        {
            "NAME": schema.name,
            "LANGUAGE": language,
            "N_WORKLOADS": str(len(schema.workloads)),
            "OP_TYPE": schema.op_type,
            "DESCRIPTION": schema.description or "see the schema",
            "ATTRS": json.dumps(schema.attrs),
            "INPUTS": ", ".join(f"`{name}`" for name in schema.inputs),
            "OUTPUTS": ", ".join(f"`{name}`" for name in schema.outputs),
            "SCHEMA_JSON": str(schema_json),
            "CONSTANT_AXES": constants or "none",
            "VARIABLE_AXES": ", ".join(schema.variable_axes) or "none",
            "COMPARE": json.dumps(schema.compare),
            "AKLR_ROOT": str(aklr_root),
            "WORKLOAD_TABLE": _workload_table(schema),
            "ANCHOR": _ANCHORS[anchor],
        },
    )


def materialize(
    schema: Schema,
    root: Path,
    *,
    aklr_root: Path,
    language: str,
    starting_kernel: dict[str, str],
    anchor: str,
) -> Run:
    """Create the task directory, the starting kernel's files by name beside the driver. A non-empty directory is
    refused: the first commit takes the whole workspace, leftovers included."""
    run = Run.at(root)
    if run.root.exists() and any(run.root.iterdir()):
        raise FileExistsError(f"run directory {run.root} is not empty; give each task its own directory")
    run.workspace.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(schema.path, run.schema_json)
    for name, text in starting_kernel.items():
        (run.workspace / name).write_text(text, encoding="utf-8")
    driver = _fill(
        (ASSETS / "aklr_forge_driver.py").read_text(encoding="utf-8"),
        {
            "AKLR_ROOT": str(aklr_root),
            "SCHEMA_JSON": str(run.schema_json),
            "OP_DIR": schema.op_dir,
            "LANGUAGE": language,
            "VALIDATE_DIR": str(run.validate_dir),
        },
    )
    run.driver.write_text(driver, encoding="utf-8")
    run.program_md.write_text(
        program_md(
            schema,
            aklr_root=aklr_root,
            language=language,
            anchor=anchor,
            schema_json=run.schema_json,
        ),
        encoding="utf-8",
    )
    (run.workspace / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    return run


def run_driver(run: Run, *args: str) -> subprocess.CompletedProcess:
    """Run the task's driver in its workspace, as KernelForge does, and capture what it prints."""
    return subprocess.run(
        [sys.executable, str(run.driver), *args],
        cwd=run.workspace,
        capture_output=True,
        text=True,
        check=False,
    )


def check_starting_kernel(run: Run) -> None:
    """Stop unless AKLR passes the starting kernel on every workload: every candidate is measured from it."""
    completed = run_driver(run)
    verdict = re.search(r"^allclose: (True|False)$", completed.stdout, re.MULTILINE)
    if verdict is None:
        raise StartingKernelError(
            f"AKLR gave no verdict:\n{completed.stdout[-2000:]}{completed.stderr[-2000:]}"
        )
    if verdict.group(1) != "True":
        findings = "\n".join(
            line for line in completed.stdout.splitlines() if line.startswith("aklr_findings:")
        )
        raise StartingKernelError(f"AKLR does not pass the starting kernel on every workload:\n{findings}")


def write_baseline(run: Run) -> Path:
    """KernelForge's scoring anchor: per workload, AKLR's baseline time from the starting kernel's validate run."""
    completed = run_driver(run, "--baseline-json", str(run.baseline_json))
    if completed.returncode:
        raise AnchorError(f"{completed.stdout[-2000:]}{completed.stderr[-2000:]}".strip())
    return run.baseline_json


def commit(run: Run) -> None:
    """KernelForge keeps and reverts candidates with git: one first commit of the whole workspace."""
    git = ["git", "-C", str(run.workspace)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    identity = ["-c", "user.name=forge_task", "-c", "user.email=forge_task@local"]
    subprocess.run([*git, *identity, "commit", "-q", "-m", "forge_task: the task"], check=True)


def forge_loop_commands(
    run: Run, *, aklr_root: Path, language: str, gpu_target: str, anchored: bool
) -> tuple[list[str], list[str]]:
    """KernelForge's own commands for this task: the one that starts the loop and the one that resumes it. Each
    candidate is timed beside the kernel kept so far, the loop stops once its known directions are mined out, and
    the agents read aklr-kb and its folder for the language. A resume keeps the kernel, driver, brief, backend
    and branch the loop started with and refuses the first three again; every other option holds only when given
    again.
    """
    loop = [sys.executable, "-m", "kernelforge.cli", "forge-loop"]
    options = [
        "--workspace",
        str(run.workspace),
        "--gpu-target",
        gpu_target,
        "--paired-keep",
        "--early-stop",
    ]
    for skill in SKILLS.get(language, ()):
        options += ["--skill-dir", str(aklr_root / skill)]
    if anchored:
        options += ["--baseline-json", str(run.baseline_json)]
    # The starting kernel's files, which KernelForge reads as the kernel's beside its --kernel.
    sources = sorted(p.name for p in run.workspace.glob("*.py") if p != run.driver)
    start = [
        *loop,
        "--kernel",
        str(run.kernel),
        "--driver",
        str(run.driver),
        "--program-md-file",
        str(run.program_md),
        *(["--source-files", ",".join(sources)] if len(sources) > 1 else []),
        "--kernel-backend",
        language,
        "--git-branch",
        "forge-optimize",
        *options,
    ]
    return start, [*loop, "--resume", *options]


def write_script(path: Path, command: list[str]) -> Path:
    """A command as a script; arguments given to it (--max-hours, --model, --gpu, ...) go to forge-loop."""
    path.write_text(f'#!/bin/sh\nexec {shlex.join(command)} "$@"\n', encoding="utf-8")
    path.chmod(0o755)
    return path
