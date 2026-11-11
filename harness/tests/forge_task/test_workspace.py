# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What a task directory holds before KernelForge sees it, and which schema it is built from."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fake_aklr import CASES, SCHEMA_NAME, install_aklr, schema_payload, write_schema
from forge_task import workspace
from forge_task.schema import SchemaError, find, load

KERNEL = '"""Start."""\n\nimport torch\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    torch.matmul(a, b.t(), out=out)\n'


def _materialize(
    aklr_root: Path, tmp_path: Path, kernel: str = KERNEL, anchor: str = "baseline"
) -> workspace.Run:
    return workspace.materialize(
        load(find(aklr_root, SCHEMA_NAME)),
        tmp_path / "run",
        aklr_root=aklr_root,
        language="flydsl",
        starting_kernel={"kernel.py": kernel},
        anchor=anchor,
    )


def _tracked(run: workspace.Run) -> list[str]:
    return sorted(
        subprocess.run(
            ["git", "-C", str(run.workspace), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )


def test_what_the_driver_writes_lives_outside_the_workspace(aklr_root: Path, tmp_path: Path):
    """The driver writes AKLR's reports on every measurement; inside the workspace KernelForge's guard reads
    that as tampering."""
    run = _materialize(aklr_root, tmp_path)
    assert run.schema_json.is_file()
    assert run.workspace not in run.schema_json.parents
    assert run.workspace not in run.validate_dir.parents


def test_the_first_commit_holds_the_whole_task(aklr_root: Path, tmp_path: Path):
    run = _materialize(aklr_root, tmp_path)
    workspace.commit(run)
    assert _tracked(run) == [".gitignore", "driver.py", "kernel.py", "program.md"]


def test_a_stop_file_is_not_a_change_to_the_workspace(aklr_root: Path, tmp_path: Path):
    """KernelForge stops at its next round when `.stop` appears; an untracked file would make its workspace
    guard refuse the rounds after."""
    run = _materialize(aklr_root, tmp_path)
    workspace.commit(run)
    (run.workspace / ".stop").touch()
    status = subprocess.run(
        ["git", "-C", str(run.workspace), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ""


def test_templates_carry_no_unfilled_placeholder(aklr_root: Path, tmp_path: Path):
    run = _materialize(aklr_root, tmp_path)
    for path in (run.driver, run.program_md):
        assert not re.search(r"@[A-Z_]+@", path.read_text(encoding="utf-8")), path


def test_the_brief_states_the_operator_and_points_at_aklrs_rules(aklr_root: Path, tmp_path: Path):
    """The implementer learns what the operator computes from the brief, not by reading the driver."""
    run = _materialize(aklr_root, tmp_path)
    brief = run.program_md.read_text(encoding="utf-8")
    assert "`gemm`: 16-bit gemm." in brief
    assert '`{"trans_b": true, "has_bias": false}`' in brief
    assert "Inputs: `a`, `b`. Outputs: `out`." in brief
    assert str(run.schema_json) in brief
    assert all(f"`{case_id}`" in brief for case_id in CASES)
    assert f"{aklr_root}/aklr/ops/README.md" in brief
    assert "bitwise" in brief
    assert "faster than the baseline on average" in brief


def test_the_start_anchor_is_said_in_the_brief(aklr_root: Path, tmp_path: Path):
    run = _materialize(aklr_root, tmp_path, anchor="start")
    assert "not faster than AKLR's baseline" in run.program_md.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("rank", "statement"),
    [(2, "torch.matmul(a, b.t()"), (3, "torch.bmm(a, b.transpose(1, 2)")],
)
def test_a_gemm_schema_starts_from_torch_with_its_contracts_call_keys(tmp_path: Path, rank, statement):
    root = install_aklr(tmp_path / "aklr")
    schema = load(write_schema(root, schema_payload(rank=rank)))
    (kernel,) = workspace.builtin_starting_kernel(schema, root).values()
    assert "def run(*, a, b, trans_b, bias, out_dtype, out):" in kernel
    assert statement in kernel
    compile(kernel, "kernel.py", "exec")


def test_another_op_has_no_built_in_starting_kernel(tmp_path: Path):
    root = install_aklr(tmp_path / "aklr")
    schema = load(write_schema(root, schema_payload("mla_demo", op_type="mla")))
    with pytest.raises(workspace.StartingKernelError, match="--starting-kernel"):
        workspace.builtin_starting_kernel(schema, root)


def test_a_scaled_gemm_schema_starts_from_its_contracts_baseline_into_the_output(tmp_path: Path):
    root = install_aklr(tmp_path / "aklr")
    payload = schema_payload()
    payload["inputs"]["a_scale"] = {"shape": ["m", 1], "dtype": "float32"}
    payload["inputs"]["b_scale"] = {"shape": [1, 1], "dtype": "float32"}
    schema = load(write_schema(root, payload))
    (kernel,) = workspace.builtin_starting_kernel(schema, root).values()
    assert "torch" not in kernel
    assert "from aklr.core.contract.registry import gemm_scaled_baseline" in kernel
    assert "def run(*, a, b, a_scale, b_scale, trans_b, bias, out_dtype, out):" in kernel
    assert "out.copy_(result)" in kernel
    compile(kernel, "kernel.py", "exec")


def test_a_starting_kernel_aklr_passes_is_kept(aklr_root: Path, tmp_path: Path):
    workspace.check_starting_kernel(_materialize(aklr_root, tmp_path))


def test_a_starting_kernel_aklr_fails_stops_with_its_findings(aklr_root: Path, tmp_path: Path):
    run = _materialize(
        aklr_root,
        tmp_path,
        kernel='"""Wrong."""\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    pass\n',
    )
    with pytest.raises(workspace.StartingKernelError, match="numeric_mismatch"):
        workspace.check_starting_kernel(run)


def test_aklr_without_a_verdict_is_not_read_as_a_failing_kernel(aklr_root: Path, tmp_path: Path):
    run = _materialize(aklr_root, tmp_path)
    (aklr_root / "aklr" / "__main__.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    with pytest.raises(workspace.StartingKernelError, match="no verdict"):
        workspace.check_starting_kernel(run)


def test_the_schema_found_by_name_is_the_kernel_less_one(aklr_root: Path):
    kernel = [{"path": "kernel.py", "content": "def run():\n    pass\n"}]
    write_schema(aklr_root, schema_payload(sources=kernel), uuid="fedcba9876543210")
    assert not load(find(aklr_root, SCHEMA_NAME)).path.name.startswith("fedcba")


def test_a_missing_schema_names_where_it_looked(aklr_root: Path):
    with pytest.raises(SchemaError, match="no kernel-less schema 'absent'"):
        find(aklr_root, "absent")


def test_two_kernel_less_schemas_of_one_name_are_refused(aklr_root: Path):
    write_schema(aklr_root, schema_payload(), uuid="fedcba9876543210")
    with pytest.raises(SchemaError, match="expected one"):
        find(aklr_root, SCHEMA_NAME)


def test_a_starting_kernels_directory_gives_each_module_by_name(tmp_path: Path):
    (tmp_path / "kernel.py").write_text("from .gemm import gemm\n", encoding="utf-8")
    (tmp_path / "gemm.py").write_text("def gemm(): pass\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("not a module\n", encoding="utf-8")
    assert sorted(workspace.read_starting_kernel(tmp_path)) == ["gemm.py", "kernel.py"]
    assert list(workspace.read_starting_kernel(tmp_path / "gemm.py")) == ["kernel.py"]


@pytest.mark.parametrize("files", [["gemm.py"], ["kernel.py", "driver.py"]])
def test_a_starting_kernels_directory_needs_kernel_py_and_no_driver(tmp_path: Path, files):
    for name in files:
        (tmp_path / name).write_text("\n", encoding="utf-8")
    with pytest.raises(workspace.StartingKernelError, match="kernel.py"):
        workspace.read_starting_kernel(tmp_path)


def test_kernelforge_is_given_every_file_of_a_kernel_of_several(aklr_root: Path, tmp_path: Path):
    """--source-files names the modules beside --kernel; a resume refuses it again, like --kernel."""
    run = workspace.materialize(
        load(find(aklr_root, SCHEMA_NAME)),
        tmp_path / "run",
        aklr_root=aklr_root,
        language="flydsl",
        starting_kernel={"kernel.py": KERNEL, "gemm.py": "\n"},
        anchor="baseline",
    )
    start, resume = workspace.forge_loop_commands(
        run, aklr_root=aklr_root, language="flydsl", gpu_target="gfx950", anchored=True
    )
    assert start[start.index("--source-files") + 1] == "gemm.py,kernel.py"
    assert "--source-files" not in resume
    single = _materialize(aklr_root, tmp_path / "single")
    start, _ = workspace.forge_loop_commands(
        single,
        aklr_root=aklr_root,
        language="flydsl",
        gpu_target="gfx950",
        anchored=True,
    )
    assert "--source-files" not in start


def test_the_op_folder_comes_from_where_aklr_keeps_the_schema(aklr_root: Path):
    assert load(find(aklr_root, SCHEMA_NAME)).op_dir == "gemm"


@pytest.mark.parametrize("parent", ["copy", "a/b/c/d"])
def test_a_schema_outside_aklr_ops_is_refused(aklr_root: Path, tmp_path: Path, parent):
    """A delivered kernel goes to the schema's op folder, which only AKLR's layout names."""
    copy = tmp_path / parent / "schema.json"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(find(aklr_root, SCHEMA_NAME), copy)
    with pytest.raises(SchemaError, match="not under aklr/ops"):
        load(copy)
