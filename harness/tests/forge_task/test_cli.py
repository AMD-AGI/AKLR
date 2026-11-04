# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The command surface: what an operator types, what it leaves on disk, and the KernelForge command it gives."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from fake_aklr import SCHEMA_NAME, install_aklr, schema_payload, write_schema
from forge_task import cli

WRONG_KERNEL = '"""Wrong."""\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    pass\n'


def _prepare(aklr_root: Path, tmp_path: Path, *extra: str) -> int:
    return cli.main(
        [
            "prepare",
            "--aklr-root",
            str(aklr_root),
            "--schema",
            SCHEMA_NAME,
            "--run-dir",
            str(tmp_path / "run"),
            *extra,
        ]
    )


def _printed_command(out: str) -> list[str]:
    line = out.split("KernelForge's command")[1].splitlines()[1]
    return shlex.split(line)


def _script_command(script: Path) -> list[str]:
    words = shlex.split(script.read_text(encoding="utf-8").split("exec ", 1)[1])
    assert words[-1] == "$@"
    return words[:-1]


def test_prepare_writes_the_task_and_prints_kernelforges_command(aklr_root: Path, tmp_path: Path, capsys):
    assert _prepare(aklr_root, tmp_path) == 0
    run_dir = tmp_path / "run"
    for relative in (
        "workspace/kernel.py",
        "workspace/driver.py",
        "workspace/program.md",
        "schema.json",
        "baseline.json",
    ):
        assert (run_dir / relative).is_file(), relative
    assert (run_dir / "workspace" / ".git").is_dir()
    command = _printed_command(capsys.readouterr().out)
    assert command[1:4] == ["-m", "kernelforge.cli", "forge-loop"]
    workspace_dir = run_dir.resolve() / "workspace"
    assert command[command.index("--kernel") + 1] == str(workspace_dir / "kernel.py")
    assert command[command.index("--driver") + 1] == str(workspace_dir / "driver.py")
    assert command[command.index("--baseline-json") + 1] == str(run_dir.resolve() / "baseline.json")
    assert _skill_dirs(command) == [
        str(aklr_root / "harness" / "skills" / "aklr-kb"),
        str(aklr_root / "harness" / "skills" / "aklr-kb/flydsl"),
    ]
    assert {"--paired-keep", "--early-stop"} <= set(command)


def test_the_saved_command_passes_its_arguments_to_forge_loop(aklr_root: Path, tmp_path: Path, capsys):
    assert _prepare(aklr_root, tmp_path) == 0
    script = tmp_path / "run" / "forge_loop.sh"
    assert os.access(script, os.X_OK)
    assert _script_command(script) == _printed_command(capsys.readouterr().out)
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\nexec ")
    echoed = subprocess.run(
        ["sh", "-c", text.replace("exec ", "exec echo ", 1), "sh", "--max-hours", "3"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert echoed.split()[-2:] == ["--max-hours", "3"]


def test_the_resume_script_gives_every_option_again_but_what_a_resume_refuses(
    aklr_root: Path, tmp_path: Path, capsys
):
    """forge-loop keeps the kernel, driver and brief it started with and refuses them on --resume; the other
    options hold for one invocation only, so a resume without them would score against another anchor.
    """
    assert _prepare(aklr_root, tmp_path) == 0
    start = _printed_command(capsys.readouterr().out)
    script = tmp_path / "run" / "forge_loop_resume.sh"
    assert os.access(script, os.X_OK)
    resume = _script_command(script)
    assert resume[:4] == start[:4] and "--resume" in resume
    assert not {"--kernel", "--driver", "--program-md-file"} & set(resume)
    for option in ("--workspace", "--gpu-target", "--skill-dir", "--baseline-json"):
        assert resume[resume.index(option) + 1] == start[start.index(option) + 1]
    assert {"--paired-keep", "--early-stop"} <= set(resume)


def test_the_start_anchor_writes_no_baseline(aklr_root: Path, tmp_path: Path, capsys):
    assert _prepare(aklr_root, tmp_path, "--anchor", "start") == 0
    assert not (tmp_path / "run" / "baseline.json").exists()
    assert "--baseline-json" not in _printed_command(capsys.readouterr().out)
    resume = _script_command(tmp_path / "run" / "forge_loop_resume.sh")
    assert "--baseline-json" not in resume


def _skill_dirs(command: list[str]) -> list[str]:
    return [command[i + 1] for i, arg in enumerate(command) if arg == "--skill-dir"]


@pytest.mark.parametrize(
    ("backend", "skill"),
    [("triton", "aklr-kb/triton"), ("gluon", "aklr-kb/triton")],
)
def test_the_agents_read_the_wiki_and_the_skill_of_the_language(
    aklr_root: Path, tmp_path: Path, capsys, backend, skill
):
    assert _prepare(aklr_root, tmp_path, "--kernel-backend", backend) == 0
    command = _printed_command(capsys.readouterr().out)
    assert _skill_dirs(command) == [
        str(aklr_root / "harness" / "skills" / "aklr-kb"),
        str(aklr_root / "harness" / "skills" / skill),
    ]


def test_a_relative_aklr_root_reaches_the_driver_and_the_brief_absolute(
    aklr_root: Path, tmp_path: Path, monkeypatch
):
    """The driver runs inside the workspace and the agents read program.md there."""
    monkeypatch.chdir(aklr_root.parent)
    assert _prepare(Path(aklr_root.name), tmp_path) == 0
    workspace_dir = tmp_path / "run" / "workspace"
    driver = (workspace_dir / "driver.py").read_text(encoding="utf-8")
    assert f'AKLR_ROOT = Path("{aklr_root.resolve()}")' in driver
    brief = (workspace_dir / "program.md").read_text(encoding="utf-8")
    assert f"{aklr_root.resolve()}/aklr/ops/README.md" in brief


def test_a_schema_file_from_another_checkout_is_read_with_this_ones_contracts(
    aklr_root: Path, tmp_path: Path
):
    other = write_schema(install_aklr(tmp_path / "other"), schema_payload())
    argv = ["prepare", "--aklr-root", str(aklr_root), "--schema-file", str(other)]
    assert cli.main([*argv, "--run-dir", str(tmp_path / "run")]) == 0


def test_a_starting_kernel_aklr_fails_exits_two_before_the_first_commit(
    aklr_root: Path, tmp_path: Path, capsys
):
    wrong = tmp_path / "wrong.py"
    wrong.write_text(WRONG_KERNEL, encoding="utf-8")
    assert _prepare(aklr_root, tmp_path, "--starting-kernel", str(wrong)) == 2
    assert "does not pass the starting kernel" in capsys.readouterr().err
    assert not (tmp_path / "run" / "workspace" / ".git").exists()


def test_an_op_without_a_built_in_start_needs_one_given(tmp_path: Path, capsys):
    root = install_aklr(tmp_path / "aklr")
    write_schema(root, schema_payload("mla_demo", op_type="mla"))
    argv = ["prepare", "--aklr-root", str(root), "--schema", "mla_demo"]
    assert cli.main([*argv, "--run-dir", str(tmp_path / "run")]) == 2
    assert "pass --starting-kernel" in capsys.readouterr().err


def test_an_unknown_schema_exits_two(aklr_root: Path, tmp_path: Path, capsys):
    argv = ["prepare", "--aklr-root", str(aklr_root), "--schema", "absent"]
    assert cli.main([*argv, "--run-dir", str(tmp_path / "run")]) == 2
    assert "no kernel-less schema 'absent'" in capsys.readouterr().err


def test_a_run_directory_another_task_left_files_in_is_refused(aklr_root: Path, tmp_path: Path, capsys):
    """The first commit takes the whole workspace, so a leftover would become the starting point."""
    leftover = tmp_path / "run" / "workspace" / "graph_harness.py"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("# another task's file\n", encoding="utf-8")
    assert _prepare(aklr_root, tmp_path) == 2
    assert "is not empty" in capsys.readouterr().err


def test_a_schema_aklr_times_no_baseline_for_needs_the_start_anchor(tmp_path: Path, capsys):
    root = install_aklr(tmp_path / "aklr", timed_baseline=False)
    write_schema(root, schema_payload())
    assert _prepare(root, tmp_path) == 2
    assert "prepare with --anchor start" in capsys.readouterr().err


def test_publish_names_the_file_it_wrote_and_validates_that_file(aklr_root: Path, tmp_path: Path, capsys):
    """validate with no path checks every schema on a branch that also changes code: the next command names the file."""
    assert _prepare(aklr_root, tmp_path) == 0
    capsys.readouterr()
    assert cli.main(["publish", str(tmp_path / "run")]) == 0
    out = capsys.readouterr().out
    path = out.split("published: ")[1].splitlines()[0]
    assert (aklr_root / path).is_file()
    assert f"aklr validate --base origin/main {path}" in out


def test_publish_refuses_a_directory_without_a_task(tmp_path: Path, capsys):
    assert cli.main(["publish", str(tmp_path)]) == 2
    assert "holds no task" in capsys.readouterr().err


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])
