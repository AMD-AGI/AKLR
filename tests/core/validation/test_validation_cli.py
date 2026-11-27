# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import subprocess

import pytest

from aklr.core.validation.command import main


@pytest.fixture
def empty_repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-qm",
            "empty repository",
        ],
        check=True,
    )
    (tmp_path / "aklr/plugins").mkdir(parents=True)  # the scope lists the op directories
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _statuses(directory):
    report = json.loads((directory / "report.json").read_text())
    return [(validator["name"], validator["status"]) for validator in report["validators"]]


def test_empty_validation_skips_every_validator(empty_repo):
    # command.main is called by aklr/__main__.py. Use its real filesystem output;
    # no bench classes, private imports, or internal replacements enter this test.
    output = empty_repo / "report"
    options = ["--num-trials", "5", "--isolate", "fused_moe", "--isolate", "mla"]
    assert main(["--devices", "0", "--out", str(output), *options]) == 0
    assert _statuses(output) == [("audit", "skipped"), ("bench", "skipped")]


@pytest.mark.parametrize(
    ("names", "ran"),
    [
        pytest.param("audit", ["audit"], id="audit_alone"),
        pytest.param("bench", ["bench"], id="bench_alone"),
        pytest.param("bench,audit", ["bench", "audit"], id="in_the_order_written"),
    ],
)
def test_the_command_runs_exactly_the_validators_named(empty_repo, names, ran):
    output = empty_repo / "report"
    assert main(["--validators", names, "--devices", "0", "--out", str(output)]) == 0
    assert _statuses(output) == [(name, "skipped") for name in ran]


@pytest.mark.parametrize(
    ("names", "message"),
    [
        pytest.param("audit,lint", "unknown Validator: 'lint'; available: audit, bench", id="unknown"),
        pytest.param("audit,audit", "validator 'audit' is named twice", id="repeated"),
        pytest.param("", "unknown Validator: ''; available: audit, bench", id="empty"),
    ],
)
def test_a_name_that_is_unknown_or_repeated_is_a_usage_error(empty_repo, capsys, names, message):
    with pytest.raises(SystemExit) as exit_info:
        main(["--validators", names, "--out", str(empty_repo / "report")])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not (empty_repo / "report").exists()
