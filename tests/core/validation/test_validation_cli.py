# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import subprocess

from aklr.core.validation.command import main


def test_empty_validation_skips_every_stage(tmp_path, monkeypatch):
    # command.main is called by aklr/__main__.py. Use its real filesystem output;
    # no bench classes, private imports, or internal replacements enter this test.
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
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "report"

    options = ["--num-trials", "5", "--isolate", "fused_moe", "--isolate", "mla"]
    assert main(["--devices", "0", "--out", str(output), *options]) == 0

    report = json.loads((output / "report.json").read_text())
    assert [(stage["stage_name"], stage["status"]) for stage in report["stages"]] == [
        ("audit", "skipped"),
        ("bench", "skipped"),
    ]
