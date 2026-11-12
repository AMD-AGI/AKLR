# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""mega_moe in the bench: its execution policy, and its kernels on their expert-parallel group."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import aklr.ops
from aklr.core.validation import ExecutionPolicy
from tests.op_case import OpCase

_AKLR = Path(aklr.ops.__file__).parents[2]


def test_a_mega_moe_task_runs_on_ep_size_ranks_in_fresh_processes() -> None:
    case = OpCase.from_id("mega_moe_gate_up_interleaved")
    policy = ExecutionPolicy.for_schema(case.schema())
    assert (policy.ranks_per_task, policy.isolate) == (case.axes["ep_size"], True)


@pytest.mark.gpu(8)
@pytest.mark.gfx950
def test_every_workload_passes_accuracy_on_eight_ranks(tmp_path: Path) -> None:
    paths = sorted(Path(aklr.ops.__file__).parent.glob("mega_moe/impls/flydsl/*/*.json"))
    argv = [sys.executable, "-m", "aklr", "validate", "--stages", "audit,bench", "--devices", "0-7"]
    done = subprocess.run(
        [*argv, "--out", str(tmp_path), *map(str, paths)],
        cwd=_AKLR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (tmp_path / "report.json").is_file(), done.stdout[-3000:] + done.stderr[-3000:]
    bench = json.loads((tmp_path / "report.json").read_text())["stages"][1]
    # no Baseline yet: accuracy alone decides, on every rank of the group
    assert [r["verdict"] for r in bench["records"]] == ["passed"] * 19
