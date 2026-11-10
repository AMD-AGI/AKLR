# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The small-token MoE schema must not silently accept the large-token algorithm."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_small_moe_entry_preserves_coverage_and_rejects_large_tokens():
    impls = Path(__file__).parents[2] / "aklr/ops/fused_moe/impls/flydsl"
    paths = impls.glob(
        "fused_moe_swiglu_has_swiglu_limit_shuffled_16x16_model_dim6144_n_experts128_topk4_inter_dim768/*.json"
    )
    (data,) = [
        json.loads(p.read_text())
        for p in paths
        if json.loads(p.read_text())["workloads"][0]["axes"]["n_tokens"] == 1
    ]
    entry = next(
        n
        for n in ast.parse(data["kernel"]["sources"][0]["content"]).body
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    launches = []

    def build(m, *args):
        return lambda *inputs: launches.append(m)

    scope = {"_build": build}
    exec(compile(ast.Module(body=[entry], type_ignores=[]), "kernel.py", "exec"), scope)
    call = dict.fromkeys(arg.arg for arg in entry.args.kwonlyargs)
    call.update(w1=SimpleNamespace(shape=(128, 1536)), topk_ids=SimpleNamespace(shape=(1, 4)))
    rows = [w["axes"]["n_tokens"] for w in data["workloads"]]
    assert max(rows) < 256
    for m in rows:
        call["hidden_states"] = SimpleNamespace(shape=(m, 6144), device="cpu")
        scope["run"](**call)
    assert launches == rows
    for m in (256, 272):
        call["hidden_states"] = SimpleNamespace(shape=(m, 6144), device="cpu")
        with pytest.raises(ValueError, match="fewer than 256 tokens"):
            scope["run"](**call)
    assert launches == rows
