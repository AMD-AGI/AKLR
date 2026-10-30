# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The qwen2_moe contracts and the Qwen3.5 decode schemas they serve."""

from pathlib import Path

import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.validation.stages._audit._stage import check_kernel
from aklr.plugins.sglang.models.qwen2_moe.contract import (
    QWEN2_MOE_FP8_EXPERTS,
    QWEN2_MOE_FP8_ROUTER_EXPERTS,
    QWEN2_MOE_ROUTER,
)

IMPLS = Path(__file__).parents[3] / "aklr/plugins/sglang/models/qwen2_moe/impls"
# Qwen3.5-397B-A17B at TP4: each schema serves the one decode row count its kernel is for
POINTS = {
    "qwen2_moe_fp8_router_experts@1.0": [1, 2, 4, 32],
    "qwen2_moe_fp8_experts@1.0": [8, 64, 128],
    "qwen2_moe_router@1.0": [8, 16, 64, 128, 256],
}
SCENARIOS = {
    "qwen2_moe_fp8_router_experts@1.0": (
        "qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256"
    ),
    "qwen2_moe_fp8_experts@1.0": (
        "qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10"
    ),
    "qwen2_moe_router@1.0": "qwen2_moe_router_model_dim4096_n_routed512_topk10",
}


def schemas():
    registry = ContractRegistry.default()
    return [
        (path, registry.parse_schema(path.read_bytes(), str(path))) for path in sorted(IMPLS.rglob("*.json"))
    ]


def test_the_router_returns_the_routing_the_caller_holds():
    assert QWEN2_MOE_ROUTER.required_outputs == ("topk_weights", "topk_ids")
    assert QWEN2_MOE_FP8_EXPERTS.required_outputs == ()
    assert QWEN2_MOE_FP8_ROUTER_EXPERTS.required_outputs == ()


def test_the_experts_are_read_as_sglang_holds_them_for_aiter():
    for contract in (QWEN2_MOE_FP8_EXPERTS, QWEN2_MOE_FP8_ROUTER_EXPERTS):
        shuffled = contract.layout_variants.by_name["shuffled_16x16"]
        assert {f.name: f.expected_value for f in shuffled.detect_facts} == {
            "w1.is_shuffled": True,
            "w2.is_shuffled": True,
        }
        assert {n: (layout.kind, layout.tile) for n, layout in shuffled.layouts.items()} == {
            "w1": ("preshuffle", (16, 16)),
            "w2": ("preshuffle", (16, 16)),
        }
        # w1 and w2 hold the routed experts, then the shared one; scales as stored
        assert contract.inputs["w1"].shape == ("n_experts", "gate_up_rows", "model_dim")
        assert contract.inputs["w1_scale"].shape == (
            "n_experts",
            "gate_up_blocks",
            "model_blocks",
        )
        assert contract.inputs["shared_gate_weight"].shape == (1, "model_dim")


def test_the_shuffle_puts_each_fp8_weight_where_the_kernels_read_it():
    """aiter's shuffle_weight(w, (16, 16)) of an fp8 [n, k] weight, the layout sglang holds:
    element (n, k) at (n // 16) * 16k + (k // 16) * 256 + (n % 16) * 16 + k % 16."""
    rows, cols = 32, 256
    shuffled = QWEN2_MOE_FP8_EXPERTS.layout_variants.by_name["shuffled_16x16"]
    codes = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols)
    weight = (codes % 251).to(torch.uint8).view(torch.float8_e4m3fn)
    stored = shuffled.layouts["w1"].encode(weight).view(torch.uint8).flatten()
    n, k = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    offset = (n // 16) * (cols * 16) + (k // 16) * 256 + (n % 16) * 16 + k % 16
    assert torch.equal(stored[offset], weight.view(torch.uint8))


def test_each_kernel_is_one_self_contained_openai_schema_at_its_decode_point():
    found = {}
    for path, schema in schemas():
        kernel = schema.kernel
        assert kernel.language == "triton" and kernel.author == "openai"
        assert kernel.requires == {"triton": ">=3.8"}
        assert kernel.entry_point == "kernel.py::run"
        assert not kernel.state_bytes and not kernel.peer_bytes
        (source,) = kernel.sources
        assert source.path == "kernel.py" and len(schema.workloads) == 1
        assert source.content.startswith("# Created by OpenAI\n")
        assert schema.name == SCENARIOS[schema.contract_id] == path.parent.name
        # only CI measures
        assert all(w.accuracy is None and w.speedup is None for w in schema.workloads)
        findings = list(
            check_kernel(
                str(path),
                kernel,
                call_keys=schema.kernel_call_keys,
                tensor_names=tuple(schema.tensors()),
                mutable_names=schema.mutable_tensors,
            )
        )
        assert findings == []
        found.setdefault(schema.contract_id, []).extend(w.axes["n_tokens"] for w in schema.workloads)
    assert {c: sorted(m) for c, m in found.items()} == POINTS
