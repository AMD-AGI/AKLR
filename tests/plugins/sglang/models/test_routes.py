# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the model routes leave to sglang before asking an op, on stand-ins of sglang 0.5.23."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.deepseek_v2 import api as deepseek_v2
from aklr.plugins.sglang.models.qwen3_5 import api as qwen3_5
from aklr.plugins.sglang.models.qwen3_next import api as qwen3_next


@pytest.fixture
def moe_checks(monkeypatch):
    """The checks after DeepSeek-V2's own, recorded when reached, which send the call back to sglang."""
    reached = []
    monkeypatch.setattr(deepseek_v2.moe_blocks, "runtime_falls_back", lambda: reached.append(True) or True)
    monkeypatch.setattr(deepseek_v2, "_moe_call", lambda *args: pytest.fail("an op was asked"))
    return reached


def forward_context(monkeypatch, bound):
    """sglang's forward context, its MoE padding bound ``bound``."""
    module = ModuleType("sglang.srt.model_executor.forward_context")
    module.has_forward_context = lambda: True
    module.get_forward_context = lambda: SimpleNamespace(moe_num_token_non_padded=lambda: bound)
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize(
    ("bound", "masked"), [(None, False), (torch.tensor([3]), True)], ids=["no_padding", "padded_rows"]
)
def test_a_deepseek_v2_moe_whose_padded_rows_sglang_masks_stays_on_sglang(
    monkeypatch, moe_checks, bound, masked
):
    """sglang 0.5.23's top-k masks padded rows by the forward context's bound, no longer an argument."""
    forward_context(monkeypatch, bound)
    assert (
        deepseek_v2._moe_forward_normal(SimpleNamespace(experts=SimpleNamespace()), torch.zeros(4, 8))
        is FallbackReason.OUT_OF_CONTRACT
    )
    assert moe_checks == ([] if masked else [True])


def test_a_deepseek_v2_moe_whose_output_sum_is_left_stays_on_sglang(moe_checks):
    """sglang 0.5.23's leave_output_sum: a later step sums the experts' output, the op's sum."""
    assert (
        deepseek_v2._moe_forward_normal(
            SimpleNamespace(experts=SimpleNamespace()), torch.zeros(4, 8), leave_output_sum=True
        )
        is FallbackReason.OUT_OF_CONTRACT
    )
    assert moe_checks == []


@pytest.mark.parametrize("api", [qwen3_5, qwen3_next], ids=["qwen3_5", "qwen3_next"])
def test_a_dp_relabeled_decode_stays_on_sglang(api):
    """sglang 0.5.23 relabels a decode batch a one-token extend under DP attention."""
    batch = SimpleNamespace(forward_mode="extend", logical_forward_mode="decode")
    assert api._gdn_forward(SimpleNamespace(), torch.zeros(4, 8), batch) is FallbackReason.OUT_OF_CONTRACT


def deepseekv41_moe(**changes):
    """A stand-in of sglang's DeepseekV2MoE as DeepSeek-V4.1 builds it on gfx950, ``changes`` aside."""
    config = SimpleNamespace(activation="silu", gate_up_interleaved=True, swiglu_limit=10.0, no_combine=False)
    runner = SimpleNamespace(config=config, runner_backend=SimpleNamespace(is_aiter=lambda: True))
    method = type("Fp8MoEMethod", (), {"is_fp4_expert": True, "runner": runner})()
    experts = type("FusedMoE", (), {})()
    experts.__dict__.update(
        quant_method=method,
        dispatcher=type("StandardDispatcher", (), {"expert_mask_gpu": None})(),
        fp8_activations=True,
        w13_weight=SimpleNamespace(is_shuffled=True),
        w13_weight_scale_inv=None,
        w2_weight=None,
        w2_weight_scale_inv=None,
        moe_runner_config=config,
    )
    routing = SimpleNamespace(
        use_grouped_topk=False,
        custom_routing_function=None,
        scoring_func="sqrtsoftplus",
        correction_bias=torch.zeros(8),
        renormalize=True,
        num_shared_topk_slots=1,
        top_k=7,
    )
    moe = SimpleNamespace(
        experts=experts,
        topk=SimpleNamespace(topk_config=routing, waterfill_balancer=None),
        gate=SimpleNamespace(weight=None),
        is_hash=False,
        num_fused_shared_experts=1,
        _fuse_shared_experts_inside_sbo=False,
        routed_scaling_factor_fused_in_topk=True,
        routed_scaling_factor=1.5,
    )
    moe.__dict__.update(changes)
    return moe


@pytest.mark.parametrize(
    ("changes", "asked"),
    [
        ({}, True),
        ({"num_fused_shared_experts": 0}, False),
        ({"routed_scaling_factor_fused_in_topk": False}, False),
    ],
    ids=["as_served", "shared_expert_apart", "scale_after_the_experts"],
)
def test_a_deepseekv41_moe_reaches_its_op_only_as_the_cookbook_serves_it(monkeypatch, changes, asked):
    """The shared expert fused last and the scale folded into the top-k: any other MoE stays on sglang."""
    calls = []
    monkeypatch.setattr(deepseek_v2.moe_blocks, "runtime_falls_back", lambda: False)
    monkeypatch.setattr(
        deepseek_v2.deepseekv41,
        "deepseekv41_mono_k2_ffn",
        lambda **call: calls.append(call) or FallbackReason.NO_WORKLOAD,
    )
    reason = deepseek_v2._moe_forward_normal(deepseekv41_moe(**changes), torch.zeros(4, 8))
    assert reason is (FallbackReason.NO_WORKLOAD if asked else FallbackReason.OUT_OF_CONTRACT)
    assert [call["topk"] for call in calls] == ([6] if asked else [])
