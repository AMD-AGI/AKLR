# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which sglang MoE experts the ops take (``experts_fall_back``)."""

from types import SimpleNamespace

import pytest

from aklr.plugins.sglang.models import moe_blocks


def fp8_experts(**traits):
    """Stand-ins of block-FP8 experts on aiter's fused_moe at sglang's standard dispatch, the ops' case."""
    method = type("Fp8MoEMethod", (), {})()
    method.block_quant, method.use_mxfp8, method.is_fp4_expert = True, False, False
    method.weight_block_size = [128, 128]
    method.runner = SimpleNamespace(
        runner_backend=SimpleNamespace(is_aiter=lambda: True),
        config=SimpleNamespace(
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            no_combine=False,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
        ),
    )
    experts = type("FusedMoE", (), {})()
    experts.quant_method = method
    experts.dispatcher = type("StandardDispatcher", (), {})()
    experts.reduce_results = False
    for name, value in traits.items():
        setattr(experts, name, value)
    return experts


@pytest.mark.parametrize(
    ("traits", "falls_back"),
    [({}, False), ({"fp8_activations": False}, False), ({"fp8_activations": True}, True)],
    ids=["no_flag", "bf16_activations", "mxfp8_activations"],
)
def test_experts_on_mxfp8_activations_stay_on_sglang(traits, falls_back):
    """sglang 0.5.23 quantizes DeepSeek-V4.1's FP4 experts' activations to MXFP8; the ops take bf16."""
    assert moe_blocks.experts_fall_back(fp8_experts(**traits), ("Fp8MoEMethod",)) is falls_back
