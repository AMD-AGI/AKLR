# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The small Points of the contracts under ``aklr/plugins/aiter`` (``tests.op_case.OpCase``)."""

from aklr.core.contract.spec import PointSpec

# full PointSpec literals; each contract's base Point first
SMALL_POINTS: tuple[PointSpec, ...] = (
    # Kimi-K3's a4w4 experts shrunk, laid out as vLLM holds them for aiter
    PointSpec(
        contract_id="situv2_a4w4_moe@1.0",
        axes={"n_tokens": 4, "model_dim": 256, "n_experts": 4, "topk": 2, "inter_dim": 128},
        dtypes={
            "hidden_states": "bfloat16",
            "w1": "float4_e2m1",
            "w2": "float4_e2m1",
            "topk_weights": "float32",
            "topk_ids": "int32",
            "w1_scale": "float8_e8m0",
            "w2_scale": "float8_e8m0",
            "out": "bfloat16",
        },
        attrs={},
        layout_variant_name="a16w4_shuffled",
    ),
)
# whose baseline runs only on gfx950
GFX950_CONTRACT_IDS = frozenset({"situv2_a4w4_moe@1.0"})
