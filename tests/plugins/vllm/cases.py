# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The small Points of the vLLM plugin's contracts (``tests.op_case.OpCase``)."""

from aklr.core.contract.spec import PointSpec

# full PointSpec literals; each contract's base Point first
SMALL_POINTS: tuple[PointSpec, ...] = (
    # Muse Spark's MoE experts shrunk: half the routed experts on the rank, as vLLM holds
    # them for aiter
    PointSpec(
        contract_id="muse_spark_moe@1.0",
        axes={
            "n_tokens": 4,
            "model_dim": 2048,
            "topk": 2,
            "n_experts": 2,
            "inter_dim": 256,
            "global_experts": 5,
        },
        dtypes={
            "hidden_states": "bfloat16",
            "norm_weight": "bfloat16",
            "topk_ids": "int32",
            "expert_mask": "int32",
            "w1": "float4_e2m1",
            "w1_scale": "float8_e8m0",
            "w2": "float4_e2m1",
            "w2_scale": "float8_e8m0",
            "out": "bfloat16",
        },
        attrs={},
        layout_variant_name="shuffled_16x16",
    ),
)
# whose baseline runs only on gfx950
GFX950_CONTRACT_IDS = frozenset({"muse_spark_moe@1.0"})
