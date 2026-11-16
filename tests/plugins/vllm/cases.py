# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The small Points of the vLLM plugin's contracts (``tests.op_case.OpCase``)."""

from aklr.core.contract.spec import PointSpec

# full PointSpec literals; each contract's base Point first
SMALL_POINTS: tuple[PointSpec, ...] = (
    # Kimi-K3's attention residual shrunk: a layer folding the pending output into the prefix,
    # then the first layer of a block, which stores its prefix as the next bank row
    PointSpec(
        contract_id="kimi_k3_attn_res@1.0",
        axes={"n_tokens": 4, "hidden_size": 256, "n_bank_rows": 4},
        dtypes={
            "prefix": "bfloat16",
            "addend": "bfloat16",
            "bank": "bfloat16",
            "score_norm_weight": "bfloat16",
            "score_proj_weight": "bfloat16",
            "norm_weight": "bfloat16",
            "out": "bfloat16",
        },
        attrs={"n_valid_rows": 2, "write_bank": False, "has_addend": True, "has_norm": True},
        layout_variant_name="row_major",
    ),
    PointSpec(
        contract_id="kimi_k3_attn_res@1.0",
        axes={"n_tokens": 4, "hidden_size": 256, "n_bank_rows": 4},
        dtypes={
            "prefix": "bfloat16",
            "addend": None,
            "bank": "bfloat16",
            "score_norm_weight": "bfloat16",
            "score_proj_weight": "bfloat16",
            "norm_weight": "bfloat16",
            "out": "bfloat16",
        },
        attrs={"n_valid_rows": 3, "write_bank": True, "has_addend": False, "has_norm": True},
        layout_variant_name="row_major",
    ),
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
