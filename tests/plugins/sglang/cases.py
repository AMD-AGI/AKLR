# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Created by OpenAI

"""The small Points of the sglang plugin's contracts (``tests.op_case.OpCase``)."""

from aklr.core.contract.spec import PointSpec

# full PointSpec literals; each contract's base Point first
SMALL_POINTS: tuple[PointSpec, ...] = (
    # GLM-5.2-MXFP4's DeepseekV2MoE: the shared expert fused last, as sglang holds the
    # experts for aiter
    PointSpec(
        contract_id="deepseek_v2_mxfp4_moe@1.0",
        axes={"n_tokens": 4, "model_dim": 256, "n_routed": 8, "inter_dim": 256},
        dtypes={
            "hidden_states": "bfloat16",
            "router_weight": "bfloat16",
            "correction_bias": "float32",
            "w1": "float4_e2m1",
            "w1_scale": "float8_e8m0",
            "w2": "float4_e2m1",
            "w2_scale": "float8_e8m0",
            "out": "bfloat16",
        },
        attrs={
            "topk": 8,
            "num_expert_group": 1,
            "topk_group": 1,
            "renormalize": True,
            "scoring_func": "sigmoid",
        },
        layout_variant_name="shuffled_16x16",
    ),
    # GLM-5.2's RoPE part; four heads (16 at TP4) and a short top-k
    PointSpec(
        contract_id="dsa_sparse_mla_decode@1.0",
        axes={
            "n_tokens": 4,
            "n_heads": 4,
            "kv_dim": 192,
            "topk": 64,
            "n_slots": 512,
        },
        dtypes={
            "q": "float8_e4m3fn",
            "kv_cache": "float8_e4m3fn",
            "topk_indices": "int32",
            "output": "bfloat16",
        },
        attrs={},
        layout_variant_name="row_major",
    ),
    # MiniMax-M2's MoE shrunk: every routed expert on the rank, then half of them (expert
    # parallelism); the experts as sglang holds them for aiter
    *(
        PointSpec(
            contract_id=f"minimax_m2_{precision}_moe@1.0",
            axes={
                "n_tokens": 4,
                "model_dim": 256,
                "n_routed": n_routed,
                "n_experts": 8,
                "inter_dim": 256,
            },
            dtypes={
                "hidden_states": "bfloat16",
                "router_weight": "float32",
                "correction_bias": "float32",
                "w1": weight,
                "w1_scale": scale,
                "w2": weight,
                "w2_scale": scale,
                "out": "bfloat16",
            },
            attrs={"topk": 8},
            layout_variant_name="shuffled_16x16",
        )
        for precision, weight, scale in (
            ("fp8", "float8_e4m3fn", "float32"),
            ("mxfp4", "float4_e2m1", "float8_e8m0"),
        )
        for n_routed in (8, 16)
    ),
    PointSpec(
        contract_id="gdn_prefill_group_fp8_quant@1.0",
        axes={
            "n_tokens": 4,
            "n_sequences": 2,
            "n_slots": 4,
            "n_v_heads": 4,
            "key_dim": 16,
            "value_dim": 128,
            "conv_dim": 544,
            "n_starts": 3,
        },
        dtypes={
            "projected_qkvz": "bfloat16",
            "projected_ba": "bfloat16",
            "conv_state": "bfloat16",
            "delta_state": "float32",
            "cache_indices": "int32",
            "cu_seqlens": "int32",
            "has_initial_state": "bool",
            "conv_weight": "bfloat16",
            "conv_bias": "bfloat16",
            "a_log": "float32",
            "dt_bias": "bfloat16",
            "norm_weight": "bfloat16",
            "normalized": "bfloat16",
            "values": "float8_e4m3fn",
            "scales": "float32",
        },
        attrs={"max_seq_len": 2},
        layout_variant_name="row_major",
    ),
    # Qwen3.5's MoE block shrunk: 16 routed experts of hidden and intermediate size 256,
    # the shared one fused in last, the experts fp8 and shuffled as sglang holds them
    PointSpec(
        contract_id="qwen2_moe_fp8_experts@1.0",
        axes={
            "n_tokens": 4,
            "model_dim": 256,
            "n_experts": 17,
            "inter_dim": 256,
            "topk": 10,
        },
        dtypes={
            "hidden_states": "bfloat16",
            "topk_weights": "float32",
            "topk_ids": "int32",
            "shared_gate_weight": "bfloat16",
            "w1": "float8_e4m3fn",
            "w1_scale": "float32",
            "w2": "float8_e4m3fn",
            "w2_scale": "float32",
            "out": "bfloat16",
        },
        attrs={},
        layout_variant_name="shuffled_16x16",
    ),
    PointSpec(
        contract_id="qwen2_moe_fp8_router_experts@1.0",
        axes={"n_tokens": 4, "model_dim": 256, "n_routed": 16, "inter_dim": 256},
        dtypes={
            "hidden_states": "bfloat16",
            "router_weight": "bfloat16",
            "shared_gate_weight": "bfloat16",
            "w1": "float8_e4m3fn",
            "w1_scale": "float32",
            "w2": "float8_e4m3fn",
            "w2_scale": "float32",
            "out": "bfloat16",
        },
        attrs={"topk": 10},
        layout_variant_name="shuffled_16x16",
    ),
    PointSpec(
        contract_id="qwen2_moe_router@1.0",
        axes={"n_tokens": 4, "model_dim": 256, "n_routed": 16, "topk": 10},
        dtypes={
            "hidden_states": "bfloat16",
            "router_weight": "bfloat16",
            "topk_weights": "float32",
            "topk_ids": "int32",
        },
        attrs={},
        layout_variant_name="row_major",
    ),
    # Qwen3-Coder's MoE block shrunk: 16 experts of hidden and intermediate size 256, top-8,
    # the experts fp8 and shuffled as sglang holds them
    PointSpec(
        contract_id="qwen3_moe_fp8_moe@1.0",
        axes={"n_tokens": 4, "model_dim": 256, "n_experts": 16, "inter_dim": 256},
        dtypes={
            "hidden_states": "bfloat16",
            "router_weight": "bfloat16",
            "w1": "float8_e4m3fn",
            "w1_scale": "float32",
            "w2": "float8_e4m3fn",
            "w2_scale": "float32",
            "out": "bfloat16",
        },
        attrs={"topk": 8},
        layout_variant_name="shuffled_16x16",
    ),
    # a slot more than rows: the drawn indices hold one -1 padding row
    PointSpec(
        contract_id="gdn_decode_group_fp8_quant@1.0",
        axes={
            "n_tokens": 4,
            "n_slots": 4,
            "n_v_heads": 4,
            "key_dim": 16,
            "value_dim": 128,
            "conv_dim": 544,
        },
        dtypes={
            "projected_qkvz": "bfloat16",
            "projected_ba": "bfloat16",
            "conv_state": "bfloat16",
            "delta_state": "float32",
            "cache_indices": "int32",
            "conv_weight": "bfloat16",
            "a_log": "float32",
            "dt_bias": "bfloat16",
            "norm_weight": "bfloat16",
            "values": "float8_e4m3fn",
            "scales": "float32",
        },
        attrs={},
        layout_variant_name="row_major",
    ),
    # a slot more than rows: the drawn indices hold one -1 padding row
    PointSpec(
        contract_id="gdn_decode@1.0",
        axes={
            "n_tokens": 4,
            "n_slots": 4,
            "n_v_heads": 4,
            "key_dim": 16,
            "value_dim": 128,
            "conv_dim": 544,
        },
        dtypes={
            "projected_qkvz": "bfloat16",
            "projected_ba": "bfloat16",
            "conv_state": "bfloat16",
            "delta_state": "float32",
            "cache_indices": "int32",
            "conv_weight": "bfloat16",
            "a_log": "float32",
            "dt_bias": "bfloat16",
            "norm_weight": "bfloat16",
            "normalized": "bfloat16",
        },
        attrs={},
        layout_variant_name="row_major",
    ),
)
# whose baseline runs only on gfx950
GFX950_CONTRACT_IDS = frozenset(
    {
        "deepseek_v2_mxfp4_moe@1.0",
        "dsa_sparse_mla_decode@1.0",
        "gdn_prefill_group_fp8_quant@1.0",
        "minimax_m2_fp8_moe@1.0",
        "minimax_m2_mxfp4_moe@1.0",
        "qwen2_moe_fp8_experts@1.0",
        "qwen2_moe_fp8_router_experts@1.0",
        "qwen2_moe_router@1.0",
        "qwen3_moe_fp8_moe@1.0",
        "gdn_decode_group_fp8_quant@1.0",
    }
)
