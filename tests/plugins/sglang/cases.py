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
    # GLM-5.2's DSA indexer shrunk: 16 heads (32 on GLM-5.2) and narrow projections, its
    # 128-value heads and keys, 64-value RoPE part and one index-K page as sglang holds them on
    # aiter
    PointSpec(
        contract_id="dsa_indexer_prepare@1.0",
        axes={
            "n_tokens": 4,
            "model_dim": 256,
            "n_positions": 64,
            "n_pages": 1,
            "q_lora_rank": 256,
            "n_heads": 16,
        },
        dtypes={
            "x": "bfloat16",
            "q_lora": "bfloat16",
            "wq_b": "bfloat16",
            "wk": "bfloat16",
            "weights_proj": "bfloat16",
            "k_norm_weight": "bfloat16",
            "k_norm_bias": "bfloat16",
            "cos_cache": "bfloat16",
            "sin_cache": "bfloat16",
            "positions": "int64",
            "out_cache_loc": "int64",
            "index_k_cache": "float8_e4m3fn",
            "q_fp8": "float8_e4m3fn",
            "head_weights": "float32",
        },
        attrs={},
        layout_variant_name="row_major",
    ),
    PointSpec(
        contract_id="dsa_indexer_key_prepare@1.0",
        axes={"n_tokens": 4, "model_dim": 256, "n_positions": 64, "n_pages": 1},
        dtypes={
            "x": "bfloat16",
            "wk": "bfloat16",
            "k_norm_weight": "bfloat16",
            "k_norm_bias": "bfloat16",
            "cos_cache": "bfloat16",
            "sin_cache": "bfloat16",
            "positions": "int64",
            "out_cache_loc": "int64",
            "index_k_cache": "float8_e4m3fn",
            "no_values": "float32",
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
    # Qwen3-Next's GDN shrunk: two key heads, so that the grouped projections differ from
    # contiguous ones
    PointSpec(
        contract_id="qwen3_next_gdn_prefill_group_fp8_quant@1.0",
        axes={
            "n_tokens": 4,
            "n_sequences": 2,
            "n_slots": 4,
            "n_v_heads": 4,
            "key_dim": 16,
            "value_dim": 128,
            "conv_dim": 576,
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
    # a slot more than rows: the drawn indices hold one -1 padding row
    PointSpec(
        contract_id="qwen3_next_gdn_decode_group_fp8_quant@1.0",
        axes={
            "n_tokens": 4,
            "n_slots": 4,
            "n_v_heads": 4,
            "key_dim": 16,
            "value_dim": 128,
            "conv_dim": 576,
        },
        dtypes={
            "projected_qkvz": "bfloat16",
            "projected_ba": "bfloat16",
            "conv_state": "bfloat16",
            "delta_state": "float32",
            "cache_indices": "int32",
            "conv_weight": "bfloat16",
            "conv_bias": "bfloat16",
            "a_log": "float32",
            "dt_bias": "bfloat16",
            "norm_weight": "bfloat16",
            "normalized": "bfloat16",
            "values": "float8_e4m3fn",
            "scales": "float32",
        },
        attrs={},
        layout_variant_name="row_major",
    ),
    # Qwen3.5's full-attention heads at TP8: four query heads and one KV head of 256, a few
    # dozen pool slots and context tokens
    PointSpec(
        contract_id="gated_attention_decode_group_fp8_quant@1.0",
        axes={
            "n_tokens": 4,
            "qkv_dim": 2560,
            "n_kv_heads": 1,
            "head_dim": 256,
            "n_slots": 64,
            "n_positions": 64,
            "n_offsets": 5,
            "n_kv_tokens": 40,
        },
        dtypes={
            "projected_qkv": "bfloat16",
            "q_norm_weight": "bfloat16",
            "k_norm_weight": "bfloat16",
            "positions": "int64",
            "cos_sin_cache": "float32",
            "cache_locations": "int64",
            "key_cache": "float8_e4m3fn",
            "value_cache": "float8_e4m3fn",
            "kv_indptr": "int32",
            "kv_indices": "int32",
            "values": "float8_e4m3fn",
            "scales": "float32",
        },
        attrs={},
        layout_variant_name="row_major",
    ),
    # Qwen3-Next-80B's full-attention heads at TP4, the same widths; at the 8192-token
    # capacity, its longest row at it and the other three a few dozen tokens
    PointSpec(
        contract_id="qwen3_next_gated_attention_decode_group_fp8_quant@1.0",
        axes={
            "n_tokens": 4,
            "qkv_dim": 2560,
            "n_kv_heads": 1,
            "head_dim": 256,
            "n_slots": 64,
            "n_positions": 64,
            "n_offsets": 5,
            "n_kv_tokens": 8288,
        },
        dtypes={
            "projected_qkv": "bfloat16",
            "q_norm_weight": "bfloat16",
            "k_norm_weight": "bfloat16",
            "positions": "int64",
            "cos_sin_cache": "float32",
            "cache_locations": "int64",
            "key_cache": "float8_e4m3fn",
            "value_cache": "float8_e4m3fn",
            "kv_indptr": "int32",
            "kv_indices": "int32",
            "values": "float8_e4m3fn",
            "scales": "float32",
        },
        attrs={"max_context": 8192},
        layout_variant_name="row_major",
    ),
    # DeepSeek-V4's MQA prepare shrunk: two query heads, a 64-wide query LoRA rank, one page
    # of sglang's 256 slots
    PointSpec(
        contract_id="deepseek_v4_attention_q_prep@1.0",
        axes={"n_tokens": 5, "n_heads": 2, "q_lora_rank": 64, "n_pages": 1, "n_positions": 16},
        dtypes={
            "qkv_a": "bfloat16",
            "q_norm_weight": "bfloat16",
            "kv_norm_weight": "bfloat16",
            "wq_b": "float8_e4m3fn",
            "wq_b_scale": "float8_e8m0",
            "freqs": "float32",
            "positions": "int64",
            "swa_loc": "int32",
            "kv_cache": "uint8",
            "q": "bfloat16",
            "q_lora_quant": "float8_e4m3fn",
            "q_lora_scale": "float8_e8m0",
        },
        attrs={"page_size": 256},
        layout_variant_name="row_major",
    ),
    # DeepSeek-V4's sparse decode with the inverse RoPE on small pools: an SWA layer, then a
    # compressed one with the extra pool
    *(
        PointSpec(
            contract_id="deepseek_v4_sparse_mla_decode_inv_rope@1.0",
            axes={
                "n_tokens": 3,
                "n_heads": 16,
                "n_pages": 3,
                "page_size": 64,
                "n_indices": 64,
                # an absent pool's axes take their lowest values
                "n_extra_pages": 2 if extra else 1,
                "extra_page_size": 128 if extra else 2,
                "n_extra_indices": 128 if extra else 64,
                "n_positions": 32,
            },
            dtypes={
                "q": "bfloat16",
                "kv_cache": "float8_e4m3fn",
                "sparse_indices": "int32",
                "extra_kv_cache": "float8_e4m3fn" if extra else None,
                "extra_sparse_indices": "int32" if extra else None,
                "sinks": "float32",
                "rope_freqs": "float32",
                "positions": "int64",
                "output": "bfloat16",
            },
            attrs={"has_extra_kv_cache": extra, "has_extra_sparse_indices": extra},
            layout_variant_name="padded_pages",
        )
        for extra in (False, True)
    ),
    # DeepSeek-V4's wo_a GEMM in sglang's gfx950 split-K regime, shrunk to its smallest K: two
    # groups, bf16 out (then on the fp8 grid), then wo_b's MXFP8 operand
    *(
        PointSpec(
            contract_id="deepseek_v4_wo_a_split_k@1.0",
            axes={"n_tokens": 5, "n_groups": 2, "rank": 128, "in_dim": 2048},
            dtypes={"x": "bfloat16", "w": "bfloat16", "out": "bfloat16"},
            attrs={"fp8_grid": fp8_grid},
            layout_variant_name="row_major",
        )
        for fp8_grid in (False, True)
    ),
    PointSpec(
        contract_id="deepseek_v4_wo_a_split_k_mxfp8@1.0",
        axes={"n_tokens": 5, "n_groups": 2, "rank": 128, "in_dim": 2048},
        dtypes={"x": "bfloat16", "w": "bfloat16", "quantized": "float8_e4m3fn", "scales": "float8_e8m0"},
        attrs={},
        layout_variant_name="row_major",
    ),
)
# whose baseline runs only on gfx950
GFX950_CONTRACT_IDS = frozenset(
    {
        "deepseek_v2_mxfp4_moe@1.0",
        "dsa_indexer_key_prepare@1.0",
        "dsa_indexer_prepare@1.0",
        "dsa_sparse_mla_decode@1.0",
        "gdn_prefill_group_fp8_quant@1.0",
        "minimax_m2_fp8_moe@1.0",
        "minimax_m2_mxfp4_moe@1.0",
        "qwen2_moe_fp8_experts@1.0",
        "qwen2_moe_fp8_router_experts@1.0",
        "qwen2_moe_router@1.0",
        "gdn_decode_group_fp8_quant@1.0",
        "qwen3_next_gdn_prefill_group_fp8_quant@1.0",
        "qwen3_next_gdn_decode_group_fp8_quant@1.0",
        "gated_attention_decode_group_fp8_quant@1.0",
        "qwen3_next_gated_attention_decode_group_fp8_quant@1.0",
        "deepseek_v4_attention_q_prep@1.0",
        "deepseek_v4_sparse_mla_decode_inv_rope@1.0",
        "deepseek_v4_wo_a_split_k@1.0",
        "deepseek_v4_wo_a_split_k_mxfp8@1.0",
    }
)
