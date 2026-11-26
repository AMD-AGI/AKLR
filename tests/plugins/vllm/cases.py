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
    # the residual before the final norm, whose mix is the result (no output norm)
    PointSpec(
        contract_id="kimi_k3_attn_res@1.0",
        axes={"n_tokens": 4, "hidden_size": 256, "n_bank_rows": 4},
        dtypes={
            "prefix": "bfloat16",
            "addend": "bfloat16",
            "bank": "bfloat16",
            "score_norm_weight": "bfloat16",
            "score_proj_weight": "bfloat16",
            "norm_weight": None,
            "out": "bfloat16",
        },
        attrs={"n_valid_rows": 3, "write_bank": False, "has_addend": True, "has_norm": False},
        layout_variant_name="row_major",
    ),
    # a block's first layer folding the pending output in: the bank row written is bf16(prefix + addend)
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
        attrs={"n_valid_rows": 2, "write_bank": True, "has_addend": True, "has_norm": True},
        layout_variant_name="row_major",
    ),
    # the same with the per-token fp8 out the linear after it takes
    PointSpec(
        contract_id="kimi_k3_attn_res_fp8@1.0",
        axes={"n_tokens": 4, "hidden_size": 256, "n_bank_rows": 4},
        dtypes={
            "prefix": "bfloat16",
            "addend": "bfloat16",
            "bank": "bfloat16",
            "score_norm_weight": "bfloat16",
            "score_proj_weight": "bfloat16",
            "norm_weight": "bfloat16",
            "quantized": "float8_e4m3fn",
            "scale": "float32",
        },
        attrs={"n_valid_rows": 2, "write_bank": True, "has_addend": True, "has_norm": True},
        layout_variant_name="row_major",
    ),
    # Kimi-K3's latent-MoE tail shrunk, a one-rank group (the generic tests draw one rank's inputs; the CPU tests of
    # the op run its eight ranks)
    PointSpec(
        contract_id="kimi_k3_latent_moe_tail@1.0",
        axes={"n_tokens": 4, "hidden_size": 256, "latent_size": 128},
        dtypes={
            "routed": "bfloat16",
            "shared": "bfloat16",
            "up_weight": "bfloat16",
            "norm_weight": "bfloat16",
            "out": "bfloat16",
        },
        attrs={"tp_size": 1},
        layout_variant_name="row_major",
    ),
    # Kimi-K3's KDA speculative decode shrunk to two heads: two sequences of three tokens (the one after the accepted
    # token, two drafts)
    PointSpec(
        contract_id="kimi_k3_kda_decode_spec@1.0",
        axes={"n_seqs": 2, "n_tokens": 6, "n_slots": 9, "n_heads": 2, "head_dim": 128},
        dtypes={
            "qkv": "bfloat16",
            "raw_g": "bfloat16",
            "raw_beta": "bfloat16",
            "a_log": "float32",
            "g_bias": "float32",
            "state_cache": "float32",
            "state_indices": "int32",
            "num_accepted_tokens": "int32",
            "out": "bfloat16",
        },
        attrs={},
        layout_variant_name="strided_rows",
    ),
    # Kimi-K3's KDA prefill shrunk to two heads: two sequences in vLLM's paged state cache
    PointSpec(
        contract_id="kimi_k3_kda_prefill@1.0",
        axes={"n_tokens": 70, "n_seqs": 2, "n_offsets": 3, "n_slots": 3, "n_heads": 2, "head_dim": 128},
        dtypes={
            "q": "bfloat16",
            "k": "bfloat16",
            "v": "bfloat16",
            "raw_g": "bfloat16",
            "raw_beta": "bfloat16",
            "a_log": "float32",
            "g_bias": "float32",
            "state_cache": "float32",
            "state_indices": "int32",
            "has_initial_state": "bool",
            "cu_seqlens": "int32",
            "out": "bfloat16",
        },
        attrs={},
        layout_variant_name="strided_rows",
    ),
    # and with a snapshot mid-prefill, as vLLM takes under prefix caching: one sequence, so that the drawn point (0,
    # or 64 tokens in) is one vLLM asks for
    PointSpec(
        contract_id="kimi_k3_kda_prefill_checkpoint@1.0",
        axes={"n_tokens": 70, "n_seqs": 1, "n_offsets": 2, "n_slots": 3, "n_heads": 2, "head_dim": 128},
        dtypes={
            "q": "bfloat16",
            "k": "bfloat16",
            "v": "bfloat16",
            "raw_g": "bfloat16",
            "raw_beta": "bfloat16",
            "a_log": "float32",
            "g_bias": "float32",
            "state_cache": "float32",
            "state_indices": "int32",
            "has_initial_state": "bool",
            "cu_seqlens": "int32",
            "checkpoint_chunks": "int32",
            "checkpoint_state": "float32",
            "out": "bfloat16",
        },
        attrs={},
        layout_variant_name="strided_rows",
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
    # Kimi-K3.1's indexer prefill at its 32 heads and top 2048, a few rows over 3072 staged keys: every row has
    # more keys than it selects
    PointSpec(
        contract_id="kimi_k3_fp4_indexer_prefill@1.0",
        axes={"n_tokens": 4, "n_heads": 32, "n_pages": 48, "topk": 2048},
        dtypes={
            "q": "bfloat16",
            "weights": "bfloat16",
            "key_values": "float4_e2m1",
            "key_scales": "float8_e8m0",
            "row_starts": "int32",
            "row_ends": "int32",
            "indices": "int32",
        },
        attrs={},
        layout_variant_name="row_major",
    ),
)
# whose baseline runs only on gfx950 (vLLM's fused KDA prefill, the indexer's FlyDSL scorer)
GFX950_CONTRACT_IDS = frozenset(
    {
        "kimi_k3_fp4_indexer_prefill@1.0",
        "kimi_k3_kda_prefill@1.0",
        "kimi_k3_kda_prefill_checkpoint@1.0",
        "muse_spark_moe@1.0",
    }
)
