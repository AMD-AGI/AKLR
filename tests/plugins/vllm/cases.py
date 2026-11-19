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
    # Kimi-K3's KDA layer after in_proj shrunk to two heads: two windows of three tokens and a padding row, in vLLM's
    # paged caches
    PointSpec(
        contract_id="kimi_k3_kda_spec_stage_c@1.0",
        axes={
            "n_seqs": 2,
            "n_tokens": 7,
            "n_slots": 9,
            "n_heads": 2,
            "head_dim": 128,
            "width": 4,
            "conv_len": 5,
        },
        dtypes={
            "qkv": "bfloat16",
            "out_gate": "bfloat16",
            "f_a": "bfloat16",
            "raw_beta": "bfloat16",
            "f_b_weight": "bfloat16",
            "conv_weight": "float32",
            "conv_state": "bfloat16",
            "a_log": "float32",
            "g_bias": "float32",
            "norm_weight": "bfloat16",
            "state_cache": "float32",
            "state_indices": "int32",
            "num_accepted_tokens": "int32",
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
)
# whose baseline runs only on gfx950 (aiter's gfx950 KDA kernel for the stage)
GFX950_CONTRACT_IDS = frozenset({"kimi_k3_kda_spec_stage_c@1.0", "muse_spark_moe@1.0"})
