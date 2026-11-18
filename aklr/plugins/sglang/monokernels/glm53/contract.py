# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3's launches on one rank of the TP group, one contract per kind:

- ``glm53_moe_layer``: an MoE layer reading the previous indexer layer's top-k (57 of the 78 layers).
- ``glm53_moe_index_layer`` (TODO): an MoE layer selecting its own top-2048 (18 layers).
- ``glm53_dense_index_layer`` (TODO): a dense layer selecting its own top-2048 (layers 0 to 2).
- ``glm53_head`` (TODO): the final norm, the LM head and the logits' all-gather.

Each has ``ranks="tp_size"``, as ``mega_moe``: every rank of the group calls it once a step, on its
own inputs, and its sums run over the group in rank order.
"""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    Preshuffle,
    SqnrBound,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.plugins.sglang.layers.dsa_backend.contract import ROPE_DIM
from aklr.plugins.sglang.models.deepseek_v2.contract import DEEPSEEK_V2_FP8_MOE, DEEPSEEK_V2_MXFP4_MOE

__all__ = ["CONTRACTS", "GLM53_MOE_LAYER"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FINITE = FiniteDistribution()
# the 128 x 128 block scales of an fp8 linear, as deepseek_v2_fp8_moe draws them
_BLOCK_SCALE = LogUniformDistribution(log2_lo="-3", log2_hi="0")
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)


def _tensor(description, shape, dtype, distribution=_FINITE, *, weight=False, **kwargs):
    return TensorSpec(
        description=description,
        shape=shape,
        dtype=dtype,
        distribution=distribution,
        lifetime="weight" if weight else "per_call",
        **kwargs,
    )


def _fp8_linear(name, rows, cols, blocks):
    """An fp8 linear's weight [rows, cols] and its f32 scale per 128 x 128 block."""
    return {
        f"{name}_weight": _tensor(
            f"{name}'s weight.", (rows, cols), _FP8, LecunNormalDistribution(fan_in=cols), weight=True
        ),
        f"{name}_scale": _tensor(
            f"{name}'s scale per 128 x 128 block.", blocks, _FP32, _BLOCK_SCALE, weight=True
        ),
    }


GLM53_MOE_LAYER = OpContract(
    contract_id="glm53_moe_layer@1.0",
    description=(
        "A GLM-5.3 decoder layer whose attention reads the previous indexer layer's top-k, on one rank "
        "of the TP group, as sglang's DeepseekV2DecoderLayer runs it with the DSA backend. "
        "Attention: x = RMSNorm(hidden_states); q_a | latent | k_pe = qkv_a(x); "
        "q_nope | q_pe = q_b(RMSNorm(q_a)) per head; q_pe and k_pe rotated GPT-J style (the pairs "
        "2i, 2i + 1) by cos_sin_cache's row at the row's position; each row writes RMSNorm(latent) "
        "| k_pe, cast to kv_cache's dtype, at its cache_locations slot (rows sharing a slot leave "
        "one of their values there); each head then attends, "
        "with scores (q_nope w_kc | q_pe) . row * sm_scale, the slots topk_indices names (-1: none) "
        "after the writes, and its softmax-weighted latent times w_vc goes through o_proj. MoE, on "
        "RMSNorm(hidden_states + the group's attention outputs summed): each token's topk experts "
        "by sigmoid score plus correction_bias, the scores those of the router's logits rounded to "
        "bf16, weighed by the renormalized unbiased scores times routed_scaling_factor, and the "
        "shared expert (the last slot) at weight 1, each expert down(silu(gate(x)) * up(x)). qkv_a, "
        "q_b, o_proj and each expert's linears take their input in bf16, then in fp8 per 1 x 128 "
        "group at a scale of its amax / 448. out is hidden_states plus the group's attention and MoE outputs, each summed over "
        "the group in rank order."
    ),
    op_type="glm53",
    axes={
        "n_tokens": axis("Rows in this call: the step's tokens.", 1, 256, is_var=True),
        "n_slots": axis("Rows of the KV pool.", 1, 2**31 - 1, is_var=True, any_size=True),
        "n_positions": axis("Rows of the rotary table.", 1, 2**21, is_var=True, any_size=True),
        "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
        "n_heads": axis("Attention heads on this rank.", 1, 128),
        "q_lora_rank": axis("q_a's width.", 128, 4096, multiple_of=128),
        "kv_lora_rank": axis("The latent's width.", 128, 1024, multiple_of=128),
        "qk_nope_dim": axis("A query head's part without RoPE.", 64, 512, multiple_of=64),
        "v_head_dim": axis("A head's output width.", 64, 512, multiple_of=64),
        "index_topk": axis("KV slots each row's top-k names.", 1, 65536),
        "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
        "inter_dim": axis("Expert intermediate size on this rank.", 128, 16384, multiple_of=128),
        "qkv_a_rows": derived_axis(
            "qkv_a's rows: q_a, the latent, then k_pe.", f"q_lora_rank + kv_lora_rank + {ROPE_DIM}"
        ),
        "q_b_rows": derived_axis(
            "q_b's rows: each head's q_nope, then its q_pe.", f"n_heads * (qk_nope_dim + {ROPE_DIM})"
        ),
        "o_cols": derived_axis("o_proj's input: each head's output.", "n_heads * v_head_dim"),
        "kv_dim": derived_axis("A KV row: the latent, then k_pe.", f"kv_lora_rank + {ROPE_DIM}"),
        "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
        "qkv_a_blocks": derived_axis("128-blocks of qkv_a's rows.", "ceil_div(qkv_a_rows, 128)"),
        "q_lora_blocks": derived_axis("128-blocks of q_a's width.", "q_lora_rank // 128"),
        "q_b_blocks": derived_axis("128-blocks of q_b's rows.", "q_b_rows // 128"),
        "o_blocks": derived_axis("128-blocks of o_proj's input.", "o_cols // 128"),
        **{
            n: DEEPSEEK_V2_FP8_MOE.axes[n]
            for n in ("n_experts", "gate_up_rows", "gate_up_blocks", "inter_blocks")
        },
    },
    attrs={
        "topk": attr("Routed experts per token.", 8, param_name="topk"),
        "tp_size": attr("Ranks of the TP group.", 1, 2, 4, 8, default=1, param_name="tp_size"),
    },
    constraints=(
        "q_b_rows % 128 == 0",
        "o_cols % 128 == 0",
        "topk <= n_routed",
        # distinct slots, and distinct top-k entries with -1 among them
        "n_slots >= n_tokens",
        "index_topk <= n_slots + 1",
    ),
    inputs={
        "hidden_states": _tensor("The residual stream entering the layer.", ("n_tokens", "model_dim"), _BF16),
        "positions": _tensor(
            "Each row's position.",
            ("n_tokens",),
            DTypeSpec(any_of=(DType.INT64,)),
            IntRangeDistribution(lo=0, hi="n_positions - 1"),
        ),
        "cache_locations": _tensor(
            "Each row's KV pool slot.",
            ("n_tokens",),
            DTypeSpec(any_of=(DType.INT64,)),
            IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        "topk_indices": _tensor(
            "The KV slots each row attends; -1: none.",
            ("n_tokens", "index_topk"),
            DTypeSpec(any_of=(DType.INT32,)),
            IntRangeDistribution(lo=-1, hi="n_slots - 1", distinct=True),
        ),
        "kv_cache": _tensor(
            "This layer's KV pool: a row per slot, the latent then k_pe; E4M3 without scales, as "
            "sglang's HIP DSA backends keep it, or bf16.",
            ("n_slots", "kv_dim"),
            DTypeSpec(any_of=(DType.FLOAT8_E4M3FN, DType.BFLOAT16)),
            mutable=True,
            # shortcut: uncalibrated; a written value one E4M3 step, or a few bf16 steps, off for the
            # kernels' other rounding points
            error_bounds=(
                (
                    "kv_cache.dtype == 'float8_e4m3fn'",
                    ToleranceBound(absolute_tolerance=2**-4, relative_tolerance=2**-3),
                ),
                (None, ToleranceBound(absolute_tolerance=2**-6, relative_tolerance=2**-5)),
            ),
        ),
        "cos_sin_cache": _tensor(
            f"Per position, the {ROPE_DIM // 2} rotary cosines then the sines.",
            ("n_positions", ROPE_DIM),
            _FP32,
        ),
        "input_norm_weight": _tensor("input_layernorm's weight.", ("model_dim",), _BF16, weight=True),
        **_fp8_linear("qkv_a", "qkv_a_rows", "model_dim", ("qkv_a_blocks", "model_blocks")),
        "q_a_norm_weight": _tensor("q_a_layernorm's weight.", ("q_lora_rank",), _BF16, weight=True),
        "kv_a_norm_weight": _tensor("kv_a_layernorm's weight.", ("kv_lora_rank",), _BF16, weight=True),
        **_fp8_linear("q_b", "q_b_rows", "q_lora_rank", ("q_b_blocks", "q_lora_blocks")),
        "w_kc": _tensor(
            "Each head's q_nope -> latent map (kv_b_proj's k part).",
            ("n_heads", "qk_nope_dim", "kv_lora_rank"),
            _FP8,
            LecunNormalDistribution(fan_in="qk_nope_dim"),
            weight=True,
        ),
        "w_vc": _tensor(
            "Each head's latent -> output map (kv_b_proj's v part).",
            ("n_heads", "kv_lora_rank", "v_head_dim"),
            _FP8,
            LecunNormalDistribution(fan_in="kv_lora_rank"),
            weight=True,
        ),
        "w_scale": _tensor("w_kc's and w_vc's one scale.", (1,), _FP32, _BLOCK_SCALE, weight=True),
        **_fp8_linear("o", "model_dim", "o_cols", ("model_blocks", "o_blocks")),
        "post_norm_weight": _tensor("post_attention_layernorm's weight.", ("model_dim",), _BF16, weight=True),
        "router_weight": DEEPSEEK_V2_FP8_MOE.inputs["router_weight"],
        # fp32, as GLM-5.3 stores it
        "correction_bias": DEEPSEEK_V2_MXFP4_MOE.inputs["correction_bias"],
        **{
            n: DEEPSEEK_V2_FP8_MOE.inputs[n]
            for n in ("w1", "w1_scale", "w2", "w2_scale", "routed_scaling_factor")
        },
        "eps": ScalarSpec(
            description="Every RMSNorm's epsilon.",
            dtype=DType.FLOAT32,
            interval=(9.999999747378752e-06, 1e-5),
            default=1e-5,
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score: 1 / sqrt(qk_nope_dim + 64).",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=0.0625,
        ),
    },
    outputs={
        "out": TensorSpec(
            description="hidden_states plus the group's attention and MoE outputs.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            # shortcut: uncalibrated, deepseek_v2_fp8_moe's bound until a kernel's SQNR is measured
            error_bounds=SqnrBound(min_sqnr_db=30.0),
        )
    },
    # row-major, or as sglang holds the fp8 weights for aiter: shuffle_weight's 16 x 16 tiles
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": DEEPSEEK_V2_FP8_MOE.layout_variants.by_name["row_major"],
            "shuffled_16x16": LayoutVariant(
                layouts=dict.fromkeys(
                    ("qkv_a_weight", "q_b_weight", "o_weight", "w1", "w2"), Preshuffle(tile=(16, 16))
                ),
                detect_facts=DEEPSEEK_V2_FP8_MOE.layout_variants.by_name["shuffled_16x16"].detect_facts,
            ),
        }
    ),
    # Bounded, not bitwise: the experts' sums are in the kernels' order.
    compare=CompareSpec(max_violation_fraction=1e-3, determinism=DeterminismLevel.BOUNDED),
    ranks="tp_size",
)

CONTRACTS = (GLM53_MOE_LAYER,)
