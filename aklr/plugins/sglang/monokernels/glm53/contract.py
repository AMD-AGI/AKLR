# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3's launches on one rank of the TP group, one contract per kind:

- ``glm53_moe_layer``: an MoE layer reading the previous indexer layer's top-k (57 of the 78 layers).
- ``glm53_moe_index_layer``: an MoE layer selecting its own top-2048 (18 layers).
- ``glm53_dense_index_layer``: a dense layer selecting its own top-2048 (layers 0 to 2).
- ``glm53_head``: the LM head and the logits' all-gather (the final norm stays sglang's).
- ``glm53_nextn_entry``: the NextN (MTP) layer's input, eh_proj(enorm(e) | hnorm(h)), e the token embeddings.
- ``glm53_embed``: the target's token embedding (embed_tokens, vocab-parallel).

Each has ``ranks="tp_size"``, as ``mega_moe``: every rank of the group calls it once a step, on its
own inputs, and its sums run over the group in rank order.
"""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DsaIndexKCacheDistribution,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    Preshuffle,
    SqnrBound,
    ToleranceBound,
    tolerance,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DeterminismLevel,
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.plugins.sglang.monokernels.glm53.baseline import (
    Glm53DenseIndexLayerBaseline,
    Glm53EmbedBaseline,
    Glm53HeadBaseline,
    Glm53MoeIndexLayerBaseline,
    Glm53MoeLayerBaseline,
    Glm53NextnEntryBaseline,
)
from aklr.plugins.sglang.monokernels.glm53.reference import (
    INDEX_DIM,
    INDEX_PAGE,
    ROPE_DIM,
    Glm53DenseIndexLayerReference,
    Glm53EmbedReference,
    Glm53HeadReference,
    Glm53MoeIndexLayerReference,
    Glm53MoeLayerReference,
    Glm53NextnEntryReference,
)

__all__ = [
    "CONTRACTS",
    "GLM53_DENSE_INDEX_LAYER",
    "GLM53_EMBED",
    "GLM53_HEAD",
    "GLM53_MOE_INDEX_LAYER",
    "GLM53_MOE_LAYER",
    "GLM53_NEXTN_ENTRY",
]

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


def _shuffled_facts(names, shuffled):
    # SGLang marks a block-FP8 weight it preshuffled for AITER
    return tuple(
        LayoutFact(location="tensor_attr", name=f"{w}.is_shuffled", expected_value=shuffled) for w in names
    )


# the MoE layers' routed and shared experts, in their last slot
_MOE_AXES = {
    "n_routed": axis("Routed experts: the router's rows.", 1, 1024),
    "inter_dim": axis("Expert intermediate size on this rank.", 128, 16384, multiple_of=128),
    "n_experts": derived_axis("Expert slots: the routed experts, then the shared one.", "n_routed + 1"),
    "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
    "gate_up_blocks": derived_axis("128-blocks of w1's rows.", "gate_up_rows // 128"),
    "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
}
_MOE_INPUTS = {
    "router_weight": _tensor(
        "The router's projection.",
        ("n_routed", "model_dim"),
        _BF16,
        LecunNormalDistribution(fan_in="model_dim"),
        weight=True,
    ),
    "correction_bias": _tensor(
        "Added to the scores to pick experts; never weighs them. fp32, as GLM-5.3 stores it.",
        ("n_routed",),
        _FP32,
        # centred on 0 and narrow beside the scores' spread: it moves most tokens' picks, yet keeps the
        # routes spread over the experts, as a balanced model's are
        LecunNormalDistribution(fan_in="model_dim"),
        weight=True,
    ),
    "w1": _tensor(
        "Gate-up projection per expert: gate rows, then up rows.",
        ("n_experts", "gate_up_rows", "model_dim"),
        _FP8,
        LecunNormalDistribution(fan_in="model_dim"),
        weight=True,
    ),
    "w1_scale": _tensor(
        "w1's scale per 128 x 128 block.",
        ("n_experts", "gate_up_blocks", "model_blocks"),
        _FP32,
        _BLOCK_SCALE,
        weight=True,
    ),
    "w2": _tensor(
        "Down projection per expert.",
        ("n_experts", "model_dim", "inter_dim"),
        _FP8,
        LecunNormalDistribution(fan_in="inter_dim"),
        weight=True,
    ),
    "w2_scale": _tensor(
        "w2's scale per 128 x 128 block.",
        ("n_experts", "model_blocks", "inter_blocks"),
        _FP32,
        _BLOCK_SCALE,
        weight=True,
    ),
    "routed_scaling_factor": ScalarSpec(
        description="Multiplies the routed weights only: GLM-5.3's.",
        dtype=DType.FLOAT32,
        values=(2.5,),
        default=2.5,
    ),
}
# the widest page table an indexer layer's call takes: its index-K pages
_MAX_PAGES = 2**16
# the keys an indexer layer's row scores at most: GLM-5.3's longest context
_MAX_ROW_KEYS = 2**20


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
        "group at a scale of its amax / 448. out is hidden_states plus the group's attention and MoE "
        "outputs, each summed over the group in rank order."
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
        **_MOE_AXES,
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
            # the common bounds of the pool's dtype: a written value an E4M3 or a bf16 step off where a
            # kernel rounds the latent's sums in another order
            error_bounds=(
                ("kv_cache.dtype == 'float8_e4m3fn'", tolerance(DType.FLOAT8_E4M3FN)),
                (None, tolerance(DType.BFLOAT16)),
            ),
        ),
        "cos_sin_cache": _tensor(
            f"Per position, the {ROPE_DIM // 2} rotary cosines then the sines.",
            ("n_positions", ROPE_DIM),
            _FP32,
            weight=True,
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
        **_MOE_INPUTS,
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
            # deepseek_v2_fp8_moe's: the layer's MoE rounds as that op's does (35 to 36 dB in its CPU
            # emulation of the kernels' rounding)
            error_bounds=SqnrBound(min_sqnr_db=30.0),
        )
    },
    # row-major, or as sglang holds the fp8 weights for aiter: shuffle_weight's 16 x 16 tiles
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_shuffled_facts(("w1", "w2"), False)),
            "shuffled_16x16": LayoutVariant(
                layouts=dict.fromkeys(
                    ("qkv_a_weight", "q_b_weight", "o_weight", "w1", "w2"), Preshuffle(tile=(16, 16))
                ),
                detect_facts=_shuffled_facts(("w1", "w2"), True),
            ),
        }
    ),
    # Bounded, not bitwise: the experts' sums are in the kernels' order.
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
    ranks="tp_size",
    reference=Glm53MoeLayerReference(),
    baseline=Glm53MoeLayerBaseline(),
)

GLM53_MOE_INDEX_LAYER = OpContract(
    contract_id="glm53_moe_index_layer@1.0",
    description=(
        "A GLM-5.3 decoder layer selecting its own top-k, on one rank of the TP group: glm53_moe_layer, its "
        "topk_indices this layer's DSA indexer's, as sglang's gfx950 indexer computes them "
        "(gfx950_fused_indexer.py). With x the attention's normalized input rounded to bf16 and q_lora "
        "RMSNorm(q_a) rounded to bf16: each row's key, wk(x) rounded to bf16, through k_norm's LayerNorm "
        "(k_norm_weight, k_norm_bias, index_eps), the interleaved RoPE of its first 64 values by index_cos "
        "and index_sin at its position and the 128-point Hadamard transform times 128^-0.5, each step "
        "rounded to bf16, is stored into index_k_cache at its cache_locations slot as E4M3 at the scale "
        "2^ceil(log2(max(amax, 1e-4) / 448)), then that scale in f32; each head of wq_b(q_lora) rounded to "
        "bf16 is turned and transformed alike and in E4M3 at its own scale (amax floored at 1e-10), its gate "
        "bf16(bf16(x weights_proj^T) * n_index_heads^-0.5) times that scale times 128^-0.5. After every "
        "row's key is stored, each row scores its seq_lens visible keys, key p at page_table[row, p // 64]'s "
        "slot p % 64: the sum over heads of gate * relu(q . k), times the key's scale; topk_indices is the "
        "index_topk highest scores' slots (page * 64 + p % 64) in any order, or for a row of fewer than "
        "index_topk keys all of its slots in order, then -1. wk and wq_b take their input in fp8 per 1 x "
        "128 group as the other linears do. A row's position is seq_lens - 1."
    ),
    op_type="glm53",
    axes={
        **GLM53_MOE_LAYER.axes,
        "n_index_heads": axis("Indexer heads.", 1, 128),
        "index_cols": derived_axis(
            "wq_b's rows: the indexer heads side by side.", f"n_index_heads * {INDEX_DIM}"
        ),
        "index_cols_blocks": derived_axis("128-blocks of wq_b's rows.", "n_index_heads"),
        "n_pages": axis("Pages of the index-K cache.", 1, 2**24, is_var=True, any_size=True),
        "max_pages": axis(
            "The page table's width: pages a row may read.", 1, _MAX_PAGES, is_var=True, any_size=True
        ),
        "n_index_positions": axis(
            "Rows of the indexer's rotary tables.", 1, 2**21, is_var=True, any_size=True
        ),
    },
    attrs=GLM53_MOE_LAYER.attrs,
    constraints=(
        *(c for c in GLM53_MOE_LAYER.constraints if "index_topk" not in c),
        # a slot is a KV pool row and an index-K page's key (SGLang's index-K cache may hold a page more); a
        # row's pages are the KV pool's; a position reads both rotary tables
        f"n_slots <= {INDEX_PAGE} * n_pages",
        f"{INDEX_PAGE} * max_pages <= n_slots",
        "n_index_positions >= n_positions",
    ),
    inputs={
        **{n: s for n, s in GLM53_MOE_LAYER.inputs.items() if n != "topk_indices"},
        **_fp8_linear("wq_b", "index_cols", "q_lora_rank", ("index_cols_blocks", "q_lora_blocks")),
        **_fp8_linear("wk", INDEX_DIM, "model_dim", (1, "model_blocks")),
        "weights_proj_weight": _tensor(
            "weights_proj's weight: a row per indexer head.",
            ("n_index_heads", "model_dim"),
            _BF16,
            LecunNormalDistribution(fan_in="model_dim"),
            weight=True,
        ),
        "k_norm_weight": _tensor("k_norm's (LayerNorm) weight.", (INDEX_DIM,), _BF16, weight=True),
        "k_norm_bias": _tensor("k_norm's bias.", (INDEX_DIM,), _BF16, weight=True),
        "index_cos": _tensor(
            "The indexer's rotary cosines, a row per position and a column per RoPE pair.",
            ("n_index_positions", ROPE_DIM // 2),
            _BF16,
            weight=True,
        ),
        "index_sin": _tensor(
            "The indexer's rotary sines, as index_cos.",
            ("n_index_positions", ROPE_DIM // 2),
            _BF16,
            weight=True,
        ),
        "index_k_cache": _tensor(
            f"This layer's index-K cache: pages of {INDEX_PAGE} keys, their {INDEX_DIM} E4M3 in 16 x 16 tiles "
            "(16 keys by 16 values), then their f32 scales, the bytes compared as E4M3.",
            ("n_pages", INDEX_PAGE * (INDEX_DIM + 4)),
            _FP8,
            # each key's scale a power of two, as the indexer stores them: a key's index scores then span
            # what a model's do, so that the top-k's near ties are near
            DsaIndexKCacheDistribution(),
            mutable=True,
            # dsa_indexer_prepare's, whose key the layer stores: one E4M3 step and the reference's own
            # rounding to E4M3, and 2 below a key's largest values; a key at a scale step passes as the
            # reference's check restates it (restated_scale_steps)
            error_bounds=ToleranceBound(absolute_tolerance=2.0, relative_tolerance=0.125 + 2**-3),
        ),
        "page_table": _tensor(
            f"Each row's pages, {INDEX_PAGE} keys a page, distinct within a row, all in the KV pool.",
            ("n_tokens", "max_pages"),
            DTypeSpec(any_of=(DType.INT32,)),
            IntRangeDistribution(lo=0, hi=f"n_slots // {INDEX_PAGE} - 1", distinct=True),
        ),
        "seq_lens": _tensor(
            f"The keys each row scores: its position + 1, at most {_MAX_ROW_KEYS}.",
            ("n_tokens",),
            DTypeSpec(any_of=(DType.INT32,)),
            IntRangeDistribution(lo=1, hi=f"min({INDEX_PAGE} * max_pages, {_MAX_ROW_KEYS})"),
        ),
        "index_eps": ScalarSpec(
            description="k_norm's epsilon.", dtype=DType.FLOAT32, interval=(0.0, 1.0), default=1e-6
        ),
    },
    outputs={
        # not "out": a contract's outputs are out alone or other names. Lower than glm53_moe_layer's out:
        # a row taking a near-tie slot the reference leaves (topk_indices) attends another key, and where
        # that moves a routed expert it differs by percents. Measured on MI355X, 8 ranks, 24 workloads:
        # sglang's own layer 28.8 dB at the lowest, the kernel 29.9 dB; 41 dB where the slots agree
        "layer_out": TensorSpec(
            description=GLM53_MOE_LAYER.outputs["out"].description,
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            error_bounds=SqnrBound(min_sqnr_db=25.0),
        ),
        # exact, but a full row's slots in any order, and a slot whose index score is near the row's
        # index_topk-th in place of a taken one (the reference's check)
        "topk_indices": TensorSpec(
            description="The KV slots each row attends; -1: none.",
            shape=("n_tokens", "index_topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": GLM53_MOE_LAYER.layout_variants.by_name["row_major"],
            "shuffled_16x16": LayoutVariant(
                layouts=dict.fromkeys(
                    ("qkv_a_weight", "q_b_weight", "o_weight", "w1", "w2", "wq_b_weight", "wk_weight"),
                    Preshuffle(tile=(16, 16)),
                ),
                detect_facts=GLM53_MOE_LAYER.layout_variants.by_name["shuffled_16x16"].detect_facts,
            ),
        }
    ),
    compare=GLM53_MOE_LAYER.compare,
    ranks="tp_size",
    reference=Glm53MoeIndexLayerReference(),
    baseline=Glm53MoeIndexLayerBaseline(widest=_MAX_PAGES * INDEX_PAGE),
)

_DENSE_FACTS = ("gate_up_weight", "down_weight")


GLM53_DENSE_INDEX_LAYER = OpContract(
    contract_id="glm53_dense_index_layer@1.0",
    description=(
        "A GLM-5.3 dense decoder layer (layers 0 to 2) on one rank of the TP group: glm53_moe_index_layer, its "
        "MLP in the MoE's place: on RMSNorm(hidden_states + the group's attention outputs summed), "
        "down(silu(gate(x)) * up(x)), gate and up gate_up's two halves; gate_up and down take their input in "
        "bf16, then in fp8 per 1 x 128 group at a scale of its amax / 448. layer_out is hidden_states plus "
        "the group's attention and MLP outputs, each summed over the group in rank order."
    ),
    op_type="glm53",
    axes={
        **{n: a for n, a in GLM53_MOE_INDEX_LAYER.axes.items() if n not in _MOE_AXES},
        "dense_inter": axis("The dense MLP's intermediate size on this rank.", 128, 65536, multiple_of=128),
        "dense_gate_up_rows": derived_axis("gate_up's rows: gate, then up.", "2 * dense_inter"),
        "dense_gate_up_blocks": derived_axis("128-blocks of gate_up's rows.", "dense_gate_up_rows // 128"),
        "dense_inter_blocks": derived_axis("128-blocks of the intermediate size.", "dense_inter // 128"),
    },
    attrs={"tp_size": GLM53_MOE_LAYER.attrs["tp_size"]},
    constraints=tuple(c for c in GLM53_MOE_INDEX_LAYER.constraints if "n_routed" not in c),
    inputs={
        **{n: s for n, s in GLM53_MOE_INDEX_LAYER.inputs.items() if n not in _MOE_INPUTS},
        **_fp8_linear("gate_up", "dense_gate_up_rows", "model_dim", ("dense_gate_up_blocks", "model_blocks")),
        **_fp8_linear("down", "model_dim", "dense_inter", ("model_blocks", "dense_inter_blocks")),
    },
    # layer_out at glm53_moe_layer's bound: the dense MLP routes no expert (sglang's own layer came to
    # 41.3 dB at the lowest of 24 workloads)
    outputs={**GLM53_MOE_INDEX_LAYER.outputs, "layer_out": GLM53_MOE_LAYER.outputs["out"]},
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_shuffled_facts(_DENSE_FACTS, False)),
            "shuffled_16x16": LayoutVariant(
                layouts=dict.fromkeys(
                    (
                        "qkv_a_weight",
                        "q_b_weight",
                        "o_weight",
                        "wq_b_weight",
                        "wk_weight",
                        "gate_up_weight",
                        "down_weight",
                    ),
                    Preshuffle(tile=(16, 16)),
                ),
                detect_facts=_shuffled_facts(_DENSE_FACTS, True),
            ),
        }
    ),
    compare=GLM53_MOE_INDEX_LAYER.compare,
    ranks="tp_size",
    reference=Glm53DenseIndexLayerReference(),
    baseline=Glm53DenseIndexLayerBaseline(widest=_MAX_PAGES * INDEX_PAGE),
)

GLM53_HEAD = OpContract(
    contract_id="glm53_head@1.0",
    description=(
        "GLM-5.3's LM head on one rank of the TP group, as sglang's LogitsProcessor._get_logits runs it at TP > 1 "
        "without DP attention: logits[:, r * vocab_rows : (r + 1) * vocab_rows] is rank r's hidden_states times "
        "its lm_head_weight^T, rounded to bf16 as its bf16 GEMM outputs them; every rank's gathered over the "
        "group, in float32."
    ),
    op_type="glm53",
    axes={
        "n_tokens": GLM53_MOE_LAYER.axes["n_tokens"],
        "model_dim": GLM53_MOE_LAYER.axes["model_dim"],
        "vocab_rows": axis("This rank's rows of the vocabulary.", 1, 2**20),
        "vocab": axis("The vocabulary: every rank's rows.", 1, 2**23),
    },
    attrs={"tp_size": GLM53_MOE_LAYER.attrs["tp_size"]},
    constraints=("vocab == tp_size * vocab_rows",),
    inputs={
        "hidden_states": _tensor("The final norm's output.", ("n_tokens", "model_dim"), _BF16),
        "lm_head_weight": _tensor(
            "This rank's rows of the LM head: rank r's, the vocabulary's from r * vocab_rows on.",
            ("vocab_rows", "model_dim"),
            _BF16,
            LecunNormalDistribution(fan_in="model_dim"),
            weight=True,
        ),
    },
    outputs={
        "logits": TensorSpec(
            description="Each row's logits over the vocabulary.",
            shape=("n_tokens", "vocab"),
            dtype=_FP32,
            # bf16's common bound: a bf16 GEMM's outputs, widened to float32
            error_bounds=tolerance(DType.BFLOAT16),
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    ranks="tp_size",
    reference=Glm53HeadReference(),
    baseline=Glm53HeadBaseline(),
)

GLM53_NEXTN_ENTRY = OpContract(
    contract_id="glm53_nextn_entry@1.0",
    description=(
        "GLM-5.3's NextN (MTP) layer input, as sglang's DeepseekModelNextN computes it before its decoder layer: "
        "eh_proj(enorm(e) | hnorm(hidden_states)). e is each row's token embedding from the vocab-parallel "
        "embed_tokens (rank r holds vocab_rows rows from r * vocab_rows); enorm and hnorm are RMSNorms (eps) in "
        "float32 rounded to bf16; eh_proj's weight is whole on every rank, its sums rounded to bf16. Rank q computes "
        "the columns out[:, q * model_dim / tp_size ..] from its own call, and every rank's out is these in rank "
        "order."
    ),
    op_type="glm53",
    axes={
        "n_tokens": GLM53_MOE_LAYER.axes["n_tokens"],
        "model_dim": GLM53_MOE_LAYER.axes["model_dim"],
        "vocab_rows": GLM53_HEAD.axes["vocab_rows"],
        "eh_cols": derived_axis("eh_proj's input: e's normalized row, then h's.", "2 * model_dim"),
    },
    attrs={"tp_size": GLM53_MOE_LAYER.attrs["tp_size"]},
    # a rank's share of eh_proj's columns
    constraints=("model_dim % tp_size == 0",),
    inputs={
        "input_ids": _tensor(
            "Each row's token.",
            ("n_tokens",),
            DTypeSpec(any_of=(DType.INT64, DType.INT32)),
            IntRangeDistribution(lo=0, hi="tp_size * vocab_rows - 1"),
        ),
        "hidden_states": _tensor("Each row's hidden state, the target's.", ("n_tokens", "model_dim"), _BF16),
        "embed_weight": _tensor(
            "This rank's rows of embed_tokens: rank r's, the vocabulary's from r * vocab_rows on.",
            ("vocab_rows", "model_dim"),
            _BF16,
            weight=True,
        ),
        "enorm_weight": _tensor("enorm's weight.", ("model_dim",), _BF16, weight=True),
        "hnorm_weight": _tensor("hnorm's weight.", ("model_dim",), _BF16, weight=True),
        "eh_proj_weight": _tensor(
            "eh_proj's weight, whole on every rank.",
            ("model_dim", "eh_cols"),
            _BF16,
            LecunNormalDistribution(fan_in="eh_cols"),
            weight=True,
        ),
        "eps": GLM53_MOE_LAYER.inputs["eps"].model_copy(
            update={"description": "enorm's and hnorm's epsilon."}
        ),
    },
    outputs={
        "out": TensorSpec(
            description="The NextN decoder layer's input.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            # bf16's common bound: a bf16 GEMM's outputs on bf16 RMSNorms
            error_bounds=tolerance(DType.BFLOAT16),
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    ranks="tp_size",
    reference=Glm53NextnEntryReference(),
    baseline=Glm53NextnEntryBaseline(),
)

GLM53_EMBED = OpContract(
    contract_id="glm53_embed@1.0",
    description=(
        "GLM-5.3's token embedding on one rank of the TP group, as sglang's vocab-parallel embed_tokens computes it "
        "before the first layer: rank r holds its vocab_rows rows of the vocabulary from r * vocab_rows; a row is the "
        "group's sum, in rank order, of each rank's lookup of its own input_ids in its rows (zeros for an id another "
        "rank holds), in float32, rounded to embed_weight's dtype."
    ),
    op_type="glm53",
    axes={
        "n_tokens": GLM53_MOE_LAYER.axes["n_tokens"],
        "model_dim": GLM53_MOE_LAYER.axes["model_dim"],
        "vocab_rows": GLM53_HEAD.axes["vocab_rows"],
    },
    attrs={"tp_size": GLM53_MOE_LAYER.attrs["tp_size"]},
    inputs={
        "input_ids": GLM53_NEXTN_ENTRY.inputs["input_ids"],
        "embed_weight": GLM53_NEXTN_ENTRY.inputs["embed_weight"],
    },
    outputs={
        "embeddings": TensorSpec(
            description="Each row's token embedding.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="embed_weight"),
            # exact: rows and zeros summed in float32, rounded once
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    ranks="tp_size",
    reference=Glm53EmbedReference(),
    baseline=Glm53EmbedBaseline(),
)

CONTRACTS = (
    GLM53_MOE_LAYER,
    GLM53_MOE_INDEX_LAYER,
    GLM53_DENSE_INDEX_LAYER,
    GLM53_HEAD,
    GLM53_NEXTN_ENTRY,
    GLM53_EMBED,
)
