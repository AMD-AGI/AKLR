# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    SqnrBound,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DeterminismLevel,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.plugins.aiter.muse_spark_layer.baseline import MuseSparkLayerBaseline
from aklr.plugins.aiter.muse_spark_layer.reference import MuseSparkLayerReference

__all__ = ["CONTRACTS", "MUSE_SPARK_LAYER"]

_HEAD_DIM = 64
_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_INT64 = DTypeSpec(any_of=(DType.INT64,))
_FINITE = FiniteDistribution()
_ROWS = ("n_rows", "model_dim")
# Past the experts every row carries the MXFP4 requantization of the expert input: the
# kernels' input differs from the reference's by bf16 ulps (the pre-FFN norm folded into
# the GEMV), which moves E2M1 codes; measured 23-24 dB against the reference
_SQNR = SqnrBound(min_sqnr_db=18.0)
# about [0.87, 1.15], a norm's gain
_GAIN = LogUniformDistribution(log2_lo="-0.2", log2_hi="0.2")


def _weight(description: str, shape: tuple, fan_in: str, *, shared: bool = False) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=shape,
        dtype=_BF16,
        distribution=LecunNormalDistribution(fan_in=fan_in),
        lifetime="weight",
        shared=shared,
    )


def _gain(description: str, width: str) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=(width,),
        dtype=_BF16,
        distribution=_GAIN,
        lifetime="weight",
        shared=True,
    )


def _mix(description: str, log2_lo: str, log2_hi: str) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=("model_dim",),
        dtype=_FP32,
        distribution=LogUniformDistribution(log2_lo=log2_lo, log2_hi=log2_hi),
        lifetime="weight",
        shared=True,
    )


def _e8m0_scales(fan_in: str) -> LogUniformDistribution:
    """One per 1 x 32 block, around 1 / sqrt(fan-in), as fused_moe's draw."""
    return LogUniformDistribution(
        log2_lo=f"-5 - bit_length({fan_in}) // 2", log2_hi=f"1 - bit_length({fan_in}) // 2"
    )


def _kv_cache(description: str) -> TensorSpec:
    return TensorSpec(
        description=description, shape=("n_slots", "n_kv_heads", _HEAD_DIM), dtype=_FP8, distribution=_FINITE
    )


def _next_kv_cache(description: str) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=("n_slots", "n_kv_heads", _HEAD_DIM),
        dtype=_FP8,
        distribution=_FINITE,
        mutable=True,
        error_bounds=ToleranceBound(absolute_tolerance=2e-2, relative_tolerance=1 / 16),
    )


def _rows(description: str, width: tuple) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=("n_rows", *width),
        dtype=_BF16,
        error_bounds=_SQNR,
    )


MUSE_SPARK_LAYER = OpContract(
    contract_id="muse_spark_layer@1.0",
    description=(
        "One decoder layer of Muse Spark with MXFP4 experts, on every rank of its tensor-parallel"
        " group, from the layer's projected queries to the next layer's, as a verify step of"
        " n_rows rows, 8 a request, runs it. Attention: each query head over its request's cached"
        " keys and values (fp8, 16 a page through page_table): context_lens keys before the"
        " request's 8 rows, then the rows' own, causal; with sliding_window > 0 only keys fewer"
        " than sliding_window positions before the row. softmax(q k^T * scale), then per head"
        " RMSNorm(o) * sigmoid(o_gate). The attention boundary: o_proj's partial rows summed over"
        " the ranks, RMSNorm without a gain (eps 1e-8), residual = alpha * residual + beta * that,"
        " x = RMSNorm(residual) * pre_norm_weight. The router: sigmoid(0.17677669 x (router_hi +"
        " router_lo)^T); a row's 8 experts are the largest of it plus router_bias (a tie to the"
        " lower id), their sigmoids normalized to sum 1 (eps 1e-15) its scores. The experts on"
        " the pooled row RMSNorm(x pooler^T) * pooled_norm_weight: per pick down(silu(gate) * up)"
        " on MXFP4 weights and MXFP4 activations, times expert_norm_weight, RMS-normalized (eps"
        " 1e-8) and times its score, summed over the picks; that unpooled, times post_ffn_weight,"
        " RMS-normalized (eps 1e-8); residual = alpha2 * residual + beta2 * that, next_input ="
        " RMSNorm(residual) * next_norm_weight. The next layer's projection of next_input: q, k,"
        " v and o_gate; q and k RMS-normalized per head, q then times 5.4730064863875, both"
        " rotated (interleaved RoPE, rotary at positions) when next_rope; k and v written to the"
        " next layer's fp8 caches, saturated at +-448, at slots (at swa_map of slots when"
        " next_sliding). Norms without an eps above take 1e-5; each stage rounds to bf16 where"
        " the model's ops do (the reference's docstring lists them)."
    ),
    op_type="muse_spark_layer",
    axes={
        "n_rows": axis("Rows of the verify step: 8 a request.", 8, 64, multiple_of=8, is_var=True),
        "n_requests": axis("Requests of the step.", 1, 8, is_var=True),
        # the kernels read the caches and the rotary table through indices: any size serves
        "n_slots": axis(
            "Rows of each KV cache, 16 a page.", 16, 2**30, multiple_of=16, is_var=True, any_size=True
        ),
        "max_blocks": axis("Pages a request in the page table.", 1, 2**26, is_var=True, any_size=True),
        "n_positions": axis("Rows of the rotary table.", 1, 2**24, is_var=True, any_size=True),
        "model_dim": axis("Hidden size.", 64, 16384, multiple_of=64),
        "n_heads": axis("Query heads of each rank, 64 values each.", 1, 128),
        "n_kv_heads": axis("KV heads of each rank.", 1, 16),
        "n_experts": axis("Routed experts.", 8, 1024),
        "pooled_dim": axis("Width between the pooler and the unpooler.", 256, 16384, multiple_of=256),
        "inter_dim": axis("Expert intermediate size.", 256, 16384, multiple_of=256),
        "q_width": derived_axis("Query and o_gate columns of each rank.", "n_heads * 64"),
        "qkvg_rows": derived_axis(
            "Rows of each rank's next projection.", "2 * n_heads * 64 + 2 * n_kv_heads * 64"
        ),
        "gate_up_slice": derived_axis(
            "Gate and up rows of each expert a rank holds.", "2 * inter_dim // tp_size"
        ),
        "pooled_slice": derived_axis(
            "Down-projection rows of each expert a rank holds.", "pooled_dim // tp_size"
        ),
        "pooled_blocks": derived_axis("w13 scale columns.", "pooled_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    attrs={
        "tp_size": attr("Ranks of the tensor-parallel group.", 8, 1, default=8, param_name="tp_size"),
    },
    constraints=(
        "n_rows == 8 * n_requests",
        "max_blocks * 16 <= n_slots",
        "n_heads % n_kv_heads == 0",
        "8 <= n_experts",
        "inter_dim % (32 * tp_size) == 0",
        "pooled_dim % (32 * tp_size) == 0",
    ),
    inputs={
        "q": TensorSpec(
            description=(
                "This rank's query heads of the layer, normed, scaled and rotated; the call"
                " leaves the next layer's."
            ),
            shape=("n_rows", "n_heads", _HEAD_DIM),
            dtype=_BF16,
            distribution=_FINITE,
            mutable=True,
            error_bounds=_SQNR,
        ),
        "o_gate": TensorSpec(
            description="This rank's output gate of the layer; the call leaves the next layer's.",
            shape=("n_rows", "n_heads", _HEAD_DIM),
            dtype=_BF16,
            distribution=_FINITE,
            mutable=True,
            error_bounds=_SQNR,
        ),
        "residual": TensorSpec(
            description="The residual stream, the same on every rank: updated in place twice.",
            shape=_ROWS,
            dtype=_FP32,
            distribution=_FINITE,
            mutable=True,
            shared=True,
            error_bounds=_SQNR,
        ),
        "k_cache": _kv_cache("This rank's KV heads of keys of the layer, a row a slot."),
        "v_cache": _kv_cache("This rank's values of the layer, a row a slot."),
        "page_table": TensorSpec(
            description="Each request's pages of the layer's caches, in context order.",
            shape=("n_requests", "max_blocks"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="0", hi="n_slots // 16 - 1"),
            shared=True,
        ),
        "context_lens": TensorSpec(
            description="Each request's keys before the step's 8 rows.",
            shape=("n_requests",),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="max_blocks * 8", hi="max_blocks * 16 - 8"),
            shared=True,
        ),
        "o_proj": _weight(
            "This rank's columns of the output projection.", ("model_dim", "q_width"), "q_width * tp_size"
        ),
        "alpha": _mix("Residual scale at the attention boundary.", "-1", "0"),
        "beta": _mix("Branch scale at the attention boundary.", "-2", "-1"),
        "pre_norm_weight": _gain("The pre-FFN norm's gain.", "model_dim"),
        "router_hi": _weight(
            "The fp32 router's bf16 high part.", ("n_experts", "model_dim"), "model_dim", shared=True
        ),
        "router_lo": _weight(
            "The fp32 router's bf16 remainder, about 2**-8 of the high part.",
            ("n_experts", "model_dim"),
            "model_dim * 65536",
            shared=True,
        ),
        "router_bias": TensorSpec(
            description="Added to the router's sigmoids to pick, not to score.",
            shape=("n_experts",),
            dtype=_FP32,
            distribution=_FINITE,
            lifetime="weight",
            shared=True,
        ),
        "pooler": _weight(
            "The pooler, the same on every rank.", ("pooled_dim", "model_dim"), "model_dim", shared=True
        ),
        "pooled_norm_weight": _gain("The pre-expert norm's gain.", "pooled_dim"),
        "w13": TensorSpec(
            description=(
                "Gate and up projections of every expert, this rank's slice of each: gate rows"
                " inter_dim / tp_size * rank on, then as many up rows from inter_dim / tp_size * rank."
            ),
            shape=("n_experts", "gate_up_slice", "pooled_dim"),
            dtype=_FP4,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "w13_scale": TensorSpec(
            description="Per 1 x 32 block of w13.",
            shape=("n_experts", "gate_up_slice", "pooled_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("pooled_dim"),
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection of every expert, this rank's slice of each: rows pooled_slice * rank on.",
            shape=("n_experts", "pooled_slice", "inter_dim"),
            dtype=_FP4,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="Per 1 x 32 block of w2.",
            shape=("n_experts", "pooled_slice", "inter_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("inter_dim"),
            lifetime="weight",
        ),
        "expert_norm_weight": _gain("Each expert output's gain before its norm.", "pooled_dim"),
        "unpooler": _weight(
            "The unpooler, the same on every rank.", ("model_dim", "pooled_dim"), "pooled_dim", shared=True
        ),
        "post_ffn_weight": _gain("The FFN branch's gain before its norm.", "model_dim"),
        "alpha2": _mix("Residual scale at the FFN boundary.", "-1", "0"),
        "beta2": _mix("Branch scale at the FFN boundary.", "-2", "-1"),
        "next_norm_weight": _gain("The next layer's input norm's gain.", "model_dim"),
        "next_qkvg": _weight(
            "This rank's rows of the next layer's q | k | v | o_gate projection.",
            ("qkvg_rows", "model_dim"),
            "model_dim",
        ),
        "rotary": TensorSpec(
            description="cos, then sin, of each position's 32 rotation angles.",
            shape=("n_positions", _HEAD_DIM),
            dtype=_FP32,
            distribution=_FINITE,
            lifetime="weight",
            shared=True,
        ),
        "positions": TensorSpec(
            description="Each row's position.",
            shape=("n_rows",),
            dtype=_INT64,
            distribution=IntRangeDistribution(lo="0", hi="n_positions - 1"),
            shared=True,
        ),
        "slots": TensorSpec(
            description="Each row's slot in the next layer's caches.",
            shape=("n_rows",),
            dtype=_INT64,
            distribution=IntRangeDistribution(lo="0", hi="n_slots - 1", distinct=True),
            shared=True,
        ),
        "swa_map": TensorSpec(
            description="The sliding pool's row of each slot.",
            shape=("n_slots",),
            dtype=_INT64,
            distribution=IntRangeDistribution(lo="0", hi="n_slots - 1", distinct=True),
            lifetime="weight",
            shared=True,
        ),
        "next_k_cache": _next_kv_cache(
            "This rank's keys of the next layer: the step's rows written in place."
        ),
        "next_v_cache": _next_kv_cache(
            "This rank's values of the next layer: the step's rows written in place."
        ),
        "scale": ScalarSpec(
            description="Multiplies every attention logit.",
            dtype=DType.FLOAT32,
            values=(0.125,),
            default=0.125,
        ),
        "sliding_window": ScalarSpec(
            description="0 for a full-attention layer, else the window.",
            dtype=DType.INT32,
            values=(0, 2047),
            default=0,
        ),
        "next_rope": ScalarSpec(
            description="Whether the next layer rotates q and k.",
            dtype=DType.BOOL,
            values=(True, False),
            default=True,
        ),
        "next_sliding": ScalarSpec(
            description="Whether the next layer's caches are the sliding pool's, reached through swa_map.",
            dtype=DType.BOOL,
            values=(False, True),
            default=False,
        ),
    },
    outputs={
        "next_input": _rows("The next layer's normed input, the same on every rank.", ("model_dim",)),
        "next_k": _rows("This rank's next-layer keys, normed and rotated.", ("n_kv_heads", _HEAD_DIM)),
        "next_v": _rows("This rank's next-layer values.", ("n_kv_heads", _HEAD_DIM)),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    # the ranks sum their partial rows in rank order: no atomics
    compare=CompareSpec(determinism=DeterminismLevel.BITWISE),
    ranks="tp_size",
    reference=MuseSparkLayerReference(),
    baseline=MuseSparkLayerBaseline(),
)

CONTRACTS = (MUSE_SPARK_LAYER,)
