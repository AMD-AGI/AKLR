# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3-Next's Gated DeltaNet core between sglang's input and output projections: the
causal convolution, the gated delta recurrence, the gated RMSNorm and the grouped FP8
quantization of out_proj's input, for prefill and for decode.

sglang's Qwen3-Next holds the Q/K/V/Z and B/A projections grouped by key head: each key
head's Q, its K, then the V and the Z of its value heads; and each key head's B, then
its A. Qwen3.5 holds them contiguous, so the qwen3_5 contracts do not describe these
inputs. A model is a scenario: Qwen3-Next-80B-A3B at attention TP4 has 4 key heads,
8 value heads, key_dim and value_dim 128."""

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    BoundedCuSeqlensDistribution,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
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

__all__ = [
    "CONTRACTS",
    "QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT",
    "QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT",
]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()


def _tensor(description, shape, dtype, *, mutable=False, bound=None, distribution=_FINITE):
    return TensorSpec(
        description=description,
        shape=shape,
        dtype=dtype,
        distribution=distribution,
        mutable=mutable,
        error_bounds=() if bound is None else bound,
    )


# the heads on one rank of the attention TP group, and the widths they give
_HEAD_AXES = {
    "n_v_heads": axis("Value heads on this rank.", 1, 64),
    "key_dim": axis("Key head width.", 16, 128),
    "value_dim": axis("FP8 group and value-head width.", 128, 128),
    "conv_dim": axis("Convolution channels: every Q, then every K, then every V.", 160, 8192),
    "value_width": derived_axis("Total value width.", "n_v_heads * value_dim"),
    "n_k_heads": derived_axis("Key heads on this rank.", "(conv_dim - value_width) // (2 * key_dim)"),
    "projection_dim": derived_axis(
        "Q/K/V/Z projection width: per key head, 2 * key_dim + 2 * value_dim * n_v_heads // n_k_heads.",
        "conv_dim + value_width",
    ),
    "gate_dim": derived_axis("B/A projection width.", "2 * n_v_heads"),
}
_HEAD_CONSTRAINTS = (
    "conv_dim > value_width",
    "(conv_dim - value_width) % (2 * key_dim) == 0",
    "n_k_heads >= 1",
    "n_v_heads % n_k_heads == 0",
)
_PROJECTIONS = {
    "projected_qkvz": _tensor(
        "Q/K/V/Z projections grouped by key head: each key head's Q, its K, then the V "
        "and the Z of its value heads.",
        ("n_tokens", "projection_dim"),
        _BF16,
    ),
    "projected_ba": _tensor(
        "B/A projections grouped by key head: the B, then the A, of its value heads.",
        ("n_tokens", "gate_dim"),
        _BF16,
    ),
}
_CONV_STATE = _tensor(
    "Live three-token convolution histories, channels as conv_dim orders them.",
    ("n_slots", "conv_dim", 3),
    _BF16,
    mutable=True,
    bound=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
)
_WEIGHTS = {
    "conv_weight": _tensor("Four causal convolution taps, oldest first.", ("conv_dim", 4), _BF16),
    "conv_bias": _tensor("Convolution bias.", ("conv_dim",), _BF16),
    "a_log": _tensor("Log decay magnitude.", ("n_v_heads",), _FP32),
    "dt_bias": _tensor("Gate bias.", ("n_v_heads",), _BF16),
    "norm_weight": _tensor("Gated RMSNorm weights.", ("value_dim",), _BF16),
    "scale": ScalarSpec(
        description="Query scale.",
        dtype=DType.FLOAT32,
        interval=(0.0883883461356163, 128**-0.5),
        default=128**-0.5,
    ),
    "eps": ScalarSpec(
        description="Gated RMSNorm epsilon.",
        dtype=DType.FLOAT32,
        interval=(9.999999974752427e-7, 1e-6),
        default=1e-6,
    ),
}
_OUTPUTS = {
    "normalized": TensorSpec(
        description="Gated normalized BF16 values.",
        shape=("n_tokens", "n_v_heads", "value_dim"),
        dtype=_BF16,
        error_bounds=ToleranceBound(absolute_tolerance=0.005, relative_tolerance=0.02),
    ),
    "values": TensorSpec(
        description="Grouped E4M3FN values.",
        shape=("n_tokens", "value_width"),
        dtype=_FP8,
        error_bounds=ToleranceBound(absolute_tolerance=2, relative_tolerance=0.125),
    ),
    "scales": TensorSpec(
        description="One FP32 scale per token and value head.",
        shape=("n_tokens", "n_v_heads"),
        dtype=_FP32,
        error_bounds=ToleranceBound(absolute_tolerance=1e-7, relative_tolerance=0.02),
    ),
}
_STEP = (
    "the width-four causal convolution of the history and the token's Q/K/V channels "
    "adds the bias to the FP32 sum of the four products, oldest tap first, and its SiLU "
    "is rounded to BF16. beta, the sigmoid of B, is rounded to BF16, and the decay is "
    "exp(-exp(a_log) * softplus(A + dt_bias)). Each value head uses its key head's Q "
    "and K, L2-normalized with epsilon 1e-6; the gated delta rule updates the FP32 "
    "delta_state in [value,key] order, and the core output, Q times the state times "
    "scale, is rounded to BF16. The core is gated RMS-normalized with norm_weight and "
    "SiLU of the head's Z, rounded to BF16, and quantized per value head to E4M3FN with "
    "scale=max(amax,1e-10)/448"
)

QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT = OpContract(
    contract_id="qwen3_next_gdn_prefill_group_fp8_quant@1.0",
    op_type="qwen3_next",
    description=(
        "Qwen3-Next GDN prefill from Q/K/V/Z and B/A projections grouped by key head. "
        "cu_seqlens partitions the tokens into positive sequences with actual maximum "
        "max_seq_len; each sequence uses its distinct cache_indices slot, and a fresh "
        "one (has_initial_state false) starts from zero history and state, ignoring the "
        f"slot's old values. For each token, {_STEP}; Q and K are rounded to BF16 after "
        "their normalization. Each sequence's slot keeps its last three tokens and its "
        "final state; unindexed slots and all other inputs are preserved."
    ),
    axes={
        "n_tokens": axis("Tokens in this prefill.", 1, 16384, is_var=True),
        "n_sequences": axis("Sequences, each with its own state slot.", 1, 2047, is_var=True),
        "n_slots": axis("Allocated state slots.", 1, 2047, is_var=True, any_size=True),
        **_HEAD_AXES,
        "n_starts": axis("Sequence boundaries.", 2, 2048, is_var=True),
    },
    attrs={
        "max_seq_len": attr(
            "Actual maximum sequence length.",
            1024,
            2,
            4096,
            5120,
            default=1024,
            param_name="max_seq_len",
        ),
    },
    constraints=(
        *_HEAD_CONSTRAINTS,
        "n_slots >= n_sequences",
        "n_starts == n_sequences + 1",
        "n_tokens >= n_sequences + max_seq_len - 1",
        "n_tokens <= n_sequences * max_seq_len",
    ),
    inputs={
        **_PROJECTIONS,
        "conv_state": _CONV_STATE,
        "delta_state": _tensor(
            "Live recurrent matrices, value dimension first.",
            ("n_slots", "n_v_heads", "value_dim", "key_dim"),
            _FP32,
            mutable=True,
            # as qwen3_5's prefill: sglang's own prefill puts Q/K/V channels one BF16
            # ulp from the reference and rounds again in its chunked recurrence
            bound=ToleranceBound(absolute_tolerance=2e-2, relative_tolerance=1e-2),
        ),
        "cache_indices": _tensor(
            "Distinct state slots for the sequences.",
            ("n_sequences",),
            _INT32,
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        "cu_seqlens": _tensor(
            "Cumulative sequence boundaries.",
            ("n_starts",),
            _INT32,
            distribution=BoundedCuSeqlensDistribution(total="n_tokens", max_seqlen="max_seq_len"),
        ),
        "has_initial_state": _tensor(
            "Continue each slot's history and state, or start from zero.",
            ("n_sequences",),
            DTypeSpec(any_of=(DType.BOOL,)),
        ),
        **_WEIGHTS,
    },
    outputs=_OUTPUTS,
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT = OpContract(
    contract_id="qwen3_next_gdn_decode_group_fp8_quant@1.0",
    op_type="qwen3_next",
    description=(
        "Qwen3-Next GDN decode, one token per row, from Q/K/V/Z and B/A projections "
        "grouped by key head. A row whose cache index is -1 is padding: it changes no "
        "state, and its core output is zero. Every other row uses its "
        f"distinct slot: {_STEP}. Q and K stay FP32 after their normalization, and the "
        "slot's history shifts the row in. Unindexed slots and all other inputs are "
        "preserved."
    ),
    axes={
        "n_tokens": axis("Decode rows: one token each, padding rows included.", 1, 2048, is_var=True),
        "n_slots": axis("Allocated state slots.", 1, 2047, is_var=True, any_size=True),
        **_HEAD_AXES,
    },
    constraints=(
        *_HEAD_CONSTRAINTS,
        # drawn indices are distinct, -1 among them at most once
        "n_tokens <= n_slots + 1",
    ),
    inputs={
        **_PROJECTIONS,
        "conv_state": _CONV_STATE,
        "delta_state": _tensor(
            "Live recurrent matrices, value dimension first.",
            ("n_slots", "n_v_heads", "value_dim", "key_dim"),
            _FP32,
            mutable=True,
            # as qwen3_5's decode: one step's change is |update x key|, and a Q/K/V
            # channel a correct kernel rounds to the neighbouring BF16 value moves it
            # by 2**-8 of that
            bound=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
        ),
        "cache_indices": _tensor(
            "Each row's distinct state slot, or -1 for a padding row.",
            ("n_tokens",),
            _INT32,
            distribution=IntRangeDistribution(lo=-1, hi="n_slots - 1", distinct=True),
        ),
        **_WEIGHTS,
    },
    outputs=_OUTPUTS,
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

CONTRACTS = (QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT, QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT)
