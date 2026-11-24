# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA layer between its in-projection and o_proj, for a speculative verify step, as vLLM runs it on ROCm
(``models/kimi_k3/amd/kda.py``): the decay gate's second projection (f_b_proj), the short convolution's update of
each window from its accepted token's place in the convolution state, the gated delta rule through the window (each
token's recurrent state written to its slot) and the output's gated RMSNorm. vLLM runs it as three kernels: a copy of
f_a, the f_b GEMM and aiter's fused convolution, recurrence and norm.

The layer hands over views of its in-projection's rows (q|k|v, the output gate, f_a and beta are columns of one row,
so their rows are apart by more than a row: ``strided_rows``), its weights, and the paged caches: the convolution
state in vLLM's SD layout and the recurrent state, slots of pages they share."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    StridedRows,
    ToleranceBound,
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
    axis,
    derived_axis,
)
from aklr.plugins.vllm.models.kimi_k3.kda_spec_stage_c.baseline import KimiK3KdaSpecStageCBaseline
from aklr.plugins.vllm.models.kimi_k3.kda_spec_stage_c.reference import KimiK3KdaSpecStageCReference

__all__ = ["CONTRACTS", "KIMI_K3_KDA_SPEC_STAGE_C", "MAX_WINDOW"]

# tokens of a window: DSpark's 8 (7 drafts) at most
MAX_WINDOW = 8

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_SAME_AS_QKV = DTypeSpec(same_as_tensor="qkv")
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()
_HEADS = ("n_tokens", "n_heads", "head_dim")


def _storage_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="storage", expected_value=value),)


KIMI_K3_KDA_SPEC_STAGE_C = OpContract(
    contract_id="kimi_k3_kda_spec_stage_c@1.0",
    description=(
        "Kimi-K3's KDA layer from its in-projection's output to its gated norm's, for windows of w = window tokens: "
        "sequence s's window is rows s * w .. s * w + w - 1, its slots state_indices[s * w .. s * w + w - 1], its "
        "start slot state_indices[s * w + a] with a = num_accepted_tokens[s] - 1, its convolution slot "
        "state_indices[s * w]. Rows from n_seqs * w on are padding. (1) raw_g = bf16(f_a @ f_b_weight.T). (2) Per "
        "packed q|k|v channel c, the window's convolution reads x = conv_state[slot, a .. a + width - 2, c] then the "
        "window's qkv inputs; token t's y = bf16(silu(sum_j conv_weight[c, j] * x[t + j])) in float32; the slot "
        "then holds conv_state[slot, a + 1 .. a + width - 2, c] followed by the window's inputs, written where the "
        "convolution slot is positive. (3) The gated delta rule on (y, raw_g, raw_beta): q and k L2-normalized "
        "(x * rsqrt(sum(x**2) + 1e-6)), q times scale, the gate lower_bound * sigmoid(exp(a_log[h]) * (raw_g + "
        "g_bias[h])), beta = sigmoid(raw_beta); from the start slot's (v, k) state S, token by token, S = S * "
        "exp(gate) along k, S = S + beta * (v - S @ k) k^T, o = bf16(S @ q), S written to the token's slot where "
        "that is positive. (4) out = o * rsqrt(mean(o**2) + eps) * norm_weight * sigmoid(out_gate), the mean over a "
        "head. A window whose start slot is <= 0 (vLLM's padding) gives zero rows and writes no slot; padding rows "
        "are zero. Slots no window writes are left unchanged."
    ),
    op_type="kda_spec_stage_c",
    axes={
        # a verify step's sequences, up to the scheduler's: each count a workload of its own, as their rows
        "n_seqs": axis("Sequences of the step, a window each.", 1, 32, is_var=True),
        "n_tokens": axis(
            "Rows of the step: every window back to back, then fewer padding rows than a window (vLLM's graph "
            "padding).",
            1,
            512,
            is_var=True,
        ),
        # the kernels reach their slots through state_indices: any cache size serves
        "n_slots": axis("Slots of the paged caches.", 2, 65536, is_var=True, any_size=True),
        "n_heads": axis("KDA heads on this rank.", 1, 128),
        "head_dim": axis(
            "Width of every q, k, v and gate head, the side of a head's square state, and f_a's width.",
            128,
            128,
        ),
        "width": axis("Taps of the short convolution.", 4, 4),
        # vLLM's conv_kernel_size - 1 + num_speculative_tokens: width - 1 inputs before a window, then one per draft
        "conv_len": axis("Inputs a convolution slot keeps.", 3, MAX_WINDOW + 2),
        "qkv_heads": derived_axis("Heads of a packed q|k|v row: q's, then k's, then v's.", "3 * n_heads"),
        "state_rows": derived_axis(
            "Rows of a slot's recurrent state: every head's, v by v.", "n_heads * head_dim"
        ),
        "conv_dim": derived_axis("Channels of the packed q|k|v convolution.", "3 * n_heads * head_dim"),
        "window": derived_axis(
            "Tokens of a window: the one after the accepted token, then the drafts.", "conv_len - width + 2"
        ),
    },
    constraints=("n_tokens >= n_seqs * window", "n_tokens < n_seqs * window + window", "n_tokens < n_slots"),
    inputs={
        "qkv": TensorSpec(
            description=(
                "The in-projection's q|k|v columns of each row, before the short convolution: heads 0 .. n_heads - 1 "
                "are q's, then k's, then v's."
            ),
            shape=("n_tokens", "qkv_heads", "head_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "out_gate": TensorSpec(
            description="The in-projection's output gate columns (g_proj's), before the sigmoid.",
            shape=_HEADS,
            dtype=_SAME_AS_QKV,
            distribution=_FINITE,
        ),
        "f_a": TensorSpec(
            description="The in-projection's f_a columns: the decay gate's low-rank input, a unit middle dim.",
            shape=("n_tokens", 1, "head_dim"),
            dtype=_SAME_AS_QKV,
            distribution=_FINITE,
        ),
        "raw_beta": TensorSpec(
            description="The in-projection's beta columns, before the sigmoid, a unit last dim.",
            shape=("n_tokens", "n_heads", 1),
            dtype=_SAME_AS_QKV,
            distribution=_FINITE,
        ),
        "f_b_weight": TensorSpec(
            description="f_b_proj's weight: raw_g = f_a @ f_b_weight.T, head by head.",
            shape=("state_rows", "head_dim"),
            dtype=_SAME_AS_QKV,
            # raw_g about N(0, 1) from f_a's N(0, 1): a_log and g_bias set the decays
            distribution=LecunNormalDistribution(fan_in="head_dim"),
            lifetime="weight",
        ),
        "conv_weight": TensorSpec(
            description="Each packed channel's taps, the oldest input's first (vLLM's conv1d weight, float32).",
            shape=("conv_dim", "width"),
            dtype=_FP32,
            distribution=LecunNormalDistribution(fan_in="width"),
            lifetime="weight",
        ),
        "conv_state": TensorSpec(
            description=(
                "The paged convolution state, vLLM's SD layout: per slot, conv_len inputs, the oldest first, each a "
                "row of every packed channel."
            ),
            shape=("n_slots", "conv_len", "conv_dim"),
            dtype=_SAME_AS_QKV,
            distribution=_FINITE,
            mutable=True,
            # copies of bf16 inputs
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
        "a_log": TensorSpec(
            description="Log of each head's gate scale (vLLM's A_log).",
            shape=("n_heads",),
            dtype=_FP32,
            # log(U[1, 16]), as KDA initializes it: exp(a_log) in [1.3, 16]
            distribution=LogUniformDistribution(log2_lo=-2, log2_hi=1.47),
            lifetime="weight",
        ),
        "g_bias": TensorSpec(
            description="The gate's per-channel bias (vLLM's dt_bias, flat (n_heads * head_dim)).",
            shape=("n_heads", "head_dim"),
            dtype=_FP32,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "norm_weight": TensorSpec(
            description="The gated RMSNorm's weight (vLLM's o_norm), one per head channel.",
            shape=("head_dim",),
            dtype=_SAME_AS_QKV,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "state_cache": TensorSpec(
            description=(
                "The paged recurrent states: slot by slot, each head's (v, k) state, the head's head_dim rows of "
                "head_dim keys."
            ),
            shape=("n_slots", "state_rows", "head_dim"),
            dtype=_FP32,
            distribution=_FINITE,
            mutable=True,
            # a window's float32 steps; a bf16 rounding of raw_g (or of a convolution output) that flips with the
            # summation order moves a decay by up to a few percent: that column's elements. The committed kernel
            # peaks at 0.19 of the bound, no element past it (MI355X, 1 to 32 windows, chained steps)
            error_bounds=ToleranceBound(absolute_tolerance=2e-3, relative_tolerance=5e-2),
        ),
        "state_indices": TensorSpec(
            description=(
                "Each row's slot, window by window (vLLM's [n_seqs, window] rows of its block table), a padding "
                "row's unread. The positive slots are distinct; 0 and -1 may repeat: every slot of a padding window "
                "(vLLM's graph padding) is its null block 0, or -1."
            ),
            shape=("n_tokens",),
            dtype=_INT32,
            # drawn positive and distinct, every window live, so that every timed window steps its states: no core
            # distribution keeps the positive values distinct while 0 and -1 repeat, and a plugin cannot add one
            # (bench draws from the schema's json). The padding windows are the reference's and the tests' (on
            # gfx950, the kernel's)
            distribution=IntRangeDistribution(lo=1, hi="n_slots - 1", distinct=True),
        ),
        "num_accepted_tokens": TensorSpec(
            description="Each sequence's tokens accepted at its last step, 1 .. window.",
            shape=("n_seqs",),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo=1, hi="window"),
        ),
        "scale": ScalarSpec(
            description="Multiplies the normalized queries; vLLM passes head_dim ** -0.5.",
            dtype=DType.FLOAT32,
            interval=(2.0**-126, 1.0),
            default=128**-0.5,
        ),
        "lower_bound": ScalarSpec(
            description="The gate's floor per token (Kimi-K3's gate_lower_bound); vLLM asserts it in [-5, 0).",
            dtype=DType.FLOAT32,
            interval=(-5.0, -(2.0**-126)),
            default=-5.0,
        ),
        "eps": ScalarSpec(
            description="Added to the mean square of a head's output in the gated norm (vLLM's o_norm.eps).",
            dtype=DType.FLOAT32,
            interval=(2.0**-126, _FLOAT32_LARGEST),
            default=1e-5,
        ),
    },
    outputs={
        "out": TensorSpec(
            description="Each row's gated, normalized heads: zero for a padding window or a padding row.",
            shape=_HEADS,
            dtype=_SAME_AS_QKV,
            # a normalized bf16 output from bf16 hand-offs (raw_g, y, o): a flipped rounding upstream is a few
            # parts in 1e3 of a head; and one bf16 rounding (2**-7) for the reference's own out
            error_bounds=ToleranceBound(absolute_tolerance=2e-3, relative_tolerance=2e-2 + 2**-7),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_storage_facts("row_major")),
            # vLLM's: q|k|v, the output gate, f_a and beta columns of the in-projection's rows; each slot of the
            # caches a part of a page they share
            "strided_rows": LayoutVariant(
                layouts={
                    "qkv": StridedRows(),
                    "out_gate": StridedRows(),
                    "f_a": StridedRows(),
                    "raw_beta": StridedRows(),
                    "conv_state": StridedRows(),
                    "state_cache": StridedRows(),
                },
                detect_facts=_storage_facts("strided_rows"),
            ),
        }
    ),
    compare=CompareSpec(determinism=DeterminismLevel.BITWISE),
    reference=KimiK3KdaSpecStageCReference(),
    baseline=KimiK3KdaSpecStageCBaseline(),
)

CONTRACTS = (KIMI_K3_KDA_SPEC_STAGE_C,)
