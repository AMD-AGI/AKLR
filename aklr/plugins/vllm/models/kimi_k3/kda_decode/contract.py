# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA decode as vLLM runs it on ROCm: ``kda.py``'s pure-decode batch, one token a sequence, through
``fused_recurrent_kda_packed_decode``, and its speculative batch, each sequence's window of drafts to verify, through
``fused_recurrent_kda`` with ``num_accepted_tokens`` (vLLM's ``models/kimi_k3/amd/ops/third_party/kda``).

The layer hands over the packed q|k|v row after the short convolution, the full-rank gate projection and the beta
projection before their activations; the step normalizes q and k, activates the gate and beta, and reads and writes
each token's recurrent state in its slot of the paged state cache. vLLM keeps that cache in pages shared with the
convolution state and beta in a column of its fused projection, so their rows are apart by more than a row
(``strided_rows``)."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DeterminismLevel,
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
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)

__all__ = ["CONTRACTS", "KIMI_K3_KDA_DECODE", "KIMI_K3_KDA_DECODE_SPEC", "L2_NORM_EPS"]

# x / sqrt(sum(x**2) + eps): vLLM's packed decode kernel
L2_NORM_EPS = 1e-6

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_SAME_AS_QKV = DTypeSpec(same_as_tensor="qkv")
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FINITE = FiniteDistribution()
_HEADS = ("n_tokens", "n_heads", "head_dim")


def _storage_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="storage", expected_value=value),)


KIMI_K3_KDA_DECODE = OpContract(
    contract_id="kimi_k3_kda_decode@1.0",
    description=(
        "Kimi-K3's KDA decode, one step of the gated delta rule per token (each sequence decodes one): token t's q, "
        "k and v are heads h, n_heads + h and 2 * n_heads + h of qkv[t], and its slot is state_indices[t], whose "
        "head h holds the state S[v, k] = state_cache[slot, h * head_dim + v, k]. Per head, q and k are "
        "L2-normalized (x / sqrt(sum(x**2) + 1e-6)) and q multiplied by scale; the gate g = lower_bound * "
        "sigmoid(exp(a_log[h]) * (raw_g + g_bias[h])) per channel; beta = sigmoid(raw_beta). Where the slot is "
        "positive, S = S * exp(g)[None, :], S = S + beta * (v - S @ k)[:, None] * k[None, :], out = S @ q, and S "
        "is written back to the slot; a slot <= 0 (vLLM's null block and padding) gives a zero output and no slot "
        "is read or written for it. Slots no token names are left unchanged."
    ),
    op_type="kda_decode",
    axes={
        "n_tokens": axis("Tokens of the batch, one a sequence.", 1, 4096, is_var=True),
        # the kernel reaches its slots through state_indices: any cache size serves
        "n_slots": axis("Slots of the state cache.", 2, 65536, is_var=True, any_size=True),
        "n_heads": axis("KDA heads on this rank.", 1, 128),
        "head_dim": axis(
            "Width of every q, k, v and gate head, and the side of a head's square state.", 128, 128
        ),
        "qkv_heads": derived_axis("Heads of a packed q|k|v row: q's, then k's, then v's.", "3 * n_heads"),
        "state_rows": derived_axis("Rows of one slot: every head's state, v by v.", "n_heads * head_dim"),
    },
    constraints=("n_tokens < n_slots",),
    inputs={
        "qkv": TensorSpec(
            description=(
                "Each token's packed q|k|v row after the short convolution (q and k before the L2 norm): heads "
                "0 .. n_heads - 1 are q's, then k's, then v's."
            ),
            shape=("n_tokens", "qkv_heads", "head_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "raw_g": TensorSpec(
            description="The full-rank gate projection (f_b_proj's output), before g_bias and the activation.",
            shape=_HEADS,
            dtype=_SAME_AS_QKV,
            # drawn small, as kimi_k3_kda_prefill's: g_bias and a_log set each channel's decay
            distribution=LecunNormalDistribution(fan_in="head_dim"),
        ),
        "raw_beta": TensorSpec(
            description=(
                "The beta projection, one per token and head, before the sigmoid; a unit last dim, so that its rows "
                "may be apart by more than a row (a column slice of vLLM's fused projection)."
            ),
            shape=("n_tokens", "n_heads", 1),
            dtype=_SAME_AS_QKV,
            distribution=_FINITE,
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
        "state_cache": TensorSpec(
            description=(
                "The paged recurrent states: slot by slot, each head's (v, k) state, the head's head_dim rows of "
                "head_dim keys."
            ),
            shape=("n_slots", "state_rows", "head_dim"),
            dtype=_FP32,
            distribution=_FINITE,
            mutable=True,
            # one float32 step from float32 operands (bf16 inputs decode exactly): about 1e-6 at N(0, 1) states
            error_bounds=ToleranceBound(absolute_tolerance=1e-4, relative_tolerance=1e-4),
        ),
        "state_indices": TensorSpec(
            description=(
                "Each token's slot; the positive ones distinct (vLLM's padding tokens name its null block 0, or -1)."
            ),
            shape=("n_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            # drawn positive, so that every timed token steps a state: the zero output of a slot <= 0 is the
            # reference's and the unit tests'
            distribution=IntRangeDistribution(lo=1, hi="n_slots - 1", distinct=True),
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
    },
    outputs={
        "out": TensorSpec(
            description="Each token's S @ q after its update, zero where its slot is <= 0.",
            shape=_HEADS,
            dtype=_SAME_AS_QKV,
            # a float32 result rounded to bf16 (its roundoff is added to the bound): a rounding that flips is one
            # more half unit
            error_bounds=ToleranceBound(absolute_tolerance=1e-3, relative_tolerance=1e-2),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_storage_facts("row_major")),
            # vLLM's: the q|k|v rows and the gate rows as their kernels leave them, beta a column of the fused
            # projection, each slot one dense row of a page that also holds the convolution state
            "strided_rows": LayoutVariant(
                layouts={
                    "qkv": StridedRows(),
                    "raw_g": StridedRows(),
                    "raw_beta": StridedRows(),
                    "state_cache": StridedRows(),
                },
                detect_facts=_storage_facts("strided_rows"),
            ),
        }
    ),
    compare=CompareSpec(determinism=DeterminismLevel.BITWISE),
)


def _speculative(contract: OpContract) -> OpContract:
    """The decode contract over a window of tokens a sequence, from the state of its accepted token."""
    axes = {
        # the window, n_tokens / n_seqs, is no axis: no tensor has it as a dim, and the kernels walk any
        "n_seqs": axis("Sequences of the batch.", 1, 1024, is_var=True),
        "n_tokens": axis(
            "Tokens of every window, back to back: n_tokens / n_seqs a window.", 1, 16384, is_var=True
        ),
        **{name: a for name, a in contract.axes.items() if name != "n_tokens"},
    }
    inputs = {}
    for name, spec in contract.inputs.items():
        if name == "state_indices":
            spec = spec.model_copy(
                update={
                    "description": (
                        "Each token's slot, window by window (vLLM's [n_seqs, window] rows of its block table); the "
                        "positive ones distinct."
                    )
                }
            )
        inputs[name] = spec
        if name == "state_indices":
            inputs["num_accepted_tokens"] = TensorSpec(
                description=(
                    "Each sequence's tokens accepted at its last step, 1 .. its window: the window starts from the "
                    "state in the slot of its token num_accepted_tokens - 1."
                ),
                shape=("n_seqs",),
                dtype=DTypeSpec(any_of=(DType.INT32,)),
                distribution=IntRangeDistribution(lo=1, hi="n_tokens // n_seqs"),
            )
        if name == "state_cache":
            inputs[name] = TensorSpec(
                description=spec.description,
                shape=spec.shape,
                dtype=_FP32,
                distribution=_FINITE,
                mutable=True,
                # a window's float32 steps, each written out
                error_bounds=ToleranceBound(absolute_tolerance=2e-4, relative_tolerance=2e-4),
            )
    out = contract.outputs["out"].model_copy(
        update={
            "description": "Each token's S @ q after its update, zero in a window whose start slot is <= 0."
        }
    )
    fields = {name: getattr(contract, name) for name in type(contract).model_fields}
    fields.update(
        contract_id="kimi_k3_kda_decode_spec@1.0",
        description=(
            "Kimi-K3's KDA speculative decode, a window of w = n_tokens / n_seqs tokens a sequence: sequence s's "
            "window is tokens s * w .. s * w + w - 1 (the token after its accepted one, then its drafts); token t's "
            "q, k and v are heads h, n_heads + h and 2 * n_heads + h of qkv[t], and its slot is state_indices[t], "
            "whose head h holds the state S[v, k] = state_cache[slot, h * head_dim + v, k]. Per head, q and k are "
            "L2-normalized (x / sqrt(sum(x**2) + 1e-6)) and q multiplied by scale; the gate g = lower_bound * "
            "sigmoid(exp(a_log[h]) * (raw_g + g_bias[h])) per channel; beta = sigmoid(raw_beta). The window starts "
            "from the state of the start slot state_indices[s * w + num_accepted_tokens[s] - 1]; where that is "
            "positive, token by token S = S * exp(g)[None, :], S = S + beta * (v - S @ k)[:, None] * k[None, :], "
            "out = S @ q, and S is written to the token's slot where that is positive. Where the start slot is <= 0 "
            "(vLLM's padding) the window's outputs are zero and no slot is written. Slots no token writes are left "
            "unchanged."
        ),
        axes=axes,
        # windows of 1 to 16 tokens (DSPARK's 8: 7 drafts), the kernels' LDS
        constraints=("n_tokens % n_seqs == 0", "n_tokens <= 16 * n_seqs", "n_tokens < n_slots"),
        inputs=inputs,
        outputs={"out": out},
    )
    return OpContract(**fields)


KIMI_K3_KDA_DECODE_SPEC = _speculative(KIMI_K3_KDA_DECODE)

CONTRACTS = (KIMI_K3_KDA_DECODE, KIMI_K3_KDA_DECODE_SPEC)
