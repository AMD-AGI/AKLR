# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA prefill (the chunked gated delta rule) as vLLM runs it on ROCm
(``models/kimi_k3/amd/ops/kda_prefill.py``'s ``chunk_kda_prefill``, as ``kda.py`` calls it).

The layer hands over q, k and v after the short convolution, the full-rank gate projection
and the beta projection before their activations; vLLM's fused backend normalizes q and k,
activates the gate and beta itself, and reads and writes each sequence's recurrent state in
the paged state cache. vLLM keeps that cache in pages shared with the convolution state, so
its rows are apart by more than a row (``strided_rows``)."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    CuSeqlensDistribution,
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

__all__ = [
    "CHECKPOINT_CHUNK",
    "CONTRACTS",
    "KIMI_K3_KDA_PREFILL",
    "KIMI_K3_KDA_PREFILL_CHECKPOINT",
    "L2_NORM_EPS",
]

# x / sqrt(sum(x**2) + eps): vLLM's Triton prefill path and recurrent kernel, and aiter's KDA
L2_NORM_EPS = 1e-6
# tokens a checkpoint chunk counts: vLLM's KDA_CHECKPOINT_ALIGNMENT, its fused kernels' chunk
CHECKPOINT_CHUNK = 64

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_SAME_AS_Q = DTypeSpec(same_as_tensor="q")
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FINITE = FiniteDistribution()
_ROWS = ("n_tokens", "n_heads", "head_dim")


def _state_storage_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="state_storage", expected_value=value),)


KIMI_K3_KDA_PREFILL = OpContract(
    contract_id="kimi_k3_kda_prefill@1.0",
    description=(
        "Kimi-K3's KDA prefill (chunked gated delta rule), every sequence's tokens back to back: "
        "sequence s owns rows cu_seqlens[s]:cu_seqlens[s + 1] and slot state_indices[s] of state_cache, "
        "whose head h holds the state S[v, k] = state_cache[slot, h * head_dim + v, k]. Per head, "
        "q and k are L2-normalized (x / sqrt(sum(x**2) + 1e-6)) and q multiplied by scale; the "
        "gate g = lower_bound * sigmoid(exp(a_log[h]) * (raw_g + g_bias[h])) per channel; "
        "beta = sigmoid(raw_beta). A sequence starts from its slot's state when has_initial_state[s], "
        "else from zero (the slot then holds stale values no kernel reads); token by token, "
        "S = S * exp(g)[None, :], S = S + beta * (v - S @ k)[:, None] * k[None, :], out = S @ q; its "
        "last S is written back to its slot. Slots no sequence names are left unchanged. vLLM's "
        "snapshots of a state mid-prefill (its checkpoint_offsets, under prefix caching) are not part of it."
    ),
    op_type="kda_prefill",
    axes={
        "n_tokens": axis("Tokens of every sequence, back to back.", 1, 131072, is_var=True),
        "n_seqs": axis("Sequences.", 1, 1024, is_var=True),
        "n_offsets": axis("Sequence offsets: sequences + 1.", 2, 1025, is_var=True),
        # the kernel reaches its slots through state_indices: any cache size serves
        "n_slots": axis("Slots of the state cache.", 1, 65536, is_var=True, any_size=True),
        "n_heads": axis("KDA heads on this rank.", 1, 128),
        "head_dim": axis(
            "Width of every q, k, v and gate head, and the side of a head's square state.", 128, 128
        ),
        "state_rows": derived_axis("Rows of one slot: every head's state, v by v.", "n_heads * head_dim"),
    },
    constraints=(
        "n_offsets == n_seqs + 1",
        "n_seqs <= n_tokens",
        "n_seqs <= n_slots",
    ),
    inputs={
        "q": TensorSpec(
            description="Queries after the short convolution, before the L2 norm.",
            shape=_ROWS,
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "k": TensorSpec(
            description="Keys after the short convolution, before the L2 norm.",
            shape=_ROWS,
            dtype=_SAME_AS_Q,
            distribution=_FINITE,
        ),
        "v": TensorSpec(
            description="Values after the short convolution.",
            shape=_ROWS,
            dtype=_SAME_AS_Q,
            distribution=_FINITE,
        ),
        "raw_g": TensorSpec(
            description="The full-rank gate projection (f_b_proj's output), before g_bias and the activation.",
            shape=_ROWS,
            dtype=_SAME_AS_Q,
            # drawn small, so that g_bias and a_log set each channel's decay and some channels keep
            # their state across thousands of tokens, as trained ones do: N(0, 1) draws here leave
            # no channel a state older than a few tokens, and the passing of states from chunk to
            # chunk untested
            distribution=LecunNormalDistribution(fan_in="head_dim"),
        ),
        "raw_beta": TensorSpec(
            description="The beta projection, one per token and head, before the sigmoid.",
            shape=("n_tokens", "n_heads"),
            dtype=_SAME_AS_Q,
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
                "The paged recurrent states: slot by slot, each head's (v, k) state, the head's "
                "head_dim rows of head_dim keys."
            ),
            shape=("n_slots", "state_rows", "head_dim"),
            dtype=_FP32,
            distribution=_FINITE,
            mutable=True,
            # a correct kernel multiplies bf16 operands into the fp32 state, and a segmented one
            # composes bf16 segment maps: its entries (up to about 5 at N(0, 1) inputs) end up to
            # about 1e-2 off
            error_bounds=ToleranceBound(absolute_tolerance=2e-2, relative_tolerance=1e-2),
        ),
        "state_indices": TensorSpec(
            description="Each sequence's slot, distinct.",
            shape=("n_seqs",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        # a device tensor in vLLM: no Point tells fresh sequences from resumed ones
        "has_initial_state": TensorSpec(
            description="Whether each sequence resumes from its slot's state, else from zero.",
            shape=("n_seqs",),
            dtype=DTypeSpec(any_of=(DType.BOOL,)),
            distribution=_FINITE,
        ),
        "cu_seqlens": TensorSpec(
            description="Where each sequence's tokens start, then where the last ends.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_tokens"),
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
            description="Each token's S @ q after its update.",
            shape=_ROWS,
            dtype=_SAME_AS_Q,
            # the outputs are small (|q| = 1 times scale: about 1e-2 at N(0, 1) inputs), and bf16
            # operands into the fp32 state leave a correct kernel about 1e-3 off at most
            error_bounds=ToleranceBound(absolute_tolerance=2e-3, relative_tolerance=2e-2),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_state_storage_facts("row_major")),
            # vLLM's: each slot one dense row of a page that also holds the convolution state
            "strided_rows": LayoutVariant(
                layouts={"state_cache": StridedRows()},
                detect_facts=_state_storage_facts("strided_rows"),
            ),
        }
    ),
    compare=CompareSpec(determinism=DeterminismLevel.BITWISE),
)

_CHECKPOINT_INPUTS = {
    "checkpoint_chunks": TensorSpec(
        description=(
            "Each sequence's snapshot point in chunks of 64 tokens; 0, or one at or past the sequence's end, asks "
            "for none."
        ),
        shape=("n_seqs",),
        dtype=DTypeSpec(any_of=(DType.INT32,)),
        distribution=IntRangeDistribution(lo=0, hi=f"n_tokens // {CHECKPOINT_CHUNK}"),
    ),
    "checkpoint_state": TensorSpec(
        description="Each sequence's snapshot: its heads' (v, k) states, as a slot of state_cache holds them.",
        shape=("n_seqs", "state_rows", "head_dim"),
        dtype=_FP32,
        distribution=_FINITE,
        mutable=True,
        error_bounds=ToleranceBound(absolute_tolerance=2e-2, relative_tolerance=1e-2),
    ),
}


def _with_checkpoint(contract: OpContract) -> OpContract:
    """The contract with vLLM's mid-prefill snapshot: its inputs, the snapshot's two after cu_seqlens."""
    inputs = {}
    for name, spec in contract.inputs.items():
        inputs[name] = spec
        if name == "cu_seqlens":
            inputs.update(_CHECKPOINT_INPUTS)
    fields = {name: getattr(contract, name) for name in type(contract).model_fields}
    fields.update(
        contract_id="kimi_k3_kda_prefill_checkpoint@1.0",
        description=contract.description.removesuffix(
            " vLLM's snapshots of a state mid-prefill (its checkpoint_offsets, under prefix caching) are not part "
            "of it."
        )
        + " Besides, as vLLM snapshots a state mid-prefill under prefix caching (its checkpoint_offsets, here in "
        "chunks of 64 tokens): where 0 < 64 * checkpoint_chunks[s] < sequence s's length, its state after its first "
        "64 * checkpoint_chunks[s] tokens is written to checkpoint_state[s]; else checkpoint_state[s] is left as it "
        "was.",
        inputs=inputs,
    )
    return OpContract(**fields)


KIMI_K3_KDA_PREFILL_CHECKPOINT = _with_checkpoint(KIMI_K3_KDA_PREFILL)

CONTRACTS = (KIMI_K3_KDA_PREFILL, KIMI_K3_KDA_PREFILL_CHECKPOINT)
