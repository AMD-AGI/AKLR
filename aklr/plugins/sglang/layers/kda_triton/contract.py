# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA (Kimi Delta Attention) prefill and decode as sglang runs them on its Triton backend
(``srt/layers/attention/linear/kernels/kda_triton.py``'s ``TritonKDAKernel``): ``extend`` calls fla's
``chunk_kda``, ``decode`` calls ``fused_sigmoid_gating_delta_rule_update``.

The KDA backend hands over q, k and v as views of the short convolution's output rows (one packed
q|k|v row a token), the full-rank gate projection and the beta projection before their activations;
the kernels normalize q and k, activate the gate (K3's safe gate, with its lower bound) and beta, and
read and write each sequence's recurrent state in its slot of sglang's state pool. sglang hands beta
over as a column slice of the [f_a | b] projection, so its rows are apart by more than a row
(``strided_beta``). A slot of -1 (sglang's padding sentinel) is read as a zero state and written
nowhere."""

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
    "CONTRACTS",
    "KIMI_K3_SGL_KDA_DECODE",
    "KIMI_K3_SGL_KDA_PREFILL",
    "KIMI_K3_SGL_KDA_PREFILL_TRACK",
    "L2_NORM_EPS",
    "TRACK_CHUNK",
]

# x / sqrt(sum(x**2) + eps): fla's l2norm_fwd and the recurrent kernel's in-kernel norm
L2_NORM_EPS = 1e-6
# tokens of one chunk of chunk_kda, the unit of sglang's prefix-cache snapshots (track_chunk_idx)
TRACK_CHUNK = 64

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_SAME_AS_QKV = DTypeSpec(same_as_tensor="qkv")
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()
_HEADS = ("n_tokens", "n_heads", "head_dim")

_STEP = (
    "token t's q, k and v are heads h, n_heads + h and 2 * n_heads + h of qkv[t]. Per head, q and k are L2-normalized "
    "(x / sqrt(sum(x**2) + 1e-6)) and q multiplied by scale; the gate g = lower_bound * sigmoid(exp(a_log[h]) * "
    "(raw_g + g_bias[h])) per key channel; beta = sigmoid(raw_beta). A step of the state S[v, k] (head h of a slot "
    "is S = state_cache[slot, h * head_dim + v, k]): S = S * exp(g)[None, :], S = S + beta * (v - S @ k)[:, None] * "
    "k[None, :], out = S @ q."
)


def _storage_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="storage", expected_value=value),)


_LAYOUTS = LayoutVariantsSpec(
    by_name={
        "row_major": LayoutVariant(detect_facts=_storage_facts("row_major")),
        # sglang's: beta a column slice of the [f_a | b] projection (rows 144 apart at TP8)
        "strided_beta": LayoutVariant(
            layouts={"raw_beta": StridedRows()},
            detect_facts=_storage_facts("strided_beta"),
        ),
    }
)


def _common_inputs() -> dict[str, TensorSpec]:
    """The token operands and the layer's gate weights, as both calls hand them over."""
    return {
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
            # drawn small, so that g_bias and a_log set each channel's decay and some channels keep their state
            # across thousands of tokens, as trained ones do: N(0, 1) draws leave no channel a state older than a
            # few tokens, and the passing of states from chunk to chunk untested
            distribution=LecunNormalDistribution(fan_in="head_dim"),
        ),
        "raw_beta": TensorSpec(
            description=(
                "The beta projection, one per token and head, before the sigmoid; a unit last dim, so that its rows "
                "may be apart by more than a row (a column slice of sglang's [f_a | b] projection)."
            ),
            shape=("n_tokens", "n_heads", 1),
            dtype=_SAME_AS_QKV,
            distribution=_FINITE,
        ),
        "a_log": TensorSpec(
            description="Log of each head's gate scale (sglang's A_log, [1, 1, n_heads, 1] there).",
            shape=("n_heads",),
            dtype=_FP32,
            # log(U[1, 16]), as KDA initializes it: exp(a_log) in [1.3, 16]
            distribution=LogUniformDistribution(log2_lo=-2, log2_hi=1.47),
            lifetime="weight",
        ),
        "g_bias": TensorSpec(
            description="The gate's per-channel bias (sglang's dt_bias, flat (n_heads * head_dim) there).",
            shape=("n_heads", "head_dim"),
            dtype=_FP32,
            distribution=_FINITE,
            lifetime="weight",
        ),
    }


def _scalars() -> dict[str, ScalarSpec]:
    return {
        "scale": ScalarSpec(
            description="Multiplies the normalized queries; sglang's kernels take head_dim ** -0.5.",
            dtype=DType.FLOAT32,
            interval=(2.0**-126, 1.0),
            default=128**-0.5,
        ),
        "lower_bound": ScalarSpec(
            description="The gate's floor per token (Kimi-K3's gate_lower_bound, -5).",
            dtype=DType.FLOAT32,
            interval=(-5.0, -(2.0**-126)),
            default=-5.0,
        ),
    }


def _state_cache(description: str, error_bounds: ToleranceBound) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=("n_slots", "state_rows", "head_dim"),
        dtype=_FP32,
        distribution=_FINITE,
        mutable=True,
        error_bounds=error_bounds,
    )


_STATE_ROWS = derived_axis("Rows of one slot: every head's state, v by v.", "n_heads * head_dim")
_QKV_HEADS = derived_axis("Heads of a packed q|k|v row: q's, then k's, then v's.", "3 * n_heads")
_HEAD_AXES = {
    "n_heads": axis("KDA heads on this rank.", 1, 128),
    "head_dim": axis(
        "Width of every q, k, v and gate head, and the side of a head's square state.", 128, 128
    ),
    "qkv_heads": _QKV_HEADS,
    "state_rows": _STATE_ROWS,
}

KIMI_K3_SGL_KDA_PREFILL = OpContract(
    contract_id="kimi_k3_sgl_kda_prefill@1.0",
    description=(
        "Kimi-K3's KDA prefill (chunked gated delta rule) as sglang's chunk_kda computes it, every sequence's "
        "tokens back to back: sequence s owns tokens cu_seqlens[s]:cu_seqlens[s + 1] and slot state_indices[s] of "
        f"state_cache. {_STEP} Token by token from the slot's state (a zero state where the slot is -1), the "
        "sequence's last S is written back to its slot (nowhere where it is -1). Slots no sequence names are left "
        "unchanged. sglang's snapshots of a state mid-prefill (its track_state, under the mamba radix cache's "
        "extra buffer) are not part of it."
    ),
    op_type="kda_triton",
    axes={
        "n_tokens": axis("Tokens of every sequence, back to back.", 1, 131072, is_var=True),
        "n_seqs": axis("Sequences.", 1, 1024, is_var=True),
        "n_offsets": axis("Sequence offsets: sequences + 1.", 2, 1025, is_var=True),
        # the kernel reaches its slots through state_indices: any pool size serves
        "n_slots": axis("Slots of the state pool.", 1, 65536, is_var=True, any_size=True),
        **_HEAD_AXES,
    },
    constraints=(
        "n_offsets == n_seqs + 1",
        "n_seqs <= n_tokens",
        "n_seqs <= n_slots",
    ),
    inputs={
        **_common_inputs(),
        "state_cache": _state_cache(
            "sglang's state pool of the layer (temporal, [slots, heads, v, k] there): slot by slot, each head's "
            "(v, k) state, the head's head_dim rows of head_dim keys.",
            # a correct kernel multiplies bf16 operands into the fp32 state, and a segmented one composes bf16
            # segment maps: its entries (up to about 5 at N(0, 1) inputs) end up to about 1e-2 off
            ToleranceBound(absolute_tolerance=2e-2, relative_tolerance=1e-2),
        ),
        "state_indices": TensorSpec(
            description="Each sequence's slot, distinct; -1 (sglang's padding) reads a zero state and writes none.",
            shape=("n_seqs",),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        "cu_seqlens": TensorSpec(
            description="Where each sequence's tokens start, then where the last ends (sglang's query_start_loc).",
            shape=("n_offsets",),
            dtype=_INT32,
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_tokens"),
        ),
        **_scalars(),
    },
    outputs={
        "out": TensorSpec(
            description="Each token's S @ q after its update.",
            shape=_HEADS,
            dtype=_SAME_AS_QKV,
            # the outputs are small (|q| = 1 times scale: about 1e-2 at N(0, 1) inputs), and bf16 operands into the
            # fp32 state leave a correct kernel about 1e-3 off at most
            error_bounds=ToleranceBound(absolute_tolerance=2e-3, relative_tolerance=2e-2),
        ),
    },
    layout_variants=_LAYOUTS,
    compare=CompareSpec(determinism=DeterminismLevel.BITWISE),
)

_TRACK_INPUTS = {
    "track_chunk_idx": TensorSpec(
        description=(
            f"Each sequence's snapshot point in chunks of {TRACK_CHUNK} tokens of its own (sglang's track_chunk_idx); "
            "-1, or one at or past the sequence's end, asks for none."
        ),
        shape=("n_seqs",),
        dtype=_INT32,
        distribution=IntRangeDistribution(lo=-1, hi=f"n_tokens // {TRACK_CHUNK}"),
    ),
    "track_state": TensorSpec(
        description="Each sequence's snapshot, in float32: its heads' (v, k) states, as a slot of state_cache holds them.",
        shape=("n_seqs", "state_rows", "head_dim"),
        dtype=_FP32,
        distribution=_FINITE,
        mutable=True,
        error_bounds=ToleranceBound(absolute_tolerance=2e-2, relative_tolerance=1e-2),
    ),
}


def _with_track(contract: OpContract) -> OpContract:
    """The prefill contract with sglang's prefix-cache snapshot: its inputs, the snapshot's two after cu_seqlens."""
    inputs = {}
    for name, spec in contract.inputs.items():
        inputs[name] = spec
        if name == "cu_seqlens":
            inputs.update(_TRACK_INPUTS)
    fields = {name: getattr(contract, name) for name in type(contract).model_fields}
    fields.update(
        contract_id="kimi_k3_sgl_kda_prefill_track@1.0",
        description=contract.description.removesuffix(
            " sglang's snapshots of a state mid-prefill (its track_state, under the mamba radix cache's extra "
            "buffer) are not part of it."
        )
        + f" Besides, as sglang snapshots a state for its prefix cache: where 0 <= c = track_chunk_idx[s] and "
        f"{TRACK_CHUNK} * c < sequence s's length, its state after its first {TRACK_CHUNK} * c tokens (the state it "
        "starts from where c = 0) is written to track_state[s]; else track_state[s] is left as it was.",
        inputs=inputs,
    )
    return OpContract(**fields)


KIMI_K3_SGL_KDA_PREFILL_TRACK = _with_track(KIMI_K3_SGL_KDA_PREFILL)

KIMI_K3_SGL_KDA_DECODE = OpContract(
    contract_id="kimi_k3_sgl_kda_decode@1.0",
    description=(
        "Kimi-K3's KDA decode as sglang's fused_sigmoid_gating_delta_rule_update computes it, one step of the gated "
        f"delta rule per token (each sequence decodes one): token t's slot is state_indices[t]. {_STEP} The step "
        "starts from the slot's state and writes the new S back to it; where the slot is -1 (sglang's padded graph "
        "rows) it starts from a zero state, still gives its out, and writes nowhere. Slots no token names are left "
        "unchanged."
    ),
    op_type="kda_triton",
    axes={
        "n_tokens": axis("Tokens of the batch, one a sequence.", 1, 4096, is_var=True),
        # the kernel reaches its slots through state_indices: any pool size serves
        "n_slots": axis("Slots of the state pool.", 1, 65536, is_var=True, any_size=True),
        **_HEAD_AXES,
    },
    constraints=("n_tokens <= n_slots",),
    inputs={
        **_common_inputs(),
        "state_cache": _state_cache(
            "sglang's state pool of the layer (temporal, [slots, heads, v, k] there): slot by slot, each head's "
            "(v, k) state, the head's head_dim rows of head_dim keys.",
            # one float32 step from float32 operands (bf16 inputs decode exactly): about 1e-6 at N(0, 1) states
            ToleranceBound(absolute_tolerance=1e-4, relative_tolerance=1e-4),
        ),
        "state_indices": TensorSpec(
            description=(
                "Each token's slot, the ones >= 0 distinct; -1 (sglang's padded graph rows) reads a zero state and "
                "writes none."
            ),
            shape=("n_tokens",),
            dtype=_INT32,
            # drawn >= 0, so that every timed token steps a state: the -1 rows are the reference's and the unit
            # tests'
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        **_scalars(),
    },
    outputs={
        "out": TensorSpec(
            description="Each token's S @ q after its update.",
            shape=_HEADS,
            dtype=_SAME_AS_QKV,
            # a float32 result rounded to bf16 (its roundoff is added to the bound): a rounding that flips is one
            # more half unit
            error_bounds=ToleranceBound(absolute_tolerance=1e-3, relative_tolerance=1e-2),
        ),
    },
    layout_variants=_LAYOUTS,
    compare=CompareSpec(determinism=DeterminismLevel.BITWISE),
)

CONTRACTS = (KIMI_K3_SGL_KDA_PREFILL, KIMI_K3_SGL_KDA_PREFILL_TRACK, KIMI_K3_SGL_KDA_DECODE)
