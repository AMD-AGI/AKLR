# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA (Kimi Delta Attention) prefill as sglang runs it on its Triton backend
(``srt/layers/attention/linear/kernels/kda_triton.py``'s ``TritonKDAKernel``): ``extend`` calls fla's
``chunk_kda``.

The KDA backend hands over q, k and v as views of the short convolution's output rows (one packed
q|k|v row a token), the full-rank gate projection and the beta projection before their activations;
the kernels normalize q and k, activate the gate (K3's safe gate, with its lower bound) and beta, and
read and write each sequence's recurrent state in its slot of sglang's state pool. sglang hands beta
over as a column slice of the [f_a | b] projection, so its rows are apart by more than a row
(``strided_beta``). A slot of -1 (sglang's padding sentinel) is read as a zero state and written
nowhere.

chunk_kda is what sglang 0.5.22 runs for extend. sglang 0.5.23 on gfx950 runs its own HIP prefill
(kda_gfx950_prefill) instead, on by default: there the route leaves every call to sglang, and serves only with that
prefill turned off (SGLANG_ROCM_KDA_HIP_PREFILL=0)."""

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    CuSeqlensDistribution,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
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
from aklr.plugins.sglang.layers.kda_backend_hip.contract import KDA_CHAIN_VERIFY_RING
from aklr.plugins.sglang.layers.kda_triton.baseline import (
    KimiK3SglKdaPrefillBaseline,
    KimiK3SglKdaPrefillTrackBaseline,
    KimiK3SglKdaVerifyBaseline,
)
from aklr.plugins.sglang.layers.kda_triton.reference import (
    TRACK_CHUNK,
    KimiK3SglKdaPrefillReference,
    KimiK3SglKdaPrefillTrackReference,
    KimiK3SglKdaVerifyReference,
)

__all__ = [
    "CONTRACTS",
    "KIMI_K3_SGL_KDA_PREFILL",
    "KIMI_K3_SGL_KDA_PREFILL_TRACK",
    "KIMI_K3_SGL_KDA_VERIFY",
]

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
    """The token operands and the layer's gate weights, as the KDA backend hands them over."""
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
            # any finite log-scale, a scale of 1 or less too; drawn N(0, 1): scales of about 0.05 to 20, the [1, 16]
            # KDA initializes them in among them
            distribution=_FINITE,
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


# sglang's chunk_kda, the engine's prefill, rounds the normalized q and k to bf16 and multiplies them and the state
# chunk by chunk on bf16 operands (float32 sums), as any chunked kernel on bf16 tensor cores does: its states move off
# the float32 recurrence by bf16 roundings (2**-8 of a product), past float32's common 1e-4; entries reach about 5 at
# N(0, 1) inputs, and a kernel computing so ends up to about 1e-2 off
_CHUNKED_STATE = ToleranceBound(absolute_tolerance=2e-2, relative_tolerance=1e-2)
_SLOTS_DRAWN = (
    # drawn >= 0 and distinct, so that every timed sequence steps a state: no distribution draws -1 any number of
    # times beside distinct slots (IntRangeDistribution's distinct draws hold -1 once at most);
    # tests/plugins/sglang/test_kda_triton_kernels.py runs the kernels on slots padded so (gfx950). A call may have
    # more sequences than slots (its -1 ones); distinct draws need n_seqs <= n_slots, which every workload keeps
    IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True)
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
    ),
    inputs={
        **_common_inputs(),
        "state_cache": TensorSpec(
            description=(
                "sglang's state pool of the layer (temporal, [slots, heads, v, k] there): slot by slot, each head's "
                "(v, k) state, the head's head_dim rows of head_dim keys."
            ),
            shape=("n_slots", "state_rows", "head_dim"),
            dtype=_FP32,
            distribution=_FINITE,
            mutable=True,
            error_bounds=_CHUNKED_STATE,
        ),
        "state_indices": TensorSpec(
            description=(
                "Each sequence's slot, the ones >= 0 distinct; -1 (sglang's padding), any number of times, reads a "
                "zero state and writes none."
            ),
            shape=("n_seqs",),
            dtype=_INT32,
            distribution=_SLOTS_DRAWN,
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
            dtype=_SAME_AS_QKV.model_copy(update={"param_name": "out_dtype"}),
            # the outputs are small (|q| = 1 times scale: about 1e-2 at N(0, 1) inputs), and bf16 operands into the
            # fp32 state leave a correct kernel about 1e-3 off at most; plus the reference's own rounding to bf16
            error_bounds=ToleranceBound(absolute_tolerance=2e-3, relative_tolerance=2e-2 + 2**-7),
        ),
    },
    layout_variants=_LAYOUTS,
    compare=CompareSpec(determinism="bitwise"),
    reference=KimiK3SglKdaPrefillReference(),
    baseline=KimiK3SglKdaPrefillBaseline(),
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
        error_bounds=_CHUNKED_STATE,
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
        reference=KimiK3SglKdaPrefillTrackReference(),
        baseline=KimiK3SglKdaPrefillTrackBaseline(),
    )
    return OpContract(**fields)


KIMI_K3_SGL_KDA_PREFILL_TRACK = _with_track(KIMI_K3_SGL_KDA_PREFILL)


def _verify(ring: OpContract) -> OpContract:
    """sglang 0.5.23's chain verify in ReplaySSM ring mode: kda_chain_verify_ring's computation (the
    contract of sglang 0.5.24's kda_verify_gfx950) on the operands KDAAttnBackend._forward_target_verify
    holds, the convolution input and beta as the rows of sglang's wider projections, against what 0.5.23
    runs for it on gfx950 (the convolution, then the Triton verify kernel)."""
    axes = {}
    for name, spec in ring.axes.items():
        if name == "conv_dim":
            axes["conv_heads"] = derived_axis(
                "Convolution heads: every query head, then every key head, then every value head.",
                "3 * n_heads",
            )
        axes[name] = spec
    inputs = dict(ring.inputs)
    inputs["mixed_qkv"] = ring.inputs["mixed_qkv"].model_copy(
        update={
            "description": (
                "Each token's convolution input, a head a row of 128 channels: every query head, then every key "
                "head, then every value head; a request's tokens are consecutive, in chain order. In sglang a "
                "column slice of its fused in-projection: a token's rows are dense, the tokens apart by more."
            ),
            "shape": ("n_tokens", "conv_heads", 128),
        }
    )
    inputs["b"] = ring.inputs["b"].model_copy(
        update={
            "description": (
                "Beta's input; a unit last dim, so that its rows may be apart by more than a row (a column "
                "slice of sglang's [f_a | b] projection)."
            ),
            "shape": ("n_tokens", "n_heads", 1),
        }
    )
    opening = "Kimi-K3's KDA chain verify in ReplaySSM ring mode."
    # sglang's causal_conv1d_update, the baseline, and the reference sum the taps in FP32, which the ring
    # contract's words do not say
    exact = "SiLU of the exact sum of the four inputs ending at it times conv_weight[c], oldest first,"
    in_fp32 = (
        "SiLU of the sum of the four inputs ending at it times conv_weight[c], each product and partial "
        "sum rounded to FP32, oldest first (as sglang's causal_conv1d_update sums them),"
    )
    if not ring.description.startswith(opening) or ring.description.count(exact) != 1:
        raise ValueError("kda_chain_verify_ring's description reads otherwise")
    description = (
        opening.removesuffix(".")
        + ", as sglang 0.5.23's KDA backend runs a target verify of dense chains on its Triton verify "
        "kernel. Token t's convolution input is mixed_qkv[t]'s heads end to end, its beta input b[t, :, 0]."
        + ring.description.removeprefix(opening).replace(exact, in_fp32)
    )
    fields = {name: getattr(ring, name) for name in type(ring).model_fields}
    fields.update(
        contract_id="kimi_k3_sgl_kda_verify@1.0",
        op_type="kda_triton",
        description=description,
        axes=axes,
        inputs=inputs,
        layout_variants=LayoutVariantsSpec(
            by_name={
                "row_major": LayoutVariant(detect_facts=_storage_facts("row_major")),
                # sglang's: the convolution input and beta as rows of its fused projections' outputs
                "strided_rows": LayoutVariant(
                    layouts={"mixed_qkv": StridedRows(), "b": StridedRows()},
                    detect_facts=_storage_facts("strided_rows"),
                ),
            }
        ),
        reference=KimiK3SglKdaVerifyReference(),
        baseline=KimiK3SglKdaVerifyBaseline(),
    )
    return OpContract(**fields)


KIMI_K3_SGL_KDA_VERIFY = _verify(KDA_CHAIN_VERIFY_RING)

CONTRACTS = (KIMI_K3_SGL_KDA_PREFILL, KIMI_K3_SGL_KDA_PREFILL_TRACK, KIMI_K3_SGL_KDA_VERIFY)
