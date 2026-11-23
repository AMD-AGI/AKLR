# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA speculative chain verify on one rank in ReplaySSM ring mode: sglang's
``kda_verify_gfx950`` with ``cache_ring``, which ``srt/layers/attention/linear/kda_backend_hip.py``
calls under ``--enable-linear-replayssm-spec``. Each request's draft chain runs through the causal
convolution and the gated delta rule from its slot's state, which stays as it is; the chain's raw
values, raw keys, gates and betas go to the slot's ring and its convolution windows to the
request's line, from which sglang's commit folds the accepted prefix.

A model is a scenario: Kimi-K3.1-Pro at TP8 has 12 KDA heads a rank and verifies chains of 5
tokens."""

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
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
from aklr.plugins.sglang.layers.kda_backend_hip.baseline import KdaChainVerifyRingBaseline
from aklr.plugins.sglang.layers.kda_backend_hip.reference import HEAD_DIM, KdaChainVerifyRingReference

__all__ = ["CONTRACTS", "KDA_CHAIN_VERIFY_RING"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()
_EXACT = ToleranceBound(absolute_tolerance=0, relative_tolerance=0)
# The reference sums the four taps in FP32 and rounds each convolution output to BF16; a kernel
# that sums them in another order lands, next to a rounding midpoint, on the other neighbour, one
# ulp (up to 2**-7 of the value) away.
_CONVOLVED = ToleranceBound(absolute_tolerance=2**-16, relative_tolerance=2**-7 + 2**-7)
# FP32 gates: a + dt_bias rounds to FP32 before exp(a_log) scales it, and exp is a hardware
# approximation; an FP32 transcription of sglang's kernel stays within 1e-5 of the float64 gates.
_GATE = ToleranceBound(absolute_tolerance=2**-16, relative_tolerance=2**-16)


def _tensor(
    description, shape, dtype, *, mutable=False, bound=None, distribution=_FINITE, lifetime="per_call"
):
    return TensorSpec(
        description=description,
        shape=shape,
        dtype=dtype,
        distribution=distribution,
        mutable=mutable,
        error_bounds=() if bound is None else bound,
        lifetime=lifetime,
    )


def _ring(description, shape, dtype, bound):
    return _tensor(
        f"Each slot's ReplaySSM ring of {description}, a row per token: a live request's tokens "
        "write rows 0 on of its slot, the other rows stay.",
        shape,
        dtype,
        mutable=True,
        bound=bound,
    )


KDA_CHAIN_VERIFY_RING = OpContract(
    contract_id="kda_chain_verify_ring@1.0",
    op_type="kda_backend_hip",
    description=(
        "Kimi-K3's KDA chain verify in ReplaySSM ring mode. Request r verifies tokens r * T to "
        "r * T + T - 1 (T = n_verify_tokens) from its distinct slot s = cache_indices[r]; a "
        "request whose slot is -1 is padding: it changes no slot or line, and its output is "
        "zero. A live request's convolution input is conv_state[s]'s three rows, oldest first, "
        "then its tokens: token t's channel c is SiLU of the exact sum of the four inputs ending at "
        "it times conv_weight[c], oldest first, rounded to BF16, and conv_window[window_indices[r], "
        "t] takes the last three inputs, oldest first. Head h takes the token's query, key and "
        "value channels: q is L2-normalized (epsilon 1e-6 added to the sum of squares) and "
        "multiplied by scale, k is L2-normalized. The gate is g = lower_bound * "
        "sigmoid(exp(a_log[h]) * (a + dt_bias[h])), or without a lower bound -exp(a_log[h]) times "
        "softplus of a + dt_bias[h] at softplus_beta, linear where softplus_beta times its input "
        "exceeds softplus_threshold; beta = sigmoid(b). From S = ssm_state[s, h] ([value, key]), "
        "each token in turn: S = S * exp(g) along the key, S = S + ((v - S k) * beta) outer k, "
        "output = S q. Rows 0 to T - 1 of slot s's rings take each token's convolved value "
        "(ring_v), convolved key before its normalization (ring_k), g (ring_g) and beta "
        "(ring_beta). ssm_state, conv_state, the other ring rows, unindexed slots and lines and "
        "all other inputs are preserved."
    ),
    axes={
        "n_requests": axis(
            "Requests verified, sglang's graph padding requests included.", 1, 2048, is_var=True
        ),
        "n_verify_tokens": axis("Tokens a request verifies: its draft chain.", 2, 8),
        "n_tokens": axis("Tokens verified.", 2, 16384, is_var=True),
        "n_heads": axis("KDA heads on this rank, query, key and value alike.", 1, 128),
        "conv_dim": derived_axis(
            "Convolution channels: every query head's, then every key head's, then every value head's.",
            f"3 * n_heads * {HEAD_DIM}",
        ),
        "n_slots": axis("Allocated state slots.", 1, 16384, is_var=True, any_size=True),
        "n_lines": axis("Allocated convolution window lines.", 1, 16384, is_var=True, any_size=True),
        "ring_len": axis("Rows of a slot's ring per head.", 2, 256, is_var=True, any_size=True),
    },
    attrs={
        "has_lower_bound": attr(
            "Whether the gate is lower-bounded (Kimi-K3's safe gate) rather than softplus.",
            True,
            False,
            default=True,
            abbrev="lower_bound",
        ),
    },
    constraints=(
        "n_tokens == n_requests * n_verify_tokens",
        "ring_len >= n_verify_tokens",
        # drawn slots are distinct, -1 among them at most once
        "n_requests <= n_slots + 1",
        "n_requests <= n_lines",
    ),
    inputs={
        "mixed_qkv": _tensor(
            "Each token's convolution input: every query head's channels, then every key head's, "
            "then every value head's; a request's tokens are consecutive rows, in chain order.",
            ("n_tokens", "conv_dim"),
            _BF16,
        ),
        "conv_weight": _tensor(
            "Each channel's four causal taps, the oldest input's first.",
            ("conv_dim", 4),
            _FP32,
            lifetime="weight",
        ),
        "conv_state": _tensor(
            "Each slot's last three convolution inputs, oldest first.",
            ("n_slots", 3, "conv_dim"),
            _BF16,
        ),
        "cache_indices": _tensor(
            "Each request's distinct state slot, or -1 for a padding request.",
            ("n_requests",),
            _INT32,
            distribution=IntRangeDistribution(lo=-1, hi="n_slots - 1", distinct=True),
        ),
        "conv_window": _tensor(
            "Convolution windows: a live request's line holds, after each of its tokens, the last "
            "three convolution inputs, oldest first.",
            ("n_lines", "n_verify_tokens", 3, "conv_dim"),
            _BF16,
            mutable=True,
            bound=_EXACT,
        ),
        "window_indices": _tensor(
            "Each request's distinct window line.",
            ("n_requests",),
            _INT32,
            distribution=IntRangeDistribution(lo=0, hi="n_lines - 1", distinct=True),
        ),
        "a": _tensor("The gate's input, one per key channel.", ("n_tokens", "n_heads", HEAD_DIM), _BF16),
        "b": _tensor("Beta's input.", ("n_tokens", "n_heads"), _BF16),
        "a_log": _tensor("Log of each head's gate scale.", ("n_heads",), _FP32, lifetime="weight"),
        "dt_bias": _tensor(
            "The gate's bias, one per key channel.", ("n_heads", HEAD_DIM), _FP32, lifetime="weight"
        ),
        "ssm_state": _tensor(
            "Each slot's recurrent state at its last committed token, value dimension first.",
            ("n_slots", "n_heads", HEAD_DIM, HEAD_DIM),
            _FP32,
        ),
        "ring_v": _ring("convolved values", ("n_slots", "n_heads", "ring_len", HEAD_DIM), _BF16, _CONVOLVED),
        "ring_k": _ring(
            "convolved keys before their normalization",
            ("n_slots", "n_heads", "ring_len", HEAD_DIM),
            _BF16,
            _CONVOLVED,
        ),
        "ring_g": _ring("gates (log decays)", ("n_slots", "n_heads", "ring_len", HEAD_DIM), _FP32, _GATE),
        "ring_beta": _ring("betas", ("n_slots", "n_heads", "ring_len"), _FP32, _GATE),
        "scale": ScalarSpec(
            description="Query scale.",
            dtype=DType.FLOAT32,
            interval=(0.0883883461356163, HEAD_DIM**-0.5),
            default=HEAD_DIM**-0.5,
        ),
        "lower_bound": ScalarSpec(
            description="The gate's lower bound (Kimi-K3's checkpoints: gate_lower_bound).",
            dtype=DType.FLOAT32,
            values=(-5.0,),
            present_when="has_lower_bound",
        ),
        "softplus_beta": ScalarSpec(
            description="The softplus gate's beta.",
            dtype=DType.FLOAT32,
            values=(1.0,),
            present_when="not has_lower_bound",
        ),
        "softplus_threshold": ScalarSpec(
            description="Where the softplus gate turns linear.",
            dtype=DType.FLOAT32,
            values=(20.0,),
            present_when="not has_lower_bound",
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Each token's attention output, zero for a padding request's tokens.",
            shape=("n_tokens", "n_heads", HEAD_DIM),
            dtype=_BF16,
            # a convolved query, key or value channel that a correct kernel rounds to the other
            # BF16 neighbour (see _CONVOLVED) moves a head's outputs by up to ~2e-4 at N(0, 1)
            # inputs, whatever their size; FP32 recurrences stay within 1e-7
            error_bounds=ToleranceBound(absolute_tolerance=2**-12, relative_tolerance=2**-10 + 2**-7),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
    reference=KdaChainVerifyRingReference(),
    baseline=KdaChainVerifyRingBaseline(),
)

CONTRACTS = (KDA_CHAIN_VERIFY_RING,)
