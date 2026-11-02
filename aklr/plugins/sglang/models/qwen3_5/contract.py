# Created by OpenAI
"""GDN convolution, recurrent state update, gated RMSNorm and grouped FP8 output."""

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

__all__ = ["CONTRACTS", "GDN_DECODE", "GDN_DECODE_GROUP_FP8_QUANT", "GDN_PREFILL_GROUP_FP8_QUANT"]

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


GDN_PREFILL_GROUP_FP8_QUANT = OpContract(
    contract_id="gdn_prefill_group_fp8_quant@1.0",
    op_type="qwen3_5",
    description=(
        "Qwen3.5 GDN prefill from contiguous Q/K/V/Z and B/A projections. Each sequence "
        "uses its distinct cache_indices slot; fresh sequences ignore the slot's old values. "
        "Causal width-four SiLU convolution updates conv_state, Q and K are L2-normalized "
        "and rounded to BF16, and the gated delta recurrence updates FP32 delta_state "
        "in [value,key] order. BF16 attention output is gated RMS-normalized, rounded to "
        "BF16 and quantized per value head to E4M3FN, with scale=max(amax,1e-10)/448. "
        "Only the indexed state slots change. cu_seqlens partitions all tokens into "
        "positive sequences with actual maximum max_seq_len. Unindexed state slots and all "
        "other inputs are preserved."
    ),
    axes={
        "n_tokens": axis("Tokens in this prefill.", 1, 16384, is_var=True),
        "n_sequences": axis("Active state slots/sequences.", 1, 2047, is_var=True),
        "n_slots": axis("Allocated state slots.", 1, 2047, is_var=True, any_size=True),
        "n_v_heads": axis("Value heads on this rank.", 1, 16),
        "key_dim": axis("Key head width.", 16, 128),
        "value_dim": axis("FP8 group and value-head width.", 128, 128),
        "conv_dim": axis("Convolution channels.", 160, 3072),
        "value_width": derived_axis("Total value width.", "n_v_heads * value_dim"),
        "n_k_heads": derived_axis("Key heads on this rank.", "(conv_dim - value_width) // (2 * key_dim)"),
        "projection_dim": derived_axis("Q/K/V/Z projection width.", "conv_dim + value_width"),
        "gate_dim": derived_axis("B/A projection width.", "2 * n_v_heads"),
        "n_starts": axis("Sequence boundaries.", 2, 2048, is_var=True),
    },
    attrs={
        "max_seq_len": attr(
            "Actual maximum sequence length.",
            2048,
            2,
            256,
            512,
            1024,
            1025,
            5000,
            8192,
            10000,
            16384,
            default=2048,
            param_name="max_seq_len",
        ),
    },
    constraints=(
        "conv_dim > value_width",
        "(conv_dim - value_width) % (2 * key_dim) == 0",
        "n_k_heads >= 1",
        "n_k_heads <= 4",
        "n_v_heads % n_k_heads == 0",
        "n_slots >= n_sequences",
        "n_starts == n_sequences + 1",
        "n_tokens >= n_sequences + max_seq_len - 1",
        "n_tokens <= n_sequences * max_seq_len",
    ),
    inputs={
        "projected_qkvz": _tensor("Contiguous Q, K, V, Z columns.", ("n_tokens", "projection_dim"), _BF16),
        "projected_ba": _tensor("Contiguous B then A columns.", ("n_tokens", "gate_dim"), _BF16),
        "conv_state": _tensor(
            "Live three-token convolution histories.",
            ("n_slots", "conv_dim", 3),
            _BF16,
            mutable=True,
            bound=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
        "delta_state": _tensor(
            "Live recurrent matrices, value dimension first.",
            ("n_slots", "n_v_heads", "value_dim", "key_dim"),
            _FP32,
            mutable=True,
            # SGLang's own prefill puts a quarter of Q/K/V one BF16 ulp from the
            # reference and rounds again in its chunked recurrence: its state is off by
            # up to 2**-8 of the largest |update x key|, ~2e-2 at N(0, 1) inputs
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
            "Use each slot's initial state, or start from zero.",
            ("n_sequences",),
            DTypeSpec(any_of=(DType.BOOL,)),
        ),
        "conv_weight": _tensor("Four causal convolution taps.", ("conv_dim", 4), _BF16),
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
    },
    outputs={
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
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

# Decode reads the same projections, weights and state pools as prefill, one token a row.
_PREFILL = GDN_PREFILL_GROUP_FP8_QUANT
_DECODE_AXES = {
    "n_tokens": axis("Decode rows: one token each, padding rows included.", 1, 2048, is_var=True),
    **{
        n: _PREFILL.axes[n]
        for n in (
            "n_slots",
            "n_v_heads",
            "key_dim",
            "value_dim",
            "conv_dim",
            "value_width",
            "n_k_heads",
            "projection_dim",
            "gate_dim",
        )
    },
}
_DECODE_CONSTRAINTS = (
    "conv_dim > value_width",
    "(conv_dim - value_width) % (2 * key_dim) == 0",
    "n_k_heads >= 1",
    "n_k_heads <= 4",
    "n_v_heads % n_k_heads == 0",
    # drawn indices are distinct, -1 among them at most once
    "n_tokens <= n_slots + 1",
)
_DECODE_INPUTS = {
    **{n: _PREFILL.inputs[n] for n in ("projected_qkvz", "projected_ba", "conv_state")},
    "delta_state": _tensor(
        "Live recurrent matrices, value dimension first.",
        ("n_slots", "n_v_heads", "value_dim", "key_dim"),
        _FP32,
        mutable=True,
        # one step's change is |update x key|; a Q/K/V channel a correct kernel
        # rounds to the neighbouring BF16 value moves it by 2**-8 of that
        bound=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
    ),
    "cache_indices": _tensor(
        "Each row's distinct state slot, or -1 for a padding row.",
        ("n_tokens",),
        _INT32,
        distribution=IntRangeDistribution(lo=-1, hi="n_slots - 1", distinct=True),
    ),
    **{n: _PREFILL.inputs[n] for n in ("conv_weight", "a_log", "dt_bias", "norm_weight", "scale", "eps")},
}
_DECODE_TEXT = (
    "Qwen3.5 GDN decode, one token per row, from contiguous Q/K/V/Z and B/A "
    "projections. A row whose cache index is -1 is padding: it reads and changes no "
    "state, and its core output is zero. Every other row uses its distinct slot: the "
    "causal width-four convolution of the slot's three-token history and the row's "
    "Q/K/V channels rounds each tap's product to BF16 and sums the four in FP32, "
    "oldest first; its SiLU is rounded to BF16, and the history shifts the row in. "
    "Q and K are L2-normalized in FP32 and Q is multiplied by scale; the gated delta "
    "rule with BF16 beta updates the slot's FP32 delta_state in [value,key] order, "
    "and the core output is rounded to BF16. The core is gated RMS-normalized with "
    "norm_weight and SiLU of the row's Z, then rounded to BF16"
)

GDN_DECODE_GROUP_FP8_QUANT = OpContract(
    contract_id="gdn_decode_group_fp8_quant@1.0",
    op_type="qwen3_5",
    description=(
        f"{_DECODE_TEXT} and quantized per value head to E4M3FN, with "
        "scale=max(amax,1e-10)/448. Unindexed state slots and all other inputs are "
        "preserved."
    ),
    axes=_DECODE_AXES,
    constraints=_DECODE_CONSTRAINTS,
    inputs=_DECODE_INPUTS,
    outputs={n: _PREFILL.outputs[n] for n in ("values", "scales")},
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

GDN_DECODE = OpContract(
    contract_id="gdn_decode@1.0",
    op_type="qwen3_5",
    description=(f"{_DECODE_TEXT}. Unindexed state slots and all other inputs are preserved."),
    axes=_DECODE_AXES,
    constraints=_DECODE_CONSTRAINTS,
    inputs=_DECODE_INPUTS,
    outputs={"normalized": _PREFILL.outputs["normalized"]},
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

CONTRACTS = (GDN_PREFILL_GROUP_FP8_QUANT, GDN_DECODE_GROUP_FP8_QUANT, GDN_DECODE)
