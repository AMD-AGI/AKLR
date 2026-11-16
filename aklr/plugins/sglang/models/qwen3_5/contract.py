# Created by OpenAI
"""GDN convolution, recurrent state update, gated RMSNorm and grouped FP8 output; the
full-attention layers' gated decode attention over their FP8 KV cache."""

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    BoundedCuSeqlensDistribution,
    CuSeqlensDistribution,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    SqnrBound,
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
    "GATED_ATTENTION_DECODE",
    "GATED_ATTENTION_DECODE_GROUP_FP8_QUANT",
    "GATED_ATTENTION_PREFILL",
    "GATED_ATTENTION_PREFILL_GROUP_FP8_QUANT",
    "GDN_DECODE",
    "GDN_DECODE_GROUP_FP8_QUANT",
    "GDN_PREFILL",
    "GDN_PREFILL_GROUP_FP8_QUANT",
    "MROPE_SECTION",
    "ROTARY_DIM",
]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_INT64 = DTypeSpec(any_of=(DType.INT64,))
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
            4096,
            5000,
            5120,
            6144,
            8192,
            10000,
            10240,
            12288,
            14336,
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

# The same prefill where out_proj takes BF16 (Qwen3.5's MXFP4 checkpoint keeps its GDN layers
# unquantized): the gated normalized output alone.
GDN_PREFILL = OpContract(
    contract_id="gdn_prefill@1.0",
    op_type="qwen3_5",
    description=(
        "Qwen3.5 GDN prefill from contiguous Q/K/V/Z and B/A projections. Each sequence "
        "uses its distinct cache_indices slot; fresh sequences ignore the slot's old values. "
        "Causal width-four SiLU convolution updates conv_state, Q and K are L2-normalized "
        "and rounded to BF16, and the gated delta recurrence updates FP32 delta_state "
        "in [value,key] order. BF16 attention output is gated RMS-normalized and rounded to "
        "BF16. Only the indexed state slots change. cu_seqlens partitions all tokens into "
        "positive sequences with actual maximum max_seq_len. Unindexed state slots and all "
        "other inputs are preserved."
    ),
    axes=GDN_PREFILL_GROUP_FP8_QUANT.axes,
    attrs=GDN_PREFILL_GROUP_FP8_QUANT.attrs,
    constraints=GDN_PREFILL_GROUP_FP8_QUANT.constraints,
    inputs=GDN_PREFILL_GROUP_FP8_QUANT.inputs,
    outputs={"normalized": GDN_PREFILL_GROUP_FP8_QUANT.outputs["normalized"]},
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

# A full-attention head's rotary part: NeoX RoPE on its first 64 channels, whose 32
# frequencies Qwen3.5's interleaved mRoPE splits over the temporal, height and width positions
ROTARY_DIM = 64
MROPE_SECTION = (11, 11, 10)
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)
# The FP8 pools' bytes that the kernels' 32-bit offsets address: larger pools stay on sglang.
_POOL_BYTES = 2**31
# A cache slot a correct kernel writes may round its BF16 key one FP8 step away
_CACHE_BOUND = ToleranceBound(absolute_tolerance=2**-9, relative_tolerance=0.125)
_ATTENTION_TEXT = (
    "Qwen3.5 full-attention decode on one rank, between qkv_proj and o_proj, one token "
    "per row. projected_qkv holds each query head's query then its output gate, then "
    "the KV heads' keys, then their values. Q and K are RMS-normalized per head with "
    "weight 1 + norm weight and rounded to BF16; each head's first 64 channels take NeoX "
    "RoPE, rounded to BF16, whose frequency f in [0, 32) reads positions[1] when "
    "f % 3 == 1 and f < 33, positions[2] when f % 3 == 2 and f < 30, else positions[0], "
    "in cos_sin_cache's row of 32 cosines then 32 sines. Each row writes its K and V, "
    "cast to E4M3FN, at its cache_locations slot of key_cache and value_cache; rows "
    "sharing a slot leave one of their values there. Query head h then attends, through "
    "KV head h // (n_heads // n_kv_heads), the written slots "
    "kv_indices[kv_indptr[r]:kv_indptr[r + 1]] with scores q . k * scale; the "
    "softmax-weighted sum of their values is rounded to BF16, multiplied by sigmoid of "
    "the gate and rounded to BF16"
)

GATED_ATTENTION_DECODE_GROUP_FP8_QUANT = OpContract(
    contract_id="gated_attention_decode_group_fp8_quant@1.0",
    op_type="qwen3_5",
    description=(
        f"{_ATTENTION_TEXT}, then quantized in groups of 128 channels to E4M3FN "
        "with scale=max(amax,1e-10)/448. Unwritten slots and all other inputs are preserved."
    ),
    axes={
        "n_tokens": axis("Decode rows: one token each, padding rows included.", 1, 2048, is_var=True),
        "qkv_dim": axis("qkv_proj's output width on this rank.", 256, 81920),
        "n_kv_heads": axis("KV heads on this rank.", 1, 16),
        "head_dim": axis("Query, key and value head width.", 64, 512, multiple_of=64),
        "n_slots": axis("Rows of the KV pools.", 1, 2**31 - 1, is_var=True, any_size=True),
        "n_positions": axis("Rows of the cos/sin table.", 1, 2**25, is_var=True, any_size=True),
        "n_offsets": axis("Offsets in kv_indptr: n_tokens + 1.", 2, 2049, is_var=True),
        "n_kv_tokens": axis("Entries of kv_indices.", 1, 2**29 - 1, is_var=True, any_size=True),
        "n_heads": derived_axis(
            "Query heads on this rank.", "(qkv_dim - 2 * n_kv_heads * head_dim) // (2 * head_dim)"
        ),
        "n_groups": derived_axis("FP8 groups of 128 channels in a row.", "n_heads * head_dim // 128"),
    },
    constraints=(
        "(qkv_dim - 2 * n_kv_heads * head_dim) % (2 * head_dim) == 0",
        "n_heads >= 1",
        "n_heads % n_kv_heads == 0",
        "(n_heads * head_dim) % 128 == 0",
        "n_offsets == n_tokens + 1",
        # every row attends at least one slot, and each row writes its own slot
        "n_kv_tokens >= n_tokens",
        "n_slots >= n_tokens",
        f"n_slots * n_kv_heads * head_dim < {_POOL_BYTES}",
    ),
    inputs={
        "projected_qkv": _tensor(
            "qkv_proj's output: each query head's query then gate, then keys, then values.",
            ("n_tokens", "qkv_dim"),
            _BF16,
        ),
        "q_norm_weight": _tensor("Query RMSNorm weight, applied as 1 + weight.", ("head_dim",), _BF16),
        "k_norm_weight": _tensor("Key RMSNorm weight, applied as 1 + weight.", ("head_dim",), _BF16),
        "positions": _tensor(
            "Each row's temporal, height and width positions.",
            (3, "n_tokens"),
            _INT64,
            distribution=IntRangeDistribution(lo=0, hi="n_positions - 1"),
        ),
        "cos_sin_cache": _tensor(
            "Per position, the 32 rotary cosines then the 32 sines.", ("n_positions", ROTARY_DIM), _FP32
        ),
        "cache_locations": _tensor(
            "Each row's KV pool slot.",
            ("n_tokens",),
            _INT64,
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        "key_cache": _tensor(
            "This layer's key pool.",
            ("n_slots", "n_kv_heads", "head_dim"),
            _FP8,
            mutable=True,
            bound=_CACHE_BOUND,
        ),
        "value_cache": _tensor(
            "This layer's value pool.",
            ("n_slots", "n_kv_heads", "head_dim"),
            _FP8,
            mutable=True,
            bound=_CACHE_BOUND,
        ),
        "kv_indptr": _tensor(
            "Where each row's slots start in kv_indices, then where the last row's end.",
            ("n_offsets",),
            _INT32,
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_kv_tokens"),
        ),
        "kv_indices": _tensor(
            "The pool slots each row attends, row after row.",
            ("n_kv_tokens",),
            _INT32,
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1"),
        ),
        "eps": ScalarSpec(
            description="Q and K RMSNorm epsilon.",
            dtype=DType.FLOAT32,
            interval=(9.999999974752427e-7, 1e-6),
            default=1e-6,
        ),
        "scale": ScalarSpec(
            description="Multiplies every score.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=256**-0.5,
        ),
    },
    # sglang's own decode is lossier than the reference: on an FP8 cache, aiter's
    # paged_attention_ragged rounds Q and each 256-token partition's softmax weights to E4M3
    # (flushing the smallest to zero) and keeps the partitions' outputs in BF16. On the small
    # test Point, CI's unittest-aklr (run 41950785417) found it 6.3x past a per-element bound
    # of 2 + 0.125|y| on 3.1% of the values and 1.9x past 3% on 25% of the scales; an FP64
    # emulation of those roundings, which gives the same numbers there, comes to 25.7-27.6 dB
    # (values) and 31.7-32.5 dB (scales) on it, 21.8-23.1 dB and 26.7-34.1 dB at the schemas'
    # Points, with scales up to 30% off. The bounds sit some 6-7 dB under that, as
    # baseline_relative's twice the baseline's error; kernels wrong on purpose (no gate,
    # sequential mRoPE sections, no Q norm, a 128**-0.5 scale, 240 for E4M3FNUZ) stay under
    # 9 dB (values) and 10 dB (scales) at those Points, at most 16.4 and 16.8 dB on the small one.
    outputs={
        "values": TensorSpec(
            description="Each query head's gated attention, grouped E4M3FN values.",
            shape=("n_tokens", "n_heads", "head_dim"),
            dtype=_FP8,
            error_bounds=SqnrBound(min_sqnr_db=15.0),
        ),
        "scales": TensorSpec(
            description="One FP32 scale per row and group of 128 channels.",
            shape=("n_tokens", "n_groups"),
            dtype=_FP32,
            error_bounds=SqnrBound(min_sqnr_db=20.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

# The same decode where sglang hands o_proj the gated output in BF16, as for an unquantized
# o_proj (Qwen3.5's MXFP4 checkpoint keeps its attention projections in BF16).
_FP8_ATTENTION = GATED_ATTENTION_DECODE_GROUP_FP8_QUANT

GATED_ATTENTION_DECODE = OpContract(
    contract_id="gated_attention_decode@1.0",
    op_type="qwen3_5",
    description=f"{_ATTENTION_TEXT}. Unwritten slots and all other inputs are preserved.",
    axes={
        **{n: _FP8_ATTENTION.axes[n] for n in ("n_tokens", "qkv_dim", "n_kv_heads", "head_dim", "n_slots")},
        # the kernels' buffer loads address the FP32 table by 32-bit byte offsets: its 2**30
        # bytes at most
        "n_positions": axis("Rows of the cos/sin table.", 1, 2**22, is_var=True, any_size=True),
        **{n: _FP8_ATTENTION.axes[n] for n in ("n_offsets", "n_kv_tokens", "n_heads")},
        "attention_width": derived_axis("Channels of a row: n_heads * head_dim.", "n_heads * head_dim"),
    },
    constraints=(
        "(qkv_dim - 2 * n_kv_heads * head_dim) % (2 * head_dim) == 0",
        "n_heads >= 1",
        "n_heads % n_kv_heads == 0",
        "n_offsets == n_tokens + 1",
        # every row attends at least one slot, and each row writes its own slot
        "n_kv_tokens >= n_tokens",
        "n_slots >= n_tokens",
        # the pool elements that the kernels' 32-bit offsets address
        f"n_slots * n_kv_heads * head_dim < {2**32}",
    ),
    inputs=_FP8_ATTENTION.inputs,
    # sglang's own decode is lossier than the reference (see the FP8 op): aiter's
    # paged_attention_ragged rounds Q and each 256-token partition's softmax weights to E4M3
    # and keeps the partitions' outputs in BF16. An FP64 emulation of those roundings gives
    # 25.2-30.6 dB on the small test Point (16 draws) and mostly 21-27 dB at the schemas'
    # Points (64 draws at 1, 2, 4 and 8 rows, two at each other count, 23-25 dB on rows of
    # 32768-131072 tokens), down to 15.0 dB on the draws where the E4M3 Q moves one peaked
    # head; the bound admits them all. The kernels round Q to E4M3 only on rows past 8192
    # (8 rows) or 16384 tokens (1 and 4 rows), as sglang, and their PV weights to FP16 or BF16:
    # 49.8 dB and up at those Points otherwise. Kernels wrong on purpose (no gate, the gate
    # without its sigmoid, sequential mRoPE sections, no Q norm, a 128**-0.5 scale) stay under
    # 8 dB at the schemas' Points and at 12.6 dB at most on the small one.
    outputs={
        "gated": TensorSpec(
            description="Each query head's gated attention, the heads side by side: o_proj's input.",
            shape=("n_tokens", "attention_width"),
            dtype=_BF16,
            error_bounds=SqnrBound(min_sqnr_db=14.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

# Prefill of fresh sequences (no cached prefix): each row attends its own sequence's rows up to
# itself, whose keys and values it has just written, so cu_seqlens alone gives what it reads.
_PREFILL_ATTENTION_TEXT = (
    "Qwen3.5 full-attention prefill of fresh sequences on one rank, between qkv_proj and "
    "o_proj. projected_qkv holds each query head's query then its output gate, then the KV "
    "heads' keys, then their values. Q and K are RMS-normalized per head with weight 1 + norm "
    "weight and rounded to BF16; each head's first 64 channels take NeoX RoPE, rounded to "
    "BF16, whose frequency f in [0, 32) reads positions[1] when f % 3 == 1 and f < 33, "
    "positions[2] when f % 3 == 2 and f < 30, else positions[0], in cos_sin_cache's row of 32 "
    "cosines then 32 sines. Each row writes its K and V, cast to E4M3FN, at its distinct "
    "cache_locations slot of key_cache and value_cache. cu_seqlens partitions the rows into "
    "sequences with actual maximum max_seq_len and no cached prefix: query head h of a row "
    "attends, through KV head h // (n_heads // n_kv_heads), the written slots of its "
    "sequence's rows up to itself with scores q . k * scale; the softmax-weighted sum of "
    "their values is rounded to BF16, multiplied by sigmoid of the gate and rounded to BF16"
)
_PREFILL_ATTENTION_AXES = {
    "n_tokens": axis("Prefill rows.", 1, 16384, is_var=True),
    **{n: _FP8_ATTENTION.axes[n] for n in ("qkv_dim", "n_kv_heads", "head_dim", "n_slots")},
    "n_positions": GATED_ATTENTION_DECODE.axes["n_positions"],
    "n_starts": axis("Sequence boundaries: sequences + 1.", 2, 2048, is_var=True),
    "n_heads": _FP8_ATTENTION.axes["n_heads"],
}
_PREFILL_ATTENTION_CONSTRAINTS = (
    "(qkv_dim - 2 * n_kv_heads * head_dim) % (2 * head_dim) == 0",
    "n_heads >= 1",
    "n_heads % n_kv_heads == 0",
    "n_tokens >= n_starts + max_seq_len - 2",
    "n_tokens <= (n_starts - 1) * max_seq_len",
    "n_slots >= n_tokens",
)
_PREFILL_ATTENTION_ATTRS = {
    "max_seq_len": attr(
        "Actual maximum sequence length.",
        1024,
        32,
        2048,
        3072,
        4096,
        5120,
        6144,
        7168,
        8192,
        10240,
        12288,
        14336,
        16384,
        default=1024,
        param_name="max_seq_len",
    ),
}
# Over thousands of rows a key's BF16 norm and RoPE, one ulp from the reference's, can round two
# FP8 steps away; a wrong slot or an unrotated key is still far past this bound.
_PREFILL_CACHE_BOUND = ToleranceBound(absolute_tolerance=2**-8, relative_tolerance=0.25)
_PREFILL_ATTENTION_INPUTS = {
    **{
        n: _FP8_ATTENTION.inputs[n]
        for n in (
            "projected_qkv",
            "q_norm_weight",
            "k_norm_weight",
            "positions",
            "cos_sin_cache",
            "cache_locations",
        )
    },
    **{
        n: _tensor(
            f"This layer's {n.split('_')[0]} pool.",
            ("n_slots", "n_kv_heads", "head_dim"),
            _FP8,
            mutable=True,
            bound=_PREFILL_CACHE_BOUND,
        )
        for n in ("key_cache", "value_cache")
    },
    "cu_seqlens": _tensor(
        "Cumulative sequence boundaries.",
        ("n_starts",),
        _INT32,
        distribution=BoundedCuSeqlensDistribution(total="n_tokens", max_seqlen="max_seq_len"),
    ),
    **{n: _FP8_ATTENTION.inputs[n] for n in ("eps", "scale")},
}

# sglang's own fresh prefill (flash_attn_varlen_fp8_pertensor_func) attends Q, K and V cast to
# E4M3FN; the bounds are the decode ops'.
GATED_ATTENTION_PREFILL_GROUP_FP8_QUANT = OpContract(
    contract_id="gated_attention_prefill_group_fp8_quant@1.0",
    op_type="qwen3_5",
    description=(
        f"{_PREFILL_ATTENTION_TEXT}, then quantized in groups of 128 channels to E4M3FN "
        "with scale=max(amax,1e-10)/448. Unwritten slots and all other inputs are preserved."
    ),
    axes={
        **_PREFILL_ATTENTION_AXES,
        "n_groups": _FP8_ATTENTION.axes["n_groups"],
    },
    attrs=_PREFILL_ATTENTION_ATTRS,
    constraints=(
        *_PREFILL_ATTENTION_CONSTRAINTS,
        "(n_heads * head_dim) % 128 == 0",
    ),
    inputs=_PREFILL_ATTENTION_INPUTS,
    outputs=_FP8_ATTENTION.outputs,
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

GATED_ATTENTION_PREFILL = OpContract(
    contract_id="gated_attention_prefill@1.0",
    op_type="qwen3_5",
    description=f"{_PREFILL_ATTENTION_TEXT}. Unwritten slots and all other inputs are preserved.",
    axes={
        **_PREFILL_ATTENTION_AXES,
        "attention_width": GATED_ATTENTION_DECODE.axes["attention_width"],
    },
    attrs=_PREFILL_ATTENTION_ATTRS,
    constraints=_PREFILL_ATTENTION_CONSTRAINTS,
    inputs=_PREFILL_ATTENTION_INPUTS,
    outputs=GATED_ATTENTION_DECODE.outputs,
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

CONTRACTS = (
    GDN_PREFILL_GROUP_FP8_QUANT,
    GDN_PREFILL,
    GDN_DECODE_GROUP_FP8_QUANT,
    GDN_DECODE,
    GATED_ATTENTION_DECODE_GROUP_FP8_QUANT,
    GATED_ATTENTION_DECODE,
    GATED_ATTENTION_PREFILL_GROUP_FP8_QUANT,
    GATED_ATTENTION_PREFILL,
)
