# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's layers on one rank of its TP group: the MoE from hidden_states to the two partial
sums its fused front path all-reduces, and the MLA layers' absorbed decode attention.

A model is a scenario: Kimi-K3 at TP8 is model_dim 7168, latent_dim 3584, n_routed 896, topk
16, inter_dim 384 and shared_inter_dim 768, and 12 MLA heads."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    BoundedCuSeqlensDistribution,
    CuSeqlensDistribution,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    GateUpInterleavedScales,
    GateUpInterleavedWeight,
    IntRangeDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    SqnrBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutFact,
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
    "KIMI_K3_LATENT_MOE",
    "KIMI_K3_MLA_DECODE",
    "KIMI_K3_MLA_PREFILL_GATE",
    "KIMI_K3_MLA_VC_OUTPUT_GATE",
]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_MXFP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))


def _e8m0_scales(fan_in: str) -> LogUniformDistribution:
    """MXFP4 block scales log-uniform over 2**-5 to 2**1 times 1 / sqrt(fan_in), as the
    other MXFP4 MoE ops draw them."""
    return LogUniformDistribution(
        log2_lo=f"-5 - bit_length({fan_in}) // 2", log2_hi=f"1 - bit_length({fan_in}) // 2"
    )


def _facts(shuffled: bool, scale_rank: int) -> tuple[LayoutFact, ...]:
    facts = {"w1.is_shuffled": shuffled, "w2.is_shuffled": shuffled, "w1_scale.ndim": scale_rank}
    return tuple(LayoutFact(location="tensor_attr", name=n, expected_value=v) for n, v in facts.items())


# row-major, or as sglang holds SiTU experts for aiter's a8w4 path: shuffle_weight_a16w4's
# gate-up-interleaved tiles, the scales shuffled 2-D (fused_moe's gate_up_interleaved)
_LAYOUTS = LayoutVariantsSpec(
    by_name={
        "row_major": LayoutVariant(layouts={}, detect_facts=_facts(False, 3)),
        "gate_up_interleaved": LayoutVariant(
            layouts={
                "w1": GateUpInterleavedWeight(gate_up=True),
                "w2": GateUpInterleavedWeight(gate_up=False),
                "w1_scale": GateUpInterleavedScales(gate_up=True),
                "w2_scale": GateUpInterleavedScales(gate_up=False),
            },
            detect_facts=_facts(True, 2),
        ),
    }
)

_SITU = "situ(g, u) = 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)"

KIMI_K3_LATENT_MOE = OpContract(
    contract_id="kimi_k3_latent_moe@1.0",
    description=(
        "sglang's KimiK3MoE._forward_fused on one rank of its TP group, up to its all-reduce: "
        "front = hidden_states front_weight^T rounded to bf16, split into the shared experts' "
        "gate and up rows, the router's logits and the latent input; each token's topk "
        "experts by sigmoid(logit) + correction_bias, weighed by their sigmoid scores "
        "renormalized; each routed expert down(situ(gate(z), up(z))) of its MXFP4 weights on "
        "the latent input z, its two linears taking their inputs in MXFP8 (aiter's a8w4), "
        f"{_SITU}; latent is the routed experts' weighed sum and shared the shared experts' "
        "down projection of situ(gate, up), both this rank's share, which the TP group sums "
        "after the call"
    ),
    op_type="kimi_k3",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "model_dim": axis("Hidden size.", 256, 16384, multiple_of=256),
        "latent_dim": axis("The routed experts' hidden size.", 256, 16384, multiple_of=256),
        "n_routed": axis("Routed experts: the router's rows.", 32, 1024, multiple_of=32),
        "inter_dim": axis("Routed expert intermediate size on this rank.", 128, 16384, multiple_of=128),
        "shared_inter_dim": axis(
            "Shared experts' intermediate size on this rank.", 128, 16384, multiple_of=128
        ),
        "shared_gate_up_rows": derived_axis(
            "front_weight's shared rows: gate, then up.", "2 * shared_inter_dim"
        ),
        "front_rows": derived_axis(
            "front_weight rows: shared gate and up, router, latent input.",
            "2 * shared_inter_dim + n_routed + latent_dim",
        ),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "latent_blocks": derived_axis("w1 scale columns.", "latent_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    attrs={"topk": attr("Routed experts per token.", 16, param_name="topk")},
    constraints=("topk <= n_routed",),
    inputs={
        "hidden_states": TensorSpec(
            description="The MoE's input: post_attention_layernorm's output.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "front_weight": TensorSpec(
            description=(
                "sglang's merged front: the shared experts' gate_up_proj (gate rows, then up "
                "rows), the gate's router rows and routed_expert_down_proj, concatenated."
            ),
            shape=("front_rows", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "correction_bias": TensorSpec(
            description="Added to the scores to pick experts; never weighs them.",
            shape=("n_routed",),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            # about 0.02 across, Kimi-K3's spread
            distribution=LecunNormalDistribution(fan_in="3 * n_routed"),
            lifetime="weight",
        ),
        "w1": TensorSpec(
            description="Gate-up projection per routed expert: gate rows, then up rows.",
            shape=("n_routed", "gate_up_rows", "latent_dim"),
            dtype=_MXFP4,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "w1_scale": TensorSpec(
            description="Per 1 x 32 block of w1.",
            shape=("n_routed", "gate_up_rows", "latent_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("latent_dim"),
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per routed expert.",
            shape=("n_routed", "latent_dim", "inter_dim"),
            dtype=_MXFP4,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="Per 1 x 32 block of w2.",
            shape=("n_routed", "latent_dim", "inter_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("inter_dim"),
            lifetime="weight",
        ),
        "shared_w2": TensorSpec(
            description="The shared experts' down projection on this rank.",
            shape=("model_dim", "shared_inter_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="shared_inter_dim"),
            lifetime="weight",
        ),
    },
    outputs={
        "latent": TensorSpec(
            description="The routed experts' weighed sum on this rank, in the latent space.",
            shape=("n_tokens", "latent_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            error_bounds=SqnrBound(min_sqnr_db=20.0),
        ),
        "shared": TensorSpec(
            description="The shared experts' output on this rank.",
            shape=("n_tokens", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            error_bounds=SqnrBound(min_sqnr_db=30.0),
        ),
    },
    layout_variants=_LAYOUTS,
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)

KIMI_K3_MLA_DECODE = OpContract(
    contract_id="kimi_k3_mla_decode@1.0",
    description=(
        "sglang's TritonAttnBackend.forward_decode for Kimi-K3's MLA layers past its KV write, "
        "one query per sequence: each query head attends the slots "
        "kv_indices[kv_indptr[r]:kv_indptr[r + 1]] of the latent cache (page size 1, no RoPE), "
        "scores q . row * sm_scale over all 576 values, and output is the softmax-weighted sum "
        "of the rows' first 512 values"
    ),
    op_type="kimi_k3",
    axes={
        "n_seqs": axis("Decode rows: one query each, padding rows included.", 1, 2048, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "n_slots": axis("Rows of the latent cache.", 1, 2**31 - 1, is_var=True, any_size=True),
        "n_offsets": axis("Offsets in kv_indptr: n_seqs + 1.", 2, 2049, is_var=True),
        "n_kv_tokens": axis("Entries of kv_indices.", 1, 2**31 - 1, is_var=True, any_size=True),
    },
    constraints=("n_offsets == n_seqs + 1", "n_kv_tokens >= n_seqs"),
    inputs={
        "q": TensorSpec(
            description="Each row's absorbed query per head: 512 latent values, then 64 rope ones.",
            shape=("n_seqs", "n_heads", 576),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "kv_cache": TensorSpec(
            description="This layer's latent cache: 512 latent values then 64 rope ones a slot.",
            shape=("n_slots", 1, 576),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "kv_indptr": TensorSpec(
            description="Where each row's slots start in kv_indices, then where the last row's end.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_kv_tokens"),
        ),
        "kv_indices": TensorSpec(
            description="The cache slots each row attends, row after row.",
            shape=("n_kv_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT64,)),
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1"),
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score: Kimi-K3's 192**-0.5.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=192**-0.5,
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Each head's softmax-weighted sum of the slots' first 512 values.",
            shape=("n_seqs", "n_heads", 512),
            dtype=DTypeSpec(same_as_tensor="q"),
            error_bounds=SqnrBound(min_sqnr_db=30.0),
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

KIMI_K3_MLA_VC_OUTPUT_GATE = OpContract(
    contract_id="kimi_k3_mla_vc_output_gate@1.0",
    description=(
        "Kimi-K3's MLA absorbed attention output to o_proj's input, as sglang's "
        "rocm_absorb_v_bmm and its output gate do it: per head, latent vc_weight^T rounded to "
        "bf16, times sigmoid(gate) rounded to bf16, the product rounded to bf16, the heads "
        "flattened"
    ),
    op_type="kimi_k3",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "v_head_dim": axis("Value head size.", 32, 512, multiple_of=32),
        "gate_dim": derived_axis("Gate and output width.", "n_heads * v_head_dim"),
    },
    inputs={
        "latent": TensorSpec(
            description="The attention output per head: 512 latent values.",
            shape=("n_tokens", "n_heads", 512),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "vc_weight": TensorSpec(
            description="Per head, kv_b_proj's value rows: v_head_dim rows of 512 latent values.",
            shape=("n_heads", "v_head_dim", 512),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in=512),
            lifetime="weight",
        ),
        "gate": TensorSpec(
            description="g_proj's output: the gate logits, heads flattened.",
            shape=("n_tokens", "gate_dim"),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
    },
    outputs={
        "output": TensorSpec(
            description="o_proj's input: the gated values, heads flattened.",
            shape=("n_tokens", "gate_dim"),
            dtype=DTypeSpec(same_as_tensor="latent"),
            error_bounds=SqnrBound(min_sqnr_db=40.0),
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

KIMI_K3_MLA_PREFILL_GATE = OpContract(
    contract_id="kimi_k3_mla_prefill_gate@1.0",
    description=(
        "sglang's forward_normal_core for a Kimi-K3 MLA layer's prefill without cached prefix, "
        "up to o_proj: each sequence of qo_indptr attends causally to itself, query i to keys "
        "0..i, scores q . k * sm_scale over 192 values; the softmax-weighted values, rounded "
        "to bf16, times sigmoid(gate) rounded to bf16, the product rounded to bf16, heads "
        "flattened"
    ),
    op_type="kimi_k3",
    axes={
        "n_tokens": axis("Tokens in this prefill.", 1, 65536, is_var=True),
        "n_offsets": axis("Offsets in qo_indptr: one more than the sequences.", 2, 2049, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "gate_dim": derived_axis("Gate and output width.", "n_heads * 128"),
    },
    attrs={
        "max_seq_len": attr(
            "The longest sequence.",
            8192,
            1024,
            2048,
            3072,
            4096,
            5120,
            6144,
            7168,
            default=8192,
            param_name="max_seq_len",
        ),
    },
    constraints=(
        # n_offsets - 1 sequences, the longest max_seq_len tokens
        "n_tokens >= n_offsets + max_seq_len - 2",
        "n_tokens <= (n_offsets - 1) * max_seq_len",
    ),
    inputs={
        "q": TensorSpec(
            description="Queries per head: 128 nope values, then 64 rope ones.",
            shape=("n_tokens", "n_heads", 192),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "k": TensorSpec(
            description="Keys per head: 128 nope values, then 64 rope ones.",
            shape=("n_tokens", "n_heads", 192),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "v": TensorSpec(
            description="Values per head.",
            shape=("n_tokens", "n_heads", 128),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "gate": TensorSpec(
            description="g_proj's output: the gate logits, heads flattened.",
            shape=("n_tokens", "gate_dim"),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "qo_indptr": TensorSpec(
            description="Where each sequence starts, then where the last one ends.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=BoundedCuSeqlensDistribution(total="n_tokens", max_seqlen="max_seq_len"),
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score: Kimi-K3's 192**-0.5.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=192**-0.5,
        ),
    },
    outputs={
        "output": TensorSpec(
            description="o_proj's input: the gated attention, heads flattened.",
            shape=("n_tokens", "gate_dim"),
            dtype=DTypeSpec(same_as_tensor="q"),
            # sglang's own path rounds Q, K and V to FP8: about 26 dB against this reference
            error_bounds=SqnrBound(min_sqnr_db=20.0),
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

CONTRACTS = (KIMI_K3_LATENT_MOE, KIMI_K3_MLA_DECODE, KIMI_K3_MLA_VC_OUTPUT_GATE, KIMI_K3_MLA_PREFILL_GATE)
