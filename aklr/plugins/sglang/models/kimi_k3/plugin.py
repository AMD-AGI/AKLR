# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's layers on one rank of its TP group: the MoE from hidden_states to the two partial
sums its fused front path all-reduces, and the MLA layers' value absorption and prefill
attention, each with the output gate.

A model is a scenario: Kimi-K3 at TP8 is model_dim 7168, latent_dim 3584, n_routed 896, topk
16, inter_dim 384 and shared_inter_dim 768, and 12 MLA heads."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    BoundedCuSeqlensDistribution,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    GateUpInterleavedScales,
    GateUpInterleavedWeight,
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
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin
from aklr.plugins.sglang.models import moe_blocks

__all__ = [
    "KIMI_K3_LATENT_MOE",
    "KIMI_K3_MLA_PREFILL_GATE",
    "KIMI_K3_MLA_VC_OUTPUT_GATE",
    "declare",
    "family",
    "kimi_k3_latent_moe",
    "kimi_k3_mla_prefill_gate",
    "kimi_k3_mla_vc_output_gate",
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


family = Plugin("sglang")
family.define(KIMI_K3_LATENT_MOE)
family.define(KIMI_K3_MLA_VC_OUTPUT_GATE)
family.define(KIMI_K3_MLA_PREFILL_GATE)
family.build()


@family.op
def kimi_k3_latent_moe(
    hidden_states,
    front_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    shared_w2,
    topk=16,
    latent=None,
    shared=None,
): ...


@family.op
def kimi_k3_mla_vc_output_gate(latent, vc_weight, gate, output=None): ...


@family.op
def kimi_k3_mla_prefill_gate(q, k, v, gate, qo_indptr, sm_scale=192**-0.5, max_seq_len=8192, output=None): ...


def _forward_fused(self, hidden_states, *, prefix_sum):
    """The front, routing, routed and shared experts as one op writing both partial sums into
    the buffer sglang all-reduces; then sglang's all-reduce, latent norm, up projection and
    tail add, as its single-collective branch runs them."""
    if moe_blocks.runtime_falls_back() or _latent_moe_falls_back(self):
        return FallbackReason.OUT_OF_CONTRACT
    from sglang.srt.models import kimi_k3

    num_tokens, hidden_size = hidden_states.shape
    latent_dim = self.moe_hidden_size
    latent_numel = num_tokens * latent_dim
    with kimi_k3.use_symmetric_memory(
        kimi_k3.get_parallel().tp_group, disabled=not kimi_k3.is_allocation_symmetric()
    ):
        buf = hidden_states.new_empty(latent_numel + num_tokens * hidden_size)
    experts = self.experts
    out = kimi_k3_latent_moe(
        hidden_states=hidden_states,
        front_weight=self._front_w,
        correction_bias=self.gate.e_score_correction_bias,
        w1=experts.w13_weight,
        w1_scale=experts.w13_weight_scale,
        w2=experts.w2_weight,
        w2_scale=experts.w2_weight_scale,
        shared_w2=self.shared_experts.down_proj.weight,
        topk=self.topk.topk_config.top_k,
        latent=buf[:latent_numel].view(num_tokens, latent_dim),
        shared=buf[latent_numel:].view(num_tokens, hidden_size),
    )
    if isinstance(out, FallbackReason):
        return out
    buf = self._reduce_latent_and_shared(buf)
    latent = self._latent_norm(buf[:latent_numel].view(num_tokens, latent_dim))
    up, _ = self.routed_expert_up_proj(latent)
    return kimi_k3._add3(up, buf[latent_numel:].view(num_tokens, hidden_size), prefix_sum, prefetch_bc=True)


def _latent_moe_falls_back(moe) -> bool:
    """Whether ``moe`` runs other than as kimi_k3_latent_moe takes it: the bf16 fused front of
    [shared gate_up | router | latent down], sigmoid routing with a bias over every expert, no
    scaling, SiTU 4 / 25 experts of MXFP4 weights on aiter's a8w4 path, a bf16 shared down
    projection, and the single all-reduce of both partial sums."""
    from sglang.srt.layers.communication import k3_ar_fusion
    from sglang.srt.runtime_context import get_parallel

    experts, shared = moe.experts, moe.shared_experts
    return (
        not moe._eligible_for_fused_front
        or moe._front_fp32
        or moe._front_w is None
        or tuple(moe._front_sizes)
        != (2 * shared.down_proj.weight.shape[1], experts.num_experts, moe.moe_hidden_size)
        or k3_ar_fusion.enabled()
        # experts on a8w4 hold the gate-up-interleaved layout the op takes
        or not _a8w4()
        or get_parallel().moe_ep_size != 1
        or moe.num_shared_experts is None
        or type(shared.act_fn).__name__ != "SituAndMul"
        or (shared.act_fn.beta, shared.act_fn.linear_beta) != (4.0, 25.0)
        or shared.down_proj.weight.dtype != moe._front_w.dtype
        or _topk_falls_back(moe.topk)
        or _experts_fall_back(experts)
        or moe.routed_expert_norm is None
        or moe.routed_expert_up_proj is None
        or type(moe.routed_expert_up_proj.quant_method).__name__ != "UnquantizedLinearMethod"
    )


def _a8w4() -> bool:
    from sglang.srt.layers.quantization import mxfp4

    return mxfp4._aiter_situ_uses_gu_interleaved_weights()


def _topk_falls_back(topk) -> bool:
    """Whether ``topk`` routes other than by sigmoid score plus the bias over one group, its
    weights renormalized and unscaled."""
    config = topk.topk_config
    return (
        not config.use_grouped_topk
        or (config.num_expert_group or 1) != 1
        or (config.topk_group or 1) != 1
        or config.custom_routing_function is not None
        or config.scoring_func != "sigmoid"
        or config.correction_bias is None
        or not config.renormalize
        or config.num_fused_shared_experts != 0
        or config.routed_scaling_factor not in (None, 1.0)
        or config.apply_routed_scaling_factor_on_output
        or topk.waterfill_balancer is not None
        or topk.enable_waterfill
    )


def _experts_fall_back(experts) -> bool:
    """Whether ``experts`` run other than as SiTU on aiter's fused_moe over MXFP4 weights at
    sglang's standard dispatch."""
    if (
        type(experts).__name__ != "FusedMoE"
        or type(experts.quant_method).__name__ != "Mxfp4MoEMethod"
        or type(experts.dispatcher).__name__ != "StandardDispatcher"
        or experts.reduce_results
    ):
        return True
    method, config = experts.quant_method, experts.moe_runner_config
    runner = getattr(method, "runner", None)
    return (
        runner is None
        or not runner.runner_backend.is_aiter()
        or config.activation != "situ"
        or (config.gemm1_alpha, config.gemm1_clamp_limit) != (4.0, 25.0)
        or config.gate_up_interleaved
        or config.apply_router_weight_on_input
        or config.no_combine
        or method.with_bias
        # the op takes the experts at their sizes, unpadded
        or getattr(method, "hidden_pad", 0)
        or getattr(method, "intermediate_pad", 0)
    )


def _forward_absorb_rocm_core(
    self,
    q_pe,
    k_pe,
    q_nope_out,
    k_nope,
    forward_batch,
    zero_allocator,
    positions,
    topk_indices,
    llama_4_scaling,
    q_nope_unabsorbed=None,
):
    """The absorbed MLA core of a Kimi-K3 layer on a plain attention backend (its last branch):
    sglang's attention, then the value absorption and the output gate as one op in place of
    rocm_absorb_v_bmm and the o_proj wrapper's gate, then o_proj."""
    if _absorb_core_falls_back(self, forward_batch, llama_4_scaling, q_nope_unabsorbed):
        return FallbackReason.OUT_OF_CONTRACT
    import torch

    tokens, heads, rank = q_nope_out.shape[0], self.num_local_heads, self.kv_lora_rank
    vc_weight = self.w_vc.transpose(1, 2)
    stand_ins = {
        "latent": q_nope_out.new_empty((tokens, heads, rank)),
        "gate": q_nope_out.new_empty((tokens, heads * vc_weight.shape[1])),
    }
    reason = kimi_k3_mla_vc_output_gate.fallback_reason(**stand_ins, vc_weight=vc_weight)
    if reason is not None:
        return reason
    q = torch.cat([q_nope_out, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe], dim=-1)
    latent = self.attn_mqa(q, k, k_nope, forward_batch, save_kv_cache=True).view(-1, heads, rank)
    # the o_proj wrapper gates only while _gate_hidden_states is set: the op takes the gate
    gate = self._compute_output_gate(self._gate_hidden_states)
    self._gate_hidden_states = None
    gated = kimi_k3_mla_vc_output_gate(latent=latent, vc_weight=vc_weight, gate=gate)
    if isinstance(gated, FallbackReason):
        raise RuntimeError(f"kimi_k3_mla_vc_output_gate declined after the gate: {gated}")
    output, _ = self.o_proj(gated)
    return output


def _absorb_core_falls_back(attn, forward_batch, llama_4_scaling, q_nope_unabsorbed) -> bool:
    """Whether ``attn`` runs a branch of forward_absorb_rocm_core other than the plain one
    (cat, attention, value absorption on bf16 weights, gated o_proj), or no gate is pending."""
    import torch
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm
    from sglang.srt.models.deepseek_common.utils import FORWARD_ABSORB_CORE_ATTENTION_BACKENDS

    return (
        attn.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
        or attn.dcp_size > 1
        or attn.is_dcp_mla_decode_phase(forward_batch)
        or attn._skip_rope_for_aiter_fused_mla()
        or llama_4_scaling is not None
        or q_nope_unabsorbed is not None
        or attn.next_skip_topk is not None
        or attn.use_deep_gemm_bmm
        or forward_mla_rocm._SGLANG_EXPERIMENTAL_LORA_OPTI
        or forward_mla_rocm.is_kv_b_lora_active(attn)
        or forward_mla_rocm.is_in_tc_piecewise_cuda_graph()
        or not (attn.w_scale is None or (isinstance(attn.w_scale, float) and attn.w_scale == 1.0))
        # rocm_absorb_v_bmm quantizes o_proj's input for a quantized o_proj
        or attn.o_proj.weight.dtype != torch.bfloat16
        or not getattr(attn, "use_output_gate", False)
        or attn._gate_hidden_states is None
    )


def _forward_normal_core(self, q, k, v, forward_batch, gate=None, key_value_num_tokens=None):
    """A Kimi-K3 prefill without cached prefix on the triton backend: the attention and the
    output gate as one op in place of attn_mha and the o_proj wrapper's gate, then o_proj."""
    if gate is not None or key_value_num_tokens is not None or _prefill_falls_back(self, forward_batch):
        return FallbackReason.OUT_OF_CONTRACT
    import torch
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
        resolve_attn_backend,
    )

    heads, tokens = self.num_local_heads, q.shape[0]
    qo_indptr = resolve_attn_backend(forward_batch).forward_metadata.qo_indptr
    q, k = q.view(tokens, heads, -1), k.view(tokens, heads, -1)
    v = v.view(tokens, heads, -1)
    arguments = {
        "q": q,
        "k": k,
        "sm_scale": self.attn_mha.scaling,
        "max_seq_len": max(forward_batch.extend_seq_lens_cpu),
    }
    stand_ins = {
        "v": v.new_empty(v.shape),
        "gate": q.new_empty((tokens, heads * v.shape[-1])),
        "qo_indptr": qo_indptr.new_empty(qo_indptr.shape, dtype=torch.int32),
    }
    reason = kimi_k3_mla_prefill_gate.fallback_reason(**stand_ins, **arguments)
    if reason is not None:
        return reason
    gate = self._compute_output_gate(self._gate_hidden_states)
    self._gate_hidden_states = None
    gated = kimi_k3_mla_prefill_gate(
        v=v.contiguous(), gate=gate, qo_indptr=qo_indptr.to(torch.int32), **arguments
    )
    if isinstance(gated, FallbackReason):
        raise RuntimeError(f"kimi_k3_mla_prefill_gate declined after the gate: {gated}")
    output, _ = self.o_proj(gated)
    return output


def _prefill_falls_back(attn, forward_batch) -> bool:
    """Whether the call is other than a plain causal prefill without cached prefix on the
    triton backend, with an output gate pending."""
    import torch
    from sglang.srt.lora.deepseek_mla_correction import is_kv_b_lora_active
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
        resolve_attn_backend,
    )

    mode, layer = forward_batch.forward_mode, attn.attn_mha
    backend = resolve_attn_backend(forward_batch)
    return (
        not mode.is_extend_without_speculative()
        or torch.cuda.is_current_stream_capturing()
        or type(backend).__name__ != "TritonAttnBackend"
        or backend.enable_deterministic
        or backend.dcp_size > 1
        or forward_batch.extend_seq_lens_cpu is None
        or any(forward_batch.extend_prefix_lens_cpu or ())
        or getattr(forward_batch, "mha_one_shot", None)
        or getattr(forward_batch, "_attn_output", None) is not None
        or layer.logit_cap
        or (layer.sliding_window_size is not None and layer.sliding_window_size > -1)
        or layer.is_cross_attention
        or is_kv_b_lora_active(attn)
        or not getattr(attn, "use_output_gate", False)
        or attn._gate_hidden_states is None
        or attn.o_proj.weight.dtype != torch.bfloat16
    )


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        "sglang.srt.models.kimi_k3.KimiK3MoE._forward_fused",
        versions=("0.5.22", "0.5.23"),
        replace=_forward_fused,
    )
    plugin.wrap(
        "sglang.srt.models.kimi_k3.KimiK3MLAAttention.forward_absorb_rocm_core",
        versions=("0.5.22", "0.5.23"),
        replace=_forward_absorb_rocm_core,
    )
    plugin.wrap(
        "sglang.srt.models.kimi_k3.KimiK3MLAAttention.forward_normal_core",
        versions=("0.5.22", "0.5.23"),
        replace=_forward_normal_core,
    )
