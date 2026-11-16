# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models import moe_blocks
from aklr.plugins.sglang.models.kimi_k3.contract import (
    KIMI_K3_KDA_VERIFY,
    KIMI_K3_LATENT_MOE,
    KIMI_K3_MLA_PREFILL_GATE,
    KIMI_K3_MLA_VC_OUTPUT_GATE,
)

__all__ = [
    "kimi_k3_kda_verify",
    "kimi_k3_latent_moe",
    "kimi_k3_mla_prefill_gate",
    "kimi_k3_mla_vc_output_gate",
]


@api(KIMI_K3_LATENT_MOE)
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


@api(KIMI_K3_MLA_VC_OUTPUT_GATE)
def kimi_k3_mla_vc_output_gate(latent, vc_weight, gate, output=None): ...


@api(KIMI_K3_MLA_PREFILL_GATE)
def kimi_k3_mla_prefill_gate(q, k, v, gate, qo_indptr, sm_scale=192**-0.5, max_seq_len=8192, output=None): ...


@api(KIMI_K3_KDA_VERIFY)
def kimi_k3_kda_verify(
    projected_qkvg,
    projected_bfa,
    f_b_weight,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    conv_state,
    kda_state,
    conv_windows,
    ring_raw_v,
    ring_raw_k,
    ring_gate,
    ring_beta,
    cache_indices,
    window_indices,
    lower_bound=-5.0,
    eps=1e-5,
    scale=128**-0.5,
): ...


@sglang_route("sglang.srt.models.kimi_k3.KimiK3MoE._forward_fused", versions=("0.5.22",))
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


@sglang_route("sglang.srt.models.kimi_k3.KimiK3MLAAttention.forward_absorb_rocm_core", versions=("0.5.22",))
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


@sglang_route("sglang.srt.models.kimi_k3.KimiK3MLAAttention.forward_normal_core", versions=("0.5.22",))
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


@sglang_route("sglang.srt.models.kimi_k3.KimiK3DeltaAttention.forward", versions=("0.5.22",))
def _kda_forward(self, hidden_states, positions, forward_batch, zero_allocator):
    """A KDA layer's speculative verify of dense chains with ReplaySSM rings, graph capture too:
    sglang's two input projections, the op in place of the f_b projection, the conv update,
    the recurrence and o_norm, then o_proj."""
    if not forward_batch.forward_mode.is_target_verify() or _kda_layer_falls_back(self, hidden_states):
        return FallbackReason.OUT_OF_CONTRACT
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend().linear_attn_backend
    if _kda_verify_falls_back(self, backend, forward_batch, hidden_states):
        return FallbackReason.OUT_OF_CONTRACT
    rows, steps = hidden_states.shape[0], forward_batch.spec_info.draft_token_num
    requests = rows // steps
    cache = backend.req_to_token_pool.mamba2_layer_cache(self.attn.layer_id)
    arguments = {
        "f_b_weight": self._bfa_f_b_w,
        "conv_weight": self.attn.conv_weights,
        "a_log": self.A_log.view(-1),
        "dt_bias": self.dt_bias,
        "norm_weight": self.o_norm.weight,
        "conv_state": cache.conv[0],
        "kda_state": cache.temporal,
        "conv_windows": cache.intermediate_conv_window[0],
        "ring_raw_v": cache.replayssm_rawv,
        "ring_raw_k": cache.replayssm_rawk,
        "ring_gate": cache.replayssm_g,
        "ring_beta": cache.replayssm_beta,
        "cache_indices": backend.forward_metadata.mamba_cache_indices[:requests],
        # select_verify_intermediate_state_indices' rows without pp_spec_stable_rows
        "window_indices": backend.verify_intermediate_state_indices[:requests],
        "lower_bound": self.attn.lower_bound,
        "eps": self.o_norm.eps,
        "scale": self.attn.head_k_dim**-0.5,
    }
    stand_ins = {
        "projected_qkvg": hidden_states.new_empty((requests, steps, 4 * self.attn.v_dim)),
        "projected_bfa": hidden_states.new_empty((requests, steps, self._bfa_w.shape[0])),
    }
    reason = kimi_k3_kda_verify.fallback_reason(**stand_ins, **arguments)
    if reason is not None:
        return reason
    from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

    qkvg, _ = self.fused_qkvg_proj(hidden_states)
    bfa = kimi_k3_tiny_gemm(hidden_states, self._bfa_w)
    normalized = kimi_k3_kda_verify(
        projected_qkvg=qkvg.view(requests, steps, -1),
        projected_bfa=bfa.view(requests, steps, -1),
        **arguments,
    )
    if isinstance(normalized, FallbackReason):
        raise RuntimeError(f"kimi_k3_kda_verify declined after the projections: {normalized}")
    return self.o_proj(normalized.view(rows, -1))[0]


def _kda_layer_falls_back(layer, hidden_states) -> bool:
    """Whether ``layer`` computes other than the op's chain: other than the full-rank gate's
    separate [f_a | b] projection (without it, or the one-GEMM in-projection at these rows), a
    convolution bias, no gate lower bound, o_norm's gate fused into the recurrence, o_proj's
    all-reduce fusion, or LoRA."""
    from sglang.srt.runtime_context import get_lora

    rows = hidden_states.shape[0]
    return (
        not layer.use_full_rank_gate
        or layer._bfa_w is None
        or (getattr(layer, "_qkvgbfa_sizes", None) is not None and 0 < rows <= layer._qkvgbfa_bs_limit)
        or layer._kda_fused_decode_ready
        or layer.all_reduce_fusion
        or layer.attn.bias is not None
        or layer.attn.lower_bound is None
        or layer.attn.activation != "silu"
        or layer.o_norm.activation != "sigmoid"
        or layer.o_norm.bias is not None
        or get_lora().enable_lora
    )


def _kda_verify_falls_back(layer, backend, forward_batch, hidden_states) -> bool:
    """Whether the verify is other than a static dense chain of draft_token_num rows per request
    on the KDA backend's triton recurrence with ReplaySSM rings: a tree or ragged layout, the
    snapshot scratch, fused accept, the fused chain kernel, pipeline-stable scratch rows, or a
    piecewise graph."""
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
    from sglang.srt.layers.attention.linear.utils import pp_spec_stable_rows_enabled
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    if type(backend) is not KDAAttnBackend:
        return True
    spec, metadata = forward_batch.spec_info, backend.forward_metadata
    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.attn.layer_id)
    steps = spec.draft_token_num
    return (
        spec.ragged_verify_layout is not None
        or metadata.retrieve_next_token is not None
        or metadata.retrieve_parent_token is not None
        or getattr(cache, "replayssm_rawv", None) is None
        or getattr(cache, "intermediate_ssm", None) is not None
        or getattr(backend, "accept_lens_pool", None) is not None
        or backend._fused_chain_verify_fn is not None
        or type(backend.kernel_dispatcher.verify_kernel) is not TritonKDAKernel
        or pp_spec_stable_rows_enabled()
        or get_tc_piecewise_forward_context() is not None
        or hidden_states.shape[0] % steps != 0
    )
