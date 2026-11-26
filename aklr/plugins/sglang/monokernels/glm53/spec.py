# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 (FP8)'s monokernel spec (RFC sgl-project/sglang#43949). In a routed decode or MTP verify step, each
decoder layer is one launch: ``glm53_moe_layer`` (MoE, reading the previous top-k), ``glm53_moe_index_layer`` (MoE,
selecting its own) or ``glm53_dense_index_layer``; the embedding is ``glm53_embed``. A layer at a width its op does
not serve, the final norm and every other step run SGLang's code. The logits processor's ``_get_logits`` runs as
``glm53_head`` in every step (``_Head``)."""

from __future__ import annotations

import functools

import torch
from sglang.kernels.plugin_kernels.amd.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from aklr.plugins.sglang.monokernels.glm53 import HF_CONFIG
from aklr.plugins.sglang.monokernels.glm53.reference import INDEX_DIM, INDEX_PAGE

# the rows a routed step may have (MTP's verify: 4 rows a request); SGLang's forward runs the others
ROWS = tuple(range(1, 9))
TP_SIZES = (8,)
# one workgroup a CU on MI355X
RESIDENT_BLOCKS = 256
# an MLA cache row: the latent, then k_pe
KV_ROW = HF_CONFIG["kv_lora_rank"] + HF_CONFIG["qk_rope_head_dim"]
# an index-K key's bytes: its E4M3 values, then its f32 scale
INDEX_ROW = INDEX_DIM + 4
# the indexer layers' top-k, which the layers after them read
INDEX_TOPK = HF_CONFIG["index_topk"]
# the keys an indexer layer's row scores at most (the contract's): GLM-5.3's longest context
INDEX_ROW_KEYS = 2**20
# the stand-in pool's rows: at least the top-k's, so its calls are in the contract
_STAND_IN_SLOTS = 2 * INDEX_TOPK

# SGLang reads the kind alone, and wants a bound with it. The contracts bound each launch against its reference
# (a layer's out at 25 dB at the least); a launch and SGLang's own layer, both within that, are this close
NUMERICS = Numerics(NumericsKind.TOLERANCE, {"layer.out.rel": 0.12})


def config_disable_reasons(cfg, hf_config) -> list[str]:
    """Before load: the checks ``Requirements`` cannot state."""
    reasons = [
        f"needs GLM-5.3's {key} = {want!r}, got {getattr(hf_config, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(hf_config, key, None) != want
    ]
    quant = getattr(hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "fp8" or list(quant.get("weight_block_size") or ()) != [128, 128]:
        reasons.append("needs the FP8 block-128 checkpoint")
    # "auto" is the model's bf16 but for a checkpoint that quantizes the cache (scaled FP8)
    if cfg.kv_cache_dtype == "auto" and str(quant.get("kv_cache_quant_algo") or "").upper() == "FP8":
        reasons.append(
            "--kv-cache-dtype auto resolves to the checkpoint's scaled FP8 cache: needs fp8_e4m3 or bfloat16"
        )
    # a verify row attends the keys up to its own position: one draft chain a request
    if cfg.speculative_algorithm is not None and (
        cfg.speculative_eagle_topk != 1 or cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
    ):
        reasons.append("MTP verifies one draft chain: topk 1, draft tokens = steps + 1")
    return reasons


# the ops of the covered layers' kinds, and those selecting their own top-k
MOE, INDEX, DENSE = "glm53_moe_layer", "glm53_moe_index_layer", "glm53_dense_index_layer"
INDEXED = (INDEX, DENSE)
HEAD = "glm53_head"
EMBED = "glm53_embed"


def _covered(model) -> dict:
    """The layers a launch replaces, with the op that runs each: every layer, an MoE one reading the previous
    indexer layer's top-k (MOE) or selecting its own (INDEX), or a dense one selecting its own (DENSE)."""
    covered = {}
    for i, layer in enumerate(model.model.layers):
        indexer, moe = layer.self_attn.indexer is not None, hasattr(layer.mlp, "experts")
        if moe or indexer:
            covered[i] = (INDEX if indexer else MOE) if moe else DENSE
    return covered


def _shuffled(linear) -> bool:
    # SGLang marks a block-FP8 linear it rewrote into AITER's (16, 16) layout
    return bool(getattr(linear, "aiter_bpreshuffled", False) or getattr(linear.weight, "is_shuffled", False))


def _fp8_reasons(i: int, owner, names, what) -> list[str]:
    reasons = []
    for name in names:
        linear = getattr(owner, name, None)
        if (
            linear is None
            or linear.weight.dtype != torch.float8_e4m3fn
            or getattr(linear, "weight_scale_inv", None) is None
        ):
            reasons.append(f"layer {i}: needs {what}{name} as block FP8")
        elif not _shuffled(linear):
            reasons.append(f"layer {i}: needs {what}{name} in AITER's (16, 16) layout")
    return reasons


def layer_reasons(i: int, layer, kind) -> list[str]:
    """Why layer i's tensors are not the ones its op's scenario takes."""
    attn = layer.self_attn
    reasons = _fp8_reasons(i, attn, ("fused_qkv_a_proj_with_mqa", "q_b_proj", "o_proj"), "")
    if attn.w_kc.dtype != torch.float8_e4m3fn or attn.w_scale is None or attn.w_scale.numel() != 1:
        reasons.append(f"layer {i}: needs w_kc and w_vc as per-tensor FP8 with a scalar w_scale")
    if kind == DENSE:
        return reasons + _fp8_reasons(i, layer.mlp, ("gate_up_proj", "down_proj"), "the MLP's ")
    experts = layer.mlp.experts
    if experts.w13_weight.dtype != torch.float8_e4m3fn or not getattr(
        experts.w13_weight, "is_shuffled", False
    ):
        reasons.append(f"layer {i}: needs the experts as block FP8 in AITER's (16, 16) layout")
    router = layer.mlp.gate.weight
    if router.dtype != torch.bfloat16 and not torch.equal(router.to(torch.bfloat16).to(router.dtype), router):
        reasons.append(f"layer {i}: needs the router's weights exactly representable in bf16")
    return reasons


def _attention_args(layer) -> dict:
    """Layer's attention tensors by its op's names: SGLang's own, but w_kc and w_vc row-major (SGLang keeps
    them transposed for its bmm)."""
    attn = layer.self_attn
    qkv_a, q_b, o = attn.fused_qkv_a_proj_with_mqa, attn.q_b_proj, attn.o_proj
    return {
        "input_norm_weight": layer.input_layernorm.weight,
        "qkv_a_weight": qkv_a.weight,
        "qkv_a_scale": qkv_a.weight_scale_inv,
        "q_a_norm_weight": attn.q_a_layernorm.weight,
        "kv_a_norm_weight": attn.kv_a_layernorm.weight,
        "q_b_weight": q_b.weight,
        "q_b_scale": q_b.weight_scale_inv,
        "w_kc": attn.w_kc.contiguous(),
        "w_vc": attn.w_vc.contiguous(),
        "w_scale": attn.w_scale.reshape(1),
        "o_weight": o.weight,
        "o_scale": o.weight_scale_inv,
        "post_norm_weight": layer.post_attention_layernorm.weight,
    }


def _moe_args(layer, hf_config) -> dict:
    """Layer's MoE tensors and attrs by its op's names, the router in bf16."""
    moe = layer.mlp
    return {
        "router_weight": moe.gate.weight.to(torch.bfloat16),
        "correction_bias": moe.gate.e_score_correction_bias,
        "w1": moe.experts.w13_weight,
        "w1_scale": moe.experts.w13_weight_scale_inv,
        "w2": moe.experts.w2_weight,
        "w2_scale": moe.experts.w2_weight_scale_inv,
        "topk": hf_config.num_experts_per_tok,
        "routed_scaling_factor": hf_config.routed_scaling_factor,
    }


def _dense_args(layer) -> dict:
    """Layer's dense MLP tensors by glm53_dense_index_layer's names."""
    gate_up, down = layer.mlp.gate_up_proj, layer.mlp.down_proj
    return {
        "gate_up_weight": gate_up.weight,
        "gate_up_scale": gate_up.weight_scale_inv,
        "down_weight": down.weight,
        "down_scale": down.weight_scale_inv,
    }


def _cos_sin(rotary) -> torch.Tensor:
    """The rotary's f32 table [positions, 64]: the cosines, then the sines (SGLang's
    ``cos_sin_cache``, or AITER's ``cos_cache`` and ``sin_cache`` joined)."""
    cache = getattr(rotary, "cos_sin_cache", None)
    if cache is None:
        cos, sin = rotary.cos_cache, rotary.sin_cache
        cache = torch.cat([cos.reshape(cos.shape[0], -1), sin.reshape(sin.shape[0], -1)], dim=-1)
    return cache.float().contiguous()


def index_reasons(i: int, layer) -> list[str]:
    """Why layer i's indexer is not the one glm53_moe_index_layer takes: SGLang's gfx950 indexer's."""
    ix = layer.self_attn.indexer
    reasons = []
    for name in ("wq_b", "wk"):
        linear = getattr(ix, name, None)
        if linear is None or linear.weight.dtype != torch.float8_e4m3fn or not _shuffled(linear):
            reasons.append(f"layer {i}: needs the indexer's {name} as block FP8 in AITER's (16, 16) layout")
    weights_proj = getattr(ix, "weights_proj", None)
    if weights_proj is None or weights_proj.weight.dtype != torch.bfloat16:
        reasons.append(f"layer {i}: needs the indexer's weights_proj in bf16")
    if getattr(ix.k_norm, "bias", None) is None or ix.k_norm.weight.dtype != torch.bfloat16:
        reasons.append(f"layer {i}: needs the indexer's k_norm as a bf16 LayerNorm")
    return reasons


def _index_args(layer) -> dict:
    """Layer's indexer tensors by its op's names."""
    ix = layer.self_attn.indexer
    return {
        "wq_b_weight": ix.wq_b.weight,
        "wq_b_scale": ix.wq_b.weight_scale_inv,
        "wk_weight": ix.wk.weight,
        "wk_scale": ix.wk.weight_scale_inv,
        "weights_proj_weight": ix.weights_proj.weight,
        "k_norm_weight": ix.k_norm.weight,
        "k_norm_bias": ix.k_norm.bias,
        "index_eps": ix.k_norm.variance_epsilon,
    }


@functools.cache  # one copy for the layers sharing a rotary
def _index_rope(rotary) -> dict:
    """The indexer's rotary tables, bf16 [positions, 32] each, as SGLang's gfx950 indexer takes them."""
    cos, sin = getattr(rotary, "cos_cache", None), getattr(rotary, "sin_cache", None)
    if cos is None or sin is None:
        cos, sin = rotary.cos_sin_cache.chunk(2, dim=-1)
    return {
        "index_cos": cos.reshape(cos.shape[0], -1).to(torch.bfloat16).contiguous(),
        "index_sin": sin.reshape(sin.shape[0], -1).to(torch.bfloat16).contiguous(),
    }


def shared_args(first) -> dict:
    """The arguments alike for every covered layer, read off the first: the rotary table, the TP size and the
    scalars."""
    from sglang.srt.runtime_context import get_parallel

    return {
        "cos_sin_cache": _cos_sin(first.self_attn.rotary_emb),
        "tp_size": get_parallel().tp_size,
        "eps": first.input_layernorm.variance_epsilon,
        "sm_scale": first.self_attn.scaling,
    }


def layer_call_args(layer, kind, hf_config) -> dict:
    """Layer's tensors and attrs, those alike for every layer (``shared_args``) aside."""
    args = _attention_args(layer)
    args |= _dense_args(layer) if kind == DENSE else _moe_args(layer, hf_config)
    if kind in INDEXED:
        args |= _index_args(layer) | _index_rope(layer.self_attn.indexer.rotary_emb)
    return args


def _kv_dtype() -> torch.dtype:
    """The MLA cache's dtype, --kv-cache-dtype as SGLang resolved it: E4M3, else bf16 (a bf16 model's "auto")."""
    from sglang.srt.runtime_context import get_model

    return torch.float8_e4m3fn if get_model().kv_cache_dtype == "fp8_e4m3" else torch.bfloat16


def stand_in_call(kind, rows: int, device) -> dict:
    """A call's per-step tensors at ``rows``: caches of its own, no slot attended (MOE) or one key a row,
    page 0's first, and the top-k's buffer (INDEXED)."""
    call = {
        "hidden_states": torch.zeros(rows, HF_CONFIG["hidden_size"], dtype=torch.bfloat16, device=device),
        "positions": torch.zeros(rows, dtype=torch.int64, device=device),
        "cache_locations": torch.arange(rows, dtype=torch.int64, device=device),
        "kv_cache": torch.zeros(_STAND_IN_SLOTS, KV_ROW, dtype=_kv_dtype(), device=device),
    }
    if kind == MOE:
        return call | {"topk_indices": torch.full((rows, INDEX_TOPK), -1, dtype=torch.int32, device=device)}
    pages = _STAND_IN_SLOTS // INDEX_PAGE
    return call | {
        "index_k_cache": torch.zeros(pages, INDEX_PAGE * INDEX_ROW, dtype=torch.float8_e4m3fn, device=device),
        "page_table": torch.zeros(rows, 1, dtype=torch.int32, device=device),
        "seq_lens": torch.ones(rows, dtype=torch.int32, device=device),
        "topk_indices": topk_buffer(rows, device),
    }


def topk_buffer(rows: int, device) -> torch.Tensor:
    # the INDEXED ops' top-k output: no input carries its index_topk axis, so the caller passes it
    return torch.empty(rows, INDEX_TOPK, dtype=torch.int32, device=device)


def api_of(kind):
    from aklr.plugins.sglang.monokernels.glm53 import api

    return getattr(api, kind)


def served_widths(op, calls: dict) -> frozenset:
    """The widths (``calls``' keys) at which AKLR serves ``op``, each run once on every rank outside the capture."""
    served = frozenset(rows for rows, call in calls.items() if op.fallback_reason(**call) is None)
    for rows in sorted(served):
        op(**calls[rows])
    return served


def launch(
    kind, args, shared, layer_id, positions, slots, hidden_states, forward_batch, topk_indices, step=None
):
    """Layer ``layer_id``'s launch: (the stream leaving it, the top-k the layers after it read). MOE reads
    ``topk_indices``; an INDEXED kind writes its own into it, from the step's indexer page table and lengths, made
    once a step in ``step``."""
    from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool

    from aklr.core.runtime.dispatch import FallbackReason

    pool = get_token_to_kv_pool()
    call = {
        "hidden_states": hidden_states,
        "positions": positions,
        "cache_locations": slots,
        "kv_cache": pool.get_key_buffer(layer_id).view(-1, KV_ROW),
        "topk_indices": topk_indices,
        **args,
        **shared,
    }
    if kind in INDEXED:
        step = {} if step is None else step
        if "page_table" not in step:
            metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
            # a row each token: SGLang expands a verify request's rows
            step["page_table"] = metadata.get_page_table_64().contiguous()
            step["seq_lens"] = metadata.get_seqlens_expanded()
        index_k = pool.get_index_k_with_scale_buffer(layer_id=layer_id).view(torch.float8_e4m3fn)
        call |= {
            "index_k_cache": index_k.view(-1, INDEX_PAGE * INDEX_ROW),
            "page_table": step["page_table"],
            "seq_lens": step["seq_lens"],
        }
    out = api_of(kind)(**call)
    if isinstance(out, FallbackReason):
        raise RuntimeError(f"GLM-5.3 monokernel: layer {layer_id}'s {kind} fell back: {out.value}")
    return out if kind in INDEXED else (out, topk_indices)


def _unserved(model, covered, hf_config, shared, widths) -> dict:
    """Each (kind, width) at which AKLR serves the kind's op no kernel, with its FallbackReason."""
    reasons = {}
    for kind in sorted(set(covered.values())):
        layer = model.model.layers[min(i for i, k in covered.items() if k == kind)]
        args = layer_call_args(layer, kind, hf_config) | shared
        device = shared["cos_sin_cache"].device
        for rows in widths:
            reason = api_of(kind).fallback_reason(**stand_in_call(kind, rows, device), **args)
            if reason is not None:
                reasons[kind, rows] = reason
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """After load: the covered layers' layouts, the served context, and a served kernel at some width."""
    from sglang.srt.runtime_context import process_model_config

    layers, covered = model.model.layers, _covered(model)
    if not covered:
        return ["no layer with an MoE or an indexer"]
    first_moe = min((i for i, kind in covered.items() if kind == MOE), default=None)
    if first_moe is not None and all(layer.self_attn.indexer is None for layer in layers[:first_moe]):
        return [f"layer {first_moe} reads the previous indexer layer's top-k, and no layer before it has one"]
    reasons = [r for i, kind in covered.items() for r in layer_reasons(i, layers[i], kind)]
    reasons += [r for i, kind in covered.items() if kind in INDEXED for r in index_reasons(i, layers[i])]
    context_len = process_model_config().context_len
    if context_len > INDEX_ROW_KEYS:
        reasons.append(f"needs a served context of at most {INDEX_ROW_KEYS} keys, got {context_len}")
    if reasons:
        return reasons
    unserved = _unserved(model, covered, hf_config, shared_args(layers[min(covered)]), ROWS)
    if len(unserved) < len(set(covered.values())) * len(ROWS):
        return []
    return [
        f"{kind} at {rows} rows: no AKLR kernel serves it ({reason.value}; AKLR_MIN_SPEEDUP=0 serves unmeasured "
        "kernels)"
        for (kind, rows), reason in unserved.items()
    ]


class _Built:
    """The covered layers' kinds and tensors, the widths each kind's launch serves, and the forward SGLang
    routes its decode and verify steps to."""

    def __init__(self, lm, layers: dict, served: dict, shared: dict, embed_widths: frozenset):
        self.lm, self.layers, self.served, self.shared = lm, layers, served, shared
        self.embed_widths = embed_widths
        # rows -> the INDEXED launches' top-k, which the layers after each read
        device = shared["cos_sin_cache"].device
        self.topk = {rows: topk_buffer(rows, device) for rows in set().union(*served.values())}

    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        """``DeepseekV2ForCausalLM.forward`` for a routed decode or verify step."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        lm = self.lm
        if input_embeds is not None:
            return type(lm).forward(lm, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)
        rows = plan.key.rows
        layers = {i: entry for i, entry in self.layers.items() if rows in self.served[entry[0]]}
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden = self._model(layers, input_ids, positions, forward_batch)
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)

    def _model(self, layers, input_ids, positions, forward_batch):
        """``DeepseekV2Model.forward`` on one pipeline stage without TBO, a layer in ``layers`` its launch:
        its input the residual stream entering it, its output the one leaving it."""
        from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.utils import BumpAllocator

        m, device = self.lm.model, positions.device
        hidden = self._embed(input_ids)
        residual_batch.start(forward_batch)
        index_topk = IndexTopKShareState(forward_batch, None)
        # only SGLang's layers take the zero allocators (each a zero fill in the step): none if every layer launches
        zero_allocator = gemm_zero_allocator = None
        if len(layers) < len(m.layers):
            zero_allocator = BumpAllocator(buffer_size=len(m.layers) * 2, dtype=torch.float32, device=device)
            size = getattr(m, "gemm_output_zero_allocator_size", 0)
            gemm_zero_allocator = BumpAllocator(size, dtype=torch.float32, device=device) if size else None
        slots = forward_batch.out_cache_loc.long()
        step = {}
        for i, layer in enumerate(m.layers):
            if i not in layers:
                hidden, topk_indices = layer(
                    positions,
                    hidden,
                    forward_batch,
                    zero_allocator,
                    gemm_zero_allocator,
                    None,
                    prev_topk_indices=index_topk.topk_indices,
                    next_full_attention_layer_id=m.next_full_attention_layer_id.get(i),
                )
                index_topk.update(topk_indices)
                continue
            kind, args = layers[i]
            topk = self.topk[slots.shape[0]] if kind in INDEXED else index_topk.topk_indices
            hidden = residual_batch.fold(hidden, forward_batch)
            out, topk = launch(
                kind, args, self.shared, i, positions, slots, hidden, forward_batch, topk, step
            )
            index_topk.update(topk)
            hidden = residual_batch.set_written(out, forward_batch)
        hidden = residual_batch.complete_output(hidden, forward_batch)
        return residual_batch.final_norm(hidden, forward_batch, m.norm)

    def _embed(self, input_ids):
        """``embed_tokens(input_ids)``: api.glm53_embed's at the widths in ``embed_widths`` while the embedding
        sums over the TP group (no scattered input), else its own."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context
        from sglang.srt.runtime_context import get_parallel

        from aklr.core.runtime.dispatch import FallbackReason

        embed, rows = self.lm.model.embed_tokens, input_ids.shape[0]
        if (
            rows in self.embed_widths
            and input_ids.dim() == 1
            and input_ids.is_contiguous()
            and not get_attn_tp_context().input_scattered
        ):
            weight = embed.weight
            out = torch.empty(rows, weight.shape[1], dtype=weight.dtype, device=weight.device)
            got = api_of(EMBED)(
                input_ids=input_ids, embed_weight=weight, tp_size=get_parallel().tp_size, embeddings=out
            )
            if not isinstance(got, FallbackReason):
                return got
        return embed(input_ids)


def plain_vocab_embedding(embed) -> bool:
    """Whether ``embed`` is sglang's plain vocab-parallel lookup over the TP group, as glm53_embed and
    glm53_nextn_entry take it: no padded or added vocabulary, quantized or LoRA table, or attention TP group."""
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.runtime_context import get_parallel

    weight = getattr(embed, "weight", None)
    return (
        isinstance(embed, VocabParallelEmbedding)
        and isinstance(embed.quant_method, UnquantizedEmbeddingMethod)
        and not hasattr(embed, "set_lora")
        and not embed.use_attn_tp_group
        and embed.tp_size == get_parallel().tp_size
        and embed.num_embeddings_padded == embed.org_vocab_size
        and embed.num_embeddings_per_partition * embed.tp_size == embed.org_vocab_size
        and weight is not None
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    )


def _embed_widths(lm, widths) -> frozenset:
    """The widths at which AKLR serves ``lm``'s embed_tokens (api.glm53_embed); none unless it is a plain lookup."""
    embed = lm.model.embed_tokens
    if not plain_vocab_embedding(embed):
        return frozenset()
    weight = embed.weight
    calls = {
        rows: {
            "input_ids": torch.zeros(rows, dtype=torch.int64, device=weight.device),
            "embed_weight": weight,
            "tp_size": embed.tp_size,
            "embeddings": torch.empty(rows, weight.shape[1], dtype=weight.dtype, device=weight.device),
        }
        for rows in widths
    }
    return served_widths(api_of(EMBED), calls)


def _head_reasons(lm) -> list[str]:
    """Why api.glm53_head cannot take ``lm``'s logits processor's logits: what the processor does beyond it."""
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
    from sglang.srt.runtime_context import get_parallel

    lp, head = lm.logits_processor, lm.lm_head
    reasons = []
    if (
        not lp.do_tensor_parallel_all_gather
        or lp.do_tensor_parallel_all_gather_dp_attn
        or lp.use_attn_tp_group
        or lp.use_tp_lm_head_all_to_all
    ):
        reasons.append("needs the logits gathered over the TP group, without DP attention or a DP LM head")
    if lp.use_fp32_lm_head or lp.use_fp32_lm_head_output:
        reasons.append("needs the LM head in bf16")
    if lp.logit_scale or lp.final_logit_softcapping:
        reasons.append("needs no logit scale or softcapping")
    weight = getattr(head, "weight", None)
    if (
        weight is None
        or weight.dtype != torch.bfloat16
        or hasattr(head, "set_lora")
        or should_apply_lm_head_quant_method(head, getattr(head, "quant_method", None))
    ):
        reasons.append("needs an unquantized bf16 LM head")
    elif weight.shape[0] * get_parallel().tp_size != lp.vocab_size:
        reasons.append("needs the vocabulary unpadded")
    return reasons


def install_head(lm, widths) -> None:
    """api.glm53_head as ``lm``'s logits processor's logits (``_Head``) at the ``widths`` AKLR serves it; nowhere if
    the processor does more than it (``_head_reasons``)."""
    from sglang.srt.runtime_context import get_parallel

    if _head_reasons(lm):
        return
    weight, vocab = lm.lm_head.weight, lm.logits_processor.vocab_size
    calls = {
        rows: {
            "hidden_states": torch.zeros(rows, weight.shape[1], dtype=torch.bfloat16, device=weight.device),
            "lm_head_weight": weight,
            "tp_size": get_parallel().tp_size,
            "logits": torch.empty(rows, vocab, dtype=torch.float32, device=weight.device),
        }
        for rows in widths
    }
    _Head(lm, served_widths(api_of(HEAD), calls))


class _Head:
    """``lm``'s logits processor's logits (its ``_get_logits``) by api.glm53_head at the widths in ``served``, its
    own elsewhere. Every rank makes the same choice: it reads the call's rows and layout alone."""

    def __init__(self, lm, served: frozenset):
        from sglang.srt.runtime_context import get_parallel

        lp = lm.logits_processor
        self.served, self.vocab, self.tp_size = served, lp.vocab_size, get_parallel().tp_size
        self.own, lp._get_logits = lp._get_logits, self._get_logits

    def _get_logits(
        self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True
    ):
        from aklr.core.runtime.dispatch import FallbackReason

        rows = hidden_states.shape[0]
        if rows in self.served and embedding_bias is None and hidden_states.is_contiguous():
            # the logits buffer if it is the call's shape, as _copy_logits_to_buffer takes it
            out = logits_metadata.next_token_logits_buffer if use_logits_buffer else None
            if out is None or tuple(out.shape) != (rows, self.vocab) or not out.is_contiguous():
                out = torch.empty(rows, self.vocab, dtype=torch.float32, device=hidden_states.device)
            logits = api_of(HEAD)(
                hidden_states=hidden_states, lm_head_weight=lm_head.weight, tp_size=self.tp_size, logits=out
            )
            if not isinstance(logits, FallbackReason):
                return logits
        return self.own(hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer)


def _build(model, hf_config, ctx) -> _Built:
    """The covered layers' kinds and tensors, each kind run once at the routed widths AKLR serves it; then the
    head and the embedding."""
    covered = _covered(model)
    shared = shared_args(model.model.layers[min(covered)])
    layers = {
        i: (kind, layer_call_args(model.model.layers[i], kind, hf_config)) for i, kind in covered.items()
    }
    device = shared["cos_sin_cache"].device
    served = {}
    for kind in sorted(set(covered.values())):
        args = next(a for k, a in layers.values() if k == kind) | shared
        served[kind] = served_widths(
            api_of(kind), {w: stand_in_call(kind, w, device) | args for w in ctx.widths}
        )
    # every step's logits, at whatever rows: a verify step's, a decode or draft step's (a row a request)
    install_head(model, ROWS)
    embed_widths = _embed_widths(model, ctx.widths)
    torch.cuda.synchronize()
    return _Built(model, layers, served, shared, embed_widths)


SPEC = MonoSpec(
    name="glm53_aklr",
    devices={"gfx950": DeviceLimits(widths=ROWS, tp_sizes=TP_SIZES, resident_blocks=RESIDENT_BLOCKS)},
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsa"}),
        server_args={
            # the launches write and read an E4M3 or a bf16 MLA cache; SGLang resolves a DSA model's "auto" to
            # bfloat16 on gfx950 before this check
            "kv_cache_dtype": frozenset({"fp8_e4m3", "bfloat16", "auto"}),
            # MTP: the NextN draft's verify steps (NEXTN resolves to EAGLE)
            "speculative_algorithm": frozenset({None, "EAGLE"}),
            # the experts in AITER's shuffled layout
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # HiSparse maps the logical KV slots to device rows; the launches write raw slots
            "enable_hisparse": frozenset({False}),
            # a layer-split pool broadcasts each layer's index keys from its owner; the launches store their own
            "enable_dsa_cache_layer_split": frozenset({False}),
            # the launches record no expert distribution
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            # a layer reading the previous top-k reads KV slots; unfused top-k hands out positions
            "SGLANG_DSA_FUSE_TOPK": True,
            # the launches route with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=config_disable_reasons,
)
