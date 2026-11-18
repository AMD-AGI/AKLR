"""A GPU activity's operator class, from its kernel name.

The first matching rule wins, so the order matters: ``gemm1_a4w4`` is a MoE kernel, not a dense GEMM; a Gemma
RMSNorm (``grouped_gemma_rmsnorm``) is a norm although its name contains ``grouped_gemm``; an attention residual
kernel is not attention; quick-reduce and cross-device reduce are communication. Patterns match whole name parts
where a substring would mislead (``prefill`` contains ``fill``). A name no rule matches is ``other`` and keeps its
full name in every table, so a new kernel is never folded into a wrong class silently.
"""

from __future__ import annotations

import functools
import re

RULES: list[tuple[str, str]] = [
    ("memcpy", r"^(Memcpy|memcpy)|^\[memcpy\]|HtoD|DtoH|DtoD"),
    ("memset", r"^(Memset|memset)|^\[memset\]"),
    (
        "communication",
        r"(?i)all_?reduce|allgather|all_gather|reduce_scatter|reducescatter|alltoall|all_to_all|nccl|"
        r"rccl|cross_device|quickreduce|qr_all|custom_ar|msccl|mori|epdispatch|epcombine|dp_gather|"
        r"two_?shot|one_?shot|deepep|mooncake|lamport|dcp_pack_a2a",
    ),
    # A whole decoder layer (attention, MoE, GEMMs) in one launch: priced by the weights it reads, not split here.
    # K3.1's k31_kda_layer_*_l68 runs 68 KDA layers whole: where it runs, the router, attention-residual and allreduce
    # launches of those layers disappear from the step (92 -> 24 router calls).
    (
        "fused_decoder_layers",
        r"_layers_s\d+_r\d+|monokernel|mono_k\d|k3_mono|megakernel|latency_specialized|"
        r"k31_kda_layer",
    ),
    # Hierarchical-cache transfers between host and device (KV pages, Mamba/KDA state): cache traffic, not the layer.
    ("kv_cache", r"transfer_mamba|transfer_kernel_impl"),
    (
        "norm",
        r"rmsnorm|RMSNorm|layernorm|LayerNorm|layer_norm|rms_norm|_norm_kernel|NormKernel|l2norm|norm_fwd",
    ),
    # Linear-attention pieces whose names would match a later class (gating, a "_fwd_kernel").
    ("linear_attention", r"gdn_gating|recompute_w_u"),
    (
        "router_topk",
        r"topk_softmax|topkGating|topk_gating|grouped_topk|biased_grouped_topk|topk_sigmoid|topk_reg|"
        r"router|gating|route_radix|route_kernel",
    ),
    # K3's a8w4 MoE (gate-up, down, its work items and its reduce) lives in sglang::k3_a8w4.
    (
        "moe",
        r"fused_moe|fmoe|moe_|_moe|moe::|MoeSorting|mfma_moe|opus_moe|flydsl_moe|ck_moe|gemm1_a|gemm2_a|"
        r"grouped_gemm(?!a)|GroupedGemm|expert|moe_align|k3_a8w4::|a8w4_moe",
    ),
    (
        "linear_attention",
        r"chunk_gated_delta|gated_delta|chunk_gla|chunk_kda|kda_|_recompute_w_u|causal_conv1d|"
        r"verify_conv|conv_depthwise|"
        r"fused_recurrent|recurrent|replayssm|mamba|ssm_|selective_scan|gdn|delta_rule|"
        r"short_conv|linear_attn|chunk_fwd_kernel|chunk_scaled_dot|chunk_local_cumsum|solve_tril",
    ),
    # The indexer scores and selects keys; the sparse attention over the selected keys is attention (with DCP, its
    # per-rank partial outputs are merged by LSE: also attention).
    ("attention", r"sparse_mla|sparse_gqa|dcp_lse_combine|^_merge_kernel$"),
    # Kimi-K3's fused latent-MoE tail (RMSNorm, the 3584->7168 up-projection, add3): a skinny GEMM streaming its weight.
    ("gemm", r"latent_tail"),
    # Kimi-K3's attention residuals ("k3_ag_attn_res") are not attention, though their names contain "_attn_".
    ("attention_residual", r"attn_res|_agg_kernel|add3_kernel"),
    (
        "sparse_attention_indexer",
        r"indexer|dsa_gfx950::logits|dsa_gfx950::topk|dsa_topk|dsa_|logits_hist|"
        r"topk_transform|fp8_mqa_logits|mqa_logits|paged_mqa|lightning_indexer|qsa_|fast_topk|"
        r"_compact_kv|dcp_topk|_localize_compact|_index_k_norm",
    ),
    (
        "attention",
        r"mla|flash_attn|fmha|attn_fwd|attention|paged_attention|decode_attention|unified_attn|"
        r"_fwd_kernel|mha_|mqa|gqa|flash_decoding|fa_fwd|fwd_kernel_stage|reduce_attn|"
        r"_attn_|attn_kernel|_prefill_kv|_fa2_|fa3_",
    ),
    ("hyper_connection", r"hc_combine|hc_split|hc_pre|hc_post|hc_mix|mhc|hyper_conn"),
    # Speculative-decoding bookkeeping between the draft and the verify: trees, cache slots, accepted-token commits.
    (
        "spec_decode_bookkeeping",
        r"chain_tree|VerifyTree|verify_commit|draft_cache_locs|draft_extend_lengths|"
        r"alloc_extend|get_last_loc|ngram_hash|replay_state_indices|accept_",
    ),
    # Inductor's matmul templates fuse an activation into the GEMM: a GEMM.
    (
        "gemm",
        r"Cijk_|gemm|gemv|matmul|bmm|wvSplitK|splitk|hipblaslt|wv_splitk|Gemm|skinny|hgemm|f8f8|a8w8|a4w4|"
        r"a16w16|mm_kernel|triton_tem_fused_.*mm",
    ),
    ("rope", r"rope|rotary|OpCachedFwd|kn_entry_\w*sbhd"),
    # A DFlash draft model's grouped convolution: draft-model compute, not the target's linear attention.
    ("draft_model", r"dflash_|dspark_|_draft_conv"),
    ("activation", r"silu|gelu|situ|swiglu|act_and_mul|and_mul|relu|sigmoid_mul"),
    ("quantization", r"quant|float8|fp8|fp4|mxfp|e4m3|e8m0|_scale"),
    (
        "sampling",
        r"sampl|softmax|SoftMax|argmax|ArgMax|top_k|top_p|multinomial|rejection|penalt|logit|draft_topk",
    ),
    ("embedding", r"embedding|Embedding|lm_head|vocab"),
    (
        "kv_cache",
        r"kv_cache|store_kv|set_kv|reshape_and_cache|cache_kernel|kv_buffer|page_first|hicache|"
        r"write_req_to_token|assign_req|copy_kv|kv_indices|transfer_kernel_impl",
    ),
    (
        "elementwise_copy",
        r"elementwise|vectorized|unroll|copy|Copy|FillFunctor|fill_kernel|\bfill|_fill\b|cat_|Cat|"
        r"index|scatter|gather|reduce_kernel|arange|where|compare|cumsum|\bsort|_sort|Sort|"
        r"triton_poi|triton_red|triton_per|rocclr|rocprim|scan_innermost|tensor_kernel_scan|"
        r"compute_cuda_kernel",
    ),
]
_COMPILED = [(name, re.compile(pattern)) for name, pattern in RULES]


# A kernel whose name is only a DSL's default (TileLang names a prim_func ``kernel`` -> ``kernel_kernel``) says nothing
# by itself; where the image's source shows what it is for one deployment, it is attributed there and only there:
# (regex on the point id, regex on the kernel name, class, where it comes from).
ATTRIBUTED: list[tuple[str, str, str, str]] = [
    (
        r"^qwen3\.8next-",
        r"^kernel_kernel$",
        "sparse_attention_indexer",
        "QSA indexer key scoring: TileLang prim_funcs named `kernel` in sglang srt/layers/attention/qsa/mqa.py "
        "(qsa_mqa_decode, qsa_mqa_prefill[_relative], the logits mask), image sglang-rocm v0.5.23-rocm10-mi35x-20261031",
    ),
    # torch.compile's names for the hyper-connection mix (GatedResidual._mix_compute in sglang
    # srt/layers/hyperconnection.py), the path taken at the batch sizes the split-K, FlyDSL and fused hc_mix kernels
    # do not cover: they trade places with _hc_mix_persistent_kernel by batch size.
    (
        r"^qwen3\.8next-",
        r"^triton_tem_fused_div_mm_silu_t_\d+$|^triton_poi_fused_mean_mul_sigmoid_view_\d+$",
        "hyper_connection",
        "GatedResidual._mix_compute (torch.compile), sglang srt/layers/hyperconnection.py",
    ),
]
_ATTRIBUTED = [(re.compile(p), re.compile(n), cls) for p, n, cls, _ in ATTRIBUTED]


@functools.lru_cache(maxsize=None)
def op_class_at(point_id: str, kind: str, name: str) -> str:
    """``op_class``, with the attributions of the point ``point_id`` applied first."""
    for p, n, cls in _ATTRIBUTED:
        if p.search(point_id or "") and n.search(name):
            return cls
    return op_class(kind, name)


@functools.lru_cache(maxsize=None)
def op_class(kind: str, name: str) -> str:
    """Operator class of a GPU activity of ``kind`` (kernel, memcpy, memset) named ``name``."""
    if kind in ("memcpy", "memset"):
        return kind
    for cls, pattern in _COMPILED:
        if pattern.search(name):
            return cls
    return "other"
