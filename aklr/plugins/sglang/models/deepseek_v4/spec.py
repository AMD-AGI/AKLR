# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang-AMD-Mono project
# From SGLang-AMD-Mono's sgl_amd_mono/deepseek_v41/spec.py; AKLR: the import paths, the
# spec names, the step counters (aklr.plugins.sglang.stats).
"""DeepSeek-V4.1-Flash on gfx950: the monokernel spec, and the forward a routed decode or
DSpark verify step runs for ``DeepseekV4ForCausalLM.forward`` (each layer as K1, then K2).
"""

from __future__ import annotations

import importlib.util
import os
import re
from typing import Any

import msgspec
import torch

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.models.deepseek_v4 import (
    SPEC_NAMES,
    TARGET,
    compressor_routes,
    embed,
    engram_routes,
    jit_routes,
    target_vllm,
)
from aklr.plugins.sglang.models.deepseek_v4.compat import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
    StepMetadata,
    is_unified_kv_triton,
    mhc_combine,
)
from aklr.plugins.sglang.models.deepseek_v4.config import CANDIDATES, HF_CONFIG
from aklr.plugins.sglang.models.deepseek_v4.kernels.common.plan import BLOCKS

MAX_ROWS = 48  # the rows a launch takes at most (``layer.MAX_TOKENS``)
IQ_GROUP_ROWS = 8  # index_q_pack's rows a group (IndexQPackParams::MAX_ROWS)
IQ_WIDE_CHECK_ROWS = (24, 40)  # the rows the Triton query is checked at, one a launch configuration
# the steps the target's monokernels ran (a device counter: graph replays included)
SERVED = "deepseek_v4.target:served"
# "0": the indexer query in SGLang's kernels, not in one launch (``_index_q``)
ENV_INDEX_Q = "AKLR_DSV41_INDEX_Q"
# "0": a ratio-2 compressor's projection in SGLang's GEMM, not in the query's launch
ENV_GATE_IN_Q = "AKLR_DSV41_GATE_IN_Q"
# "0": past eight rows the query stays in index_q_pack's groups of eight
ENV_INDEX_Q_WIDE = "AKLR_DSV41_INDEX_Q_WIDE"
# "0": the head's combine reads an fp32 copy of the residual, as sglang's forward makes it
ENV_COMBINE_BF16 = "AKLR_DSV41_COMBINE_BF16"
# "0": the DSpark target layers' means as a list sglang concatenates, not one packed tensor
ENV_PACK_AUX = "AKLR_DSV41_PACK_AUX"
# "0": every REINDEX layer rebuilds the step's candidate table
ENV_REINDEX_ONCE = "AKLR_DSV41_REINDEX_ONCE"
# "0": the last layer's post by aiter, then the head's combine (two launches)
ENV_POST_COMBINE = "AKLR_DSV41_POST_COMBINE"
# "0": each Engram layer's hash ids copied out of the strided [tokens, layers, ..] hash
ENV_ENGRAM_IDS = "AKLR_DSV41_ENGRAM_IDS"

_SHARED = ("gate_proj", "up_proj", "down_proj")
# the checkpoint's shared expert tensors, as the model loader reads them
_CKPT_SHARED = re.compile(r"layers\.(?P<i>\d+)\.ffn\.shared_experts\.(?P<w>w[123])\.(?P<kind>weight|scale)")
_CKPT_PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
# the (freqs_cis, table) pairs built in this process: the target's and the DSpark
# draft's equal tables share one copy
ROPE_TABLES: list = []

# The SGLang names this glue reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    # an index source whose rows SGLang's decode indexer does not score
    "kernels/ops/attention/dsv4/__init__.py": ("topk_transform_paged",),
    "kernels/ops/attention/dsv4/topk.py": (
        "topk_transform_paged(scores)",
        "topk_transform_paged(seq_lens)",
        "topk_transform_paged(page_tables)",
        "topk_transform_paged(out_page_indices)",
        "topk_transform_paged(page_size)",
        "topk_transform_paged(out_raw_indices)",
    ),
    # SGLang's indexer query at decode row counts (``_index_query``)
    "kernels/ops/attention/dsv4/fp4_indexer_hip.py": (
        "index_q_rope_pack_weights_flydsl",
        "index_q_rope_pack_weights_flydsl(q)",
        "index_q_rope_pack_weights_flydsl(freqs_cis)",
        "index_q_rope_pack_weights_flydsl(positions)",
        "index_q_rope_pack_weights_flydsl(rope_dim)",
        "index_q_rope_pack_weights_flydsl(head_weight_partials)",
        "index_q_rope_pack_weights_flydsl(weight_scale)",
        "index_q_rope_pack_weights_flydsl(num_heads)",
        # whether SGLANG_HIP_DSV41_INDEX_K_PAGE8 takes (``_page8_ratios``)
        "aiter_has_fp4_rowgroup",
    ),
    "kernels/ops/gemm/router_gemv_hip.py": (
        "rocm_router_gemv_split_k",
        "rocm_router_gemv_split_k(x)",
        "rocm_router_gemv_split_k(w)",
    ),
    "kernels/ops/layernorm/mhc.py": ("hc_broadcast",),
    "kernels/ops/layernorm/mhc_boundary_hip.py": ("tensors",),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "moe_runner_backend",
        "speculative_algorithm",
        "enable_encoder_swa_bounded_replay",
        "fp8_gemm_runner_backend",
        "context_length",
        "expert_distribution_recorder_mode",
    ),
    "srt/environ.py": (
        "SGLANG_DSV4_FP4_EXPERTS",
        "SGLANG_DSV4_FP4_DEQUANT",
        "SGLANG_USE_AITER_MOE_GU_ITLV",
        "SGLANG_DSV4_KV_LAYOUT",
        "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE",
        "SGLANG_HIP_DSV41_INDEX_K_PAGE8",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ),
    # an index source's compressor, between K1 and K2
    "srt/layers/attention/deepseek_v4_backend.py": (
        "forward_low_ratio_sources(layer)",
        "forward_low_ratio_sources(x)",
        "forward_low_ratio_sources(q_lora)",
        "forward_low_ratio_sources(positions)",
        "forward_low_ratio_sources(forward_batch)",
        "forward_low_ratio_sources(run_indexer)",
    ),
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": (
        "forward_low_ratio_sources",
        "forward_metadata",
        "core_attn_metadata",
        "get_swa_out_cache_loc",
        "swa_page_indices",
        "swa_topk_lengths",
        "sparse_page_indices",
        "sparse_topk_lengths",
        # its indexer's inputs and outputs: K2i stands in for its scorer
        "low_ratio_indexer_metadata",
        "sparse_raw_indices",
        "drop_folded_sparse_indices",
    ),
    # SGLang's decode indexer: its per-step decisions
    "srt/layers/attention/dsv4/low_ratio_backend_hip.py": (
        "low_ratio_decode_rows_are_identity",
        "low_ratio_decode_rows_are_identity(backend)",
        "low_ratio_decode_rows_are_identity(forward_batch)",
        "low_ratio_decode_rows_are_identity(ratio)",
        "low_ratio_decode_rows_fit_candidate_span",
        "low_ratio_decode_rows_fit_candidate_span(backend)",
        "low_ratio_decode_rows_fit_candidate_span(forward_batch)",
    ),
    "srt/layers/attention/dsv4/metadata.py": (
        "page_table",
        "compressed_seq_lens",
        "compressed_page_size",
    ),
    "srt/layers/attention/dsv4/dsv41_sparse.py": (
        "is_candidate_source",
        "uses_candidates",
        "candidate_topk_blocks",
        "candidate_block_size",
        "n_heads",
        "wq_b",
        "weights_proj",
        "weights_proj_hip_max_tokens",
        "rope_head_dim",
        "head_weight_scale",
    ),
    "srt/layers/quantization/fp8.py": (
        "aiter_bpreshuffled",
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        '"w2_weight_scale_inv"',
    ),
    "srt/layers/quantization/fp8_hip.py": ("mxfp8_native_ready",),
    "srt/layers/quantization/mxfp4.py": ("is_shuffled",),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "get_attn_tp_context",
        "maybe_input_scattered",
    ),
    "srt/layers/logits_processor.py": ("forward(hidden_states_before_norm)",),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "get_swa_raw_buffer",
        "get_swa_key_page_size",
        "get_extra_key_buffer",
        "get_extra_key_page_size",
        "get_index_k_fp4_payload_buffer",
        "get_index_k_fp4_scale_buffer",
        "low_ratio_index_k_is_split",
        "low_ratio_index_k_page8",
        "get_index_k_page_size",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/configs/deepseek_v4.py": ("vision_n_layers", "image_token_id"),
    "srt/layers/engram.py": ("layer_hash_index", "forward(cp_all_tokens)"),
    "srt/models/deepseek_common/amd/deepseek_v4_hip.py": ("hc_boundary_fused",),
    "srt/models/deepseek_v2.py": (
        "experts",
        "gate",
        "e_score_correction_bias",
        "is_nextn",
        "is_hash",
        "forward_normal",
    ),
    # DeepseekV4ForCausalLM.forward, which the routed forward stands in for
    "srt/models/deepseek_v4.py": (
        "prepare_language_model_inputs",
        "logits_processor",
        "lm_head",
        "capture_aux_hidden_states",
        "model",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "embed_tokens",
        "hc_mult",
        "norm",
        "start_layer",
        "end_layer",
        "engram_hasher",
        '"freqs_cis_c4"',
        '"freqs_cis_c128"',
        "layers",
        "config",
        "dspark_layers_to_capture",
        "self_attn",
        "mlp",
        "engram",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "input_layernorm",
        "post_attention_layernorm",
        "hc_post",
        "compressor",
        "indexer",
        "q_head_norm",
        "fuse_wqa_wkv",
        "n_local_heads",
        "wqkv_a",
        "wq_b",
        "wo_a",
        "wo_b",
        "q_norm",
        "kv_norm",
        "freqs_cis",
        "attn_sink",
        "compress_ratio",
    ),
    "srt/configs/model_config.py": ("context_len",),
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
    "srt/utils/common.py": ("get_bool_env_var",),
}

# Per launch output against the normal path on the same GPU (the kernel tests)
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # K1: the seam's mixes and the SWA KV records as the normal path writes them
        "k1.kv.max_abs": 0.0,
        "k1.mix.max_abs": 1e-6,
        # wq_b summed in another order; bf16 rounding flips only
        "k1.q.cosine": 0.99999,
        "k1.q.rel": 5e-3,
        # an index source's normed input as the normal path's norm writes it, and
        # its qr (q_norm's sum of squares in another order: bf16 flips)
        "k1.normed.max_abs": 0.0,
        "k1.qr.rel": 4e-3,
        # K2: the attention core output, wo_a's MXFP8 output and wo_b's partial
        "k2.attn_out.cosine": 0.9998,
        "k2.wo_a.rel": 5e-3,
        "k2.wo_b.rel": 1e-3,
        # the FFN launch (K2 runs the same stages)
        "ffn.mix.rel": 1e-4,
        "ffn.normed.cosine": 0.9999,
        "ffn.normed.rel": 0.012,
        "ffn.moe.cosine": 0.9999,
        "ffn.moe.rel": 0.015,
        # K2i: a row's largest logit difference from SGLang's FP4 logits over its largest
        # logit; a candidate consumer's top-512 equals SGLang's candidate-block top-k as sets
        "k2i.logits.rel": 1e-5,
        "k2i.sel.max_abs": 0.0,
    },
)


def _covered_layers(config) -> list[int]:
    """The layers K1 and K2 can run: not hash-routed, compress ratio 0, 1 or 2."""
    first = getattr(config, "num_hash_layers", 0) or 0
    return [i for i in range(first, config.num_hidden_layers) if config.compress_ratios[i] in (0, 1, 2)]


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: a value SGLang's attention gate must not
    take, aiter's own flag, and FlyDSL."""
    from sglang.srt.utils import get_bool_env_var

    reasons = []
    if is_unified_kv_triton():
        reasons.append("needs the paged SWA cache, not unified_kv")
    if get_bool_env_var("AITER_FORCE_A8W4"):
        reasons.append(
            "needs the MXFP4 experts in aiter's gate / up interleaved layout, not AITER_FORCE_A8W4"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.attention.plan import (
        Dims,
    )

    heads = Dims(get_parallel().tp_size).heads
    covered = _covered_layers(hf_config)
    other = sorted(set(range(hf_config.num_hidden_layers)) - set(covered))
    reasons = [f"layers {other}: no kernels (hash-routed, or another compress ratio)"]
    reasons, unseen = reasons if other else [], []
    for i in covered:
        layer = model.model.layers[i]
        a, moe = layer.self_attn, layer.mlp
        if not layer.hc_boundary_fused or moe.is_nextn or moe.is_hash:
            reasons.append(f"layer {i}: needs the fused mHC boundary and a routed MoE")
        if not getattr(moe.experts.w13_weight, "is_shuffled", False):
            reasons.append(f"layer {i}: MoE weights not in aiter's shuffle")
        if any(("shared expert fp8", i, p, k) not in copies for p in _SHARED for k in ("weight", "scale")):
            unseen.append(i)
        if a.q_head_norm or not a.fuse_wqa_wkv or a.n_local_heads != heads:
            reasons.append(f"layer {i}: K1 needs the fused wqkv_a, {heads} local heads and no q head norm")
        if (a.compressor is not None or a.indexer is not None) and (a.compress_ratio not in (1, 2)):
            reasons.append(
                f"layer {i}: a compressor or indexer at compress ratio "
                f"{a.compress_ratio}; SGLang's low-ratio sources serve ratio 1 or 2"
            )
        reasons += _indexer_reasons(i=i, attn=a)
        if any(
            getattr(lin, "aiter_bpreshuffled", False) or getattr(lin, "mxfp8_native_ready", False)
            for lin in (a.wqkv_a, a.wq_b, a.wo_a, a.wo_b)
        ):
            reasons.append(
                f"layer {i}: an FP8 linear method reordered an attention weight in "
                "place; K1 and K2 read them row-major"
            )
    if unseen:
        reasons.append(f"the shared expert's FP8 checkpoint tensors were not seen at load (layers {unseen})")
    return reasons


def _indexer_reasons(*, i: int, attn) -> list[str]:
    """What K2i needs of layer i's indexer past ``HF_CONFIG``: a candidate source's
    blocks (optional HF-config keys, so read off the indexer), and SGLang's
    decode-row query path, which ``_index_query`` runs."""
    ix = attn.indexer
    if ix is None:
        return []
    reasons = []
    if (ix.is_candidate_source or ix.uses_candidates) and (
        ix.candidate_topk_blocks,
        ix.candidate_block_size,
    ) != (CANDIDATES["candidate_topk_blocks"], CANDIDATES["candidate_block_size"]):
        reasons.append(f"layer {i}: K2i scores candidate blocks of 2048 x 8 rows")
    if ix.weights_proj_hip_max_tokens < MAX_ROWS or attn.freqs_cis.dtype != (torch.complex64):
        reasons.append(
            f"layer {i}: K2i's query needs SGLang's split-K head-weight GEMV for "
            f"{MAX_ROWS} rows and a complex64 RoPE table"
        )
    return reasons


def _load_tensor(copies: dict, model, name: str, tensor: torch.Tensor) -> None:
    """Keep this rank's shard of each covered layer's FP8 shared expert, by its
    checkpoint name (``layers.{i}.ffn.shared_experts.{w1,w2,w3}.{weight,scale}``;
    the checkpoint's ``mtp.*`` layers do not match)."""
    from sglang.srt.runtime_context import get_parallel

    m = _CKPT_SHARED.fullmatch(name)
    if m is None or int(m["i"]) not in _covered_layers(model.config):
        return
    i, proj = int(m["i"]), _CKPT_PROJ[m["w"]]
    p = get_parallel()
    dim = 1 if proj == "down_proj" else 0  # the TP-sharded intermediate dimension
    t = tensor.view(torch.uint8)
    n = t.shape[dim] // p.tp_size
    shard = t.narrow(dim, p.tp_rank * n, n).to(torch.cuda.current_device())
    copies["shared expert fp8", i, proj, m["kind"]] = shard.contiguous()


class _Layer(msgspec.Struct):
    """One covered layer: its module and weights, and its caches (bound at the
    layer's first launch)."""

    i: int
    module: Any
    ffn: Any  # FFNWeights
    k1: Any = None  # K1Weights
    k2: Any = None  # K2Weights
    ratio: int = 0  # the compress ratio: which sparse key table K1 reads (0: none)
    # an index source: SGLang's compressor and indexer run between K1 and K2
    index: bool = False
    swa: Any = None  # the SWA cache, uint8 [pages, page bytes]
    swa_block: int = 0
    comp: Any = None  # the compressed cache, the same layout
    comp_block: int = 0
    plane: Any = None  # an indexer's FP4 index plane: (payload, scales)
    # an indexer's query in one launch (``_index_q``): its weights, checked against
    # SGLang's query at the layer's first launch
    iq: Any = None
    iq_checked: bool = False
    # past index_q_pack's eight rows a group, the query in one Triton launch (``index_q.py``)
    iq_wide: bool = True


def _keep(copies: dict, t: torch.Tensor, *key) -> torch.Tensor:
    copies[key] = t
    return t


def _codes(copies: dict, i: int, name: str, lin) -> torch.Tensor:
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.weights import (
        e8m0,
    )

    return _keep(
        copies,
        e8m0(lin.weight_scale_inv, f"layer {i} {name}"),
        "e8m0 scale codes",
        i,
        name,
    )


def _weights(copies: dict, tables: list, layer, i: int) -> _Layer:
    """Layer i's tensors as the launches read them."""
    from sglang.srt.runtime_context import get_parallel

    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.attention.plan import (
        Dims,
    )
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.runner import (
        FFNWeights,
        K1Weights,
        K2Weights,
    )
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.weights import (
        mxfp8_exact,
        rope_table,
    )

    p, a, moe = get_parallel(), layer.self_attn, layer.mlp
    d, e, u8, e4m3 = Dims(p.tp_size), moe.experts, torch.uint8, torch.float8_e4m3fn
    n = moe.gate.weight.shape[0]  # the routed experts, before a fused shared one

    def gate_up(kind):  # the shared expert's gate and up projections, one tensor
        parts = [copies.pop(("shared expert fp8", i, q, kind)) for q in _SHARED[:2]]
        return _keep(copies, torch.cat(parts), "shared expert fp8", i, "gate_up", kind)

    def down(kind):
        return copies["shared expert fp8", i, "down_proj", kind]

    ffn = FFNWeights(
        hc_ffn_fn=layer.hc_ffn_fn,
        hc_ffn_scale=layer.hc_ffn_scale,
        hc_ffn_base=layer.hc_ffn_base,
        ffn_norm=layer.post_attention_layernorm.weight,
        gate_w=moe.gate.weight,
        bias=_keep(copies, moe.gate.e_score_correction_bias.float(), "f32 router bias", i),
        w13=e.w13_weight[:n],
        w13_s=e.w13_weight_scale_inv.view(u8)[: n * e.w13_weight.shape[1]],
        w2=e.w2_weight[:n],
        w2_s=e.w2_weight_scale_inv.view(u8)[: n * e.w2_weight.shape[1]],
        sgu=gate_up("weight").view(e4m3),
        sgu_s=gate_up("scale"),
        sw2=down("weight").view(e4m3),
        sw2_s=down("scale"),
    )
    ffn.check(p.tp_size, n)
    L = _Layer(i, layer, ffn)
    if a.fuse_wqa_wkv:
        wqkv, wqkv_s = a.wqkv_a.weight, _codes(copies, i, "wqkv_a", a.wqkv_a)
    else:  # the DSpark draft's wq_a and wkv as K1's one weight (a copy)
        u8 = [lin.weight.view(torch.uint8) for lin in (a.wq_a, a.wkv)]
        wqkv = _keep(copies, torch.cat(u8).view(a.wq_a.weight.dtype), "wqkv fp8", i)
        wqkv_s = _keep(
            copies,
            torch.cat([_codes({}, i, n_, lin) for n_, lin in (("wq_a", a.wq_a), ("wkv", a.wkv))]),
            "e8m0 scale codes",
            i,
            "wqkv",
        )
    cos_sin = rope_table(a.freqs_cis, tables)
    L.k1 = K1Weights(
        hc_fn=layer.hc_attn_fn,
        hc_scale=layer.hc_attn_scale,
        hc_base=layer.hc_attn_base,
        norm=layer.input_layernorm.weight,
        wqkv=wqkv,
        wqkv_s=wqkv_s,
        q_norm=a.q_norm.weight,
        kv_norm=a.kv_norm.weight,
        wqb=a.wq_b.weight,
        wqb_s=_codes(copies, i, "wq_b", a.wq_b),
        cos_sin=cos_sin,
    )
    L.k1.check(d)
    if a.wo_a.weight.dtype == e4m3:
        woa, woa_s = a.wo_a.weight, _codes(copies, i, "wo_a", a.wo_a)
    else:  # SGLang dequantized the checkpoint's FP8 wo_a to bf16 at load
        woa, woa_s = mxfp8_exact(a.wo_a.weight.reshape(d.groups * 1024, d.group_k), f"layer {i} wo_a")
        _keep(copies, woa, "wo_a mxfp8", i, "weight")
        _keep(copies, woa_s, "wo_a mxfp8", i, "scale")
    sink = a.attn_sink.detach()[p.tp_rank * d.heads : (p.tp_rank + 1) * d.heads]
    L.k2 = K2Weights(
        woa=woa,
        woa_s=woa_s,
        wob=a.wo_b.weight,
        wob_s=_codes(copies, i, "wo_b", a.wo_b),
        sink=_keep(copies, sink.float().contiguous(), "f32 attention sinks", i),
        cos_sin=cos_sin,
    )
    L.k2.check(d)
    L.ratio = a.compress_ratio
    # the DSpark draft's window-only attention has neither
    L.index = getattr(a, "compressor", None) is not None or getattr(a, "indexer", None) is not None
    if getattr(a, "indexer", None) is not None:
        L.iq = _index_q_weights(copies, a, i)
    return L


def _index_q_weights(copies: dict, attn, i: int):
    """The indexer's query weights as ``hipops.index_q_pack`` reads them, or None
    (SGLang's query, the reason logged)."""
    ix, why = attn.indexer, None
    w, wp, f = ix.wq_b.weight, ix.weights_proj.weight, attn.freqs_cis
    shape = (ix.n_heads * 128, HF_CONFIG["q_lora_rank"])
    if os.environ.get(ENV_INDEX_Q, "1") == "0":
        why = f"{ENV_INDEX_Q}=0"
    elif w.dtype != torch.float8_e4m3fn or tuple(w.shape) != shape:
        why = f"wq_b {w.dtype} {tuple(w.shape)}, not e4m3 {shape}"
    elif getattr(w, "is_shuffled", False) or not w.is_contiguous():
        why = "wq_b not row-major"
    elif wp.dtype != torch.bfloat16 or not wp.is_contiguous():
        why = f"weights_proj {wp.dtype}"
    elif f.dtype != torch.complex64 or not f.is_contiguous():
        why = f"freqs_cis {f.dtype}"
    else:
        try:
            codes = _codes(copies, i, "indexer wq_b", ix.wq_b)
        except RuntimeError as e:
            why = str(e)
    if why is not None:
        stats.set_info(f"index_q layer {i}", why)
        return None
    return (
        w.view(torch.uint8),
        codes,
        wp,
        torch.view_as_real(f),
        float(ix.head_weight_scale),
    )


def counted() -> bool:
    """Whether this step's metadata launch added one to the served counter
    (``metadata_routes._counter``)."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    core = getattr(get_attn_backend().forward_metadata, "core_attn_metadata", None)
    return bool(getattr(core, "_aklr_counted", False))


# whether the fused post and combine agreed with aiter's mhc_post and sglang's mhc_combine
_POST_COMBINE_AGREED: dict = {}


def _post_combine_enabled() -> bool:
    return os.environ.get(ENV_POST_COMBINE, "1") != "0" and _POST_COMBINE_AGREED.get("ok", True)


def _settled(layer, pending):
    """The residual ``pending`` (x, residual, post, comb) settles to: ``layer``'s post where
    one is owed (x not None)."""
    x, residual, post, comb = pending
    return residual if x is None else layer.hc_post(x, residual, post, comb)


def head_post_combine(layer, pending, pre, hc: int):
    """The streams after the last layer and their combine by ``pre``; an owed post and the
    combine in one launch (``prologue.post_combine``) where its first eager call matches
    aiter's ``mhc_post`` and sglang's ``mhc_combine`` bit for bit."""
    x, residual, post, comb = pending
    if x is None:
        # the combine widens its input to fp32 as it loads it: the same sums from bf16
        return residual, mhc_combine(_widened(residual.flatten(1)), pre, hc, residual.dtype)
    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    def separate():
        out = layer.hc_post(x, residual, post, comb)
        return out, mhc_combine(_widened(out.flatten(1)), pre, hc, out.dtype)

    fits = (
        residual.dtype == torch.bfloat16
        and x.dtype == torch.bfloat16
        and all(t.is_contiguous() for t in (x, residual, post, comb, pre))
        and post.dtype == comb.dtype == pre.dtype == torch.float32
        and residual.dim() == 3
        and residual.shape[1] == hc
        and tuple(post.shape) == (residual.shape[0], hc)
        and tuple(comb.shape) == (residual.shape[0], hc, hc)
    )
    if not fits:
        return separate()
    if "ok" not in _POST_COMBINE_AGREED:
        if torch.cuda.is_current_stream_capturing():
            return separate()
        want = separate()
        got = prologue.post_combine(x, residual, post, comb, pre)
        ok = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(want, got))
        _POST_COMBINE_AGREED["ok"] = ok
        stats.set_info(
            "head_post_combine",
            "one launch" if ok else "aiter's post, then the combine",
        )
        return got if ok else want
    if not _POST_COMBINE_AGREED["ok"]:
        return separate()
    return prologue.post_combine(x, residual, post, comb, pre)


# whether the packed aux means agreed with torch's mean at their first eager call
_PACK_AGREED: dict = {}


def _pack_aux_ok(residual, captured: int, plan) -> bool:
    """Whether this capture writes its mean into the packed aux buffer (each target layer's
    in turn, a contiguous bf16 [M, HC, H] residual), its kernel agreeing with torch's mean.
    """
    if (
        os.environ.get(ENV_PACK_AUX, "1") == "0"
        or residual.dim() != 3
        or residual.dtype != torch.bfloat16
        or not residual.is_contiguous()
        or (captured > 0 and plan.state.get("dspark_aux_packed") is None)
    ):
        return False
    if "ok" not in _PACK_AGREED:
        if torch.cuda.is_current_stream_capturing():
            return False
        from aklr.plugins.sglang.models.deepseek_v4 import prologue

        got = prologue.hc_mean_into(residual, torch.empty_like(residual[:, 0]))
        want = residual.mean(dim=1)
        _PACK_AGREED["ok"] = bool(torch.equal(got.view(torch.int16), want.view(torch.int16)))
        stats.set_info(
            "dspark_aux",
            "packed means" if _PACK_AGREED["ok"] else "torch's mean (disagreed)",
        )
    return _PACK_AGREED["ok"]


def _widened(x):
    """``x`` for ``mhc_combine``: as it is where the combine's loads widen it to fp32 bit for
    bit (bf16, unit stride), else ``x.float()`` (``AKLR_DSV41_COMBINE_BF16=0``: always).
    """
    if x.dtype == torch.bfloat16 and x.stride(-1) == 1 and os.environ.get(ENV_COMBINE_BF16, "1") != "0":
        return x
    return x.float()


def _boundary_out(layer, defer_post, x, residual, post, comb, pre):
    """What the fused boundary returns: (hidden_states, pre, pending post); with
    ``defer_post`` this layer's post stays pending for the next layer."""
    post = post.view(post.shape[0], -1)
    if defer_post:
        return None, pre, (x, residual, post, comb)
    return layer.hc_post(x, residual, post, comb), pre, None


def _launch(
    runner,
    L,
    plan,
    positions,
    forward_batch,
    pending_post,
    prev_pre,
    capture=None,
    vllm=None,
):
    """K1, then K2, of layer ``L``: the previous layer's pending post and pre in (x None: the
    residual arrives settled), K2's (x, residual, post, comb, pre) out. ``vllm``: the layer
    in vLLM's launches (``target_vllm``), K2i and the top-k still the runner's."""
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    x, residual, post, comb = pending_post
    if L.swa is None:  # the layer's caches, at its first launch
        pool = get_token_to_kv_pool()
        L.swa, L.swa_block = pool.get_swa_raw_buffer(L.i), pool.get_swa_key_page_size()
        if L.ratio:
            # SGLang types the compressed cache as e4m3; K2 addresses its records' bytes
            L.comp = pool.get_extra_key_buffer(L.i).view(torch.uint8)
            L.comp_block = pool.get_extra_key_page_size(L.i)
    backend, M = get_attn_backend(), residual.shape[0]
    if "k1" not in plan.state:
        # the forward's first K1 converts them; a graph records the copies
        slot = backend.get_swa_out_cache_loc(forward_batch)
        core = getattr(backend.forward_metadata, "core_attn_metadata", None)
        slot64 = getattr(core, "_aklr_slot64", None)  # metadata_routes' int64 copy
        if slot64 is None or slot is not core.swa_out_cache_loc or slot64.shape[0] < M:
            slot64 = slot[:M].long().contiguous()
        plan.state["k1"] = (positions[:M].long().contiguous(), slot64[:M])
    pos, slot = plan.state["k1"]
    md = backend.forward_metadata.core_attn_metadata
    if L.ratio:  # the top-k of the layer's index source (its own, after K1)
        topk = (md.sparse_page_indices(L.ratio), md.sparse_topk_lengths(L.ratio))
    else:  # the window only
        topk = (runner.no_topk, runner.no_topk_lens)
    keys = (md.swa_page_indices, md.swa_topk_lengths, *topk)
    if x is None:
        residual = residual.contiguous()  # the broadcast embedding may be a view
        if prev_pre is None:
            prev_pre = runner.first_pre[:M]

    def select(normed, qr):
        _select(
            runner=runner,
            L=L,
            plan=plan,
            backend=backend,
            positions=positions,
            pos=pos,
            forward_batch=forward_batch,
            M=M,
            normed=normed,
            qr=qr,
        )

    if vllm is not None:
        out = vllm.forward(
            L,
            x,
            residual,
            post,
            comb,
            prev_pre,
            pos,
            slot,
            keys,
            select=select if L.index else None,
        )
        if capture is not None:  # K2 leaves the seam's residual in place
            capture(L, vllm.res_mid[:M])
        return out
    _, res_mid, *_ = runner.k1(
        L.k1,
        x,
        residual,
        post,
        comb,
        prev_pre,
        pos,
        slot,
        L.swa,
        L.swa_block,
        keys,
        index=L.index,
    )
    if capture is not None:
        capture(L, res_mid)
    if L.index:
        select(runner.normed[:M], runner.qr[:M])
    # K2 reads K1's outputs from the runner's buffers
    return runner.k2(
        L.k2,
        L.ffn,
        M,
        pos,
        L.swa,
        L.swa_block,
        L.comp,
        L.comp_block,
        slot=slot,
        keys=keys,
        index=L.index,
    )


def _select(*, runner, L, plan, backend, positions, pos, forward_batch, M, normed, qr) -> None:
    """An index source between K1 and K2: SGLang's compressor on K1's normed input, then the
    layer's top-512 into SGLang's sparse indices as SGLang's decode indexer decides (every
    row in order, or K2i's logits and SGLang's top-k kernels)."""
    from sglang.kernels.ops.attention.dsv4 import topk_transform_paged
    from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
        low_ratio_decode_rows_are_identity,
        low_ratio_decode_rows_fit_candidate_span,
    )

    attn, r = L.module.self_attn, L.ratio
    query = None
    if attn.compressor is not None:
        kv_score, gate_w = None, _compressor_gate(attn)
        if gate_w is not None:  # the compressor's GEMV in the query's launch
            query, kv_score = _index_query(L=L, attn=attn, x=normed, q_lora=qr, positions=pos, gate_w=gate_w)
        with compressor_routes.given(attn.compressor, normed, kv_score):
            backend.forward_low_ratio_sources(
                layer=attn,
                x=normed,
                q_lora=None,
                positions=positions,
                forward_batch=forward_batch,
                run_indexer=False,
            )
    ix = attn.indexer
    if ix is None:
        return
    fm = backend.forward_metadata
    core, meta = fm.core_attn_metadata, fm.low_ratio_indexer_metadata(r)
    page_idx, raw_idx = core.sparse_page_indices(r), core.sparse_raw_indices(r)
    # as SGLang's indexer: it rewrites this ratio's indices in place
    core.drop_folded_sparse_indices(r)
    if low_ratio_decode_rows_are_identity(backend=backend, forward_batch=forward_batch, ratio=r):
        # every row sees at most index_topk rows: its top-k is every row, in order
        topk_transform_paged(
            scores=torch.empty((page_idx.shape[0], 1), dtype=torch.float32, device=page_idx.device),
            seq_lens=meta.compressed_seq_lens,
            page_tables=meta.page_table,
            out_page_indices=page_idx,
            page_size=meta.compressed_page_size,
            out_raw_indices=raw_idx,
        )
        return
    two_level = (
        ix.is_candidate_source or ix.uses_candidates
    ) and not low_ratio_decode_rows_fit_candidate_span(backend=backend, forward_batch=forward_batch)
    kind = "full" if not two_level else "reindex" if ix.uses_candidates else "producer"
    if L.plane is None:  # the layer's index plane, at its first launch (eager)
        L.plane = _index_plane(L=L, M=M, meta=meta, raw_idx=raw_idx)
    bounds = meta.compressed_seq_lens
    if bounds.dtype != torch.int32 or not bounds.is_contiguous():
        bounds = bounds.to(torch.int32).contiguous()
    k2i, per_req = runner.k2i, plan.key.tokens_per_request
    if kind != "reindex" and ("k2i plan", r) not in plan.state:
        # the step's walk plan of this ratio: its FULL and producer layers share it
        k2i.walk_plan(ratio=r, bounds=bounds, rows=M, per_req=per_req)
        plan.state["k2i plan", r] = True
    published = k2i.select(
        kind=kind,
        rows=M,
        per_req=per_req,
        ratio=r,
        query=(
            query
            if query is not None
            else _index_query(L=L, attn=attn, x=normed, q_lora=qr, positions=pos)[0]
        ),
        plane=L.plane,
        page_table=meta.page_table,
        bounds=bounds,
        raw_out=raw_idx,
        page_out=page_idx,
        # the candidate source's blocks of this step, for the layers after it
        candidates=plan.state["k2i candidates"] if kind == "reindex" else None,
        # the step's later REINDEX layers read the first one's candidate table (the same
        # candidates, page table and bounds: one compress ratio)
        table_ready=kind == "reindex" and plan.state.get("k2i table") == (r, M, per_req),
    )
    if kind == "producer":
        plan.state["k2i candidates"] = published
        plan.state.pop("k2i table", None)
    elif kind == "reindex" and os.environ.get(ENV_REINDEX_ONCE, "1") != "0":
        plan.state["k2i table"] = (r, M, per_req)


def _index_query(*, L, attn, x, q_lora, positions, gate_w=None) -> tuple:
    """(the FP4 query, its scales and head weights; ``x . gate_w^T`` in fp32 or None) in one
    launch (``_index_q``) where the layer's check passed, else (SGLang's query, None).
    """
    if L.iq is not None and not L.iq_checked:
        if torch.cuda.is_current_stream_capturing():
            return (
                _sglang_index_query(attn=attn, x=x, q_lora=q_lora, positions=positions),
                None,
            )
        L.iq_checked = True
        why = _index_q_disagrees(L, attn, IQ_GROUP_ROWS - 2, x.device, gate_w)
        wide_why = None
        if why is None and os.environ.get(ENV_INDEX_Q_WIDE, "1") == "0":
            wide_why = f"{ENV_INDEX_Q_WIDE}=0"
        elif why is None:
            # each of its launch configurations compiled here, before any capture
            for rows in IQ_WIDE_CHECK_ROWS:
                wide_why = wide_why or _index_q_disagrees(L, attn, rows, x.device, gate_w)
        L.iq_wide = why is None and wide_why is None
        how = "one launch" + (" with the compressor's GEMV" if gate_w is not None else "")
        if why is None:
            how += "; past 8 rows one Triton launch" if L.iq_wide else f"; past 8 rows in groups ({wide_why})"
        stats.set_info(f"index_q layer {L.i}", why or how)
        if why is not None:
            L.iq = None
    if L.iq is not None:
        out = _index_q(L.iq, x, q_lora, positions, gate_w, wide=L.iq_wide)
        return out[:3], out[3] if gate_w is not None else None
    return _sglang_index_query(attn=attn, x=x, q_lora=q_lora, positions=positions), None


def _index_q(iq, x, q_lora, positions, gate_w=None, wide=False) -> tuple:
    from aklr.plugins.sglang.models.deepseek_v4 import index_q
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    w, codes, wp, table, scale = iq
    if wide and q_lora.shape[0] > IQ_GROUP_ROWS and index_q.supported(q_lora, w, x, table):
        return index_q.index_q(q_lora, w, codes, x, wp, positions, table, scale, gate_w)
    return tuple(hipops.ops().index_q_pack(q_lora, w, codes, x, wp, positions, table, scale, gate_w))


def _compressor_gate(attn):
    """An index source's ratio-2 compressor's ``| kv | score |`` weight, for its
    projection in the indexer query's launch (``compressor_routes``), or None."""
    compressor = attn.compressor
    if (
        attn.indexer is None
        or not getattr(compressor, "use_fused_gate", False)
        or os.environ.get(ENV_GATE_IN_Q, "1") == "0"
    ):
        return None
    w = compressor.wkv_gate.weight
    if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.shape[1:] != (HF_CONFIG["hidden_size"],):
        return None
    return w


def _index_q_disagrees(L, attn, rows: int, device, gate_w=None) -> str | None:
    """Why ``_index_q`` cannot stand in for SGLang's query, or None: on the same random rows
    the FP4 bytes and scales equal but for bf16 rounding flips, the head weights within a
    bf16 step, ``gate_w``'s GEMV within fp32 noise of SGLang's ``linear_bf16_fp32``."""
    from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

    g = torch.Generator(device=device).manual_seed(L.i)
    q_lora = torch.randn(rows, HF_CONFIG["q_lora_rank"], device=device, generator=g)
    x = torch.randn(rows, HF_CONFIG["hidden_size"], device=device, generator=g)
    q_lora, x = q_lora.bfloat16(), x.bfloat16()
    positions = torch.arange(rows, device=device) * 997
    try:
        got = _index_q(L.iq, x, q_lora, positions, gate_w, wide=rows > IQ_GROUP_ROWS)
    except RuntimeError as e:
        return f"the kernel: {e}"
    want = _sglang_index_query(attn=attn, x=x, q_lora=q_lora, positions=positions)
    same = [
        (a.view(torch.uint8) == b.view(torch.uint8)).float().mean().item() for a, b in zip(got[:2], want[:2])
    ]
    weights = torch.allclose(got[2].float(), want[2].float(), rtol=1e-2, atol=1e-6)
    if min(same) < 0.98 or not weights:
        return (
            f"disagrees with SGLang's query: fp4 bytes {same[0]:.3f}, scales "
            f"{same[1]:.3f} the same, head weights {'close' if weights else 'apart'}"
        )
    if gate_w is not None:
        want_gate = linear_bf16_fp32(x, gate_w)
        apart = (got[3] - want_gate).abs().max().item()
        if not apart <= 1e-4 * want_gate.abs().max().item():
            return f"the compressor's GEMV apart from SGLang's by {apart:.3g}"
    return None


def _sglang_index_query(*, attn, x, q_lora, positions) -> tuple:
    """SGLang's indexer query for decode rows (``low_ratio_backend_hip``'s split-K
    GEMV path, its path at K2i's row counts)."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        index_q_rope_pack_weights_flydsl,
    )
    from sglang.kernels.ops.gemm.router_gemv_hip import rocm_router_gemv_split_k

    ix = attn.indexer
    q, _ = ix.wq_b(q_lora)
    return index_q_rope_pack_weights_flydsl(
        q=q,
        freqs_cis=attn.freqs_cis,
        positions=positions,
        rope_dim=ix.rope_head_dim,
        head_weight_partials=rocm_router_gemv_split_k(x=x, w=ix.weights_proj.weight),
        weight_scale=ix.head_weight_scale,
        num_heads=ix.n_heads,
    )


def _index_plane(*, L, M: int, meta, raw_idx) -> tuple:
    """The layer's FP4 index plane, after the checks K2i needs of SGLang's index
    cache (it exists only after the forward's KV sizing, so not at attach)."""
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.indexer.runner import (
        PLANE_ROWS,
        check_plane,
    )

    pool = get_token_to_kv_pool()
    reasons = []
    if not pool.low_ratio_index_k_is_split(L.i):
        reasons.append("the index keys are not in the split FP4 plane")
    if pool.get_index_k_page_size(L.ratio) != PLANE_ROWS:
        reasons.append(f"index pages of {pool.get_index_k_page_size(L.ratio)} rows")
    if meta.compressed_page_size != PLANE_ROWS or meta.page_table.shape[0] != M:
        reasons.append(
            f"an index page table of {tuple(meta.page_table.shape)} at "
            f"{meta.compressed_page_size} rows a page, for {M} rows"
        )
    if raw_idx is None:
        reasons.append("no sparse_raw_indices for the step")
    if reasons:
        raise RuntimeError(
            f"monokernel: layer {L.i}: K2i needs SGLang's FP4 index plane of "
            f"{PLANE_ROWS}-row pages: " + "; ".join(reasons)
        )
    page8 = hasattr(pool, "low_ratio_index_k_page8") and pool.low_ratio_index_k_page8(L.i)
    if page8 != (L.ratio in _page8_ratios()):
        raise RuntimeError(
            f"monokernel: layer {L.i}: its index keys are in the page-"
            f"{8 if page8 else 64} layout; K2i was built for the other"
        )
    payload = pool.get_index_k_fp4_payload_buffer(L.i)
    scale = pool.get_index_k_fp4_scale_buffer(L.i)
    check_plane(payload=payload, scale=scale)
    return payload.view(torch.uint8), scale, page8


def _page8_ratios() -> frozenset:
    """The compress ratios whose index-K pool SGLang lays out in row-group 8-slot pages, as
    its memory pool decides it; known before the pool exists, so K2i builds before capture.
    """
    from sglang.srt.utils import get_bool_env_var

    if not get_bool_env_var("SGLANG_HIP_DSV41_INDEX_K_PAGE8"):
        return frozenset()
    try:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            aiter_has_fp4_rowgroup,
        )
    except ImportError:
        return frozenset()
    return frozenset({1}) if aiter_has_fp4_rowgroup() else frozenset()


class _Built:
    """One rank's monokernels: the runner, each layer's weights, and the forward."""

    def __init__(self, lm, runner, layers: dict, vllm=None):
        self.lm, self.model, self.runner, self._layers = lm, lm.model, runner, layers
        self.vllm = vllm  # the target layers in vLLM's launches (``target_vllm``)
        self._normal_moe = lm.model.layers[min(layers)].mlp
        self._served = stats.device_counter(SERVED, runner.device)
        stats.set_info(SERVED, {"layers": sorted(layers), "of": lm.config.num_hidden_layers})

    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None, *_, **__):
        """``DeepseekV4ForCausalLM.forward`` for a routed step: SGLang's input
        preparation and logits processor around the backbone."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        if not counted():
            self._served.add_(1)
        if not torch.cuda.is_current_stream_capturing():
            stats.record(SERVED, "served", str(plan.key))
        lm = self.lm
        input_ids, input_embeds = lm.prepare_language_model_inputs(input_ids, forward_batch, input_embeds)
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            out = self._backbone(plan, input_ids, positions, forward_batch, input_embeds)
        aux = None
        if lm.capture_aux_hidden_states:
            out, aux = out
        hidden, pre_hc_head = out
        return lm.logits_processor(
            input_ids,
            hidden,
            lm.lm_head,
            forward_batch,
            aux,
            hidden_states_before_norm=None if aux is not None else pre_hc_head,
        )

    def _backbone(self, plan, input_ids, positions, forward_batch, input_embeds):
        """``DeepseekV4Model.forward``: the embedding, each layer as K1 and K2,
        SGLang's Engram writes, then the head's combine and norm."""
        from sglang.kernels.ops.layernorm.mhc import hc_broadcast

        m = self.model
        self._warm_normal_moe()
        x = None
        if input_embeds is None:
            x = embed.lookup(m.embed_tokens, input_ids, m.hc_mult)
        if x is None:
            x = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
            x = hc_broadcast(x, m.hc_mult)
        # the compressor caches each step's rotary tables on the batch
        for attr in ("freqs_cis_c4", "freqs_cis_c128"):
            if hasattr(forward_batch, attr):
                delattr(forward_batch, attr)
        hash_ids = None
        if m.engram_hasher is not None:
            hash_ids = m.engram_hasher(input_ids, forward_batch)
            if hash_ids.dim() == 3 and hash_ids.shape[1] > 1 and os.environ.get(ENV_ENGRAM_IDS, "1") != "0":
                # every layer's ids contiguous in one copy: the lookups' flattens are views
                hash_ids = hash_ids.transpose(0, 1).contiguous().transpose(0, 1)
        aux = None if m.dspark_layers_to_capture is None else []
        plan.state["dspark_aux"] = aux
        plan.state.pop("dspark_aux_packed", None)
        pending, pre = (None, x, None, None), None
        engram_post = engram_routes.post_enabled()
        for i in range(m.start_layer, m.end_layer):
            engram = m.layers[i].engram
            if engram is not None:
                ids = hash_ids[:, engram.layer_hash_index]
                # every image-token row keeps its residual, as in SGLang's forward: a
                # captured graph cannot branch on whether the batch has mm inputs
                if m.config.vision_n_layers > 0:
                    residual = engram_routes.forward_keeping_owed(
                        engram,
                        m.layers[i - 1],
                        pending,
                        ids,
                        forward_batch,
                        input_ids,
                        m.config.image_token_id,
                    )
                else:
                    residual = engram(_settled(m.layers[i - 1], pending), ids, forward_batch)
                pending = (None, residual, None, None)
            # an Engram layer reads the settled residual: a post owed into it only where
            # its write takes the post (engram_routes.forward_keeping_owed)
            defer = i + 1 < m.end_layer and (m.layers[i + 1].engram is None or engram_post)
            # the last layer's post owed to the head's combine (post_combine)
            defer = defer or (i + 1 == m.end_layer and _post_combine_enabled())
            hidden, pre, post = self._layer(
                self._layers[i], plan, positions, forward_batch, pending, pre, defer
            )
            pending = post if defer else (None, hidden, None, None)
        residual, combined = head_post_combine(m.layers[m.end_layer - 1], pending, pre, m.hc_mult)
        pre_hc_head = residual.flatten(1)
        hidden = m.norm(combined)
        if aux is not None:
            packed = plan.state.get("dspark_aux_packed")
            if packed is not None and len(aux) == len(m.dspark_layers_to_capture):
                aux = packed  # sglang's pack_aux_hidden_states passes a packed tensor through
            return (hidden, pre_hc_head), aux
        return hidden, pre_hc_head

    def _layer(self, L, plan, positions, forward_batch, pending_post, prev_pre, defer_post):
        """One layer: the previous layer's pending post (x, residual, post, comb)
        and pre in; the fused boundary's outputs out."""
        self._warm_normal_moe()
        out = _launch(
            self.runner,
            L,
            plan,
            positions,
            forward_batch,
            pending_post,
            prev_pre,
            capture=lambda L, residual: self._capture(plan, L, residual),
            vllm=self.vllm,
        )
        return _boundary_out(L.module, defer_post, *out)

    def _capture(self, plan, L, residual) -> None:
        """DSpark reads the mean of each target layer's input residual."""
        targets = self.model.dspark_layers_to_capture
        if targets is not None and L.i in targets:
            aux = plan.state["dspark_aux"]
            mean = None
            if _pack_aux_ok(residual, len(aux), plan):
                from aklr.plugins.sglang.models.deepseek_v4 import prologue

                M, H = residual.shape[0], residual.shape[-1]
                packed = plan.state.get("dspark_aux_packed")
                if packed is None:
                    packed = plan.state["dspark_aux_packed"] = torch.empty(
                        (M, len(targets) * H),
                        dtype=residual.dtype,
                        device=residual.device,
                    )
                j = len(aux)
                mean = prologue.hc_mean_into(residual, packed[:, j * H : (j + 1) * H])
            else:
                plan.state.pop("dspark_aux_packed", None)
            aux.append(residual.mean(dim=1) if mean is None else mean)

    def _warm_normal_moe(self) -> None:
        """At the first eager forward, run the normal MoE at 1 to MAX_ROWS rows: the
        decode graphs do not run it, so no later extend step compiles its
        kernels (per rank, no collective)."""
        moe = self._normal_moe
        if moe is None or torch.cuda.is_current_stream_capturing():
            return
        self._normal_moe = None
        x = torch.randn(
            MAX_ROWS,
            HF_CONFIG["hidden_size"],
            dtype=torch.bfloat16,
            device=self.runner.device,
        )
        for m in range(1, MAX_ROWS + 1):
            moe.forward_normal(x[:m], return_moe_output=True)


def _longest_context(hf_config) -> int:
    """The longest context a request can have, SGLang's ``context_len``; a ratio-1
    index source sees that many compressed rows (plus a verify step's draft rows)."""
    return int(hf_config.context_len) + 2 * MAX_ROWS


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_parallel

    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.runner import (
        DSV41MonoRunner,
    )

    first = len(ROPE_TABLES)
    layers = {
        i: _weights(ctx.copies, ROPE_TABLES, model.model.layers[i], i) for i in _covered_layers(hf_config)
    }
    for n, (_, t) in enumerate(ROPE_TABLES[first:], first):
        ctx.copies["rope table", n] = t
    # the peer handshake is collective: every rank gets here (attach agreed first)
    p = get_parallel()
    embed.replicate(model.model.embed_tokens, p.tp_group, ctx.copies)
    device = model.model.layers[0].input_layernorm.weight.device
    runner = DSV41MonoRunner(p.tp_size, p.tp_rank, p.tp_group.cpu_group, device)
    runner.set_deadline(ctx.deadline_s)
    # K2i's builds: one for each index-plane layout the indexed layers read
    page8 = _page8_ratios()
    k2i = {L.ratio in page8 for L in layers.values() if L.module.self_attn.indexer is not None}
    if k2i:
        # the indexer's buffers, sized here: before SGLang sizes the KV cache
        runner.enable_k2i(_longest_context(hf_config))
    # K1 with and without the owed post (the residual arrives settled on the
    # first layer and after an Engram write), K2 with and without top-k rows
    folds = (True, False)
    widths = list(ctx.widths)
    vllm = target_vllm.build(
        p.tp_size,
        p.tp_rank,
        p.tp_group.cpu_group,
        device,
        layers,
        widths,
        {(L.ratio, fold, L.index) for L in layers.values() for fold in folds},
    )
    k1 = k2 = ()
    if vllm is None:
        k1 = {(fold, L.index) for L in layers.values() for fold in folds}
        k2 = {(L.ratio > 0, L.index) for L in layers.values()}
    # Every rank builds every width. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    start = len(widths) * p.tp_rank // p.tp_size
    for m in widths[start:] + widths[:start]:
        runner.precompile(m, k1=k1, k2=k2, k2i=k2i)
    jit_routes.prewarm(device)
    return _Built(model, runner, layers, vllm)


SPEC = MonoSpec(
    name=SPEC_NAMES[TARGET],
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, MAX_ROWS + 1)),
            tp_sizes=(2, 4),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsv4"}),
        server_args={
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # the other FP8 GEMM backends reorder wqkv_a, wq_b and wo_b in place;
            # K1 and K2 read them row-major
            "fp8_gemm_runner_backend": frozenset({"aiter"}),
            # DSpark's target steps are uniform verify steps; other drafts differ
            "speculative_algorithm": frozenset({None, "DSPARK"}),
            # the paged SWA cache, not a request window
            "enable_encoder_swa_bounded_replay": frozenset({False}),
            # the launches record no expert distribution (SGLang's forward sets it per layer)
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        hf_config={**HF_CONFIG, "model_type": "deepseek_v41"},
        env={
            # the MXFP4 experts in aiter's gate / up interleaved layout
            "SGLANG_DSV4_FP4_EXPERTS": True,
            "SGLANG_DSV4_FP4_DEQUANT": False,
            "SGLANG_USE_AITER_MOE_GU_ITLV": True,
            # fp8 nope and bf16 rope, 584 B a token
            "SGLANG_DSV4_KV_LAYOUT": "v4",
            # K1 quantizes the SWA nope payload from f32, as SGLang's fused writer does
            "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE": False,
            # benchmark routing overrides: K2 routes with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
    load_tensor=_load_tensor,
    # a routed step scores every index source with K2i (_select)
    unread_metadata=frozenset({StepMetadata.LOW_RATIO_DECODE_SCHEDULE}),
)
