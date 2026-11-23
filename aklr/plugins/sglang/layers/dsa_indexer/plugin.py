# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSA indexer's preparation in sglang's Indexer on HIP (srt/layers/attention/dsa/
dsa_indexer.py): the query heads and the key it scores with, before the scores. The key
goes into the layer's index-K cache; the scores and their top-k stay on sglang."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
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
    axis,
    derived_axis,
)
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin

__all__ = [
    "DSA_INDEXER_KEY_PREPARE",
    "DSA_INDEXER_PREPARE",
    "FP8_MAX",
    "HEAD_DIM",
    "PAGE_SIZE",
    "ROPE_DIM",
    "SCALE_FLOOR",
    "declare",
    "dsa_indexer_key_prepare",
    "dsa_indexer_prepare",
    "family",
]


# An indexer head and key: the Hadamard transform's length and the cache's 128 bytes a key
HEAD_DIM = 128
# The RoPE part: a head's and a key's first 64 values, in interleaved pairs (2i, 2i + 1)
ROPE_DIM = 64
# Keys per index-K page: aiter's preshuffled layout of 16 x 16 tiles
PAGE_SIZE = 64
# A key's or head's fp8 scale: 2 ** ceil(log2(max(amax, SCALE_FLOOR) / FP8_MAX))
SCALE_FLOOR = 1e-4
FP8_MAX = 448.0

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_INT64 = DTypeSpec(any_of=(DType.INT64,))
_FINITE = FiniteDistribution()
# The var axes' bounds: pages of an index-K pool, rows of a rotary table
_MAX_PAGES = 2**24
_MAX_POSITIONS = 2**24

# The bounds, from fp32 emulations of sglang's path and of the kernels' split-K sums at GLM-5.2's
# shapes (1 to 256 tokens with 30 input sets each, 1024 and 4096 with 4), against an exact
# reference (the float32 one sums in one more such order): the projections' other summation
# order rounds a few values to the next bf16, so a few fp8 values sit one fp8 step from the
# reference's, and a few rows (2 of 3.4e6 query heads,
# none of 1.1e5 keys) have their amax cross a power of two times 448, the fp8 scale then twice
# or half the reference's and the values half or twice. The query side takes SQNR, which admits
# such a row: one of a token's 32 heads leaves 15 dB, and the emulations stay at 43 dB or more.
# References broken on purpose (no or NeoX RoPE, cos and sin swapped, the position one off, no
# Hadamard transform or its 128^-0.5, a scale not a power of two, n_heads^-0.5, the scale or
# 128^-0.5 left out of head_weights) come to 10 dB or less on q_fp8 or 9 dB on head_weights.
_QUERY_BOUND = SqnrBound(min_sqnr_db=12.0)
_WEIGHTS_BOUND = SqnrBound(min_sqnr_db=10.0)
# The cache is bounded per byte, so that a scale's bytes count as much as a key's: one fp8 step
# (0.125 of a value) and 2 below a row's largest values (a few one-ulp bf16 differences summed
# by the Hadamard transform). The emulations meet it everywhere; a key missing, not
# preshuffled, without LayerNorm or its bias, or at twice its scale, and a scale not stored,
# fail it. A scale step of one key fails it too unless the pool's bytes outnumber that key's
# 128 values by more than 1 / max_violation_fraction, as the bench's pools of the fewest pages
# do from 3200 tokens on; a scale stored at exactly twice the one its values were divided by
# is not seen (its fp32 bytes differ in a sign bit of fp8 zero and at most one fp8 step).
_CACHE_BOUND = ToleranceBound(absolute_tolerance=2.0, relative_tolerance=0.125)
# The cache's bytes before the call: 64 to 256, far from a stored scale's bytes (0, 0, 0 and
# about 1.5 as fp8 values), so that a scale that is not stored fails the bound
_UNWRITTEN = LogUniformDistribution(log2_lo=6, log2_hi=8)
_COMPARE = CompareSpec(max_violation_fraction=3e-4)
# no_values has no element to compare
_EMPTY_BOUND = SqnrBound(min_sqnr_db=0.0)

_KEY_AXES = {
    "n_tokens": axis("Tokens in this call, padding rows included.", 1, 65536, is_var=True),
    "model_dim": axis("The hidden states' width: wk's and weights_proj's input.", 128, 16384),
    "n_positions": axis("Rows of the rotary tables.", 1, _MAX_POSITIONS, is_var=True, any_size=True),
    "n_pages": axis("Pages of the index-K cache.", 1, _MAX_PAGES, is_var=True, any_size=True),
}
# distinct slots: a token writes its own
_KEY_CONSTRAINTS = (f"n_tokens <= {PAGE_SIZE} * n_pages",)

_KEY_INPUTS = {
    "x": TensorSpec(
        description="Each token's hidden states, the layer's input.",
        shape=("n_tokens", "model_dim"),
        dtype=_BF16,
        distribution=_FINITE,
    ),
    "wk": TensorSpec(
        description="wk: the key's projection.",
        shape=(HEAD_DIM, "model_dim"),
        dtype=_BF16,
        distribution=LecunNormalDistribution(fan_in="model_dim"),
        lifetime="weight",
    ),
    "k_norm_weight": TensorSpec(
        description="k_norm's (LayerNorm) weight.",
        shape=(HEAD_DIM,),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "k_norm_bias": TensorSpec(
        description="k_norm's bias.",
        shape=(HEAD_DIM,),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "cos_cache": TensorSpec(
        description="The rotary cosines, a row per position and a column per RoPE pair.",
        shape=("n_positions", ROPE_DIM // 2),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "sin_cache": TensorSpec(
        description="The rotary sines, as cos_cache.",
        shape=("n_positions", ROPE_DIM // 2),
        dtype=_BF16,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "positions": TensorSpec(
        description="Each token's position: its row of the rotary tables.",
        shape=("n_tokens",),
        dtype=_INT64,
        distribution=IntRangeDistribution(lo=0, hi="n_positions - 1"),
    ),
    "out_cache_loc": TensorSpec(
        description=(
            f"Each token's slot of the index-K cache (page slot // {PAGE_SIZE}, key "
            f"slot % {PAGE_SIZE}), distinct but for sglang's graph padding rows, which all write "
            "slot 0, a slot no sequence reads."
        ),
        shape=("n_tokens",),
        dtype=_INT64,
        distribution=IntRangeDistribution(lo=0, hi=f"{PAGE_SIZE} * n_pages - 1", distinct=True),
    ),
    "index_k_cache": TensorSpec(
        description=(
            f"This layer's index-K cache, as sglang views it: pages of {PAGE_SIZE} keys, their "
            f"{HEAD_DIM} fp8 values in 16 x 16 tiles (16 keys by 16 values) and then their fp32 "
            "scales, each page's bytes compared as fp8 values. The tokens' keys are stored "
            "into it."
        ),
        shape=("n_pages", PAGE_SIZE, HEAD_DIM + 4),
        dtype=_FP8,
        distribution=_UNWRITTEN,
        mutable=True,
        error_bounds=_CACHE_BOUND,
    ),
    "k_norm_eps": ScalarSpec(
        description="k_norm's epsilon.",
        dtype=DType.FLOAT32,
        interval=(0.0, 1.0),
        default=1e-6,
    ),
}

_KEY_TEXT = (
    "Each token's key is LayerNorm(x wk^T) (k_norm_weight, k_norm_bias, k_norm_eps), its "
    f"first {ROPE_DIM} values turned by RoPE in interleaved pairs (2i, 2i + 1) by cos_cache "
    "and sin_cache at the token's position, then the Hadamard transform over its "
    f"{HEAD_DIM} values (Sylvester order) times {HEAD_DIM}^-0.5, each step rounded to bf16 as "
    "sglang rounds it; it is stored into index_k_cache at its out_cache_loc slot as fp8 "
    f"values at a scale of 2^ceil(log2(max(amax, {SCALE_FLOOR}) / {FP8_MAX:g})) of its values, "
    "and that scale in fp32."
)

DSA_INDEXER_PREPARE = OpContract(
    contract_id="dsa_indexer_prepare@1.0",
    description=(
        "sglang's DSA indexer before its scores on one rank (Indexer.forward_cuda on HIP, the "
        "Hadamard-rotated index-K format). Each query head is a head of q_lora wq_b^T, rounded "
        "to bf16, turned and transformed as the key, and quantized to q_fp8 at its own scale as "
        "the key is; its head weight is bf16(bf16(x weights_proj^T) * n_heads^-0.5) times that "
        f"scale times {HEAD_DIM}^-0.5. {_KEY_TEXT}"
    ),
    op_type="dsa_indexer",
    axes={
        **_KEY_AXES,
        "q_lora_rank": axis("q_lora's width: wq_b's input.", 128, 16384),
        "n_heads": axis("Indexer heads.", 1, 128),
        "index_cols": derived_axis("wq_b's rows: the heads side by side.", f"n_heads * {HEAD_DIM}"),
    },
    constraints=_KEY_CONSTRAINTS,
    inputs={
        "x": _KEY_INPUTS["x"],
        "q_lora": TensorSpec(
            description="Each token's query latent after q_a_layernorm: wq_b's input.",
            shape=("n_tokens", "q_lora_rank"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "wq_b": TensorSpec(
            description="wq_b: the query heads' projection, each head's rows in turn.",
            shape=("index_cols", "q_lora_rank"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="q_lora_rank"),
            lifetime="weight",
        ),
        "wk": _KEY_INPUTS["wk"],
        "weights_proj": TensorSpec(
            description="weights_proj: each head's weight, a row per head.",
            shape=("n_heads", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        **{n: s for n, s in _KEY_INPUTS.items() if n not in ("x", "wk")},
    },
    outputs={
        "q_fp8": TensorSpec(
            description="Each token's query heads, turned and transformed, in fp8 at their scales.",
            shape=("n_tokens", "n_heads", HEAD_DIM),
            dtype=_FP8,
            error_bounds=_QUERY_BOUND,
        ),
        "head_weights": TensorSpec(
            description="Each token's head weights, each with its head's fp8 scale folded in.",
            shape=("n_tokens", "n_heads"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=_WEIGHTS_BOUND,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_COMPARE,
)

DSA_INDEXER_KEY_PREPARE = OpContract(
    contract_id="dsa_indexer_key_prepare@1.0",
    description=(
        "sglang's DSA indexer's key-only path on one rank (Indexer._forward_cuda_k_only on HIP, "
        "the Hadamard-rotated index-K format), which stores the tokens' keys and computes no "
        f"query. {_KEY_TEXT}"
    ),
    op_type="dsa_indexer",
    axes=_KEY_AXES,
    constraints=_KEY_CONSTRAINTS,
    inputs=_KEY_INPUTS,
    outputs={
        "no_values": TensorSpec(
            description=(
                "No values: the op's one result is index_k_cache, and a contract declares an output."
            ),
            shape=(0,),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=_EMPTY_BOUND,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_COMPARE,
)


family = Plugin("sglang")
family.define(DSA_INDEXER_PREPARE)
family.define(DSA_INDEXER_KEY_PREPARE)
family.build()


# Indexer's quantization, as deepseek_v2 builds it: one scale per 128 values, rounded up to a
# power of two
_QUANT_BLOCK = 128
_SCALE_FMT = "ue8m0"


@family.op
def dsa_indexer_prepare(
    x,
    q_lora,
    wq_b,
    wk,
    weights_proj,
    k_norm_weight,
    k_norm_bias,
    cos_cache,
    sin_cache,
    positions,
    out_cache_loc,
    index_k_cache,
    k_norm_eps=1e-6,
    q_fp8=None,
    head_weights=None,
): ...


@family.op
def dsa_indexer_key_prepare(
    x,
    wk,
    k_norm_weight,
    k_norm_bias,
    cos_cache,
    sin_cache,
    positions,
    out_cache_loc,
    index_k_cache,
    k_norm_eps=1e-6,
    no_values=None,
): ...


def _forward_cuda(
    self, x, q_lora, positions, forward_batch, layer_id, return_indices=True, use_cuda_graph=True
):
    """forward_cuda's standard path with its preparation (_get_q_k_bf16, act_quant,
    _store_index_k_cache, _get_logits_head_gate) as the op, then sglang's scores and top-k.
    sglang keeps every other branch: the key-only one (routed on _forward_cuda_k_only), its
    gfx950 fused decode, the graph split ops, the fused writers, dual streams and CP."""
    import torch
    from sglang.srt.layers.attention.dsa import dsa_indexer as indexer
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.runner import get_is_capture_mode

    hidden = _hidden_bf16(x)
    mode = forward_batch.forward_mode
    if (
        hidden is None
        or not isinstance(q_lora, torch.Tensor)
        or _module_falls_back(self, ("wq_b", "wk", "weights_proj"))
        or _batch_falls_back(indexer, forward_batch)
        or indexer.is_graph_dsa_split_op_surface(forward_batch)
        or self.dsa_enable_prefill_cp
        # the key-only branch, whose route is _forward_cuda_k_only's
        or self._should_skip_logits_computation(forward_batch)
        or _gfx950_fused_decode_may_serve(self, mode, hidden)
    ):
        return FallbackReason.OUT_OF_CONTRACT
    enable_dual_stream = (
        self.alt_stream is not None
        and get_is_capture_mode()
        and 0 < q_lora.shape[0] <= indexer.DUAL_STREAM_TOKEN_THRESHOLD
    )
    if enable_dual_stream:
        return FallbackReason.OUT_OF_CONTRACT
    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
    pool, index_k_cache = _index_k_cache(layer_id)
    if metadata is None or index_k_cache is None:
        return FallbackReason.OUT_OF_CONTRACT
    arguments = _key_arguments(self, hidden, positions, forward_batch, index_k_cache) | {
        "q_lora": q_lora,
        "wq_b": self.wq_b.weight,
        "weights_proj": self.weights_proj.weight,
    }
    reason = dsa_indexer_prepare.fallback_reason(**arguments)
    if reason is not None:
        return reason
    _invalidate(pool, layer_id)
    result = dsa_indexer_prepare(**arguments)
    if isinstance(result, FallbackReason):
        return result
    q_fp8, head_weights = result
    # [tokens, heads, 1], as _get_logits_head_gate returns them
    weights = head_weights.unsqueeze(-1)
    if forward_batch.seq_lens.numel() == 0:
        topk_result = torch.full(
            (hidden.shape[0], self.index_topk), -1, dtype=torch.int, device=hidden.device
        )
    elif mode.is_decode_or_idle() or mode.is_target_verify() or mode.is_draft_extend_v2():
        topk_result = self._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)
    else:
        topk_result = self._get_topk_ragged(False, forward_batch, layer_id, q_fp8, weights, metadata)
    topk_result = indexer._broadcast_indexer_topk_from_rank0(topk_result)
    return indexer.maybe_capture_indexer_topk(layer_id, topk_result)


def _forward_cuda_k_only(
    self,
    x,
    positions,
    forward_batch,
    layer_id,
    act_quant,
    metadata,
    return_indices=True,
    *,
    num_tokens=None,
    topk_result=None,
):
    """_forward_cuda_k_only's standard branch with its keys (_get_k_bf16,
    _store_index_k_cache) as the op, then sglang's select-all top-k on zero logits. The graph
    split ops' calls (num_tokens) and the fused writers' branches stay on sglang."""
    import torch
    from sglang.srt.layers.attention.dsa import dsa_indexer as indexer

    hidden = _hidden_bf16(x)
    mode = forward_batch.forward_mode
    if (
        num_tokens is not None
        or hidden is None
        or not (mode.is_extend_without_speculative() or mode.is_decode_or_idle())
        or _module_falls_back(self, ("wk",))
        or _batch_falls_back(indexer, forward_batch)
        # sglang would make it contiguous on the batch itself
        or not forward_batch.out_cache_loc.is_contiguous()
    ):
        return FallbackReason.OUT_OF_CONTRACT
    pool, index_k_cache = _index_k_cache(layer_id)
    if index_k_cache is None:
        return FallbackReason.OUT_OF_CONTRACT
    arguments = _key_arguments(self, hidden, positions, forward_batch, index_k_cache)
    reason = dsa_indexer_key_prepare.fallback_reason(**arguments)
    if reason is not None:
        return reason
    _invalidate(pool, layer_id)
    result = dsa_indexer_key_prepare(**arguments)
    if isinstance(result, FallbackReason):
        return result
    if not return_indices:
        return None
    # the select-all top-k: with every length within index_topk, zero logits pick each slot
    seq_lens_expanded = metadata.get_seqlens_expanded()
    dummy_logits = torch.zeros(
        seq_lens_expanded.shape[0], self.index_topk, dtype=torch.float32, device=hidden.device
    )
    raw_topk_result = metadata.topk_transform(dummy_logits, self.index_topk)
    if topk_result is not None:
        topk_result[: raw_topk_result.shape[0]] = raw_topk_result
        return None
    return raw_topk_result


def _hidden_bf16(x):
    """The bf16 hidden states _get_q_k_bf16 and _get_k_bf16 project: x, or the third of
    fused_rms_fp8_group_quant's (fp8, scale, bf16); None for anything else."""
    import torch

    if isinstance(x, tuple):
        x = x[2] if len(x) == 3 else None
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        return None
    return x


def _module_falls_back(module, linears) -> bool:
    """Whether the layer prepares its index keys other than the op: keys not Hadamard-rotated
    (a fused writer), an RMSNorm k_norm, NeoX RoPE or rotary tables the op cannot read, another
    quantization, or a LoRA adapter on one of ``linears``."""
    import torch
    from sglang.srt.layers.layernorm import LayerNorm

    rotary = module.rotary_emb
    tables = [getattr(rotary, name, None) for name in ("cos_cache", "sin_cache")]
    k_norm = module.k_norm
    return (
        module.use_dsa_indexer_fusion
        or module.use_aiter_fused_fp8_writer
        or not isinstance(k_norm, LayerNorm)
        or not (k_norm.elementwise_affine and k_norm.use_bias)
        or module.is_neox_style
        or (module.head_dim, module.rope_head_dim) != (HEAD_DIM, ROPE_DIM)
        or (module.block_size, module.scale_fmt) != (_QUANT_BLOCK, _SCALE_FMT)
        or not all(
            isinstance(t, torch.Tensor)
            and t.dtype == torch.bfloat16
            and t.is_contiguous()
            and t.device == k_norm.weight.device
            and t.shape[-1] == ROPE_DIM // 2
            for t in tables
        )
        or any(getattr(getattr(module, name, None), "set_lora", False) for name in linears)
    )


def _batch_falls_back(indexer, forward_batch) -> bool:
    """Whether the call is one the op's branch does not reach in sglang: inside a piecewise or
    breakable CUDA graph (sglang's split ops), or with context parallelism (CP gathers the
    keys); and whether sglang stores the keys elsewhere than aiter's preshuffled fp8 pages."""
    return (
        indexer._is_in_piecewise_or_breakable_cuda_graph()
        or forward_batch.attn_cp_metadata is not None
        or indexer.is_cp_active(forward_batch)
        or not (indexer._use_aiter and indexer._use_aiter_preshuffle)
        or indexer._is_fp8_fnuz
    )


def _gfx950_fused_decode_may_serve(module, mode, hidden) -> bool:
    """Whether sglang's gfx950 fused decode indexer (four kernels, logits and top-k included)
    may take the call: it is left to sglang, which falls through to its standard path when it
    declines."""
    if not module.use_gfx950_fused_indexer or not (mode.is_decode_or_idle() or mode.is_target_verify()):
        return False
    from sglang.srt.layers.attention.dsa.gfx950_fused_indexer import MAX_ROWS

    return hidden.shape[0] <= MAX_ROWS


def _index_k_cache(layer_id):
    """(pool, the layer's index-K pages [pages, PAGE_SIZE, HEAD_DIM + 4] in fp8), as
    _store_index_k_cache views them; (pool, None) where sglang would store elsewhere (another
    page size, a layer-split pool that owns only some layers)."""
    import torch
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    pool = get_token_to_kv_pool()
    if pool.index_page_size != PAGE_SIZE or hasattr(pool, "_is_layer_owned"):
        return pool, None
    buffer = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
    if not buffer.is_contiguous() or buffer.numel() % (PAGE_SIZE * (HEAD_DIM + 4)):
        return pool, None
    return pool, buffer.view(-1, PAGE_SIZE, HEAD_DIM + 4).view(torch.float8_e4m3fn)


def _invalidate(pool, layer_id) -> None:
    """As _store_index_k_cache does before its store."""
    if hasattr(pool, "invalidate_index_buffer_for_layer"):
        pool.invalidate_index_buffer_for_layer(layer_id)


def _key_arguments(module, hidden, positions, forward_batch, index_k_cache):
    """The key's arguments of both ops: the layer's wk, k_norm and rotary tables, the batch's
    positions and slots (aiter stores the first of them, one a key row)."""
    rotary = module.rotary_emb
    out_cache_loc = forward_batch.out_cache_loc
    if out_cache_loc.shape[0] > hidden.shape[0]:
        out_cache_loc = out_cache_loc[: hidden.shape[0]]
    return {
        "x": hidden,
        "wk": module.wk.weight,
        "k_norm_weight": module.k_norm.weight,
        "k_norm_bias": module.k_norm.bias,
        "cos_cache": rotary.cos_cache.view(-1, ROPE_DIM // 2),
        "sin_cache": rotary.sin_cache.view(-1, ROPE_DIM // 2),
        "positions": positions,
        "out_cache_loc": out_cache_loc,
        "index_k_cache": index_k_cache,
        "k_norm_eps": module.k_norm.variance_epsilon,
    }


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer.forward_cuda",
        versions=("0.5.23",),
        replace=_forward_cuda,
    )
    plugin.wrap(
        "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._forward_cuda_k_only",
        versions=("0.5.22", "0.5.23"),
        replace=_forward_cuda_k_only,
    )
