###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""Schemas of the paged attention ops, one per KV cache storage."""

from __future__ import annotations

from typing import Any

import torch

from aklr.core.schema import Const, Scalar, Schema, Standard, Tensor, Var, default_check
from aklr.ops.attention.baseline import paged_attention_v1_baseline, unified_attention_baseline
from aklr.ops.attention.reference import (
    paged_attention_decode_reference,
    unified_attention_reference,
)

__all__ = [
    "PAGED_ATTENTION_DECODE",
    "PAGED_ATTENTION_DECODE_KV_SPLIT_PAGES",
    "UNIFIED_ATTENTION",
    "UNIFIED_ATTENTION_KV_SPLIT_PAGES",
    "paged_attention_decode_op_schema",
    "unified_attention_op_schema",
]


def _attention_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    # flashinfer's attention tests: rtol = atol = 1e-2 in bfloat16, 1e-3 in float16, the dtype's default
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=rtol,
        atol=atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


def _attention_rows(rows, n_heads, head_dim, split, device):
    # split: vLLM's query, rows of a wider fused projection's output
    stored = torch.randn(rows, 2 if split else 1, n_heads * head_dim, device=device).to(torch.bfloat16)
    return stored[:, 0].unflatten(1, (n_heads, head_dim))


def _kv_caches(n_blocks, block_size, n_kv_heads, head_dim, split, device):
    # split: vLLM's, the key and the value the two halves of each row of one (blocks, heads, tokens, 2 * dim) buffer
    if not split:
        return tuple(
            torch.randn(n_blocks, block_size, n_kv_heads, head_dim, device=device).to(torch.bfloat16)
            for _ in range(2)
        )
    stored = torch.randn(n_blocks, n_kv_heads, block_size, 2 * head_dim, device=device).to(torch.bfloat16)
    return stored[..., :head_dim].transpose(1, 2), stored[..., head_dim:].transpose(1, 2)


def _paged_attention_decode_inputs(
    batch, n_heads, n_kv_heads, head_dim, n_blocks, block_size, max_blocks, split, device, seed
):
    torch.manual_seed(seed)
    key_cache, value_cache = _kv_caches(n_blocks, block_size, n_kv_heads, head_dim, split, device)
    return {
        "query": _attention_rows(batch, n_heads, head_dim, split, device),
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": torch.randint(0, n_blocks, (batch, max_blocks), device=device).to(torch.int32),
        "context_lens": torch.randint(1, max_blocks * block_size + 1, (batch,), device=device).to(
            torch.int32
        ),
        "scale": head_dim**-0.5,
        "kv_cache_storage": "kv_split_pages" if split else "row_major",
    }


def _paged_attention_decode_init(
    *, batch, n_heads, n_kv_heads, head_dim, n_blocks, block_size, max_blocks, device="cuda", seed=0
):
    return _paged_attention_decode_inputs(
        batch, n_heads, n_kv_heads, head_dim, n_blocks, block_size, max_blocks, False, device, seed
    )


def _paged_attention_decode_kv_split_pages_init(
    *, batch, n_heads, n_kv_heads, head_dim, n_blocks, block_size, max_blocks, device="cuda", seed=0
):
    return _paged_attention_decode_inputs(
        batch, n_heads, n_kv_heads, head_dim, n_blocks, block_size, max_blocks, True, device, seed
    )


def _unified_attention_inputs(
    total_q,
    batch,
    n_heads,
    n_kv_heads,
    head_dim,
    n_blocks,
    block_size,
    max_blocks,
    max_seqlen_q,
    sliding_window,
    split,
    device,
    seed,
):
    torch.manual_seed(seed)
    # each request has 1 to max_seqlen_q new tokens, total_q in all: the spare ones dealt at random
    lengths = torch.ones(batch, dtype=torch.long)
    for request in torch.randperm(batch * (max_seqlen_q - 1))[: total_q - batch] % batch:
        lengths[request] += 1
    k, v = _kv_caches(n_blocks, block_size, n_kv_heads, head_dim, split, device)
    context = max_blocks * block_size
    return {
        "q": _attention_rows(total_q, n_heads, head_dim, split, device),
        "k": k,
        "v": v,
        "cu_seqlens_q": torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32).to(device),
        "seqused_k": (lengths + torch.randint(0, context - max_seqlen_q + 1, (batch,)))
        .to(torch.int32)
        .to(device),
        "block_table": torch.randint(0, n_blocks, (batch, max_blocks), device=device).to(torch.int32),
        "softmax_scale": head_dim**-0.5,
        "max_seqlen_q": max_seqlen_q,
        # vLLM's (sliding_window - 1, 0), or (-1, -1) without one
        "window_size": (sliding_window - 1, 0) if sliding_window else (-1, -1),
        "causal": True,
        "kv_cache_storage": "kv_split_pages" if split else "row_major",
    }


def _unified_attention_init(
    *,
    total_q,
    batch,
    n_heads,
    n_kv_heads,
    head_dim,
    n_blocks,
    block_size,
    max_blocks,
    max_seqlen_q,
    sliding_window,
    device="cuda",
    seed=0,
):
    return _unified_attention_inputs(
        total_q,
        batch,
        n_heads,
        n_kv_heads,
        head_dim,
        n_blocks,
        block_size,
        max_blocks,
        max_seqlen_q,
        sliding_window,
        False,
        device,
        seed,
    )


def _unified_attention_kv_split_pages_init(
    *,
    total_q,
    batch,
    n_heads,
    n_kv_heads,
    head_dim,
    n_blocks,
    block_size,
    max_blocks,
    max_seqlen_q,
    sliding_window,
    device="cuda",
    seed=0,
):
    return _unified_attention_inputs(
        total_q,
        batch,
        n_heads,
        n_kv_heads,
        head_dim,
        n_blocks,
        block_size,
        max_blocks,
        max_seqlen_q,
        sliding_window,
        True,
        device,
        seed,
    )


def _paged_attention_decode(name: str, split: bool, init: Any) -> Schema:
    cache = ["n_blocks", "block_size", "n_kv_heads", "head_dim"]
    return Schema(
        name_prefix=name,
        op_type="paged_attention_decode",
        description=(
            "Decode attention over a paged KV cache: query head h of each request attends to KV head"
            " h // (n_heads / n_kv_heads) of the first context_lens tokens its block_tables row names,"
            " block_size tokens a block; with sliding_window > 0 only to the last sliding_window of them."
            " Logits are q . k * scale."
            + (
                " vLLM's storage: the key and the value caches are the two halves of each row of one"
                " (blocks, heads, tokens, 2 * head_dim) buffer, the query rows of a wider buffer."
                if split
                else ""
            )
        ),
        axes={
            "batch": Var("Requests, one query token each."),
            "n_heads": Const("Query heads."),
            "n_kv_heads": Const("KV heads."),
            "head_dim": Const("Values per head."),
            "n_blocks": Var("Blocks of the KV cache."),
            "block_size": Const("Tokens per block."),
            "max_blocks": Var("Blocks per request in the block table."),
        },
        constraints=[
            "n_heads % n_kv_heads == 0",
            "n_heads // n_kv_heads <= 16",
            "head_dim in (64, 128, 256)",
            "block_size in (16, 32)",
            "n_blocks * block_size <= 2147483648",
            "query.dtype in ('bfloat16', 'float16')",
            "key_cache.dtype == value_cache.dtype == query.dtype",
            f"kv_cache_storage == {'kv_split_pages' if split else 'row_major'!r}",
            "kv_cache_dtype == 'auto' and kv_cache_layout == 'NHD'",
            "alibi_slopes is None and logits_soft_cap == 0",
        ],
        inputs={
            "query": Tensor(["batch", "n_heads", "head_dim"], description="The query of each request."),
            "key_cache": Tensor(cache, description="The keys of every cached token, block by block."),
            "value_cache": Tensor(cache, description="The values of every cached token, block by block."),
            "block_tables": Tensor(
                ["batch", "max_blocks"],
                dtype="int32",
                description="The blocks of each request, in context order.",
            ),
            "context_lens": Tensor(
                ["batch"],
                dtype="int32",
                description="Cached tokens of each request, the query's own included.",
            ),
            "scale": Scalar("float32", description="Multiplies every logit."),
            "sliding_window": Scalar(
                "int32",
                optional=True,
                description="Trailing tokens each query reads; 0 reads the whole context.",
            ),
        },
        outputs={
            "out": Tensor(
                ["batch", "n_heads", "head_dim"],
                description="Softmax-weighted sum of the read values.",
                dtype_from="query",
            ),
        },
        standard=Standard(
            reference=paged_attention_decode_reference,
            check=_attention_check,
            init=init,
            baseline=paged_attention_v1_baseline,
        ),
    )


def _unified_attention(name: str, split: bool, init: Any) -> Schema:
    cache = ["n_blocks", "block_size", "n_kv_heads", "head_dim"]
    return Schema(
        name_prefix=name,
        op_type="unified_attention",
        description=(
            "Causal varlen attention of a batch of requests' new tokens over a paged KV cache, vLLM's"
            " multi-token decode: request i owns query rows cu_seqlens_q[i] to cu_seqlens_q[i + 1] and the"
            " first seqused_k[i] tokens its block_table row names, block_size tokens a block, its new"
            " tokens last. Query j of a request of L new tokens sits at position seqused_k - L + j and"
            " reads the tokens at or before it; with sliding_window > 0 only the last sliding_window of"
            " them. Query head h reads KV head h // (n_heads / n_kv_heads); logits are q . k *"
            " softmax_scale."
            + (
                " vLLM's storage: k and v are the two halves of each row of one"
                " (blocks, heads, tokens, 2 * head_dim) buffer, q rows of a wider buffer."
                if split
                else ""
            )
        ),
        axes={
            "total_q": Var("Query rows of every request."),
            "batch": Var("Requests."),
            "n_offsets": Var("Offsets in cu_seqlens_q: batch + 1."),
            "n_heads": Const("Query heads."),
            "n_kv_heads": Const("KV heads."),
            "head_dim": Const("Values per head."),
            "n_blocks": Var("Blocks of the KV cache."),
            "block_size": Const("Tokens per block."),
            "max_blocks": Var("Blocks per request in the block table."),
            # vLLM's decode_max_query_len; aiter reads 1 as one query per request
            "max_seqlen_q": Const("The longest request's new tokens.", abbrev="max_q"),
            "sliding_window": Const(
                "Tokens a query reads, itself included; 0 reads the whole context.",
                abbrev="window",
                extract=lambda call: call.get("window_size", (-1, -1))[0] + 1,
            ),
        },
        constraints=[
            "n_offsets == batch + 1",
            "batch <= total_q <= batch * max_seqlen_q",
            "n_heads % n_kv_heads == 0",
            "n_heads // n_kv_heads <= 16",
            "head_dim in (64, 128, 256)",
            "block_size in (16, 32, 64)",
            "n_blocks * block_size <= 2147483648",
            "max_seqlen_q <= max_blocks * block_size",
            "q.dtype in ('bfloat16', 'float16')",
            "k.dtype == v.dtype == q.dtype",
            f"kv_cache_storage == {'kv_split_pages' if split else 'row_major'!r}",
            "causal and softcap == 0",
            "window_size[0] == -1 or window_size[1] == 0",
            # aiter scales the logits and values by the descales' first element
            "q_descale is None and k_descale is None and v_descale is None",
        ],
        inputs={
            "q": Tensor(
                ["total_q", "n_heads", "head_dim"],
                description="The new tokens' queries, request after request.",
            ),
            "k": Tensor(cache, description="The keys of every cached token, block by block."),
            "v": Tensor(cache, description="The values of every cached token, block by block."),
            "cu_seqlens_q": Tensor(
                ["n_offsets"],
                dtype="int32",
                description="Where each request's query rows start, then total_q.",
            ),
            "seqused_k": Tensor(
                ["batch"],
                dtype="int32",
                description="Cached tokens of each request, its new tokens included.",
            ),
            "block_table": Tensor(
                ["batch", "max_blocks"],
                dtype="int32",
                description="The blocks of each request, in context order.",
            ),
            "softmax_scale": Scalar("float32", description="Multiplies every logit."),
        },
        outputs={
            "out": Tensor(
                ["total_q", "n_heads", "head_dim"],
                description="Softmax-weighted sum of the read values, per query row.",
                dtype_from="q",
            ),
        },
        standard=Standard(
            reference=unified_attention_reference,
            check=_attention_check,
            init=init,
            baseline=unified_attention_baseline,
        ),
    )


PAGED_ATTENTION_DECODE = _paged_attention_decode(
    "paged_attention_decode", False, _paged_attention_decode_init
)
PAGED_ATTENTION_DECODE_KV_SPLIT_PAGES = _paged_attention_decode(
    "paged_attention_decode_kv_split_pages", True, _paged_attention_decode_kv_split_pages_init
)
UNIFIED_ATTENTION = _unified_attention("unified_attention", False, _unified_attention_init)
UNIFIED_ATTENTION_KV_SPLIT_PAGES = _unified_attention(
    "unified_attention_kv_split_pages", True, _unified_attention_kv_split_pages_init
)


def paged_attention_decode_op_schema(
    *_: Any,
    kv_cache_dtype: str = "auto",
    kv_cache_layout: str = "NHD",
    kv_cache_storage: str = "row_major",
    alibi_slopes: Any = None,
    logits_soft_cap: float = 0.0,
    **__: Any,
) -> Schema | None:
    if kv_cache_dtype != "auto" or kv_cache_layout != "NHD" or alibi_slopes is not None or logits_soft_cap:
        return None
    return {"row_major": PAGED_ATTENTION_DECODE, "kv_split_pages": PAGED_ATTENTION_DECODE_KV_SPLIT_PAGES}.get(
        kv_cache_storage
    )


paged_attention_decode_op_schema.schemas = (PAGED_ATTENTION_DECODE, PAGED_ATTENTION_DECODE_KV_SPLIT_PAGES)


def unified_attention_op_schema(
    *_: Any,
    window_size: tuple[int, int] = (-1, -1),
    causal: bool = True,
    kv_cache_storage: str = "row_major",
    softcap: float = 0.0,
    q_descale: Any = None,
    k_descale: Any = None,
    v_descale: Any = None,
    **__: Any,
) -> Schema | None:
    if not causal or softcap or not (window_size[0] == -1 or window_size[1] == 0):
        return None
    if q_descale is not None or k_descale is not None or v_descale is not None:
        return None
    return {"row_major": UNIFIED_ATTENTION, "kv_split_pages": UNIFIED_ATTENTION_KV_SPLIT_PAGES}.get(
        kv_cache_storage
    )


unified_attention_op_schema.schemas = (UNIFIED_ATTENTION, UNIFIED_ATTENTION_KV_SPLIT_PAGES)
