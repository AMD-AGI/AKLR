###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import Any

import torch

from aklr.core.schema import Const, Scalar, Schema, Tensor, Var, default_check
from aklr.ops.mla.reference import (
    flash_mla_with_kvcache_dense_reference,
    flash_mla_with_kvcache_ragged_reference,
    flash_mla_with_kvcache_sparse_reference,
    fp8_paged_mqa_logits_reference,
)

__all__ = [
    "FLASH_MLA_DENSE",
    "FLASH_MLA_RAGGED",
    "FLASH_MLA_SPARSE",
    "FP8_PAGED_MQA_LOGITS",
    "flash_mla_with_kvcache_op_schema",
    "fp8_paged_mqa_logits_op_schema",
]


def _flash_mla_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    # aiter's flash_mla tests: allclose(atol=1e-2, rtol=1e-2)
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=1e-2 if rtol is None else rtol,
        atol=1e-2 if atol is None else atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


def _dsv4_fp8_kv_cache(n_pages, page_size, device):
    payload_bytes, page_bytes = page_size * 576, page_size * 584
    page_stride = (page_bytes + 575) // 576 * 576
    pool = torch.zeros(n_pages, page_stride, dtype=torch.uint8, device=device)
    distinct = min(n_pages, 251)
    values = torch.randn(distinct, page_size, 512, device=device)
    codes = values[..., :448].clamp(-448, 448).to(torch.float8_e4m3fn)
    rope = values[..., 448:].to(torch.bfloat16)
    payloads = torch.cat((codes.view(torch.uint8), rope.view(torch.uint8)), dim=-1)
    payloads = payloads.flatten(1)
    for start in range(0, n_pages, distinct):
        stop = min(start + distinct, n_pages)
        pool[start:stop, :payload_bytes] = payloads[: stop - start]
    slots = pool[:, payload_bytes:page_bytes].unflatten(1, (page_size, 8))
    slots[..., :7] = torch.randint(124, 128, (n_pages, page_size, 7), dtype=torch.uint8, device=device)
    return pool.view(torch.float8_e4m3fn).as_strided((n_pages, page_size, 1, 584), (page_stride, 584, 584, 1))


def _flash_mla_with_kvcache_sparse_init(
    *,
    batch,
    n_heads,
    n_pages,
    page_size,
    n_indices,
    n_extra_pages=None,
    extra_page_size=None,
    n_extra_indices=None,
    device="cuda",
    seed=0,
):
    torch.manual_seed(seed)
    call = {
        "q": torch.randn(batch, 1, n_heads, 512, device=device).to(torch.bfloat16),
        "kv_cache": _dsv4_fp8_kv_cache(n_pages, page_size, device),
        "sparse_indices": torch.randint(-1, n_pages * page_size, (batch, 1, n_indices), device=device).to(
            torch.int32
        ),
        "sparse_lens": torch.randint(0, n_indices + 1, (batch,), device=device).to(torch.int32),
        "sinks": torch.randn(n_heads, device=device),
        "sm_scale": 512**-0.5,
        "kv_scale": 1.0,
        "causal": False,
        "kv_cache_format": "dsv4_fp8",
    }
    if extra_page_size is not None:
        call["extra_kv_cache"] = _dsv4_fp8_kv_cache(n_extra_pages, extra_page_size, device)
        call["extra_sparse_indices"] = torch.randint(
            -1,
            n_extra_pages * extra_page_size,
            (batch, 1, n_extra_indices),
            device=device,
        ).to(torch.int32)
        call["extra_sparse_lens"] = torch.randint(0, n_extra_indices + 1, (batch,), device=device).to(
            torch.int32
        )
    return call


def _flash_mla_with_kvcache_dense_init(
    *,
    batch,
    n_heads,
    n_pages,
    page_size,
    table_width,
    device="cuda",
    seed=0,
):
    # shortcut: an fp8 pool only, the one dense kernel's; draw bf16 too once a bf16 kernel comes
    torch.manual_seed(seed)
    return {
        "q": torch.randn(batch, 1, n_heads, 576, device=device).to(torch.bfloat16),
        "kv_cache": torch.randn(n_pages, page_size, 1, 576, device=device).to(torch.float8_e4m3fn),
        "block_table": torch.randint(0, n_pages * page_size, (batch, table_width), device=device).to(
            torch.int32
        ),
        "cache_seqlens": torch.randint(1, table_width + 1, (batch,), device=device).to(torch.int32),
        "sm_scale": 192**-0.5,  # Kimi K3's: 128 nope + 64 rope dims
        "kv_scale": 1.0,
        "causal": False,
        "kv_cache_format": "fp8_e4m3",
    }


def _flash_mla_with_kvcache_ragged_init(
    *,
    batch,
    query_length,
    n_heads,
    n_pages,
    page_size,
    table_width,
    device="cuda",
    seed=0,
):
    # shortcut: an fp8 pool only, the one ragged kernel's; draw bf16 too once a bf16 kernel comes
    torch.manual_seed(seed)
    # each request reads at least query_length rows; the spare rows are split by weights exp(N(0, 1) clipped to +-2)
    spare = table_width - batch * query_length
    shares = torch.randn(batch, dtype=torch.float64).clamp(-2, 2).exp().cumsum(0)
    ends = (shares / shares[-1:] * spare).floor().long()
    ends[-1:] = spare
    ends += query_length * torch.arange(1, batch + 1)
    return {
        "q": torch.randn(batch, query_length, n_heads, 576, device=device).to(torch.bfloat16),
        "kv_cache": torch.randn(n_pages, page_size, 1, 576, device=device).to(torch.float8_e4m3fn),
        "block_table": torch.randint(0, n_pages * page_size, (table_width,), device=device).to(torch.int32),
        "cu_seqlens_k": torch.cat((ends.new_zeros(1), ends)).to(torch.int32).to(device),
        "sm_scale": 192**-0.5,  # Kimi K3's: 128 nope + 64 rope dims
        "kv_scale": 1.0,
        "causal": query_length > 1,
        "kv_cache_format": "fp8_e4m3",
    }


def _flash_mla(
    name: str,
    reference: Any,
    head_dim: int,
    required: tuple[str, ...],
    absent: tuple[str, ...],
    block_table: list[str],
    constraints: tuple[str, ...],
    init: Any = None,
) -> Schema:
    kv_cache = Tensor(
        ["n_pages", "page_size", "one", "kv_dim"],
        description=(
            "The KV pool: packed dsv4 slots of 584 bytes, 576-byte payloads then"
            " 8-byte scales. sglang's DeepSeekV4TokenToKVPool pads each page to"
            " ceil_div(page_size * 584, 576) * 576 bytes."
            if head_dim == 512
            else "The KV pool: one 576-value row a page."
        ),
    )
    return Schema(
        name_prefix=name,
        op_type="flash_mla_with_kvcache",
        description=(
            "MLA attention over a paged KV pool, FlashMLA's API. Sparse: each token "
            "attends to the slots its indices name, in the pool and the optional extra "
            "one; sinks join the normalisation only. Dense: each request attends to "
            "its context rows; q's rope part is last, scores are q . row * sm_scale * "
            "kv_scale and values row[:head_dim_v] * kv_scale. With causal, query i of "
            "a request of length L reads its first L - query_length + i + 1 rows."
        ),
        axes={
            "batch": Var("Rows: one token each (sparse) or one request each (dense)."),
            "query_length": Const("Queries per row."),
            "n_heads": Const("Query heads."),
            "n_pages": Var("Pages of the KV pool."),
            "page_size": Const("Slots per page of the KV pool."),
            "n_indices": Const("Sparse indices per token."),
            "n_extra_pages": Var("Pages of the extra KV pool."),
            "extra_page_size": Const("Slots per page of the extra KV pool."),
            "n_extra_indices": Const("Extra sparse indices per token."),
            "table_width": Var("Columns of the 2-D block_table, or length of the flat one."),
            "n_offsets": Var("Offsets in cu_seqlens_k: batch + 1."),
            "head_dim": Const("Query values per head.", abbrev="", value=head_dim),
            "kv_dim": Const(
                "Bytes (sparse) or values (dense) per pool slot.",
                abbrev="",
                value=584 if head_dim == 512 else 576,
            ),
            "v_dim": Const("Output values per head.", abbrev="", value=512),
            "one": Const("A single KV head.", abbrev="", value=1),
        },
        constraints=[
            *(f"{t} is None" for t in absent),
            "n_pages * page_size <= 2147483648",
            *constraints,
        ],
        inputs={
            "q": Tensor(
                ["batch", "query_length", "n_heads", "head_dim"],
                dtype="bfloat16",
                description="The queries of each row; dense: the rope part last.",
            ),
            "kv_cache": kv_cache,
            "sparse_indices": Tensor(
                ["batch", "one", "n_indices"],
                dtype="int32",
                optional="sparse_indices" not in required,
                description="Slots of the KV pool each token reads; negative: skipped.",
            ),
            "sparse_lens": Tensor(
                ["batch"],
                dtype="int32",
                optional=True,
                description="Leading sparse_indices each token reads; all when absent.",
            ),
            "extra_kv_cache": Tensor(
                ["n_extra_pages", "extra_page_size", "one", "kv_dim"],
                optional=True,
                description=("The extra KV pool, packed dsv4 slots in pages padded as kv_cache's."),
            ),
            "extra_sparse_indices": Tensor(
                ["batch", "one", "n_extra_indices"],
                dtype="int32",
                optional=True,
                description=("Slots of the extra KV pool each token reads; negative: skipped."),
            ),
            "extra_sparse_lens": Tensor(
                ["batch"],
                dtype="int32",
                optional=True,
                description=("Leading extra_sparse_indices each token reads; all when absent."),
            ),
            "block_table": Tensor(
                block_table,
                dtype="int32",
                optional="block_table" not in required,
                description=(
                    "Pool rows of each request: a (batch, width) table read up to "
                    "cache_seqlens, or every request's rows back to back."
                ),
            ),
            "cache_seqlens": Tensor(
                ["batch"],
                dtype="int32",
                optional="cache_seqlens" not in required,
                description=("Leading block_table entries each request reads, at least one."),
            ),
            "cu_seqlens_k": Tensor(
                ["n_offsets"],
                dtype="int32",
                optional="cu_seqlens_k" not in required,
                description=("Where each request's rows start in block_table, then its length."),
            ),
            "causal": Scalar(
                "bool",
                optional=True,
                description=(
                    "Whether query i of a request of length L reads only its first "
                    "L - query_length + i + 1 rows."
                ),
            ),
            "sm_scale": Scalar(
                "float32",
                optional=True,
                description="Multiplies every logit; q.shape[-1] ** -0.5 when absent.",
            ),
            "sinks": Tensor(
                ["n_heads"],
                dtype="float32",
                optional=True,
                description="Per-head logit added to the normalisation only, not to lse.",
            ),
            "kv_cache_format": Scalar(
                "string",
                optional=True,
                description=(
                    "The pool rows: DeepSeek-V4's packed fp8 slots (sparse), or 576 "
                    "fp8 or bf16 values (dense)."
                ),
            ),
            "kv_scale": Scalar(
                "float32",
                optional=True,
                description="Multiplies every KV value, in the logits and the values.",
            ),
        },
        outputs={
            "output": Tensor(
                ["batch", "query_length", "n_heads", "v_dim"],
                description="Softmax-weighted sum of the read values; 0 for no row read.",
                dtype_from="q",
            ),
            "lse": Tensor(
                ["batch", "query_length", "n_heads"],
                dtype="float32",
                description=(
                    "Log-sum-exp of the scaled logits, sinks excluded; for no row "
                    "read +inf (sparse) or -inf (dense)."
                ),
            ),
        },
        reference=reference,
        check=_flash_mla_check,
        init=init,
    )


FLASH_MLA_SPARSE = _flash_mla(
    "flash_mla_with_kvcache_sparse",
    flash_mla_with_kvcache_sparse_reference,
    512,
    ("sparse_indices",),
    ("block_table", "cache_seqlens", "cu_seqlens_k"),
    ["batch", "table_width"],
    (
        "kv_cache_format == 'dsv4_fp8'",
        "kv_cache.dtype in ('uint8', 'float8_e4m3fn')",
        "head_dim_v is None",
        "(extra_kv_cache is None) == (extra_sparse_indices is None)",
        "extra_kv_cache is not None or extra_sparse_lens is None",
        "extra_kv_cache is None or extra_kv_cache.dtype in ('uint8', 'float8_e4m3fn')",
        "extra_kv_cache is None or n_extra_pages * extra_page_size <= 2147483648",
        "query_length == 1",
        "not causal",
    ),
    _flash_mla_with_kvcache_sparse_init,
)
FLASH_MLA_DENSE = _flash_mla(
    "flash_mla_with_kvcache_dense",
    flash_mla_with_kvcache_dense_reference,
    576,
    ("block_table", "cache_seqlens"),
    (
        "sparse_indices",
        "sparse_lens",
        "extra_kv_cache",
        "extra_sparse_indices",
        "extra_sparse_lens",
        "sinks",
        "cu_seqlens_k",
    ),
    ["batch", "table_width"],
    (
        "kv_cache_format in ('fp8_e4m3', 'bf16')",
        "(kv_cache.dtype == 'bfloat16') == (kv_cache_format == 'bf16')",
        "kv_cache_format != 'fp8_e4m3' or kv_cache.dtype == 'float8_e4m3fn'",
        "head_dim_v == 512",
        # a dense pool row is a page
        "page_size == 1",
    ),
    _flash_mla_with_kvcache_dense_init,
)
FLASH_MLA_RAGGED = _flash_mla(
    "flash_mla_with_kvcache_ragged",
    flash_mla_with_kvcache_ragged_reference,
    576,
    ("block_table", "cu_seqlens_k"),
    (
        "sparse_indices",
        "sparse_lens",
        "extra_kv_cache",
        "extra_sparse_indices",
        "extra_sparse_lens",
        "sinks",
        "cache_seqlens",
    ),
    ["table_width"],
    (
        "kv_cache_format in ('fp8_e4m3', 'bf16')",
        "(kv_cache.dtype == 'bfloat16') == (kv_cache_format == 'bf16')",
        "kv_cache_format != 'fp8_e4m3' or kv_cache.dtype == 'float8_e4m3fn'",
        "head_dim_v == 512",
        "page_size == 1",
        "n_offsets == batch + 1",
        "table_width >= batch * query_length",
    ),
    _flash_mla_with_kvcache_ragged_init,
)


def flash_mla_with_kvcache_op_schema(
    *_: Any,
    cu_seqlens_k: Any = None,
    kv_cache_format: str = "dsv4_fp8",
    **__: Any,
) -> Schema:
    if kv_cache_format == "dsv4_fp8":
        return FLASH_MLA_SPARSE
    if cu_seqlens_k is None:
        return FLASH_MLA_DENSE
    return FLASH_MLA_RAGGED


flash_mla_with_kvcache_op_schema.schemas = (
    FLASH_MLA_SPARSE,
    FLASH_MLA_DENSE,
    FLASH_MLA_RAGGED,
)


def _fp8_paged_mqa_logits_check(reference_outputs, actual_outputs, **_unused):
    # the old SQNR comparison: 20 * log10(|ref| / |actual - ref|) >= 30 dB
    reference, actual = (
        o[0] if isinstance(o, (list, tuple)) else o for o in (reference_outputs, actual_outputs)
    )
    if reference.shape != actual.shape:
        return False
    reference, actual = reference.double(), actual.double()
    error = (actual - reference).norm().item()
    return error <= reference.norm().item() * 10 ** (-30.0 / 20)


def _fp8_paged_mqa_logits_init(
    *,
    batch,
    n_heads,
    n_pages,
    page_tokens,
    max_blocks,
    max_seq_len,
    device="cuda",
    seed=0,
):
    torch.manual_seed(seed)
    return {
        "q": torch.randn(batch, 1, n_heads, 128, device=device).to(torch.float8_e4m3fn),
        "kv_cache": torch.randint(0, 64, (n_pages, page_tokens, 1, 132), device=device).to(torch.uint8),
        "weights": torch.randn(batch, n_heads, device=device),
        "context_lens": torch.full((batch,), max_seq_len, dtype=torch.int32, device=device),
        "block_tables": torch.randint(0, n_pages, (batch, max_blocks), device=device).to(torch.int32),
        "page_size": page_tokens,
        "logits": torch.empty(batch, max_seq_len, device=device),
        "preshuffled": True,
    }


FP8_PAGED_MQA_LOGITS = Schema(
    op_type="fp8_paged_mqa_logits",
    description=(
        "DeepSeek-V4 indexer logits: for each request and each cached token t, "
        "sum over heads h of weights[h] * relu(q[h] . k_t) * scale_t, over a paged "
        "pool of fp8 keys with an fp32 scale per token."
    ),
    axes={
        "batch": Var("Requests, one decode query each."),
        "n_heads": Const("Indexer heads."),
        "n_pages": Var("Pages of the key pool."),
        "page_tokens": Const("Tokens per page of the key pool: page_size."),
        "max_blocks": Var("Pages per request in the block table."),
        "max_seq_len": Var("Logits per request: one per slot of its block table row."),
        "one": Const("A single decode query, a single key head.", abbrev="", value=1),
        "index_dim": Const("Values per query head.", abbrev="", value=128),
        "token_bytes": Const(
            "Bytes per token of a page: 128 fp8 keys, then an fp32 scale.",
            abbrev="",
            value=132,
        ),
    },
    constraints=[
        "page_tokens == page_size",
        "preshuffled",
        "page_size % 16 == 0",
        "max_seq_len == max_blocks * page_size",
    ],
    inputs={
        "q": Tensor(
            ["batch", "one", "n_heads", "index_dim"],
            dtype="float8_e4m3fn",
            description="The decode query of each request.",
        ),
        "kv_cache": Tensor(
            ["n_pages", "page_tokens", "one", "token_bytes"],
            dtype="uint8",
            description=(
                "The key pool, a page a row: page_size x 128 fp8 keys in aiter's"
                " preshuffled 16 x 16 tiles, then page_size fp32 scales."
            ),
        ),
        "weights": Tensor(
            ["batch", "n_heads"],
            dtype="float32",
            description="Per-head weight of each request.",
        ),
        "context_lens": Tensor(["batch"], dtype="int32", description="Cached tokens of each request."),
        "block_tables": Tensor(
            ["batch", "max_blocks"],
            dtype="int32",
            description="The pages of each request, in context order.",
        ),
        "page_size": Scalar("int32", description="Tokens per page."),
    },
    outputs={
        "logits": Tensor(
            ["batch", "max_seq_len"],
            dtype="float32",
            description="The logits of each request's cached tokens.",
        ),
    },
    reference=fp8_paged_mqa_logits_reference,
    check=_fp8_paged_mqa_logits_check,
    init=_fp8_paged_mqa_logits_init,
)


def fp8_paged_mqa_logits_op_schema(*_: Any, preshuffled: bool = True, **__: Any) -> Schema | None:
    return FP8_PAGED_MQA_LOGITS if preshuffled else None


fp8_paged_mqa_logits_op_schema.schemas = (FP8_PAGED_MQA_LOGITS,)
