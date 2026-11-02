# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The per-step attention metadata of a served step: placeholders for what it never reads
(the ratio-4 / ratio-128 fields), and its pieces from ``prologue``'s launches where their
first eager build agrees with sglang's."""

from __future__ import annotations

import contextvars
import os

import torch

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import compat, serving
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

ENV_SKIP = "AKLR_DSV41_SKIP_C4C128"
HIP = "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix"

SGLANG_READS = {
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": (
        "DeepseekV4HipRadixBackend",
        "make_forward_metadata_from_raw_verify(raw_metadata)",
        "init_forward_metadata_in_graph(forward_batch)",
        "has_c4",
        "has_c128",
        "target_verify_num_draft_tokens",
        "__init__(model_runner)",
        "DSV4AttnMetadata",
        "init_compression_metadata(num_tokens)",
        "init_compression_metadata(unified_swa_pages)",
        "seq_lens_casual",
        "raw_out_loc",
        "low_ratios",
        "c4_out_loc",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_out_loc",
        "c128_topk_lengths_raw",
        "c128_topk_lengths_clamp1",
        "c128_page_indices",
        "swa_page_indices",
        "_pad_last_dim",
        "low_ratio_compression_metadata",
        "expand_extend_with_same_length(bs)",
        "expand_extend_with_same_length(qo_len)",
        "expand_extend_with_same_length(seq_lens)",
        "expand_extend_with_same_length(req_pool_indices)",
        "get_swa_page_indices(seq_lens_casual)",
        "get_swa_page_indices(req_pool_indices_repeated)",
        "SWA_WINDOW",
        "req_to_token",
        "token_to_kv_pool",
        "PAGE_INDEX_ALIGNED_SIZE",
        "page_table_from_req_to_token",
        "create_paged_compressor_data",
        "_create_dummy_paged_compress_data",
        "DSV4Metadata",
        "init_flashmla_related",
        "_attach_unified_kv_prefill_meta",
        "_attach_unified_kv_decode_streams",
        "init_forward_metadata_indexer",
        "_init_low_ratio_indexer_metadata",
        "need_compress",
        "MAX_SEQ_LEN_FOR_CAPTURE",
        "_create_flashmla_metadata",
        "dsa_topk_backend",
        "index_topk",
        "cuda_int32_kwargs",
    ),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "DeepSeekV4TokenToKVPool",
        "translate_loc_from_full_to_swa",
        "full_to_swa_index_mapping",
    ),
}

_UNREAD = contextvars.ContextVar("aklr_dsv41_c4c128_unread", default=False)


def _enabled() -> bool:
    value = os.environ.get(ENV_SKIP) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV_SKIP}={value!r}: expected '1' or '0'")
    return value == "1"


def _served(backend, batch_size: int) -> bool:
    """Whether AKLR's monokernels run this verify step (not another backend's)."""
    runner = backend.__dict__.get("_aklr_runner")
    if runner is None or not serving() or compat.backend() != "aklr":
        return False
    key = compat.ShapeKey(
        forward_mode="target_verify",
        batch_size=batch_size,
        tokens_per_request=backend.target_verify_num_draft_tokens,
    )
    return key in compat.served_keys(runner.model)


def _served_verify(backend, raw_metadata) -> bool:
    if backend.has_c4 or backend.has_c128 or not _enabled():
        return False
    return _served(backend, len(raw_metadata.seq_lens))


@sglang_route(f"{HIP}.DeepseekV4HipRadixBackend.__init__", versions=VERSIONS, around=True)
def _backend_init(orig, self, model_runner, *args, **kwargs):
    # release/v0.5.22's backend keeps no reference to its runner
    self._aklr_runner = model_runner
    return orig(self, model_runner, *args, **kwargs)


@sglang_route(
    f"{HIP}.DeepseekV4HipRadixBackend.make_forward_metadata_from_raw_verify",
    versions=VERSIONS,
    around=True,
)
def _verify_metadata(orig, self, raw_metadata, *args, **kwargs):
    served = _served_verify(self, raw_metadata)
    token = _UNREAD.set(served)
    try:
        if served and not args and not kwargs and _rows_enabled():
            rows = _rows_inputs(self, raw_metadata)
            meta = rows is not None and _meta_inputs(self)
            if meta and _agreed(
                "verify_meta",
                lambda: _metadata_disagrees(orig, self, raw_metadata, rows, meta),
            ):
                return _fused_verify_metadata(self, raw_metadata, rows, meta)
            if rows is not None and _agreed(
                "verify_rows",
                lambda: _metadata_disagrees(orig, self, raw_metadata, rows),
            ):
                return _fused_verify_metadata(self, raw_metadata, rows)
        return orig(self, raw_metadata, *args, **kwargs)
    finally:
        _UNREAD.reset(token)


ENV_ROWS = "AKLR_DSV41_VERIFY_ROWS"
# the fused rows' ratio-4 / ratio-128 placeholders (out_loc, lengths) for _compression_metadata
_PLACEHOLDERS = contextvars.ContextVar("aklr_dsv41_c4c128_placeholders", default=None)


def _rows_enabled() -> bool:
    value = os.environ.get(ENV_ROWS) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV_ROWS}={value!r}: expected '1' or '0'")
    return value == "1" and serving()


def _rows_inputs(backend, raw):
    """(req_to_token, mapping) for ``prologue.verify_rows`` on this raw verify metadata, or
    None where sglang's ``make_core_attn_metadata`` takes another branch."""
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        PAGE_INDEX_ALIGNED_SIZE,
        SWA_WINDOW,
    )

    seq_lens, req, loc = raw.seq_lens, raw.req_pool_indices, raw.out_cache_loc
    rows = seq_lens.shape[0] * backend.target_verify_num_draft_tokens
    if (
        backend.token_to_kv_pool.request_window is not None
        or SWA_WINDOW % PAGE_INDEX_ALIGNED_SIZE
        or seq_lens.dim() != 1
        or not seq_lens.is_contiguous()
        or seq_lens.dtype not in (torch.int32, torch.int64)
        or req.dim() != 1
        or req.shape != seq_lens.shape
        or not req.is_contiguous()
        or loc is None
        or loc.shape[0] != rows
        or not seq_lens.is_cuda
        or not getattr(backend, "need_compress", False)
    ):
        return None
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    pool, r2t = backend.token_to_kv_pool, backend.req_to_token
    mapping = getattr(pool, "full_to_swa_index_mapping", None)
    if (
        type(pool).translate_loc_from_full_to_swa
        is not DeepSeekV4TokenToKVPool.translate_loc_from_full_to_swa
        or mapping is None
        or mapping.dim() != 1
        or not mapping.is_contiguous()
        or r2t.dim() != 2
        or r2t.stride(1) != 1
    ):
        return None
    return r2t, mapping


def _meta_inputs(backend):
    """(blocks a page of the ratio-1 and ratio-2 indexer pools) for ``prologue.verify_meta``,
    or None where its tensors are not the whole of the step's metadata."""
    if (
        set(backend.low_ratios) != {1, 2}
        or backend.has_c4
        or backend.has_c128
        or compat.is_unified_kv_triton()
        or getattr(backend.token_to_kv_pool, "unified_swa_pages", 0)
        or backend.page_size != 256
    ):
        return None
    bpp = []
    for ratio in (1, 2):
        index_page = backend.token_to_kv_pool.get_index_k_page_size(ratio)
        slots = backend.page_size // ratio
        if slots % index_page:
            return None
        n = slots // index_page
        if n < 1 or n & (n - 1):
            return None
        bpp.append(n)
    return tuple(bpp)


def _meta_metadata(self, raw_metadata, rows, meta):
    """``_fused_verify_metadata`` with every tensor from ``prologue.verify_meta``'s launch."""
    from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    r2t, mapping = rows
    bs, n = len(raw_metadata.seq_lens), self.target_verify_num_draft_tokens
    ext = raw_metadata.extend_seq_lens
    if ext is not None and ext.numel() != bs:
        ext = None
    if ext is not None:
        ext = ext.to(device=raw_metadata.seq_lens.device).contiguous()
    width = -(-self.index_topk // hip.PAGE_INDEX_ALIGNED_SIZE) * hip.PAGE_INDEX_ALIGNED_SIZE
    counter = _counter(TARGET_SERVED, raw_metadata.seq_lens.device)
    t = prologue.verify_meta(
        seq_lens=raw_metadata.seq_lens,
        req_pool_indices=raw_metadata.req_pool_indices,
        extend_seq_lens=ext,
        qo_len=n,
        req_to_token=r2t,
        mapping=mapping,
        window=hip.SWA_WINDOW,
        raw_out_loc=raw_metadata.out_cache_loc,
        max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
        page_size=self.page_size,
        index_topk=self.index_topk,
        width=width,
        bpp1=meta[0],
        bpp2=meta[1],
        counter=counter,
    )
    rows_n = bs * n
    core = hip.DSV4AttnMetadata(
        page_size=self.page_size,
        raw_out_loc=raw_metadata.out_cache_loc,
        seq_lens_casual=t["seq_lens_casual"],
        cuda_int32_kwargs=self.cuda_int32_kwargs,
        positions_casual=t["positions_casual"],
        page_table=t["page_table"],
        swa_page_indices=t["swa_page_indices"],
        swa_topk_lengths=t["swa_topk_lengths"],
        index_topk=self.index_topk,
        low_ratios=self.low_ratios,
        request_window_layout=None,
    )
    # init_compression_metadata on a served step (_compression_metadata's placeholders)
    loc, zero = t["c4c128_out_loc"], t["zero_lengths"]
    core.c4_out_loc, core.c128_out_loc = loc[0], loc[1]
    core.c4_topk_lengths_raw = core.c4_topk_lengths_clamp1 = zero
    core.c128_topk_lengths_raw = core.c128_topk_lengths_clamp1 = zero
    core.c128_page_indices = torch.empty((rows_n, 0), dtype=torch.int32, device=zero.device)
    for name in (
        "c1_out_loc",
        "c1_topk_lengths_clamp1",
        "c2_out_loc",
        "c2_topk_lengths_clamp1",
    ):
        setattr(core, name, t[name])
    stats.set_info("c4c128_metadata", "skipped on served verify steps")
    # init_flashmla_related
    core.c4_sparse_topk_lengths, core.c4_sparse_topk_lengths_raw = t["sparse_zero"]
    for name in (
        "c4_sparse_page_indices",
        "c1_sparse_topk_lengths",
        "c1_sparse_page_indices",
        "c2_sparse_topk_lengths",
        "c2_sparse_page_indices",
        "c1_sparse_raw_indices",
        "c2_sparse_raw_indices",
    ):
        setattr(core, name, t[name])
    core.c0_flashmla_metadata = hip._create_flashmla_metadata()
    core.c4_flashmla_metadata = hip._create_flashmla_metadata()
    core.c128_flashmla_metadata = hip._create_flashmla_metadata()
    core.c1_flashmla_metadata = hip._create_flashmla_metadata()
    core.c2_flashmla_metadata = hip._create_flashmla_metadata()
    indexers = {}
    for ratio in (1, 2):
        indexers[f"c{ratio}_indexer_metadata"] = PagedIndexerMetadata(
            page_size=self.page_size,
            compressed_page_size=self.token_to_kv_pool.get_index_k_page_size(ratio),
            page_table=(t[f"c{ratio}_page_table"] if meta[ratio - 1] > 1 else t["page_table"]),
            compressed_seq_lens=t[f"c{ratio}_topk_lengths_clamp1"],
            use_topk_v2=self.dsa_topk_backend.should_use_topk_v2(),
            compress_ratio=ratio,
        )
    if t["req_pool_indices_repeated"].dtype == torch.int64:
        core._aklr_req_rep = t["req_pool_indices_repeated"]  # _token_req_indices
    core._aklr_counted = counter is not None  # spec.counted
    dummy = hip._create_dummy_paged_compress_data
    return hip.DSV4Metadata(
        core,
        None,
        **{k: indexers[k] for k in (f"c{r}_indexer_metadata" for r in core.low_ratios)},
        c4_compress_metadata=dummy(compress_ratio=4),
        c128_compress_metadata=dummy(compress_ratio=128),
    )


def _fused_verify_metadata(self, raw_metadata, rows, meta=None):
    """sglang's ``make_forward_metadata_from_raw_verify`` (``make_core_attn_metadata``'s
    plain-window branch) on a served step, its row pieces from ``prologue.verify_rows``;
    with ``meta`` every tensor from ``prologue.verify_meta`` (``_meta_metadata``)."""
    import functools

    from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip

    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    if meta:
        return _meta_metadata(self, raw_metadata, rows, meta)
    r2t, mapping = rows
    req_pool_indices = raw_metadata.req_pool_indices
    out_cache_loc = raw_metadata.out_cache_loc
    bs, n = len(raw_metadata.seq_lens), self.target_verify_num_draft_tokens
    ext = raw_metadata.extend_seq_lens
    if ext is not None and ext.numel() != bs:
        ext = None
    if ext is not None:
        ext = ext.to(device=raw_metadata.seq_lens.device).contiguous()
    seq_lens, extend_seq_lens, casual, req_rep, positions, swa, swa_len, loc, zero = prologue.verify_rows(
        seq_lens=raw_metadata.seq_lens,
        req_pool_indices=req_pool_indices,
        extend_seq_lens=ext,
        qo_len=n,
        req_to_token=r2t,
        mapping=mapping,
        window=hip.SWA_WINDOW,
    )
    core = hip.DSV4AttnMetadata(
        page_size=self.page_size,
        raw_out_loc=out_cache_loc,
        seq_lens_casual=casual,
        cuda_int32_kwargs=self.cuda_int32_kwargs,
        positions_casual=positions,
        page_table=hip.page_table_from_req_to_token(
            self.req_to_token, req_rep, self.MAX_SEQ_LEN_FOR_CAPTURE, self.page_size
        ),
        swa_page_indices=swa,
        swa_topk_lengths=swa_len,
        index_topk=self.index_topk,
        low_ratios=self.low_ratios,
        request_window_layout=None,
    )
    token = _PLACEHOLDERS.set((loc, zero))
    try:
        core.init_compression_metadata(
            num_tokens=None,
            unified_swa_pages=getattr(self.token_to_kv_pool, "unified_swa_pages", 0),
        )
    finally:
        _PLACEHOLDERS.reset(token)
    core.init_flashmla_related()
    self._attach_unified_kv_prefill_meta(core, req_pool_indices, req_rep, seq_lens, extend_seq_lens, n * bs)
    self._attach_unified_kv_decode_streams(core, req_rep, num_draft=n)
    indexer_metadata = self.init_forward_metadata_indexer(core) if self.has_c4 else None
    create = functools.partial(
        hip.create_paged_compressor_data,
        is_prefill=True,
        token_to_kv_pool=self.token_to_kv_pool,
        req_to_token=self.req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_lens=extend_seq_lens,
        seq_lens_cpu=None,
        extend_lens_cpu=None,
        use_prefill_cuda_graph=True,
        num_q_tokens=n * bs,
    )
    dummy = hip._create_dummy_paged_compress_data
    return hip.DSV4Metadata(
        core,
        indexer_metadata,
        **self._init_low_ratio_indexer_metadata(core, is_prefill=False),
        c4_compress_metadata=(create(compress_ratio=4) if self.has_c4 else dummy(compress_ratio=4)),
        c128_compress_metadata=(create(compress_ratio=128) if self.has_c128 else dummy(compress_ratio=128)),
    )


def _metadata_disagrees(orig, self, raw_metadata, rows, meta=None):
    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    want = orig(self, raw_metadata)
    got = _fused_verify_metadata(self, raw_metadata, rows, meta)
    return prologue.metadata_disagrees(want, got)


@sglang_route(f"{HIP}.DSV4AttnMetadata.init_compression_metadata", versions=VERSIONS, around=True)
def _compression_metadata(orig, self, num_tokens=None, unified_swa_pages=0):
    if not _UNREAD.get() or unified_swa_pages:
        return orig(self, num_tokens, unified_swa_pages)
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        PAGE_INDEX_ALIGNED_SIZE,
        _pad_last_dim,
        low_ratio_compression_metadata,
    )

    rows, writes = self.seq_lens_casual.shape[0], self.raw_out_loc.shape[0]
    device = self.seq_lens_casual.device
    made = _PLACEHOLDERS.get()
    if made is not None:  # the fused rows' (verify_rows), already filled
        out_loc, lengths = made
    else:
        out_loc = torch.full((2, writes), -1, dtype=torch.int64, device=device)
        lengths = torch.zeros(rows, dtype=torch.int32, device=device)
    self.c4_out_loc, self.c128_out_loc = out_loc[0], out_loc[1]
    self.c4_topk_lengths_raw = self.c4_topk_lengths_clamp1 = lengths
    self.c128_topk_lengths_raw = self.c128_topk_lengths_clamp1 = lengths
    self.c128_page_indices = torch.empty((rows, 0), dtype=torch.int32, device=device)
    if made is None or self.swa_page_indices.shape[-1] % PAGE_INDEX_ALIGNED_SIZE:
        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)
    for name, value in low_ratio_compression_metadata(
        self.seq_lens_casual, self.raw_out_loc, self.low_ratios
    ).items():
        setattr(self, name, value)
    stats.set_info("c4c128_metadata", "skipped on served verify steps")


@sglang_route(
    f"{HIP}.DeepseekV4HipRadixBackend.init_forward_metadata_in_graph",
    versions=VERSIONS,
    around=True,
)
def _metadata_in_graph(orig, self, forward_batch, *args, **kwargs):
    held = {"backend": self}
    token_graph = _IN_GRAPH.set(held if _rows_enabled() else None)
    try:
        result = orig(self, forward_batch, *args, **kwargs)
    finally:
        _IN_GRAPH.reset(token_graph)
    loc = held.get("swa_loc")
    core = getattr(self.forward_metadata, "core_attn_metadata", None)
    if loc is not None and core is not None and core.swa_out_cache_loc is loc[0]:
        core._aklr_slot64 = loc[1]  # K1's slots (spec._launch)
    return result


# inside init_forward_metadata_in_graph: {"backend": the backend, "swa_loc": (int32, int64)}
_IN_GRAPH = contextvars.ContextVar("aklr_dsv41_in_graph", default=None)


@sglang_route(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool.translate_loc_from_full_to_swa",
    versions=VERSIONS,
    around=True,
)
def _translate_loc(orig, self, kv_indices, *args, **kwargs):
    """In ``init_forward_metadata_in_graph`` (whose one call casts it to int32) the SWA
    locations come out as int32 from one launch, with their int64 copy for K1."""
    held = _IN_GRAPH.get()
    mapping = getattr(self, "full_to_swa_index_mapping", None)
    if (
        held is None
        or args
        or kwargs
        or "swa_loc" in held
        or mapping is None
        or mapping.dim() != 1
        or not mapping.is_contiguous()
        or kv_indices.dim() != 1
        or not kv_indices.is_contiguous()
        or not kv_indices.is_cuda
        or kv_indices.dtype not in (torch.int32, torch.int64)
    ):
        return orig(self, kv_indices, *args, **kwargs)
    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    if not _agreed(
        "swa_loc",
        lambda: prologue.swa_disagrees(
            orig(self, kv_indices).to(torch.int32),
            prologue.swa_loc(kv_indices, mapping)[0],
        ),
    ):
        return orig(self, kv_indices)
    held["swa_loc"] = prologue.swa_loc(kv_indices, mapping)
    return held["swa_loc"][0]


@sglang_route(
    "sglang.srt.layers.attention.dsv4.dsv41_sparse.token_req_indices",
    versions=VERSIONS,
    around=True,
)
def _token_req_indices(orig, forward_batch, *args, **kwargs):
    """A served verify step's per-token request slots: the rows ``verify_meta`` expanded."""
    held = _IN_GRAPH.get()
    if held is not None and not args and not kwargs.get("num_tokens"):
        core = getattr(held["backend"].forward_metadata, "core_attn_metadata", None)
        rep = getattr(core, "_aklr_req_rep", None)
        if (
            rep is not None
            and forward_batch.forward_mode.is_target_verify()
            and rep.dtype == torch.int64
            and rep.numel() == forward_batch.batch_size * int(forward_batch.spec_info.draft_token_num)
        ):
            return rep
    return orig(forward_batch, *args, **kwargs)


ENV_PROLOGUE = "AKLR_DSV41_PROLOGUE"
# a piece -> whether its kernel agreed with sglang's at its first eager call
_AGREED: dict[str, bool] = {}


def _prologue_enabled() -> bool:
    value = os.environ.get(ENV_PROLOGUE) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV_PROLOGUE}={value!r}: expected '1' or '0'")
    return value == "1" and serving()


def _agreed(piece: str, why) -> bool | None:
    """The piece's verdict; at its first eager call ``why()`` (None: the same
    integers as sglang's) decides it. None under a capture before any verdict."""
    if piece not in _AGREED:
        if torch.cuda.is_current_stream_capturing():
            return None
        reason = why()
        _AGREED[piece] = reason is None
        stats.set_info(f"deepseek_v4.prologue:{piece}", reason or "one launch")
    return _AGREED[piece]


@sglang_route(
    f"{HIP}.DeepseekV4HipRadixBackend.expand_extend_with_same_length",
    versions=VERSIONS,
    around=True,
)
def _expand(orig, self, bs, qo_len, seq_lens, req_pool_indices):
    if not (_prologue_enabled() and seq_lens.is_cuda and seq_lens.dim() == 1):
        return orig(self, bs, qo_len, seq_lens, req_pool_indices)
    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    ok = _agreed(
        "expand",
        lambda: prologue.expand_disagrees(orig, self, bs, qo_len, seq_lens, req_pool_indices),
    )
    if not ok:
        return orig(self, bs, qo_len, seq_lens, req_pool_indices)
    return prologue.expand(bs, qo_len, seq_lens, req_pool_indices)


def _swa_inputs(backend, seq_lens_casual, req_pool_indices_repeated):
    """(req_to_token, mapping) the kernel reads, or None where sglang's path differs."""
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    pool, r2t = backend.token_to_kv_pool, backend.req_to_token
    mapping = getattr(pool, "full_to_swa_index_mapping", None)
    if (
        type(pool).translate_loc_from_full_to_swa
        is not DeepSeekV4TokenToKVPool.translate_loc_from_full_to_swa
        or mapping is None
        or mapping.dim() != 1
        or not mapping.is_contiguous()
        or r2t.dim() != 2
        or r2t.stride(1) != 1
        or not seq_lens_casual.is_cuda
        or seq_lens_casual.dtype != torch.int32
        or not seq_lens_casual.is_contiguous()
        or not req_pool_indices_repeated.is_contiguous()
    ):
        return None
    return r2t, mapping


@sglang_route(
    f"{HIP}.DeepseekV4HipRadixBackend.get_swa_page_indices",
    versions=VERSIONS,
    around=True,
)
def _swa_page_indices(orig, self, seq_lens_casual, req_pool_indices_repeated):
    inputs = _prologue_enabled() and _swa_inputs(self, seq_lens_casual, req_pool_indices_repeated)
    if not inputs:
        return orig(self, seq_lens_casual, req_pool_indices_repeated)
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import SWA_WINDOW

    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    r2t, mapping = inputs
    got = prologue.swa_page_indices(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
        req_to_token=r2t,
        mapping=mapping,
        window=SWA_WINDOW,
    )
    ok = _agreed(
        "swa_page_indices",
        lambda: prologue.swa_disagrees(orig(self, seq_lens_casual, req_pool_indices_repeated), got),
    )
    if not ok:
        return orig(self, seq_lens_casual, req_pool_indices_repeated)
    return got


@sglang_route(
    f"{HIP}.DeepseekV4HipRadixBackend.init_forward_metadata_dspark_draft_block",
    versions=VERSIONS,
    around=True,
)
def _draft_block_metadata(orig, self, *args, **kwargs):
    """The DSpark draft block's metadata with its rows from one launch (``prologue.block_rows``)
    where the first eager build agrees with sglang's; ``AKLR_DSV41_VERIFY_ROWS=0`` keeps
    sglang's."""
    if args or not _rows_enabled() or not _block_inputs_ok(self, **kwargs):
        return orig(self, *args, **kwargs)
    ok = _agreed(
        "block_rows",
        lambda: _block_disagrees(orig, self, kwargs),
    )
    if not ok:
        return orig(self, *args, **kwargs)
    return _fused_draft_block_metadata(self, **kwargs)


def _block_inputs_ok(self, *, max_seq_len, req_pool_indices, seq_lens, out_cache_loc, block_size):
    return bool(
        self.token_to_kv_pool.request_window is None
        and self._uses_dspark_draft_window()
        and block_size == self.target_verify_num_draft_tokens
        and seq_lens.is_cuda
        and seq_lens.dim() == 1
        and seq_lens.is_contiguous()
        and seq_lens.dtype in (torch.int32, torch.int64)
        and req_pool_indices.shape == seq_lens.shape
        and req_pool_indices.is_contiguous()
        and out_cache_loc.shape[0] >= block_size * seq_lens.shape[0]
    )


def _fused_draft_block_metadata(self, *, max_seq_len, req_pool_indices, seq_lens, out_cache_loc, block_size):
    """sglang's ``init_forward_metadata_dspark_draft_block`` (``make_core_attn_metadata``'s
    DSpark branch without compression), the rows from ``prologue.block_rows``."""
    from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip

    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    pool = self.token_to_kv_pool
    mapping = getattr(pool, "full_to_swa_index_mapping", None)
    counter = None
    if (
        getattr(pool, "swa_req_ring_size", None) is None
        and mapping is not None
        and mapping.dim() == 1
        and mapping.is_contiguous()
        and out_cache_loc.is_contiguous()
        and self.req_to_token.stride(1) == 1
    ):
        # the rows, the window's SWA page indices and the page table in one launch
        counter = _counter(DRAFT_SERVED, seq_lens.device)
        valid = getattr(pool, "swa_valid_from", None)
        t = prologue.block_meta(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            qo_len=block_size,
            req_to_token=self.req_to_token,
            mapping=mapping,
            out_loc=out_cache_loc,
            valid_from=valid,
            window=hip.SWA_WINDOW,
            width=-(-(hip.SWA_WINDOW + block_size) // hip.PAGE_INDEX_ALIGNED_SIZE)
            * hip.PAGE_INDEX_ALIGNED_SIZE,
            max_seq_len=max_seq_len + block_size,
            page_size=self.page_size,
            counter=counter,
        )
        casual, positions, page_table = (
            t["seq_lens_casual"],
            t["positions_casual"],
            t["page_table"],
        )
        swa_page_indices, swa_topk_lengths = (
            t["swa_page_indices"],
            t["swa_topk_lengths"],
        )
    else:
        casual, req_rep, positions = prologue.block_rows(seq_lens, req_pool_indices, block_size)
        swa_page_indices, swa_topk_lengths = self.get_dspark_swa_page_indices(
            seq_lens_casual=casual,
            req_pool_indices_repeated=req_rep,
            out_loc=out_cache_loc,
            block_size=block_size,
        )
        page_table = hip.page_table_from_req_to_token(
            self.req_to_token, req_rep, max_seq_len + block_size, self.page_size
        )
    core = hip.DSV4AttnMetadata(
        page_size=self.page_size,
        raw_out_loc=out_cache_loc,
        seq_lens_casual=casual,
        cuda_int32_kwargs=self.cuda_int32_kwargs,
        positions_casual=positions,
        page_table=page_table,
        swa_page_indices=swa_page_indices,
        swa_topk_lengths=swa_topk_lengths,
        index_topk=self.index_topk,
        low_ratios=self.low_ratios,
        request_window_layout=None,
    )
    core.c4_sparse_topk_lengths = None
    core.c4_sparse_topk_lengths_raw = None
    core.c4_sparse_page_indices = None
    core.c4_sparse_raw_indices = None
    core.c0_flashmla_metadata = hip._create_flashmla_metadata()
    core.c4_flashmla_metadata = None
    core.c128_flashmla_metadata = None
    core._aklr_counted = counter is not None  # spec.counted
    dummy = hip._create_dummy_paged_compress_data
    return hip.DSV4Metadata(
        core,
        None,
        c4_compress_metadata=dummy(compress_ratio=4),
        c128_compress_metadata=dummy(compress_ratio=128),
    )


TARGET_SERVED, DRAFT_SERVED = "deepseek_v4.target:served", "deepseek_v4.draft:served"
ENV_COUNT = "AKLR_DSV41_COUNT_IN_META"


def _counter(name, device):
    """The served counter (``stats.device_counter``) the metadata launch adds one to for the
    forward (``spec.counted``), or None: ``AKLR_DSV41_COUNT_IN_META=0`` or not made yet.
    """
    if os.environ.get(ENV_COUNT, "1") == "0":
        return None
    return stats.existing_device_counter(name, device)


def _block_disagrees(orig, self, kwargs):
    from aklr.plugins.sglang.models.deepseek_v4 import prologue

    want = orig(self, **kwargs)
    got = _fused_draft_block_metadata(self, **kwargs)
    return prologue.metadata_disagrees(want, got)
