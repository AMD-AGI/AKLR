# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The token embedding from a whole table on every rank: no all-reduce on a step. SGLang's
vocab-parallel sum has one owner a row, so it equals the owner's bits; ``replicate`` gathers
the table once at build and ``lookup`` writes the ``hc`` copies in one launch."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from aklr.plugins.sglang import stats

ENV_EMBED = "AKLR_DSV41_EMBED"
INFO = "deepseek_v4.embed"
# the whole table, on the module SGLang's target and draft share
_ATTR = "_aklr_dsv41_table"

SGLANG_READS = {
    "srt/layers/vocab_parallel_embedding.py": (
        "VocabParallelEmbedding",
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_end_index",
        "num_added_embeddings",
        "org_vocab_size",
        "output_dtype",
        "use_attn_tp_group",
        "tp_size",
        "get_attn_tp_context",
    ),
    "srt/layers/layer_boundary/adapters/attention.py": ("input_scattered",),
    "srt/distributed/parallel_state.py": ("device_group", "cpu_group", "world_size"),
}


@triton.jit
def _lookup_kernel(IDS, W, O, n_vocab, D: tl.constexpr, HC: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    mask = col < D
    i = tl.load(IDS + token).to(tl.int64)
    valid = (i >= 0) & (i < n_vocab)
    row = tl.load(W + tl.where(valid, i, 0) * D + col, mask & valid, 0)
    for c in tl.static_range(HC):
        tl.store(O + (token * HC + c) * D + col, row, mask)


def _lookup(table: torch.Tensor, ids: torch.Tensor, hc: int) -> torch.Tensor:
    n, d = ids.shape[0], table.shape[1]
    out = table.new_empty((n, hc, d))
    if n:
        b = 1024
        _lookup_kernel[(n, triton.cdiv(d, b))](ids, table, out, table.shape[0], d, hc, b, num_warps=4)
    return out


def lookup(module, input_ids: torch.Tensor, hc: int) -> torch.Tensor | None:
    """``module(input_ids)`` broadcast to ``[T, hc, H]``, or None where the module
    holds no whole table (the caller runs SGLang's)."""
    table = getattr(module, _ATTR, None)
    if table is None or input_ids.ndim != 1 or not input_ids.is_contiguous():
        return None
    if _input_scattered():  # SGLang's leaves the shards' sum to a later collective
        return None
    return _lookup(table, input_ids, hc)


def _input_scattered() -> bool:
    from sglang.srt.layers.vocab_parallel_embedding import get_attn_tp_context

    return get_attn_tp_context().input_scattered


def _refusal(module) -> str | None:
    value = os.environ.get(ENV_EMBED) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV_EMBED}={value!r}: expected '1' or '0'")
    if value == "0":
        return f"SGLang's ({ENV_EMBED}=0)"
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    if type(module) is not VocabParallelEmbedding:
        return f"SGLang's (a {type(module).__name__})"
    if module.tp_size == 1 or module.use_attn_tp_group:
        return "SGLang's (no vocab-parallel all-reduce)"
    w = getattr(module, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or not w.is_cuda:
        return "SGLang's (not a BF16 device table)"
    if module.num_added_embeddings or module.output_dtype not in (None, w.dtype):
        return "SGLang's (added vocab or a cast output)"
    return None


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def replicate(module, group, copies: dict) -> None:
    """Gather ``module``'s vocab shards into the whole table on every rank of
    ``group`` (SGLang's TP group); collective: every rank calls it."""
    import torch.distributed as dist

    why = _refusal(module)
    # every rank takes the same branch: the gathers below are collective
    refused = torch.tensor([why is not None], dtype=torch.int32)
    dist.all_reduce(refused, op=dist.ReduceOp.MAX, group=group.cpu_group)
    if refused.item():
        stats.set_info(INFO, why or "SGLang's (another rank keeps SGLang's)")
        return
    w, s = module.weight, module.shard_indices
    start, end = s.org_vocab_start_index, s.org_vocab_end_index
    ranges = torch.tensor([start, end], dtype=torch.int64, device=w.device)
    every = torch.empty(group.world_size * 2, dtype=torch.int64, device=w.device)
    dist.all_gather_into_tensor(every, ranges, group=group.device_group)
    shards = w.new_empty((group.world_size * w.shape[0], w.shape[1]))
    dist.all_gather_into_tensor(shards, w.contiguous(), group=group.device_group)
    table = w.new_zeros((module.org_vocab_size, w.shape[1]))
    for r, (a, b) in enumerate(every.view(-1, 2).tolist()):
        table[a:b] = shards[r * w.shape[0] : r * w.shape[0] + b - a]
    del shards
    torch.cuda.empty_cache()
    n = table.shape[0]
    inside = torch.cat([torch.arange(0, n, 61), torch.arange(start, end, 67)])
    inside = torch.cat([inside, torch.tensor([n - 1])]).to(w.device)
    outside = torch.tensor([-1, n, n + 7], device=w.device)
    got = _lookup(table, torch.cat([inside, outside]), 2)
    want = torch.cat([table[inside], table.new_zeros(len(outside), table.shape[1])])
    ok = _same(table[start:end], w[: end - start]) and _same(got[:, 0], want) and _same(got[:, 1], want)
    del got, want
    agreed = torch.tensor([ok], dtype=torch.int32)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if not agreed.item():
        stats.set_info(INFO, "SGLang's (a rank's table or lookup differs)")
        return
    copies["embed table",] = table
    setattr(module, _ATTR, table)
    stats.set_info(INFO, f"whole table on each rank ({tuple(table.shape)})")
