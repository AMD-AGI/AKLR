# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's persistent plan for its fp8 MLA prefill as ``get_ps_metadata_v1(..., need_lse=True)`` builds it
(``csrc/kernels/mla/metadata/v1_2_host.cuh``), in Python: every query tile, split over CUs or not, leaves fp32
partials with their log-sum-exp for the reduce. aiter's planner takes ``need_lse`` from 0.1.23 on; older ones (the
one in sglang 0.5.22's image) leave a tile one CU reads whole unreduced, so its lse is never written."""

from __future__ import annotations

import math


def _ceil_div(a: int, b: int) -> int:
    """C's ``(a + b - 1) / b``, which truncates toward zero."""
    n = a + b - 1
    return n // b if n >= 0 else -(-n // b)


def ps_plan(
    qo_indptr: list[int],
    kv_indptr: list[int],
    kv_lens: list[int],
    num_heads_k: int,
    *,
    num_cus: int,
    qlen_granularity: int = 256,
    kvlen_granularity: int = 128,
    block_size: int = 1,
    is_causal: bool = True,
) -> tuple[list[int], list[list[int]], list[int], list[list[int]], list[int]]:
    """``(work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map)`` as lists, as aiter's planner
    fills them with ``need_lse`` and one query head per key head (only its used rows: ``num_cus + 1``, one
    ``[batch, partial_qo_loc, qo_start, qo_end, kv_start, kv_end, kv_offset, q_head_range]`` per work, one entry per
    query tile), for sequences at ``qo_indptr``'s query rows over ``kv_indptr``'s key blocks, ``kv_lens`` keys each."""
    clusters = math.gcd(num_heads_k, num_cus)
    tgs = num_cus // clusters
    heads = num_heads_k // clusters
    per_unit = kvlen_granularity // block_size

    # each sequence's query tiles, head and tail alternately (0, n-1, 1, n-2, ...), with the key blocks they read
    tiles, units = [], 0
    for b in range(len(qo_indptr) - 1):
        qo_len, kv_len = qo_indptr[b + 1] - qo_indptr[b], kv_lens[b]
        ranges = [(s, min(s + qlen_granularity, qo_len)) for s in range(0, qo_len, qlen_granularity)]
        for i in range(len(ranges)):
            start, end = ranges[i // 2 if i % 2 == 0 else len(ranges) - 1 - i // 2]
            kv = min(kv_len - qo_len + end, kv_len) if is_causal else kv_len
            n = _ceil_div(kv, kvlen_granularity)
            tiles.append((b, start + qo_indptr[b], end + qo_indptr[b], n * per_unit, kv))
            units += n
    average, remainder = units // tgs, units % tgs

    work_indptr, works = [0] * (num_cus + 1), []
    for cluster in range(clusters):
        # each CU of the cluster takes its share of key blocks, the same for every head of the cluster
        tile = block = partial = 0
        for tg in range(tgs):
            for h in range(heads):
                head = cluster * heads + h
                head_range = ((head + 1) << 16) | (head & 0xFFFF)
                saved = (tile, block, partial)
                capacity = (average + 1 if tg < remainder else average) * per_unit
                while tile < len(tiles) and capacity > 0:
                    b, qo_start, qo_end, n_blocks, kv = tiles[tile]
                    kv_start = block + kv_indptr[b]
                    if kv - block * block_size <= capacity * block_size:
                        # the rest of the tile; reduced even when whole (need_lse), unless it reads nothing
                        consumed = n_blocks - block
                        loc = -1 if consumed == 0 else qlen_granularity * partial
                        partial += consumed != 0
                        kv_end = min(kv_start + consumed, kv_indptr[b + 1])
                        works.append([b, loc, qo_start, qo_end, kv_start, kv_end, 0, head_range])
                        tile, block = tile + 1, 0
                    else:
                        consumed = capacity
                        kv_end = min(kv_start + consumed, kv_indptr[b + 1])
                        kv_offset = kv_lens[b] - (kv_end - kv_indptr[b]) * block_size
                        works.append([b, qlen_granularity * partial, qo_start, qo_end, kv_start, kv_end, kv_offset, head_range])  # fmt: skip
                        partial += 1
                        block += consumed
                    capacity -= consumed
                if h != heads - 1:
                    tile, block, partial = saved
            work_indptr[cluster * tgs + tg + 1] = len(works)

    # one reduce per query tile, its partials in ascending order; every query row in exactly one
    groups: dict[tuple[int, int], set[int]] = {}
    for w in works:
        if w[1] != -1:
            groups.setdefault((w[2], w[3]), set()).add(w[1])
    reduce_indptr, final, partials, covered = [0], [], [], 0
    for start, end in sorted(groups):
        if start != covered:
            raise ValueError(
                f"query rows [{covered}, {start}) are in no reduce: their lse would be unwritten"
            )
        covered = end
        partials += sorted(groups[(start, end)])
        reduce_indptr.append(len(partials))
        final.append([start, end])
    if covered != qo_indptr[-1]:
        raise ValueError(
            f"query rows [{covered}, {qo_indptr[-1]}) are in no reduce: their lse would be unwritten"
        )
    return work_indptr, works, reduce_indptr, final, partials
