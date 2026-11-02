# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 68506823, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.: atom/models/deepseek_v41/mono/index_plan.py, via sglang#43984
"""The FP4 scorer's tasks of a step, defined once for the kernel (traced Int32) and the CPU test
(Python ints). ``bound`` is how many columns a token's scorer may read; only non-negative operands
reach ``//`` and ``%``, where Python's floor and the device's truncation agree."""

import numpy as np

from aklr.plugins.sglang.models.deepseek_v4.kernels.common.plan import cdiv, sel

CHUNK = 128  # columns a score task: 16 tiles of 8 rows
TOPK = 512
TOPK_BLOCKS = 2048
BLOCK_ROWS = 8  # a candidate block, and an index tile (an FP4 plane's page)
STEP = 64  # keys an FP4 walk step: a wave's 4 MFMA tiles, 8 pages
WALK_ROWS = 6  # query rows an FP4 walk scores off each key load
HEADS = 32  # the indexer's heads (``index_n_heads``)
DIM = 128  # an index head's dims (``index_head_dim``)


def scored(bound):
    """Does the scorer run for a token? Not while its top-k is every column it
    sees (bound <= TOPK: SGLang's top-k then takes them all, reading no logits)."""
    return bound > TOPK


def score_chunks(bound):
    """The chunks the scorer scores for a token: chunk ch covers columns
    [CHUNK ch, CHUNK (ch + 1))."""
    return sel(scored(bound), cdiv(bound, CHUNK), 0)


def fill_walk_plan(*, bounds, owners, plan):
    """A FULL layer's score tasks on the host: ``plan`` [ctas, 4] gets (lead, span, ch0, ch1) a CTA,
    chunks of a piece (at most WALK_ROWS adjacent tokens of one owner), zeros past the last. Each piece
    gets ceil(need / width) even tasks at the smallest width that fits."""
    n, ctas = bounds.shape[0], plan.shape[0]
    plan.fill(0)
    t = np.arange(n)
    run_start = np.ones(n, dtype=bool)
    run_start[1:] = owners[1:] != owners[:-1]
    run_first = np.maximum.accumulate(np.where(run_start, t, 0))
    lead = np.flatnonzero(run_start | ((t - run_first) % WALK_ROWS == 0))
    span = np.diff(lead, append=n)
    need = np.maximum.reduceat(np.where(scored(bounds), cdiv(bounds, CHUNK), 0), lead)
    live = need > 0
    lead, span, need = lead[live], span[live], need[live]
    if not need.size:
        return
    assert need.size < ctas, (need.size, ctas)
    # the width search's range: ceil(total / ctas), the least that could fit,
    # to ceil(total / (ctas - pieces)), which does (a piece's ceil adds under
    # one task); feasibility is monotone, so the first width that fits
    total = int(need.sum())
    lo = max(cdiv(total, ctas), 1)
    widths = np.arange(lo, max(cdiv(total, ctas - need.size), lo) + 1)[:, None]
    width = lo + int(np.argmax(cdiv(need, widths).sum(1) <= ctas))
    k = cdiv(need, width)
    piece = np.repeat(np.arange(need.size), k)
    i = np.arange(piece.size) - np.repeat(np.cumsum(k) - k, k)
    tasks = plan[: piece.size]
    tasks[:, 0] = lead[piece]
    tasks[:, 1] = span[piece]
    tasks[:, 2] = i * need[piece] // k[piece]
    tasks[:, 3] = (i + 1) * need[piece] // k[piece]


# a REINDEX token's chunks at most: its candidate blocks' rows
REINDEX_CHUNKS = TOPK_BLOCKS * BLOCK_ROWS // CHUNK


def reindex_split(tokens, ctas, least):
    """A REINDEX layer's tasks a token (each its own piece), REINDEX_CHUNKS shared over them and
    fixed by the step's width, so a CTA finds its task from its index alone; ``least`` chunks a task
    at least."""
    return max(1, min(ctas // tokens, REINDEX_CHUNKS // least))


def reindex_chunks(i, per):
    """Chunks [ch0, ch1) of a REINDEX token's task i of ``per``."""
    return i * REINDEX_CHUNKS // per, (i + 1) * REINDEX_CHUNKS // per
