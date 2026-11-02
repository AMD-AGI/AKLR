# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang#43984 (kernels/ops/mono/gfx950/deepseek_v41/indexer/k2i.py), whose
# scorer comes from ROCm/ATOM at 68506823 (MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.)
"""K2i: ATOM K2a's FP4 scorer (``score_fp4``) as a launch of its own between K1 and K2 of an
index source, writing each row's logits into ``ilog`` for SGLang's top-k. Each CTA scores the task
the walk plan (FULL) or its index (REINDEX) gives it, and none waits on another."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from aklr.plugins.sglang.models.deepseek_v4.kernels.common.epoch import kernel_symbol
from aklr.plugins.sglang.models.deepseek_v4.kernels.common.plan import (
    BLOCKS,
    THREADS,
    kernel_sources,
)

from .score_fp4 import ROWS_LDS_WORDS, run_scores4

SOURCES = kernel_sources(__name__)

_STREAM = fx.Stream(None)
MAX_ROWS = 48  # a step's rows at most (8 DSpark requests of 6)


def build_k2i(tokens: int, page8: bool = False):
    """K2i for a step of ``tokens`` rows, on an index plane of 64-row pages, or
    of row-group 8-slot pages (``page8``: SGLANG_HIP_DSV41_INDEX_K_PAGE8)."""
    assert 0 < tokens <= MAX_ROWS, tokens
    name = kernel_symbol("dsv41_mono_k2i", s=tokens, **({"p8": 1} if page8 else {}))
    keyed = (("tokens", tokens), ("page8", page8), ("sources", SOURCES))

    @fx.struct
    class RowsLds:
        rows: fx.Array[fx.Int32, ROWS_LDS_WORDS, 16]

    @fx.union
    class K2iLds:
        scorer: RowsLds

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2i(
        iq: Int64,
        iqs: Int64,
        iw: Int64,
        plane: Int64,
        pscale: Int64,
        iplan: Int64,
        itab: Int64,
        itab_stride: Int32,
        itab_len: Int32,
        ibatch: Int64,
        ishift: Int32,
        ibound: Int64,
        ilog: Int64,
        lstride: Int32,
        reindex: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(K2iLds).scorer.peek()
        args = {
            "iq": iq,
            "iqs": iqs,
            "iw": iw,
            "plane": plane,
            "pscale": pscale,
            "iplan": iplan,
            "itab": itab,
            "itab_stride": itab_stride,
            "itab_len": itab_len,
            "ibatch": ibatch,
            "ishift": ishift,
            "ibound": ibound,
            "ilog": ilog,
            "lstride": lstride,
            "reindex": reindex,
        }
        c = {
            "tid": tid,
            "lane": tid % 64,
            "wave": tid // 64,
            "args": args,
            "rows_lds": lds.rows.ptr,
            "page8": page8,
        }
        run_scores4(c, tokens, fx.block_idx.x)

    @flyc.jit
    def launch(
        iq: Int64,
        iqs: Int64,
        iw: Int64,
        plane: Int64,
        pscale: Int64,
        iplan: Int64,
        itab: Int64,
        itab_stride: Int32,
        itab_len: Int32,
        ibatch: Int64,
        ishift: Int32,
        ibound: Int64,
        ilog: Int64,
        lstride: Int32,
        reindex: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2i(
            iq,
            iqs,
            iw,
            plane,
            pscale,
            iplan,
            itab,
            itab_stride,
            itab_len,
            ibatch,
            ishift,
            ibound,
            ilog,
            lstride,
            reindex,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
