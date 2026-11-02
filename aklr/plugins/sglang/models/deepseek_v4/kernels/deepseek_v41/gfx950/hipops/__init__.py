# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Infera-Flash's HIP kernels as one torch extension built at first use (``csrc/ops.hip``
registers the ops): the DSpark draft's Markov step and greedy argmax with the TP ranks' bests swapped
in-kernel (``markov_step``, ``argmax_xchg``), ``index_q_pack`` and ``swa_store``."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import torch

_CSRC = Path(__file__).resolve().parent / "csrc"
# markov.h: the one-sequence steps' packets in slots 0-15, a batch's from 512, the argmax's
# from 1024, the chain's from 4096 (uint32x4 each)
EXCHANGE_BYTES = 128 << 10
MAX_SEQUENCES = 32  # MARKOV_BATCH_MAX
DIM = 256  # MarkovParams::DIM


def _sources() -> dict:
    return {
        str(p.relative_to(_CSRC)): p.read_text(encoding="utf-8")
        for p in sorted(_CSRC.rglob("*"))
        if p.suffix in (".h", ".hip")
    }


@functools.cache
def ops():
    """``torch.ops.<library>`` with ``markov_step``, ``argmax_xchg`` and ``index_q_pack``,
    built first on a new toolchain or source."""
    from aklr.core.jit import hip_extension
    from aklr.core.runtime.builder.builder import get_build_cache_path

    sources = _sources()
    digest = hashlib.sha256("".join(f"{k}\0{v}\0" for k, v in sorted(sources.items())).encode()).hexdigest()[
        :16
    ]
    name = f"aklr_dsv41_hipops_{digest}"
    hip_extension.load_library(name, sources, get_build_cache_path() / "dsv41" / name, "gfx950")
    return getattr(torch.ops, name)


class Exchange:
    """A TP group's exchange: ``peers`` (every rank's buffer address, int64 on the host) and the
    steps' and argmax's call counters and last tokens. ``group`` is a CPU (gloo) group; collective.
    """

    def __init__(self, group, rank: int, size: int, device: torch.device):
        from aklr.plugins.sglang.models.deepseek_v4.kernels.common.peer import (
            PeerBuffer,
        )

        self.rank, self.size = rank, size
        self._buffer = PeerBuffer(EXCHANGE_BYTES, group, rank, size, device)
        self.peers = self._buffer.addresses.cpu()
        i32 = {"dtype": torch.int32, "device": device}
        i64 = {"dtype": torch.int64, "device": device}
        self.calls, self.tok = torch.zeros(16, **i32), torch.zeros(1, **i64)
        self.seq_calls = torch.zeros(64, **i32)
        self.seq_tok = torch.zeros(64, **i64)
        self.argmax_calls = torch.zeros(1, **i32)
