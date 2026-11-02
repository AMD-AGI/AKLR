# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/sync.py
"""The tagged mailbox: hand-offs between the CTAs of one GPU (device scope) and
between GPUs (system scope), every address a named region (``Addr``)."""

from dataclasses import dataclass
from enum import Enum

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aklr.plugins.sglang.models.deepseek_v4.kernels.common.aiter_flydsl import (
    buffer_ops as bo,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.common.ops import (
    CM_DEV,
    CM_SYS,
    bf16_pair,
    rsrc,
    traced,
)

POLL_MAX = 12  # mailbox specs polled per batch (bounds live registers)
SPIN_SLEEP = 4  # infera-flash: s_sleep units (64 clocks) between polls; back-to-back polls slow the producers


@traced
def _spin(load_all, pending):
    """Re-issue ``load_all`` until ``pending`` clears; a side-effecting op in the
    loop keeps the loads from being hoisted."""
    v = load_all()
    while pending(v):
        rocdl.s_sleep(SPIN_SLEEP)
        v = load_all()
    return v


class Space(Enum):
    SCRATCH = "scratch"  # this GPU's memory: device-scope hand-off between CTAs
    PEER = "peer"  # a peer-symmetric region: system-scope hand-off between GPUs


@dataclass(frozen=True)
class Addr:
    """A mailbox region's address: the region's name and space, and the traced
    address. The space sets the hand-off's scope: device for SCRATCH, system for
    PEER."""

    region: str
    space: Space
    value: object


def sreg(base, offset, region):
    """Scratch region ``region`` at byte ``offset`` of ``base``."""
    return Addr(region, Space.SCRATCH, base + fx.Int64(offset))


def preg(base, offset, region):
    """Peer-symmetric region ``region`` at byte ``offset`` of a rank's buffer
    ``base``."""
    return Addr(region, Space.PEER, base + fx.Int64(offset))


# the store / load cache policy of a hand-off in each space
_SCOPE_CM = {Space.SCRATCH: CM_DEV, Space.PEER: CM_SYS}


class Mailbox:
    """Tagged-pair hand-off: each 32-bit value is stored next to this launch's tag (its
    epoch, so an earlier step's pair is never live), polled until all tags match: one
    round trip, no drain or flag. An ``Addr``'s space picks the scope."""

    def __init__(self, tag):
        self.tag = tag

    def put(self, a, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(a.value),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def put_bf(self, a, i, vs):
        """Elements i .. i+len(vs) (2 or 4, i aligned) as packed bf16 pairs, one
        store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(a.value),
            i,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def put_words(self, a, i, words):
        """Raw 32-bit words (1 or 2) at pairs i.. as (word, tag). Integer payloads
        (fp8 words) must come this way: through an f32 value a NaN bit pattern is
        not kept."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(a.value),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def poll(self, specs, batch=POLL_MAX):
        """Batched poll of ``specs`` = [(Addr, pair index, npairs in {1, 2})], one space:
        all re-loaded together while any tag is stale (one round trip after the last
        producer lands). Returns the value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        spaces = {a.space for a, _, _ in specs}
        assert len(spaces) == 1, f"one poll batch spans {spaces}"
        cm = _SCOPE_CM[spaces.pop()]
        bases = [a.value for a, _, _ in specs]
        tag = self.tag

        def load_all():
            words = []
            for b, (_, i, n) in zip(bases, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v = _spin(load_all, pending)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs


@traced
def publish(put, a, i, v, who):
    """This CTA's plain stores drained, then (thread ``who``) the pair (``a``, i) := v. A
    workgroup release fence would not wait for them: LLVM's AMDGPU model omits vmcnt
    outside tgsplit mode, but the readers here are on other CUs."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if who:
        put(a, i, v)
