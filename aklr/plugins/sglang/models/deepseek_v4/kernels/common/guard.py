# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang-AMD-Mono project
"""Deadlines, sticky error words and the stop of a failed launch. A wait past its deadline sets
its GPU's error word, flags every peer and stops the launch, which never runs on with words that did
not arrive. The guard block is the first ``GUARD_WORDS`` int32 of the epoch buffer."""

import contextlib
import os

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from aklr.plugins.sglang.models.deepseek_v4.kernels.common.aiter_flydsl import (
    buffer_ops as bo,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.common.ops import (
    CM_DEV,
    CM_SYS,
    global_ptr,
    load_ptr64,
    rsrc,
)

# re-exported: sync.py calls guard.clock
from aklr.plugins.sglang.models.deepseek_v4.kernels.common.ops import (
    clock as clock,  # noqa: PLC0414
)

GUARD_DEADLINE = 1  # int64 at words 1-2
ERR_DEADLINE = 3
ERR_PEER = 4
ERR_FAULT = 5
ERROR_WORDS = (ERR_DEADLINE, ERR_PEER, ERR_FAULT)
GUARD_WORDS = 8  # the epoch buffer's words before the per-CTA marks

FLAG_WORDS = 8  # one a rank; TP 8 at most
FLAG_BYTES = 256  # the peer buffer's flags region, padded to a cache line pair

CLOCK_HZ = 100_000_000  # s_memrealtime: the constant 100 MHz reference clock


def deadline_ticks(seconds: float) -> int:
    return int(seconds * CLOCK_HZ)


def fault_injection(rank: int) -> tuple[int, int] | None:
    """(site, after): the wait this rank stalls at once ``after`` launches have run,
    from ``SGLANG_AMD_MONO_FAULT_INJECT`` (``rank:site[:after]``), or None. A test
    hook: the stalled wait times out."""
    spec = os.environ.get("SGLANG_AMD_MONO_FAULT_INJECT", "")
    if not spec:
        return None
    r, site, after = (spec.split(":") + ["0", "0"])[:3]
    return (int(site or 0), int(after or 0)) if int(r) == rank else None


def stop_launch():
    """Stop the launch: a volatile store to the null page faults at 0x0 (told apart from a real
    fault) and the runtime tears the queue down. Unlike ``llvm.trap`` it asks nothing of the
    kernel's ABI."""
    fx.generic_store(global_ptr(0), fx.Int32(1), volatile=True)


class Guard:
    """One launch's deadline and error words, and optionally the peers' flags. Without ``own`` /
    ``peers`` (peer buffer bases) a failure stays on this GPU; ``stop=False`` keeps a failed
    launch running (the one-GPU test reads the words afterwards)."""

    def __init__(
        self,
        epoch,
        own=None,
        peers=None,
        rank=None,
        npes=1,
        flags_offset=0,
        fault_site=None,
        fault_after=0,
        stop=True,
    ):
        self.epoch = epoch
        self.own = own
        self.peers = peers
        self.rank = rank
        self.npes = npes
        self.flags_offset = flags_offset
        self.fault_site = fault_site
        self.fault_after = fault_after
        self.stop = stop
        self._sites = 0

    def armed(self):
        """The fault injection's stall is on: ``fault_after`` launches have run."""
        ep = fx.Int32(bo.buffer_load(rsrc(self.epoch), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
        return ep >= fx.Int32(self.fault_after)

    def deadline(self):
        """The deadline in clock ticks, loaded where a check needs it."""
        r = rsrc(self.epoch)
        lo = fx.Int32(bo.buffer_load(r, GUARD_DEADLINE, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
        hi = fx.Int32(bo.buffer_load(r, GUARD_DEADLINE + 1, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
        return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    def next_site(self) -> int:
        self._sites += 1
        return self._sites

    def _word(self, i):
        return fx.Int32(bo.buffer_load(rsrc(self.epoch), i, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))

    def flags(self):
        """This rank's flags as its peers wrote them (system scope); 0 when clean."""
        if self.own is None:
            return fx.Int32(0)
        acc = fx.Int32(0)
        # a buffer load carries at most 4 dwords: TP 8 reads its flags in two
        for c0 in range_constexpr(0, self.npes, 4):
            n = min(4, self.npes - c0)
            w = bo.buffer_load(
                rsrc(self.own + fx.Int64(self.flags_offset)),
                c0,
                vec_width=n,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
            if const_expr(n == 1):
                acc = acc | fx.Int32(w)
            else:
                v = fx.Vector(w)
                for p in range_constexpr(n):
                    acc = acc | v[p]
        return acc

    def stopped(self):
        """Whether this GPU already failed or saw a peer fail: later waits return at once."""
        own = self._word(ERR_DEADLINE) | self._word(ERR_PEER) | self._word(ERR_FAULT)
        return (own | self.flags()) != 0

    def fail(self, site):
        """This GPU's wait at ``site`` passed its deadline: set the error word, flag
        every peer, fence, and stop the launch."""
        bo.buffer_store(fx.Int32(site + 1), rsrc(self.epoch), ERR_DEADLINE, cache_modifier=CM_DEV)
        if self.peers is not None:
            for p in range_constexpr(self.npes):
                bo.buffer_store(
                    fx.Int32(1),
                    rsrc(load_ptr64(self.peers, p) + fx.Int64(self.flags_offset)),
                    self.rank,
                    cache_modifier=CM_SYS,
                )
        rocdl.s_waitcnt(vmcnt=0)
        if const_expr(self.stop):
            stop_launch()

    def note_peer(self):
        """A wait saw a failure elsewhere: stop the launch (the failed rank flagged
        everyone), or without the stop keep a sticky ``peer`` word."""
        if const_expr(self.stop):
            stop_launch()
        else:
            old = self._word(ERR_PEER)
            bo.buffer_store(
                (old == 0).select(fx.Int32(1), old),
                rsrc(self.epoch),
                ERR_PEER,
                cache_modifier=CM_DEV,
            )


_CURRENT: list[Guard] = []


@contextlib.contextmanager
def scope(guard: Guard | None):
    """Make ``guard`` the one the waits traced inside this block take."""
    _CURRENT.append(guard)
    try:
        yield guard
    finally:
        _CURRENT.pop()


def current() -> Guard | None:
    return _CURRENT[-1] if _CURRENT else None
