###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The graph timer: the Candidate and its opponent timed in CUDA graphs on every rank of the task's group."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

__all__ = ["GraphTimer", "TimingFault"]


class TimingFault(RuntimeError):
    """A timing that failed: the participant named raised, or, with none named, the tracer failed twice."""

    def __init__(self, participant: str | None, error: BaseException) -> None:
        super().__init__(f"{participant or 'tracer'}: {type(error).__name__}: {error}")
        self.participant = participant


class GraphTimer:
    def __init__(self, budget_ms: float) -> None:
        """budget_ms: the profile's graph budget, the time a participant's graph replays fill."""
        self.budget_ms = budget_ms

    def time(
        self,
        candidate: Callable[[], object],
        opponent: Callable[[], object],
        seed: int,
    ) -> tuple[float, float]:
        """The Candidate's and its opponent's GPU milliseconds per call in a CUDA graph, one timing each, in a
        seeded shuffled order; on a group, each call the slowest rank's. A tracer fault is timed again once, on
        every rank together: a timing that raised raises on every rank, the participant's if one rank's raised."""
        import torch.distributed as dist

        from aklr.core.utils.testing import bench_cudagraph

        participants = {"candidate": candidate, "opponent": opponent}
        order = list(participants)
        random.Random(seed).shuffle(order)
        timings: dict[str, float] = {}
        for participant in order:
            fn = participants[participant]
            raised: list[BaseException] = []

            def call(fn: Callable[[], object] = fn, raised: list[BaseException] = raised) -> None:
                try:
                    fn()
                except BaseException as error:
                    raised.append(error)
                    raise

            fault: BaseException = RuntimeError("no timing")
            for _ in range(2):
                try:
                    ms = bench_cudagraph(call, rep=self.budget_ms, group=dist.group.WORLD)
                except Exception as error:  # noqa: BLE001 -- the participant's, else the tracer's
                    own = f"{type(raised[0]).__name__}: {raised[0]}" if raised else None
                    messages: list[str | None] = [None] * dist.get_world_size()
                    dist.all_gather_object(messages, own)
                    for rank, message in enumerate(messages):
                        if message is not None:
                            cause = (
                                raised[0] if len(messages) == 1 else RuntimeError(f"rank {rank}: {message}")
                            )
                            raise TimingFault(participant, cause) from cause
                    fault = error
                    continue
                if math.isfinite(ms) and ms > 0:
                    timings[participant] = ms
                    break
                fault = RuntimeError(f"the tracer read {ms} ms")
            else:
                raise TimingFault(None, fault)
        return timings["candidate"], timings["opponent"]
