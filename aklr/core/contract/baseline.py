# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The base classes of an op's baseline: the baseline code each op writes, what the engine runs
without AKLR, and the profile that times one call, the kernel's and the baseline's the one way.
A baseline holds no contract: the contract calls it and passes in what it needs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["GRAPH_BUDGET_MILLISECONDS", "DistOpBaseline", "OpBaseline"]

# a CUDA graph's replays per reading fill about this long
GRAPH_BUDGET_MILLISECONDS = 20.0


class OpBaseline(ABC):
    def setup(self) -> None:  # noqa: B027 - optional
        """Before a process's first call: what the calls take for granted beyond their arguments,
        set up as the engine does before its first (mega_moe's mori heap). Nothing by default."""

    def teardown(self) -> None:  # noqa: B027 - optional
        """After a process's last call: undoes ``setup``."""

    @abstractmethod
    def baseline(self, **call: Any) -> Mapping[str, Tensor]:
        """The outputs, by name, of what the engine runs without AKLR on the kernel's call ``call``;
        the mutable inputs it writes are read from ``call``."""

    def profile(self, fn: Callable[[], object]) -> float:
        """Milliseconds of one call of ``fn`` in a CUDA graph, its cache flushes taken out."""
        from aklr.core.utils.testing import bench_cudagraph

        return bench_cudagraph(fn, rep=GRAPH_BUDGET_MILLISECONDS)


class DistOpBaseline(OpBaseline):
    """An op that ranks call together. ``baseline`` runs in each rank's process on that rank's
    call, ``setup`` once the process is in torch.distributed's default group."""

    def profile(self, fn: Callable[[], object]) -> float:
        """Every rank starts each call of ``fn`` together; the slowest rank's milliseconds."""
        import torch.distributed as dist

        from aklr.core.utils.testing import bench_cudagraph

        return bench_cudagraph(fn, rep=GRAPH_BUDGET_MILLISECONDS, group=dist.group.WORLD)
