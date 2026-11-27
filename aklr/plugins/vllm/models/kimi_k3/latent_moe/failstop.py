# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The latent-MoE tail's fail-stop in vLLM: after every step's launches, this rank's group FAILED word (the kernel's,
``TailGroup.failed_word`` of its runtime) is copied to the host, and the step's output leaves the worker only through
a check of that copy, which raises once the word is set (the vLLM plugin's ``AFTER_STEP``). A wait of any rank that
gave up sets the word on every rank before anything more is stored, so a step whose launches read anything made from
such a wait never reaches the scheduler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

__all__ = ["after_step", "watch"]

# the watch of each group set up in this process
_WATCHES: list[_FailStop] = []


def watch(group: Any) -> None:
    """Check ``group``'s FAILED word after every step from now on (``after_step`` among the plugin's step watches)."""
    from aklr.plugins import vllm as plugin

    _WATCHES.append(_FailStop(group.hip, group.failed_word, group.peer_table.device))
    if after_step not in plugin.AFTER_STEP:
        plugin.AFTER_STEP.append(after_step)


def after_step() -> Callable[[], None] | None:
    """The fail-stop watch of every group of this process, the vLLM plugin's step watch: once a step, on every rank,
    after the step's launches. Its check, run before the step's output leaves the worker, raises once the group's
    FAILED word was set in that step or before; None inside a graph capture (no host call there) or without a
    group."""
    checks = [check for check in (w.after_step() for w in list(_WATCHES)) if check is not None]
    if not checks:
        return None

    def check() -> None:
        for one in checks:
            one()

    return check


class _FailStop:
    """A step's verdict: this rank's group FAILED word as the step's launches left it, copied to a pinned slot on a
    stream of its own once the step's launches are done (the step's next kernels do not wait for the copy, nor does
    the host: no collective, no device read). Its check waits for that copy alone, in vLLM's output thread, and
    raises where the word is set. A slot is reused RING steps on, long after vLLM has read the step's output. The
    word is sticky, so a copy never reads it clearer than the step left it."""

    RING = 64

    def __init__(self, hip: Any, word: int, device: torch.device) -> None:
        self.hip, self.word = hip, word
        cuda = device.type == "cuda"
        self.host = torch.zeros(self.RING, dtype=torch.int32, pin_memory=cuda)
        self.stream = torch.cuda.Stream(device) if cuda else None
        # each slot's events: the step's launches done (on the step's stream), its copy landed
        self.ready = [torch.cuda.Event() for _ in range(self.RING)] if cuda else None
        self.copied = [torch.cuda.Event(blocking=True) for _ in range(self.RING)] if cuda else None
        self.steps = 0

    def after_step(self) -> Callable[[], None] | None:
        if torch.cuda.is_current_stream_capturing():
            return None
        slot = self.steps % self.RING
        self.steps += 1
        host, copied = self.host[slot : slot + 1], None
        if self.stream is None:
            self.hip.copy_word(host, self.word, None)
        else:
            self.ready[slot].record(torch.cuda.current_stream(self.stream.device))
            self.stream.wait_event(self.ready[slot])
            self.hip.copy_word(host, self.word, self.stream.cuda_stream)
            copied = self.copied[slot]
            copied.record(self.stream)

        def check() -> None:
            if copied is not None:
                copied.synchronize()
            if host.item():
                raise RuntimeError(
                    "Kimi-K3 latent-MoE tail: a TP rank gave up an in-kernel wait in this step or before (each rank's "
                    "TailGroup.health records its own); stopping (AKLR_DISABLE=kimi_k3_latent_moe_tail@1.0 runs vLLM's "
                    "tail)"
                )

        return check
