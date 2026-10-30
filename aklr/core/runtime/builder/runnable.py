###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Source: flashinfer_bench/compile/runnable.py
# Modified by Advanced Micro Devices, Inc. for the AKLR project.
# Modified for aklr/core/runtime/builder
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of aiter (see LICENSE).
###############################################################################


from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from pydantic import BaseModel, Field

from aklr.core.runtime.peer_memory import PeerMemory, open_peer_memory

__all__ = ["Runnable", "RunnableMetadata"]


class RunnableMetadata(BaseModel):
    """How a runnable was built and what it implements."""

    build_type: str
    """The language of the builder that produced it."""
    contract_id: str
    """The contract whose call keys it takes."""
    schema_name: str
    """The name of the first schema it was built for."""
    content_hash: str
    """``kernel.content_hash``: the registry's cache key."""
    misc: dict[str, Any] = Field(default_factory=dict)
    """Builder-specific extras: ``module_name``, ``entry_symbol``."""


class Runnable:
    """A schema's imported kernel, called with one call's api params (``api_params`` names) as keywords; it fills the outputs.

    A kernel with ``state_bytes`` also gets ``state``: that many bytes on the device of the
    call's tensors, zeroed at its first call there and kept for every later one. A kernel with
    ``peer_bytes`` also gets ``peers`` and ``rank`` of its peer memory on that device, opened
    at the same first call (``peer_memory.open_peer_memory``). The dispatcher makes that first
    call outside a graph capture (``FallbackReason.NOT_WARMED``)."""

    metadata: RunnableMetadata

    _callable: Callable[..., Any]
    _cleaner: Callable[[], None] | None
    _state_bytes: int
    _peer_bytes: int
    _states: dict[torch.device, torch.Tensor]
    _peers: dict[torch.device, PeerMemory]

    def __init__(
        self,
        fn: Callable[..., Any],
        metadata: RunnableMetadata,
        cleaner: Callable[[], None] | None = None,
        state_bytes: int = 0,
        peer_bytes: int = 0,
    ) -> None:
        self._callable = fn
        self.metadata = metadata
        self._cleaner = cleaner
        self._state_bytes = state_bytes
        self._peer_bytes = peer_bytes
        self._states = {}
        self._peers = {}

    def __call__(self, **call: Any) -> Any:
        if self._state_bytes or self._peer_bytes:
            device = next(v.device for v in call.values() if isinstance(v, torch.Tensor))
            if self._state_bytes:
                if device not in self._states:
                    self._states[device] = torch.zeros(self._state_bytes, dtype=torch.uint8, device=device)
                call["state"] = self._states[device]
            if self._peer_bytes:
                if device not in self._peers:
                    self._peers[device] = open_peer_memory(self._peer_bytes, device)
                peers = self._peers[device]
                call["peers"], call["rank"] = peers.addresses, peers.rank
        return self._callable(**call)

    def cleanup(self) -> None:
        """Run the cleaner once (the kernel's package leaves ``sys.modules``); safe to call more than once."""
        if self._cleaner:
            try:
                self._cleaner()
            finally:
                self._cleaner = None
