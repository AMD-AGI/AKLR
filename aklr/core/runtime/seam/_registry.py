###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import ClassVar

from aklr.core.runtime.seam._adapter import Adapter
from aklr.core.runtime.seam._patch_manager import PatchManager

__all__ = ["SeamRegistry"]


class SeamRegistry:
    """The declared adapters of backend entry points; ``install`` patches in those this environment has."""

    _default: ClassVar[SeamRegistry | None] = None

    def __init__(self) -> None:
        self._patches = PatchManager()
        self._adapters: dict[str, Adapter] = {}

    @classmethod
    def default(cls) -> SeamRegistry:
        if cls._default is None:
            cls._default = cls()
        return cls._default

    def add(self, adapter: Adapter) -> None:
        if adapter.entry in self._adapters:
            raise ValueError(f"{adapter.entry}: two adapters for one entry point")
        self._adapters[adapter.entry] = adapter

    def adapters(self, op_type: str) -> tuple[Adapter, ...]:
        return tuple(a for a in self._adapters.values() if any(s.op_type == op_type for s in a.api.schemas))

    def install(self) -> None:
        import aklr.seam  # noqa: F401 -- declares every adapter: aiter, a submodule, does not import it

        for adapter in self._adapters.values():
            self._patches.patch(adapter)

    def uninstall(self) -> None:
        self._patches.unpatch_all()
