# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The registry of adapters and the patch manager that installs them."""

from __future__ import annotations

from typing import ClassVar

from aklr.core.runtime.seam._adapter import Adapter
from aklr.core.runtime.seam._patch_manager import PatchManager

__all__ = ["SeamRegistry"]


class SeamRegistry:
    """Every declared adapter, installed through one patch manager."""

    _default: ClassVar[SeamRegistry | None] = None

    def __init__(self) -> None:
        self._patches = PatchManager()
        self._adapters: dict[str, Adapter] = {}

    @classmethod
    def default(cls) -> SeamRegistry:
        """The process-wide registry."""
        if cls._default is None:
            cls._default = cls()
        return cls._default

    def add(self, adapter: Adapter) -> None:
        """Declare ``adapter``; ``install`` patches it."""
        if adapter.entry in self._adapters:
            raise ValueError(f"{adapter.entry}: two adapters for one entry point")
        self._adapters[adapter.entry] = adapter

    def adapters(self, op_type: str) -> tuple[Adapter, ...]:
        """The adapters whose api serves a schema of ``op_type``, in declaration order."""
        return tuple(a for a in self._adapters.values() if any(s.op_type == op_type for s in a.api.schemas))

    def install(self) -> None:
        """Patch every declared entry point this environment has."""
        import aklr.seam  # noqa: F401 -- declares every adapter: aiter, a submodule, does not import it

        for adapter in self._adapters.values():
            self._patches.patch(adapter)

    def uninstall(self) -> None:
        """Restore every patched entry point."""
        self._patches.unpatch_all()
