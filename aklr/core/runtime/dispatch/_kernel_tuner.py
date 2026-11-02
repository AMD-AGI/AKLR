###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from aklr.core.runtime.dispatch._dispatch_key import DispatchKey


class KernelTuner(Protocol):
    """Picks among a key's ranked kernels online when no baseline was measured; ``run(name)`` runs one."""

    def choose(
        self,
        schema_name: str,
        key: DispatchKey,
        candidates: list[str],
        run: Callable[[str], Any],
    ) -> str: ...
