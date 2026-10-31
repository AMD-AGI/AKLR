# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from aklr.core.runtime.dispatch._dispatch_key import DispatchKey


class KernelTuner(Protocol):
    def choose(
        self,
        schema_name: str,
        key: DispatchKey,
        candidates: list[str],
        run: Callable[[str], Any],
    ) -> str: ...
