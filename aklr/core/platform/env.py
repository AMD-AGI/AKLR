# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
import os
from pathlib import Path

__all__ = [
    "ENV_BUILD_CACHE_DIR",
    "ENV_DISABLE",
    "ENV_ENABLE_TRACING",
    "ENV_MIN_SPEEDUP",
    "ENV_SCHEMA_PATH",
    "RuntimeEnv",
]

# RuntimeEnv() reads them; e2e writes them for its child processes
ENV_DISABLE = "AKLR_DISABLE"
ENV_MIN_SPEEDUP = "AKLR_MIN_SPEEDUP"
ENV_BUILD_CACHE_DIR = "AKLR_BUILD_CACHE_DIR"
ENV_ENABLE_TRACING = "AKLR_ENABLE_TRACING"
ENV_SCHEMA_PATH = "AKLR_SCHEMA_PATH"


class RuntimeEnv:
    """The environment that decides what the runtime builds, serves and records;
    ``""`` is unset, a bad value raises ``ValueError`` naming the variable."""

    def __init__(self) -> None:
        # AKLR_DISABLE: "1" all, "0" or unset none, else op types the runtime checks;
        # an older AKLR's id "<op_type>@<version>" names its op type
        disable = os.environ.get(ENV_DISABLE) or "0"
        self.is_all_disabled = disable == "1"
        self.disabled_op_types: frozenset[str] = (
            frozenset()
            if disable in ("0", "1")
            else frozenset(item.strip().split("@", 1)[0] for item in disable.split(","))
        )
        tracing = os.environ.get(ENV_ENABLE_TRACING) or "0"
        if tracing not in ("0", "1"):
            raise ValueError(f"{ENV_ENABLE_TRACING}={tracing!r}: expected '1' or '0'")
        self.is_tracing_enabled = tracing == "1"
        directory = os.environ.get(ENV_SCHEMA_PATH)
        self.schema_path = Path(directory).absolute() if directory else None
        # AKLR_MIN_SPEEDUP: least speedup over aiter to serve; <= 0 also serves unmeasured
        speedup = os.environ.get(ENV_MIN_SPEEDUP) or "1.03"
        try:
            self.min_speedup = float(speedup)
        except ValueError as e:
            raise ValueError(f"{ENV_MIN_SPEEDUP}={speedup!r}: {e}") from e
        if not math.isfinite(self.min_speedup):
            raise ValueError(f"{ENV_MIN_SPEEDUP}={speedup!r}: expected a finite number")
        # expanduser, unlike Path.home(), never raises when HOME and the password database both miss
        if directory := os.environ.get(ENV_BUILD_CACHE_DIR):
            self.build_cache_path = Path(directory).absolute()
        elif directory := os.environ.get("XDG_CACHE_HOME"):
            self.build_cache_path = Path(directory, "aklr", "kernels").absolute()
        else:
            self.build_cache_path = Path(os.path.expanduser("~/.cache/aklr/kernels")).absolute()
