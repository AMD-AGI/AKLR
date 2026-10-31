# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
import os
from pathlib import Path

__all__ = [
    "DEFAULT_EXPORT_DIR",
    "ENV_DISABLE",
    "ENV_EXPORT",
    "ENV_EXPORT_DIR",
    "ENV_KERNEL_CACHE_DIR",
    "ENV_MEASURED_DIR",
    "ENV_MIN_SPEEDUP",
    "ENV_SGLANG_PLUGIN",
    "RuntimeEnv",
]

# RuntimeEnv() reads them; e2e writes them for its child processes
ENV_DISABLE = "AKLR_DISABLE"
ENV_MIN_SPEEDUP = "AKLR_MIN_SPEEDUP"
ENV_KERNEL_CACHE_DIR = "AKLR_KERNEL_CACHE_DIR"
ENV_EXPORT = "AKLR_EXPORT"
ENV_EXPORT_DIR = "AKLR_EXPORT_DIR"
ENV_MEASURED_DIR = "AKLR_MEASURED_DIR"
ENV_SGLANG_PLUGIN = "AKLR_SGLANG_PLUGIN"
DEFAULT_EXPORT_DIR = "aklr_export"


class RuntimeEnv:
    """The environment that decides what the runtime builds, serves and records;
    ``""`` is unset, a bad value raises ``ValueError`` naming the variable."""

    def __init__(self) -> None:
        # AKLR_DISABLE: "1" all, "0" or unset none, else contract ids the runtime checks
        disable = os.environ.get(ENV_DISABLE) or "0"
        self.is_all_disabled = disable == "1"
        self.disabled_contracts: frozenset[str] = (
            frozenset() if disable in ("0", "1") else frozenset(item.strip() for item in disable.split(","))
        )
        export = os.environ.get(ENV_EXPORT) or "0"
        if export not in ("0", "1"):
            raise ValueError(f"{ENV_EXPORT}={export!r}: expected '1' or '0'")
        self.is_export_enabled = export == "1"
        # AKLR_SGLANG_PLUGIN: "1" hooks sglang's layers to the plugin's ops
        plugin = os.environ.get(ENV_SGLANG_PLUGIN) or "0"
        if plugin not in ("0", "1"):
            raise ValueError(f"{ENV_SGLANG_PLUGIN}={plugin!r}: expected '1' or '0'")
        self.is_sglang_plugin_enabled = plugin == "1"
        self.export_dir = Path(os.environ.get(ENV_EXPORT_DIR) or DEFAULT_EXPORT_DIR).absolute()
        # AKLR_MIN_SPEEDUP: least speedup over aiter to serve; <= 0 also serves unmeasured
        speedup = os.environ.get(ENV_MIN_SPEEDUP) or "1.03"
        try:
            self.min_speedup = float(speedup)
        except ValueError as e:
            raise ValueError(f"{ENV_MIN_SPEEDUP}={speedup!r}: {e}") from e
        if not math.isfinite(self.min_speedup):
            raise ValueError(f"{ENV_MIN_SPEEDUP}={speedup!r}: expected a finite number")
        # AKLR_MEASURED_DIR: schemas laid out as the checkout that replace the committed
        # ones they name; validation's e2e serves its bench's measurements through it
        directory = os.environ.get(ENV_MEASURED_DIR)
        self.measured_dir = Path(directory).absolute() if directory else None
        # reads HOME itself, not Path.home(), which falls back to the password database
        self.build_cache_path: Path | None = None
        if directory := os.environ.get(ENV_KERNEL_CACHE_DIR):
            self.build_cache_path = Path(directory).absolute()
        elif directory := os.environ.get("XDG_CACHE_HOME"):
            self.build_cache_path = Path(directory, "aklr", "kernels").absolute()
        elif directory := os.environ.get("HOME"):
            self.build_cache_path = Path(directory, ".cache", "aklr", "kernels").absolute()
