# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
import os
from pathlib import Path

__all__ = [
    "DEFAULT_EXPORT_DIR",
    "DEFAULT_MIN_SPEEDUP",
    "ENV_DISABLE",
    "ENV_EXPORT",
    "ENV_EXPORT_DIR",
    "ENV_FLYDSL_FASTCALL",
    "ENV_KERNEL_CACHE_DIR",
    "ENV_MEASURED_DIR",
    "ENV_MIN_SPEEDUP",
    "ENV_SGLANG_PLUGIN",
    "ENV_TRACE",
    "ENV_VLLM_PLUGIN",
    "RuntimeEnv",
    "trace_mode",
]

# RuntimeEnv() reads them
ENV_DISABLE = "AKLR_DISABLE"
ENV_MIN_SPEEDUP = "AKLR_MIN_SPEEDUP"
ENV_KERNEL_CACHE_DIR = "AKLR_KERNEL_CACHE_DIR"
ENV_EXPORT = "AKLR_EXPORT"
ENV_EXPORT_DIR = "AKLR_EXPORT_DIR"
ENV_MEASURED_DIR = "AKLR_MEASURED_DIR"
ENV_FLYDSL_FASTCALL = "AKLR_FLYDSL_FASTCALL"
ENV_SGLANG_PLUGIN = "AKLR_SGLANG_PLUGIN"
ENV_VLLM_PLUGIN = "AKLR_VLLM_PLUGIN"
ENV_TRACE = "AKLR_TRACE"
DEFAULT_EXPORT_DIR = "aklr_export"
# the least published speedup over the baseline that validation admits a kernel at and
# dispatch serves a workload at
DEFAULT_MIN_SPEEDUP = 1.05


def trace_mode() -> str:
    """AKLR_TRACE: "0" or unset traces nothing, "plugin" the plugins' routes, "op" the op
    dispatches, "all" both."""
    mode = os.environ.get(ENV_TRACE) or "0"
    if mode not in ("0", "plugin", "op", "all"):
        raise ValueError(f"{ENV_TRACE}={mode!r}: expected '0', 'plugin', 'op' or 'all'")
    return mode


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
        # AKLR_VLLM_PLUGIN: "1" sends the vLLM calls the plugin routes to AKLR
        plugin = os.environ.get(ENV_VLLM_PLUGIN) or "0"
        if plugin not in ("0", "1"):
            raise ValueError(f"{ENV_VLLM_PLUGIN}={plugin!r}: expected '1' or '0'")
        self.is_vllm_plugin_enabled = plugin == "1"
        self.trace_mode = trace_mode()
        # AKLR_FLYDSL_FASTCALL: "1" or unset launches FlyDSL kernels through a compiled function
        # per argument signature, "0" through FlyDSL's per-call checks
        fastcall = os.environ.get(ENV_FLYDSL_FASTCALL) or "1"
        if fastcall not in ("0", "1"):
            raise ValueError(f"{ENV_FLYDSL_FASTCALL}={fastcall!r}: expected '1' or '0'")
        self.is_flydsl_fastcall_enabled = fastcall == "1"
        self.export_dir = Path(os.environ.get(ENV_EXPORT_DIR) or DEFAULT_EXPORT_DIR).absolute()
        # AKLR_MIN_SPEEDUP: least speedup over aiter to serve; <= 0 also serves unmeasured
        speedup = os.environ.get(ENV_MIN_SPEEDUP) or str(DEFAULT_MIN_SPEEDUP)
        try:
            self.min_speedup = float(speedup)
        except ValueError as e:
            raise ValueError(f"{ENV_MIN_SPEEDUP}={speedup!r}: {e}") from e
        if not math.isfinite(self.min_speedup):
            raise ValueError(f"{ENV_MIN_SPEEDUP}={speedup!r}: expected a finite number")
        # AKLR_MEASURED_DIR: schemas laid out as the checkout that replace the committed
        # ones they name, e.g. the schemas/ directory a validation run writes
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
