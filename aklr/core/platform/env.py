###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import math
import os
from pathlib import Path

__all__ = [
    "DEFAULT_MIN_SPEEDUP",
    "ENV_BUILD_CACHE_DIR",
    "ENV_DISABLE",
    "ENV_ENABLE_TRACING",
    "ENV_MIN_SPEEDUP",
    "ENV_SCHEMA_PATH",
    "ENV_TRACE",
    "RuntimeEnv",
    "trace_mode",
]

ENV_DISABLE = "AKLR_DISABLE"
ENV_MIN_SPEEDUP = "AKLR_MIN_SPEEDUP"
ENV_BUILD_CACHE_DIR = "AKLR_BUILD_CACHE_DIR"
ENV_ENABLE_TRACING = "AKLR_ENABLE_TRACING"
ENV_SCHEMA_PATH = "AKLR_SCHEMA_PATH"
ENV_TRACE = "AKLR_TRACE"
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
    """AKLR's environment variables, parsed: "" counts as unset, a bad value raises ValueError naming it."""

    def __init__(self) -> None:
        # "1": all op types, "0": none, else a comma list of them (older AKLRs' "<op_type>@<version>" too)
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
        self.trace_mode = trace_mode()
        directory = os.environ.get(ENV_SCHEMA_PATH)
        self.schema_path = Path(directory).absolute() if directory else None
        # least speedup over the replaced baseline to serve a kernel; <= 0 also serves unmeasured ones
        speedup = os.environ.get(ENV_MIN_SPEEDUP) or str(DEFAULT_MIN_SPEEDUP)
        try:
            self.min_speedup = float(speedup)
        except ValueError as e:
            raise ValueError(f"{ENV_MIN_SPEEDUP}={speedup!r}: {e}") from e
        if not math.isfinite(self.min_speedup):
            raise ValueError(f"{ENV_MIN_SPEEDUP}={speedup!r}: expected a finite number")
        if directory := os.environ.get(ENV_BUILD_CACHE_DIR):
            self.build_cache_path = Path(directory).absolute()
        elif directory := os.environ.get("XDG_CACHE_HOME"):
            self.build_cache_path = Path(directory, "aklr", "kernels").absolute()
        else:
            # expanduser, unlike Path.home(), never raises when HOME and the password database both miss
            self.build_cache_path = Path(os.path.expanduser("~/.cache/aklr/kernels")).absolute()
