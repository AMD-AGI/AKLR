# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR plugin for sglang, loaded through the ``sglang.srt.plugins`` entry point."""

from __future__ import annotations

from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.dispatch import register_sglang_plugin
from aklr.plugins.sglang import hook

__all__ = ["register"]


def register() -> None:
    """Entry point sglang calls once per process; a plugin that cannot start ends the process,
    which sglang would only log."""
    try:
        _register()
    except Exception as e:
        raise SystemExit(f"aklr: the sglang plugin cannot start: {e!r}") from e


def _register() -> None:
    register_sglang_plugin()
    env = RuntimeEnv()
    if env.is_all_disabled:
        return
    from aklr.ops import load_routes  # aklr.ops imports this package's contracts

    # imports every op's api.py, the plugin's routes among them
    load_routes()
    if env.is_sglang_plugin_enabled:
        hook.install()
