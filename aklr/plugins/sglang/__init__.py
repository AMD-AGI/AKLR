# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR plugin for sglang, loaded through the ``sglang.srt.plugins`` entry point."""

from __future__ import annotations

import importlib

from aklr.core.contract.locate import op_packages
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
    if env.is_all_disabled or not env.is_sglang_plugin_enabled:
        return
    # import the modules that declare routes: the layer routes, then each plugin op's api
    importlib.import_module(f"{__name__}.layers")
    for package in op_packages():
        if package.startswith(f"{__name__}."):
            importlib.import_module(f"{package}.api")
    hook.install()
