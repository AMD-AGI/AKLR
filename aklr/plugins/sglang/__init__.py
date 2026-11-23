# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR plugin for sglang, loaded through the ``sglang.srt.plugins`` entry point."""

from __future__ import annotations

import importlib
from threading import RLock

from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.dispatch import register_sglang_plugin
from aklr.plugins import Plugin, _declare_families
from aklr.plugins.sglang import hook

__all__ = ["plugin", "register"]

plugin = Plugin("sglang")
_MODULES = (
    "layers.aiter_backend_mla_prefill",
    "layers.attn_residual.plugin",
    "layers.dsa_backend.plugin",
    "layers.dsa_indexer.plugin",
    "layers.kda_backend_hip.plugin",
    "layers.logits_processor_torch_gemm",
    "layers.mla_decode.plugin",
    "models.deepseek_common.plugin",
    "models.deepseek_v2.plugin",
    "models.dspark_torch_gemm",
    "models.forward_mla_rocm.plugin",
    "models.kimi_k3.plugin",
    "models.kimi_k3_torch_gemm",
    "models.minimax_m2.plugin",
    "models.qwen2_moe.plugin",
    "models.qwen3_5.plugin",
    "models.qwen3_moe.plugin",
    "models.qwen3_next.plugin",
)


def _build():
    return plugin.build(
        _declare_families, *(importlib.import_module(f"{__name__}.{name}").declare for name in _MODULES)
    )


def register() -> None:
    """Entry point sglang calls once per process; a plugin that cannot start ends the process,
    which sglang would only log."""
    try:
        with _lock:
            _start()
    except Exception as e:
        raise SystemExit(f"aklr: the sglang plugin cannot start: {e!r}") from e


_lock = RLock()
_bootstrapped = False


def _start() -> None:
    global _bootstrapped
    env = RuntimeEnv()
    enabled = env.is_sglang_plugin_enabled and not env.is_all_disabled
    if enabled:
        _build()
    if not _bootstrapped:
        register_sglang_plugin()
        _bootstrapped = True
    if env.is_aiter_plugin_enabled and not env.is_all_disabled:
        from aklr.plugins.aiter.provider import enable

        enable("AKLR sglang plugin, AKLR_AITER_PLUGIN=1")
    if enabled:
        hook.install()
