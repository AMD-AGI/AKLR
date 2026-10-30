# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR plugin for sglang, loaded through the ``sglang.srt.plugins`` entry point."""

from __future__ import annotations

import importlib

from aklr.core.contract.locate import op_packages
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime import peer_memory
from aklr.core.runtime.dispatch import register_sglang_plugin
from aklr.plugins.sglang.hook import install

__all__ = ["register"]


def _tp_group():
    """sglang's TP group, which the kernels with peer memory sum over: its host (gloo) group,
    as ``peer_memory.set_group`` takes it."""
    from sglang.srt.distributed import get_tp_group

    return get_tp_group().cpu_group


def register() -> None:
    """Entry point sglang calls once per process: AKLR starts once torch.distributed is
    up; unless AKLR_DISABLE=1, aiter's entry points are proxied and, with
    AKLR_SGLANG_PLUGIN=1, the layer routes hook and kernels with peer memory span the
    TP group."""
    register_sglang_plugin()
    env = RuntimeEnv()
    if env.is_all_disabled:
        return
    from aklr.ops import load_routes  # aklr.ops imports this package's contracts

    load_routes()
    if not env.is_sglang_plugin_enabled:
        return
    for package in op_packages():
        if package.startswith(f"{__name__}."):
            importlib.import_module(f"{package}.api")
    install()
    peer_memory.set_group(_tp_group)
