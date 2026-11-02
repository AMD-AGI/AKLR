# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR plugin for sglang, loaded through the ``sglang.srt.plugins`` entry point."""

from __future__ import annotations

import importlib
import logging
import os
import sys

from aklr.core.contract.locate import op_packages
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime import peer_memory
from aklr.core.runtime.dispatch import register_sglang_plugin
from aklr.plugins.sglang import hook
from aklr.plugins.sglang.hook import install

__all__ = ["BACKENDS", "register"]

_LOGGER = logging.getLogger(__name__)

ENV_LOG_LEVEL = "AKLR_LOG_LEVEL"
# Packages that serve a whole model through an sglang extension point of their own
# (not op packages: no contract.py); each has register() -> None.
BACKENDS = ("aklr.plugins.sglang.models.deepseek_v4",)


def _tp_group():
    """sglang's TP group, which the kernels with peer memory sum over: its host (gloo) group,
    as ``peer_memory.set_group`` takes it."""
    from sglang.srt.distributed import get_tp_group

    return get_tp_group().cpu_group


def _attach_log_handler() -> None:
    """AKLR's own ``[aklr]`` lines, visible at sglang's default log level (sglang's
    handlers serve its own loggers only); AKLR_LOG_LEVEL overrides INFO."""
    logger = logging.getLogger("aklr")
    if any(getattr(h, "_aklr", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler._aklr = True
    handler.setFormatter(
        logging.Formatter("[aklr %(asctime)s pid %(process)d] %(levelname)s %(message)s",
                          "%Y-%m-%d %H:%M:%S")
    )  # fmt: skip
    logger.addHandler(handler)
    logger.setLevel(os.environ.get(ENV_LOG_LEVEL, "INFO").upper())
    logger.propagate = False


def register() -> None:
    """Entry point sglang calls once per process: AKLR starts once torch.distributed is
    up; unless AKLR_DISABLE=1, aiter's entry points are proxied and, with
    AKLR_SGLANG_PLUGIN=1, the layer routes hook, the model backends register and kernels
    with peer memory span the TP group. A backend that fails to register leaves its
    models to sglang, with one warning."""
    register_sglang_plugin()
    env = RuntimeEnv()
    if env.is_all_disabled:
        return
    from aklr.ops import load_routes  # aklr.ops imports this package's contracts

    # a server keeps the image's aiter where the routes describe another version
    skipped = load_routes(strict=False)
    if skipped:
        _LOGGER.warning(
            "%d aiter entry points not proxied, the routes describe another aiter: %s",
            len(skipped),
            ", ".join(skipped),
        )
    if not env.is_sglang_plugin_enabled:
        return
    if env.is_package_disabled(f"{__name__}._"):
        return  # AKLR_DISABLE=aklr.plugins
    _attach_log_handler()
    for package in op_packages():
        if package.startswith(f"{__name__}."):
            importlib.import_module(f"{package}.api")
    for package in BACKENDS:
        declared = len(hook._ROUTES)
        try:
            importlib.import_module(package).register()
        except Exception as e:  # noqa: BLE001
            # a backend declares its routes before it registers with sglang: drop them
            del hook._ROUTES[declared:]
            _LOGGER.warning(
                "%s not registered (%s: %s); sglang runs its models", package,
                type(e).__name__, e,
            )  # fmt: skip
    install()
    peer_memory.set_group(_tp_group)
