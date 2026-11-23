# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR as a kernel plugin of aiter (``aiter.plugin``, RFC ROCm/aiter#7240).

Before each call of an op aiter routes, aiter asks the enabled providers whether they serve it. This
one answers from AKLR's dispatch table without running anything, and declines whatever AKLR has no
qualified kernel for, so aiter runs its own. aiter finds it through the ``aiter.plugins`` entry point
but asks it only once it is enabled: :func:`enable` does that, called by the sglang and vLLM plugins
under ``AKLR_AITER_PLUGIN=1`` and by any other host itself.

One module of this package per op family registers the selectors of the aiter ops it serves."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Mapping
from typing import Any

from aklr.plugins.aiter.provider._base import NAME, SELECTORS, declining, is_declining, ready

__all__ = ["NAME", "Provider", "declining", "enable", "provider"]

_LOGGER = logging.getLogger("aklr.plugins.aiter")

for _module in pkgutil.iter_modules(__path__):
    if not _module.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module.name}")


class Provider:
    """The provider aiter asks: ``select`` per call of an op in ``op_versions``."""

    name = NAME
    # the aiter.plugin.API_VERSION this provider speaks
    api_versions = frozenset({1})
    op_versions = {op: frozenset({version}) for op, (version, _) in SELECTORS.items()}

    def select(self, op: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        """AKLR's candidate for this call, or ``None`` for aiter to run its own kernel."""
        entry = SELECTORS.get(op.name)
        if entry is None or is_declining() or not ready():
            return None
        return entry[1](args, kwargs)


provider = Provider()


def enable(source: str = "AKLR") -> bool:
    """Have aiter ask AKLR first for the ops it routes, keeping the providers already enabled after
    it; ``False`` when this aiter has no plugin interface. ``source`` is the provenance aiter records
    with every routing decision."""
    try:
        from aiter import plugin
    except ImportError:
        _LOGGER.info("aklr: this aiter has no plugin interface (aiter.plugin); its ops stay aiter's")
        return False
    enabled = [name for name in plugin.status()["enabled"] if name != NAME]
    plugin.configure(source=source, enable=[NAME, *enabled])
    return True
