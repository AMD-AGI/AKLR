# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR as a kernel plugin of aiter (``aiter.plugin``, RFC ROCm/aiter#7240).

Before each call of an op aiter routes, aiter asks the enabled providers whether they serve it. This
one answers from AKLR's dispatch table without running anything, and declines whatever AKLR has no
qualified kernel for, so aiter runs its own. aiter finds it through the ``aiter.plugins`` entry point
but asks it only once it is enabled: :func:`enable` does that, called by the sglang and vLLM plugins
under ``AKLR_AITER_PLUGIN=1`` and by any other host itself.

Each operator family's plugin module declares the AITER ops and ABI versions it serves."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from aklr.plugins import _declare_families
from aklr.plugins.aiter import plugin as declarations
from aklr.plugins.aiter._candidate import NAME, declining, is_declining, ready
from aklr.plugins.aiter.fused_moe import plugin as moe
from aklr.plugins.aiter.gemm import plugin as gemm
from aklr.plugins.aiter.kda_conv import plugin as conv
from aklr.plugins.aiter.mhc import plugin as mhc
from aklr.plugins.aiter.mla import plugin as mla
from aklr.plugins.aiter.mla_prefill import plugin as mla_prefill
from aklr.plugins.aiter.quant import plugin as quant
from aklr.plugins.aiter.rms_norm import plugin as norm
from aklr.plugins.aiter.topk import plugin as topk

__all__ = ["NAME", "Provider", "declining", "enable", "provider"]

_LOGGER = logging.getLogger("aklr.plugins.aiter")


class Provider:
    """The provider aiter asks: ``select`` per call of an op in ``op_versions``."""

    name = NAME
    # the aiter.plugin.API_VERSION this provider speaks
    api_versions = frozenset({1})

    def __init__(self) -> None:
        declarations.build(
            _declare_families,
            conv.declare,
            gemm.declare,
            mhc.declare,
            mla.declare,
            mla_prefill.declare,
            moe.declare,
            norm.declare,
            quant.declare,
            topk.declare,
        )
        self._selectors = MappingProxyType(declarations._selectors)
        versions: dict[str, set[int]] = {}
        for op, version in self._selectors:
            versions.setdefault(op, set()).add(version)
        self.op_versions = MappingProxyType({op: frozenset(v) for op, v in versions.items()})

    def select(self, op: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        """AKLR's candidate for this call, or ``None`` for aiter to run its own kernel."""
        entry = self._selectors.get((op.name, op.version))
        if entry is None or is_declining() or not ready():
            return None
        return entry(args, kwargs)


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
    status = plugin.status()
    enabled = [name for name in status["enabled"] if name != NAME]
    plugin.configure(source=source, enable=[NAME, *enabled], exclude=status.get("excluded", ()))
    return True
