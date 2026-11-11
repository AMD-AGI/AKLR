# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``@vllm_route``: how a vLLM call reaches AKLR ops, declared beside them.

vLLM compiles most models' forward with torch.compile, so Python there runs once, while
tracing. A route therefore replaces no traced function: it replaces a function a vLLM function
looks up when it runs, either an aiter function inside one of vLLM's opaque custom ops
(``vllm::moe_forward``, ``vllm::rocm_unquantized_gemm``) or a function of a model vLLM runs
uncompiled (``@ignore_torch_compile``). Those run at every eager call and at every CUDA graph
capture, where the route's choice is recorded."""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import version

from packaging.version import Version

from aklr.plugins._calls import wrap_calls

__all__ = ["install", "vllm_route", "vllm_version"]

_LOGGER = logging.getLogger(__name__)

_ROUTES: list[tuple[str, tuple[str, ...], tuple[str, ...], Callable]] = []


def vllm_route(caller: str, *callees: str, versions: tuple[str, ...]) -> Callable[[Callable], Callable]:
    """Declare that vLLM's ``caller`` reaches AKLR ops through this function where it calls
    ``callees`` (each as the dotted name ``caller`` looks up at run time), on the vLLM releases
    ``versions`` it was tested on; on any other, vLLM runs its own code."""

    def declare(route: Callable) -> Callable:
        _ROUTES.append((caller, callees, versions, route))
        return route

    return declare


def vllm_version() -> str:
    """The installed vLLM's release, without a local label (0.33.0rc1.dev910+g4fab5a074 is
    0.33.0rc1.dev910)."""
    return Version(version("vllm")).public


def install() -> list[str]:
    """Wrap the callees of the routes tested on the installed vLLM; returns them."""
    installed = vllm_version()
    hooked, untested = [], []
    for caller, callees, versions, route in _ROUTES:
        if installed not in versions:
            untested.append(caller)
            continue
        hooked += wrap_calls(caller, callees, route)
    # a warning, so that the line shows at any vLLM log level
    _LOGGER.warning(
        "aklr hooks %d calls of vLLM %s%s",
        len(hooked),
        installed,
        f"; not those of {', '.join(untested)}, untested on it" if untested else "",
    )
    return hooked
