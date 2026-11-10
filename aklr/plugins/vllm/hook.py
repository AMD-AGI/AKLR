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

import functools
import inspect
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version

import torch
from packaging.version import Version

from aklr.core.runtime.dispatch import FallbackReason
from aklr.core.runtime.hook_registry import HookRegistry, HookSpec, HookType

__all__ = ["install", "original", "vllm_route", "vllm_version"]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Route:
    caller: str
    versions: tuple[str, ...]
    group: str | None


# Index declarations by callable and callee so fallback never scans the route set.
_ROUTES: dict[Callable, dict[str, _Route]] = {}


def vllm_route(
    caller: str, *callees: str, versions: tuple[str, ...], group: str | None = None
) -> Callable[[Callable], Callable]:
    """Declare that vLLM's ``caller`` reaches AKLR ops through this function where it calls
    ``callees`` (each as the dotted name ``caller`` looks up at run time), on the vLLM releases
    ``versions`` it was tested on; on any other, vLLM runs its own code.

    Routes in a ``group`` are selected together: if one is untested on the installed
    version, none is installed. The registry checks and installs the selected batch
    atomically. Only the declared callee bindings change; aliases are not propagated.
    """

    def declare(route: Callable) -> Callable:
        declaration = _Route(caller, versions, group)
        targets = _ROUTES.get(route, {})
        for callee in callees:
            if callee in targets and targets[callee] != declaration:
                raise ValueError(f"{callee}: conflicting declaration for route {route.__qualname__}")
        _ROUTES.setdefault(route, {}).update(dict.fromkeys(callees, declaration))
        return route

    return declare


def vllm_version() -> str:
    """The installed vLLM's release, without a local label (0.33.0rc1.dev910+g4fab5a074 is
    0.33.0rc1.dev910)."""
    return Version(version("vllm")).public


def install() -> list[str]:
    """Validate the selected routes, then install them in one device-ready batch."""
    installed = vllm_version()
    blocked_groups = {
        declaration.group
        for targets in _ROUTES.values()
        for declaration in targets.values()
        if declaration.group is not None and installed not in declaration.versions
    }
    hooked, untested, blocked, specs = [], [], [], []
    sources: dict[str, str] = {}
    for route, targets in _ROUTES.items():
        for callee, declaration in targets.items():
            if installed not in declaration.versions:
                untested.append(declaration.caller)
                continue
            if declaration.group in blocked_groups:
                blocked.append(f"{declaration.caller} (group={declaration.group})")
                continue
            caller = declaration.caller
            if caller not in sources:
                sources[caller] = inspect.getsource(pkgutil.resolve_name(caller))
            attr = callee.rsplit(".", 1)[-1]
            if f"{attr}(" not in sources[caller]:
                raise RuntimeError(f"{caller} does not call {callee}")
            specs.append(
                HookSpec(
                    hook_id=_hook_id(route, callee),
                    target=callee,
                    hook=_around(route),
                    hook_type=HookType.AROUND,
                    phase="device_ready",
                    group=declaration.group,
                    source=f"{route.__module__}.{route.__qualname__}",
                    exclusive=True,
                    validate=_signature_validator(route, callee),
                )
            )
            hooked.append(callee)
    registry = HookRegistry.default()
    for spec in specs:
        registry.register(spec)
    registry.apply(phase="device_ready")
    skipped = []
    if untested:
        skipped.append(f"untested callers: {', '.join(dict.fromkeys(untested))}")
    if blocked:
        skipped.append(f"blocked by untested group members: {', '.join(dict.fromkeys(blocked))}")
    # a warning, so that the line shows at any vLLM log level
    _LOGGER.warning(
        "aklr hooks %d calls of vLLM %s%s",
        len(hooked),
        installed,
        f"; {'; '.join(skipped)}" if skipped else "",
    )
    return hooked


def _parameters(fn: Callable) -> list[tuple[str, inspect._ParameterKind]]:
    # Defaults can contain framework enums unavailable when routes are declared.
    return [(p.name, p.kind) for p in inspect.signature(fn).parameters.values()]


def _hook_id(route: Callable, callee: str) -> str:
    return f"vllm.route:{route.__module__}.{route.__qualname__}:{callee}"


@functools.cache
def _signature_validator(route: Callable, callee: str) -> Callable:
    def validate(orig: Callable) -> None:
        # The Registry supplies the exact callable that its wrapper will invoke,
        # including cls for classmethods and the bound receiver for singletons.
        if _parameters(route) != _parameters(orig):
            raise RuntimeError(f"{callee}: route {route.__qualname__} has incompatible parameters")

    return validate


def original(route: Callable, callee: str | None = None) -> Callable:
    """The installed route's saved callable, independent of the current wrapper chain.

    Specify ``callee`` for routes with multiple targets. Unlike ``next_fn`` in an
    AROUND callback, this is the implementation saved before any hooks in the batch.
    """
    try:
        targets = _ROUTES[route]
    except KeyError:
        raise ValueError(f"{route.__qualname__}: route is not declared") from None
    if callee is None:
        if len(targets) != 1:
            raise ValueError(f"{route.__qualname__}: select one of its declared callees")
        callee = next(iter(targets))
    elif callee not in targets:
        raise ValueError(f"{route.__qualname__}: undeclared callee {callee}")
    # Keep the Registry lookup: restore invalidates handles there, and a later
    # installation may legitimately save a different native implementation.
    return HookRegistry.default().original(_hook_id(route, callee))


@functools.cache
def _around(route: Callable) -> Callable:
    def around(orig, *args, **kwargs):
        # Dynamo would trace AKLR's table lookup and recompile for every new shape
        if not torch.compiler.is_compiling():
            result = route(*args, **kwargs)
            if not isinstance(result, FallbackReason):
                return result
        return orig(*args, **kwargs)

    return around
