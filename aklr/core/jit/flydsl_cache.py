# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from typing import Any

__all__ = ["install", "is_needed"]

_KERNEL_PACKAGE_PREFIX = "flydsl_"
_lock = threading.Lock()
_original_call: Callable[..., Any] | None = None


def is_needed() -> bool:
    import flydsl
    from packaging.version import Version

    return Version(flydsl.__version__).release >= (0, 5)


def install() -> bool:
    global _original_call
    with _lock:
        if _original_call is not None:
            return True
        if not is_needed():
            return False
        from flydsl.compiler.jit_function import JitFunction

        original = JitFunction.__call__

        def __call__(self, *args, **kwargs):
            return _call(self, args, kwargs)

        __call__.__wrapped__ = original
        _original_call = original
        JitFunction.__call__ = __call__
        return True


def _launches(jit_function) -> dict | None:
    try:
        return jit_function._aklr_launches
    except AttributeError:
        module = getattr(jit_function.func, "__module__", None) or ""
        launches = {} if module.startswith(_KERNEL_PACKAGE_PREFIX) else None
        jit_function._aklr_launches = launches
        return launches


def _call(jit_function, args, kwargs):
    launches = _launches(jit_function)
    if launches is None or _tracing():
        return _original_call(jit_function, *args, **kwargs)
    bound_self, call_args = jit_function._split_self(args)
    if bound_self is not None:
        return _original_call(jit_function, *args, **kwargs)
    try:
        bound = jit_function._sig.bind(*call_args, **kwargs)
    except TypeError:
        return _original_call(jit_function, *args, **kwargs)
    bound.apply_defaults()
    values = tuple(bound.arguments.values())
    try:
        key = _key(jit_function, bound)
    except (TypeError, ValueError, RuntimeError):
        return _original_call(jit_function, *args, **kwargs)
    compiled = launches.get(key)
    if compiled is None:
        return _compile(jit_function, launches, key, args, kwargs, values)
    return compiled(*values)


def _tracing() -> bool:
    from flydsl.compiler.jit_function import ir

    return ir.Context.current is not None


def _compile(jit_function, launches, key, args, kwargs, values):
    from flydsl.compiler.jit_function import _compile_impl

    if not _would_launch(values):
        return _original_call(jit_function, *args, **kwargs)
    compiled = _compile_impl(jit_function, *args, **kwargs)
    if compiled is not None:
        launches[key] = compiled


def _would_launch(values) -> bool:
    from flydsl.compiler import jit_function as flydsl_jit

    has_device_arguments = any(flydsl_jit._has_live_device_argument(v) for v in values)
    return not any(
        flydsl_jit._is_compile_placeholder(v, allow_null_pointers=has_device_arguments) for v in values
    )


def _key(jit_function, bound) -> tuple:
    from flydsl.compiler import jit_function as flydsl_jit
    from flydsl.compiler.protocol import cache_signature

    target = flydsl_jit.get_backend().target
    hints = jit_function._effective_compile_hints()
    parts: list[Any] = [
        flydsl_jit._cache_invalidating_env_values(target),
        target,
        tuple(sorted((name, type(value).__qualname__, repr(value)) for name, value in hints.items())),
    ]
    parameters = jit_function._sig.parameters
    for name, value in bound.arguments.items():
        annotation = parameters[name].annotation
        if annotation is not inspect.Parameter.empty:
            if flydsl_jit.Constexpr.is_constexpr_annotation(annotation):
                parts.append((name, flydsl_jit.Constexpr.value_signature(value)))
                continue
            if flydsl_jit.is_type_param_annotation(annotation):
                if getattr(value, "__dsl_member_behavior__", False):
                    value = (value, value.__cache_signature__())
                parts.append((name, value))
                continue
        if not isinstance(value, flydsl_jit.JitArgument):
            if isinstance(annotation, type) and issubclass(annotation, flydsl_jit.JitArgument):
                value = annotation(value)
            else:
                constructor, _ = flydsl_jit.JitArgumentRegistry.get(type(value))
                if constructor is None:
                    raise TypeError(f"{name}: {type(value).__name__} has no FlyDSL argument")
                value = constructor(value)
        parts.append((name, cache_signature(value)))
    return tuple(parts)
