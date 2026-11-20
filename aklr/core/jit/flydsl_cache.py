# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The launch cache FlyDSL 0.5 dropped, for the JitFunctions of AKLR's FlyDSL kernels.

FlyDSL before 0.5 kept a JitFunction's call state per specialization: a call whose arguments sign as an earlier
call's launched without tracing. From 0.5 every call traces the Python function again, lowers it and hashes the
result before it launches (mla_prefill_varlen_lse's launcher took 25 ms a call, mla_decode_fwd's 320 ms, against
0.16 and 0.09 ms on FlyDSL 0.4.2); only the CompiledFunction ``flyc.compile`` returns skips that.

:func:`install` puts the cache back for the JitFunctions defined in AKLR's kernel packages (other callers of FlyDSL
in the process, aiter's kernels among them, keep FlyDSL's own path): the first call of a specialization in the
process goes through ``flyc.compile``, which traces, compiles or loads, and launches it; every later call launches
its CompiledFunction. A specialization is keyed as FlyDSL 0.4.2 keyed it: the environment and target that invalidate
an artifact, the compile hints, and per parameter a Constexpr's value, a type parameter's type, or the runtime
argument's cache signature (a tensor's dtype, alignment and layout). Like a CompiledFunction, a call does not check
the globals the function read when traced: AKLR's kernels do not change them after import."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from typing import Any

__all__ = ["install", "is_needed"]

# the packages FlyDSLBuilder imports AKLR's FlyDSL kernels as
_KERNEL_PACKAGE_PREFIX = "_aklr_flydsl_"
_lock = threading.Lock()
_original_call: Callable[..., Any] | None = None


def is_needed() -> bool:
    """Whether this FlyDSL traces every call: 0.5 and later."""
    import flydsl
    from packaging.version import Version

    return Version(flydsl.__version__).release >= (0, 5)


def install() -> bool:
    """Put the cache in front of the JitFunctions of AKLR's kernels, once a process; ``False`` on a FlyDSL that
    keeps its own."""
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
    """The CompiledFunction of each specialization of ``jit_function`` called so far, by key; ``None`` for a
    function no AKLR kernel defines."""
    try:
        return jit_function._aklr_launches
    except AttributeError:
        module = getattr(jit_function.func, "__module__", None) or ""
        launches = {} if module.startswith(_KERNEL_PACKAGE_PREFIX) else None
        jit_function._aklr_launches = launches
        return launches


def _call(jit_function, args, kwargs):
    launches = _launches(jit_function)
    # a call inside another function's trace inlines it (FlyDSL's own path does it first) and returns its value
    if launches is None or _tracing():
        return _original_call(jit_function, *args, **kwargs)
    bound_self, call_args = jit_function._split_self(args)
    if bound_self is not None:  # a method: FlyDSL's own path
        return _original_call(jit_function, *args, **kwargs)
    try:
        bound = jit_function._sig.bind(*call_args, **kwargs)
    except TypeError:  # FlyDSL's own path reports it
        return _original_call(jit_function, *args, **kwargs)
    bound.apply_defaults()
    values = tuple(bound.arguments.values())
    try:
        key = _key(jit_function, bound)
    except (TypeError, ValueError, RuntimeError):  # an argument FlyDSL cannot sign: its own path reports it
        return _original_call(jit_function, *args, **kwargs)
    compiled = launches.get(key)
    if compiled is None:
        return _compile(jit_function, launches, key, args, kwargs, values)
    return compiled(*values)


def _tracing() -> bool:
    """Whether a FlyDSL trace is under way: an MLIR context is current."""
    from flydsl.compiler.jit_function import ir

    return ir.Context.current is not None


def _compile(jit_function, launches, key, args, kwargs, values):
    """The first call of a specialization in the process: ``flyc.compile`` traces it, compiles it or loads its
    artifact, and launches it. ``flyc.compile`` launches only a call with no compile placeholder (a tensor off
    the device, or a null pointer where no device buffer is passed): any other takes FlyDSL's own path and is not
    kept."""
    from flydsl.compiler.jit_function import _compile_impl

    if not _would_launch(values):
        return _original_call(jit_function, *args, **kwargs)
    compiled = _compile_impl(jit_function, *args, **kwargs)
    if compiled is not None:  # None under FLYDSL_COMPILE_ONLY
        launches[key] = compiled


def _would_launch(values) -> bool:
    """Whether ``flyc.compile`` launches a call of these argument values, by FlyDSL's own test."""
    from flydsl.compiler import jit_function as flydsl_jit

    has_device_arguments = any(flydsl_jit._has_live_device_argument(v) for v in values)
    return not any(
        flydsl_jit._is_compile_placeholder(v, allow_null_pointers=has_device_arguments) for v in values
    )


def _key(jit_function, bound) -> tuple:
    """FlyDSL 0.4.2's cache key of a call, from FlyDSL's own signatures."""
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
