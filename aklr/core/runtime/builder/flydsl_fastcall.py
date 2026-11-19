# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A host fast path for the ``@flyc.jit`` launchers of AKLR's FlyDSL kernels.

Each call of a ``@flyc.jit`` function rebuilds FlyDSL's cache key (argument binding, a
snapshot of every global the function reads, the env vars that invalidate it, a JitArgument
per argument): about 40 us of host time per launch on MI355X hosts, against about 8 us for
the ``flyc.compile()``d function of the same specialization. Eager steps launch AKLR's
kernels hundreds of times each.

The patched ``JitFunction.__call__`` keeps one ``CompiledFunction`` per launcher and argument
signature, for launchers defined in AKLR's kernel packages only. The signature holds what
FlyDSL's cache key derives from an argument and more: a GPU tensor's dtype, shape, stride and
device, a scalar's type and value; a stream, FlyDSL's or torch's, stands for its kind. Keyword
arguments and defaults are passed to the compiled function by position, as it takes them. A
call with an argument of another kind, of a method or a launcher with keyword-only or variadic
parameters, inside a trace, or under ``FLYDSL_COMPILE_ONLY`` takes FlyDSL's own path.

What a hit skips is what AKLR's kernels never change while serving: the globals a launcher
reads (constant tables), the ``FLYDSL_*`` env vars and compile hints. A process that changes
them after a launch sets ``AKLR_FLYDSL_FASTCALL=0``."""

from __future__ import annotations

import inspect
import threading
from typing import Any

import torch

__all__ = ["install"]

_MISS = object()
_lock = threading.Lock()
# set while flyc.compile runs: FlyDSL 0.3.2's compile launches through JitFunction.__call__
_compiling = threading.local()


def _arg_key(value: Any, stream_types: tuple[type, ...]) -> Any:
    if isinstance(value, torch.Tensor):
        # flyc.compile launches with GPU tensors only; it compiles any other as a placeholder
        if value.layout is not torch.strided or value.device.type != "cuda":
            return _MISS
        return (value.dtype, tuple(value.shape), value.stride(), value.device.index)
    if isinstance(value, stream_types):
        return type(value)
    if value is None or type(value) in (bool, int, float, str):
        return (type(value), value)
    return _MISS


def _signature(func: Any) -> inspect.Signature | None:
    """The launcher's signature; ``None`` for a method or a signature ``CompiledFunction``'s
    positional call cannot fill."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        return None
    if any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params):
        return None
    return sig


def install(package_prefix: str) -> None:
    """Patch ``flydsl``'s ``JitFunction.__call__`` once per process; a no-op without flydsl."""
    with _lock:
        try:
            import flydsl.compiler as flyc
            import flydsl.expr as fx
            from flydsl._mlir import ir
            from flydsl.compiler.jit_function import CompiledFunction as compiled_cls
            from flydsl.compiler.jit_function import JitFunction as jit_cls
            from flydsl.utils import env as flydsl_env
        except ImportError:
            return
        original = jit_cls.__call__
        if getattr(original, "_aklr_fastcall", False):
            return
        compile_env = flydsl_env.compile
        compile_fn = flyc.compile
        # a stream keys by its kind, FlyDSL's or torch's, which the compiled function converts
        stream_types = (fx.Stream, torch.Stream)

        def fast_call(self, *args, **kwargs):
            entries = self.__dict__.get("_aklr_compiled")
            if entries is None:
                ours = getattr(self.func, "__module__", "").startswith(package_prefix)
                entries = {} if ours else False
                self._aklr_compiled = entries
            # a compiled function launches even under FLYDSL_COMPILE_ONLY, which must not
            if (
                entries is False
                or getattr(_compiling, "active", False)
                or ir.Context.current is not None
                or compile_env.compile_only
            ):
                return original(self, *args, **kwargs)
            key = []
            for value in args:
                k = _arg_key(value, stream_types)
                if k is _MISS:
                    return original(self, *args, **kwargs)
                key.append(k)
            for name in sorted(kwargs):
                k = _arg_key(kwargs[name], stream_types)
                if k is _MISS:
                    return original(self, *args, **kwargs)
                key.append((name, k))
            key = tuple(key)
            entry = entries.get(key)
            if entry is not None:
                compiled, names, defaults = entry
                if not names:
                    return compiled(*args)
                return compiled(*args, *(kwargs[n] if n in kwargs else defaults[n] for n in names))
            sig = _signature(self.func)
            if sig is None:
                return original(self, *args, **kwargs)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            positional = tuple(bound.arguments.values())
            # flyc.compile with no arguments returns the launcher without launching it
            if not positional:
                return original(self, *args, **kwargs)
            # the first call of a signature launches through flyc.compile, which compiles it
            _compiling.active = True
            try:
                compiled = compile_fn(self, *positional)
            finally:
                _compiling.active = False
            if isinstance(compiled, compiled_cls):
                names = list(sig.parameters)[len(args) :]
                entries[key] = (compiled, names, {n: sig.parameters[n].default for n in names})
            return None

        fast_call._aklr_fastcall = True
        jit_cls.__call__ = fast_call
