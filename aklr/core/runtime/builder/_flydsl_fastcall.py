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
FlyDSL's cache key derives from an argument and more: a tensor's dtype, shape, stride and
device, a scalar's type and value; a stream, FlyDSL's or torch's, stands for its kind. A call
with an argument of another kind or a non-strided tensor, of a method, inside a trace, or under
``FLYDSL_COMPILE_ONLY`` takes FlyDSL's own path.

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
_installed = False


def _arg_key(value: Any, stream_types: tuple[type, ...]) -> Any:
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            return _MISS
        return (value.dtype, tuple(value.shape), value.stride(), value.device.index)
    if isinstance(value, stream_types):
        return type(value)
    if value is None or type(value) in (bool, int, float, str):
        return (type(value), value)
    return _MISS


def _trailing(func: Any, n_args: int) -> tuple[list[str], dict[str, Any]] | None:
    """The parameters after the first ``n_args`` and their defaults; ``None`` for a method or
    a signature ``CompiledFunction``'s positional call cannot fill."""
    params = list(inspect.signature(func).parameters.values())
    if params and params[0].name == "self":
        return None
    if any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params):
        return None
    rest = params[n_args:]
    return [p.name for p in rest], {p.name: p.default for p in rest}


def install(package_prefix: str) -> None:
    """Patch ``flydsl``'s ``JitFunction.__call__`` once per process; a no-op without flydsl."""
    global _installed
    with _lock:
        if _installed:
            return
        try:
            import flydsl.compiler as flyc
            import flydsl.expr as fx
            from flydsl._mlir import ir
            from flydsl.compiler.jit_function import JitFunction as jit_cls
            from flydsl.utils import env as flydsl_env
        except ImportError:
            return
        compile_env = flydsl_env.compile
        original = jit_cls.__call__
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
            if entries is False or ir.Context.current is not None or compile_env.compile_only:
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
            trailing = _trailing(self.func, len(args))
            if trailing is None:
                return original(self, *args, **kwargs)
            # the first call of this signature launches through FlyDSL, which compiles it
            compiled = compile_fn(self, *args, **kwargs)
            if compiled is not None:
                names, defaults = trailing
                entries[key] = (compiled, names, defaults)
            return None

        jit_cls.__call__ = fast_call
        _installed = True
