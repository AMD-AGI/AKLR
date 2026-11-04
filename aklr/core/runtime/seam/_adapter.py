###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable, Mapping
from typing import Any

import torch

from aklr.core.runtime.dispatch import KernelDispatcher
from aklr.core.runtime.dispatch._adapt import _is_inplace_api
from aklr.core.runtime.tracing import TracingRuntime

__all__ = ["Adapter"]


class Adapter:
    """Serves a backend entry point through an AKLR api: ``bind_fn`` maps its call, None to decline."""

    def __init__(
        self,
        entry: str,
        api: Callable[..., Any],
        bind_fn: Callable[..., dict[str, Any] | None],
        baseline_fn: Callable[..., Callable[[], Any] | None] | None,
    ) -> None:
        module, _, name = entry.rpartition(".")
        if not module or not name:
            raise ValueError(f"not a dotted entry point: {entry!r}")
        self.entry = entry
        self.api = api
        # the entry point's top-level package: the baseline dispatch must beat
        self.backend = module.partition(".")[0]
        self.bind_fn = bind_fn
        self.baseline_fn = baseline_fn

    def __repr__(self) -> str:
        return f"<Adapter {self.entry} -> {self.api.__name__}>"

    def original(self) -> Callable | None:
        """The entry point's unpatched function, or None when this environment lacks it."""
        module, _, name = self.entry.rpartition(".")
        try:
            entry = getattr(importlib.import_module(module), name, None)
        except ModuleNotFoundError as e:
            # only the entry's own module (or a parent) missing means this environment lacks it
            if e.name is None or not f"{module}.".startswith(f"{e.name}."):
                raise
            return None
        if entry is None:
            return None
        return entry.__wrapped__ if getattr(entry, "aklr_adapter", None) is self else entry

    def baseline(self, call: Mapping[str, Any]) -> Callable[[], Any] | None:
        """A no-argument call of the backend's original on api kwargs ``call``, for timing; else None."""
        if self.baseline_fn is None:
            return None
        original = self.original()
        return None if original is None else self.baseline_fn(original, **call)

    def wrap(self, original: Callable) -> Callable:
        """The stand-in for ``original``: dispatch to AKLR, else run ``original``."""
        is_inplace = _is_inplace_api(original)

        @functools.wraps(original)
        def stand_in(*args: Any, **kwargs: Any) -> Any:
            # Compiling: dispatch reads data_ptr and keys on each call's sizes, so the graph keeps the backend.
            if torch.compiler.is_compiling() or (
                KernelDispatcher.get_instance() is None and TracingRuntime.get_instance() is None
            ):
                return original(*args, **kwargs)
            schema_kwargs = self.bind_fn(*args, **kwargs)
            if schema_kwargs is None:
                return original(*args, **kwargs)
            return self.api.dispatch(
                (),
                schema_kwargs,
                lambda *_, **__: original(*args, **kwargs),
                self.backend,
                is_inplace,
            )

        stand_in.aklr_adapter = self  # type: ignore[attr-defined]
        return stand_in
