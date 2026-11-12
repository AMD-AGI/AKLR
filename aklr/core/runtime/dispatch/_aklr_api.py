###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any

from aklr.core.runtime.dispatch._adapt import _is_inplace_api
from aklr.core.runtime.dispatch._dispatcher import KernelDispatcher
from aklr.core.runtime.tracing import TracingRuntime
from aklr.core.schema import Schema, extract_axes


def aklr_api(
    schema_or_resolver: Schema | Callable[..., Schema | None],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate an op's api with its schema, or a resolver (listing its ``.schemas``) picking one per call."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = signature(fn)
        resolver = None if isinstance(schema_or_resolver, Schema) else schema_or_resolver
        schemas = (schema_or_resolver,) if resolver is None else resolver.schemas
        for template in schemas:
            Schema.declare_template(template)
        extractors = {id(s): s.const_axis_extractors() for s in schemas}
        is_inplace = _is_inplace_api(fn)
        api = f"{fn.__module__}.{fn.__qualname__}"
        warned: set[type[Exception]] = set()

        def dispatch(
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            fallback: Callable[..., Any] | None,
            baseline: str | None,
            is_inplace: bool,
        ) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            dispatcher = KernelDispatcher.get_instance()
            if dispatcher is None and fallback is None:
                raise RuntimeError(f"{fn.__name__}: no KernelDispatcher is active")
            schema = schemas[0] if resolver is None else resolver(**bound.arguments)
            if schema is None:
                if fallback is None:
                    raise RuntimeError(f"{fn.__name__}: no schema describes this call")
                return fallback(**bound.arguments)
            extractor_map = extractors.get(id(schema))
            if extractor_map is None:
                raise RuntimeError(
                    f"{fn.__name__}: {resolver.__name__} chose {schema.name!r}, not one of its schemas"
                )
            schema_name = schema.schema_name(extract_axes(extractor_map, bound.arguments))
            tracing = TracingRuntime.get_instance()
            # only calls that fall back are traced: the workloads no AKLR kernel serves
            if fallback is not None and tracing is not None:
                untraced = fallback

                def traced_fallback(*fallback_args: Any, **fallback_kwargs: Any) -> Any:
                    try:
                        tracing.collect(schema, schema_name, bound.arguments)
                    except Exception as e:  # noqa: BLE001
                        if type(e) not in warned:
                            warned.add(type(e))
                            warnings.warn(
                                f"{api}: tracing failed, {type(e).__name__}: {e}; "
                                "further ones of this type are not reported",
                                # point at the patched entry point's caller; dispatcher.dispatch adds a frame
                                stacklevel=4 if dispatcher is None else 5,
                            )
                    return untraced(*fallback_args, **fallback_kwargs)

                fallback = traced_fallback
            if dispatcher is None:
                return fallback(**bound.arguments)
            return dispatcher.dispatch(
                schema_name,
                (),
                bound.arguments,
                fallback,
                baseline,
                is_inplace,
            )

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            # the api body is a stub: with no fallback, a miss raises
            return dispatch(args, kwargs, None, None, is_inplace)

        wrapped.dispatch = dispatch
        wrapped.schemas = schemas
        return wrapped

    return decorator
