###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""An op's api: the function ops and seam adapters call."""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import inspect
import pkgutil
import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar

from pydantic import TypeAdapter

from aklr.core.runtime.dispatch._dispatcher import Dispatcher
from aklr.core.runtime.tracing import TracingRuntime
from aklr.core.schema import Schema, extract_axes


def _is_inplace_api(fn: Callable) -> bool:
    """Whether the api returns None, i.e. writes its results into its arguments."""
    try:
        annotation = inspect.signature(fn).return_annotation
    except (TypeError, ValueError):
        return False
    return annotation is None or annotation is type(None) or annotation == "None"


class OpVariants:
    """An op's api: called like the function it decorates, it resolves the call's variant among its schemas and
    dispatches it."""

    _ops: ClassVar[dict[str, OpVariants]] = {}  # by op type, each put in by its __init__

    @classmethod
    def ops(cls) -> dict[str, OpVariants]:
        """The apis declared in code, by op type: those of every op (aklr/ops/<op>/api.py), and any declared since."""
        import aklr.ops

        for package in pkgutil.iter_modules(aklr.ops.__path__):
            if package.ispkg:
                importlib.import_module(f"aklr.ops.{package.name}.api")  # its aklr_api declares its apis
        return cls._ops

    @classmethod
    def load(cls, text: str | bytes) -> Schema:
        """The schema of a json, with the standard of its variant: the json leaves it out. None when no variant
        is its."""
        schema = TypeAdapter(Schema).validate_json(text)
        api = cls.ops().get(schema.op_type)
        variant = None if api is None else api.variant_of(schema)
        return schema if variant is None else replace(schema, standard=variant.standard)

    def __init__(self, fn: Callable[..., Any], variants: tuple[Schema, ...]) -> None:
        """Declare fn the api of variants, in resolution order; fn gives the signature and is never run. A second
        api of one op type raises."""
        self._signature = inspect.signature(fn)
        if len({s.op_type for s in variants}) != 1:
            raise ValueError(f"{fn.__name__}: its variants are of op types {[s.op_type for s in variants]}")
        if len({s.variant_name for s in variants}) != len(variants):
            raise ValueError(f"{fn.__name__}: two variants are named alike, {[s.op_name for s in variants]}")
        for schema in variants:
            for condition in schema.variant.values():
                unknown = {
                    node.id
                    for node in ast.walk(ast.parse(condition, mode="eval"))
                    if isinstance(node, ast.Name)
                } - {*self._signature.parameters, "torch", *vars(builtins)}
                if unknown:
                    raise ValueError(
                        f"{schema.op_name}: {sorted(unknown)} are not arguments of {fn.__name__}"
                    )
        functools.update_wrapper(self, fn)
        self.variants = variants
        self._is_inplace = _is_inplace_api(fn)
        self._warned: set[type[Exception]] = set()
        op_type = variants[0].op_type
        if op_type in OpVariants._ops:
            raise ValueError(f"{fn.__name__}: {op_type} already has an api")
        OpVariants._ops[op_type] = self

    def variant_of(self, schema: Schema) -> Schema | None:
        """The variant schema is an instance of: of its op name, named schema's name at its Const axes; None
        when none is."""
        return next(
            (
                v
                for v in self.variants
                if v.op_name == schema.op_name and v.schema_name(schema.const_axes) == schema.name
            ),
            None,
        )

    def resolve(self, call: dict[str, Any]) -> Schema | None:
        """The first variant whose condition matches call, the api's bound arguments; None when none does."""
        return next((schema for schema in self.variants if schema.matches(call)), None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # a direct call has no fallback: a miss raises
        return self.dispatch(
            self._signature.bind_partial(*args, **kwargs).arguments, is_inplace=self._is_inplace
        )

    def dispatch(
        self,
        call: dict[str, Any],
        *,
        fallback: Callable[[], Any] | None = None,
        baseline: str | None = None,
        is_inplace: bool = False,
    ) -> Any:
        """Run call, the api's arguments by name, on a kernel of its variant that beats baseline; with none,
        return fallback(), recorded as a workload when tracing, or raise when there is no fallback."""
        bound = self._signature.bind_partial(**call)
        bound.apply_defaults()
        call = bound.arguments
        dispatcher = Dispatcher.get_instance()
        if dispatcher is None and fallback is None:
            raise RuntimeError(f"{self.__name__}: no Dispatcher is active")
        schema = self.resolve(call)
        if schema is None:
            if fallback is None:
                raise RuntimeError(f"{self.__name__}: no variant takes this call")
            return fallback()
        tracing = TracingRuntime.get_instance()
        # only calls that fall back are traced: the workloads no AKLR kernel serves
        if fallback is not None and tracing is not None:
            untraced = fallback

            def fallback() -> Any:
                try:
                    name = schema.schema_name(extract_axes(schema.const_axis_extractors, call))
                    tracing.collect(schema, name, call)
                except Exception as e:  # noqa: BLE001
                    if type(e) not in self._warned:
                        self._warned.add(type(e))
                        warnings.warn(
                            f"{self.__module__}.{self.__qualname__}: tracing failed, {type(e).__name__}: {e}; "
                            "further ones of this type are not reported",
                            # point at the patched entry point's caller; Dispatcher.call adds a frame
                            stacklevel=4 if dispatcher is None else 5,
                        )
                return untraced()

        if dispatcher is None:
            return fallback()
        return dispatcher.call(schema, call, fallback=fallback, baseline=baseline, is_inplace=is_inplace)


def aklr_api(variants: Schema | tuple[Schema, ...]) -> Callable[[Callable[..., Any]], OpVariants]:
    """Decorate an op's api with its schema, or with its variants in resolution order."""
    return lambda fn: OpVariants(fn, (variants,) if isinstance(variants, Schema) else variants)
