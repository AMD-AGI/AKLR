###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


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
    try:
        annotation = inspect.signature(fn).return_annotation
    except (TypeError, ValueError):
        return False
    return annotation is None or annotation is type(None) or annotation == "None"


class OpVariants:
    _ops: ClassVar[dict[str, OpVariants]] = {}

    @classmethod
    def ops(cls) -> dict[str, OpVariants]:
        import aklr.ops

        for package in pkgutil.iter_modules(aklr.ops.__path__):
            if package.ispkg:
                importlib.import_module(f"aklr.ops.{package.name}.api")
        return cls._ops

    @classmethod
    def load(cls, text: str | bytes) -> Schema:
        schema = TypeAdapter(Schema).validate_json(text)
        api = cls.ops().get(schema.op_type)
        variant = None if api is None else api.variant_of(schema)
        return schema if variant is None else replace(schema, standard=variant.standard)

    def __init__(self, fn: Callable[..., Any], variants: tuple[Schema, ...]) -> None:
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
        return next(
            (
                v
                for v in self.variants
                if v.op_name == schema.op_name and v.schema_name(schema.const_axes) == schema.name
            ),
            None,
        )

    def resolve(self, call: dict[str, Any]) -> Schema | None:
        return next((schema for schema in self.variants if schema.matches(call)), None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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
                            stacklevel=4 if dispatcher is None else 5,
                        )
                return untraced()

        if dispatcher is None:
            return fallback()
        return dispatcher.call(schema, call, fallback=fallback, baseline=baseline, is_inplace=is_inplace)


def aklr_api(variants: Schema | tuple[Schema, ...]) -> Callable[[Callable[..., Any]], OpVariants]:
    return lambda fn: OpVariants(fn, (variants,) if isinstance(variants, Schema) else variants)
