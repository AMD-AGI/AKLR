# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The argument specs of an op contract: tensors, their dtypes and scalars."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, model_validator

from aklr.core.contract import expr
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DeterminismLevel,
    Distribution,
    DType,
    ElementOrder,
    ErrorBound,
)
from aklr.core.contract.semantics._compare import ErrorBoundUnion
from aklr.core.contract.semantics._distributions import DistributionUnion
from aklr.core.contract.semantics._frozen_model import FrozenModel
from aklr.core.contract.spec._points import case_key


class TensorLifetime(str, Enum):
    """Whether a tensor's values may change from one call to the next."""

    PER_CALL = "per_call"
    WEIGHT = "weight"


# Shapes are in logical elements ; stored shape is ``semantics.dtype_storage_shape``.
class ShapeByAttrSpec(FrozenModel):
    """A ``TensorSpec`` shape chosen by an attr; ``cases`` is keyed by ``case_key``."""

    attr_name: str
    cases: dict[str, tuple[str | int, ...]]


class DTypeSpec(FrozenModel):
    # any_of[0], or same_as_tensor's dtype when both are set, is left out of a schema name
    any_of: tuple[DType, ...] | None = None
    same_as_tensor: str | None = None
    default: DType | None = None

    @model_validator(mode="after")
    def _check(self) -> DTypeSpec:
        if self.default is not None and self.any_of and self.default != self.any_of[0]:
            raise ValueError(f"dtype default {self.default} is not any_of[0]")
        return self


def _one_branch(value: Any) -> Any:
    """A bare item stands for ``((None, item),)``: one branch, always taken."""
    return ((None, value),) if isinstance(value, Distribution | ErrorBound) else value


class TensorSpec(FrozenModel):
    kind: Literal["tensor"] = "tensor"
    description: str = Field(min_length=1)
    shape: tuple[str | int, ...] | ShapeByAttrSpec
    dtype: DTypeSpec
    present_when: str | None = None  # set exactly on an optional tensor
    can_squeeze: bool = False  # also accept the shape without its trailing 1
    distribution: Annotated[  # inputs only
        tuple[tuple[str | None, DistributionUnion], ...], BeforeValidator(_one_branch)
    ] = ()
    # floating outputs and mutable inputs only; first true predicate wins
    error_bounds: Annotated[tuple[tuple[str | None, ErrorBoundUnion], ...], BeforeValidator(_one_branch)] = ()
    lifetime: TensorLifetime = TensorLifetime.PER_CALL
    # inputs only: the kernel may write it in place (a KV cache), and it is compared as an
    # output after the call
    mutable: bool = False

    @model_validator(mode="after")
    def _check(self) -> TensorSpec:
        for field_name in ("distribution", "error_bounds"):
            predicates = [p for p, _ in getattr(self, field_name)]
            if predicates and (predicates[-1] is not None or None in predicates[:-1]):
                raise ValueError(f"{field_name}: only the last predicate, and it, must be None")
        return self

    def shape_cases(self, attrs: Mapping[str, Any] | None = None) -> tuple[tuple[str | int, ...], ...]:
        """The shapes this tensor may take; the one chosen by ``attrs`` when given."""
        if not isinstance(self.shape, ShapeByAttrSpec):
            return (self.shape,)
        if attrs is None:
            return tuple(self.shape.cases.values())
        return (self.shape.cases[case_key(attrs[self.shape.attr_name])],)

    def distribution_for(self, symbols: Mapping[str, Any]) -> Distribution:
        """The item of the first ``distribution`` branch true on ``symbols``."""
        for predicate, distribution in self.distribution:
            if predicate is None or expr.compile(predicate)[0](symbols):
                return distribution
        raise ValueError("an output has no distribution")

    def narrow(self, attrs: Mapping[str, Any], dtype: str | None, symbols: Mapping[str, Any]) -> TensorSpec:
        """This tensor in a schema (``OpContract.narrow``): its shape case at ``attrs``;
        when present (``dtype`` not ``None``) only ``dtype`` and its branches decided
        on ``symbols``."""
        (shape,) = self.shape_cases(attrs)
        update: dict[str, Any] = {"shape": shape}
        if dtype is not None:
            update["dtype"] = DTypeSpec(any_of=(DType(dtype),))
            update["distribution"] = self._decide_branches(self.distribution, symbols)
            update["error_bounds"] = self._decide_branches(self.error_bounds, symbols)
        return type(self)(**{**dict(self), **update})

    @staticmethod
    def _decide_branches(
        branches: tuple[tuple[str | None, Any], ...], symbols: Mapping[str, Any]
    ) -> tuple[tuple[str | None, Any], ...]:
        decided: list[tuple[str | None, Any]] = []
        for predicate, item in branches:
            if predicate is None:
                is_true: bool | None = True
            else:
                closure, reads = expr.compile(predicate)
                is_true = bool(closure(symbols)) if reads <= symbols.keys() else None
            if is_true is False and not decided:
                continue
            values = {}
            for field_name in type(item).expr_fields:
                text = getattr(item, field_name)
                if isinstance(text, str):
                    closure, reads = expr.compile(text)
                    if reads <= symbols.keys() and type(value := closure(symbols)) is int:
                        values[field_name] = value
            kept = item.model_copy(update=values) if values else item
            if is_true:
                decided.append((None, kept))
                break
            decided.append((predicate, kept))
        return tuple(decided)


class ScalarSpec(FrozenModel):
    """An argument scalar, never part of a ``PointSpec``; ``default`` is the API
    default."""

    kind: Literal["scalar"] = "scalar"
    description: str = Field(min_length=1)
    dtype: DType
    values: tuple[int | float | bool, ...] | None = None
    interval: tuple[float, float] | None = None
    default: int | float | bool | None = None
    present_when: str | None = None

    def normalize(self, value: object) -> bool | int | float:
        """``value`` as the kernel receives it: bool, int or float (float32 narrowed,
        -0.0 is 0.0); a wrong kind or out-of-range value raises ``ValueError``."""
        dtype = self.dtype
        if not isinstance(value, (bool, int, float)):
            # ValueError: callers report every bad value alike
            raise ValueError(  # noqa: TRY004
                f"expected a {dtype.value} scalar, got {type(value).__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite scalar {value!r}")
        if (dtype is DType.BOOL) != isinstance(value, bool):
            raise ValueError(f"expected a {dtype.value} scalar, got {value!r}")
        if dtype is DType.BOOL:
            return value
        if DTYPE_TRAITS[dtype].is_integral:
            if int(value) != value:
                raise ValueError(f"expected an integer scalar, got {value!r}")
            return int(value)
        try:  # a huge int overflows the double; a float32 overflow packs as inf
            narrowed = float(value) if value else 0.0
            if dtype is DType.FLOAT32:
                narrowed = struct.unpack("f", struct.pack("f", narrowed))[0]
        except OverflowError:
            narrowed = math.inf
        if not math.isfinite(narrowed):
            raise ValueError(f"{value!r} does not fit in {dtype.value}")
        return narrowed

    def value_error(self, value: bool | float) -> str | None:
        """Why ``value`` is not a workload value of this scalar, else ``None``."""
        try:
            normalized = self.normalize(value)
        except ValueError as e:
            return str(e)
        if (type(normalized), normalized) != (type(value), value):
            return f"{value!r} is not normalized ({normalized!r})"
        if self.values is not None and (type(value), value) not in {(type(v), v) for v in self.values}:
            return f"{value!r} is not one of {list(self.values)}"
        if self.interval is not None and not (self.interval[0] <= value <= self.interval[1]):
            return f"{value!r} is outside {list(self.interval)}"
        return None


ArgSpec = Annotated[TensorSpec | ScalarSpec, Field(discriminator="kind")]


class CompareSpec(FrozenModel):
    max_violation_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    determinism: DeterminismLevel = DeterminismLevel.BITWISE
    element_order: ElementOrder = ElementOrder.FIXED


def evaluated_json(value: Distribution | ErrorBound, symbols: Mapping[str, Any]) -> dict[str, Any]:
    """The json of ``value`` (defaults included) with ``expr_fields`` texts evaluated."""
    evaluated = value.model_dump(mode="json")
    for field_name in type(value).expr_fields:
        if isinstance(evaluated[field_name], str):
            evaluated[field_name] = expr.compile(evaluated[field_name])[0](symbols)
    return evaluated
