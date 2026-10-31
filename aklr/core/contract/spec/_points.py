# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The coordinates of a contract's domain: the contract id naming it, axis and attr
specs, and the Point."""

from __future__ import annotations

import functools
import re
from typing import Any, Literal

from pydantic import PositiveInt, model_validator

from aklr.core.contract.semantics import sorted_key_json
from aklr.core.contract.semantics._frozen_model import FrozenModel

ContractId = str  # "<contract_name>@<major>.<minor>"; the version is for humans
CONTRACT_ID_PATTERN = re.compile(r"^([a-z][a-z0-9_]*)@(\d+)\.(\d+)$")


def contract_name(contract_id: ContractId) -> str:
    """``"gemm_a8w8@1.0"`` -> ``"gemm_a8w8"``."""
    match = CONTRACT_ID_PATTERN.fullmatch(contract_id)
    if match is None:
        raise ValueError(f"not a contract id: {contract_id!r}")
    return match.group(1)


AttrValue = str | int | bool


def attr_membership_key(value: AttrValue | None) -> tuple[bool, AttrValue | None]:
    """``(is_bool, value)``: membership key keeping ``True`` and ``1`` distinct."""
    return type(value) is bool, value


def case_key(value: AttrValue) -> str:
    """The text key of an attr value: ``str(value)``, but ``true``/``false`` for a bool."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def attr_value(raw: Any) -> AttrValue | None:
    """An api argument as an attr value: an enum by its lower-case name, ``None``/
    bool/int/str as is. Unhashable: ``TypeError`` (a generated closure declines)."""
    if raw is None or type(raw) in (bool, int, str):
        return raw
    return _enum_name(raw)


# ``typed``: members of two IntEnums with one int compare equal, names differ.
@functools.lru_cache(maxsize=None, typed=True)
def _enum_name(raw: Any) -> str:  # structure: allow S3, S4, S5 lru_cache  # fmt: skip
    return raw.name.lower()


class AxisSpec(FrozenModel):
    kind: Literal["int", "derived"]
    description: str
    lo: int | None = None
    hi: int | None = None
    multiple_of: PositiveInt = 1
    expr: str | int | None = None  # derived only; an int in a schema (decided)
    is_var: bool = False  # varies across a schema's workloads; else fixed by the schema
    # a var axis the kernel does not depend on (a KV cache's rows): a schema serving one
    # value serves them all, its workloads' values only what bench runs
    any_size: bool = False

    @model_validator(mode="after")
    def _check(self) -> AxisSpec:
        if self.any_size and not self.is_var:
            raise ValueError("an any_size axis is a var axis")
        if self.kind == "derived":
            if self.expr is None or self.lo is not None or self.hi is not None:
                raise ValueError("a derived axis has expr and no lo/hi")
            if self.is_var:
                raise ValueError("a derived axis is not a var axis")
        elif self.lo is None or self.hi is None or self.expr is not None:
            raise ValueError("an int axis has lo, hi and no expr")
        elif self.lo > self.hi or self.lo % self.multiple_of:
            raise ValueError("int axis: need lo <= hi and lo % multiple_of == 0")
        return self


class AttrSpec(FrozenModel):
    description: str
    values: tuple[AttrValue, ...]  # values[0] is left out of a schema name
    default: AttrValue | None = None
    # None: derived from whether an optional tensor is passed
    param_name: str | None = None
    abbrev: str | None = None  # stands for the attr in a schema name in place of its name
    # added after schemas were committed: at its default a schema leaves it out, and
    # its kernel takes neither its param nor the inputs whose presence reads it
    optional: bool = False

    @model_validator(mode="after")
    def _check(self) -> AttrSpec:
        keys = [attr_membership_key(v) for v in self.values]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("attr values must be non-empty and distinct")
        if self.optional and self.default is None and len(keys) > 1:
            raise ValueError("an optional attr has a default")
        if self.default is not None and attr_membership_key(self.default) != keys[0]:
            raise ValueError(f"attr default {self.default!r} is not the first of {self.values!r}")
        return self

    @property
    def is_str(self) -> bool:
        """Whether every value is a ``str``: a str symbol of expressions."""
        return all(isinstance(v, str) for v in self.values)

    def in_name(self, name: str, value: AttrValue) -> str:
        """Attr ``name``'s part of a schema name at ``value``: ``""`` at
        ``values[0]``, else ``<label>``/``no_<label>`` (bool), ``<label><value>``
        (int) or the lowercased str, ``<label>`` being ``abbrev`` or ``name``."""
        if attr_membership_key(value) == attr_membership_key(self.values[0]):
            return ""
        label = self.abbrev or name
        if isinstance(value, bool):
            return label if value else f"no_{label}"
        if isinstance(value, int):
            return f"{label}{value}"
        return value.lower()


def axis(
    description: str,
    lo: int,
    hi: int,
    *,
    multiple_of: int = 1,
    is_var: bool = False,
    any_size: bool = False,
) -> AxisSpec:
    return AxisSpec(
        kind="int",
        description=description,
        lo=lo,
        hi=hi,
        multiple_of=multiple_of,
        is_var=is_var,
        any_size=any_size,
    )


def derived_axis(description: str, expr: str) -> AxisSpec:
    return AxisSpec(kind="derived", description=description, expr=expr)


def attr(
    description: str,
    *values: AttrValue,
    default: AttrValue | None = None,
    param_name: str | None = None,
    abbrev: str | None = None,
    optional: bool = False,
) -> AttrSpec:
    return AttrSpec(
        description=description,
        values=values,
        default=default,
        param_name=param_name,
        abbrev=abbrev,
        optional=optional,
    )


class PointSpec(FrozenModel):
    """One coordinate of a contract's domain: every int axis, tensor dtype (``None``:
    absent), attr and the layout; ``OpContract.check_point`` enforces completeness."""

    contract_id: ContractId
    axes: dict[str, int]
    dtypes: dict[str, str | None]
    attrs: dict[str, AttrValue]
    layout_variant_name: str

    def key(self) -> str:
        """``sorted_key_json`` of every field."""
        return sorted_key_json(self.model_dump(mode="json"))

    def symbols(self) -> dict[str, int | str | bool]:
        """Expression symbols of the Point, without derived axes (``Domain.symbols``)."""
        return {
            **self.axes,
            **{f"{t}.dtype": d for t, d in self.dtypes.items() if d is not None},
            **self.attrs,
            "layout_variant": self.layout_variant_name,
        }
