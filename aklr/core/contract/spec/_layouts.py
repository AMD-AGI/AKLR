# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import itertools

from pydantic import model_validator

from aklr.core.contract.semantics import Layout, RowMajor
from aklr.core.contract.semantics._frozen_model import FrozenModel
from aklr.core.contract.semantics._layouts import FactLocation, LayoutUnion
from aklr.core.contract.spec._points import AttrValue, attr_membership_key


class LayoutFact(FrozenModel):
    """How a call shows its layout: the value its ``location`` reads for ``name`` equals
    ``expected_value``."""

    location: FactLocation
    name: str
    expected_value: AttrValue


class LayoutVariant(FrozenModel):
    """Named by its key in ``LayoutVariantsSpec.by_name``."""

    layouts: dict[str, LayoutUnion] = {}  # noqa: RUF012 -- pydantic copies defaults
    detect_facts: tuple[LayoutFact, ...] = ()

    def layout_of(self, tensor: str) -> Layout:
        """The layout of ``tensor``; ``RowMajor`` when the variant does not list it."""
        return self.layouts.get(tensor, RowMajor())


class LayoutVariantsSpec(FrozenModel):
    by_name: dict[str, LayoutVariant]

    @model_validator(mode="after")
    def _check(self) -> LayoutVariantsSpec:
        if not self.by_name:
            raise ValueError("there must be a layout variant")
        if "row_major" in self.by_name and self.by_name["row_major"].layouts:
            raise ValueError('layout variant "row_major" lists no tensor layout')
        if len(self.by_name) > 1 and not all(variant.detect_facts for variant in self.by_name.values()):
            raise ValueError("with several layout variants each needs a fact")
        for (name, facts), (other_name, other_facts) in itertools.combinations(
            ((name, variant.detect_facts) for name, variant in self.by_name.items()),
            2,
        ):
            expected = {fact.name: fact.expected_value for fact in facts}
            if not any(
                fact.name in expected
                and attr_membership_key(expected[fact.name]) != attr_membership_key(fact.expected_value)
                for fact in other_facts
            ):
                raise ValueError(f"layout variants {name!r} and {other_name!r} are not told apart by a fact")
        return self

    def names(self) -> tuple[str, ...]:
        """In declaration order: the index is the fast key's layout_variant_idx."""
        return tuple(self.by_name)
