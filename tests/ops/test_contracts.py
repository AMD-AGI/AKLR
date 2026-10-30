# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests of the registered op contracts: registration and variant names."""

import itertools
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import aklr.ops
from aklr.core.contract import export, expr
from aklr.core.contract.op import OpContract, api_name
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import AxisSpec, PointSpec
from aklr.core.contract.spec._points import attr_membership_key
from aklr.core.runtime import seam
from tests.op_case import OpCase


class TestOpRegistration:
    """Every registered contract has an ``OpCase`` and an api; a route is optional."""

    def test_every_contract_has_an_op_case(self) -> None:
        """``OpCase.all()``'s distinct contract ids are exactly
        ``ContractRegistry.default()``'s, in order."""
        assert list(dict.fromkeys(case.contract_id for case in OpCase.all())) == [
            c.contract_id for c in ContractRegistry.default().contracts()
        ]

    def test_load_routes_declares_ck_blockscale_first(self) -> None:
        """``aklr.ops.load_routes()`` imports every op's api; a new op may have no
        route (it is its own baseline)."""
        aklr.ops.load_routes()
        blockscale = OpCase.of("gemm_a8w8_blockwise_scaled@1.0").contract.api_name
        # FIXME(main-kernels): the bench baseline is the first route that binds; main's is CK
        assert (
            next(r.api for r in seam.routes() if api_name(r.api_fn) == blockscale)
            == "aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale"
        )

    def test_committed_schemas_have_no_errors(self) -> None:
        """``export.repository_errors()`` (``python -m aklr export check``) is ``[]``."""
        assert export.repository_errors() == []


_TESTS = Path(__file__).parents[1]
# read at import: the autouse ``aklr_env`` fixture unsets every AKLR_* variable
_UPDATE_NAMES = os.environ.get("AKLR_UPDATE_NAMES") == "1"


class _Variants:
    """A contract's variants: one Point per distinct attrs, dtypes and layout that the
    relations reading no axis allow (a superset of the domain's), each int axis at
    ``axis(spec)``."""

    def __init__(self, contract: OpContract) -> None:
        self.contract = contract

    def points(self, axis: Callable[[AxisSpec], int | str]) -> Iterator[PointSpec]:
        c = self.contract
        roots = [t for t in c.tensors() if c._dtype_root(t) == t]
        relations = [expr.compile(r) for r in c.domain.relations]
        free = [f for f, reads in relations if not reads & c.axes.keys()]
        seen = set()
        for values, root_dtypes, layout in itertools.product(
            itertools.product(*(a.values for a in c.attrs.values())),
            itertools.product(*(tuple(d.value for d in c.dtypes_of(t)) for t in roots)),
            c.layout_variants.names(),
        ):
            attrs = dict(zip(c.attrs, values, strict=True))
            dtype_of = dict(zip(roots, root_dtypes, strict=True))
            # presence reads attrs alone (ExprError otherwise), so a name tells it
            dtypes = {
                t: (
                    None
                    if s.present_when and not expr.compile(s.present_when)[0](attrs)
                    else dtype_of[c._dtype_root(t)]
                )
                for t, s in c.tensors().items()
            }
            key = (tuple(map(attr_membership_key, values)), *dtypes.values(), layout)
            symbols = {**attrs, "layout_variant": layout}
            symbols |= {f"{t}.dtype": d for t, d in dtypes.items()}
            try:
                allowed = all(f(symbols) for f in free)
            except (expr.ExprError, ArithmeticError, TypeError):
                allowed = False
            if key in seen or not allowed:
                continue
            seen.add(key)
            yield PointSpec.model_construct(
                contract_id=c.contract_id,
                axes={n: axis(a) for n, a in c.axes.items() if a.kind == "int"},
                dtypes=dtypes,
                attrs=attrs,
                layout_variant_name=layout,
            )


class TestNames:
    """``OpContract.name_at`` names every variant apart, in a file name."""

    def test_names_are_unique_across_contracts(self) -> None:
        """No two variants of any contracts share a name, const axes left as ``#``
        (shortcut: an int attr part beside an axis digit is not told apart from
        the axis value; a real clash of that kind also needs equal axis values)."""
        owner: dict[str, str] = {}
        for c in ContractRegistry.default().contracts():
            for point in _Variants(c).points(lambda a: "#"):
                name = c.name_at(point)
                assert name not in owner, f"{name}: {owner.get(name)} and {point}"
                owner[name] = str(point)

    def test_names_fit_a_file_name(self) -> None:
        """Every name, each int axis at its ``hi``, is at most 255 bytes of
        ``_``-joined lowercase words."""
        for c in ContractRegistry.default().contracts():
            for point in _Variants(c).points(lambda a: a.hi):
                name = c.name_at(point)
                assert len(name.encode()) <= 255, name
                assert re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", name), name

    def test_names_match_the_snapshot(self) -> None:
        """Each ``names.txt`` (``_names_file``) lists its contracts' variant names: a rename
        shows in its diff. Regenerate: ``AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py``.
        """
        texts: dict[Path, str] = {}
        for c in ContractRegistry.default().contracts():
            names = "".join(f"{c.name_at(p)}\n" for p in _Variants(c).points(lambda a: "#"))
            texts[_names_file(c)] = texts.get(_names_file(c), "") + names
        for path, text in texts.items():
            if _UPDATE_NAMES:
                path.write_text(text)
            assert path.read_text() == text


def _names_file(contract: OpContract) -> Path:
    """Where ``contract``'s variant names are kept: tests/plugins/<framework>/names.txt for a
    plugin op, else tests/ops/names.txt."""
    parts = contract.op_package.split(".")
    if parts[1] == "plugins":
        return _TESTS / "plugins" / parts[2] / "names.txt"
    return _TESTS / "ops" / "names.txt"
