# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One of main's contracts re-cut for a host target: the target's own param names in place of main's,
and what the target fixes no longer a choice."""

from __future__ import annotations

import ast
import dataclasses
import operator
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from aklr.core.contract.baseline import DistOpBaseline
from aklr.core.contract.op import OpContract
from aklr.core.contract.reference import DistOpReference, OpReference
from aklr.core.contract.spec import (
    LayoutVariant,
    LayoutVariantsSpec,
    PointSpec,
    ShapeByAttrSpec,
    TensorSpec,
    contract_name,
)
from aklr.core.contract.spec._points import attr_membership_key, case_key
from codegen.kernel_entry import KernelEntry

__all__ = ["Adaptation"]

_NOTHING: Mapping[str, Any] = MappingProxyType({})
_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class _Expressions(ast.NodeTransformer):
    """A contract's expression texts under an adaptation: main's names replaced by the target's, and
    the names the target fixes by their values, folded."""

    def __init__(self, rename: Mapping[str, str], pin: Mapping[str, Any]) -> None:
        self.rename, self.pin, self.is_changed = rename, pin, False

    def text(self, text: str) -> bool | str:
        """``text`` re-cut: a bool once what the target fixes decides it, else the expression left."""
        self.is_changed = False
        node = self.visit(ast.parse(text, mode="eval").body)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        return self._rendered(node) if self.is_changed else text

    def branches(self, branches: tuple[tuple[str | None, Any], ...]) -> tuple[tuple[str | None, Any], ...]:
        """A tensor's distribution or error-bound branches re-cut: a branch whose predicate the target
        makes false is gone, one it makes true is the last."""
        kept = []
        for predicate, item in branches:
            when = None if predicate is None else self.text(predicate)
            if when is False:
                continue
            fields = {
                name: self.text(value)
                for name in getattr(type(item), "expr_fields", ())
                if isinstance(value := getattr(item, name), str)
            }
            kept.append((None if when is True else when, item.model_copy(update=fields)))
            if when is True:
                break
        return tuple(kept)

    def _rendered(self, node: ast.expr) -> str:
        """``node`` as text, an and, or or not inside another in parentheses, as a contract writes them."""
        nested = [
            f"({self._rendered(v)})" if isinstance(v, ast.BoolOp | ast.UnaryOp) else self._rendered(v)
            for v in (node.values if isinstance(node, ast.BoolOp) else [getattr(node, "operand", None)])
            if v is not None
        ]
        if isinstance(node, ast.BoolOp):
            return (" and " if isinstance(node.op, ast.And) else " or ").join(nested)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"not {nested[0]}"
        return ast.unparse(node)

    def visit_Name(self, node: ast.Name) -> ast.expr:  # noqa: N802 -- ast's visitor name
        if node.id in self.pin:
            self.is_changed = True
            return ast.Constant(self.pin[node.id])
        if node.id in self.rename:
            self.is_changed = True
            return ast.Name(self.rename[node.id], node.ctx)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:  # noqa: N802 -- ast's visitor name
        self.generic_visit(node)
        is_and = isinstance(node.op, ast.And)
        undecided = []
        for value in node.values:
            if not isinstance(value, ast.Constant):
                undecided.append(value)
            elif bool(value.value) != is_and:  # a false term of an and, a true one of an or
                return ast.Constant(not is_and)
        if not undecided:
            return ast.Constant(is_and)
        return undecided[0] if len(undecided) == 1 else ast.BoolOp(node.op, undecided)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:  # noqa: N802 -- ast's visitor name
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.Constant(not node.operand.value)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.expr:  # noqa: N802 -- ast's visitor name
        self.generic_visit(node)
        try:
            values = [ast.literal_eval(operand) for operand in (node.left, *node.comparators)]
        except ValueError:  # an operand the target does not fix
            return node
        return ast.Constant(
            all(
                _COMPARISONS[type(op)](a, b)
                for op, a, b in zip(node.ops, values[:-1], values[1:], strict=True)
            )
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Adaptation:
    """Main's contract ``source`` re-cut for the host target ``target`` on the host versions ``versions``,
    into the family ``family``. ``target`` is ``None`` for a contract no host function takes: it moves
    as it is, for validation only.

    ``rename`` maps main's call names (inputs, outputs, attrs and their params) to the target's params.
    ``pin`` holds what the target fixes, by main's attr name or ``"layout_variant"``: no longer a choice of
    the re-cut contract, nor a param. ``facts`` are the layout facts (main's names) the fixed variant
    still shows in a call of the target; the others are gone with the arguments main's caller passed.
    ``override`` is merged over the re-cut contract's json, in the target's names: what the target
    differs in beyond names (``None`` removes a key)."""

    source: str
    family: str
    target: str | None = None
    versions: tuple[int, ...] | tuple[str, ...] = ()
    contract_id: str | None = None
    rename: Mapping[str, str] = _NOTHING
    pin: Mapping[str, Any] = _NOTHING
    facts: tuple[str, ...] = ()
    override: Mapping[str, Any] = _NOTHING

    @property
    def is_move(self) -> bool:
        """Whether the contract only moves: main's names, attrs and layouts as they are."""
        return not (self.rename or self.pin or self.override)

    def contract(self, main: OpContract) -> OpContract:
        """The re-cut contract of ``main``, which is ``source``; its reference and baseline are main's,
        for the ones ``class_source`` writes to take their place."""
        expressions = _Expressions(self.rename, self.pin)
        inputs = {}
        for name, spec in main.inputs.items():
            present_when = None if spec.present_when is None else expressions.text(spec.present_when)
            if present_when is False:  # the target never passes it
                continue
            update = {
                "present_when": None if present_when is True else present_when,
                "description": self._described(spec.description),
            }
            if isinstance(spec, TensorSpec):
                update |= self._tensor(spec, expressions)
            inputs[self._name(name)] = spec.model_copy(update=update)
        variants = {}
        for name, variant in main.layout_variants.by_name.items():
            if self.pin.get("layout_variant", name) != name:
                continue
            facts = []
            for fact in variant.detect_facts:
                if "layout_variant" in self.pin and fact.name not in self.facts:
                    continue
                tensor, _, attribute = fact.name.partition(".")
                renamed = f"{self._name(tensor)}.{attribute}" if attribute else self._name(tensor)
                facts.append(fact.model_copy(update={"name": renamed}))
            layouts = {
                self._name(t): layout for t, layout in variant.layouts.items() if self._name(t) in inputs
            }
            variants[name] = LayoutVariant(layouts=layouts, detect_facts=tuple(facts))
        constraints = [expressions.text(text) for text in main.constraints]
        if False in constraints:
            raise ValueError(f"{self.source}: what {self.target} fixes is outside its constraints")
        contract = OpContract(
            contract_id=self.contract_id
            or (f"{self.target.rpartition('.')[2]}@1.0" if self.target else self.source),
            description=self._described(main.description),
            op_type=self.family,
            axes={
                name: a.model_copy(update={"expr": expressions.text(a.expr)}) if a.kind == "derived" else a
                for name, a in main.axes.items()
            },
            attrs={
                self._name(name): a.model_copy(
                    update={"param_name": a.param_name and self._name(a.param_name)}
                )
                for name, a in main.attrs.items()
                if name not in self.pin
            },
            constraints=tuple(text for text in constraints if text is not True),
            inputs=inputs,
            outputs={
                self._name(name): spec.model_copy(update=self._tensor(spec, expressions))
                for name, spec in main.outputs.items()
            },
            layout_variants=LayoutVariantsSpec(by_name=variants),
            compare=main.compare,
            ranks=main.ranks and self._name(main.ranks),
            reference=main.reference,
            baseline=main.baseline,
        )
        if not self.override:
            return contract
        merged = _merged(contract.model_dump(mode="json"), self.override)
        return OpContract.model_validate({**merged, "reference": main.reference, "baseline": main.baseline})

    def point(self, main_point: PointSpec, contract: OpContract) -> PointSpec | None:
        """``main_point``, a Point of ``source``, as a Point of the re-cut ``contract``; ``None`` where the
        target fixes otherwise, or ``contract`` does not hold it."""
        for name, value in self.pin.items():
            actual = main_point.layout_variant_name if name == "layout_variant" else main_point.attrs[name]
            if attr_membership_key(actual) != attr_membership_key(value):
                return None
        attrs = {self._name(name): value for name, value in main_point.attrs.items() if name not in self.pin}
        # an attr the target adds, at the value main's kernels were written for: its first
        attrs |= {name: a.values[0] for name, a in contract.attrs.items() if name not in attrs}
        tensors = contract.tensors()
        point = PointSpec(
            contract_id=contract.contract_id,
            axes=main_point.axes,
            dtypes={self._name(t): d for t, d in main_point.dtypes.items() if self._name(t) in tensors},
            attrs=attrs,
            layout_variant_name=main_point.layout_variant_name,
        )
        return point if contract.domain.contains(point) else None

    def schema(
        self, main_schema: Mapping[str, Any], main: OpContract, contract: OpContract
    ) -> OpContract | None:
        """The schema file json ``main_schema`` of ``main`` as a schema of the re-cut ``contract``: the
        same workloads with their measurements, the same kernel with its entry taking the target's
        names. ``None`` where the target fixes otherwise than the schema."""
        workloads, point, main_point = [], None, None
        for workload in main_schema["workloads"]:
            main_point = main.point_from_json(main_schema, workload["axes"])
            point = self.point(main_point, contract)
            if point is None:
                return None
            scalars = {self._name(n): v for n, v in workload.get("scalars", {}).items()}
            scalars = {n: v for n, v in scalars.items() if n in contract.inputs}
            measured = {k: workload[k] for k in ("accuracy", "speedup", "microseconds") if k in workload}
            workloads.append({**contract.workload_for(point, scalars).model_dump(), **measured})
        kernel = dict(main_schema["kernel"])
        entry = KernelEntry(
            kernel["language"], kernel["entry_point"], {s["path"]: s["content"] for s in kernel["sources"]}
        )
        call_keys = contract.narrow(point, workloads=workloads, kernel=kernel).kernel_call_keys
        constants = {name: repr(value) for name, value in self._fixed(main).items()}
        sources = entry.retargeted(self.rename, call_keys, constants)
        kernel["sources"] = [{"path": path, "content": content} for path, content in sources.items()]
        return contract.narrow(point, workloads=workloads, kernel=kernel)

    def class_name(self, contract: OpContract, role: str) -> str:
        """The name of the re-cut ``contract``'s ``role`` class, ``"reference"`` or ``"baseline"``."""
        words = contract_name(contract.contract_id).split("_")
        return "".join(word.capitalize() for word in words) + role.capitalize()

    def class_source(self, main: OpContract, contract: OpContract, role: str) -> str:
        """The source of the re-cut ``contract``'s ``role`` class, ``"reference"`` or ``"baseline"``: one
        over main's class (``Main<its name>`` where the source runs) that takes the target's call, calls
        main's with main's names, and names its results the target's."""
        base = type(getattr(main, role))
        if isinstance(getattr(main, role), DistOpReference | DistOpBaseline):
            raise NotImplementedError(f"{self.source}: an op of ranks is re-cut by moving only")
        if role == "reference" and base.check is not OpReference.check:
            raise NotImplementedError(f"{self.source}: {base.__name__} has a check of its own names")
        fixed, marks, shown = self._fixed(main), "", {}
        if role == "baseline" and "layout_variant" in self.pin:
            # a fact main's caller showed on a tensor, which the target's caller does not: shown here
            for fact in main.layout_variants.by_name[self.pin["layout_variant"]].detect_facts:
                tensor, _, attribute = fact.name.partition(".")
                if attribute and fact.name not in self.facts:
                    if tensor not in shown:
                        new = self._name(tensor)
                        marks += f"        {tensor} = call[{new!r}].view(call[{new!r}].shape)\n"
                        shown[tensor] = tensor
                    marks += f"        {tensor}.{attribute} = {fact.expected_value!r}\n"
        kwargs = ", ".join(
            f"{p.name}="
            + (
                repr(fixed[p.name])
                if p.name in fixed
                else shown.get(p.name, f"call.get({self._name(p.name)!r})")
            )
            for p in main.api_params
        )
        renamed = {old: new for old, new in self.rename.items() if old in main.tensors()}
        return (
            f"\n\nclass {self.class_name(contract, role)}(Main{base.__name__}):\n"
            f'    """main\'s ``{base.__name__}`` under the names of ``{contract.contract_id}``."""\n\n'
            f"    def {role}(self, **call):\n"
            f"{marks}"
            f"        results = super().{role}({kwargs})\n"
            f"        return {{{renamed!r}.get(name, name): tensor for name, tensor in results.items()}}\n"
        )

    def _name(self, name: str) -> str:
        return self.rename.get(name, name)

    def _described(self, text: str) -> str:
        """``text``, a description of main's, with the target's names in its formulas: a name beside an
        operator or a bracket, or a one-letter name. A longer name elsewhere is a word of the sentence, and
        so is an "a" before a word."""
        if not self.rename:
            return text
        names = "|".join(map(re.escape, sorted(self.rename, key=len, reverse=True)))

        def named(match: re.Match) -> str:
            old, before, after = match.group(), text[: match.start()].rstrip(), text[match.end() :]
            if len(old) == 1:
                is_name = not (old == "a" and re.match(r" [a-z0-9]", after))
            else:
                is_name = (
                    before[-1:] in ("=", "+", "*", "@", "(", ";")
                    or after.lstrip()[:1] in ("=", "+", "*", "@", ")", ";")
                    or re.match(r"\.[A-Za-z]", after) is not None
                )
            return self.rename[old] if is_name else old

        return re.sub(rf"(?<![\w.])({names})\b", named, text)

    def _tensor(self, spec: TensorSpec, expressions: _Expressions) -> dict[str, Any]:
        """The fields of main's tensor ``spec`` that the adaptation changes."""
        shape, dtype = spec.shape, spec.dtype
        if isinstance(shape, ShapeByAttrSpec):
            if shape.attr_name in self.pin:
                shape = shape.cases[case_key(self.pin[shape.attr_name])]
            else:
                shape = shape.model_copy(update={"attr_name": self._name(shape.attr_name)})
        return {
            "description": self._described(spec.description),
            "shape": shape,
            "dtype": dtype.model_copy(
                update={
                    "same_as_tensor": dtype.same_as_tensor and self._name(dtype.same_as_tensor),
                    "param_name": dtype.param_name and self._name(dtype.param_name),
                }
            ),
            "distribution": expressions.branches(spec.distribution),
            "error_bounds": expressions.branches(spec.error_bounds),
        }

    def _fixed(self, main: OpContract) -> dict[str, Any]:
        """Main's api params the target fixes, each with its value: the pinned attrs' params, the fixed
        layout variant's argument facts, and ``None`` for an input the target never passes."""
        fixed = {a.param_name: self.pin[n] for n, a in main.attrs.items() if n in self.pin and a.param_name}
        if "layout_variant" in self.pin:
            for fact in main.layout_variants.by_name[self.pin["layout_variant"]].detect_facts:
                if fact.location.param_name(fact.name) is not None and fact.name not in self.facts:
                    fixed[fact.name] = fact.expected_value
        expressions = _Expressions(self.rename, self.pin)
        for name, spec in main.inputs.items():
            if spec.present_when is not None and expressions.text(spec.present_when) is False:
                fixed[name] = None
        return fixed


def _merged(base: Any, patch: Any) -> Any:
    """``base`` with ``patch`` merged over it key by key, ``base``'s order kept: a json merge patch."""
    if not isinstance(patch, Mapping) or not isinstance(base, Mapping):
        return patch
    merged = {key: _merged(value, patch[key]) if key in patch else value for key, value in base.items()}
    merged |= {key: value for key, value in patch.items() if key not in base}
    return {key: value for key, value in merged.items() if value is not None}
