# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ast
import inspect
import itertools
import math
import textwrap
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final

from pydantic import Field, ValidationError, model_validator

from aklr.core.contract import expr, semantics
from aklr.core.contract.locate import (
    OP_PACKAGE_PATTERN,
    op_module,
    op_package_of,
    schema_relative_path,
    schema_uuid,
)
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DeterminismLevel,
    DType,
    ElementOrder,
    sorted_key_json,
)
from aklr.core.contract.spec import (
    CONTRACT_ID_PATTERN,
    ArgSpec,
    AttrSpec,
    AxisSpec,
    CompareSpec,
    ContractId,
    FrozenModel,
    KernelSpec,
    LayoutVariantsSpec,
    PointSpec,
    ScalarSpec,
    ShapeByAttrSpec,
    TensorLifetime,
    TensorSpec,
    WorkloadSpec,
    axes_text,
    contract_name,
    evaluated_json,
)
from aklr.core.contract.spec._points import AttrValue, attr_membership_key, case_key
from aklr.core.contract.spec._sources import source_error_parts

__all__ = [
    "CONTRACT_VERSION",
    "ApiParam",
    "Domain",
    "OpContract",
    "api_name",
]

CONTRACT_VERSION: Final = 5  # the json format of a schema file (its "schema_version")
_RESERVED_NAMES = frozenset({"out_dtype", "layout_variant"})
_DISTINCT_PRIME_DIMS = (10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079)


class OpContract(FrozenModel):
    """A registered contract (``name`` is ``None``) or a schema (``name``, ``models``,
    ``workloads`` and ``kernel`` filled); a schema's file is ``schema_text``."""

    # a schema's name, a path segment: impls/<language>/<name>/
    name: Annotated[str, Field(min_length=1)] | None = None
    contract_id: ContractId
    description: str
    op_type: str  # the op family: "gemm" for aklr/ops/gemm/
    # the models whose serving makes a schema's calls, sorted: "kimi-k3", "glm-5.2"
    models: tuple[Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")], ...] = ()
    workloads: tuple[WorkloadSpec, ...] = ()
    axes: dict[str, AxisSpec]
    attrs: dict[str, AttrSpec] = {}  # noqa: RUF012 -- pydantic copies defaults
    constraints: tuple[str, ...] = ()
    inputs: dict[str, ArgSpec]
    outputs: dict[str, TensorSpec]  # "out" alone, or names other than "out"
    layout_variants: LayoutVariantsSpec
    compare: CompareSpec = CompareSpec()
    kernel: KernelSpec | None = None
    # the attr whose value is how many ranks, one a device, call the op together; None: one
    ranks: str | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> OpContract:
        if self.name is None:
            if self.workloads or self.kernel is not None or self.models:
                raise ValueError("a registered contract has no workloads, kernel or models")
            if self.layout_variants.names()[0] != "row_major":
                raise ValueError('the first layout variant of a registered contract is "row_major"')
            return self
        if not self.workloads or self.kernel is None:
            raise ValueError("a schema has non-empty workloads and a kernel")
        if not self.models or list(self.models) != sorted(set(self.models)):
            raise ValueError(
                f"models {list(self.models)}: a schema names the models it serves, sorted, once each"
            )
        tensors = self.tensors()
        for name in self.mutable_tensors:
            spec = tensors[name]
            is_floating = not DTYPE_TRAITS[spec.dtype.any_of[0]].is_integral
            if is_floating and (
                len(spec.error_bounds) != 1
                or any(
                    isinstance(getattr(spec.error_bounds[0][1], f), str)
                    for f in type(spec.error_bounds[0][1]).expr_fields
                )
            ):
                raise ValueError(
                    f"{name}: a schema's floating mutable tensor has one bound, its expressions decided"
                )
        return self

    @model_validator(mode="after")
    def _check_names(self) -> OpContract:
        if not CONTRACT_ID_PATTERN.fullmatch(self.contract_id):
            raise ValueError(f"not a contract id: {self.contract_id!r}")
        if not OP_PACKAGE_PATTERN.fullmatch(self.op_package):
            raise ValueError(f"op_type {self.op_type!r}: {self.op_package!r} is not an op package")
        names = [
            *self.axes,
            *self.attrs,
            *self.inputs,
            *self.outputs,
            *self._layout_arg_names(),
        ]
        repeated = [n for n, count in Counter(names).items() if count > 1]
        if repeated or _RESERVED_NAMES & set(names):
            raise ValueError(f"repeated {repeated} or reserved {sorted(_RESERVED_NAMES)} names")
        if not self.outputs or ("out" in self.outputs and len(self.outputs) > 1):
            raise ValueError("outputs are 'out' alone, or names other than 'out'")
        return self

    def _scope(self) -> tuple[frozenset[str], frozenset[str]]:
        bools_ints = {n for n, a in self.attrs.items() if not a.is_str}
        strs = {n for n, a in self.attrs.items() if a.is_str}
        dtypes = {f"{t}.dtype" for t in self.tensors()}
        return frozenset(self.axes) | bools_ints, frozenset(dtypes | strs | {"layout_variant"})

    @model_validator(mode="after")
    def _check_expressions(self) -> OpContract:
        derived = [n for n, a in self.axes.items() if a.kind == "derived"]
        ints, strs = self._scope()
        const = ints - self.var_axes.keys()
        const_no_derived = const - set(derived)

        def check(text: str, int_symbols: frozenset[str] = ints) -> None:
            expr.check_scope(expr.parse(str(text)), int_symbols, strs)

        for i, name in enumerate(derived):  # only derived axes declared before it
            check(self.axes[name].expr, const_no_derived | set(derived[:i]))
        for text in self.constraints:
            check(text)
        for spec in self.inputs.values():
            if spec.present_when is not None:
                check(spec.present_when, const)
        for spec in self.tensors().values():  # error bounds read point.symbols()
            for predicate, bound in spec.error_bounds:
                if predicate is not None:
                    check(predicate, const_no_derived)
                for field_name in type(bound).expr_fields:
                    check(getattr(bound, field_name), const_no_derived)
        return self

    @model_validator(mode="after")
    def _check_tensors(self) -> OpContract:
        tensors = self.tensors()
        for name, spec in self.outputs.items():
            if spec.present_when is not None or spec.can_squeeze:
                raise ValueError(f"{name}: an output is never optional or squeezed")
        for name, spec in tensors.items():
            for shape in spec.shape_cases():
                if any(isinstance(d, str) and d not in self.axes for d in shape):
                    raise ValueError(f"{name}: shape {shape} names an unknown axis")
            if isinstance(spec.shape, ShapeByAttrSpec):
                attr = self.attrs.get(spec.shape.attr_name)
                keys = {case_key(v) for v in attr.values} if attr else None
                if set(spec.shape.cases) != keys:
                    raise ValueError(f"{name}: cases are not the attr's values")
            dtype_spec = spec.dtype
            source = dtype_spec.same_as_tensor
            if (
                source not in (*tensors, None)
                or (dtype_spec.any_of is None and source is None)
                or (source is not None and tensors[source].dtype.same_as_tensor is not None)
            ):  # no chains: dtypes_of/_dtype_root recurse at most once
                raise ValueError(f"{name}: needs any_of or the same_as_tensor of a tensor without one")
            if (
                name in self.outputs
                and "out" not in self.outputs
                and not (
                    (dtype_spec.same_as_tensor is None and len(dtype_spec.any_of or ()) == 1)
                    or (dtype_spec.any_of is None and dtype_spec.same_as_tensor in self.inputs)
                )
            ):
                raise ValueError(
                    f"{name}: an output other than 'out' has one dtype or the same_as_tensor of an input"
                )
            if (
                dtype_spec.default is not None
                and dtype_spec.any_of
                and dtype_spec.default not in dtype_spec.any_of
            ):
                raise ValueError(f"{name}: default dtype is not in any_of")
            storage = [s for d in dtype_spec.any_of or () for s in DTYPE_TRAITS[d].storage_dtypes]
            if len(storage) != len(set(storage)):  # Point <-> FastKey stays bijective
                raise ValueError(f"{name}: any_of dtypes share a storage dtype")
        # alloc_outputs makes the logical, not the packed, shape
        for name in self.outputs:
            if any(DTYPE_TRAITS[d].bits == 4 for d in self.dtypes_of(name)):
                raise ValueError(f"{name}: an output is never a 4-bit dtype")
        inputs = {n: s for n, s in self.inputs.items() if isinstance(s, TensorSpec)}
        for name, spec in inputs.items():
            if spec.mutable and (spec.present_when is not None or spec.lifetime != TensorLifetime.PER_CALL):
                raise ValueError(f"{name}: a mutable input is required and per_call")
        presence_attrs = {n for n, a in self.attrs.items() if a.param_name is None}
        for axis, readers in self._axis_readers.items():
            first = tensors[readers[0]] if readers else None
            if (
                first is None
                or "out" in readers  # "out" is allocated, so never required
                or (
                    first.present_when is not None
                    and (len(readers) > 1 or first.present_when not in presence_attrs)
                )
            ):
                raise ValueError(
                    f"axis {axis!r} is not readable from a required input, an output "
                    "other than 'out', or the one optional tensor of a presence attr"
                )
        presence = Counter(s.present_when for s in self.inputs.values())
        bools = {attr_membership_key(False), attr_membership_key(True)}
        for name, a in self.attrs.items():
            if a.param_name is not None:
                continue
            keys = {attr_membership_key(v) for v in a.values}
            if not keys <= bools or presence[name] != 1:
                raise ValueError(f"{name!r}: a presence attr is bool, one optional tensor")
            if self.name is None and keys != bools:
                raise ValueError(f"{name!r}: a presence attr of a registered contract takes both bools")
        for name, spec in self.inputs.items():
            if isinstance(spec, ScalarSpec) and not (
                spec.values
                or (
                    spec.interval is not None
                    and all(map(math.isfinite, spec.interval))
                    and spec.interval[0] <= spec.interval[1]
                )
            ):
                raise ValueError(f"scalar {name!r} needs values or a finite interval")
        for variant, v in self.layout_variants.by_name.items():
            named = {f.location.tensor_name(f.name) for f in v.detect_facts}
            named = (named - {None}) | set(v.layouts)
            if not named <= set(inputs):
                raise ValueError(f"layout {variant!r} names unknown tensors")
        return self

    @model_validator(mode="after")
    def _check_distributions_and_bounds(self) -> OpContract:
        ints, strs = self._scope()
        for name, spec in self.tensors().items():
            if (name in self.outputs) == bool(spec.distribution):
                raise ValueError(f"{name}: an input needs distribution, an output none")
            if name in self.outputs or spec.mutable:
                integrality = {DTYPE_TRAITS[d].is_integral for d in self.dtypes_of(name)}
                if len(integrality) > 1:
                    raise ValueError(f"{name}: a mutable tensor is floating or integer, not both")
                if bool(spec.error_bounds) == (True in integrality):
                    raise ValueError(
                        f"{name}: a floating mutable tensor needs error_bounds, an integer one none"
                    )
                if name in self.outputs:
                    continue
            elif spec.error_bounds:
                raise ValueError(f"{name}: only a mutable input has error_bounds")
            dtypes_read = {f"{name}.dtype"}
            for predicate, distribution in spec.distribution:
                # so a schema's present input has one item: ``narrow`` decides it
                if predicate and expr.parse(predicate).symbols & self.var_axes.keys():
                    raise ValueError(f"{name}: a distribution predicate reads a var axis")
                bounds = [str(getattr(distribution, f)) for f in type(distribution).expr_fields]
                for text in (*bounds, *([predicate] if predicate else [])):
                    e = expr.parse(text)
                    expr.check_scope(e, ints, strs)
                    dtypes_read |= {s for s in e.symbols if s.endswith(".dtype")}
                # a (lo, hi) kind; a one-field kind (fan_in) has no order to check
                if len(bounds) == 2:
                    lo, hi = map(distribution.constant_value, type(distribution).expr_fields)
                    if lo is not None and hi is not None and lo > hi:
                        raise ValueError(f"{name}: {distribution.kind} needs lo <= hi")
            for dtypes in self._dtype_assignments(dtypes_read):
                dtype = dtypes[f"{name}.dtype"]
                if dtype is None:  # an absent optional tensor
                    continue
                # the items selectable at ``dtypes``: a dtype-only predicate is evaluated, others both ways
                for predicate, distribution in spec.distribution:
                    if predicate is not None:
                        closure, reads = expr.compile(predicate)
                        if reads <= dtypes.keys() and not closure(dtypes):
                            continue
                    if not distribution.accepts_dtype(dtype):
                        raise ValueError(f"{name}: {distribution.kind} does not accept {dtype}")
                    if predicate is None or reads <= dtypes.keys():
                        break
        return self

    @model_validator(mode="after")
    def _check_compare(self) -> OpContract:
        # a mutable tensor's dtypes are all integer or none (_check_distributions_and_bounds)
        integer_mutable = [o for o in self.mutable_tensors if DTYPE_TRAITS[self.dtypes_of(o)[0]].is_integral]
        compare = self.compare
        if integer_mutable and compare.max_violation_fraction:
            raise ValueError("compare: an integer mutable tensor allows no violation")
        if compare.element_order != ElementOrder.FIXED and (
            len(integer_mutable) < len(self.mutable_tensors)
            or compare.determinism == DeterminismLevel.BITWISE
        ):
            raise ValueError(
                f"compare: element_order {compare.element_order.value!r} needs integer "
                "mutable tensors and bounded determinism"
            )
        return self

    @model_validator(mode="after")
    def _check_ranks(self) -> OpContract:
        if self.ranks is None:
            return self
        attr = self.attrs.get(self.ranks)
        if attr is None or not all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in attr.values
        ):
            raise ValueError(f"ranks: {self.ranks!r} is not an attr of positive ints")
        # the reference task runs on one device: it cannot call the group's baseline
        if any(b.kind == "baseline_relative" for s in self.tensors().values() for _, b in s.error_bounds):
            raise ValueError("ranks: a baseline_relative bound needs the baseline in the reference task")
        return self

    def _axis_dims(self, axis: str, name: str) -> tuple[int, ...] | None:
        layouts = [v.layout_of(name) for v in self.layout_variants.by_name.values()]
        dims = []
        for shape in self.tensors()[name].shape_cases():
            # each layout keeps a dim when it stores it unchanged
            primes = _DISTINCT_PRIME_DIMS[: len(shape)]
            stored = [layout.storage_shape(primes) for layout in layouts]
            dim = next(
                (
                    i
                    for i, d in enumerate(shape)
                    if d == axis and all(len(s) == len(shape) and s[i] == primes[i] for s in stored)
                ),
                None,
            )
            if dim is None:
                return None
            dims.append(dim)
        return tuple(dims)

    @cached_property
    def _axis_readers(self) -> dict[str, tuple[str, ...]]:
        tensors, readers = self.tensors(), {}
        for axis in (n for n, a in self.axes.items() if a.kind != "derived"):
            able = [n for n in tensors if self._axis_dims(axis, n) is not None]
            groups = (
                [n for n in able if n in self.inputs and tensors[n].present_when is None],
                [n for n in able if n in self.outputs],
                able,
            )
            readers[axis] = next((tuple(g) for g in groups if g), ())
        return readers

    def _layout_arg_names(self) -> list[str]:
        facts = (f for v in self.layout_variants.by_name.values() for f in v.detect_facts)
        names = dict.fromkeys(f.location.param_name(f.name) for f in facts)
        return [name for name in names if name is not None]

    def _dtype_root(self, tensor: str) -> str:
        dtype_spec = self.tensors()[tensor].dtype
        source = dtype_spec.same_as_tensor
        if source and (tensor in self.inputs or dtype_spec.any_of is None):
            return self._dtype_root(source)
        return tensor

    @property
    def op_package(self) -> str:
        """``"aklr.ops.gemm"`` for ``op_type`` ``"gemm"``; a plugin op's package
        under ``aklr.plugins``."""
        return op_package_of(self.op_type)

    @property
    def api_name(self) -> str:
        """``<op_package>.api.<contract_name>``; a plain property so a
        ``model_copy`` never carries a stale one."""
        return f"{self.op_package}.api.{contract_name(self.contract_id)}"

    @cached_property
    def var_axes(self) -> dict[str, AxisSpec]:
        """The axes that vary across a schema's workloads, in declaration order."""
        return {n: a for n, a in self.axes.items() if a.is_var}

    @cached_property
    def required_outputs(self) -> tuple[str, ...]:
        """The outputs carrying an axis no required input carries: the caller passes
        them, nothing allocates them."""
        readers = {t for tensors in self._axis_readers.values() for t in tensors}
        return tuple(o for o in self.outputs if o in readers)

    @cached_property
    def allocated_outputs(self) -> tuple[str, ...]:
        """The outputs dispatch allocates: those not required."""
        return tuple(o for o in self.outputs if o not in self.required_outputs)

    def outputs_by_name(self, result: Any) -> dict[str, Any]:
        """An api-shaped ``result`` (one output, else a tuple) by output name."""
        return _by_name(tuple(self.outputs), result)

    @cached_property
    def mutable_inputs(self) -> tuple[str, ...]:
        """The inputs the kernel may write in place."""
        return tuple(n for n, s in self.inputs.items() if isinstance(s, TensorSpec) and s.mutable)

    @cached_property
    def mutable_tensors(self) -> tuple[str, ...]:
        """What the kernel writes, each checked against the reference after a call: the
        outputs, then the mutable inputs; the reference returns them in this order."""
        return (*self.outputs, *self.mutable_inputs)

    def mutable_tensors_by_name(self, values: Any) -> dict[str, Any]:
        """A reference's ``values`` (one tensor, else a tuple) by name in ``mutable_tensors``."""
        return _by_name(self.mutable_tensors, values)

    def ranks_at(self, point: PointSpec) -> int:
        """How many ranks call the op together at ``point``."""
        return 1 if self.ranks is None else point.attrs[self.ranks]

    def reference_outputs(
        self,
        point: PointSpec,
        group_args: Sequence[Mapping[str, Any]],
        rank: int,
        decode: Callable,
    ) -> dict[str, Any]:
        """The reference of ``rank``'s call, its inputs read through ``decode``, by name in
        ``mutable_tensors``; ``group_args`` holds every rank's row-major args in rank order,
        each on its rank's device. A contract with ``ranks`` has a reference of ``(point,
        group_args, rank, decode)``, any other one of ``(point, row_major_args, decode)``."""
        reference_fn = getattr(op_module(self.op_package, "reference"), self.reference_name)
        if self.ranks is None:
            return self.mutable_tensors_by_name(reference_fn(point, group_args[rank], decode))
        return self.mutable_tensors_by_name(reference_fn(point, group_args, rank, decode))

    def call_mutable_tensors(self, result: Any, call: Mapping[str, Any]) -> dict[str, Any]:
        """A call's mutable tensors by name: its api ``result``'s outputs, then the mutable
        inputs as the call left them in ``call``."""
        return self.outputs_by_name(result) | {n: call[n] for n in self.mutable_inputs}

    @cached_property
    def optional_axes(self) -> dict[str, str]:
        """Each axis carried by one optional input only -> its presence attr."""
        tensors = self.tensors()
        return {
            axis: tensors[readers[0]].present_when
            for axis, readers in self._axis_readers.items()
            if tensors[readers[0]].present_when is not None
        }

    def absent_axes(self, attrs: Mapping[str, Any]) -> frozenset[str]:
        """The optional axes whose presence attr is false in ``attrs``: pinned to ``lo``
        and left out of the schema name."""
        return frozenset(a for a, p in self.optional_axes.items() if not attrs[p])

    @cached_property
    def domain(self) -> Domain:
        relations = list(self.constraints)
        relations += [f"{p} or {a} == {self.axes[a].lo}" for a, p in self.optional_axes.items()]
        allowed_values = {n: frozenset(a.values) for n, a in self.attrs.items()}
        for name, spec in self.tensors().items():
            dtypes = tuple(d.value for d in self.dtypes_of(name))
            allowed_values[f"{name}.dtype"] = frozenset(dtypes) | (
                {None} if spec.present_when is not None else set()
            )
            if (root := self._dtype_root(name)) != name:
                relations.append(f"{name}.dtype == {root}.dtype")
            if spec.present_when is not None:
                relations.append(f"({spec.present_when}) == ({name}.dtype in {dtypes!r})")
        allowed_values["layout_variant"] = frozenset(self.layout_variants.names())
        return Domain(
            contract_id=self.contract_id,
            axes={n: a for n, a in self.axes.items() if a.kind != "derived"},
            derived_axis_exprs={n: str(a.expr) for n, a in self.axes.items() if a.kind == "derived"},
            allowed_values=allowed_values,
            relations=tuple(dict.fromkeys(relations)),
        )

    @cached_property
    def api_params(self) -> tuple[ApiParam, ...]:
        """The api's params, in order: required inputs and outputs, attrs, layout args
        (defaulting to the first variant's facts), optional inputs, ``out_dtype`` if the
        output is ``out``, then the outputs not required; the kernel entry takes the subset
        ``kernel_call_keys``.
        """
        required, optional = [], []
        for name, spec in self.inputs.items():
            default = spec.default if isinstance(spec, ScalarSpec) else None
            is_required = default is None and spec.present_when is None
            p = ApiParam(name=name, is_required=is_required, default=default)
            (required if is_required else optional).append(p)
        attrs = [
            ApiParam(name=a.param_name, is_required=False, default=a.default)
            for a in self.attrs.values()
            if a.param_name is not None
        ]
        first_variant = self.layout_variants.by_name[self.layout_variants.names()[0]]
        layout_arg_defaults = {f.name: f.expected_value for f in first_variant.detect_facts}
        layout = [
            ApiParam(name=name, is_required=False, default=layout_arg_defaults[name])
            for name in self._layout_arg_names()
        ]
        outputs = [ApiParam(name=o, is_required=o in self.required_outputs) for o in self.outputs]
        out_dtype = [ApiParam(name="out_dtype", is_required=False)]
        return (
            *required,
            *(p for p in outputs if p.is_required),
            *attrs,
            *layout,
            *optional,
            *(out_dtype if "out" in self.outputs else ()),
            *(p for p in outputs if not p.is_required),
        )

    @property
    def kernel_call_keys(self) -> tuple[str, ...]:
        """The call keys the kernel entry takes, in ``api_params`` order: all but a
        schema's optional attrs at their default, and the inputs whose presence reads one."""
        hidden = {n for n, a in self.attrs.items() if a.optional and self.name}
        left_out = {self.attrs[n].param_name for n in hidden} | {
            n
            for n, s in self.inputs.items()
            if s.present_when and expr.parse(s.present_when).symbols & hidden
        }
        return tuple(p.name for p in self.api_params if p.name not in left_out)

    @property
    def reference_name(self) -> str:
        """The reference function's name in the op's ``reference.py``."""
        return f"{contract_name(self.contract_id)}_reference"

    @property
    def group_init_name(self) -> str:
        """The group set-up function's name in the op's ``group.py``."""
        return f"{contract_name(self.contract_id)}_group_init"

    def group_init(self) -> Callable | None:
        """The op's ``group.py`` function ``group_init_name``, if it has one: each rank
        of a ``ranks`` group calls it once, its torch.distributed group up, to set up
        what the op's calls take for granted (as an engine does before the first)."""
        return self._optional_function("group", self.group_init_name)

    @property
    def baseline_name(self) -> str:
        """The baseline function's name in the op's ``baseline.py``."""
        return f"{contract_name(self.contract_id)}_baseline"

    def baseline(self) -> Callable | None:
        """The op's ``baseline.py`` function ``baseline_name``, if it has one: it takes the
        api's params and returns the outputs, as the api does, and raises
        ``NotImplementedError`` for a call it does not serve."""
        return self._optional_function("baseline", self.baseline_name)

    def _optional_function(self, submodule: str, name: str) -> Callable | None:
        try:
            module = op_module(self.op_package, submodule)
        except ModuleNotFoundError as e:
            if e.name != f"{self.op_package}.{submodule}":  # the module exists but its own import failed
                raise
            return None
        return getattr(module, name, None)

    def check_point(self, point: PointSpec) -> None:
        """Raise ``ValueError`` unless ``point`` names exactly this contract's keys (no
        default is filled)."""
        detail = self.domain._key_mismatch(point)
        if detail is not None:
            raise ValueError(f"{self.contract_id}: {detail}")

    def tensors(self) -> dict[str, TensorSpec]:
        """Tensor inputs and outputs, in declaration order."""
        tensors = {n: s for n, s in self.inputs.items() if isinstance(s, TensorSpec)}
        return {**tensors, **self.outputs}

    def dtypes_of(self, tensor: str) -> tuple[DType, ...]:
        """The dtypes ``tensor`` may have, ``same_as_tensor`` expanded."""
        dtype_spec = self.tensors()[tensor].dtype
        return (
            dtype_spec.any_of if dtype_spec.any_of is not None else self.dtypes_of(dtype_spec.same_as_tensor)
        )

    def dtype_storage_shape_of(self, tensor: str, point: PointSpec) -> tuple[int, ...]:
        """The packed (``semantics.dtype_storage_shape``) shape of ``tensor``, present
        at ``point``."""
        symbols = self.domain.symbols(point)
        (shape,) = self.tensors()[tensor].shape_cases(point.attrs)
        dims = [d if isinstance(d, int) else symbols[d] for d in shape]
        return semantics.dtype_storage_shape(point.dtypes[tensor], dims)

    def storage_shape_of(self, tensor: str, point: PointSpec) -> tuple[int, ...]:
        """``dtype_storage_shape_of`` in ``tensor``'s layout at ``point``."""
        layout = self.layout_variants.by_name[point.layout_variant_name].layout_of(tensor)
        return layout.storage_shape(self.dtype_storage_shape_of(tensor, point))

    def name_at(self, point: PointSpec) -> str:
        """The schema name for ``point``: the contract name, each attr's part, root
        tensor dtype and layout variant off its first declared value, and
        ``<axis><value>`` per present const int axis, ``_``-joined."""
        assert self.name is None, "a schema has one value per attr: name a contract"
        parts = [contract_name(self.contract_id)]
        parts += (a.in_name(n, point.attrs[n]) for n, a in self.attrs.items())
        for name, spec in self.tensors().items():
            dtype, source = point.dtypes[name], spec.dtype.same_as_tensor
            if dtype is None or self._dtype_root(name) != name:
                continue  # absent, or another tensor's dtype
            first = point.dtypes[source] if source else self.dtypes_of(name)[0].value
            parts.append("" if dtype == first else f"{name}_{dtype}")
        if point.layout_variant_name != self.layout_variants.names()[0]:
            parts.append(point.layout_variant_name)
        absent = self.absent_axes(point.attrs)
        parts += (
            f"{name}{point.axes[name]}"
            for name, a in self.axes.items()
            if a.kind == "int" and not a.is_var and name not in absent
        )
        return "_".join(p for p in parts if p)

    def narrow(
        self,
        point: PointSpec,
        *,
        workloads: Sequence[WorkloadSpec | Mapping[str, Any]],
        kernel: KernelSpec | Mapping[str, Any],
        models: Sequence[str],
    ) -> OpContract:
        """The schema serving ``point`` (its var axes not read): every other axis, attr,
        dtype and layout variant pinned to ``point``'s, with ``workloads`` (stored
        sorted by axes), ``kernel`` and ``models``."""
        if not self.domain.contains(point):
            raise ValueError(f"{point.key()} is outside the domain of {self.contract_id}")
        symbols = self.domain.symbols(point)
        absent = self.absent_axes(point.attrs)
        var = {n for n in self.var_axes if n not in absent}
        pinned = {k: v for k, v in symbols.items() if k not in var}
        axes = {}
        for name, a in self.axes.items():
            if name in var:
                axes[name] = a
            elif a.kind == "int":
                value = a.lo if name in absent else point.axes[name]
                update = {"lo": value, "hi": value, "is_var": False, "any_size": False}
                axes[name] = a.model_copy(update=update)
            else:
                axes[name] = a.model_copy(update={"expr": symbols[name]})
        # a schema's attr is its one value; the default is the registered contract's
        attrs = {
            name: a.model_copy(
                update={
                    "values": (point.attrs[name],),
                    "default": None,
                    "optional": a.optional
                    and attr_membership_key(point.attrs[name]) == attr_membership_key(a.default),
                }
            )
            for name, a in self.attrs.items()
        }
        variant = point.layout_variant_name
        fields = {f: getattr(self, f) for f in type(self).model_fields}
        schema = OpContract.model_validate(
            {
                **fields,
                "name": self.name_at(point),
                "models": models,
                "workloads": workloads,
                "axes": axes,
                "attrs": attrs,
                "constraints": tuple(t for t in self.constraints if expr.parse(t).symbols & var),
                "inputs": {
                    n: (s.narrow(point.attrs, point.dtypes[n], pinned) if isinstance(s, TensorSpec) else s)
                    for n, s in self.inputs.items()
                },
                "outputs": {
                    n: s.narrow(point.attrs, point.dtypes[n], pinned) for n, s in self.outputs.items()
                },
                "layout_variants": LayoutVariantsSpec(
                    by_name={variant: self.layout_variants.by_name[variant]}
                ),
                "kernel": kernel,
            }
        )
        return schema.model_copy(
            update={"workloads": tuple(sorted(schema.workloads, key=lambda w: tuple(w.axes.items())))}
        )

    def point(self, var_axes: Mapping[str, int]) -> PointSpec:
        """The Point a schema serves at ``var_axes`` (exactly its var axes, else
        ``ValueError``); the domain is not checked."""
        if list(var_axes) != list(self.var_axes):
            raise ValueError(f"axes {list(var_axes)} are not {list(self.var_axes)}")
        axes = {n: var_axes[n] if a.is_var else a.lo for n, a in self.axes.items() if a.kind == "int"}
        attrs = {n: a.values[0] for n, a in self.attrs.items()}
        layout_variant_name = self.layout_variants.names()[0]
        tensors = self.tensors()
        symbols = {
            **axes,
            **{n: a.expr for n, a in self.axes.items() if a.kind == "derived"},
            **attrs,
            "layout_variant": layout_variant_name,
            **{f"{n}.dtype": s.dtype.any_of[0].value for n, s in tensors.items() if s.present_when is None},
        }
        dtypes = {
            n: (
                s.dtype.any_of[0].value
                if s.present_when is None or expr.compile(s.present_when)[0](symbols)
                else None
            )
            for n, s in tensors.items()
        }
        return PointSpec(
            contract_id=self.contract_id,
            axes=axes,
            dtypes=dtypes,
            attrs=attrs,
            layout_variant_name=layout_variant_name,
        )

    def point_with_contract_id(self, var_axes: Mapping[str, int], contract_id: ContractId) -> PointSpec:
        """``point(var_axes)`` labelled with ``contract_id`` (another version's)."""
        return self.point(var_axes).model_copy(update={"contract_id": contract_id})

    def point_from_json(self, data: Mapping[str, Any], var_axes: Mapping[str, int]) -> PointSpec:
        """The Point a schema file's json ``data`` pins at ``var_axes`` (an unwritten
        axis is absent: ``lo``; an unwritten tensor ``None``)."""
        try:
            attrs = {
                n: (a.default if a.optional and n not in data["attrs"] else data["attrs"][n])
                for n, a in self.attrs.items()
            }
            written = data["axes"]
            axes = {
                name: (
                    var_axes[name]
                    if name in var_axes
                    else written[name]["value"]
                    if name in written
                    else a.lo
                )
                for name, a in self.domain.axes.items()
            }
            tensors = {**data["inputs"], **data["outputs"]}
            dtypes = {name: tensors[name]["dtype"] if name in tensors else None for name in self.tensors()}
            (layout_variant_name,) = data["layout_variants"]["by_name"]
            return PointSpec(
                contract_id=self.contract_id,
                axes=axes,
                dtypes=dtypes,
                attrs=attrs,
                layout_variant_name=layout_variant_name,
            )
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
            raise ValueError(f"not a schema of {self.contract_id}: {e!r}") from None

    def relative_path(self) -> PurePosixPath:
        """A schema file's path in its op directory; the uuid hashes the authored
        content."""
        uuid = schema_uuid(
            self.name,
            self.authored_parts(self.model_dump(mode="json", include={"workloads", "kernel"})),
        )
        return schema_relative_path(self.name, self.kernel.language, uuid)

    def _present_inputs(self, point: PointSpec, symbols: Mapping[str, Any]) -> dict[str, ArgSpec]:
        return {
            name: spec
            for name, spec in self.inputs.items()
            if (
                point.dtypes[name] is not None
                if isinstance(spec, TensorSpec)
                else spec.present_when is None or expr.compile(spec.present_when)[0](symbols)
            )
        }

    def _present_scalars(self, point: PointSpec) -> dict[str, ScalarSpec]:
        present = self._present_inputs(point, self.domain.symbols(point))
        return {n: s for n, s in present.items() if isinstance(s, ScalarSpec)}

    def distributions_at(self, point: PointSpec) -> dict[str, dict[str, Any]]:
        """Each present tensor input's ``distribution`` item at ``point``, evaluated, as
        json."""
        symbols = self.domain.symbols(point)
        return {
            name: evaluated_json(spec.distribution_for(symbols), symbols)
            for name, spec in self._present_inputs(point, symbols).items()
            if isinstance(spec, TensorSpec)
        }

    def workload_for(self, point: PointSpec, scalars: Mapping[str, Any]) -> WorkloadSpec:
        """The workload at ``point``: its present var axes and every present scalar
        (``scalars``'s value, else the default), normalized. Raises ``ValueError``."""
        if not self.domain.contains(point):
            raise ValueError("outside the domain")
        present = self._present_scalars(point)
        if unknown := sorted(scalars.keys() - present.keys()):
            raise ValueError(f"{unknown} are not scalar inputs of this schema")
        normalized = {
            name: spec.normalize(
                scalars[name] if name in scalars else semantics.default_scalar(spec.model_dump(mode="json"))
            )
            for name, spec in present.items()
        }
        absent = self.absent_axes(point.attrs)
        axes = {a: point.axes[a] for a in self.var_axes if a not in absent}
        return WorkloadSpec.model_validate({"axes": axes, "scalars": normalized})

    def schema_text(self) -> str:
        """The text of a schema's file: present axes, attrs, inputs and outputs at their
        values, the rest as dumped."""
        point = self.point(self.workloads[0].axes)
        symbols = self.domain.symbols(point)
        absent = self.absent_axes(point.attrs)

        def item(name: str, spec: ArgSpec) -> dict[str, Any]:
            is_tensor = isinstance(spec, TensorSpec)
            text = {
                "shape": list(spec.shape) if is_tensor else None,
                "dtype": point.dtypes[name] if is_tensor else spec.dtype.value,
            }
            if is_tensor and spec.distribution:
                text["distribution"] = spec.distribution[0][1].model_dump(mode="json")
            return text | {"description": spec.description}

        return expr.json_file_text(
            {
                "schema_version": CONTRACT_VERSION,
                "name": self.name,
                "contract_id": self.contract_id,
                "description": self.description,
                "op_type": self.op_type,
                "models": list(self.models),
                "axes": {
                    name: ({"type": "var"} if a.is_var else {"type": "const", "value": symbols[name]})
                    | {"description": a.description}
                    for name, a in self.axes.items()
                    if name not in absent
                },
                "attrs": {n: a.values[0] for n, a in self.attrs.items() if not a.optional},
                "inputs": {
                    name: item(name, spec) for name, spec in self._present_inputs(point, symbols).items()
                },
                "outputs": {name: item(name, spec) for name, spec in self.outputs.items()},
                **self.model_dump(
                    mode="json",
                    include={"workloads", "layout_variants", "compare", "kernel"},
                ),
            }
        )

    @staticmethod
    def error_lines(where: str, parsed: Any, error: ValidationError) -> list[str]:
        """One line per item of ``error``: ``<where>:<path>:<line>: <reason>`` for a
        line of a ``kernel.sources`` file of the validated json ``parsed``, else
        ``<where>: <dotted loc>: <reason>``."""
        lines = []
        for detail in error.errors():
            loc = detail["loc"]
            line_number, reason = source_error_parts(detail)
            if loc[:2] == ("kernel", "sources") and line_number is not None:
                try:
                    path = parsed["kernel"]["sources"][loc[2]]["path"]
                except (KeyError, IndexError, TypeError):
                    path = f"kernel.sources[{loc[2]}]"
                lines.append(f"{where}:{path}:{line_number}: {reason}")
            else:
                at = f"{where}: {'.'.join(map(str, loc))}" if loc else where
                lines.append(f"{at}: {reason}")
        return lines

    @staticmethod
    def authored_parts(schema: Mapping[str, Any]) -> dict[str, Any]:
        """The kernel and each workload's axes and scalars of a schema's raw json,
        workloads sorted for comparison only."""
        workloads = []
        for w in OpContract._raw_workloads(schema):
            scalars = w.get("scalars", {})
            if not isinstance(scalars, Mapping):
                raise ValueError(  # noqa: TRY004
                    f"workload scalars {scalars!r} are not a mapping"
                )
            workloads.append({"axes": w["axes"], "scalars": dict(scalars)})
        return {
            "kernel": schema.get("kernel"),
            "workloads": sorted(workloads, key=sorted_key_json),
        }

    @staticmethod
    def measured_parts(schema: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Each workload's written ``speedup`` and ``accuracy`` by ``axes_text``, from a
        schema's raw json. ``ValueError`` as ``authored_parts``."""
        return {
            axes_text(w["axes"]): {k: w[k] for k in ("speedup", "accuracy") if k in w}
            for w in OpContract._raw_workloads(schema)
        }

    @staticmethod
    def _raw_workloads(schema: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        raw_workloads = schema.get("workloads", [])
        if not isinstance(raw_workloads, list):
            raise ValueError(  # noqa: TRY004
                f"workloads {raw_workloads!r} are not a list"
            )
        for w in raw_workloads:
            axes = w.get("axes") if isinstance(w, Mapping) else None
            if not isinstance(axes, Mapping) or not all(
                isinstance(k, str) and type(v) is int for k, v in axes.items()
            ):
                raise ValueError(f"workload axes {axes!r} are not names to ints")
        return raw_workloads

    def with_measurements(self, measurements: Mapping[str, WorkloadSpec]) -> OpContract:
        """A copy with each workload's ``speedup`` and ``accuracy`` from
        ``measurements`` by ``axes_text``; ``ValueError`` on a key mismatch or an
        invalid measurement."""
        keys = [axes_text(w.axes) for w in self.workloads]
        if set(keys) != set(measurements):
            raise ValueError(
                f"measurements of {sorted(set(keys) ^ set(measurements))} do not match the workloads"
            )
        workloads, lines = [], []
        for w, k in zip(self.workloads, keys, strict=True):
            m = measurements[k]
            try:
                workloads.append(
                    WorkloadSpec.model_validate(
                        {**w.model_dump(), "speedup": m.speedup, "accuracy": m.accuracy}
                    )
                )
            except ValidationError as e:
                lines += OpContract.error_lines(f"workload {{{k}}}", {}, e)
        if lines:
            raise ValueError("\n".join(lines))
        return self.model_copy(update={"workloads": tuple(workloads)})

    def rank_key(self, workload: WorkloadSpec) -> tuple[bool, float, str]:
        """A ``max`` key among candidates for a Point: failed last, then speedup
        (unmeasured lowest), then ``kernel.content_hash``."""
        speedup = -math.inf if workload.speedup is None else workload.speedup
        return (workload.accuracy != "failed", speedup, self.kernel.content_hash)

    def schema_errors(self, path: Path, schema: OpContract) -> list[str]:
        """The errors of ``schema`` at ``path`` against this registered contract, one
        line each; never raises."""
        errors = []
        workloads = schema.workloads
        location = PurePosixPath(self.op_type) / schema.relative_path()
        if PurePosixPath(path).parts[-len(location.parts) :] != location.parts:  # E1
            errors.append(f"{path}: not at <ops>/{location}")

        in_domain = []
        for w in workloads:  # E5
            at = f"{path}: workload {{{axes_text(w.axes)}}}"
            try:
                point = schema.point(w.axes)
            except ValueError as e:
                errors.append(f"{at}: cannot build a point: {e}")
                continue
            if not self.domain.contains(point):
                errors.append(f"{at}: outside the domain of {self.contract_id}")
                continue
            in_domain.append((at, w, point))

        for at, w, point in in_domain:  # E7
            scalars = schema._present_scalars(point)
            if list(w.scalars) != list(scalars):
                errors.append(f"{at}: scalars {list(w.scalars)} are not {list(scalars)}")
                continue
            for name, value in w.scalars.items():
                if (reason := scalars[name].value_error(value)) is not None:
                    errors.append(f"{at}: {name}: {reason}")

        items = [tuple(w.axes.items()) for w in workloads]  # E8
        if (repeated := next((i for i in items if items.count(i) > 1), None)) is not None:
            errors.append(f"{path}: workload {{{axes_text(dict(repeated))}}} repeats")

        measured = [w.accuracy is not None for w in workloads]  # E9
        if any(measured):
            if not all(measured):
                errors.append(f"{path}: some workloads have an accuracy, some not")
            targets = schema.kernel.targets
            if len(targets) != 1 or not targets[0].is_complete:
                errors.append(f"{path}: measurements need one kernel target with arch and hardware")
        return errors

    def _dtype_assignments(
        self, symbols: Iterable[str]
    ) -> Iterator[
        dict[str, str | None]
    ]:  # structure: allow S4 generator; inlining pushes _check_distributions_and_bounds past 60 lines
        tensors = [s.removesuffix(".dtype") for s in symbols]
        root_of = {t: self._dtype_root(t) for t in tensors}
        roots = list(dict.fromkeys(root_of.values()))
        choices = [
            [d for d in self.dtypes_of(r) if all(d in self.dtypes_of(t) for t in tensors if root_of[t] == r)]
            for r in roots
        ]
        optional = [t for t in tensors if self.tensors()[t].present_when is not None]
        for chosen in itertools.product(*choices):
            root_dtype = dict(zip(roots, chosen))
            for absent in itertools.product((False, True), repeat=len(optional)):
                gone = set(itertools.compress(optional, absent))
                yield {f"{t}.dtype": None if t in gone else root_dtype[root_of[t]].value for t in tensors}

    def check_api_signature(self, fn: Callable) -> None:
        """Raise ``TypeError`` unless ``fn`` is this contract's api function: named
        params, every required one without a non-``None`` default, extra ones
        defaulted, named and placed at ``api_name``, a body of exactly ``...``."""
        params = inspect.signature(fn).parameters
        named = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        unnamed = [p.name for p in params.values() if p.kind not in named]
        if unnamed:
            raise TypeError(f"{fn.__name__}: no positional-only or *args/**kwargs, got {unnamed}")
        for p in self.api_params:
            if not p.is_required:
                continue
            if p.name not in params:
                raise TypeError(f"{fn.__name__}: missing {p.name!r}, required by {self.contract_id}")
            default = params[p.name].default
            if default is not inspect.Parameter.empty and default is not None:
                raise TypeError(f"{fn.__name__}: required {p.name!r} may default only to None")
        known = {p.name for p in self.api_params}
        undefaulted = [
            name for name, p in params.items() if name not in known and p.default is inspect.Parameter.empty
        ]
        if undefaulted:
            raise TypeError(f"{fn.__name__}: {undefaulted} not in {self.contract_id} need a default")
        if fn.__name__ != contract_name(self.contract_id):
            raise TypeError(f"{fn.__name__} must be named {contract_name(self.contract_id)}")
        if api_name(fn) != self.api_name:
            raise TypeError(f"{api_name(fn)} != {self.api_name}")
        (node,) = ast.parse(textwrap.dedent(inspect.getsource(fn))).body
        if ast.dump(ast.Module(body=node.body, type_ignores=[])) != _ELLIPSIS_BODY:
            raise TypeError(f"{fn.__name__}: the body must be exactly `...`")


# Relations see an absent optional tensor's ``<t>.dtype`` as ``None``, so presence is a relation.
class Domain(FrozenModel):
    contract_id: ContractId
    axes: dict[str, AxisSpec]  # int axes
    derived_axis_exprs: dict[str, str] = {}  # noqa: RUF012 -- pydantic copies defaults
    # attr names, "<t>.dtype", "layout_variant"
    allowed_values: dict[str, frozenset[AttrValue | None]]
    relations: tuple[str, ...] = ()

    def contains(self, point: PointSpec) -> bool:
        """``point`` has exactly the domain's keys and lies in it; never raises."""
        if self._key_mismatch(point) is not None:
            return False
        dtypes = {f"{t}.dtype": d for t, d in point.dtypes.items()}
        symbols = {**point.attrs, **dtypes, "layout_variant": point.layout_variant_name}
        for name, v in point.axes.items():
            a = self.axes[name]
            if not a.lo <= v <= a.hi or v % a.multiple_of:
                return False
        for name, v in symbols.items():
            if attr_membership_key(v) not in {attr_membership_key(m) for m in self.allowed_values[name]}:
                return False
        try:
            symbols = {**self.symbols(point), **dtypes}  # absent dtypes bound to None
            return all(expr.compile(r)[0](symbols) for r in self.relations)
        except (expr.ExprError, ArithmeticError, TypeError):
            return False

    def symbols(self, point: PointSpec) -> dict[str, int | str | bool]:
        """``point.symbols()`` plus the derived axes, in declaration order."""
        symbols = point.symbols()
        for name, text in self.derived_axis_exprs.items():
            symbols[name] = expr.compile(text)[0](symbols)
        return symbols

    def _key_mismatch(self, point: PointSpec) -> str | None:
        dtypes = {n.removesuffix(".dtype") for n in self.allowed_values if n.endswith(".dtype")}
        attrs = {n for n in self.allowed_values if "." not in n and n != "layout_variant"}
        if point.contract_id != self.contract_id:
            return f"point is of {point.contract_id!r}"
        if point.axes.keys() != self.axes.keys():
            return f"axes {sorted(point.axes)} are not {sorted(self.axes)}"
        if point.dtypes.keys() != dtypes:
            return f"dtypes {sorted(point.dtypes)} are not {sorted(dtypes)}"
        if point.attrs.keys() != attrs:
            return f"attrs {sorted(point.attrs)} are not {sorted(attrs)}"
        if point.layout_variant_name not in self.allowed_values["layout_variant"]:
            return f"unknown layout variant {point.layout_variant_name!r}"
        return None


# An api function is its signature only: no ``*args``/``**kwargs``, a body of exactly ``...``.
_ELLIPSIS_BODY = ast.dump(ast.parse("..."))


class ApiParam(FrozenModel):
    name: str
    is_required: bool
    default: Any = None


def _by_name(names: tuple[str, ...], values: Any) -> dict[str, Any]:
    """``values`` (one, else a tuple) by ``names``, in order."""
    return dict(zip(names, (values,) if len(names) == 1 else values, strict=True))


def api_name(fn: Callable) -> str:
    """Dotted ``module.qualname`` of a function, which is what a route is keyed by."""
    module = getattr(fn, "__module__", "") or ""
    qualname = getattr(fn, "__qualname__", "") or ""
    return f"{module}.{qualname}" if module else qualname
