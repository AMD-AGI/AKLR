###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer) and
# FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from functools import cached_property
from typing import Annotated, Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from pydantic import Field, StringConstraints, TypeAdapter, model_validator

from aklr.core.platform.device import DeviceTarget
from aklr.core.schema._args import Const, Scalar, Tensor, Var
from aklr.core.schema._frozen_model import NonEmptyStr, frozen_dataclass
from aklr.core.schema._kernel import Kernel
from aklr.core.schema._standard import Standard
from aklr.core.schema._workload import Workload
from aklr.core.utils import torch_dtype_to_dtype_str


def _get_tensor(
    kwargs: Dict[str, Any],
    param: str,
    tuple_idx: Optional[int] = None,
) -> Optional[torch.Tensor]:
    val = kwargs.get(param)
    if val is None:
        return None
    if tuple_idx is not None:
        if isinstance(val, (tuple, list)) and len(val) > tuple_idx:
            val = val[tuple_idx]
        else:
            return None
    return val if isinstance(val, torch.Tensor) else None


def extract_axes(
    extractors: dict[str, Callable[[dict[str, Any]], int | None]],
    call: dict[str, Any],
) -> dict[str, int]:
    """Each axis's value from its extractor on call, left out if the extractor raises or returns None."""
    axes: dict[str, int] = {}
    for axis, extract in extractors.items():
        try:
            value = extract(call)
        except Exception:  # noqa: BLE001
            value = None
        if value is not None:
            axes[axis] = int(value)
    return axes


@frozen_dataclass
class Schema:
    """An op's schema: its contract, at most one kernel and the workloads measured on it, kept as json. One
    declared in code, a variant of an op api, also holds its test standard; the json leaves it out."""

    # first in the json dump; its "" default, never validated, is filled in by _default_name
    name: NonEmptyStr = Field(default="", kw_only=True)
    op_type: NonEmptyStr
    axes: Dict[NonEmptyStr, Union[Var, Const]]
    # at most one item, {variant name: condition}: the calls of the op that are this variant's, as a Python
    # expression over the api's arguments and torch; empty is the default variant, of every call
    variant: Dict[Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(_[a-z0-9]+)*$")], NonEmptyStr] = (
        Field(default_factory=dict)
    )

    workloads: List[Workload] = Field(default_factory=list)
    inputs: Dict[NonEmptyStr, Union[Tensor, Scalar]] = Field(kw_only=True)
    outputs: Dict[NonEmptyStr, Union[Tensor, Scalar]] = Field(kw_only=True)
    standard: Optional[Standard] = Field(default=None, exclude=True, repr=False, kw_only=True)
    # one "<model>/tp<n>[+dp<n>][+dcp<n>][+ep<n>]" per deployment calling it, e.g. "glm-5.2/tp8+dp8"
    tags: List[NonEmptyStr] = Field(default_factory=list)
    description: Optional[str] = None
    constraints: List[NonEmptyStr] = Field(default_factory=list)

    kernel: Optional[Kernel] = Field(default=None)

    @model_validator(mode="after")
    def _default_name(self) -> Schema:
        if not self.name:
            object.__setattr__(self, "name", self.schema_name(self.const_axes))
        return self

    @model_validator(mode="after")
    def _validate_input_output_names(self) -> Schema:
        if set(self.inputs.keys()) & set(self.outputs.keys()):
            raise ValueError("Input and output names must not overlap")
        return self

    @model_validator(mode="after")
    def _validate_expressions(self) -> Schema:
        if len(self.variant) > 1:
            raise ValueError(f"a schema is one variant, not {list(self.variant)}")
        for expression in (*self.constraints, *self.variant.values()):
            try:
                ast.parse(expression, mode="eval")
            except SyntaxError as e:
                raise ValueError(
                    f"Constraints and variant conditions must be valid Python expressions: {e}"
                ) from e
        return self

    @property
    def variant_name(self) -> str:
        return next(iter(self.variant), "default")

    @property
    def op_name(self) -> str:
        """Its op's variant: ``gemm_bf16_nt.shuffled``."""
        return f"{self.op_type}.{self.variant_name}"

    @cached_property
    def _variant_condition(self) -> Any:
        return (
            compile(next(iter(self.variant.values())), f"<{self.op_name}>", "eval") if self.variant else None
        )

    def matches(self, call: Dict[str, Any]) -> bool:
        """Whether call, the api's bound arguments, is one of this variant's."""
        condition = self._variant_condition
        return condition is None or bool(eval(condition, {"torch": torch}, call))  # noqa: S307 -- template code

    def ordered_inputs(self, call: Dict[str, Any]) -> List[Any]:
        """The call's input values in the schema's order, None where absent: kernels take them positionally."""
        values: List[Any] = []
        for key, spec in self.inputs.items():
            value = call.get(spec.param or key)
            index = getattr(spec, "tuple_idx", None)
            if index is not None and value is not None:
                value = value[index] if isinstance(value, (tuple, list)) and len(value) > index else None
            values.append(value)
        return values

    def serves(self, workload: Workload, target: DeviceTarget, min_speedup: float, baseline: str) -> bool:
        """Whether dispatch may serve workload by this schema's kernel on target, see Workload.qualifies."""
        return (
            self.kernel is not None
            and self.kernel.build.runs_on(target)
            and workload.qualifies(target, min_speedup, baseline)
        )

    @model_validator(mode="after")
    def _validate_tensor_axis_references(self) -> Schema:
        all_tensors = {**self.inputs, **self.outputs}

        for tensor_name, tensor in all_tensors.items():
            if isinstance(tensor, Tensor):
                for axis_name in tensor.dim_names:
                    if axis_name not in self.axes:
                        tensor_type = "input" if tensor_name in self.inputs else "output"
                        raise ValueError(
                            f'{tensor_type.capitalize()} "{tensor_name}" references undefined '
                            f'axis "{axis_name}"'
                        )
        return self

    @model_validator(mode="after")
    def _validate_workload_axes(self) -> Schema:
        # an axis left out is not an error here: tracing records a call's axes off its inputs alone
        var_axes = set(self.var_axes)
        for index, workload in enumerate(self.workloads):
            if extra := workload.axes.keys() - var_axes:
                raise ValueError(f"workloads[{index}] contains non-variable axes {sorted(extra)}")
        return self

    @cached_property
    def const_axes(self) -> Dict[str, int]:
        return {
            name: axis.value
            for name, axis in self.axes.items()
            if isinstance(axis, Const) and axis.value is not None
        }

    @cached_property
    def var_axes(self) -> List[str]:
        return [name for name, axis in self.axes.items() if isinstance(axis, Var)]

    @cached_property
    def uuid(self) -> str:
        """16 hex of the json's sha256, workloads excluded so that recording one keeps the file name."""
        key = json.dumps(
            TypeAdapter(Schema).dump_python(self, mode="json", exclude={"workloads"}),
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @cached_property
    def var_axes_bindings(self) -> Dict[str, Tuple[str, int]]:
        bindings: Dict[str, Tuple[str, int]] = {}
        for inp_name, tensor in self.inputs.items():
            if isinstance(tensor, Scalar):
                continue
            for dim_idx, axis in enumerate(tensor.dim_names):
                ax_def = self.axes.get(axis)
                if isinstance(ax_def, Var) and axis not in bindings:
                    bindings[axis] = (inp_name, dim_idx)
        return bindings

    def get_axes_values(self, input_shapes: Iterable[Optional[Tuple[int, ...]]]) -> Dict[str, int]:
        var_axes_values: Dict[str, int] = {}
        for (inp_name, inp), inp_shape in zip(self.inputs.items(), input_shapes):
            if isinstance(inp, Scalar) or inp_shape is None:
                continue
            if len(inp.dim_names) != len(inp_shape):
                raise ValueError(
                    f"Input '{inp_name}''s defined dimension is {len(inp.dim_names)} but the "
                    f"actual dimension is {len(inp_shape)}"
                )
            for axis_name, axis_value in zip(inp.dim_names, inp_shape):
                if axis_name in self.axes and self.axes[axis_name].type == "var":
                    if axis_name in var_axes_values:
                        if var_axes_values[axis_name] != axis_value:
                            raise ValueError(
                                f"Axis '{axis_name}' has different values for different input "
                                f"tensors: {var_axes_values[axis_name]} and {axis_value}"
                            )
                    else:
                        var_axes_values[axis_name] = axis_value
        return var_axes_values

    def get_axes_values_from_inputs(self, inputs: Iterable[Any]) -> Dict[str, int]:
        """Var axis values from the shapes of input values in schema input order; raises if two disagree."""
        shapes = [tuple(val.shape) if hasattr(val, "shape") else None for val in inputs]
        return self.get_axes_values(shapes)

    def _get_shapes(
        self, tensors: Iterable[Tensor], var_axes_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
        var_axes_values = var_axes_values or {}
        shapes = []

        for tensor in tensors:
            if isinstance(tensor, Scalar):
                shapes.append(None)
                continue
            shape = []
            for axis_name in tensor.dim_names:
                axis = self.axes[axis_name]
                if isinstance(axis, Const):
                    shape.append(axis.value)
                elif isinstance(axis, Var):
                    if axis_name not in var_axes_values:
                        raise ValueError(f"Missing value for variable axis '{axis_name}'")
                    shape.append(var_axes_values[axis_name])
            shapes.append(shape)

        return shapes

    def get_input_shapes(
        self, var_axes_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
        return self._get_shapes(self.inputs.values(), var_axes_values)

    def get_output_shapes(
        self, var_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
        return self._get_shapes(self.outputs.values(), var_values)

    def merge_kwargs_to_args(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
        if not kwargs:
            return args

        param_names = list(self.inputs.keys()) + list(self.outputs.keys())

        if len(args) > len(param_names):
            raise TypeError(
                f"Too many positional arguments: got {len(args)}, expected at most {len(param_names)}"
            )

        positional_arg_names = set(param_names[: len(args)])
        for name in kwargs:
            if name in positional_arg_names:
                raise TypeError(f"Got multiple values for argument '{name}'")

        result = list(args)
        for i in range(len(args), len(param_names)):
            name = param_names[i]
            if name in kwargs:
                result.append(kwargs[name])
            else:
                break
        return tuple(result)

    def _build_axis_extractors(  # structure: allow S4 the upstream builder, kept whole
        self,
    ) -> Dict[str, Callable[[Dict[str, Any]], Optional[int]]]:
        extractors: Dict[str, Callable[[Dict[str, Any]], Optional[int]]] = {}

        def _make_tensor_extractor(
            p: str, ti: Optional[int], di: int
        ) -> Callable[[Dict[str, Any]], Optional[int]]:
            def extractor(kw: Dict[str, Any]) -> Optional[int]:
                t = _get_tensor(kw, p, ti)
                if t is None or di >= t.ndim:
                    return None
                return int(t.shape[di])

            return extractor

        def _make_scalar_extractor(
            name: str,
        ) -> Callable[[Dict[str, Any]], Optional[int]]:
            def extractor(kw: Dict[str, Any]) -> Optional[int]:
                val = kw.get(name)
                if val is None:
                    return None
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return None

            return extractor

        def _make_fixed_extractor(
            value: int,
        ) -> Callable[[Dict[str, Any]], Optional[int]]:
            def extractor(_kw: Dict[str, Any]) -> Optional[int]:
                return value

            return extractor

        def _tensor_source(
            axis_name: str, allow_optional: bool
        ) -> Optional[Callable[[Dict[str, Any]], Optional[int]]]:
            for json_key, descriptor in self.inputs.items():
                if not isinstance(descriptor, Tensor):
                    continue
                if descriptor.optional and not allow_optional:
                    continue
                if axis_name not in descriptor.dim_names:
                    continue
                param = descriptor.param if descriptor.param is not None else json_key
                return _make_tensor_extractor(
                    param, descriptor.tuple_idx, descriptor.dim_names.index(axis_name)
                )
            return None

        for axis_name in self.axes:
            marker = self.axes[axis_name]
            extractor: Optional[Callable[[Dict[str, Any]], Optional[int]]]
            if isinstance(marker, Const) and marker.value is not None:
                extractor = _make_fixed_extractor(marker.value)
            elif isinstance(marker, Const) and marker.extract is not None:
                extractor = marker.extract
            else:
                extractor = None
            if extractor is None:
                extractor = _tensor_source(axis_name, allow_optional=False)
            if extractor is None:
                scalar_desc = self.inputs.get(axis_name)
                if isinstance(scalar_desc, Scalar):
                    param = scalar_desc.param if scalar_desc.param is not None else axis_name
                    extractor = _make_scalar_extractor(param)
            # optional tensors come after scalars, as a call may omit them and leave a Const axis unset
            if extractor is None:
                extractor = _tensor_source(axis_name, allow_optional=True)
            if extractor is None:
                extractor = _make_scalar_extractor(axis_name)
            extractors[axis_name] = extractor

        return extractors

    @cached_property
    def const_axis_extractors(
        self,
    ) -> dict[str, Callable[[dict[str, Any]], int | None]]:
        return {
            axis: extract
            for axis, extract in self._build_axis_extractors().items()
            if isinstance(self.axes[axis], Const)
        }

    def schema_name(self, axis_values: Dict[str, int]) -> str:
        """Its name at these Const axis values; dispatch and for_call both name by it, so they agree."""
        prefix = self.op_type if self.variant_name == "default" else f"{self.op_type}_{self.variant_name}"
        parts = []
        for n, marker in self.axes.items():
            if not isinstance(marker, Const) or n not in axis_values:
                continue
            pfx = marker.abbrev if marker.abbrev is not None else n
            if pfx == "":
                continue
            parts.append(f"{pfx}{axis_values[n]}")
        return prefix + ("_" + "_".join(parts) if parts else "")

    def call_dtypes(self, call: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """The dtype call gives each tensor input and output, None where neither it nor the schema names one."""

        def call_dtype(key: str, tensor: Tensor) -> Optional[str]:
            t = _get_tensor(call, tensor.param or key, tensor.tuple_idx)
            return torch_dtype_to_dtype_str(t.dtype) if t is not None else None

        def input_dtype(key: str, tensor: Tensor) -> Optional[str]:
            dtype = call_dtype(key, tensor)
            if dtype is None and tensor.optional:
                dtype = tensor.dtype
            return dtype

        def output_dtype(tensor: Tensor) -> Optional[str]:
            if tensor.dtype_from is not None:
                t = _get_tensor(call, tensor.dtype_from)
                scalar = call.get(tensor.dtype_from_scalar)
                if t is not None:
                    return torch_dtype_to_dtype_str(t.dtype)
                if scalar is not None:
                    return torch_dtype_to_dtype_str(scalar)
                return tensor.dtype
            if tensor.dtype is not None:
                return tensor.dtype
            for key, inp in self.inputs.items():
                if isinstance(inp, Tensor) and set(inp.dim_names) & set(tensor.dim_names):
                    dtype = call_dtype(key, inp)
                    if dtype is not None:
                        return dtype
            return None

        dtypes = {key: input_dtype(key, inp) for key, inp in self.inputs.items() if isinstance(inp, Tensor)}
        return dtypes | {
            key: output_dtype(out) for key, out in self.outputs.items() if isinstance(out, Tensor)
        }

    def for_call(self, call: Dict[str, Any]) -> Schema:
        """This schema at call: Const axes, name and tensor dtypes read from the call's arguments."""
        values = extract_axes(self.const_axis_extractors, call)
        dtypes = self.call_dtypes(call)
        return replace(
            self,
            name=self.schema_name(values),
            axes={
                axis: (
                    replace(marker, value=values[axis])
                    if isinstance(marker, Const) and axis in values
                    else marker
                )
                for axis, marker in self.axes.items()
                if not (isinstance(marker, Const) and marker.extract is not None and axis not in values)
            },
            inputs={
                key: (replace(inp, dtype=dtypes[key]) if isinstance(inp, Tensor) else inp)
                for key, inp in self.inputs.items()
            },
            outputs={
                key: (replace(out, dtype=dtypes[key]) if isinstance(out, Tensor) else out)
                for key, out in self.outputs.items()
            },
        )
