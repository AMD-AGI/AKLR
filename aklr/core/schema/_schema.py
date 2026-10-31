# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Schema and associated classes for the fi_trace system.

Design
------
A :class:`Schema` describes the schema of a FlashInfer operation
independently from any specific Python function.  Templates live in
``flashinfer/trace/templates/`` and are referenced by the
``@flashinfer_api(trace=<template>)`` decorator.

Axis extraction is **automatic**: the extraction logic is derived from the
``dim_names`` of the ``Tensor`` inputs — no lambda functions required.

Example::

    from aklr.core.schema import Schema, Var, Const, Tensor, Scalar

    rmsnorm_trace = Schema(
        op_type="rmsnorm",
        axes={"num_tokens": Var(), "hidden_size": Const()},
        inputs={
            "input":  Tensor(["num_tokens", "hidden_size"]),
            "weight": Tensor(["hidden_size"]),
            "eps":    Scalar("float32"),
        },
        outputs={"output": Tensor(["num_tokens", "hidden_size"])},
    )
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from pydantic import (
    Field,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from aklr.core._frozen_model import NonEmptyStr, frozen_dataclass
from aklr.core.schema._args import Const, Scalar, Tensor, Var
from aklr.core.schema._check import standard_check
from aklr.core.schema._function_source import FUNCTION_SOURCES
from aklr.core.schema._kernel import Kernel
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
    # First in the json dump. The unvalidated "" default is filled by _default_name.
    name: NonEmptyStr = Field(default="", kw_only=True)
    op_type: NonEmptyStr
    axes: Dict[NonEmptyStr, Union[Var, Const]]
    # None falls back to op_type in schema_name.
    name_prefix: Optional[NonEmptyStr] = Field(default=None, exclude=True, repr=False)

    workloads: List[Workload] = Field(default_factory=list)
    # kw_only: a required field after defaulted ones (a positional one would not compile).
    inputs: Dict[NonEmptyStr, Union[Tensor, Scalar]] = Field(kw_only=True)
    outputs: Dict[NonEmptyStr, Union[Tensor, Scalar]] = Field(kw_only=True)
    # Standalone Python source; a function passed here is rendered on construction.
    reference: Optional[Union[NonEmptyStr, Callable[..., Any]]] = None
    check: Union[str, Callable[..., Any]] = Field(default=standard_check, validate_default=True)
    init: Optional[Union[str, Callable[..., Any]]] = None
    # Only "<model>/<parallel>", one per deployment calling it, e.g. "glm-5.2/tp8+dp8":
    # tp<n>, then dp<n> (attention DP), dcp<n> (decode context parallel), ep<n> if > 1.
    tags: List[NonEmptyStr] = Field(default_factory=list)
    description: Optional[str] = None
    constraints: List[NonEmptyStr] = Field(default_factory=list)

    kernel: Optional[Kernel] = Field(default=None)

    @field_validator(*FUNCTION_SOURCES, mode="before")
    @classmethod
    def _function_to_source(cls, value: Any, info: ValidationInfo) -> Any:
        """Render a function passed for a source field into its standalone source."""
        if callable(value):
            assert info.field_name is not None  # always set for a field validator
            return FUNCTION_SOURCES[info.field_name].render(value)
        return value

    def source_to_function(self, field: str) -> Optional[Callable]:
        """Load the source of *field* (``reference``, ``check``, ``init``) as a function.

        Returns None when the field holds no source.
        """
        source = getattr(self, field)
        if source is None:
            return None
        return FUNCTION_SOURCES[field].load(source, f"<{self.name}.{field}>")

    @model_validator(mode="after")
    def _default_name(self) -> Schema:
        """Fill an unset ``name`` from :meth:`schema_name` over the set const axes."""
        if not self.name:
            object.__setattr__(self, "name", self.schema_name(self.const_axes))  # frozen
        return self

    @model_validator(mode="after")
    def _validate_function_sources(self) -> Schema:
        """Validate that each source field is valid Python binding its entry name.

        Raises
        ------
        ValueError
            If a source does not parse or does not bind its entry name.
        """
        for field, function_source in FUNCTION_SOURCES.items():
            source = getattr(self, field)
            if source is not None:
                function_source.validate(source)
        return self

    @model_validator(mode="after")
    def _validate_input_output_names(self) -> Schema:
        """Validate that input and output names are unique and do not overlap.

        Raises
        ------
        ValueError
            If the input or output names are not unique or overlap.
        """
        if set(self.inputs.keys()) & set(self.outputs.keys()):
            raise ValueError("Input and output names must not overlap")
        return self

    @model_validator(mode="after")
    def _validate_constraints_syntax(self) -> Schema:
        """Validate that constraints are valid Python expressions.

        Raises
        ------
        ValueError
            If any constraint is not a valid Python expression.
        """
        if self.constraints is not None:
            for constraint in self.constraints:
                try:
                    ast.parse(constraint, mode="eval")
                except SyntaxError as e:
                    raise ValueError(f"Constraints must be valid Python expressions: {e}") from e
        return self

    @model_validator(mode="after")
    def _validate_tensor_axis_references(self) -> Schema:
        """Validate that tensor shapes reference defined axes.

        Ensures that all axis names used in input and output tensor shapes
        are properly defined in the axes dictionary.

        Raises
        ------
        ValueError
            If any tensor shape references an undefined axis.
        """
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

    @cached_property
    def const_axes(self) -> Dict[str, int]:
        """Get all constant axes and their values.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping constant axis names to their fixed values; a ``Const``
        whose value is still unset is left out.
        """
        return {
            name: axis.value
            for name, axis in self.axes.items()
            if isinstance(axis, Const) and axis.value is not None
        }

    @cached_property
    def var_axes(self) -> List[str]:
        """Get all variable axis names.

        Returns
        -------
        List[str]
            List of all variable axis names defined in this Definition.
        """
        return [name for name, axis in self.axes.items() if isinstance(axis, Var)]

    @cached_property
    def uuid(self) -> str:
        key = json.dumps(
            TypeAdapter(Schema).dump_python(self, mode="json", exclude={"workloads"}),
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @cached_property
    def var_axes_bindings(self) -> Dict[str, Tuple[str, int]]:
        """Get the bindings of variable axes to input tensor dimensions.

        Determines which input tensor and dimension index corresponds to each
        variable axis. If multiple input tensors share the same axis, the
        binding will be to the first tensor encountered.

        Returns
        -------
        Dict[str, Tuple[str, int]]
            Dictionary mapping axis names to tuples of (input_tensor_name, dimension_index).
            Only includes variable axes that appear in input tensor shapes.
        """
        bindings: Dict[str, Tuple[str, int]] = {}
        for inp_name, tensor in self.inputs.items():
            if isinstance(tensor, Scalar):  # scalar, no shape
                continue
            for dim_idx, axis in enumerate(tensor.dim_names):
                ax_def = self.axes.get(axis)
                if isinstance(ax_def, Var) and axis not in bindings:
                    bindings[axis] = (inp_name, dim_idx)
        return bindings

    def get_axes_values(self, input_shapes: Iterable[Optional[Tuple[int, ...]]]) -> Dict[str, int]:
        """Get concrete variable axis values from input shapes.

        Parameters
        ----------
        input_shapes : Iterable[Optional[Tuple[int, ...]]]
            Iterable of input tensor shapes.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping variable axis names to their concrete values; an axis
            only an absent input or an output carries is left out.

        Raises
        ------
        ValueError
            If a axis occurs in multiple input tensors, but the values are not consistent.
        """
        var_axes_values: Dict[str, int] = {}
        for (inp_name, inp), inp_shape in zip(self.inputs.items(), input_shapes):
            if isinstance(inp, Scalar) or inp_shape is None:  # scalar or absent, no shape
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
        """Get concrete variable axis values directly from input values.

        Convenience method that combines extract_shapes and get_var_axes_values.

        Parameters
        ----------
        inputs : Iterable[Any]
            Iterable of input values (tensors or other types).

        Returns
        -------
        Dict[str, int]
            Dictionary mapping variable axis names to their concrete values.
        """
        shapes = [tuple(val.shape) if hasattr(val, "shape") else None for val in inputs]
        return self.get_axes_values(shapes)

    def _get_shapes(
        self, tensors: Iterable[Tensor], var_axes_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
        """Get concrete tensor shapes given variable axis values.

        Parameters
        ----------
        tensors : List[Tensor]
            List of tensor specifications to compute shapes for.
        var_values : Optional[Dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        List[Optional[Tuple[int, ...]]]
            List of concrete shapes as tuples of integers. None for scalar tensors.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
        var_axes_values = var_axes_values or {}
        shapes = []

        for tensor in tensors:
            if isinstance(tensor, Scalar):  # scalar, no shape
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
        """Get concrete input shapes given variable axis values.

        Parameters
        ----------
        var_values : Optional[Dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping input tensor names to their concrete shapes.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
        return self._get_shapes(self.inputs.values(), var_axes_values)

    def get_output_shapes(
        self, var_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
        """Get concrete output shapes given variable axis values.

        Parameters
        ----------
        var_values : Optional[Dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping output tensor names to their concrete shapes.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
        return self._get_shapes(self.outputs.values(), var_values)

    def merge_kwargs_to_args(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
        """Merge keyword arguments into positional arguments based on input/output order.

        Parameters
        ----------
        args : Tuple[Any, ...]
            Positional arguments.
        kwargs : Dict[str, Any]
            Keyword arguments to merge.

        Returns
        -------
        Tuple[Any, ...]
            Merged positional arguments in the order: inputs, then outputs.
        """
        if not kwargs:
            return args

        param_names = list(self.inputs.keys()) + list(self.outputs.keys())

        if len(args) > len(param_names):
            raise TypeError(
                f"Too many positional arguments: got {len(args)}, expected at most {len(param_names)}"
            )

        # Check for duplicate arguments
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

    # ------------------------------------------------------------------
    # Axis extraction (automatic)
    # ------------------------------------------------------------------

    def _build_axis_extractors(  # structure: allow S4 the upstream builder, kept whole
        self,
    ) -> Dict[str, Callable[[Dict[str, Any]], Optional[int]]]:
        """Build per-axis extraction callables from tensor dim_names or scalars.

        For each axis, pick a source that is *always present* in a generated
        trace sample, in priority order:

          0. an explicit ``Const(value=...)`` or ``Const(extract=...)``;
          1. a required (non-optional) ``Tensor`` whose ``dim_names`` include
             the axis — read ``shape[dim_idx]``;
          2. a ``Scalar`` input named after the axis — read the kwarg value;
          3. an *optional* ``Tensor`` as a last resort;
          4. a scalar kwarg matching the axis name.

        Optional tensors are deprioritized because trace samples omit optional
        inputs: reading their shape would return ``None`` and leave a ``Const``
        axis without a value (e.g. ``top_k``, whose only tensor source is the
        optional ``pre_idx`` but which is always passed as a scalar).
        """
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
            # 0. Explicitly fixed or derived Const.
            if isinstance(marker, Const) and marker.value is not None:
                extractor = _make_fixed_extractor(marker.value)
            elif isinstance(marker, Const) and marker.extract is not None:
                extractor = marker.extract
            else:
                extractor = None
            # 1. Required tensor whose shape carries the axis.
            if extractor is None:
                extractor = _tensor_source(axis_name, allow_optional=False)
            # 2. Scalar input named after the axis (always present).
            if extractor is None:
                scalar_desc = self.inputs.get(axis_name)
                if isinstance(scalar_desc, Scalar):
                    param = scalar_desc.param if scalar_desc.param is not None else axis_name
                    extractor = _make_scalar_extractor(param)
            # 3. Optional tensor, best-effort (None when the sample omits it).
            if extractor is None:
                extractor = _tensor_source(axis_name, allow_optional=True)
            # 4. Fallback: scalar kwarg by axis name (undeclared integer args).
            if extractor is None:
                extractor = _make_scalar_extractor(axis_name)
            extractors[axis_name] = extractor

        return extractors

    def const_axis_extractors(
        self,
    ) -> dict[str, Callable[[dict[str, Any]], int | None]]:
        return {
            axis: extract
            for axis, extract in self._build_axis_extractors().items()
            if isinstance(self.axes[axis], Const)
        }

    def schema_name(self, axis_values: Dict[str, int]) -> str:
        """Schema name for a concrete const-axis vector.

        ``name_prefix`` (or ``op_type`` fallback) followed by each const axis's
        ``abbrev`` + value (``abbrev=""`` omits the axis; ``abbrev=None`` uses the
        axis name). This is the single source of truth for the naming convention,
        shared by :meth:`for_call` (trace time) and ``trace_apply``'s
        name-routing (apply time), so the two always agree.
        """
        prefix = self.name_prefix if self.name_prefix is not None else self.op_type
        parts = []
        for n, marker in self.axes.items():
            if not isinstance(marker, Const) or n not in axis_values:
                continue
            pfx = marker.abbrev if marker.abbrev is not None else n
            if pfx == "":
                continue
            parts.append(f"{pfx}{axis_values[n]}")
        return prefix + ("_" + "_".join(parts) if parts else "")

    def for_call(self, call: Dict[str, Any]) -> Schema:
        values = extract_axes(self.const_axis_extractors(), call)

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
                key: (replace(inp, dtype=input_dtype(key, inp)) if isinstance(inp, Tensor) else inp)
                for key, inp in self.inputs.items()
            },
            outputs={
                key: (replace(out, dtype=output_dtype(out)) if isinstance(out, Tensor) else out)
                for key, out in self.outputs.items()
            },
        )
