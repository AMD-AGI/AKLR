# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The generated ``fast_key(call)`` closure: a call read as a Point's FastKey.

Locals are the contract's names, ``_dt_<t>`` (dtypes), ``_s_<t>`` (shapes),
``_ptr_<t>`` (data pointers), ``_bytes_<o>`` (output sizes), ``_r_<a>`` (raw attrs),
``_v<i>`` (variant facts) and ``_layout_variant_idx``.

Like ``alloc_outputs``, it may write into ``call``: a strided view of a layout's
bytes becomes the stored tensor (``Layout.view_rewrite``), which the kernel is passed.
"""

from __future__ import annotations

import ast
import copy
import functools
from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from aklr.core.contract import expr
from aklr.core.contract.closure import (
    fill_expr,
    fill_statements,
    join_tests,
    name_node,
    shape_tuple,
)
from aklr.core.contract.fastkey._generator import ClosureGenerator
from aklr.core.contract.semantics import DTYPE_TRAITS, DType, torch_dtypes
from aklr.core.contract.spec import ScalarSpec, TensorSpec, attr_value

if TYPE_CHECKING:
    import torch

    from aklr.core.contract.fastkey._codec import FastKeyCodec

MIN_PTR_ALIGN_BYTES = 16
# Everything a call can make a generated closure raise: it declines instead.
_DECLINE = (
    LookupError,
    TypeError,
    ValueError,
    AttributeError,
    ArithmeticError,
    RuntimeError,
)


class FastKeyFnGenerator(ClosureGenerator):
    """Generates ``fast_key(call)``; ``body`` grows by one ``_emit_*`` per step.
    ``read_tensors`` (tensor inputs, then required outputs) carry the axes."""

    def __init__(self, codec: FastKeyCodec, device: torch.device | None) -> None:
        import torch

        contract = codec.contract
        self.inputs = {n: s for n, s in contract.inputs.items() if isinstance(s, TensorSpec)}
        self.read_tensors = self.inputs | {o: contract.outputs[o] for o in contract.required_outputs}
        scope = {
            "_DECLINE": _DECLINE,
            "_attr_value": attr_value,
            "_as_float": self._as_float,
            # generated code runs without builtins
            "_bool": bool,
            "_int": int,
            "_type": type,
            "_getattr": getattr,
            "_strided": torch.strided,
        }
        super().__init__(codec, device, scope)
        self.body: list[ast.stmt] = []

    def generate(self) -> Callable:
        """Every step, then ``return <key>``, in a ``try`` returning ``None`` on
        ``_DECLINE``. Scalars come last: their ``present_when`` reads earlier locals."""
        for name in (*self.contract.inputs, *self.contract.outputs):
            self.body += fill_statements("T = call[K]", T=name, K=ast.Constant(name))
        self._emit_attrs()
        # the storage checks read the variant; its facts read ``call``, not the locals
        self._emit_layout()
        self._emit_tensors()
        self._emit_axes()
        # Each present read tensor's shape is its stored shape.
        for name, spec in self.read_tensors.items():
            shape = name_node(f"_s_{name}")
            check = self._decline_unless(self._shape_test(name, spec, shape))
            if spec.present_when is not None:
                check = [ast.If(fill_expr("T is not None", T=name), check, [])]
            self.body += check
        self._emit_outputs()
        self._emit_scalars()
        key = ast.Tuple(list(map(name_node, self.codec.key_names)), ast.Load())
        body = fill_statements("try:\n    pass\nexcept _DECLINE:\n    return None")
        body[0].body = [*self.body, ast.Return(key)]
        return self.compile("fast_key", "call", body)

    def _emit_attrs(self) -> None:  # structure: allow S4 a generate() step
        for name, a in self.contract.attrs.items():
            if a.param_name is None:  # presence of its optional tensor
                (t,) = (t for t, s in self.contract.inputs.items() if s.present_when == name)
                self.body += fill_statements("A = T is not None", A=name, T=t)
                continue
            # Two levels, by ``type(v) is bool``: keeps ``True`` apart from ``1``.
            values = {flag: {} for flag in (True, False)}
            for v in a.values:
                values[type(v) is bool][v] = v
            if a.default is not None:
                values[False][None] = a.default
            self.scope[f"_A_{name}"] = MappingProxyType(
                {flag: MappingProxyType(m) for flag, m in values.items()}
            )
            raw, p = f"_r_{name}", ast.Constant(a.param_name)
            self.body += fill_statements(
                "R = _attr_value(call[P])\nA = M[R is True or R is False][R]",
                R=raw,
                A=name,
                M=f"_A_{name}",
                P=p,
            )

    def _emit_tensors(self) -> None:  # structure: allow S4 a generate() step
        for name, spec in self.read_tensors.items():
            self.scope[f"_D_{name}"] = self.codec.primary_storage_dtypes[name]
            dt = f"_dt_{name}"
            read = fill_statements("D = M[T.dtype]", D=dt, M=f"_D_{name}", T=name)
            read += self._storage_checks(name)
            read += fill_statements("S = T.shape", S=f"_s_{name}", T=name)
            if spec.present_when is not None:
                self.body += fill_statements("D = None", D=dt)
                read = [ast.If(fill_expr("T is not None", T=name), read, [])]
            self.body += read
        # the kernel writes a mutable input in place: it shares no byte with another input
        for name in self.contract.mutable_inputs:
            self.body += self._overlap_checks(name, [n for n in self.inputs if n != name])

    def _emit_layout(self) -> None:  # structure: allow S4 a generate() step
        variants = self.contract.layout_variants.by_name.values()
        if not any(v.detect_facts for v in variants):  # one fact-less variant
            self.body += fill_statements("_layout_variant_idx = 0")
            return
        holds = []
        for i, v in enumerate(variants):
            facts = []
            for fact in v.detect_facts:
                read = fact.location.read_source(fact.name)
                facts.append(fill_expr(f"{read} == V", V=ast.Constant(fact.expected_value)))
            self.body += fill_statements("V = X", V=f"_v{i}", X=join_tests(ast.And(), facts))
            holds.append(name_node(f"_v{i}"))
        if len(holds) == 1:
            self.body += self._decline_unless(holds[0])
            self.body += fill_statements("_layout_variant_idx = 0")
            return
        n_matches = functools.reduce(lambda a, b: fill_expr("A + B", A=a, B=b), holds)
        self.body += fill_statements("if N != 1:\n    return None", N=n_matches)
        idx = ast.Constant(len(holds) - 1)
        for i in reversed(range(len(holds) - 1)):
            idx = fill_expr("I if V else B", I=i, V=holds[i], B=idx)
        self.body += fill_statements("_layout_variant_idx = B", B=idx)

    def _emit_axes(self) -> None:  # structure: allow S4 a generate() step
        for axis in self.contract.domain.axes:
            # The tensor ``axis`` is read from and its dim in each shape case.
            name = self.contract._axis_readers[axis][0]
            spec = self.contract.tensors()[name]
            dims = self.contract._axis_dims(axis, name)
            is_packed = name in self.codec.packed_tensors
            reads = []
            for shape, i in zip(spec.shape_cases(), dims, strict=True):
                read = fill_expr("S[I]", S=f"_s_{name}", I=i)
                if is_packed and i == len(shape) - 1:
                    read = fill_expr("X * 2", X=read)
                reads.append(read)
            value = self.by_case(spec, reads)
            presence = self.contract.optional_axes.get(axis)
            if presence is not None:  # absent: its ``lo``, as the Point holds it
                lo = self.contract.axes[axis].lo
                value = fill_expr("X if P else L", X=value, P=presence, L=lo)
            self.body += fill_statements("A = X", A=axis, X=value)
            a = self.contract.axes[axis]
            # no workload holds an any_size axis's value: its bounds and the
            # constraints reading it are checked here
            if a.any_size:
                self.body += self._decline_unless(
                    fill_expr(
                        "L <= A <= H and A % M == 0",
                        L=a.lo,
                        A=axis,
                        H=a.hi,
                        M=a.multiple_of,
                    )
                )
        self.body += self.derived_axis_statements()
        any_size = {n for n, a in self.contract.axes.items() if a.any_size}
        for text in self.contract.constraints:
            constraint = expr.parse(text)
            if constraint.symbols & any_size:
                self.body += self._decline_unless(expr.to_ast(constraint, self.symbol))

    def _emit_outputs(self) -> None:  # structure: allow S4 a generate() step
        for i, (name, spec) in enumerate(self.contract.outputs.items()):
            dtype_spec, dt = spec.dtype, f"_dt_{name}"
            is_required = name in self.contract.required_outputs
            fallback = None  # without one, ``out``'s dtype must be passed
            if dtype_spec.same_as_tensor is not None:
                fallback = name_node(f"_dt_{dtype_spec.same_as_tensor}")
            elif not is_required and (dtype_spec.default is not None or name != "out"):
                constant = dtype_spec.default or dtype_spec.any_of[0]
                self.scope[f"_DTYPE_{name}"] = torch_dtypes(constant)[0]
                fallback = name_node(f"_DTYPE_{name}")
            if name == "out":
                self.scope["_D_out"] = self.codec.primary_storage_dtypes["out"]
                p = ast.Constant("out_dtype")
                read = fill_expr("_D_out[call[P]]", P=p)
                if fallback is not None:
                    read = fill_expr("F if call[P] is None else X", F=fallback, P=p, X=read)
                self.body += fill_statements("_dt_out = X", X=read)
            elif not is_required:
                self.body += fill_statements("D = F", D=dt, F=fallback)
            if dtype_spec.any_of is None and (name == "out" or is_required):
                self.body += self._decline_unless(fill_expr("D == F", D=dt, F=fallback))
            check = self._overlap_checks(name, [*self.inputs, *list(self.contract.outputs)[:i]])
            if not is_required:
                self.scope[f"_D_{name}"] = self.codec.primary_storage_dtypes[name]
                given_ok = [
                    fill_expr("M[O.dtype] == D", M=f"_D_{name}", O=name, D=dt),
                    self._shape_test(name, spec, fill_expr("O.shape", O=name)),
                ]
                check = [
                    *self._storage_checks(name),
                    *self._decline_unless(ast.BoolOp(ast.And(), given_ok)),
                    *check,
                ]
                check = [ast.If(fill_expr("O is not None", O=name), check, [])]
            self.body += check

    def _overlap_checks(self, written: str, others: list[str]) -> list[ast.stmt]:  # structure: allow S4 long caller  # fmt: skip
        """Decline when the tensor the kernel writes, ``written``, shares a byte with
        any of ``others``."""
        size, tests = f"_bytes_{written}", []
        for name in others:
            spec = self.read_tensors.get(name)  # None: an allocated output
            test = fill_expr(
                "P < Q + T.nbytes and Q < P + B",
                P=f"_ptr_{written}",
                Q=f"_ptr_{name}",
                T=name,
                B=size,
            )
            if spec is None or spec.present_when is not None:
                test = fill_expr("T is not None and X", T=name, X=test)
            tests.append(test)
        if not tests:
            return []
        overlaps = join_tests(ast.Or(), tests)
        return fill_statements("B = O.nbytes\nif X:\n    return None", B=size, O=written, X=overlaps)

    def _emit_scalars(self) -> None:  # structure: allow S4 a generate() step
        for name, spec in self.contract.inputs.items():
            if not isinstance(spec, ScalarSpec):
                continue
            tests = [fill_expr("T is not None", T=name)]
            value = "_as_float(T)"  # declines a bool or non-number
            if spec.dtype is DType.BOOL or DTYPE_TRAITS[spec.dtype].is_integral:
                # the kernel gets T as is: an int scalar never sees 2.0 or True
                kind = "_bool" if spec.dtype is DType.BOOL else "_int"
                tests.append(fill_expr(f"_type(T) is {kind}", T=name))
                value = "T"
            if spec.values is not None:
                tests.append(fill_expr(f"{value} in V", T=name, V=ast.Constant(spec.values)))
            if spec.interval is not None:
                lo, hi = spec.interval
                tests.append(fill_expr("L <= _as_float(T) <= H", L=lo, T=name, H=hi))
            check = self._decline_unless(join_tests(ast.And(), tests))
            if spec.present_when is None:
                self.body += check
            else:
                when = expr.to_ast(expr.parse(spec.present_when), self.symbol)
                absent = fill_statements("if T is not None:\n    return None", T=name)
                self.body.append(ast.If(when, check, absent))

    def _storage_checks(self, tensor: str) -> list[ast.stmt]:
        layouts = [v.layout_of(tensor) for v in self.contract.layout_variants.by_name.values()]
        read = []
        for i, layout in enumerate(layouts):
            if (text := layout.view_rewrite()) is not None:
                when = fill_expr("not T.is_contiguous() and _layout_variant_idx == I", T=tensor, I=i)
                rewrite = fill_statements(text, T=tensor, K=ast.Constant(tensor))
                read.append(ast.If(when, rewrite, []))
        read += fill_statements("P = T.data_ptr()", P=f"_ptr_{tensor}", T=tensor)
        stride_test = self._by_variant([fill_expr(layout.storage_test(), T=tensor) for layout in layouts])
        tests = [fill_expr("T.layout is _strided", T=tensor), stride_test]
        # Only a complex dtype carries the conj bit; the dtype map declined others.
        storages = (s for d in self.contract.dtypes_of(tensor) for s in torch_dtypes(d))
        if any(s.is_complex for s in storages):
            tests.append(fill_expr("not T.is_conj()", T=tensor))
        tests.append(fill_expr("not T.is_neg()", T=tensor))
        tests.append(fill_expr("P % A == 0", P=f"_ptr_{tensor}", A=MIN_PTR_ALIGN_BYTES))
        if self.device is not None:
            tests.append(fill_expr("T.device == _device", T=tensor))
        return [*read, *self._decline_unless(join_tests(ast.And(), tests))]

    def _shape_test(self, tensor: str, spec: TensorSpec, actual: ast.expr) -> ast.expr:
        is_packed = tensor in self.codec.packed_tensors
        tests = []
        for shape in spec.shape_cases():
            forms = [shape]
            if spec.can_squeeze and shape[-1] == 1:
                forms.append(shape[:-1])
            alternatives = []
            for form in forms:
                dtype_storage_dims = shape_tuple(form)
                if is_packed:  # ``semantics.dtype_storage_shape``, inlined
                    dtype_storage_dims.elts[-1] = fill_expr("D // 2", D=dtype_storage_dims.elts[-1])
                stored = self._storage_dims(tensor, dtype_storage_dims)
                alternatives.append(fill_expr("T == S", T=actual, S=stored))
            tests.append(join_tests(ast.Or(), alternatives))
        return self.by_case(spec, tests)

    def _storage_dims(self, tensor: str, dtype_storage_dims: ast.Tuple) -> ast.expr:  # structure: allow S4 long  # fmt: skip
        names = tuple(f"d{i}" for i in range(len(dtype_storage_dims.elts)))
        texts = [
            v.layout_of(tensor).storage_shape(names) for v in self.contract.layout_variants.by_name.values()
        ]
        if all(t == names for t in texts):
            return dtype_storage_dims
        dims = dict(zip(names, dtype_storage_dims.elts, strict=True))
        return self._by_variant(
            [
                ast.Tuple(
                    [expr.to_ast(expr.parse(d), lambda n: copy.deepcopy(dims[n])) for d in text],
                    ast.Load(),
                )
                for text in texts
            ]
        )

    @staticmethod
    def _by_variant(exprs: list[ast.expr]) -> ast.expr:
        """``exprs[i]`` (one per layout variant) chosen by ``_layout_variant_idx``."""
        if len({ast.dump(x) for x in exprs}) == 1:
            return exprs[0]
        *branches, result = exprs
        for i, x in reversed(list(enumerate(branches))):
            result = fill_expr("X if _layout_variant_idx == I else Y", X=x, I=i, Y=result)
        return result

    @staticmethod
    def _decline_unless(test: ast.expr) -> list[ast.stmt]:
        return fill_statements("if not X:\n    return None", X=test)

    @staticmethod
    def _as_float(v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"not an int or float: {type(v).__name__}")
        return float(v)
