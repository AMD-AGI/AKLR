# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aklr.seam  # noqa: F401
from aklr.core.runtime.seam import SeamRegistry
from tests.op_case import OpCase


class TestReference:
    def test_outputs_have_the_declared_shapes(self, op_case: OpCase) -> None:
        schema = op_case.schema()
        outputs = op_case.reference(op_case.call())
        assert [tuple(t.shape) for t in outputs] == [
            tuple(s) for s in schema.get_output_shapes({**schema.const_axes, **op_case.axes})
        ]
        assert all(t.isfinite().all() for t in outputs)

    def test_check_passes_the_reference_in_the_output_dtypes(self, op_case: OpCase) -> None:
        reference = op_case.reference(op_case.call())
        rounded = [t.to(d) for t, d in zip(reference, op_case.output_dtypes, strict=True)]
        assert op_case.template.source_to_function("check")(reference, rounded)

    def test_check_fails_an_output_off_the_reference(self, op_case: OpCase) -> None:
        reference = op_case.reference(op_case.call())
        off = [
            (-t - 1 if t.is_floating_point() else t + 1).to(d)
            for t, d in zip(reference, op_case.output_dtypes, strict=True)
        ]
        assert not op_case.template.source_to_function("check")(reference, off)

    @pytest.mark.gpu
    @pytest.mark.gfx950
    def test_backend_matches_the_reference(self, op_case: OpCase) -> None:
        schema = op_case.schema()
        call = op_case.call(device="cuda")
        shapes = schema.get_output_shapes({**schema.const_axes, **op_case.axes})
        for name, shape, dtype in zip(schema.outputs, shapes, op_case.output_dtypes, strict=True):
            call.setdefault(name, torch.empty(shape, dtype=dtype, device="cuda"))
        runs = [
            run
            for adapter in SeamRegistry.default().adapters(schema.op_type)
            if (run := adapter.baseline(call)) is not None
        ]
        if not runs:
            pytest.skip("no backend reads this storage")
        result = runs[0]()
        returned = list(result) if isinstance(result, (tuple, list)) else [result]
        actual = (
            returned
            if all(isinstance(t, torch.Tensor) for t in returned)
            else [call[name] for name in schema.outputs]
        )
        check = op_case.template.source_to_function("check")
        assert check([t.cpu() for t in op_case.reference(call)], [t.cpu() for t in actual])
