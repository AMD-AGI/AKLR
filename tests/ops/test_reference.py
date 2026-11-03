# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.contract import semantics
from aklr.core.platform.device import DeviceTarget
from aklr.core.validation.stages._bench._evaluation._comparison import (
    is_reference_comparable,
)
from tests.op_case import OpCase


class TestReference:
    """The reference's outputs are as declared, admitted by the contract's compare."""

    def test_outputs_are_declared(self, op_case: OpCase) -> None:
        """On the CPU, ``reference_outputs`` holds every result, float32 or float64 (an
        exact one), finite."""
        reference = op_case.reference_outputs(op_case.inputs())
        assert list(reference) == list(op_case.contract.mutable_tensors)
        for name, tensor in reference.items():
            assert tensor.device.type == "cpu", name
            assert tensor.dtype in (torch.float32, torch.float64), name
            assert tensor.isfinite().all(), name
            assert tuple(tensor.shape) == op_case.contract.dtype_storage_shape_of(name, op_case.point), name

    def test_rounded_reference_passes_compare(self, op_case: OpCase) -> None:
        """The reference rounded to each output's dtype passes ``op_case.compare``."""
        args = op_case.inputs()
        assert is_reference_comparable(
            op_case.point.dtypes,
            op_case.reference_outputs(args),
            lambda rounded: op_case.compare(rounded, args, baseline_outputs=rounded),
        )

    @pytest.mark.gpu
    def test_backend_matches_reference(self, op_case: OpCase) -> None:
        if op_case.needs_gfx950 and DeviceTarget.current().arch != "gfx950":
            pytest.skip("the backend runs the Point only on gfx950")
        args = op_case.inputs()
        call = op_case.call(args, "cuda")
        outputs = op_case.contract.call_mutable_tensors(op_case.contract.baseline()(**call), call)
        results = op_case.compare(outputs, args, baseline_outputs=outputs)
        output_specs = semantics.mutable_tensor_declarations(op_case.request.schema_json)
        baseline_relative = {
            o
            for o, spec in output_specs.items()
            if spec["error_bounds"] and spec["error_bounds"][0][1]["kind"] == "baseline_relative"
        }
        failed = {o: r.detail for o, r in results.items() if not r.is_passed and o not in baseline_relative}
        assert failed == {}
        # self-baselined compare is vacuous: check the reference by relative L2 error
        # shortcut: 0.1 uncalibrated (MI300X skips), set to ~3-5x the gfx950 ratio once measured
        reference = op_case.reference_outputs(args)
        for o in baseline_relative:
            error = (outputs[o].cpu().double() - reference[o]).norm()
            assert error <= 0.1 * reference[o].norm(), (
                o,
                (error / reference[o].norm()).item(),
            )


def test_mqa_logits_reference_is_the_same_in_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One row a chunk gives fp8_paged_mqa_logits' reference bit for bit."""
    import aklr.ops.mla.reference as mla_reference

    case = OpCase.of("fp8_paged_mqa_logits@1.0")
    args = case.inputs()
    whole = case.reference_outputs(args)["logits"]
    monkeypatch.setattr(mla_reference, "_CHUNK_ELEMENTS", 1)
    assert torch.equal(case.reference_outputs(args)["logits"], whole)
