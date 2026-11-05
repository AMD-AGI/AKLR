# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.semantics import to_float64
from aklr.core.runtime.builder import HipBuilder
from aklr.ops.mla_prefill.reference import sparse_prefill_mxfp8_reference
from tests.op_case import OpCase


@pytest.mark.gpu
@pytest.mark.parametrize("empty", [False, True])
def test_csr_reference_preserves_operand_device(empty):
    case = OpCase.of("sparse_prefill_mxfp8@1.0")
    if empty:
        axes = {**case.point.axes, "prefix_nnz": 0, "extend_nnz": 0}
        case = OpCase(case.point.model_copy(update={"axes": axes}))
    inputs = case.inputs()
    expected = sparse_prefill_mxfp8_reference(case.point, inputs, to_float64)
    device_inputs = {key: value.cuda() if torch.is_tensor(value) else value for key, value in inputs.items()}
    actual = sparse_prefill_mxfp8_reference(case.point, device_inputs, to_float64)
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.gpu
@pytest.mark.gfx950
def test_csr_prefill_repeated_calls_are_bitwise():
    """Arrival order of tail chunks must not alter the published BF16 result."""
    schemas, errors = ContractRegistry.default().read_schemas("aklr.ops.mla_prefill")
    assert not errors
    for _, schema in schemas:
        if schema.kernel.language != "hip" or not schema.kernel.sources:
            continue
        workload = next((w for w in schema.workloads if w.axes["n_tokens"] == 344), None)
        if workload is None:
            continue
        case = OpCase(schema.point(workload.axes))
        call = case.call(case.inputs(), "cuda")
        call["output"] = torch.empty(
            case.contract.storage_shape_of("output", case.point), device="cuda", dtype=torch.bfloat16
        )
        runnable = HipBuilder().build(schema)
        runnable(**call)
        expected = call["output"].clone()
        for _ in range(16):
            runnable(**call)
            assert torch.equal(call["output"].view(torch.uint8), expected.view(torch.uint8))
