# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
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
    expected = sparse_prefill_mxfp8_reference(case.point, inputs)
    device_inputs = {key: value.cuda() if torch.is_tensor(value) else value for key, value in inputs.items()}
    actual = sparse_prefill_mxfp8_reference(case.point, device_inputs)
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual.cpu(), expected)


def _csr_tail_workloads():
    schemas, errors = ContractRegistry.default().read_schemas("aklr.ops.mla_prefill")
    assert not errors
    workloads = []
    for path, schema in schemas:
        if schema.kernel.language != "hip" or not schema.kernel.sources:
            continue
        for workload in schema.workloads:
            if workload.axes["n_tokens"] == 344:
                workloads.append(
                    pytest.param(schema, workload, id=f"{path.stem}-prefix{workload.axes['prefix_rows']}")
                )
    # a HIP CSR kernel withdrawn to a kernel-less schema leaves nothing for the tail check to run on
    return workloads or [
        pytest.param(None, None, marks=pytest.mark.skip(reason="no committed HIP CSR kernel"))
    ]


@pytest.mark.gpu
@pytest.mark.gfx950
@pytest.mark.parametrize("schema,workload", _csr_tail_workloads())
@pytest.mark.parametrize("input_set", ["numeric/0", "numeric/1"])
def test_csr_prefill_repeated_calls_are_bitwise(
    schema, workload, input_set, committed_kernel_cache, monkeypatch
):
    """Tail partials must be visible before merging, in eager calls and graph replays."""
    monkeypatch.setenv("AKLR_KERNEL_CACHE_DIR", str(committed_kernel_cache))
    case = OpCase(schema.point(workload.axes))
    call = case.call(case.inputs(input_set), "cuda")
    call["output"] = torch.empty(
        case.contract.storage_shape_of("output", case.point), device="cuda", dtype=torch.bfloat16
    )
    runnable = HipBuilder().build(schema)
    runnable(**call)
    expected = call["output"].clone().view(torch.uint8)
    for repeat in range(128):
        runnable(**call)
        assert torch.equal(call["output"].view(torch.uint8), expected), f"eager repeat {repeat}"

    # Capture on a side stream after warming it up, then reuse the graph's scratch.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            runnable(**call)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        runnable(**call)
    for repeat in range(128):
        graph.replay()
        assert torch.equal(call["output"].view(torch.uint8), expected), f"graph replay {repeat}"
