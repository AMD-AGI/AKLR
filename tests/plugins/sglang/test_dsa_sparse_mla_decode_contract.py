# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSA decode schemas: self-contained OpenAI kernels at GLM-5.2's TP8 Points."""

import ast
from pathlib import Path

import pytest
import torch

from aklr.core.contract import semantics
from aklr.core.contract.registry import ContractRegistry
from aklr.core.validation.stages._audit._stage import check_kernel
from aklr.core.validation.stages._bench._evaluation._trials import TrialRequest
from aklr.plugins.sglang.layers.dsa_backend.reference import (
    dsa_sparse_mla_decode_reference,
)

IMPLS = Path(__file__).parents[3] / "aklr/plugins/sglang/layers/dsa_backend/impls"
# what the sources import, siblings aside
IMPORTS = {"math", "typing", "torch", "triton"}
# (heads, tokens): TP8 decode batches, none of 16 tokens or fewer, which the
# DeepseekV2DecoderLayer route's ops serve whole
POINTS = {(8, 32), (8, 128), (8, 256)}


def schemas():
    registry = ContractRegistry.default()
    return [
        (path, registry.parse_schema(path.read_bytes(), str(path))) for path in sorted(IMPLS.rglob("*.json"))
    ]


def test_the_schemas_serve_glm52s_decode_points_once_each():
    served = []
    for _, schema in schemas():
        assert schema.contract_id == "dsa_sparse_mla_decode@1.0"
        for workload in schema.workloads:
            point = schema.point(workload.axes)
            assert point.axes["kv_dim"] == 576 and point.axes["topk"] == 2048
            assert point.dtypes["q"] == point.dtypes["kv_cache"] == "float8_e4m3fn"
            assert workload.scalars == {"sm_scale": 0.0625}
            assert workload.accuracy is None and workload.speedup is None
            served.append((point.axes["n_heads"], point.axes["n_tokens"]))
    assert sorted(served) == sorted(POINTS)


@pytest.mark.parametrize("index", range(len(POINTS)))
def test_each_kernel_is_openais_self_contained_and_passes_the_audit(index):
    path, schema = schemas()[index]
    kernel = schema.kernel
    assert kernel.language == "triton" and kernel.author == "openai"
    assert kernel.requires == {"triton": ">=3.8"}
    assert kernel.entry_point == "kernel.py::run"
    assert not kernel.state_bytes and not kernel.peer_bytes
    for source in kernel.sources:
        assert source.content.startswith("# Created by OpenAI\n"), source.path
        for node in ast.walk(ast.parse(source.content)):
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and not node.level:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert roots <= IMPORTS, (source.path, roots)
    findings = list(
        check_kernel(
            str(path),
            kernel,
            call_keys=schema.kernel_call_keys,
            tensor_names=tuple(schema.tensors()),
            mutable_names=schema.mutable_tensors,
        )
    )
    assert findings == []


@pytest.mark.gpu
@pytest.mark.gfx950
@pytest.mark.parametrize("index", range(len(POINTS)))
def test_each_kernel_reads_a_short_context_and_a_padded_row(index, builder_registry):
    """The top-k as decode leaves it at a 1536-token context (its last 512 entries -1)
    and a CUDA graph's padded row (every entry -1): the output within the contract's
    bound, the padded row 0."""
    path, schema = schemas()[index]
    contract = ContractRegistry.default().get(schema.contract_id)
    request = TrialRequest.of_workload(
        str(path),
        schema,
        contract,
        schema.workloads[0],
        thresholds={},
        incumbent=None,
        baseline=None,
    )
    args = semantics.row_major_inputs(
        request.schema_json,
        request.workload_json,
        request.distributions,
        input_set="numeric/0",
    )
    topk_indices = args["topk_indices"]
    topk_indices[:, 1536:] = -1
    padded = topk_indices.shape[0] > 1
    if padded:
        topk_indices[-1] = -1
    call = request.assemble_call(args, "cuda")
    tokens, heads, _ = call["q"].shape
    call["output"] = torch.full((tokens, heads, 512), torch.nan, dtype=torch.bfloat16, device="cuda")
    builder_registry.build(schema)(**call)
    output = call["output"].cpu()
    expected = dsa_sparse_mla_decode_reference(request.point, args, semantics.to_float64)
    results = semantics.compare_outputs(request.schema_json, {"output": output}, {"output": expected})
    assert results["output"].is_passed, results["output"].detail
    if padded:
        assert bool(output[-1].eq(0).all())
