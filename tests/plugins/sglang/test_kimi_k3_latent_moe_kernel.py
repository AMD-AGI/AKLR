# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's latent MoE kernel for SGLang's decode at 1-16 tokens (gfx950), whose 16-token schedule splits the front
GEMM four ways along K into float32 partials its postprocess sums: every committed workload, its scratch drawn from
blocks an earlier tensor filled with ones in every bit, within the contract's bounds of the reference."""

import json
from pathlib import Path

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from tests.op_case import OpCase

_OP = Path(__file__).resolve().parents[3] / "aklr/plugins/sglang/models/kimi_k3/impls/triton"


def _schema():
    """The committed schema of the 1-16 token schedules: the file whose kernel splits the 16-token front."""
    (path,) = [
        p
        for p in sorted(_OP.glob("kimi_k3_latent_moe_*/*.json"))
        if any(
            "def _m16_front_split(" in s["content"] for s in json.loads(p.read_text())["kernel"]["sources"]
        )
    ]
    return path


def _workloads():
    return [w["axes"]["n_tokens"] for w in json.loads(_schema().read_text())["workloads"]]


@pytest.mark.gfx950
@pytest.mark.parametrize("n_tokens", _workloads())
@pytest.mark.parametrize("input_set", ["numeric/0", "numeric/1"])
def test_the_kernel_matches_the_reference_on_a_poisoned_allocator(builder_registry, n_tokens, input_set):
    """The split front's partials and the rest of the call's scratch come from freed blocks of all-ones bytes (NaN
    as float32): a scratch word read before it is written would carry a NaN into the outputs."""
    data = json.loads(_schema().read_text())
    contract = ContractRegistry.default().for_schema_json(data)
    (axes,) = [w["axes"] for w in data["workloads"] if w["axes"]["n_tokens"] == n_tokens]
    case = OpCase(contract.point_from_json(data, axes))
    schema = contract.narrow(
        case.point, workloads=[contract.workload_for(case.point, {})], kernel=data["kernel"]
    )
    runnable = builder_registry.build(schema)
    args = case.inputs(input_set)
    call = dict(case.call({n: t.cuda() if torch.is_tensor(t) else t for n, t in args.items()}, "cuda"))
    for name in contract.outputs:
        shape = contract.dtype_storage_shape_of(name, case.point)
        call[name] = torch.full(
            shape, float("nan"), dtype=getattr(torch, case.point.dtypes[name]), device="cuda"
        )
    torch.empty(1 << 30, dtype=torch.uint8, device="cuda").fill_(255)  # freed, kept by the allocator
    runnable(**call)
    torch.cuda.synchronize()
    assert case.compare({n: call[n] for n in contract.outputs}, args) == {}
