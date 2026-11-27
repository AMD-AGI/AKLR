# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The committed kernel of kimi_k3_sgl_kda_verify (gfx950) on calls validation does not draw: built from its schema at
a committed workload, it serves convolution rows any distance apart, those that are no multiple of eight channels
apart too, as it serves dense ones, and it sums a convolution's taps in FP32 as sglang's causal_conv1d_update does."""

import json
from pathlib import Path

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from tests.op_case import OpCase

_IMPLS = Path(__file__).resolve().parents[3] / "aklr/plugins/sglang/layers/kda_triton/impls/hip"


def _case(builder_registry, n_requests: int):
    """The case of the committed schema at its workload of ``n_requests`` requests, and the schema's kernel built."""
    (path,) = sorted(_IMPLS.glob("kimi_k3_sgl_kda_verify_*/*.json"))
    data = json.loads(path.read_text())
    (axes,) = [w["axes"] for w in data["workloads"] if w["axes"]["n_requests"] == n_requests]
    contract = ContractRegistry.default().for_schema_json(data)
    case = OpCase(contract.point_from_json(data, axes))
    schema = contract.narrow(
        case.point, workloads=[contract.workload_for(case.point, {})], kernel=data["kernel"]
    )
    return case, builder_registry.build(schema)


def _call(case: OpCase, args: dict, row_pitch: int | None = None) -> dict:
    """The api call of ``args`` on the GPU, its output allocated; ``row_pitch``: the convolution input's rows that many
    channels apart instead."""
    call = dict(case.call(args, "cuda"))
    contract = case.contract
    for name in contract.outputs:
        shape = contract.dtype_storage_shape_of(name, case.point)
        call[name] = torch.empty(shape, dtype=getattr(torch, case.point.dtypes[name]), device="cuda")
    if row_pitch is not None:
        rows = call["mixed_qkv"]
        storage = torch.empty(rows.shape[0], row_pitch, dtype=rows.dtype, device="cuda")
        call["mixed_qkv"] = storage[:, : rows[0].numel()].unflatten(1, rows.shape[1:]).copy_(rows)
    return call


def _written(case: OpCase, call: dict) -> dict:
    return {n: call[n] for n in case.contract.mutable_tensors}


@pytest.mark.gfx950
@pytest.mark.parametrize("extra", [1, 3, 8], ids=["one_channel", "three_channels", "eight_channels"])
def test_the_committed_kernel_serves_rows_any_distance_apart(builder_registry, extra):
    """Rows a row and ``extra`` channels apart: one and three channels off a multiple of eight, which the kernel
    copies dense, and eight, which it reads in place; every output and written input as on dense rows, and within the
    contract's bounds of the reference."""
    case, runnable = _case(builder_registry, 2)
    args = case.inputs()
    dense = _call(case, args)
    runnable(**dense)
    apart = _call(case, args, row_pitch=dense["mixed_qkv"][0].numel() + extra)
    assert apart["mixed_qkv"].stride(0) % 8 == extra % 8
    runnable(**apart)
    torch.cuda.synchronize()
    for name, tensor in _written(case, dense).items():
        assert torch.equal(_written(case, apart)[name], tensor), name
    on_gpu = {n: v.cuda() if torch.is_tensor(v) else v for n, v in args.items()}
    expected = case.contract.reference.reference(**case.request.reference_call(on_gpu, "cuda"))
    assert case.contract.reference.check(_written(case, apart), expected, case.request.schema) == {}


@pytest.mark.gfx950
def test_the_committed_kernel_sums_the_taps_in_fp32_as_sglang_does(builder_registry):
    """Inputs of 1 and taps (2**26, 1, -2**26, 0): the exact sum is 1, the FP32 sum, oldest first, 0 (2**26 + 1 rounds
    to 2**26). The contract's convolution is sglang's FP32 sum: the kernel's convolved keys and values in the rings
    are sglang's causal_conv1d_update's, SiLU(0) = 0."""
    pytest.importorskip("sglang")
    case, runnable = _case(builder_registry, 1)
    args = case.inputs()
    args["conv_weight"] = (
        torch.tensor([2.0**26, 1.0, -(2.0**26), 0.0]).expand_as(args["conv_weight"]).contiguous()
    )
    args["conv_state"].fill_(1)
    args["mixed_qkv"].fill_(1)
    kernels = _call(case, args)
    runnable(**kernels)
    sglangs = _call(case, args)
    case.contract.baseline.baseline(**sglangs)
    torch.cuda.synchronize()
    slot = int(args["cache_indices"][0])
    steps = case.point.axes["n_verify_tokens"]
    for name in ("ring_k", "ring_v"):
        written = kernels[name][slot, :, :steps]
        assert torch.equal(written, sglangs[name][slot, :, :steps]), name
        assert not written.float().any(), name
