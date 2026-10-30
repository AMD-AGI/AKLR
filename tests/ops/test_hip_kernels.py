# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests of the committed hip kernels through the api, as a serving framework calls them."""

import pytest
import torch

import aklr.ops.gemm
from aklr.core.contract.registry import ContractRegistry
from aklr.core.platform.env import ENV_MIN_SPEEDUP
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason
from tests.op_case import OpCase


def _hip_gemm_points() -> list[tuple[str, dict[str, int]]]:
    """(name, every int axis of its smallest workload) of each committed hip gemm@1.0 schema."""
    contract = OpCase.of("gemm@1.0").contract
    schemas, _ = ContractRegistry.default().read_schemas(contract.op_package)
    return [
        (s.name, s.point(min((w.axes for w in s.workloads), key=lambda x: x["m"])).axes)
        for _, s in schemas
        if s.contract_id == contract.contract_id
        and s.kernel is not None
        and s.kernel.language == "hip"
        and s.kernel.sources
    ]


@pytest.mark.gfx950
@pytest.mark.parametrize(("name", "axes"), _hip_gemm_points(), ids=lambda v: str(v))
def test_gemm_out_dtype_none_is_the_input_dtype(
    default_dispatchers: DispatcherRegistry,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    axes: dict[str, int],
) -> None:
    """vLLM's ``tgemm.mm`` passes ``out_dtype=None``: the kernel serves it as a's dtype,
    bit for bit the result of passing that dtype."""
    monkeypatch.setenv(ENV_MIN_SPEEDUP, "0")
    default_dispatchers.init()
    torch.manual_seed(0)
    a = torch.randn(axes["m"], axes["k"], device="cuda").to(torch.bfloat16)
    b = (torch.randn(axes["n"], axes["k"], device="cuda") * 0.05).to(torch.bfloat16)
    implicit = aklr.ops.gemm.gemm(a=a, b=b, trans_b=True, out_dtype=None)
    explicit = aklr.ops.gemm.gemm(a=a, b=b, trans_b=True, out_dtype=torch.bfloat16)
    assert not isinstance(implicit, FallbackReason), (name, implicit)
    assert not isinstance(explicit, FallbackReason), (name, explicit)
    assert implicit.dtype == torch.bfloat16
    assert torch.equal(implicit, explicit)
