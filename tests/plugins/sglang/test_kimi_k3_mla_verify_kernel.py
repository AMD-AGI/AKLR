# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The committed kernel of kimi_k3_mla_verify_q (gfx950) on calls validation does not draw: built from its schema at a
committed workload, it writes torch's float8_e4m3fn cast of every bf16 bit pattern byte for byte (signed zeros,
subnormals, rounding ties, values at and past the type's range, infinities and NaNs), and reads operand rows any
distance apart: rows past 2 GiB and past 4 GiB of the operand's first, rows not 16-byte aligned."""

import json
from pathlib import Path

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.plugins.sglang.models.kimi_k3.reference import KimiK3MlaVerifyQReference

_IMPLS = Path(__file__).resolve().parents[3] / "aklr/plugins/sglang/models/kimi_k3/impls/flydsl"
FP8 = torch.float8_e4m3fn
RANK, ROPE = 512, 64


def _schema() -> Path:
    (path,) = sorted(_IMPLS.glob("kimi_k3_mla_verify_q_*/*.json"))
    return path


def _kernel(builder_registry, n_tokens: int):
    """The committed kernel built for its workload of ``n_tokens`` tokens, and the workload's operand rows."""
    data = json.loads(_schema().read_text())
    (axes,) = [w["axes"] for w in data["workloads"] if w["axes"]["n_tokens"] == n_tokens]
    contract = ContractRegistry.default().for_schema_json(data)
    point = contract.point_from_json(data, axes)
    schema = contract.narrow(point, workloads=[contract.workload_for(point, {})], kernel=data["kernel"])
    return builder_registry.build(schema), point.axes


def _packed(runnable, axes, q_nope, q_pe) -> torch.Tensor:
    """The kernel's query from these operand rows, its output first filled with a byte no cast gives."""
    q = torch.empty(axes["n_tokens"], axes["n_kernel_heads"], RANK + ROPE, dtype=FP8, device="cuda")
    q.view(torch.uint8).fill_(0x55)
    runnable(q_nope=q_nope, q_pe=q_pe, q=q, storage="strided_rows")
    torch.cuda.synchronize()
    return q


def _rows_apart(rows: int, width: int, pitch: int) -> torch.Tensor:
    """(rows, 1, width) bf16 random rows ``pitch`` values apart, on the GPU (the gaps are never written)."""
    storage = torch.empty((rows - 1) * pitch + width, dtype=torch.bfloat16, device="cuda")
    view = storage.as_strided((rows, 1, width), (pitch, width, 1))
    view.copy_(torch.randn(rows, 1, width, device="cuda").to(torch.bfloat16))
    return view


def _check(q, q_nope, q_pe):
    expected = KimiK3MlaVerifyQReference().reference(q_nope=q_nope.contiguous(), q_pe=q_pe.contiguous(), q=q)
    failures = KimiK3MlaVerifyQReference().check({"q": q}, expected, None)
    assert failures == {}, failures


@pytest.mark.gfx950
def test_the_committed_kernel_casts_every_bf16_bit_pattern_as_torch_does(builder_registry):
    """Every bf16 bit pattern in the latent values, and again in the rotated ones, at the 16-token workload."""
    runnable, axes = _kernel(builder_registry, 16)
    rows = axes["n_rows"]
    patterns = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    latent = torch.zeros(rows * RANK, dtype=torch.bfloat16, device="cuda")
    latent[: patterns.numel()] = patterns
    rotated = torch.zeros(rows * ROPE, dtype=torch.bfloat16, device="cuda")
    rotated[: patterns.numel()] = patterns
    q_nope, q_pe = latent.view(rows, 1, RANK), rotated.view(rows, 1, ROPE)
    _check(_packed(runnable, axes, q_nope, q_pe), q_nope, q_pe)


@pytest.mark.gfx950
@pytest.mark.parametrize(
    "value",
    [0.0, -0.0, 2.0**-133, -(2.0**-133), 2.0**-9, 1.0625, 1.1875, 448.0, -448.0, 464.0, -464.0, 466.0, -466.0]
    + [float("inf"), float("-inf"), float("nan")],
)
def test_the_committed_kernel_casts_edge_values_as_torch_does(builder_registry, value):
    """Signed zeros, a bf16 subnormal, fp8's smallest subnormal, rounding ties, its largest value and the values
    around 464, where torch's cast turns to the NaN code, infinities and a NaN, in every slot of both operands."""
    runnable, axes = _kernel(builder_registry, 8)
    rows = axes["n_rows"]
    q_nope = torch.full((rows, 1, RANK), value, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.full((rows, 1, ROPE), -value, dtype=torch.bfloat16, device="cuda")
    _check(_packed(runnable, axes, q_nope, q_pe), q_nope, q_pe)


@pytest.mark.gfx950
@pytest.mark.parametrize(
    ("n_tokens", "nope_pitch", "pe_pitch"),
    [
        (
            384,
            32768,
            192,
        ),  # the 384-token workload's latent rows 64 KiB apart: 2.4 GB from the first to the last
        (8, 6_000_000, 6_000_000),  # rows 12 MB apart: the last ones past 2**32 bytes of the first
        (8, 577, 193),  # rows 2 bytes past a 16-byte boundary in turn
    ],
    ids=["past_2gib", "past_4gib", "unaligned_rows"],
)
def test_the_committed_kernel_reads_rows_any_distance_apart(builder_registry, n_tokens, nope_pitch, pe_pitch):
    runnable, axes = _kernel(builder_registry, n_tokens)
    q_nope = _rows_apart(axes["n_rows"], RANK, nope_pitch)
    q_pe = _rows_apart(axes["n_rows"], ROPE, pe_pitch)
    _check(_packed(runnable, axes, q_nope, q_pe), q_nope, q_pe)
