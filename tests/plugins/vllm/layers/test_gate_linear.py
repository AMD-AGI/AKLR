# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's MoE router gate to AKLR's gemm: what is asked for and what goes back."""

from types import SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.layers import gate_linear


def gate(**changes):
    """DeepSeek-V4's gate as vLLM builds it on ROCm: a bf16 weight, fp32 logits, no bias."""
    layer = SimpleNamespace(
        weight=torch.zeros(384, 64, dtype=torch.bfloat16),
        out_dtype=torch.float32,
        allow_cublas_router_gemm=True,
        return_bias=True,
    )
    layer.__dict__.update(changes)
    return layer


@pytest.mark.parametrize("rows", [1, 32, 33, 4096], ids=lambda rows: f"{rows}_rows")
def test_a_bf16_gates_fp32_logits_are_one_gemm_at_any_row_count(monkeypatch, rows):
    """vLLM's own dispatch changes kernel past 32 rows; the op is asked the same way at each."""
    calls, logits = [], torch.ones(rows, 384)
    monkeypatch.setattr(gate_linear, "gemm", lambda a, b, **call: calls.append((a, b, call)) or logits)
    layer, x = gate(), torch.zeros(rows, 64, dtype=torch.bfloat16)
    out, bias = gate_linear._forward(layer, x)
    assert out is logits and bias is None
    ((a, b, call),) = calls
    assert a is x and b is layer.weight and call == {"out_dtype": torch.float32}


def test_a_gate_asked_for_logits_alone_gets_them_alone(monkeypatch):
    logits = torch.ones(4, 384)
    monkeypatch.setattr(gate_linear, "gemm", lambda a, b, **call: logits)
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    assert gate_linear._forward(gate(return_bias=False), x) is logits


def test_a_gate_no_kernel_serves_stays_on_vllm(monkeypatch):
    monkeypatch.setattr(gate_linear, "gemm", lambda a, b, **call: FallbackReason.NO_WORKLOAD)
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    assert gate_linear._forward(gate(), x) is FallbackReason.NO_WORKLOAD


def test_a_gate_on_another_of_vllms_tiers_is_not_asked_for(monkeypatch):
    """An fp32 weight, a bias, a quantized gate or bf16 logits: vLLM clears the flag for each."""
    monkeypatch.setattr(gate_linear, "gemm", lambda a, b, **call: pytest.fail("asked"))
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    assert gate_linear._forward(gate(allow_cublas_router_gemm=False), x) is FallbackReason.OUT_OF_CONTRACT
