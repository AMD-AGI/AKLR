# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route on sglang 0.5.24.dev809's _k3_bf16_gemm (Kimi-K3.1-Pro's image), on stand-ins of the sglang modules it
reads: the fp32 MoE front reaches kimi_k3_moe_front_gemm only where sglang's own ROCm routing leaves it to torch;
every other call stays sglang's. The op is a recorder."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.kimi_k3 import api as kimi_k3

TOKENS, COLUMNS, HIDDEN = 4, 64, 256


@pytest.fixture
def sglang(monkeypatch):
    """sglang's kimi_k3 on ROCm and its bf16 GEMM backend; ``hip[0]`` is whether k3_bf16_gemm_hip takes a call."""
    hip, backend = [False], SimpleNamespace(cutedsl=False)

    def module(name, **attrs):
        m = ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    def k3_bf16_gemm_hip(x, weight, out):
        return out.fill_(1.0) if hip[0] else None

    model = module("sglang.srt.models.kimi_k3", _is_hip=True, k3_bf16_gemm_hip=k3_bf16_gemm_hip)
    module("sglang.srt.models", kimi_k3=model)
    unquant = module(
        "sglang.srt.layers.quantization.unquant",
        get_bf16_gemm_backend=lambda: SimpleNamespace(is_cutedsl=lambda: backend.cutedsl),
    )
    module("sglang.srt.layers.quantization", unquant=unquant)
    return SimpleNamespace(hip=hip, backend=backend)


@pytest.fixture
def op(monkeypatch):
    """The op's calls, each answered with its ``out``."""
    calls = []

    def record(**call):
        calls.append(call)
        return call["out"]

    monkeypatch.setattr(kimi_k3, "kimi_k3_moe_front_gemm", record)
    return calls


def operands():
    """hidden_states and the merged front weight, bf16."""
    x = torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16)
    return x, torch.zeros(COLUMNS, HIDDEN, dtype=torch.bfloat16)


def test_the_fp32_front_sglang_leaves_to_torch_reaches_the_op(sglang, op):
    """_fused_front's call off an fp32 front, which k3_bf16_gemm_hip declines: the op writes the fp32 output the
    route allocates, as sglang's torch.mm would."""
    x, weight = operands()
    result = kimi_k3._k3_bf16_gemm(x, weight, out_dtype=torch.float32)
    (call,) = op
    assert call["x"] is x and call["weight"] is weight and call["out_dtype"] == torch.float32
    assert call["out"].dtype == torch.float32 and call["out"].shape == (TOKENS, COLUMNS)
    assert result is call["out"]


def test_the_call_as_sglang_makes_it_is_a_point_of_the_op(sglang, op):
    """K3.1's front at TP8 (6016 x 7168): the op's call has a FastKey, at its m, n and k."""
    from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
    from aklr.plugins.sglang.models.kimi_k3.contract import KIMI_K3_MOE_FRONT_GEMM

    x, weight = torch.zeros(5, 7168, dtype=torch.bfloat16), torch.zeros(6016, 7168, dtype=torch.bfloat16)
    kimi_k3._k3_bf16_gemm(x, weight, out_dtype=torch.float32)
    (call,) = op
    codec = FastKeyCodec.from_contract(KIMI_K3_MOE_FRONT_GEMM)
    fast_key = FastKeyFnGenerator(codec, None).generate()
    key = fast_key({p.name: call.get(p.name, p.default) for p in KIMI_K3_MOE_FRONT_GEMM.api_params})
    assert key is not None
    assert codec.to_point(key).axes == {"m": 5, "n": 6016, "k": 7168}


def test_what_aiters_tile_takes_stays_with_it(sglang, op):
    """Where k3_bf16_gemm_hip runs aiter's FlyDSL tile, its output is the result and the op is not asked."""
    sglang.hip[0] = True
    x, weight = operands()
    result = kimi_k3._k3_bf16_gemm(x, weight, out_dtype=torch.float32)
    assert op == [] and result.dtype == torch.float32 and bool((result == 1.0).all())


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"out_dtype": torch.bfloat16}, {"out": torch.empty(TOKENS, COLUMNS, dtype=torch.bfloat16)}],
    ids=["bf16_front", "bf16_out_dtype", "shared_down_into_out"],
)
def test_other_calls_stay_on_sglang(sglang, op, kwargs):
    """A bf16 output, or an output given (the shared experts' down projection), is not the op's call."""
    x, weight = operands()
    assert kimi_k3._k3_bf16_gemm(x, weight, **kwargs) is FallbackReason.OUT_OF_CONTRACT
    assert op == []


def test_the_cutedsl_backend_stays_on_sglang(sglang, op):
    sglang.backend.cutedsl = True
    x, weight = operands()
    assert kimi_k3._k3_bf16_gemm(x, weight, out_dtype=torch.float32) is FallbackReason.OUT_OF_CONTRACT
    assert op == []


def test_the_route_is_declared_for_kimi_k31s_image_only():
    from aklr.plugins.sglang import hook

    declared = [versions for target, versions, _ in hook._ROUTES if target.endswith("kimi_k3._k3_bf16_gemm")]
    assert declared == [("0.5.24.dev809",)]
