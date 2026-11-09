# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Qwen3.5 GDN route's choice of op, on stand-ins of sglang 0.5.22: a block-FP8 layer (the
FP8 checkpoint) decodes on the FP8 op first, an unquantized one (the MXFP4 checkpoint) on the
BF16 op alone, on contiguous projections."""

import enum
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.qwen3_5 import api

ROWS = 4


class ForwardMode(enum.Enum):
    DECODE = 1
    EXTEND = 2


class Fp8LinearMethod:
    block_quant, use_mxfp8, block_fp8_as_mxfp8 = True, False, False

    def __init__(self):
        self.w8a8_block_fp8_linear = aiter_w8a8_block_fp8_linear


class UnquantizedLinearMethod:
    pass


def aiter_w8a8_block_fp8_linear(*args, **kwargs):
    raise AssertionError("out_proj's GEMM is a stand-in")


class Op:
    """An op's api: takes every call, records what it was handed."""

    def __init__(self, result, ask=None):
        self.result, self.ask, self.asked, self.calls = result, ask, 0, []

    def fallback_reason(self, **kwargs):
        self.asked += 1
        return self.ask

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class Linear:
    def __init__(self, method):
        self.quant_method, self.weight, self.inputs = method, torch.empty(4096, 2048), []

    def __call__(self, x):
        self.inputs.append(x)
        return "out_proj output", None


class Layer:
    """A GDN layer at TP4 whose projections are one packed GEMM's column slices, as sglang's
    BF16 in_proj hands them for decode batches of up to 64 rows."""

    def __init__(self, methods):
        self.in_proj_qkvz, self.in_proj_ba, self.out_proj = (Linear(m) for m in methods)
        self.attn = SimpleNamespace(
            activation="silu", layer_id=0, head_k_dim=128, conv_weights=torch.zeros(3072, 4)
        )
        self.norm = SimpleNamespace(activation="swish", eps=1e-6, weight=torch.zeros(128))
        self.A_log, self.dt_bias = torch.zeros(16), torch.zeros(16)
        packed = torch.randn(ROWS, 5120 + 32).to(torch.bfloat16)
        self.projections = packed[:, :5120], packed[:, 5120:]

    def _forward_input_proj(self, hidden_states):
        return self.projections


@pytest.fixture
def sglang(monkeypatch):
    """sglang 0.5.22's modules the route imports, as stand-ins; returns its attention backend."""
    cache = SimpleNamespace(
        conv=[torch.zeros(9, 3072, 3)],
        temporal=torch.zeros(9, 16, 128, 128),
        replayssm_d=None,
        replayssm_k=None,
        replayssm_g=None,
    )
    metadata = SimpleNamespace(
        mamba_cache_indices=torch.zeros(ROWS, dtype=torch.int32),
        replayssm_write_pos=None,
        replayssm_force_flush=None,
    )
    backend = SimpleNamespace(
        forward_metadata=metadata,
        req_to_token_pool=SimpleNamespace(mamba2_layer_cache=lambda layer_id: cache),
        tracked=[],
    )
    backend._track_mamba_state_decode = lambda *args: backend.tracked.append(args)
    modules = {
        "sglang.srt.model_executor.forward_batch_info": {"ForwardMode": ForwardMode},
        "sglang.srt.model_executor.forward_context": {
            "get_attn_backend": lambda: SimpleNamespace(linear_attn_backend=backend)
        },
        "sglang.srt.runtime_context": {"get_lora": lambda: SimpleNamespace(enable_lora=False)},
        "sglang.srt.layers.quantization.fp8": {"Fp8LinearMethod": Fp8LinearMethod},
        "sglang.srt.layers.quantization.fp8_utils": {
            "aiter_w8a8_block_fp8_linear": aiter_w8a8_block_fp8_linear,
            "use_aiter_triton_gemm_w8a8_tuned_gfx950": lambda n, k: False,
        },
        "sglang.srt.layers.quantization.unquant": {"UnquantizedLinearMethod": UnquantizedLinearMethod},
    }
    for name, attrs in modules.items():
        stand_in = ModuleType(name)
        stand_in.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stand_in)
    return backend


@pytest.fixture
def ops(monkeypatch):
    """The two decode ops, each taking every call."""
    fp8 = Op((torch.zeros(ROWS, 2048), torch.zeros(ROWS, 16)))
    bf16 = Op(torch.zeros(ROWS, 16, 128, dtype=torch.bfloat16))
    monkeypatch.setattr(api, "gdn_decode_group_fp8_quant", fp8)
    monkeypatch.setattr(api, "gdn_decode", bf16)
    return fp8, bf16


def forward(layer, mode=ForwardMode.DECODE):
    return api._gdn_forward(layer, torch.zeros(ROWS, 4096), SimpleNamespace(forward_mode=mode))


@pytest.mark.parametrize("fp8_declines", [False, True], ids=["fp8_op", "bf16_op"])
def test_a_block_fp8_layer_decodes_on_the_fp8_op_first(sglang, ops, fp8_declines):
    """The FP8 checkpoint's layer: the FP8 op, else the BF16 op, both handed sglang's
    projections as they are."""
    fp8, bf16 = ops
    fp8.ask = FallbackReason.NO_WORKLOAD if fp8_declines else None
    layer = Layer((Fp8LinearMethod(), UnquantizedLinearMethod(), Fp8LinearMethod()))
    assert forward(layer) == "out_proj output"
    served = bf16 if fp8_declines else fp8
    (call,) = served.calls
    assert call["projected_qkvz"] is layer.projections[0] and call["projected_ba"] is layer.projections[1]
    assert (fp8.asked, bf16.asked) == ((1, 1) if fp8_declines else (1, 0))
    (projected,) = layer.out_proj.inputs
    if fp8_declines:
        assert projected.shape == (ROWS, 2048)
    else:
        assert projected is fp8.result
    assert len(sglang.tracked) == 1


def test_an_unquantized_layer_decodes_on_the_bf16_op_alone(sglang, ops):
    """The MXFP4 checkpoint's layer: the BF16 op only, handed contiguous copies of the packed
    projections; out_proj takes its BF16 output."""
    fp8, bf16 = ops
    layer = Layer((UnquantizedLinearMethod(),) * 3)
    assert forward(layer) == "out_proj output"
    assert fp8.asked == 0 and not fp8.calls and bf16.asked == 1
    (call,) = bf16.calls
    for name, projection in zip(("projected_qkvz", "projected_ba"), layer.projections, strict=True):
        assert call[name].is_contiguous() and not projection.is_contiguous()
        assert torch.equal(call[name], projection)
    (projected,) = layer.out_proj.inputs
    assert projected.shape == (ROWS, 2048) and projected.dtype == torch.bfloat16
    assert len(sglang.tracked) == 1


@pytest.mark.parametrize(
    ("methods", "mode"),
    [
        ((UnquantizedLinearMethod(),) * 3, ForwardMode.EXTEND),
        ((UnquantizedLinearMethod(), UnquantizedLinearMethod(), object()), ForwardMode.DECODE),
        ((object(), UnquantizedLinearMethod(), UnquantizedLinearMethod()), ForwardMode.DECODE),
    ],
    ids=["unquantized_prefill", "other_out_proj", "other_in_proj"],
)
def test_other_layers_stay_on_sglang(sglang, ops, methods, mode):
    """Prefill of an unquantized layer (no BF16 prefill op), and projections quantized other
    than block FP8 on out_proj: sglang's own path, before any op or projection runs."""
    assert forward(Layer(methods), mode) is FallbackReason.OUT_OF_CONTRACT
    assert all(op.asked == 0 and not op.calls for op in ops)
