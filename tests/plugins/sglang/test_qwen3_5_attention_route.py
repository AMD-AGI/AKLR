# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Qwen3.5 attention route's choice of op, on stand-ins of sglang 0.5.22: decode on the decode
ops, the prefill of fresh sequences on the prefill ops with the backend's sequence boundaries and
longest sequence, and every other extend batch on sglang's own path."""

import enum
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.qwen3_5 import api

LENGTHS = [3, 5]
ROWS = sum(LENGTHS)


class ForwardMode(enum.Enum):
    DECODE = 1
    EXTEND = 2
    MIXED = 3

    def is_decode(self):
        return self is ForwardMode.DECODE


class Op:
    """An op's api: takes every call, records what it was handed."""

    def __init__(self, result):
        self.result, self.asked, self.calls = result, 0, []

    def fallback_reason(self, **kwargs):
        self.asked += 1

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class Linear:
    def __init__(self, output=None):
        self.output, self.inputs = output, []

    def __call__(self, x):
        self.inputs.append(x)
        return (self.output if self.output is not None else "o_proj output"), None


def _layer():
    norm = SimpleNamespace(weight=torch.zeros(256), variance_epsilon=1e-6)
    return SimpleNamespace(
        q_norm=norm,
        k_norm=norm,
        q_size=2048,
        kv_size=256,
        attn=SimpleNamespace(layer_id=0),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(64, 64)),
        qkv_proj=Linear(torch.zeros(ROWS, 4608, dtype=torch.bfloat16)),
        o_proj=Linear(),
    )


@pytest.fixture
def backend(monkeypatch):
    """sglang 0.5.22's modules the route imports, as stand-ins; returns the attention backend."""
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda layer_id: ("keys", "values")),
        forward_metadata=SimpleNamespace(kv_indptr="kv_indptr", kv_indices="kv_indices"),
        qo_indptr=torch.tensor([0, 3, 8, 0, 0], dtype=torch.int32),
        scale=256**-0.5,
    )
    model = {
        "_is_hip": True,
        "_use_aiter": True,
        "_fp8_tuple_input": lambda o_proj, rows: False,
    }
    modules = {
        "sglang.srt.models.qwen3_5": model,
        "sglang.srt.model_executor.forward_batch_info": {"ForwardMode": ForwardMode},
        "sglang.srt.model_executor.forward_context": {"get_attn_backend": lambda: backend},
    }
    for name, attrs in modules.items():
        stand_in = ModuleType(name)
        stand_in.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stand_in)
    # the route imports the model module from its package
    models = ModuleType("sglang.srt.models")
    models.qwen3_5 = sys.modules["sglang.srt.models.qwen3_5"]
    monkeypatch.setitem(sys.modules, "sglang.srt.models", models)
    monkeypatch.setattr(api, "_attention_falls_back", lambda layer, backend: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return backend


@pytest.fixture
def ops(monkeypatch):
    """The four attention ops, each taking every call."""
    made = {
        "gated_attention_decode_group_fp8_quant": Op((torch.zeros(ROWS, 8, 256), torch.zeros(ROWS, 16))),
        "gated_attention_decode": Op(torch.zeros(ROWS, 2048)),
        "gated_attention_prefill_group_fp8_quant": Op((torch.zeros(ROWS, 8, 256), torch.zeros(ROWS, 16))),
        "gated_attention_prefill": Op(torch.zeros(ROWS, 2048)),
    }
    for name, op in made.items():
        monkeypatch.setattr(api, name, op)
    return made


def _batch(mode=ForwardMode.EXTEND, prefixes=(0, 0), **overrides):
    batch = SimpleNamespace(
        forward_mode=mode,
        spec_info=None,
        out_cache_loc=torch.arange(ROWS),
        extend_seq_lens_cpu=LENGTHS,
        extend_prefix_lens_cpu=list(prefixes),
        batch_size=len(LENGTHS),
        attn_attend_prefix_cache=False,
    )
    batch.__dict__.update(overrides)
    return batch


def forward(layer, batch):
    return api._attention_self_attention(
        layer, torch.zeros(3, ROWS, dtype=torch.int64), torch.zeros(ROWS, 4096), batch
    )


@pytest.mark.parametrize("fp8_groups", [True, False], ids=["fp8_o_proj", "bf16_o_proj"])
def test_fresh_sequences_prefill_on_the_prefill_ops(backend, ops, monkeypatch, fp8_groups):
    """A block-FP8 o_proj takes the FP8 prefill op's values and scales, a BF16 one the BF16 op's
    rows; both handed the backend's boundaries of the batch's sequences and the longest one."""
    monkeypatch.setattr(api, "_takes_fp8_groups", lambda linear: fp8_groups)
    layer = _layer()
    layer.o_proj.weight = SimpleNamespace()
    assert forward(layer, _batch()) == "o_proj output"
    name = "gated_attention_prefill_group_fp8_quant" if fp8_groups else "gated_attention_prefill"
    (call,) = ops[name].calls
    assert all(not op.calls for n, op in ops.items() if n != name)
    assert torch.equal(call["cu_seqlens"], torch.tensor([0, 3, 8], dtype=torch.int32))
    assert call["max_seq_len"] == max(LENGTHS) and "kv_indices" not in call
    (projected,) = layer.o_proj.inputs
    if fp8_groups:
        assert projected[0].shape == (ROWS, 2048) and projected[1] is ops[name].result[1]
    else:
        assert projected is ops[name].result


def test_decode_stays_on_the_decode_ops(backend, ops, monkeypatch):
    """Decode: the decode op, handed the backend's KV indices."""
    monkeypatch.setattr(api, "_takes_fp8_groups", lambda linear: False)
    layer = _layer()
    assert forward(layer, _batch(ForwardMode.DECODE)) == "o_proj output"
    (call,) = ops["gated_attention_decode"].calls
    assert call["kv_indices"] == "kv_indices" and "cu_seqlens" not in call


@pytest.mark.parametrize(
    "batch",
    [
        _batch(prefixes=(0, 4)),
        _batch(attn_attend_prefix_cache=True),
        _batch(extend_prefix_lens_cpu=None),
        _batch(extend_seq_lens_cpu=[3, 4]),
        _batch(ForwardMode.MIXED),
    ],
    ids=["prefix", "prefix_chunk", "no_host_prefixes", "padded_rows", "mixed"],
)
def test_other_extend_batches_stay_on_sglang(backend, ops, monkeypatch, batch):
    """A cached prefix (chunked prefill past its first chunk), rows the host lengths do not
    cover, or another mode than extend: sglang's own path, before any op or projection runs."""
    monkeypatch.setattr(api, "_takes_fp8_groups", lambda linear: True)
    layer = _layer()
    layer.o_proj.weight = SimpleNamespace()
    assert forward(layer, batch) is FallbackReason.OUT_OF_CONTRACT
    assert all(op.asked == 0 and not op.calls for op in ops.values())
    assert not layer.qkv_proj.inputs


def test_a_captured_prefill_stays_on_sglang(backend, ops, monkeypatch):
    """A prefill in a graph capture: the capture would replay its lengths."""
    monkeypatch.setattr(api, "_takes_fp8_groups", lambda linear: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert forward(_layer(), _batch()) is FallbackReason.OUT_OF_CONTRACT
    assert all(op.asked == 0 for op in ops.values())
