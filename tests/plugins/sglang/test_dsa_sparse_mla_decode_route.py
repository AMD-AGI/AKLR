# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the DSA decode route tells its op, and when it falls back to sglang."""

from types import SimpleNamespace

import pytest
import torch

import aklr.plugins.sglang.layers.dsa_backend.api as dsa_backend
from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.layers.dsa_backend.contract import DSA_SPARSE_MLA_DECODE

FP8 = torch.float8_e4m3fn
# GLM-5.2's pool rows and top-k; sglang's pool keeps one page past its slots
KV_DIM, LATENT, TOPK, POOL_ROWS = 576, 512, 2048, 4096 + 64


def contract_point(arguments):
    """The Point dispatch keys the op call at, on any device; None outside the contract."""
    codec = FastKeyCodec.from_contract(DSA_SPARSE_MLA_DECODE)
    call = {p.name: p.default for p in DSA_SPARSE_MLA_DECODE.api_params} | arguments
    key = FastKeyFnGenerator(codec, None).generate()(call)
    return None if key is None else codec.to_point(key)


@pytest.fixture
def op(monkeypatch):
    """The op calls the route makes; each returns 3s as [tokens, heads, latent]."""
    calls = []

    def fake(**arguments):
        calls.append(arguments)
        q = arguments["q"]
        return torch.full((*q.shape[:2], LATENT), 3.0, dtype=torch.bfloat16)

    monkeypatch.setattr(dsa_backend, "dsa_sparse_mla_decode", fake)
    return calls


def decode_call(tokens=4, heads=16, device="cpu"):
    """forward_decode's call of _forward_triton_decode, as sglang 0.5.22 makes it on HIP:
    q_nope and q_rope views of q_all, the fused RoPE and cache write's fp8 output, the
    layer's [rows, 1, kv_dim] pool and the top-k slots."""
    q_all = torch.zeros((tokens, heads, KV_DIM), dtype=FP8, device=device)
    return q_all, {
        "q_nope": q_all[:, :, :LATENT],
        "q_rope": q_all[:, :, LATENT:],
        "kv_cache": torch.zeros((POOL_ROWS, 1, KV_DIM), dtype=FP8, device=device),
        "v_head_dim": LATENT,
        "page_table_1": torch.full((tokens, TOPK), -1, dtype=torch.int32),
        "sm_scale": 0.0625,
    }


@pytest.mark.parametrize("heads", [16, 8], ids=["tp4", "tp8"])
def test_decode_reaches_its_op_on_the_whole_query(sglang, op, heads):
    q_all, call = decode_call(heads=heads)
    result = dsa_backend._forward_triton_decode(SimpleNamespace(), **call)
    (arguments,) = op
    q, kv_cache = arguments["q"], arguments["kv_cache"]
    assert q.shape == q_all.shape and q.is_contiguous()
    assert q.data_ptr() == q_all.data_ptr()
    assert kv_cache.shape == (POOL_ROWS, KV_DIM)
    assert kv_cache.data_ptr() == call["kv_cache"].data_ptr()
    assert arguments["topk_indices"] is call["page_table_1"]
    assert arguments["sm_scale"] == 0.0625
    # sglang's kernel's [1, tokens, heads, v_head_dim]
    assert result.shape == (1, 4, heads, LATENT) and bool(result.eq(3).all())
    point = contract_point(arguments)
    assert point is not None
    assert point.axes == {
        "n_tokens": 4,
        "n_heads": heads,
        "kv_dim": KV_DIM,
        "topk": TOPK,
        "n_slots": POOL_ROWS,
    }


def test_speculative_decoding_falls_back_to_sglang(sglang, op):
    sglang.spec.speculative_algorithm = "EAGLE"
    _, call = decode_call()
    result = dsa_backend._forward_triton_decode(SimpleNamespace(), **call)
    assert result is FallbackReason.OUT_OF_CONTRACT and op == []


@pytest.mark.parametrize("apart", ["tensors", "offset", "strides"])
def test_a_query_not_held_whole_falls_back_to_sglang(sglang, op, apart):
    """q_nope and q_rope as two tensors, or views that are not one row's two parts."""
    q_all, call = decode_call()
    if apart == "tensors":
        call["q_nope"] = torch.zeros((4, 16, LATENT), dtype=FP8)
        call["q_rope"] = torch.zeros((4, 16, KV_DIM - LATENT), dtype=FP8)
    elif apart == "offset":
        call["q_rope"] = q_all[:, :, LATENT - 64 : LATENT]
    else:
        wide = torch.zeros((4, 16, KV_DIM + 64), dtype=FP8)
        call["q_nope"], call["q_rope"] = wide[..., :LATENT], wide[..., LATENT:KV_DIM]
    result = dsa_backend._forward_triton_decode(SimpleNamespace(), **call)
    assert result is FallbackReason.OUT_OF_CONTRACT and op == []


def test_the_ops_fallback_reason_reaches_sglang(sglang, monkeypatch):
    monkeypatch.setattr(
        dsa_backend,
        "dsa_sparse_mla_decode",
        lambda **arguments: FallbackReason.NO_WORKLOAD,
    )
    _, call = decode_call()
    result = dsa_backend._forward_triton_decode(SimpleNamespace(), **call)
    assert result is FallbackReason.NO_WORKLOAD


def call_arguments(q, kv_cache, topk_indices):
    return {"q": q, "kv_cache": kv_cache, "topk_indices": topk_indices}


def meta(*shape, dtype=FP8):
    return torch.empty(shape, dtype=dtype, device="meta")


@pytest.mark.parametrize(
    "change",
    ["bf16 q", "bf16 pool", "rope 32", "pool past 2^31 bytes", "strided top-k"],
)
def test_the_contract_declines_what_the_kernels_do_not_read(change):
    """The route passes these on; the contract leaves them to sglang: no Point, or one
    outside the domain (which no schema serves)."""
    q, kv_cache = meta(4, 16, KV_DIM), meta(POOL_ROWS, KV_DIM)
    topk_indices = meta(4, TOPK, dtype=torch.int32)
    assert contract_point(call_arguments(q, kv_cache, topk_indices)) is not None
    if change == "bf16 q":
        q = meta(4, 16, KV_DIM, dtype=torch.bfloat16)
    elif change == "bf16 pool":
        kv_cache = meta(POOL_ROWS, KV_DIM, dtype=torch.bfloat16)
    elif change == "rope 32":
        q, kv_cache = meta(4, 16, LATENT + 32), meta(POOL_ROWS, LATENT + 32)
    elif change == "pool past 2^31 bytes":
        kv_cache = meta((2**31 - 2) // KV_DIM + 1, KV_DIM)
    else:
        topk_indices = meta(4, 2 * TOPK, dtype=torch.int32)[:, :TOPK]
    point = contract_point(call_arguments(q, kv_cache, topk_indices))
    assert point is None or not DSA_SPARSE_MLA_DECODE.domain.contains(point)


def test_the_baseline_calls_sglangs_kernel_as_the_backend_does(monkeypatch):
    """dsa_sparse_mla_decode_baseline hands triton_sparse_mla_decode_splitk what
    _forward_triton_decode does and returns its [tokens, heads, latent] output, with or
    without the output passed."""
    import sys
    from types import ModuleType

    from aklr.plugins.sglang.layers.dsa_backend.baseline import (
        dsa_sparse_mla_decode_baseline,
    )

    calls = []

    def splitk(q_nope, q_rope, kv, indices, sm_scale, d_v=512, workspace=None):
        calls.append(locals())
        return torch.full((1, *q_nope.shape[:2], d_v), 2.0, dtype=torch.bfloat16)

    module = ModuleType("sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode")
    module.triton_sparse_mla_decode_splitk = splitk
    monkeypatch.setitem(sys.modules, module.__name__, module)
    q = torch.zeros((4, 16, KV_DIM), dtype=FP8)
    kv_cache = torch.zeros((POOL_ROWS, KV_DIM), dtype=FP8)
    topk_indices = torch.full((4, TOPK), -1, dtype=torch.int32)
    output = torch.empty((4, 16, LATENT), dtype=torch.bfloat16)
    for given in (None, output):
        result = dsa_sparse_mla_decode_baseline(q, kv_cache, topk_indices, 0.0625, output=given)
        assert result.shape == (4, 16, LATENT) and bool(result.eq(2).all())
    for call in calls:
        assert call["q_nope"].shape == (4, 16, LATENT)
        assert call["q_rope"].data_ptr() == q.data_ptr() + LATENT
        assert call["kv"].shape == (POOL_ROWS, 1, KV_DIM)
        assert call["indices"].shape == (4, 1, TOPK)
        assert call["sm_scale"] == 0.0625 and call["d_v"] == LATENT
