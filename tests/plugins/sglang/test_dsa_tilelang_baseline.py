# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""dsa_sparse_mla_decode's and dsa_sparse_mla_prefill's native "tilelang": its schemas time sglang's
tilelang_sparse_fwd as _forward_tilelang calls it, on a stand-in of sglang's tilelang kernel."""

import sys
from types import ModuleType

import pytest
import torch

from aklr.plugins.sglang.layers.dsa_backend.contract import DSA_SPARSE_MLA_DECODE, DSA_SPARSE_MLA_PREFILL


@pytest.mark.parametrize(
    "contract", [DSA_SPARSE_MLA_DECODE, DSA_SPARSE_MLA_PREFILL], ids=["decode", "prefill"]
)
def test_native_tilelang_times_tilelang_sparse_fwd(monkeypatch, contract):
    calls = []

    def tilelang_sparse_fwd(q, kv, indices, sm_scale, d_v=512):
        calls.append(dict(q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=d_v))
        return torch.ones(1, q.shape[0], q.shape[1], d_v, dtype=torch.bfloat16)

    module = ModuleType("sglang.kernels.ops.attention.dsa.tilelang_kernel")
    module.tilelang_sparse_fwd = tilelang_sparse_fwd
    monkeypatch.setitem(sys.modules, module.__name__, module)
    q = torch.zeros(3, 2, 576, dtype=torch.float8_e4m3fn)
    kv_cache = torch.zeros(100, 576, dtype=torch.float8_e4m3fn)
    topk_indices = torch.zeros(3, 100, dtype=torch.int32)
    result = contract.baseline.baseline(
        q=q, kv_cache=kv_cache, topk_indices=topk_indices, sm_scale=0.0625, native="tilelang"
    )
    ((kwargs,),) = [calls]
    assert kwargs["kv"].shape == (100, 1, 576) and kwargs["d_v"] == 512
    # padded with -1 to 64 entries a block, as _forward_tilelang pads it
    assert kwargs["indices"].shape == (3, 1, 128) and (kwargs["indices"][..., 100:] == -1).all()
    assert result["output"].shape == (3, 2, 512)


def test_native_defaults_to_triton():
    assert DSA_SPARSE_MLA_DECODE.attrs["native"].default == "triton"
    assert DSA_SPARSE_MLA_PREFILL.attrs["native"].default == "triton"
