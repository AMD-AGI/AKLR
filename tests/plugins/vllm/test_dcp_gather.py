# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DCP query all-gather route on CPU, with stand-ins for vLLM's groups and aiter's custom all-gather: the route's
layout equals vLLM's all_gather(dim=1) byte for byte (bf16 and fp8 queries), what it does not take goes back to
vLLM, and a first eager call that disagrees with vLLM's own gather turns the route off."""

import sys
import types

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.layers import dcp

W, M, H, D = 8, 10, 12, 576


class _CustomGather:
    """aiter's custom all-gather as the TP group's communicator holds it: every rank's input, concatenated on dim 0."""

    def __init__(self, ranks_inputs):
        self.ranks_inputs = ranks_inputs
        self.disabled = False

    def should_custom_ag(self, inp):
        return inp.dtype in (torch.bfloat16, torch.float16, torch.float32)

    def custom_all_gather(self, inp, dim=0):
        assert dim == 0
        return torch.cat([x.contiguous().view(inp.dtype) for x in self.ranks_inputs], dim=0)


class _PyncclGather:
    """vLLM's own all_gather along a dim (the pynccl path)."""

    def __init__(self, ranks_inputs):
        self.ranks_inputs = ranks_inputs

    def all_gather(self, input_, dim):
        return torch.cat(self.ranks_inputs, dim=dim)


def _groups(monkeypatch, ranks_inputs, vllm_inputs=None):
    dcp_group = types.SimpleNamespace(
        world_size=W,
        ranks=list(range(W)),
        device_group=None,
        device_communicator=_PyncclGather(vllm_inputs or ranks_inputs),
    )
    tp_group = types.SimpleNamespace(ranks=list(range(W)), device_communicator=types.SimpleNamespace(
        aiter_ar_comm=_CustomGather(ranks_inputs)))  # fmt: skip
    state = types.ModuleType("vllm.distributed.parallel_state")
    state.get_dcp_group = lambda: dcp_group
    state.get_tp_group = lambda: tp_group
    for name in ("vllm", "vllm.distributed"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", state)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None, group=None: None)
    monkeypatch.setattr(dcp, "_CHECKED", [])
    return dcp_group


def _inputs(dtype):
    torch.manual_seed(0)
    return [torch.randn(M, H, D).to(dtype) for _ in range(W)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_the_gather_lays_ranks_out_along_the_head_dim_as_vllm_does(monkeypatch, dtype):
    xs = _inputs(dtype)
    group = _groups(monkeypatch, xs)
    out = dcp._dcp_query_all_gather(group, xs[3], dim=1)
    ref = torch.cat(xs, dim=1)
    assert out.shape == (M, W * H, D) and out.dtype == dtype
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
    assert dcp._CHECKED == [True]


def test_what_it_does_not_take_goes_back_to_vllm(monkeypatch):
    xs = _inputs(torch.bfloat16)
    group = _groups(monkeypatch, xs)
    assert dcp._dcp_query_all_gather(group, xs[0], dim=0) is FallbackReason.OUT_OF_CONTRACT
    assert dcp._dcp_query_all_gather(group, xs[0][0], dim=1) is FallbackReason.OUT_OF_CONTRACT
    other = types.SimpleNamespace(world_size=W, ranks=list(range(W)))
    assert dcp._dcp_query_all_gather(other, xs[0], dim=1) is FallbackReason.OUT_OF_CONTRACT
    odd = torch.zeros(M, H, D - 1, dtype=torch.float8_e4m3fn)
    assert dcp._dcp_query_all_gather(group, odd, dim=1) is FallbackReason.OUT_OF_CONTRACT


def test_a_first_call_that_disagrees_with_vllms_gather_turns_the_route_off(monkeypatch):
    xs = _inputs(torch.bfloat16)
    wrong = [x.clone() for x in xs]
    wrong[5][0, 0, 0] += 1
    group = _groups(monkeypatch, xs, vllm_inputs=wrong)
    out = dcp._dcp_query_all_gather(group, xs[0], dim=1)
    assert torch.equal(out, torch.cat(wrong, dim=1))  # the check's call returns vLLM's own result
    assert dcp._CHECKED == [False]
    assert dcp._dcp_query_all_gather(group, xs[0], dim=1) is FallbackReason.INACCURATE
