# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DCP query all-gather route on CPU, with stand-ins for vLLM's DCP and TP groups: the TP group's gather it
asks for is laid out as the DCP group's all_gather(dim=1), byte for byte (bf16 and fp8 queries); what it does not
take goes back to vLLM; a TP gather of another shape, a first eager call that disagrees with the DCP group's own
gather, and AKLR_DISABLE each leave the gather with vLLM."""

import sys
import types

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.layers import dcp

W, M, H, D = 8, 10, 12, 576


class _Gather:
    """A group's all_gather along a dim over every rank's input, as vLLM's communicators return it; ``calls`` counts
    them."""

    def __init__(self, ranks_inputs):
        self.ranks_inputs = ranks_inputs
        self.calls = 0

    def all_gather(self, input_, dim):
        self.calls += 1
        return torch.cat([x.contiguous().view(input_.dtype) for x in self.ranks_inputs], dim=dim)


def _groups(monkeypatch, ranks_inputs, dcp_inputs=None, capturing=False):
    """(the DCP group, the TP group): the DCP group's own communicator gathers ``dcp_inputs`` (default: the same),
    the TP group's, with vLLM's aiter gather on, ``ranks_inputs``."""
    dcp_group = types.SimpleNamespace(
        world_size=W,
        ranks=list(range(W)),
        device_group=None,
        device_communicator=_Gather(dcp_inputs or ranks_inputs),
    )
    gather = _Gather(ranks_inputs)
    tp_group = types.SimpleNamespace(
        world_size=W,
        ranks=list(range(W)),
        device_communicator=types.SimpleNamespace(use_aiter_tp_all_gather=True),
        all_gather=gather.all_gather,
        gather=gather,
    )
    state = types.ModuleType("vllm.distributed.parallel_state")
    state.get_dcp_group = lambda: dcp_group
    state.get_tp_group = lambda: tp_group
    for name in ("vllm", "vllm.distributed"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", state)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None, group=None: None)
    monkeypatch.setattr(dcp, "_CHECKED", [])
    monkeypatch.delenv("AKLR_DISABLE", raising=False)
    dcp._enabled.cache_clear()
    return dcp_group, tp_group


def _inputs(dtype):
    torch.manual_seed(0)
    return [torch.randn(M, H, D).to(dtype) for _ in range(W)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_the_tp_groups_gather_lays_ranks_out_along_the_head_dim_as_the_dcp_groups(monkeypatch, dtype):
    xs = _inputs(dtype)
    group, tp = _groups(monkeypatch, xs)
    out = dcp._dcp_query_all_gather(group, xs[3], dim=1)
    ref = torch.cat(xs, dim=1)
    assert out.shape == (M, W * H, D) and out.dtype == dtype
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
    assert dcp._CHECKED == [True]
    # the first eager call alone is compared with the DCP group's own gather
    dcp._dcp_query_all_gather(group, xs[3], dim=1)
    assert (tp.gather.calls, group.device_communicator.calls) == (2, 1)


def test_what_it_does_not_take_goes_back_to_vllm(monkeypatch):
    xs = _inputs(torch.bfloat16)
    group, tp = _groups(monkeypatch, xs)
    assert dcp._dcp_query_all_gather(group, xs[0], dim=0) is FallbackReason.OUT_OF_CONTRACT
    assert dcp._dcp_query_all_gather(group, xs[0][0], dim=1) is FallbackReason.OUT_OF_CONTRACT
    other = types.SimpleNamespace(world_size=W, ranks=list(range(W)))
    assert dcp._dcp_query_all_gather(other, xs[0], dim=1) is FallbackReason.OUT_OF_CONTRACT
    odd = torch.zeros(M, H, D - 1, dtype=torch.float8_e4m3fn)
    assert dcp._dcp_query_all_gather(group, odd, dim=1) is FallbackReason.OUT_OF_CONTRACT
    assert dcp._dcp_query_all_gather(group, xs[0].to(torch.int32), dim=1) is FallbackReason.OUT_OF_CONTRACT
    # the TP group's own call of all_gather, which the hook wraps too
    assert dcp._dcp_query_all_gather(tp, xs[0], dim=1) is FallbackReason.OUT_OF_CONTRACT
    assert tp.gather.calls == 0


@pytest.mark.parametrize("why", ["other ranks", "no aiter gather on the TP group"])
def test_a_tp_group_that_cannot_do_the_gather_faster_leaves_it_with_vllm(monkeypatch, why):
    xs = _inputs(torch.bfloat16)
    group, tp = _groups(monkeypatch, xs)
    if why == "other ranks":
        tp.ranks = list(range(W, 2 * W))
    else:
        tp.device_communicator.use_aiter_tp_all_gather = False
    assert dcp._dcp_query_all_gather(group, xs[0], dim=1) is FallbackReason.OUT_OF_CONTRACT
    assert tp.gather.calls == 0


def test_a_tp_gather_of_another_shape_goes_back_to_vllm(monkeypatch):
    """A gather that hands back its input's shape (an aiter that returns zeros in a capture's warm-up forward)."""
    xs = _inputs(torch.bfloat16)
    group, tp = _groups(monkeypatch, xs)
    tp.all_gather = lambda input_, dim: torch.zeros_like(input_)
    assert dcp._dcp_query_all_gather(group, xs[0], dim=1) is FallbackReason.OUT_OF_CONTRACT
    assert dcp._CHECKED == []


def test_a_first_call_that_disagrees_with_the_dcp_groups_gather_turns_the_route_off(monkeypatch):
    xs = _inputs(torch.bfloat16)
    wrong = [x.clone() for x in xs]
    wrong[5][0, 0, 0] += 1
    group, _ = _groups(monkeypatch, xs, dcp_inputs=wrong)
    out = dcp._dcp_query_all_gather(group, xs[0], dim=1)
    assert torch.equal(out, torch.cat(wrong, dim=1))  # the check's call returns the DCP group's own result
    assert dcp._CHECKED == [False]
    assert dcp._dcp_query_all_gather(group, xs[0], dim=1) is FallbackReason.INACCURATE


def test_a_capture_is_not_where_the_gathers_are_compared(monkeypatch):
    xs = _inputs(torch.bfloat16)
    group, _ = _groups(monkeypatch, xs, capturing=True)
    out = dcp._dcp_query_all_gather(group, xs[0], dim=1)
    assert torch.equal(out, torch.cat(xs, dim=1))
    assert dcp._CHECKED == [] and group.device_communicator.calls == 0


def test_aklr_disable_names_the_route(monkeypatch):
    xs = _inputs(torch.bfloat16)
    group, tp = _groups(monkeypatch, xs)
    monkeypatch.setenv("AKLR_DISABLE", f"gemm@1.0,{dcp.ROUTE_ID}")
    dcp._enabled.cache_clear()
    try:
        assert dcp._dcp_query_all_gather(group, xs[0], dim=1) is FallbackReason.DISABLED
        assert tp.gather.calls == 0
    finally:
        monkeypatch.delenv("AKLR_DISABLE")
        dcp._enabled.cache_clear()
