# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's padding of the MLA decode query to aiter's head count, to AKLR's mla_query_heads: which calls of
AiterMLAHelper.get_mla_padded_q the route takes, what it hands over, and the committed schema."""

import inspect

import pytest
import torch

from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.layers.mla_query_heads import api
from aklr.plugins.vllm.layers.mla_query_heads.baseline import MlaQueryHeadsBaseline
from aklr.plugins.vllm.layers.mla_query_heads.reference import MlaQueryHeadsReference
from tests.op_case import OpCase

FP8 = torch.float8_e4m3fn


def query(tokens=3, heads=12, dim=32):
    gen = torch.Generator().manual_seed(0)
    return torch.randint(0, 255, (tokens, heads, dim), generator=gen, dtype=torch.uint8).view(FP8)


def vllm_tiled(q, m):
    """vLLM 0.34.0's get_mla_padded_q where m is no multiple of the heads and the rows are dense."""
    reps = -(-m // q.shape[1])
    return q.repeat(1, reps, 1)[:, :m, :].contiguous()


@pytest.fixture
def op(monkeypatch):
    """The op as the route calls it: its calls kept, the reference's output returned."""
    calls = []

    def mla_query_heads(q, out_heads=128, out_dtype=None, out=None):
        calls.append((q, out_heads))
        return MlaQueryHeadsReference().reference(q=q, out_heads=out_heads)["out"]

    monkeypatch.setattr(api, "mla_query_heads", mla_query_heads)
    return calls


def test_the_route_takes_vllms_parameters():
    """vLLM 0.34.0's AiterMLAHelper.get_mla_padded_q."""
    assert list(inspect.signature(api._get_mla_padded_q).parameters) == ["num_heads", "q", "target_heads"]


@pytest.mark.parametrize("target", [16, 32], ids=["16", "32"])
def test_a_target_that_is_no_multiple_of_the_heads_is_the_ops_tiling(op, target):
    """As vLLM tiles them: the query's heads, then its first ones again; vLLM reads the first n_heads back."""
    q = query()
    out = api._get_mla_padded_q(12, q, target_heads=target)
    assert len(op) == 1 and op[0][0] is q and op[0][1] == target
    assert out.dtype == FP8 and out.shape == (3, target, 32)
    assert torch.equal(out.view(torch.uint8), vllm_tiled(q, target).view(torch.uint8))
    assert torch.equal(out[:, :12].view(torch.uint8), q.view(torch.uint8))


def test_what_vllm_pads_another_way_stays_vllms(op):
    """No target (vLLM's own head count), no padding, a multiple of the heads (vLLM repeats each head), rows already
    as long as the target (vLLM takes a view), a query of other dims or strides."""
    q = query()
    assert api._get_mla_padded_q(12, q) is FallbackReason.OUT_OF_CONTRACT
    assert api._get_mla_padded_q(12, q, target_heads=12) is FallbackReason.OUT_OF_CONTRACT
    assert api._get_mla_padded_q(12, q, target_heads=24) is FallbackReason.OUT_OF_CONTRACT
    assert api._get_mla_padded_q(4, q, target_heads=16) is FallbackReason.OUT_OF_CONTRACT
    wide = torch.zeros(3, 16, 32, dtype=torch.uint8).view(FP8)[:, :12]
    assert api._get_mla_padded_q(12, wide, target_heads=16) is FallbackReason.OUT_OF_CONTRACT
    assert api._get_mla_padded_q(12, q[0], target_heads=16) is FallbackReason.OUT_OF_CONTRACT
    assert api._get_mla_padded_q(12, q[:, :, ::2], target_heads=16) is FallbackReason.OUT_OF_CONTRACT
    assert op == []


def test_the_reference_and_the_baseline_are_vllms_tiling():
    case = OpCase.of("mla_query_heads@1.0")
    call = case.call(case.inputs())
    q, m = call["q"], call["out_heads"]
    want = vllm_tiled(q, m).view(torch.uint8)
    assert torch.equal(case.contract.reference.reference(**call)["out"].view(torch.uint8), want)
    assert torch.equal(MlaQueryHeadsBaseline().baseline(**call)["out"].view(torch.uint8), want)


def test_the_committed_schema_is_its_canonical_text_at_its_path():
    """Kimi-K3's query (96 gathered heads of 576 bytes, to 128): the decode steps' token counts, five a sequence up
    to 96 sequences."""
    contract = ContractRegistry.default().get("mla_query_heads@1.0")
    (path,) = committed_schema_paths(op_dir(contract.op_package), "mla_query_heads_n_heads96_head_dim576")
    schema = ContractRegistry.default().parse_schema(path.read_bytes(), str(path))
    assert schema.schema_text() == path.read_text()
    assert path.relative_to(op_dir(contract.op_package)).as_posix() == schema.relative_path().as_posix()
    assert schema.kernel.language == "flydsl" and schema.kernel.entry_point == "kernel.py::run"
    assert [w.axes["n_tokens"] for w in schema.workloads] == list(range(5, 485, 5))
    assert contract.schema_errors(path, schema) == []
