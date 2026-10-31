# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSA decode reference against the contract's text, one head at a time in float64:
the reference computes in fp32, so they agree to fp32's rounding."""

import math

import pytest
import torch

from aklr.core.contract import semantics
from aklr.plugins.sglang.layers.dsa_backend import reference
from aklr.plugins.sglang.layers.dsa_backend.contract import ROPE_DIM
from tests.op_case import OpCase

CONTRACT_ID = "dsa_sparse_mla_decode@1.0"


@pytest.fixture
def case():
    return OpCase.of(CONTRACT_ID)


def attend(case, args):
    return case.reference_outputs(args)["output"]


def by_head(args, latent):
    """Each head's softmax over its token's named rows, a slot at a time."""
    q = args["q"].double()
    pool = args["kv_cache"].double()
    tokens, heads, _ = q.shape
    output = torch.zeros((tokens, heads, latent), dtype=torch.float64)
    for t in range(tokens):
        slots = [int(s) for s in args["topk_indices"][t] if s >= 0]
        for h in range(heads):
            scores = [float(q[t, h] @ pool[s]) * args["sm_scale"] for s in slots]
            if not scores:
                continue
            peak = max(scores)
            weights = [math.exp(s - peak) for s in scores]
            total = sum(weights)
            for w, s in zip(weights, slots, strict=True):
                output[t, h] += w / total * pool[s, :latent]
    return output


def test_each_head_attends_the_rows_its_slots_name(case):
    args = case.inputs()
    args["topk_indices"][1, ::3] = -1  # every third entry of a token reads no row
    latent = args["q"].shape[-1] - ROPE_DIM
    expected = by_head(args, latent)
    assert torch.allclose(attend(case, args), expected, rtol=1e-5, atol=1e-6)


def test_a_token_that_reads_no_row_gets_zeros(case):
    args = case.inputs()
    args["topk_indices"][2] = -1
    output = attend(case, args)
    assert bool(output[2].eq(0).all()) and bool(output[[0, 1, 3]].ne(0).any())


def test_the_order_of_the_slots_does_not_matter(case):
    args = case.inputs()
    expected = attend(case, args)
    args["topk_indices"] = args["topk_indices"].flip(-1)
    assert torch.allclose(attend(case, args), expected, rtol=1e-5, atol=1e-6)


def test_sm_scale_multiplies_every_score(case):
    """A zero scale weighs a token's rows alike: their latent parts' mean."""
    args = case.inputs()
    args["sm_scale"] = 0.0
    args["topk_indices"][0, 1:] = -1
    args["topk_indices"][0, 0] = 5
    latent = args["q"].shape[-1] - ROPE_DIM
    output = attend(case, args)
    assert torch.equal(output[0], args["kv_cache"][5, :latent].double().expand(4, -1))
    named = args["topk_indices"][1]
    named = named[named >= 0]
    mean = args["kv_cache"].double()[named.long(), :latent].mean(0)
    assert torch.allclose(output[1], mean.expand(4, -1), rtol=1e-5, atol=1e-6)


def test_the_reference_leaves_its_inputs_and_chunks_alike(case, monkeypatch):
    """Inputs untouched; one token a chunk gives the same output bit for bit."""
    args = case.inputs()
    before = {n: v.clone() for n, v in args.items() if torch.is_tensor(v)}
    whole = attend(case, args)
    for name, tensor in before.items():
        assert torch.equal(args[name].view(torch.uint8), tensor.view(torch.uint8))
    monkeypatch.setattr(reference, "_CHUNK_TOKENS", 1)
    assert torch.equal(attend(case, args), whole)


def test_the_default_inputs_spread_the_softmax(case):
    """fp8 N(0, 1) queries and rows at GLM-5's scale: no head is a one-hot argmax, so
    the compare sees the weighting."""
    args = case.inputs()
    q = semantics.to_float64(args["q"], "float8_e4m3fn")
    rows = semantics.to_float64(args["kv_cache"], "float8_e4m3fn")
    slots = args["topk_indices"].long()
    scores = torch.einsum("thc,tkc->thk", q, rows[slots.clamp(min=0)])
    weights = (scores * args["sm_scale"]).softmax(-1)
    assert float(weights.max()) < 0.5
