# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aklr.ops
from aklr.ops.mla_prefill.ps_plan import ps_plan
from tests.op_sample import OpSample

_PREFILL = {"n_q": 40, "n_kv": 40, "n_offsets": 4, "n_heads": 2, "qk_head_dim": 64, "v_head_dim": 64}
_PREFIX = _PREFILL | {"n_kv": 300}
_SAMPLES = [
    OpSample(
        aklr.ops.mla_decode_fwd,
        "mla_decode_fwd",
        {"batch": 3, "n_heads": 16, "n_slots": 300, "n_indices": 70, "n_offsets": 4},
    ),
    OpSample(aklr.ops.mla_prefill_varlen, "mla_prefill_varlen", _PREFILL),
    OpSample(aklr.ops.mla_prefill_varlen, "mla_prefill_varlen_prefix", _PREFIX),
    OpSample(aklr.ops.mla_prefill_varlen_lse, "mla_prefill_varlen_lse", _PREFILL),
    OpSample(aklr.ops.mla_prefill_varlen_lse, "mla_prefill_varlen_lse_no_causal", _PREFILL),
    OpSample(aklr.ops.mla_prefill_varlen_lse, "mla_prefill_varlen_lse_prefix", _PREFIX | {"n_offsets": 2}),
    OpSample(aklr.ops.mla_prefill_varlen_lse, "mla_prefill_varlen_lse_prefix_no_causal", _PREFIX),
]


@pytest.mark.parametrize("sample", _SAMPLES, ids=lambda s: s.name)
def test_the_init_draws_a_call_of_its_variant_that_the_reference_takes(sample: OpSample) -> None:
    call = sample.call()
    # the api's bound arguments: a call without a prefix leaves cu_seqlens_k at its default
    assert sample.api.resolve({"cu_seqlens_k": None, **call}) is sample.template
    schema = sample.schema()
    assert schema.name.startswith(sample.name + "_n_heads")
    outputs = schema.standard.reference(**call)
    outputs = list(outputs) if isinstance(outputs, tuple) else [outputs]
    assert [list(o.shape) for o in outputs] == schema.get_output_shapes({**schema.const_axes, **sample.axes})
    assert all(o.isfinite().all() for o in outputs)
    assert schema.standard.check(outputs, [o.to(torch.bfloat16 if o.ndim == 3 else o.dtype) for o in outputs])


def test_the_prefill_reference_is_each_sequences_causal_softmax() -> None:
    sample = _SAMPLES[5]
    call = sample.call() | {"qkv_fp8": False}
    output, lse = sample.template.standard.reference(**call)
    q, k, v = (call[n].float() for n in ("q", "k", "v"))
    n_q, n_kv = q.shape[0], k.shape[0]
    logits = torch.einsum("qhd,khd->hqk", q, k) * call["sm_scale"]
    hidden = torch.arange(n_kv)[None] > torch.arange(n_q)[:, None] + (n_kv - n_q)
    logits = logits.masked_fill(hidden, -torch.inf)
    torch.testing.assert_close(
        output, torch.einsum("hqk,khd->qhd", logits.softmax(-1), v), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(lse, logits.logsumexp(-1).T, atol=1e-4, rtol=1e-4)


def test_a_decode_request_reading_no_row_is_zero_with_lse_minus_inf() -> None:
    sample = _SAMPLES[0]
    call = sample.call()
    call["kv_indptr"] = torch.tensor([0, 0, 30, 70], dtype=torch.int32)
    o, lse = sample.template.standard.reference(**call)
    assert o[0].eq(0).all()
    assert lse[0].eq(-torch.inf).all()
    assert lse[1:].isfinite().all()


@pytest.mark.parametrize("causal", [True, False])
def test_the_prefill_plan_reduces_every_query_row_once(causal: bool) -> None:
    work_indptr, works, reduce_indptr, final, partials = ps_plan(
        [0, 300, 1000], [0, 900, 2000], [900, 1100], 16, num_cus=80, is_causal=causal
    )
    assert [row for start, end in final for row in range(start, end)] == list(range(1000))
    assert (work_indptr[-1], reduce_indptr[-1]) == (len(works), len(partials))
