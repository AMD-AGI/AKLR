# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aklr.plugins.aiter
from aklr.plugins.aiter.mla_prefill.ps_plan import ps_plan
from tests.op_sample import OpSample

_PREFILL = {"n_q": 40, "n_kv": 40, "n_offsets": 4, "n_heads": 2, "qk_head_dim": 64, "v_head_dim": 64}
_PREFIX = _PREFILL | {"n_kv": 300}
_SAMPLES = [
    OpSample(
        aklr.plugins.aiter.mla_decode_fwd,
        "mla_decode_fwd",
        {"batch": 3, "n_heads": 16, "n_slots": 300, "n_indices": 70, "n_offsets": 4},
    ),
    OpSample(aklr.plugins.aiter.mla_prefill_varlen, "mla_prefill_varlen", _PREFILL),
    OpSample(aklr.plugins.aiter.mla_prefill_varlen, "mla_prefill_varlen_prefix", _PREFIX),
    OpSample(aklr.plugins.aiter.mla_prefill_varlen_lse, "mla_prefill_varlen_lse", _PREFILL),
    OpSample(aklr.plugins.aiter.mla_prefill_varlen_lse, "mla_prefill_varlen_lse_no_causal", _PREFILL),
    OpSample(aklr.plugins.aiter.mla_prefill_varlen_lse, "mla_prefill_varlen_lse_prefix", _PREFIX | {"n_offsets": 2}),
    OpSample(aklr.plugins.aiter.mla_prefill_varlen_lse, "mla_prefill_varlen_lse_prefix_no_causal", _PREFIX),
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


def test_a_cprr_query_reads_its_shards_rows_at_or_before_its_global_position() -> None:
    axes = {"batch": 2, "query_length": 3, "n_heads": 16, "n_slots": 300, "n_indices": 40, "n_offsets": 3}
    sample = OpSample(aklr.plugins.aiter.mla_decode_fwd_cprr, "mla_decode_fwd_cprr", axes | {"cp_world_size": 4})
    call = sample.call()
    assert sample.api.resolve(call) is sample.template
    assert sample.schema().name == "mla_decode_fwd_cprr_query_length3_n_heads16_cp4"
    o, lse = sample.template.standard.reference(**call)
    assert sample.template.standard.check([o, lse], [o.to(torch.bfloat16), lse])
    one_query = aklr.plugins.aiter.mla_decode_fwd.variants[0].standard.reference
    world, rank = call["cp_world_size"], call["cp_rank"]
    read = []
    for b in range(2):
        start, end = call["kv_indptr"][b : b + 2].tolist()
        global_tokens = int(call["g_kv_indptr"][b + 1] - call["g_kv_indptr"][b])
        for i in range(3):
            # local row j holds global position j * world + rank
            rows = min(end - start, max(0, (global_tokens - 3 + i - rank) // world + 1))
            read.append(rows)
            expected_o, expected_lse = one_query(
                call["q"][b : b + 1, i],
                call["kv_buffer"],
                torch.tensor([0, rows], dtype=torch.int32),
                call["kv_indices"][start:],
                call["q_scale"],
                call["kv_scale"],
                call["sm_scale"],
            )
            torch.testing.assert_close(o[b, i], expected_o[0])
            torch.testing.assert_close(lse[b, i], expected_lse[0])
    # the drafts of a request read different prefixes of its shard, none of them the row past its global length
    assert read == [18, 19, 19] * 2


@pytest.mark.parametrize("causal", [True, False])
def test_the_prefill_plan_reduces_every_query_row_once(causal: bool) -> None:
    work_indptr, works, reduce_indptr, final, partials = ps_plan(
        [0, 300, 1000], [0, 900, 2000], [900, 1100], 16, num_cus=80, is_causal=causal
    )
    assert [row for start, end in final for row in range(start, end)] == list(range(1000))
    assert (work_indptr[-1], reduce_indptr[-1]) == (len(works), len(partials))


def test_the_verify_logits_are_each_query_rows_logits_over_its_requests_pages() -> None:
    axes = {"batch": 2, "next_n": 3, "n_heads": 32, "n_pages": 4, "page_tokens": 64, "max_blocks": 2}
    sample = OpSample(
        aklr.plugins.aiter.fp8_paged_mqa_logits_verify, "fp8_paged_mqa_logits_verify", axes | {"max_seq_len": 128}
    )
    call = sample.call()
    assert sample.api.resolve(call) is sample.template
    assert sample.schema().name == "fp8_paged_mqa_logits_verify_next_n3_n_heads32_page_tokens64"
    logits = sample.template.standard.reference(**call)
    assert logits.shape == (2, 3, 128)
    assert sample.template.standard.check([logits], [logits.float()])
    one_row = aklr.plugins.aiter.fp8_paged_mqa_logits.variants[0].standard.reference
    for b, j in [(0, 0), (1, 2)]:
        row = one_row(
            call["q"][b, j][None, None],
            call["kv_cache"],
            call["weights"][b, j][None],
            call["context_lens"][b, j][None],
            call["block_tables"][b][None],
            None,
            page_size=64,
        )
        torch.testing.assert_close(logits[b, j], row[0])
