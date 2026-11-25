# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

import aklr.plugins.aiter
from tests.op_sample import OpSample


def test_the_paged_topk_init_draws_untied_scores_and_distinct_pages_the_reference_takes() -> None:
    axes = {"n_tokens": 3, "n_candidates": 64, "n_pages": 5, "k": 8, "page_size": 16}
    sample = OpSample(aklr.plugins.aiter.topk_transform_paged, "topk_transform_paged", axes)
    call = sample.call()
    assert sample.schema().name == "topk_transform_paged_k8_ps16"
    assert all(row.unique().numel() == 64 for row in call["scores"])
    assert all(sorted(row.tolist()) == list(range(5)) for row in call["page_tables"])
    assert int(call["seq_lens"].max()) <= 64
    out = sample.template.standard.reference(**call)
    assert list(out.shape) == [3, 8]
    assert sample.template.standard.check([out], [out.flip(-1)])


def test_the_biased_topk_reference_ranks_by_its_float32_key() -> None:
    (template,) = aklr.plugins.aiter.biased_grouped_topk.variants
    # 0.5 + (1.5 + 2**-23) rounds to 2.0 in float32: the two experts tie there, and the lower one wins
    bias = torch.tensor([1.5, 1.5 + 2**-23], dtype=torch.float32)
    _, ids = template.standard.reference(torch.zeros(1, 2), bias, 1, 1, 1, False)
    assert ids.tolist() == [[0]]
