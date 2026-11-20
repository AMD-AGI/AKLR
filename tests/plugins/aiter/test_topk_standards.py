# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

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
