# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.speculative.eagle_worker_v2.contract import (
    EAGLE_WORKER_V2_DRAFT_TOP1,
)

__all__ = ["eagle_worker_v2_draft_top1"]


@api(EAGLE_WORKER_V2_DRAFT_TOP1)
def eagle_worker_v2_draft_top1(logits): ...


# At topk 1 on HIP the draft worker takes argmaxes with p = 1: the draft steps' tokens from
# draft_topk1_postprocess, which also advances the rows' positions; the post-verify draft
# extend's from draft_topk1_argmax_only, which calls draft_topk1_postprocess where it is defined
# (sglang patches that binding too); the prefill's draft extend's as
# fast_topk(renorm_draft_probs(logits), topk), where without rejection sampling only the
# logits' order counts. These routes take the argmax op in all three.
@sglang_route(
    "sglang.srt.speculative.eagle_worker_v2.renorm_draft_probs",
    versions=">=0.5.22,<0.5.23",
)
def _renorm_draft_probs(next_token_logits, sampling_info, use_rejection_sampling):
    """The logits themselves, not their softmax: the same highest one for fast_topk."""
    from sglang.srt.runtime_context import get_spec

    if use_rejection_sampling or get_spec().speculative_eagle_topk != 1:
        return FallbackReason.OUT_OF_CONTRACT
    return next_token_logits


@sglang_route("sglang.srt.speculative.eagle_worker_v2.fast_topk", versions=">=0.5.22,<0.5.23")
def _fast_topk(values, topk, dim):
    """topk 1 over the last axis of rows."""
    if topk != 1 or values.dim() != 2 or dim not in (-1, 1):
        return FallbackReason.OUT_OF_CONTRACT
    return eagle_worker_v2_draft_top1(logits=values)


@sglang_route(
    "sglang.srt.speculative.eagle_worker_v2.draft_topk1_postprocess",
    versions=">=0.5.22,<0.5.23",
)
def _draft_topk1_postprocess(next_token_logits, positions, draft_tokens=None, draft_token_column=0):
    """Each row's argmax with p = 1 and the rows' positions advanced by one, as sglang's
    kernel leaves them; on HIP sglang passes no draft_tokens for the argmaxes."""
    if draft_tokens is not None:
        return FallbackReason.OUT_OF_CONTRACT
    out = eagle_worker_v2_draft_top1(logits=next_token_logits)
    if not isinstance(out, FallbackReason):
        positions.add_(1)
    return out
