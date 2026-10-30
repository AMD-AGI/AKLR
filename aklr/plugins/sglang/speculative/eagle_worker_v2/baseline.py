# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22's draft worker runs on HIP at topk 1: draft_topk1_postprocess, the split
argmax of EAGLEWorkerV2's draft steps (and, through draft_topk1_argmax_only, of its draft extend
after verify)."""

from __future__ import annotations

import functools

__all__ = ["eagle_worker_v2_draft_top1_baseline"]


@functools.cache
def _positions(n_rows, device):
    """The draft steps' positions, which draft_topk1_postprocess advances: one buffer per row
    count, as each of the draft's cuda graphs keeps one."""
    import torch

    return torch.zeros(n_rows, dtype=torch.int64, device=device)


def eagle_worker_v2_draft_top1_baseline(logits, topk_p=None, topk_index=None):
    """draft_topk1_postprocess as a draft step calls it on HIP (no draft_tokens), sglang's own
    body, not the plugin's hook on it: a triton argmax of each row's 8192-logit splits, then one
    per row that takes the highest split's, writes the index and p = 1 and advances the row's
    position. The draft extend after verify zeroes a scratch positions buffer for it first
    (draft_topk1_argmax_only): one fill more."""
    from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess

    postprocess = getattr(draft_topk1_postprocess, "__wrapped__", draft_topk1_postprocess)
    return postprocess(logits, _positions(logits.shape[0], logits.device))
