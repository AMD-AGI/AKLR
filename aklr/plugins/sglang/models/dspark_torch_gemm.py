# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSpark draft's lm_head GEMM, which sglang runs on torch (``srt/models/dspark.py``): each
step, the draft projects its last hidden states through the target's lm_head, ``hidden @ weight.T``
over the rank's slice of the vocabulary.

The draft's other torch GEMMs, the projection of the target's hidden states and the Markov head's
bias, stay on torch: there torch is as fast as AKLR's kernels."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []


@sglang_route("sglang.srt.models.dspark.project_through_lm_head", versions=("0.5.22",))
def _project_through_lm_head(hidden, lm_head):
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    if should_apply_lm_head_quant_method(lm_head, lm_head.quant_method):
        return FallbackReason.OUT_OF_CONTRACT
    weight = lm_head.weight
    return gemm(hidden.to(weight.dtype), weight)
