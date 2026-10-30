# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.norms.contract import RMS_NORM, RMS_NORM_ADD

__all__ = ["rms_norm", "rms_norm_add"]


@api(RMS_NORM)
def rms_norm(x, weight, eps): ...


@api(RMS_NORM_ADD)
def rms_norm_add(x, residual, weight, eps): ...


# forward_aiter, not forward: sglang still picks the backend, and this is the one the
# baseline measures.
@sglang_route("sglang.srt.layers.layernorm.RMSNorm.forward_aiter", versions=">=0.5.22,<0.5.23")
def _rms_norm_forward_aiter(self, x, residual=None, post_residual_addition=None, quant_linear=None):
    """Batch-invariant, padded and post-residual-addition calls fall back to sglang; the
    contract checks shapes."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    # quant_linear is no reason: forward_aiter ignores it too.
    if (
        post_residual_addition is not None
        or self._fused_pad_kernel is not None
        or is_batch_invariant_mode_enabled()
    ):
        return FallbackReason.OUT_OF_CONTRACT
    if residual is None:
        return rms_norm(x=x, weight=self.weight, eps=self.variance_epsilon)
    return rms_norm_add(x=x, residual=residual, weight=self.weight, eps=self.variance_epsilon)
