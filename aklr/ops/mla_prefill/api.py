# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.mla_prefill.schema import SPARSE_PREFILL_MXFP8

__all__ = ["sparse_prefill_mxfp8"]


@aklr_api(SPARSE_PREFILL_MXFP8)
def sparse_prefill_mxfp8(
    q_nope,
    q_rope,
    prefix_nope,
    prefix_rope,
    prefix_indices,
    prefix_indptr,
    extend_nope,
    extend_rope,
    extend_indices,
    extend_indptr,
    sinks,
    softmax_scale=512**-0.5,
    output=None,
): ...
