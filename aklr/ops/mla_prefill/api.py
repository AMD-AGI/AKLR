# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.runtime.dispatch import api

from .contract import SPARSE_PREFILL_MXFP8


@api(SPARSE_PREFILL_MXFP8)
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
    *,
    softmax_scale=512**-0.5,
): ...
