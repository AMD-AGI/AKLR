###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the mla_prefill op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations

import torch


def pa_sparse_prefill_fp8_opus_baseline(
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
    **_,
):
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

    return lambda: pa_sparse_prefill_fp8_opus(
        q_nope.view(torch.float8_e4m3fn),
        q_rope,
        prefix_nope.view(torch.float8_e4m3fn),
        prefix_rope,
        prefix_indices,
        prefix_indptr,
        extend_nope.view(torch.float8_e4m3fn),
        extend_rope,
        extend_indices,
        extend_indptr,
        sinks,
        softmax_scale,
        out=output,
    )
