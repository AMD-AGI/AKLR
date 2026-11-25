# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.plugins.aiter.muse_spark_layer.contract import MUSE_SPARK_LAYER

__all__ = ["muse_spark_layer"]


@api(MUSE_SPARK_LAYER)
def muse_spark_layer(
    q,  # in: this layer's queries; out: the next layer's
    o_gate,  # in and out, as q
    residual,  # fp32, updated in place
    k_cache,
    v_cache,
    page_table,
    context_lens,
    o_proj,
    alpha,
    beta,
    pre_norm_weight,
    router_hi,
    router_lo,
    router_bias,
    pooler,
    pooled_norm_weight,
    w13,  # [256, 1024, 4096], fp4x2: this rank's gate and up rows of every expert
    w13_scale,
    w2,  # [256, 512, 4096], fp4x2: this rank's down rows of every expert
    w2_scale,
    expert_norm_weight,
    unpooler,
    post_ffn_weight,
    alpha2,
    beta2,
    next_norm_weight,
    next_qkvg,
    rotary,
    positions,
    slots,
    swa_map,
    next_k_cache,  # written in place at slots
    next_v_cache,
    scale=0.125,
    sliding_window=0,
    next_rope=True,
    next_sliding=False,
    tp_size=8,
    next_input=None,
    next_k=None,
    next_v=None,
): ...
