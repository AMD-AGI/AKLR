# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The launches' entry functions, which ``spec.py`` calls."""

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.plugins.sglang.monokernels.glm53.contract import GLM53_MOE_LAYER

__all__ = ["glm53_moe_layer"]


@api(GLM53_MOE_LAYER)
def glm53_moe_layer(
    hidden_states,
    positions,
    cache_locations,
    topk_indices,
    kv_cache,  # written in place
    cos_sin_cache,
    input_norm_weight,
    qkv_a_weight,
    qkv_a_scale,
    q_a_norm_weight,
    kv_a_norm_weight,
    q_b_weight,
    q_b_scale,
    w_kc,
    w_vc,
    w_scale,
    o_weight,
    o_scale,
    post_norm_weight,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk=8,
    tp_size=1,
    routed_scaling_factor=2.5,
    eps=1e-5,
    sm_scale=0.0625,
    out_dtype=None,
): ...
