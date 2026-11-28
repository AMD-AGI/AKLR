# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The launches' entry functions, which ``spec.py`` calls."""

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.plugins.sglang.monokernels.glm53.contract import (
    GLM53_DENSE_INDEX_LAYER,
    GLM53_EMBED,
    GLM53_HEAD,
    GLM53_MOE_INDEX_LAYER,
    GLM53_MOE_LAYER,
    GLM53_NEXTN_ENTRY,
)

__all__ = [
    "glm53_dense_index_layer",
    "glm53_embed",
    "glm53_head",
    "glm53_moe_index_layer",
    "glm53_moe_layer",
    "glm53_nextn_entry",
]


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


@api(GLM53_MOE_INDEX_LAYER)
def glm53_moe_index_layer(
    hidden_states,
    positions,
    cache_locations,
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
    wq_b_weight,
    wq_b_scale,
    wk_weight,
    wk_scale,
    weights_proj_weight,
    k_norm_weight,
    k_norm_bias,
    index_cos,
    index_sin,
    index_k_cache,  # written in place
    page_table,
    seq_lens,
    topk=8,
    tp_size=1,
    routed_scaling_factor=2.5,
    eps=1e-5,
    sm_scale=0.0625,
    index_eps=1e-6,
    layer_out=None,
    topk_indices=None,
): ...


@api(GLM53_DENSE_INDEX_LAYER)
def glm53_dense_index_layer(
    hidden_states,
    positions,
    cache_locations,
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
    wq_b_weight,
    wq_b_scale,
    wk_weight,
    wk_scale,
    weights_proj_weight,
    k_norm_weight,
    k_norm_bias,
    index_cos,
    index_sin,
    index_k_cache,  # written in place
    page_table,
    seq_lens,
    gate_up_weight,
    gate_up_scale,
    down_weight,
    down_scale,
    tp_size=1,
    eps=1e-5,
    sm_scale=0.0625,
    index_eps=1e-6,
    layer_out=None,
    topk_indices=None,
): ...


@api(GLM53_HEAD)
def glm53_head(
    hidden_states,
    lm_head_weight,
    tp_size=1,
    logits=None,
): ...


@api(GLM53_NEXTN_ENTRY)
def glm53_nextn_entry(
    input_ids,
    hidden_states,
    embed_weight,
    enorm_weight,
    hnorm_weight,
    eh_proj_weight,
    eps=1e-5,
    tp_size=1,
    out_dtype=None,
    out=None,
): ...


@api(GLM53_EMBED)
def glm53_embed(
    input_ids,
    embed_weight,
    tp_size=1,
    embeddings=None,
): ...
