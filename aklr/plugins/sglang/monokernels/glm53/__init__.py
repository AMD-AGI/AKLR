# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 (FP8) on gfx950: the spec in ``spec.py``, the layer ops in this package."""

# GLM-5.3's shapes, routing, rotary layout and indexer schedule, which the kernels build in
HF_CONFIG = dict(
    model_type="glm_moe_dsa",
    hidden_size=6144,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_shared_experts=1,
    moe_intermediate_size=2048,
    routed_scaling_factor=2.5,
    scoring_func="sigmoid",
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    hidden_act="silu",
    swiglu_limit=None,
    rms_norm_eps=1e-5,
    attention_bias=False,
    num_attention_heads=64,
    q_lora_rank=2048,
    kv_lora_rank=512,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    rope_interleave=True,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    indexer_rope_interleave=True,
    index_topk_freq=4,
    index_skip_topk_offset=3,
)
