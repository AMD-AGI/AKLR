# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang-AMD-Mono project
"""DeepSeek-V4.1-Flash's shapes, routing and activation, which the kernels build in
(``sgl_amd_mono/deepseek_v41/__init__.py`` at b27d604)."""

HF_CONFIG = {
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "n_routed_experts": 384,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "norm_topk_prob": True,
    "topk_method": "noaux_tc",
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "rms_norm_eps": 1e-20,
    "hc_pre_from_prev_sublayer": True,
    "head_dim": 512,
    "q_lora_rank": 1280,
    "qk_rope_head_dim": 64,
    "sliding_window": 128,
    # the indexer K2i scores: 32 heads of 128 dims, and its top-512 rows
    "index_topk": 512,
    "index_n_heads": 32,
    "index_head_dim": 128,
}
# a candidate source's (``candidate_source_layer_id``) top blocks, which K2i scores
CANDIDATES = {"candidate_topk_blocks": 2048, "candidate_block_size": 8}
