# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ops aiter routes at the commit stack.json pins (a6c759b1), as far as AKLR declares them: each
op's version and the params of its native function, copied from aiter's source."""

from collections.abc import Callable

import torch
from torch import Tensor

# op name -> (op version, a function with the native one's params and defaults)
ROUTED_OPS: dict[str, tuple[int, Callable]] = {
    # aiter/ops/topk.py: topk_gating, biased_grouped_topk
    "topk_gating": (
        1,
        lambda topk_weights, topk_indices, gating_output, correction_bias=None, need_renorm=True, routed_scaling_factor=1.0, score_func="sqrtsoftplus", num_shared_experts=0, num_valid_rows=None: (
            None
        ),
    ),
    "biased_grouped_topk": (
        1,
        lambda gating_output, correction_bias, topk_weights, topk_ids, num_expert_group, topk_group, need_renorm, routed_scaling_factor=1.0, num_valid_rows=None: (
            None
        ),
    ),
    # aiter/ops/mhc.py: mhc_pre
    "mhc_pre": (
        1,
        lambda residual, fn, hc_scale, hc_base, rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=1.0, sinkhorn_repeat=20, norm_weight=None, norm_eps=1e-6, large_m_splitk=False, w_preshuffle_bf16=0, res_preshuffle=False: (
            Tensor
        ),
    ),
    # aiter/tuned_gemm.py: _gemm_a16w16
    "gemm_a16w16": (
        1,
        lambda A, B, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None, out=None: Tensor,
    ),
    # aiter/ops/triton/gemm/basic/gemm_a16w16.py: _gemm_a16w16_routed
    "gemm_a16w16_activation": (
        1,
        lambda x, w, bias=None, dtype=torch.bfloat16, y=None, config=None, activation=None, skip_reduce=False, kernel_type="bandwidth_bound", persistent=False: (
            Tensor
        ),
    ),
    # aiter/ops/gemm_op_a8w8.py: gemm_a8w8_bpreshuffle, gemm_a8w8_blockscale, gemm_a8w8_blockscale_bpreshuffle
    "gemm_a8w8_bpreshuffle": (
        1,
        lambda XQ, WQ, x_scale, w_scale, bias=None, dtype=torch.bfloat16, check=False: Tensor,
    ),
    "gemm_a8w8_blockscale": (
        1,
        lambda XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16, isBpreshuffled=False, split_k=None: Tensor,
    ),
    "gemm_a8w8_blockscale_bpreshuffle": (
        1,
        lambda XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16, out=None: Tensor,
    ),
    # aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py: _gemm_a8w8_blockscale_routed
    "gemm_a8w8_blockscale_out": (
        1,
        lambda x, w, x_scale, w_scale, dtype=torch.bfloat16, y=None, config=None, skip_reduce=False, kernel_type="bandwidth_bound": (
            Tensor
        ),
    ),
    # aiter/ops/rmsnorm.py: _rms_norm_fwd_dispatch, rmsnorm2d_fwd_with_add
    "rmsnorm": (2, lambda input, weight, epsilon, use_model_sensitive_rmsnorm: Tensor),
    "rmsnorm_with_add": (
        1,
        lambda out, input, residual_in, residual_out, weight, epsilon, gemma_norm=False, use_model_sensitive_rmsnorm=0: (
            None
        ),
    ),
}
