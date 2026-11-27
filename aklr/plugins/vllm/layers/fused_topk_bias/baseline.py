# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.baseline import OpBaseline

__all__ = ["DeepseekV4TopkBaseline"]


class DeepseekV4TopkBaseline(OpBaseline):
    def baseline(
        self,
        *,
        router_logits,
        correction_bias,
        topk_weights,
        topk_ids,
        renormalize,
        padding_mask=None,
        input_tokens=None,
        bias_vl=None,
        image_sentinel_lo=0,
        routed_scaling_factor=1.0,
    ):
        """What vLLM 0.34's fused_topk_bias runs for this router with aiter's fused MoE on: its
        topk_hash_softplus_sqrt without a hash table, with the token_expert_indices scratch it allocates,
        into the caller's outputs."""
        import torch
        from vllm import _custom_ops as ops

        ops.topk_hash_softplus_sqrt(
            topk_weights,
            topk_ids,
            torch.empty_like(topk_ids),
            router_logits,
            renormalize,
            routed_scaling_factor,
            correction_bias,
            input_tokens,
            None,
            is_padding=padding_mask,
            bias_vl=bias_vl,
            image_sentinel_lo=image_sentinel_lo,
        )
        return {"topk_weights": topk_weights, "topk_ids": topk_ids}
