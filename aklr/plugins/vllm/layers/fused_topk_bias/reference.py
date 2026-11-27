# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference
from aklr.plugins.sglang.layers.moe_topk.reference import DeepseekV4RouterReference

__all__ = ["DeepseekV4TopkReference"]


class DeepseekV4TopkReference(OpReference):
    def reference(
        self,
        *,
        router_logits,
        correction_bias,
        topk_ids,
        renormalize,
        padding_mask=None,
        input_tokens=None,
        bias_vl=None,
        image_sentinel_lo=0,
        routed_scaling_factor=1.0,
        **_,
    ):
        """deepseek_v4_router's routed columns, without its shared one: the same scores, roundings and ranking
        (aiter's wave64 order of equal biased scores), image sentinel rows ranked with bias_vl; rows padding_mask
        marks take ids -1 and weights 0."""
        n_tokens, n_slots = topk_ids.shape

        def routed(bias):
            out = DeepseekV4RouterReference().reference(
                router_logits=router_logits,
                correction_bias=bias,
                topk_ids=torch.empty((n_tokens, n_slots + 1), dtype=topk_ids.dtype, device=topk_ids.device),
                n_shared=1,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
            )
            return out["topk_weights"][:, :n_slots].contiguous(), out["topk_ids"][:, :n_slots].contiguous()

        weights, ids = routed(correction_bias)
        if bias_vl is not None:
            image = (input_tokens >= image_sentinel_lo) & (input_tokens < image_sentinel_lo + 5)
            vl_weights, vl_ids = routed(bias_vl)
            weights[image] = vl_weights[image]
            ids[image] = vl_ids[image]
        if padding_mask is not None:
            ids[padding_mask] = -1
            weights[padding_mask] = 0
        return {"topk_weights": weights, "topk_ids": ids}
