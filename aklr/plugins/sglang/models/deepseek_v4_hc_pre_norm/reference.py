# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["HcPreNormReference"]


class HcPreNormReference(OpReference):
    def reference(
        self,
        *,
        residual,
        pre_prev,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps,
        hc_eps,
        norm_eps,
        post_multiplier,
        sinkhorn_iters,
        **_,
    ):
        """The mixing statistics of the flattened copies give this boundary's pre / post / comb;
        the collapse under the previous pre, summed in copy order in float32 and rounded to bf16,
        is what the RMS norm takes. quantized and scales follow fp8_grid_quant's rule: the scale
        multiply rounds to float32, its exponent rounds upward and is clamped to [1, 254]."""
        copies = residual.float()
        n_tokens, hc, _ = copies.shape
        flat = copies.flatten(1)
        rsqrt = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
        logits = (flat.double() @ hc_fn.double().mT).float() * rsqrt
        scale, base = hc_scale.float(), hc_base.float()
        sizes = (hc, hc, hc * hc)
        pre_logits, post_logits, comb_logits = logits.split(sizes, dim=-1)
        pre_base, post_base, comb_base = base.split(sizes)
        pre = torch.sigmoid(pre_logits * scale[0] + pre_base) + hc_eps
        post = post_multiplier * torch.sigmoid(post_logits * scale[1] + post_base)
        comb = (comb_logits * scale[2] + comb_base).reshape(n_tokens, hc, hc)
        comb = comb.softmax(dim=-1) + hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
        for _ in range(int(sinkhorn_iters) - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

        y = torch.zeros_like(copies[:, 0])
        for k in range(hc):
            y = y + pre_prev[:, k : k + 1].float() * copies[:, k]
        y = y.to(torch.bfloat16)
        rows = y.float()
        rstd = torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + norm_eps)
        out_norm = (rows * rstd * norm_weight.float()).to(torch.bfloat16)

        grouped = out_norm.float().reshape(n_tokens, -1, 32)
        maximum = grouped.abs().amax(-1).clamp_min(1e-10)
        raw = (maximum * (1.0 / 448.0)).contiguous().view(torch.int32)
        exponent = (((raw >> 23) & 0xFF) + ((raw & 0x7FFFFF) != 0).to(torch.int32)).clamp(1, 254)
        scale_value = (exponent << 23).contiguous().view(torch.float32)
        quantized = (grouped / scale_value.unsqueeze(-1)).clamp(-448.0, 448.0)
        return {
            "y": y,
            "out_norm": out_norm,
            "pre": pre,
            "post": post,
            "comb": comb,
            "quantized": quantized.to(torch.float8_e4m3fn).reshape(n_tokens, -1),
            "scales": exponent.to(torch.uint8),
        }
