# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What mla_qkv_a_norm's and mla_qkv_a_norm_fp8's contracts say, in torch, in float32: the inputs
decoded exactly, nothing rounded in between but what a block-fp8 projection quantizes (the
kernels' bf16 steps are theirs, left to the bounds)."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["FP8_GROUP", "MlaQkvANormFp8Reference", "MlaQkvANormReference"]

# The width of the groups a block-fp8 projection quantizes its activations in (aiter's per-1x128
# quantization): a group's fp8 codes at a scale of max(amax, 1e-10) / 448.
FP8_GROUP = 128
# fp8 e4m3fn's largest finite value
_FP8_MAX = 448.0


def _rms_normalized(x, weight, eps):
    """x times the reciprocal rms of its row, plus eps under the root, times weight."""
    mean_square = x.square().mean(dim=-1, keepdim=True)
    return x * (mean_square + eps).rsqrt() * weight


class MlaQkvANormReference(OpReference):
    def reference(self, *, hidden_states, w_qkv_a, q_norm_weight, kv_norm_weight, eps, **_):
        """Each token's hidden state times w_qkv_a, its q and KV parts RMS-normalized and scaled
        by their weights, its RoPE part as projected.
        Rounding points: the outputs' (sglang's bf16 qkv latent, before the norms, is left to the
        bounds)."""
        q_rank, kv_rank = q_norm_weight.shape[0], kv_norm_weight.shape[0]
        latent = hidden_states.float() @ w_qkv_a.float().T
        outputs = {
            "q_latent": _rms_normalized(latent[:, :q_rank], q_norm_weight.float(), eps),
            "kv_latent": _rms_normalized(latent[:, q_rank : q_rank + kv_rank], kv_norm_weight.float(), eps),
            "k_rope": latent[:, q_rank + kv_rank :],
        }
        return {n: t.to(torch.bfloat16) for n, t in outputs.items()}


def _group_codes(x):
    """x [rows, k]'s fp8 e4m3fn codes per 128-wide group of k and their scales [rows, groups]:
    max(amax, 1e-10) / 448 per group, the codes x * (1 / scale) in float32, as the kernels and
    aiter's per-1x128 quantization compute them."""
    groups = x.float().unflatten(-1, (-1, FP8_GROUP))
    scale = groups.abs().amax(-1, keepdim=True).clamp(min=1e-10) * (1.0 / _FP8_MAX)
    codes = (groups * (1.0 / scale)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return codes.flatten(-2), scale.squeeze(-1)


class MlaQkvANormFp8Reference(OpReference):
    def reference(self, *, hidden_states, w_qkv_a, w_qkv_a_scale, q_norm_weight, kv_norm_weight, eps, **_):
        """Each token's hidden state in fp8 codes per 128-wide group, times w_qkv_a and the two
        groups' scales; its q and KV parts RMS-normalized and scaled by their weights, its RoPE
        part as projected; the q part, in bf16, quantized per group again, the scales group-major.
        Rounding points: the hidden state's fp8 codes, q_fp8 from q_latent in bf16 (as sglang
        hands q_b_proj its bf16 latent), and the outputs' (the qkv latent, before the norms, is
        left to the bounds)."""
        q_rank, kv_rank = q_norm_weight.shape[0], kv_norm_weight.shape[0]
        codes, scales = _group_codes(hidden_states)
        weight = w_qkv_a.float()
        w_scale = w_qkv_a_scale.float().repeat_interleave(FP8_GROUP, 0)[: weight.shape[0]]
        latent = None
        for g in range(scales.shape[-1]):
            cols = slice(g * FP8_GROUP, (g + 1) * FP8_GROUP)
            part = (codes[:, cols].float() @ weight[:, cols].T) * scales[:, g : g + 1] * w_scale[:, g]
            latent = part if latent is None else latent + part
        q = _rms_normalized(latent[:, :q_rank], q_norm_weight.float(), eps).to(torch.bfloat16)
        q_codes, q_scales = _group_codes(q)
        kv = _rms_normalized(latent[:, q_rank : q_rank + kv_rank], kv_norm_weight.float(), eps)
        return {
            "q_latent": q,
            "q_fp8": q_codes,
            "q_scale": q_scales.T.contiguous(),
            "kv_latent": kv.to(torch.bfloat16),
            "k_rope": latent[:, q_rank + kv_rank :].to(torch.bfloat16),
        }
