# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What mla_qkv_a_norm's contract says, in torch, in float32: the inputs decoded exactly, nothing
rounded in between (the kernels' bf16 steps are theirs, left to the bounds)."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["MlaQkvANormReference"]


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
