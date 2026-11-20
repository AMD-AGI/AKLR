# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The attention residual from its math, in float32; the roundings are p's and the outputs', to the prefix's
dtype."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["KimiK3AttnResFp8Reference", "KimiK3AttnResReference"]

_CHUNK_TOKENS = 2048  # float32 rows of 9 x 8192 per token: about 0.6 GB a chunk


class KimiK3AttnResReference(OpReference):
    def reference(
        self,
        *,
        prefix,
        bank,
        score_norm_weight,
        score_proj_weight,
        n_valid_rows,
        write_bank,
        score_eps,
        norm_eps,
        addend=None,
        norm_weight=None,
        **_,
    ):
        """out, prefix and bank as the call leaves them."""
        p = prefix.float()
        if addend is not None:
            p = (p + addend.float()).to(prefix.dtype).float()
        bank = bank.float()
        score_weight = score_norm_weight.float() * score_proj_weight.float()
        norm_weight = None if norm_weight is None else norm_weight.float()
        out = torch.empty_like(p)
        for start in range(0, p.shape[0], _CHUNK_TOKENS):
            tokens = slice(start, start + _CHUNK_TOKENS)
            if n_valid_rows == 0:
                # one source, whose softmax weight is 1 whatever its score (an infinite p scores NaN)
                mix = p[tokens]
            else:
                rows = torch.cat((bank[tokens, :n_valid_rows], p[tokens, None]), dim=1)
                scores = (rows @ score_weight) * (rows.square().mean(dim=-1) + score_eps).rsqrt()
                mix = (torch.softmax(scores, dim=-1)[..., None] * rows).sum(dim=1)
            if norm_weight is not None:
                mix = mix * (mix.square().mean(dim=-1, keepdim=True) + norm_eps).rsqrt() * norm_weight
            out[tokens] = mix
        if write_bank:
            bank[:, n_valid_rows] = p
        dtype = prefix.dtype
        return {"out": out.to(dtype), "prefix": p.to(dtype), "bank": bank.to(dtype)}


class KimiK3AttnResFp8Reference(KimiK3AttnResReference):
    def reference(self, **call):
        """quantized, scale, prefix and bank as the call leaves them: the attention residual's out, in the prefix's
        dtype, quantized per token as vLLM does it in float32 (the fp8 rounding is the operation itself, not an
        error)."""
        from aklr.plugins.vllm.models.kimi_k3.attn_res.contract import FP8_MAX

        expected = dict(super().reference(**call))
        rounded = expected.pop("out").float()
        floor, inverse_max = (
            torch.tensor(1e-10, dtype=torch.float32),
            torch.tensor(1.0 / FP8_MAX, dtype=torch.float32),
        )
        scale = torch.maximum(rounded.abs().amax(dim=-1, keepdim=True), floor) * inverse_max
        values = (rounded * (1.0 / scale)).clamp(-FP8_MAX, FP8_MAX)
        return {"quantized": values.to(torch.float8_e4m3fn), "scale": scale, **expected}
