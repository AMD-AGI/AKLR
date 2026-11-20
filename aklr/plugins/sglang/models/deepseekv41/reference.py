# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.reference import OpReference
from aklr.core.contract.semantics import to_float32
from aklr.plugins.aiter.provider.mla.reference import FlashMlaWithKvcacheReference

__all__ = ["Deepseekv41Mxfp8LinearReference", "Deepseekv41SparseMlaReference"]


class Deepseekv41Mxfp8LinearReference(OpReference):
    def reference(self, *, x, weight, weight_scale, x_scale=None, **_):
        import torch

        from aklr.plugins.sglang.models.deepseekv41.contract import MXFP8_BLOCK

        blocks = x.float().unflatten(-1, (-1, MXFP8_BLOCK))
        if x_scale is not None:
            scale = to_float32(x_scale, "float8_e8m0")[..., None]
        else:
            amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
            mantissa, exponent = torch.frexp(amax / 448.0)
            scale = torch.exp2((exponent - (mantissa == 0.5).int()).clamp(-127, 127).float())
            blocks = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
        scales = to_float32(weight_scale, "float8_e8m0")
        scales = scales.repeat_interleave(MXFP8_BLOCK, dim=0).repeat_interleave(MXFP8_BLOCK, dim=1)
        result = (blocks * scale).flatten(-2) @ (weight.float() * scales).mT
        return {"output": result.to(torch.bfloat16)}


class Deepseekv41SparseMlaReference(OpReference):
    def reference(
        self,
        *,
        q,
        kv_cache,
        indices,
        sinks,
        sm_scale,
        extra_kv_cache=None,
        extra_indices=None,
        rope_freqs=None,
        positions=None,
        **_,
    ):
        import torch

        from aklr.plugins.sglang.models.deepseekv41.contract import ROPE_DIM

        output = (
            FlashMlaWithKvcacheReference()
            .reference(
                q=q.float(),
                kv_cache=kv_cache,
                kv_cache_format="dsv4_fp8",
                causal=False,
                sparse_indices=indices,
                extra_kv_cache=extra_kv_cache,
                extra_sparse_indices=extra_indices,
                sinks=sinks,
                sm_scale=sm_scale,
                kv_scale=1.0,
            )["output"]
            .float()
        )
        if rope_freqs is not None:
            freqs = rope_freqs.float()[positions.long()].view(-1, 1, 1, ROPE_DIM // 2, 2)
            cos, sin = freqs.unbind(-1)
            rope = output[..., -ROPE_DIM:].unflatten(-1, (ROPE_DIM // 2, 2))
            a, b = rope.unbind(-1)
            rope.copy_(torch.stack((a * cos + b * sin, b * cos - a * sin), dim=-1))
        return {"output": output.to(torch.bfloat16)}
