# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch: the packed pools decoded exactly, nothing rounded in
between."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from aklr.ops.mla.reference import flash_mla_with_kvcache_reference
from aklr.plugins.sglang.models.deepseekv41.contract import MXFP8_BLOCK, ROPE_DIM

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["deepseekv41_mxfp8_linear_reference", "deepseekv41_sparse_mla_reference"]


def deepseekv41_mxfp8_linear_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Rounding point: a bf16 x rounds to fp8 e4m3 at its scale per 1 x 32 before the product."""
    import torch

    args = row_major_args
    x = decode(args["x"], point.dtypes["x"])
    blocks = x.unflatten(-1, (-1, MXFP8_BLOCK))
    if point.attrs["has_x_scale"]:
        x_scale = decode(args["x_scale"], point.dtypes["x_scale"]).to(x.dtype)[..., None]
    else:
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        mantissa, exponent = torch.frexp(amax / 448.0)
        # ceil(log2): frexp's exponent, one less at a power of two
        x_scale = torch.exp2((exponent - (mantissa == 0.5).int()).clamp(-127, 127).to(x.dtype))
        blocks = (blocks / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype)
    weight = decode(args["weight"], point.dtypes["weight"]).to(x.dtype)
    weight_scale = decode(args["weight_scale"], point.dtypes["weight_scale"]).to(x.dtype)
    weight_scale = weight_scale.repeat_interleave(MXFP8_BLOCK, dim=0).repeat_interleave(MXFP8_BLOCK, dim=1)
    return (blocks * x_scale).flatten(-2) @ (weight * weight_scale).mT


def deepseekv41_sparse_mla_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """flash_mla_with_kvcache's sparse attention over the same pools, then the rotation."""
    import torch

    args = row_major_args
    # the sparse branch of flash_mla_with_kvcache reads these of its Point, and no lengths
    sparse = SimpleNamespace(attrs={"kv_cache_format": "dsv4_fp8"}, dtypes=point.dtypes)
    output, _ = flash_mla_with_kvcache_reference(
        sparse,
        {
            "q": args["q"],
            "kv_cache": args["kv_cache"],
            "sparse_indices": args["indices"],
            "extra_kv_cache": args.get("extra_kv_cache"),
            "extra_sparse_indices": args.get("extra_indices"),
            "sinks": args["sinks"],
            "sm_scale": args["sm_scale"],
            "kv_scale": 1.0,
        },
        decode,
    )
    if not point.attrs["has_rope_freqs"]:
        return output
    freqs = decode(args["rope_freqs"], point.dtypes["rope_freqs"]).to(output.dtype)
    freqs = freqs[args["positions"].long()].view(-1, 1, 1, ROPE_DIM // 2, 2)
    cos, sin = freqs.unbind(-1)
    rope = output[..., -ROPE_DIM:].unflatten(-1, (ROPE_DIM // 2, 2))
    a, b = rope.unbind(-1)
    rope.copy_(torch.stack((a * cos + b * sin, b * cos - a * sin), dim=-1))
    return output
