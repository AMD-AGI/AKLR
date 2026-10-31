# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "bmm_bf16_nn_reference",
    "bmm_bf16_nn_shuffled_reference",
    "bmm_bf16_nt_reference",
    "bmm_bf16_nt_shuffled_reference",
    "gemm_a8w8_nt_reference",
    "gemm_a8w8_nt_shuffled_reference",
    "gemm_bf16_nt_reference",
    "gemm_bf16_nt_shuffled_reference",
    "gemm_fp8_nt_blockscaled_mn_reference",
    "gemm_fp8_nt_blockscaled_reference",
    "gemm_fp8_nt_blockscaled_shuffled_k_reference",
    "gemm_fp8_nt_blockscaled_shuffled_mn_reference",
    "gemm_mxfp4_nt_reference",
    "gemm_mxfp4_nt_shuffled_reference",
    "unshuffle_e8m0_scales",
    "unshuffle_tiles",
    "unshuffle_weight",
]


def unshuffle_tiles(stored: Tensor, split: tuple[int, ...], order: tuple[int, ...]) -> Tensor:
    """The inverse of splitting the last two dims of a tensor into ``split`` and
    permuting them by ``order``: the bytes of ``stored``, in its shape and dtype."""
    import torch

    x = stored.view(torch.uint8) if stored.element_size() == 1 else stored
    lead = tuple(x.shape[:-2])
    inverse = tuple(sorted(range(len(order)), key=order.__getitem__))
    n = len(lead)
    x = x.reshape(*lead, *(split[i] for i in order))
    x = x.permute(*range(n), *(n + i for i in inverse))
    return x.contiguous().view(stored.shape).view(stored.dtype)


def unshuffle_weight(stored: Tensor, tile: tuple[int, int]) -> Tensor:
    """aiter ``shuffle_weight(x, layout=tile)`` undone on the last two dims: tiles of
    ``tile[0]`` rows by ``2 * tile[1]`` columns, each in 16-byte vectors."""
    rows, cols = stored.shape[-2:]
    tile_rows, tile_cols = tile[0], tile[1] * 2
    vector = 16 // stored.element_size()  # elements per 16-byte load
    split = (
        rows // tile_rows,
        tile_rows,
        cols // tile_cols,
        tile_cols // vector,
        vector,
    )
    return unshuffle_tiles(stored, split, (0, 2, 3, 1, 4))


def unshuffle_e8m0_scales(stored: Tensor, rows: int, cols: int) -> Tensor:
    """aiter ``shuffle_scale`` undone on the last two dims, then the padding to
    multiples of (256, 8) dropped: the ``(..., rows, cols)`` scales."""
    padded_rows, padded_cols = stored.shape[-2:]
    split = (padded_rows // 32, 2, 16, padded_cols // 8, 2, 4)
    order = (0, 3, 5, 2, 4, 1)
    return unshuffle_tiles(stored, split, order)[..., :rows, :cols].contiguous()


def _plus_bias(product: Tensor, bias: Tensor | None) -> Tensor:
    return product if bias is None else product + bias.double()


def gemm_bf16_nt_reference(a, b, bias=None) -> Tensor:
    """``a @ b.T + bias``."""
    return _plus_bias(a.double() @ b.double().mT, bias)


def gemm_bf16_nt_shuffled_reference(a, b, bias=None) -> Tensor:
    """The row-major maths on b unshuffled (16 x 16)."""
    return gemm_bf16_nt_reference(a, unshuffle_weight(b, (16, 16)), bias)


def gemm_a8w8_nt_reference(a, b, a_scale, b_scale, bias=None) -> Tensor:
    """Per-row scales."""
    a = a.double() * a_scale.double().reshape(-1, 1)
    b = b.double() * b_scale.double().reshape(-1, 1)
    return _plus_bias(a @ b.mT, bias)


def gemm_a8w8_nt_shuffled_reference(a, b, a_scale, b_scale, bias=None) -> Tensor:
    """The row-major maths on b unshuffled (16 x 16)."""
    return gemm_a8w8_nt_reference(a, unshuffle_weight(b, (16, 16)), a_scale, b_scale, bias)


def gemm_fp8_nt_blockscaled_reference(a, b, a_scale, b_scale) -> Tensor:
    """A scale per ``(1, 128)`` of ``a`` and ``(128, 128)`` of ``b``; tails partial."""
    a, b = a.double(), b.double()
    (n, k), block = b.shape, 128
    a_scale = a_scale.double().repeat_interleave(block, dim=1)
    b_scale = b_scale.double().repeat_interleave(block, dim=0)
    b_scale = b_scale.repeat_interleave(block, dim=1)
    return (a * a_scale[:, :k]) @ (b * b_scale[:n, :k]).mT


def gemm_fp8_nt_blockscaled_mn_reference(a, b, a_scale, b_scale) -> Tensor:
    return gemm_fp8_nt_blockscaled_reference(a, b, a_scale.mT, b_scale)


def gemm_fp8_nt_blockscaled_shuffled_mn_reference(a, b, a_scale, b_scale) -> Tensor:
    """The K-major maths on b unshuffled (16 x 16) and an MN-major a_scale."""
    return gemm_fp8_nt_blockscaled_reference(a, unshuffle_weight(b, (16, 16)), a_scale.mT, b_scale)


def gemm_fp8_nt_blockscaled_shuffled_k_reference(a, b, a_scale, b_scale) -> Tensor:
    """The K-major maths on b unshuffled (16 x 16)."""
    return gemm_fp8_nt_blockscaled_reference(a, unshuffle_weight(b, (16, 16)), a_scale, b_scale)


def gemm_mxfp4_nt_reference(a, b, a_descale, b_descale, bias=None) -> Tensor:
    """``a @ b.T + bias``, one e8m0 scale per 32 fp4 values along ``k``; fp4 packs
    two values a byte, low nibble first."""
    import torch

    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    e2m1 = torch.tensor(magnitudes, dtype=torch.float64, device=a.device)
    e2m1 = torch.cat((e2m1, -e2m1))

    def dequantize(x, scale):
        codes = x.view(torch.uint8).long()
        x = e2m1[torch.stack((codes & 0xF, codes >> 4), dim=-1).flatten(-2)]
        codes = scale.view(torch.uint8)
        scale = torch.exp2(codes.double() - 127).masked_fill(codes == 255, math.nan)
        return x * scale.repeat_interleave(32, dim=-1)

    return _plus_bias(dequantize(a, a_descale) @ dequantize(b, b_descale).mT, bias)


def gemm_mxfp4_nt_shuffled_reference(a, b, a_descale, b_descale, bias=None) -> Tensor:
    """The row-major maths on b unshuffled (16 x 16) and both scales unshuffled and
    unpadded."""
    m, n, scale_columns = a.shape[0], b.shape[0], a.shape[1] * 2 // 32
    return gemm_mxfp4_nt_reference(
        a,
        unshuffle_weight(b, (16, 16)),
        unshuffle_e8m0_scales(a_descale, m, scale_columns),
        unshuffle_e8m0_scales(b_descale, n, scale_columns),
        bias,
    )


def bmm_bf16_nt_reference(A, B) -> Tensor:
    """``B`` is ``(batch_size, n, k)``."""
    return A.double() @ B.double().mT


def bmm_bf16_nn_reference(A, B) -> Tensor:
    """``B`` is ``(batch_size, k, n)``."""
    return A.double() @ B.double()


def bmm_bf16_nt_shuffled_reference(A, B) -> Tensor:
    """The nt maths on B unshuffled (16 x 16)."""
    return bmm_bf16_nt_reference(A, unshuffle_weight(B, (16, 16)))


def bmm_bf16_nn_shuffled_reference(A, B) -> Tensor:
    """The nn maths on B unshuffled (16 x 16)."""
    return bmm_bf16_nn_reference(A, unshuffle_weight(B, (16, 16)))
