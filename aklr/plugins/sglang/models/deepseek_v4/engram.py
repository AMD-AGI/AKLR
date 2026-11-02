# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""An Engram layer's write with its image-token rows kept, in one launch: sglang's
``engram_gate`` kernel with its ``torch.where`` made at the store (a text row gets the same
FP32 arithmetic and single cast, an image row its input unchanged)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gate_keeping_kernel(
    X,
    KV,
    QW,
    KW,
    IDS,
    O,
    image_token_id,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(tl.sum(key * key, 0) / D + EPS)
    dot = tl.sum((x * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    image = tl.load(IDS + token) == image_token_id
    tl.store(O + row * D + col, tl.where(image, x, x + gate * value), mask)


@triton.jit
def _gate_keeping_post_kernel(
    XO,
    R,
    POST,
    COMB,
    KV,
    QW,
    KW,
    IDS,
    O,
    image_token_id,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    # the previous layer's post (aiter's mhc_post as its compiler contracts it, prologue's
    # _post_combine_kernel) of this stream, rounded to bf16 as mhc_post stores it
    xo = tl.load(XO + token * D + col, mask, 0).to(tl.float32)
    r0 = tl.load(R + token * HC * D + col, mask, 0).to(tl.float32)
    v = tl.fma(xo, tl.load(POST + token * HC + hc), r0 * tl.load(COMB + token * HC * HC + hc))
    for j in tl.static_range(1, HC):
        r = tl.load(R + token * HC * D + j * D + col, mask, 0).to(tl.float32)
        v = tl.fma(r, tl.load(COMB + token * HC * HC + j * HC + hc), v)
    x = v.to(tl.bfloat16).to(tl.float32)
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(tl.sum(key * key, 0) / D + EPS)
    dot = tl.sum((x * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    image = tl.load(IDS + token) == image_token_id
    tl.store(O + row * D + col, tl.where(image, x, x + gate * value), mask)


def post_applies(x, residual, post, comb) -> bool:
    """Whether ``gate_keeping_post`` takes the previous layer's owed post: x [T, D] and
    residual [T, HC, D] bf16, post [T, HC] and comb [T, HC, HC] fp32, all contiguous."""
    return (
        residual.dim() == 3
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == torch.float32
        and tuple(x.shape) == (residual.shape[0], residual.shape[2])
        and tuple(post.shape) == residual.shape[:2]
        and tuple(comb.shape) == (*residual.shape[:2], residual.shape[1])
        and all(t.is_contiguous() for t in (x, residual, post, comb))
    )


def gate_keeping_post(
    x,
    residual,
    post,
    comb,
    kv,
    q_weight,
    k_weight,
    eps,
    clamp_value,
    input_ids,
    image_token_id,
):
    """``gate_keeping(mhc_post(x, residual, post, comb), kv, ...)`` in one launch: the
    previous layer's post computed where the gate loads its streams."""
    t, hc, d = residual.shape
    out = torch.empty_like(residual)
    if t:
        _gate_keeping_post_kernel[(t * hc,)](
            x,
            residual,
            post,
            comb,
            kv,
            q_weight,
            k_weight,
            input_ids,
            out,
            image_token_id,
            d,
            hc,
            eps,
            clamp_value,
            triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def applies(x, kv, q_weight, k_weight, input_ids) -> bool:
    """Where sglang's ``engram_gate`` takes its fused kernel (and the ids are one a row)."""
    return (
        x.ndim == 3
        and kv.ndim == 2
        and x.dtype == kv.dtype
        and x.dtype in (torch.bfloat16, torch.float32)
        and q_weight.dtype in (torch.bfloat16, torch.float32)
        and k_weight.dtype in (torch.bfloat16, torch.float32)
        and all(t.is_contiguous() for t in (x, kv, q_weight, k_weight, input_ids))
        and input_ids.shape == (x.shape[0],)
        and kv.shape == (x.shape[0], (x.shape[1] + 1) * x.shape[2])
    )


def gate_keeping(x, kv, q_weight, k_weight, eps, clamp_value, input_ids, image_token_id):
    """``engram_gate(x, kv, ...)`` with each row whose ``input_ids`` is
    ``image_token_id`` kept at ``x``."""
    t, hc, d = x.shape
    out = torch.empty_like(x)
    if t:
        _gate_keeping_kernel[(t * hc,)](
            x,
            kv,
            q_weight,
            k_weight,
            input_ids,
            out,
            image_token_id,
            d,
            hc,
            eps,
            clamp_value,
            triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
