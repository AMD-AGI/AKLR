# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's GEMMs: the 16-bit ones (the tuned ``gemm_a16w16`` and the Triton one, routed as
``gemm_a16w16_activation``) served by AKLR's ``gemm``, the fp8 ones with a fp32 scale per
128 x 128 of the weight and per 1 x 128 of the activations (CK's, its shuffled-weight one and the
Triton one, routed as ``gemm_a8w8_blockscale_out``) served by AKLR's ``gemm_a8w8_blockwise_scaled``,
and the per-token one on a shuffled weight (``gemm_a8w8_bpreshuffle``) served by AKLR's
``gemm_a8w8``. aiter takes each layout of the weight as its own op, AKLR as an ``operand_format``
or a layout variant of one contract; each op keeps its own ``run``."""

from __future__ import annotations

import torch

from aklr.ops.gemm.api import gemm, gemm_a8w8, gemm_a8w8_blockwise_scaled
from aklr.plugins.aiter.provider._base import offer, selector, served


def _fp32_scales(x_scale, w_scale) -> bool:
    """aiter also takes e8m0 scales per 32 (group32); AKLR's contract has fp32 per 128 only."""
    return x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32


def _out_dtype(dtype, y):
    """The dtype aiter's Triton GEMMs return: ``y``'s when given (``dtype`` only sizes a new one)."""
    return dtype if y is None else y.dtype


@selector("gemm_a8w8_blockscale")
def _select_gemm_a8w8_blockscale(args, kwargs):
    xq, wq, x_scale, w_scale, dtype, is_b_preshuffled, split_k = args
    # a shuffled weight is gemm_a8w8_blockscale_bpreshuffle's contract (aiter has no fp32 kernel
    # for it behind this op on gfx950); split_k has no counterpart in AKLR's
    if is_b_preshuffled or split_k is not None or not _fp32_scales(x_scale, w_scale):
        return None
    return offer(
        gemm_a8w8_blockwise_scaled.serving(xq, wq, x_scale, w_scale, out_dtype=dtype),
        _run_gemm_a8w8_blockscale,
    )


def _run_gemm_a8w8_blockscale(xq, wq, x_scale, w_scale, dtype, is_b_preshuffled, split_k):
    return served(gemm_a8w8_blockwise_scaled(xq, wq, x_scale, w_scale, out_dtype=dtype))


@selector("gemm_a8w8_blockscale_bpreshuffle")
def _select_gemm_a8w8_blockscale_bpreshuffle(args, kwargs):
    xq, wq, x_scale, w_scale, dtype, out = args
    # aiter reads x_scale's bytes column-major; AKLR's is_row_major marker would read them row-major
    if not _fp32_scales(x_scale, w_scale) or getattr(x_scale, "is_row_major", False):
        return None
    # x_scale holds aiter's column-major bytes, AKLR's default for a shuffled weight
    return offer(
        gemm_a8w8_blockwise_scaled.serving(
            xq, wq, x_scale, w_scale, out_dtype=dtype, out=out, operand_format="none:preshuffled"
        ),
        _run_gemm_a8w8_blockscale_bpreshuffle,
    )


def _run_gemm_a8w8_blockscale_bpreshuffle(xq, wq, x_scale, w_scale, dtype, out):
    return served(
        gemm_a8w8_blockwise_scaled(
            xq, wq, x_scale, w_scale, out_dtype=dtype, out=out, operand_format="none:preshuffled"
        )
    )


@selector("gemm_a8w8_blockscale_out")
def _select_gemm_a8w8_blockscale_out(args, kwargs):
    x, w, x_scale, w_scale, dtype, y, config, skip_reduce, kernel_type = args
    # config and kernel_type pick aiter's schedule, not the result. aiter's kernel reads x_scale
    # through its strides, so a transposed view is valid there; AKLR's row-major operands are
    # contiguous (no ColumnMajor a_scale)
    if (
        skip_reduce
        or _out_dtype(dtype, y) is None
        or not _fp32_scales(x_scale, w_scale)
        or not x_scale.is_contiguous()
    ):
        return None
    return offer(
        gemm_a8w8_blockwise_scaled.serving(
            x, w, x_scale, w_scale, out_dtype=_out_dtype(dtype, y), out=y, operand_format="none:none"
        ),
        _run_gemm_a8w8_blockscale_out,
    )


def _run_gemm_a8w8_blockscale_out(x, w, x_scale, w_scale, dtype, y, config, skip_reduce, kernel_type):
    return served(
        gemm_a8w8_blockwise_scaled(
            x, w, x_scale, w_scale, out_dtype=_out_dtype(dtype, y), out=y, operand_format="none:none"
        )
    )


def _per_row(scale, rows: int) -> bool:
    """A fp32 scale per row of ``rows``: (rows, 1), or the (rows,) AKLR's contract takes too."""
    return scale.dtype == torch.float32 and tuple(scale.shape) in ((rows, 1), (rows,))


def _shuffled(w):
    """``w`` carrying the ``is_shuffled`` by which AKLR's ``gemm_a8w8`` tells a shuffled b. aiter's
    op reads its weight as shuffled whatever it carries, and vLLM's weight, re-wrapped as a
    Parameter, carries nothing: then a view of it carries the mark, and the caller's tensor is left
    as it was."""
    if getattr(w, "is_shuffled", False) is True:
        return w
    marked = w.view(w.shape)
    marked.is_shuffled = True
    return marked


@selector("gemm_a8w8_bpreshuffle")
def _select_gemm_a8w8_bpreshuffle(args, kwargs):
    xq, wq, x_scale, w_scale, bias, dtype, check = args
    # check is aiter's debug flag: a debugging caller keeps aiter's kernel. aiter raises on a bias,
    # reads a wider weight as K zero-padded and has rules of its own for int8 (bf16 out only, K from
    # 256); AKLR's scales are fp32 per token and per output column, no per-tensor one (but one
    # token's, which is per token)
    if (
        check
        or bias is not None
        or xq.dtype != torch.float8_e4m3fn
        or wq.dtype != xq.dtype
        or xq.dim() != 2
        or wq.dim() != 2
        or wq.shape[1] != xq.shape[1]
        or not _per_row(x_scale, xq.shape[0])
        or not _per_row(w_scale, wq.shape[0])
    ):
        return None
    return offer(
        gemm_a8w8.serving(xq, _shuffled(wq), x_scale, w_scale, out_dtype=dtype),
        _run_gemm_a8w8_bpreshuffle,
    )


def _run_gemm_a8w8_bpreshuffle(XQ, WQ, x_scale, w_scale, bias, dtype, check):
    return served(gemm_a8w8(XQ, _shuffled(WQ), x_scale, w_scale, out_dtype=dtype))


def _rows(a, b, out):
    """``(a, out)`` as the 2-D views aiter's tuned GEMM computes on (a's leading dims are the rows),
    or ``None`` where a view is not possible and aiter runs ``F.linear``."""
    try:
        rows = a.view(-1, a.shape[-1]) if a.dim() >= 3 else a
        return rows, None if out is None else out.view(-1, b.shape[0])
    except RuntimeError:
        return None


@selector("gemm_a16w16")
def _select_gemm_a16w16(args, kwargs):
    a, b, bias, otype, scale_a, scale_b, scale_c, out = args
    # scaled (fp8) GEMMs and a shuffled weight are outside AKLR's gemm
    if scale_a is not None or scale_b is not None or scale_c is not None:
        return None
    if getattr(b, "is_shuffled", False) is True:
        return None
    # an out of another dtype than otype is an error aiter raises
    if out is not None and out.dtype != (a.dtype if otype is None else otype):
        return None
    rows = _rows(a, b, out)
    if rows is None:
        return None
    a_rows, out_rows = rows
    return offer(gemm.serving(a_rows, b, bias=bias, out_dtype=otype, out=out_rows), _run_gemm_a16w16)


def _run_gemm_a16w16(A, B, bias, otype, scale_a, scale_b, scale_c, out):
    a_rows, out_rows = _rows(A, B, out)
    result = served(gemm(a_rows, B, bias=bias, out_dtype=otype, out=out_rows))
    return result.view(*A.shape[:-1], B.shape[0]) if A.dim() >= 3 else result


@selector("gemm_a16w16_activation")
def _select_gemm_a16w16_activation(args, kwargs):
    x, w, bias, dtype, y, config, activation, skip_reduce, kernel_type, persistent = args
    # config, kernel_type and persistent pick aiter's schedule, not the result; an activation and
    # the unreduced split-K partials are outside AKLR's gemm
    if activation is not None or skip_reduce or _out_dtype(dtype, y) is None:
        return None
    return offer(
        gemm.serving(x, w, bias=bias, out_dtype=_out_dtype(dtype, y), out=y),
        _run_gemm_a16w16_activation,
    )


def _run_gemm_a16w16_activation(
    x, w, bias, dtype, y, config, activation, skip_reduce, kernel_type, persistent
):
    return served(gemm(x, w, bias=bias, out_dtype=_out_dtype(dtype, y), out=y))
