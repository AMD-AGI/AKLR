# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4_hc_pre_norm.contract import (
    DEEPSEEK_V4_HC_PRE_NORM,
    DEEPSEEK_V4_HC_PRE_NORM_MXFP8,
)

__all__ = ["deepseek_v4_hc_pre_norm", "deepseek_v4_hc_pre_norm_mxfp8"]


@api(DEEPSEEK_V4_HC_PRE_NORM)
def deepseek_v4_hc_pre_norm(
    residual,
    pre_prev,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    y=None,
    out_norm=None,
    pre=None,
    post=None,
    comb=None,
    *,
    rms_eps,
    hc_eps,
    norm_eps,
    post_multiplier=2.0,
    sinkhorn_iters=20,
): ...


@api(DEEPSEEK_V4_HC_PRE_NORM_MXFP8)
def deepseek_v4_hc_pre_norm_mxfp8(
    residual,
    pre_prev,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    y=None,
    out_norm=None,
    pre=None,
    post=None,
    comb=None,
    quantized=None,
    scales=None,
    *,
    rms_eps,
    hc_eps,
    norm_eps,
    post_multiplier=2.0,
    sinkhorn_iters=20,
): ...


def _deferred_class():
    """A HcCoefficients that holds the boundary's inputs instead of its partials, so that the
    norm launch that follows can run both in one kernel; any other reader materializes it the
    way sglang would, which also fills the collapse."""
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        HcCoefficients,
        _hc_boundary_partials,
        hc_mix_reduce_sinkhorn_vec,
    )

    class _Deferred(HcCoefficients):
        pending: object = None

        def materialize(self) -> None:
            if self.materialized:
                return
            held, self.pending = self.pending, None
            if held is not None:
                part_mix, part_sq = _hc_boundary_partials(
                    None,
                    held["residual"],
                    None,
                    None,
                    held["pre_prev"],
                    held["hc_fn"],
                    None,
                    held["y"],
                    hc_mult=self.hc_mult,
                )
                self.part_mix, self.part_sq = part_mix, part_sq
            self.materialized = True
            hc_mix_reduce_sinkhorn_vec(
                self.part_mix,
                self.part_sq,
                self.hc_scale,
                self.hc_base,
                self._pre,
                self._post,
                self._comb,
                k=self.k,
                rms_eps=self.rms_eps,
                mix=self.mix,
                hc_mult=self.hc_mult,
                num_slices=self.num_slices,
                sinkhorn_iters=self.sinkhorn_iters,
                hc_eps=self.hc_eps,
            )

    return _Deferred


_DEFERRED = None


def _deferred():
    global _DEFERRED
    if _DEFERRED is None:
        _DEFERRED = _deferred_class()
    return _DEFERRED


def _qualifies(x, residual, pre_prev, hc_fn, hc_mult) -> bool:
    import torch
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        _HC_BOUNDARY_BLOCK_K,
        _hc_boundary_use_prefill,
    )

    m, _, h = residual.shape
    return (
        x is None
        and pre_prev is not None
        and hc_mult == 4
        and m > 0
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and h % _HC_BOUNDARY_BLOCK_K == 0
        and not _hc_boundary_use_prefill(m, False, True, None)
    )


@sglang_route(
    "sglang.kernels.ops.layernorm.mhc_boundary_hip.hc_boundary_fused_deferred",
    versions=("0.5.23",),
)
def _hc_boundary_fused_deferred(
    x,
    residual,
    post_in,
    comb_in,
    pre_prev,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult,
    sinkhorn_iters,
    rms_eps,
    hc_eps,
):
    """A boundary with no pending post, at decode rows: the collapse, the mixing statistics and
    the Sinkhorn all wait for the norm launch that reads the collapse, which runs them in one
    kernel (``_rmsnorm_with_sinkhorn``). Any other reader of the coefficients -- a norm sglang
    takes down another path, or the fused all-reduce asking for post / comb -- materializes them
    through sglang's own two launches. Rows with a pending post stay on sglang."""
    import torch

    if not _qualifies(x, residual, pre_prev, hc_fn, hc_mult):
        return FallbackReason.OUT_OF_CONTRACT
    m, _, h = residual.shape
    slices = h // 64
    device = residual.device
    coefficients = _deferred()(
        torch.empty((slices, m, hc_fn.shape[0]), dtype=torch.float32, device=device),
        torch.empty((slices, m), dtype=torch.float32, device=device),
        hc_scale,
        hc_base,
        k=hc_mult * h,
        rms_eps=rms_eps,
        mix=hc_fn.shape[0],
        hc_mult=hc_mult,
        num_slices=slices,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
    )
    y = torch.empty((m, h), dtype=residual.dtype, device=device)
    coefficients.pending = {
        "residual": residual,
        "pre_prev": pre_prev.contiguous(),
        "hc_fn": hc_fn,
        "y": y,
    }
    return None, y, coefficients


@sglang_route(
    "sglang.kernels.ops.layernorm.mhc_boundary_hip.rmsnorm_with_sinkhorn",
    versions=("0.5.23",),
)
def _rmsnorm_with_sinkhorn(x, weight, eps, coefficients, *, emit_fp8=False, fake_quant=True):
    """The pending boundary of ``x`` and this norm in one launch: the MXFP8 operand the next
    projection takes (the attention input norm, and the FFN norm under SGLANG_HIP_FFN_NORM_MXFP8)
    or the plain norm. sglang's bf16 fake quant, another tensor's norm and shapes the op has no
    kernel for materialize the coefficients as sglang would and leave the launch to sglang."""
    held = getattr(coefficients, "pending", None)
    mxfp8 = fake_quant and emit_fp8
    if held is None or x is not held["y"] or x.stride(1) != 1 or (fake_quant and not emit_fp8):
        if held is not None:
            coefficients.materialize()
        return FallbackReason.OUT_OF_CONTRACT
    call = (
        held["residual"],
        held["pre_prev"],
        held["hc_fn"],
        coefficients.hc_scale,
        coefficients.hc_base,
        weight.contiguous(),
        held["y"],
        None,
        coefficients._pre,
        coefficients._post,
        coefficients._comb,
    )
    scalars = {
        "rms_eps": coefficients.rms_eps,
        "hc_eps": coefficients.hc_eps,
        "norm_eps": eps,
        "sinkhorn_iters": coefficients.sinkhorn_iters,
    }
    op = deepseek_v4_hc_pre_norm_mxfp8 if mxfp8 else deepseek_v4_hc_pre_norm
    out = op(*call, **scalars)
    if isinstance(out, FallbackReason):
        coefficients.materialize()
        return out
    coefficients.pending = None
    coefficients.materialized = True
    if not mxfp8:
        return None, out[1]
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation

    return Mxfp8Activation(out[5], out[6]), out[1]
