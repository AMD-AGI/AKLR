# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.baseline import OpBaseline

__all__ = [
    "HcPreNormBaseline",
    "HcPreNormMxfp8Baseline",
    "HcPreNormPostBaseline",
    "HcPreNormPostMxfp8Baseline",
]


def _sglang_pair(call, emit_fp8):
    """What sglang 0.5.23 runs on HIP below its prefill rows: ``_hc_boundary_partials``' Triton
    launch (one 64-column slice of every copy per program), then ``rmsnorm_with_sinkhorn``, whose
    launch norms the collapse and hosts the partials' reduce and the Sinkhorn."""
    import torch
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        HcCoefficients,
        _hc_boundary_partials,
        rmsnorm_with_sinkhorn,
    )

    residual, hc_fn, x = call["residual"], call["hc_fn"], call.get("x")
    hc_mult, hidden = residual.shape[1], residual.shape[2]
    y = torch.empty_like(residual[:, 0])
    residual_out = torch.empty_like(residual) if x is not None else None
    part_mix, part_sq = _hc_boundary_partials(
        x,
        residual,
        call.get("pending_post"),
        call.get("pending_comb"),
        call["pre_prev"],
        hc_fn,
        residual_out,
        y,
        hc_mult=hc_mult,
        prefill=False,
    )
    coefficients = HcCoefficients(
        part_mix,
        part_sq,
        call["hc_scale"],
        call["hc_base"],
        k=hc_mult * hidden,
        rms_eps=call["rms_eps"],
        mix=hc_fn.shape[0],
        hc_mult=hc_mult,
        num_slices=part_sq.shape[0],
        sinkhorn_iters=int(call["sinkhorn_iters"]),
        hc_eps=call["hc_eps"],
    )
    quant, out_norm = rmsnorm_with_sinkhorn(
        y,
        call["norm_weight"],
        call["norm_eps"],
        coefficients,
        emit_fp8=emit_fp8,
        fake_quant=emit_fp8,
    )
    out = {
        **({"residual_out": residual_out} if x is not None else {}),
        "y": y,
        "out_norm": out_norm,
        "pre": coefficients._pre,
        "post": coefficients._post,
        "comb": coefficients._comb,
    }
    if emit_fp8:
        out["quantized"], out["scales"] = quant.q, quant.scale
    return out


class HcPreNormBaseline(OpBaseline):
    def baseline(self, **call):
        """The pair with the plain norm: sglang's FFN boundary without the MXFP8 operand."""
        return _sglang_pair(call, emit_fp8=False)


class HcPreNormMxfp8Baseline(OpBaseline):
    def baseline(self, **call):
        """The pair whose norm launch also writes the next projection's MXFP8 operand."""
        return _sglang_pair(call, emit_fp8=True)


class HcPreNormPostBaseline(OpBaseline):
    def baseline(self, **call):
        """The pair with a pending post and the plain norm: the FFN boundary past the rows the fused
        all-reduce covers, so its post is still the boundary's to apply."""
        return _sglang_pair(call, emit_fp8=False)


class HcPreNormPostMxfp8Baseline(OpBaseline):
    def baseline(self, **call):
        """The same with the norm launch also writing the next projection's MXFP8 operand."""
        return _sglang_pair(call, emit_fp8=True)
