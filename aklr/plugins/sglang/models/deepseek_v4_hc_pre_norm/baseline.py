# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.baseline import OpBaseline

__all__ = ["HcPreNormBaseline"]


class HcPreNormBaseline(OpBaseline):
    def baseline(
        self,
        *,
        residual,
        pre_prev,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps,
        hc_eps,
        norm_eps,
        post_multiplier,
        sinkhorn_iters,
        **_,
    ):
        """What sglang 0.5.23 runs on HIP below its prefill rows: ``_hc_boundary_partials``' Triton
        launch (one 64-column slice of every copy per program), then ``rmsnorm_with_sinkhorn``,
        whose launch norms the collapse and hosts the partials' reduce and the Sinkhorn."""
        import torch
        from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
            HcCoefficients,
            _hc_boundary_partials,
            rmsnorm_with_sinkhorn,
        )

        hc_mult, hidden = residual.shape[1], residual.shape[2]
        y = torch.empty_like(residual[:, 0])
        part_mix, part_sq = _hc_boundary_partials(
            None, residual, None, None, pre_prev, hc_fn, None, y, hc_mult=hc_mult, prefill=False
        )
        coefficients = HcCoefficients(
            part_mix,
            part_sq,
            hc_scale,
            hc_base,
            k=hc_mult * hidden,
            rms_eps=rms_eps,
            mix=hc_fn.shape[0],
            hc_mult=hc_mult,
            num_slices=part_sq.shape[0],
            sinkhorn_iters=int(sinkhorn_iters),
            hc_eps=hc_eps,
        )
        quant, out_norm = rmsnorm_with_sinkhorn(y, norm_weight, norm_eps, coefficients, emit_fp8=True)
        return {
            "y": y,
            "out_norm": out_norm,
            "pre": coefficients._pre,
            "post": coefficients._post,
            "comb": coefficients._comb,
            "quantized": quant.q,
            "scales": quant.scale,
        }
