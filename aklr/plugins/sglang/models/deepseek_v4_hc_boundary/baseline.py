# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.baseline import OpBaseline

__all__ = ["HcBoundaryBaseline"]


class HcBoundaryBaseline(OpBaseline):
    def baseline(self, *, residual, pre_prev, hc_fn, x=None, post=None, comb=None, **_):
        """What sglang 0.5.23 runs on HIP below its prefill rows: _hc_boundary_partials' Triton kernel, one
        64-column slice per program (the wide 32-row tile where it applies)."""
        import torch
        from sglang.kernels.ops.layernorm.mhc_boundary_hip import _hc_boundary_partials

        residual_out = torch.empty_like(residual) if x is not None else None
        y = torch.empty_like(residual[:, 0])
        part_mix, part_sq = _hc_boundary_partials(
            x,
            residual,
            post,
            comb,
            pre_prev,
            hc_fn,
            residual_out,
            y,
            hc_mult=residual.shape[1],
            prefill=False,
        )
        out = {"y": y, "part_mix": part_mix, "part_sq": part_sq}
        if x is not None:
            out["residual_out"] = residual_out
        return out
