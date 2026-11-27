# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["HcBoundaryReference"]


class HcBoundaryReference(OpReference):
    def reference(self, *, residual, pre_prev, hc_fn, part_mix, x=None, post=None, comb=None, **_):
        """The post's multiplies and adds are float32 in the stated order (separate torch ops round each); the
        collapse sums in copy order in float32; the partials are exact per slice, rounded once to float32."""
        out = {}
        copies = residual.float()
        if x is not None:
            xf = x.float()
            rows = []
            for k in range(copies.shape[1]):
                v = xf * post[:, k : k + 1]
                for j in range(copies.shape[1]):
                    v = v + copies[:, j] * comb[:, j, k : k + 1]
                rows.append(v.to(torch.bfloat16))
            out["residual_out"] = torch.stack(rows, 1)
            copies = out["residual_out"].float()
        y = torch.zeros_like(copies[:, 0])
        for k in range(copies.shape[1]):
            y = y + pre_prev[:, k : k + 1].float() * copies[:, k]
        out["y"] = y.to(torch.bfloat16)
        n_tokens, hc, hidden = copies.shape
        slices = part_mix.shape[0]
        width = hidden // slices
        c = copies.double().reshape(n_tokens, hc, slices, width)
        w = hc_fn.double().reshape(hc_fn.shape[0], hc, slices, width)
        out["part_mix"] = torch.einsum("mksw,nksw->smn", c, w).float()
        out["part_sq"] = c.square().sum((1, 3)).T.contiguous().float()
        return out
