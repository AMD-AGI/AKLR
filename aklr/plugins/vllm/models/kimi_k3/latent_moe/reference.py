# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The latent-MoE tail as vLLM composes it, in float32: the all-reduced latent rounded to bf16
(the all-reduce's output), its RMSNorm rounded to bf16 (one rounding of latent * rstd * weight, as aiter's norm
kernel rounds), each rank's own columns of its shared output rounded to bf16 after the addmm (its epilogue adds the
shared partial in float32; the call's shared as it leaves it), then the all-reduce of the shared outputs, rounded to
bf16 once."""

import torch

from aklr.core.contract.reference import DistOpReference

__all__ = ["KimiK3LatentMoeTailReference"]


def _bf16(x):
    """``x`` rounded to bf16 (to nearest even), in its own dtype."""
    return x.to(torch.bfloat16).to(x.dtype)


class KimiK3LatentMoeTailReference(DistOpReference):
    def reference(self, calls, rank):
        """out and shared as ``rank``'s call leaves them; out is every rank's. The ranks' calls in rank order (each
        on its rank's device, read on ``rank``'s), sums in rank order."""
        call = calls[rank]
        device = call["routed"].device

        def read(c, name):
            return c[name].float().to(device)

        latent = read(calls[0], "routed")
        for c in calls[1:]:
            latent = latent + read(c, "routed")
        latent = _bf16(latent)
        rstd = (latent.square().mean(-1, keepdim=True) + call["eps"]).rsqrt()
        out = mine = None
        for r, c in enumerate(calls):
            shared = read(c, "shared").clone()
            up_weight = read(c, "up_weight")
            normed = _bf16(latent * rstd * read(c, "norm_weight"))
            own = slice(r * up_weight.shape[0], (r + 1) * up_weight.shape[0])
            shared[:, own] = _bf16(shared[:, own] + normed @ up_weight.T)
            mine = shared if r == rank else mine
            out = shared if out is None else out + shared
        dtype = call["shared"].dtype
        return {"out": out.to(dtype), "shared": mine.to(dtype)}
