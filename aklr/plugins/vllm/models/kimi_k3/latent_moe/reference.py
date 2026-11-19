# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The latent-MoE tail as vLLM composes it, in the dtype ``decode`` gives: the all-reduced latent rounded to bf16
(the all-reduce's output), its RMSNorm rounded to bf16 (one rounding of latent * rstd * weight, as aiter's norm
kernel rounds), each rank's own columns of its shared output rounded to bf16 after the addmm (its epilogue adds the
shared partial in float32), then the all-reduce of the shared outputs, left unrounded."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_latent_moe_tail_reference"]


def _bf16(x: Tensor) -> Tensor:
    """``x`` rounded to bf16 (to nearest even), in its own dtype."""
    import torch

    return x.to(torch.bfloat16).to(x.dtype)


def kimi_k3_latent_moe_tail_reference(
    point: Any,
    group_args: Sequence[Mapping[str, Any]],
    rank: int,
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """``out`` of ``rank``'s call, which is every rank's: the ranks' args in rank order (each on its rank's device,
    read on ``rank``'s), sums in rank order."""
    dtypes = point.dtypes
    eps = group_args[rank]["eps"]
    device = group_args[rank]["routed"].device

    def read(args, name):
        return decode(args[name], dtypes[name]).to(device)

    latent = read(group_args[0], "routed")
    for args in group_args[1:]:
        latent = latent + read(args, "routed")
    latent = _bf16(latent)
    rstd = (latent.square().mean(-1, keepdim=True) + eps).rsqrt()
    out = None
    for r, args in enumerate(group_args):
        shared = read(args, "shared").clone()
        up_weight = read(args, "up_weight")
        normed = _bf16(latent * rstd * read(args, "norm_weight"))
        own = slice(r * up_weight.shape[0], (r + 1) * up_weight.shape[0])
        shared[:, own] = _bf16(shared[:, own] + normed @ up_weight.T)
        out = shared if out is None else out + shared
    return out
