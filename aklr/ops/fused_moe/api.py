# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

from aklr.core.runtime.dispatch import api
from aklr.ops.fused_moe.contract import FUSED_MOE

if TYPE_CHECKING:
    import torch

__all__ = ["fused_moe"]


@api(FUSED_MOE)
def fused_moe(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weights,
    topk_ids,
    expert_mask: torch.Tensor | None = None,  # EP
    activation=None,
    quant_type=None,
    doweight_stage1=False,
    # following for quant
    w1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, inter_dim]
    bias1=None,
    bias2=None,
    # clamped SwiGLU, swiglu and SiTUv2 activation maths, not tuning knobs
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode: str | None = None,
    out_dtype=None,
    num_local_tokens: torch.Tensor | None = None,  # EP: [1] int32 rows received
    act_quant=4,  # bits per quantized activation element: SiTUv2 quantizes to fp8 (8)
    out: torch.Tensor | None = None,  # the caller's output buffer; None: dispatch allocates it
): ...
