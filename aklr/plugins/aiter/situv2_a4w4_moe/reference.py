# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch: fused_moe's reference at its SiTUv2 Point with MXFP4 activations (act_quant 4),
gate and up separated, no expert parallelism."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["situv2_a4w4_moe_reference"]


def situv2_a4w4_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each token through its top-k experts, weighted by its routing weights, summed."""
    from aklr.ops.fused_moe.reference import fused_moe_reference

    attrs = {
        "activation": "situv2",
        "act_quant": 4,
        "gate_mode": "separated",
        "doweight_stage1": False,
        "has_expert_mask": False,
        "has_num_local_tokens": False,
    }
    fused_moe_point = SimpleNamespace(axes=point.axes, dtypes=point.dtypes, attrs=attrs)
    return fused_moe_reference(fused_moe_point, row_major_args, decode)
