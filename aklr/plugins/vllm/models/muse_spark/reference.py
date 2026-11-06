# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch: the RMSNorm unrounded, rounded to hidden_states' dtype
as vLLM's fused_rmsnorm_expand writes it, then each route through fused_moe's reference
(its MXFP4 activations and bf16 intermediate) with a weight of one."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["muse_spark_moe_reference"]


def muse_spark_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each row normalized, then each of its routes through its expert, unweighted."""
    import torch

    from aklr.ops.fused_moe.reference import fused_moe_reference

    hidden = row_major_args["hidden_states"]
    x = decode(hidden, point.dtypes["hidden_states"])
    gain = decode(row_major_args["norm_weight"], point.dtypes["norm_weight"])
    gain = gain + row_major_args["gain_center"]
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + row_major_args["eps"])
    normalized = (x * rstd * gain).to(hidden.dtype)
    n_tokens, topk = row_major_args["topk_ids"].shape
    routes = normalized.repeat_interleave(topk, dim=0)
    ids = row_major_args["topk_ids"].reshape(n_tokens * topk, 1)
    route_point = SimpleNamespace(
        axes={"n_tokens": n_tokens * topk},
        dtypes={"hidden_states": point.dtypes["hidden_states"], "topk_weights": "float32"},
        attrs={
            "has_expert_mask": True,
            "has_num_local_tokens": False,
            "activation": "silu",
            "doweight_stage1": False,
            "gate_mode": "separated",
            "act_quant": 4,
        },
    )
    route_args = {
        "hidden_states": routes,
        "topk_ids": ids,
        "topk_weights": torch.ones(ids.shape, dtype=torch.float32, device=ids.device),
        **{k: row_major_args[k] for k in ("expert_mask", "w1", "w1_scale", "w2", "w2_scale")},
    }
    out = fused_moe_reference(route_point, route_args, decode)
    return out.view(n_tokens, topk, -1)
