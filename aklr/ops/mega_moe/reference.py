# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.ops.fused_moe.reference import _mxfp8, _weight

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["mega_moe_reference"]


def mega_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each token through its top-k experts of the whole group, weighted, summed.

    The experts are indexed by their global id: ``w1``, ``w2`` and their scales are
    the group's, every rank's experts in rank order (with ``ep_size`` 1, this rank's).
    Rounding points: the input of each linear in MXFP8, the gated activation in the
    input's dtype before it.
    """
    import torch.nn.functional as F

    hidden, ids = row_major_args["hidden_states"], row_major_args["topk_ids"].long()
    group_experts = point.axes["n_experts"] * point.attrs["ep_size"]
    if row_major_args["w1"].shape[0] != group_experts:
        raise ValueError(
            f"the reference needs the group's {group_experts} experts, got {row_major_args['w1'].shape[0]}"
        )
    limit = float(row_major_args["swiglu_limit"])
    x = _mxfp8(to_float64(hidden, point.dtypes["hidden_states"]))
    weights = to_float64(row_major_args["topk_weights"], point.dtypes["topk_weights"])
    out = x.new_zeros((*ids.shape, x.shape[-1]))
    for expert in ids.unique().tolist():
        routed = ids == expert
        tokens = routed.nonzero(as_tuple=True)[0]  # topk_ids rows are distinct
        projected = x[tokens] @ _weight(row_major_args, "w1", expert, to_float64).mT
        gate, up = projected.chunk(2, dim=-1)  # w1 stores [gate | up] along its rows
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        gated = (F.silu(gate) * up).to(hidden.dtype).double()
        w2 = _weight(row_major_args, "w2", expert, to_float64)
        out[routed] = _mxfp8(gated) @ w2.mT
    return (out * weights.unsqueeze(-1)).sum(dim=1)
