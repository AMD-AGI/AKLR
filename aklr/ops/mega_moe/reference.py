# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from aklr.ops.fused_moe.reference import _mxfp8, _weight

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["mega_moe_reference"]


def mega_moe_reference(
    point: Any,
    group_args: Sequence[Mapping[str, Any]],
    rank: int,
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each token of ``rank`` through its top-k experts of the whole group, weighted, summed.

    Global expert ``e`` is local expert ``e % n_experts`` of rank ``e // n_experts``.
    Rounding points: the input of each linear in MXFP8, the gated activation in the
    input's dtype before it.
    """
    import torch.nn.functional as F

    args, n_experts = group_args[rank], point.axes["n_experts"]
    hidden, ids = args["hidden_states"], args["topk_ids"].long()
    limit = float(args["swiglu_limit"])
    x = _mxfp8(decode(hidden, point.dtypes["hidden_states"]))
    weights = decode(args["topk_weights"], point.dtypes["topk_weights"])
    out = x.new_zeros((*ids.shape, x.shape[-1]))
    for expert in ids.unique().tolist():
        owner, local = divmod(expert, n_experts)
        routed = ids == expert
        tokens = routed.nonzero(as_tuple=True)[0]  # topk_ids rows are distinct
        w1 = _weight(group_args[owner], "w1", local, decode).to(x.device)
        projected = x[tokens] @ w1.mT
        gate, up = projected.chunk(2, dim=-1)  # w1 stores [gate | up] along its rows
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        gated = (F.silu(gate) * up).to(hidden.dtype).to(x.dtype)
        w2 = _weight(group_args[owner], "w2", local, decode).to(x.device)
        out[routed] = _mxfp8(gated) @ w2.mT
    return (out * weights.unsqueeze(-1)).sum(dim=1)
