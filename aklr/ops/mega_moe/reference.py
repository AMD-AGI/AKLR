# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch.nn.functional as F

from aklr.core.contract.reference import DistOpReference
from aklr.ops.fused_moe.reference import mxfp4_weight, mxfp8

__all__ = ["MegaMoeReference"]


class MegaMoeReference(DistOpReference):
    def reference(self, calls, rank):
        """Each token of ``rank`` through its top-k experts of the whole group, weighted, summed.

        Global expert ``e`` is local expert ``e % n_experts`` of rank ``e // n_experts``.
        Rounding points: the input of each linear in MXFP8, the gated activation in the input's
        dtype before it.
        """
        call = calls[rank]
        hidden_states, n_experts = call["hidden_states"], call["w1"].shape[0]
        ids, limit = call["topk_ids"].long(), float(call["swiglu_limit"])
        x = mxfp8(hidden_states.float())
        out = x.new_zeros((*ids.shape, x.shape[-1]))
        for expert in ids.unique().tolist():
            owner, local = divmod(expert, n_experts)
            owner_call = calls[owner]
            routed = ids == expert
            tokens = routed.nonzero(as_tuple=True)[0]  # topk_ids rows are distinct
            w1 = mxfp4_weight(owner_call["w1"][local], owner_call["w1_scale"][local]).to(x.device)
            gate, up = (x[tokens] @ w1.mT).chunk(2, dim=-1)  # w1: [gate | up]
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
            gated = (F.silu(gate) * up).to(hidden_states.dtype).float()
            w2 = mxfp4_weight(owner_call["w2"][local], owner_call["w2_scale"][local]).to(x.device)
            out[routed] = mxfp8(gated) @ w2.mT
        out = (out * call["topk_weights"].float().unsqueeze(-1)).sum(dim=1)
        return {"out": out.to(hidden_states.dtype)}
