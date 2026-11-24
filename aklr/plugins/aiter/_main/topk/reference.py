# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["AnyOrderInFullRowsReference", "BiasedGroupedTopkReference", "TopkTransformPagedReference"]


class BiasedGroupedTopkReference(OpReference):
    def reference(self, *, router_logits, correction_bias, routed_scaling_factor, renormalize, topk_ids, **_):
        """topk_weights and topk_ids: each token's topk experts by biased score, ties to the lower expert.

        Every group is kept (the contract's constraint), so the groups mask nothing.
        """
        scores = router_logits.float().sigmoid()
        # torch.topk orders equal values as its algorithm happens to: a stable sort fixes them
        ranked = (scores + correction_bias.float()).sort(dim=-1, descending=True, stable=True).indices
        chosen = ranked[..., : topk_ids.shape[-1]]
        weights = scores.gather(-1, chosen)
        if renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return {"topk_weights": weights * routed_scaling_factor, "topk_ids": chosen.to(topk_ids.dtype)}


class AnyOrderInFullRowsReference(OpReference):
    def check(self, outputs, expected, schema):
        """``OpReference.check``, but a row of ``out_page_indices`` the reference fills (no -1) may
        list its slots in any order: ties and a full row's order are the kernel's choice."""
        out, reference = outputs["out_page_indices"], expected["out_page_indices"]
        is_full = (reference >= 0).all(dim=-1, keepdim=True)
        out = torch.where(is_full, out.sort(dim=-1).values, out)
        reference = torch.where(is_full, reference.sort(dim=-1).values, reference)
        return super().check(
            {**outputs, "out_page_indices": out}, {**expected, "out_page_indices": reference}, schema
        )


class TopkTransformPagedReference(AnyOrderInFullRowsReference):
    def reference(self, *, scores, seq_lens, page_size, out_page_indices, page_tables=None, **_):
        """Each row's selected candidates, mapped through its page table when given."""
        topk = out_page_indices.shape[-1]
        width = min(topk, scores.shape[-1])
        lengths = seq_lens.long().unsqueeze(-1)
        positions = torch.arange(scores.shape[-1], device=scores.device)
        # float32 compares exactly: the scores rank as stored.
        ranked = scores.masked_fill(positions >= lengths, -torch.inf).argsort(
            dim=-1, descending=True, stable=True
        )[:, :width]
        selected = torch.where(lengths <= topk, positions[:width], ranked)
        is_selected = positions[:width] < lengths
        if page_tables is not None:
            rows = is_selected.nonzero(as_tuple=True)[0]
            chosen = selected[is_selected]
            pages = page_tables[rows, chosen // page_size].long()
            selected = selected.masked_scatter(is_selected, pages * page_size + chosen % page_size)
        out = torch.nn.functional.pad(torch.where(is_selected, selected, -1), (0, topk - width), value=-1)
        return {"out_page_indices": out.to(out_page_indices.dtype)}
