# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The query's heads tiled to ``out_heads``, byte for byte: head h of the output is head h mod n_heads of the query."""

from aklr.core.contract.reference import OpReference

__all__ = ["MlaQueryHeadsReference"]


class MlaQueryHeadsReference(OpReference):
    def reference(self, *, q, out_heads=128, **_):
        import torch

        heads = torch.arange(out_heads, device=q.device) % q.shape[1]
        return {"out": q.view(torch.uint8)[:, heads].view(q.dtype)}
