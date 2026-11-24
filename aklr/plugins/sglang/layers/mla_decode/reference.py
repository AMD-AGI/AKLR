# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What kimi_k3_mla_decode's contract says, in torch, in fp32."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["KimiK3MlaDecodeReference"]


class KimiK3MlaDecodeReference(OpReference):
    def reference(self, *, q, kv_cache, kv_indptr, kv_indices, sm_scale, **_):
        """Each row's softmax over its slots of q . row * sm_scale, applied to the rows' first
        512 values; no rounding point but the output's."""
        dtype, q = q.dtype, q.float()
        cache = kv_cache.float()[:, 0]
        indptr, indices = kv_indptr.long().tolist(), kv_indices.long()
        output = q.new_zeros((q.shape[0], q.shape[1], 512))
        for r in range(q.shape[0]):
            rows = cache[indices[indptr[r] : indptr[r + 1]]]
            scores = (q[r] @ rows.T) * float(sm_scale)
            output[r] = torch.softmax(scores, dim=-1) @ rows[:, :512]
        return {"output": output.to(dtype)}
