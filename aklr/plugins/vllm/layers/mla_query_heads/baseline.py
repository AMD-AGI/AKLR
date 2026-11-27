# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM 0.34.0 runs where the padded head count is no multiple of the query's and the rows are dense
(``AiterMLAHelper.get_mla_padded_q``, ``v1/attention/backends/mla/rocm_aiter_mla.py``): torch's ``repeat`` along
the head dim, then a contiguous copy of the first out_heads heads; both allocate."""

from aklr.core.contract.baseline import OpBaseline

__all__ = ["MlaQueryHeadsBaseline"]


class MlaQueryHeadsBaseline(OpBaseline):
    def baseline(self, *, q, out_heads=128, out_dtype=None, out=None):
        reps = -(-out_heads // q.shape[1])
        return {"out": q.repeat(1, reps, 1)[:, :out_heads, :].contiguous()}
