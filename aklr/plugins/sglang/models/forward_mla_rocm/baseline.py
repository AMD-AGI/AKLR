# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.24.dev809 runs on gfx950 for rocm_absorb_v_bmm on bf16 weights at unit scale, w_vc
not contiguous (Kimi-K3.1-Pro's): torch.bmm into the (heads, tokens, v) view of a (tokens, heads, v)
buffer."""

import torch

from aklr.core.contract.baseline import OpBaseline

__all__ = ["MlaVAbsorbBaseline"]


class MlaVAbsorbBaseline(OpBaseline):
    def baseline(self, *, attn_output, w_vc, output=None):
        """torch.bmm as rocm_absorb_v_bmm calls it: attn_output's (heads, tokens, 512) view times the
        (heads, 512, v) view of w_vc sglang holds, into the transposed view of its _bmm_buf."""
        if output is None:
            output = attn_output.new_empty((attn_output.shape[0], w_vc.shape[0], w_vc.shape[1]))
        torch.bmm(
            attn_output.to(torch.bfloat16).transpose(0, 1), w_vc.transpose(1, 2), out=output.transpose(0, 1)
        )
        return {"output": output}
