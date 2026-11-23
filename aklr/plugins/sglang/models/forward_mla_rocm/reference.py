# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch, in float32: the bf16 inputs decoded exactly, each dot
product summed in float32."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["MlaVAbsorbReference"]


class MlaVAbsorbReference(OpReference):
    def reference(self, *, attn_output, w_vc, **_):
        """Per token and head, attn_output . w_vc^T; no rounding point but the output's."""
        output = torch.einsum("thk,hvk->thv", attn_output.float(), w_vc.float())
        return {"output": output.to(torch.bfloat16)}
