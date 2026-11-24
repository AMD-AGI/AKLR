# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.baseline import OpBaseline

__all__ = ["KimiK3TinyGemmBaseline"]


class KimiK3TinyGemmBaseline(OpBaseline):
    def baseline(self, *, x, w, **_):
        """torch's linear, which sglang runs for this gemm."""
        return {"out": torch.nn.functional.linear(x, w)}
