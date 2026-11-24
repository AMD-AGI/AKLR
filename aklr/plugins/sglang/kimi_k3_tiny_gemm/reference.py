# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.reference import OpReference

__all__ = ["KimiK3TinyGemmReference"]


class KimiK3TinyGemmReference(OpReference):
    def reference(self, *, x, w, **_):
        """``x @ w.T``."""
        return {"out": (x.float() @ w.float().mT).to(x.dtype)}
