# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR kernel contracts and host integration for fused_moe."""

__all__ = ["fused_moe", "op_kwargs"]


def __getattr__(name):
    if name == "fused_moe":
        from aiter.fused_moe import fused_moe

        return fused_moe
    if name == "op_kwargs":
        from ._arguments import op_kwargs

        return op_kwargs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
