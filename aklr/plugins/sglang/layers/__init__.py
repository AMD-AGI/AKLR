# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routes from sglang's layers to the ops in ``aklr/ops``, laid out like sglang's ``srt/layers``.
The subpackages here are plugin ops of their own."""

from aklr.plugins.sglang.layers import fused_moe

__all__ = ["fused_moe"]
