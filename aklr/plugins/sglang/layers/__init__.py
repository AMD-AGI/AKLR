# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routes from sglang's layers to the ops of ``aklr/ops`` (the plugin's own ops are its
subpackages); mirrors sglang's ``srt/layers``."""

from aklr.plugins.sglang.layers import fused_moe

__all__ = ["fused_moe"]
