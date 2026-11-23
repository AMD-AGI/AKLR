# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek MLA's query and value absorbs around the attention (kv_b_proj in fp8)."""

from aklr.ops.mla_absorb import api

__all__ = ["api"]
