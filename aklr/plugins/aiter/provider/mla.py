# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility import for callers which announce a target-verify logits call."""

from aklr.plugins.aiter._verify import note_target_verify

__all__ = ["note_target_verify"]
