# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

__all__ = ["LoadError"]


class LoadError(RuntimeError):
    """Aklr cannot start serving: a bad ``AKLR_*`` value, an unknown device, a bad schema."""
