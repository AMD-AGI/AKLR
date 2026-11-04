###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

__all__ = ["LoadError"]


class LoadError(RuntimeError):
    """AKLR cannot load something it serves, e.g. a kernel that fails to build."""
