# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AITER kernel provider. Call host operators through AITER or aklr.plugins.aiter.api."""

from aklr.plugins import Plugin

plugin = Plugin("aiter")

__all__ = ["plugin"]
