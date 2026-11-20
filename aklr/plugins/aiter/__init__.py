# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR as a provider of aiter's kernel plugin interface (``aklr.plugins.aiter.provider``).

aiter routes its ops to the providers enabled for it; sglang and vLLM reach AKLR through it once they
enable that interface, with no route of their own on an aiter function. The ops it serves are AKLR's
(``aklr/ops``)."""

__all__: list[str] = []
