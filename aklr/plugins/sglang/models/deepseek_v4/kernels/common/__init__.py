# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang-AMD-Mono project
"""Shared device primitives of the mono kernels: the resident grid (``plan``), wave ops and buffer
access (``ops``), MX helpers (``mx``), epoch tags (``epoch``), the mailbox (``sync``) and wait deadlines
(``guard``). Adapted from ROCm/ATOM (MIT License); each file names its origin."""
