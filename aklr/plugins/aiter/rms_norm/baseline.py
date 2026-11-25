###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the rms_norm op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations


def rmsnorm2d_fwd_with_add_baseline(x, residual, weight, epsilon, normalized, residual_out, **_):
    from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_add

    return lambda: rmsnorm2d_fwd_with_add(
        out=normalized,
        input=x,
        residual_in=residual,
        residual_out=residual_out,
        weight=weight,
        epsilon=epsilon,
    )


def rmsnorm2d_fwd_baseline(x, weight, epsilon, **_):
    from aiter.ops.rmsnorm import rmsnorm2d_fwd

    return lambda: rmsnorm2d_fwd(x, weight, epsilon)
