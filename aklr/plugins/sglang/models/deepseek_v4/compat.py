# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the DeepSeek-V4.1 glue reads from sglang's monokernel runtime and its few kernels."""

from __future__ import annotations

from sglang.srt.monokernels.spec import (
    BuildContext,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Plan,
    Requirements,
    ShapeKey,
    StepMetadata,
)

# The sglang names read here (``SGLANG_READS`` as in spec.py).
SGLANG_READS = {
    "srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "BuildContext",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Plan",
        "Requirements",
        "ShapeKey",
        "StepMetadata",
        "tokens_per_request",
        "unread_metadata",
    ),
    "srt/monokernels/runtime.py": ("_backend", "_monokernel_unread_metadata"),
    "kernels/ops/layernorm/mhc.py": ("mhc_combine",),
    "srt/layers/attention/dsv4/unified_kv_gate.py": ("is_unified_kv_triton",),
}

__all__ = [
    "BuildContext",
    "DeviceLimits",
    "Feature",
    "MonoSpec",
    "Numerics",
    "NumericsKind",
    "Plan",
    "Requirements",
    "ShapeKey",
    "StepMetadata",
    "backend",
    "is_unified_kv_triton",
    "mhc_combine",
    "served_keys",
    "shape_key",
]


def backend() -> str | None:
    """The monokernel backend this server runs (``aklr`` when AKLR's specs run)."""
    from sglang.srt.monokernels.runtime import _backend

    return _backend()


def shape_key(forward_batch) -> ShapeKey | None:
    """The graph key of a DECODE or TARGET_VERIFY step, else None."""
    mode = forward_batch.forward_mode
    if mode.is_decode():
        name = "decode"
    elif mode.is_target_verify():
        name = "target_verify"
    else:
        return None
    bs, rows = int(forward_batch.batch_size), int(forward_batch.input_ids.shape[0])
    if bs <= 0 or rows % bs:
        return None
    return ShapeKey(forward_mode=name, batch_size=bs, tokens_per_request=rows // bs)


def served_keys(model) -> frozenset:
    """The graph keys the model's monokernels run (none before they are built)."""
    served, _ = model.__dict__.get("_monokernel_unread_metadata", ((), ()))
    return frozenset(served)


def mhc_combine(x_flat, pre, hc: int, out_dtype):
    from sglang.kernels.ops.layernorm.mhc import mhc_combine as combine

    return combine(x_flat, pre, hc, out_dtype)


def is_unified_kv_triton() -> bool:
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton as gate

    return gate()
