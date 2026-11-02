###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch


def default_tolerances(dtype: Union[torch.dtype, str]) -> Tuple[float, float]:
    """Return ``(rtol, atol)`` defaults for a trace correctness check.

    These defaults are intentionally conservative for low-precision inference
    data types. Template-specific ``check`` functions can override them by
    passing explicit ``rtol`` and ``atol`` to :func:`default_check`.
    """
    dtype_name = str(dtype).replace("torch.", "")
    if dtype_name in ("float64", "double"):
        return 1e-7, 1e-7
    if dtype_name in ("float32", "float"):
        return 1e-5, 1e-5
    if dtype_name in ("float16", "half"):
        return 1e-3, 1e-3
    if dtype_name == "bfloat16":
        return 1e-2, 1e-2
    if dtype_name.startswith("float8"):
        return 1e-1, 1e-1
    if dtype_name.startswith("float4") or "fp4" in dtype_name:
        return 1.0, 1.0
    return 0.0, 0.0


def _as_output_list(outputs: Any) -> List[Any]:
    if isinstance(outputs, list):
        return outputs
    if isinstance(outputs, tuple):
        return list(outputs)
    return [outputs]


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.reshape(-1).to(torch.float32)
    expected = expected.reshape(-1).to(torch.float32)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not finite.any():
        return 1.0
    actual = actual[finite]
    expected = expected[finite]
    actual_norm = torch.linalg.vector_norm(actual)
    expected_norm = torch.linalg.vector_norm(expected)
    if actual_norm.item() == 0 or expected_norm.item() == 0:
        return 1.0 if torch.equal(actual, expected) else 0.0
    return ((actual * expected).sum() / (actual_norm * expected_norm)).item()


def default_check(
    reference_outputs: Any,
    actual_outputs: Any,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    max_mismatch_pct: float = 0.0,
    min_cos_sim: Optional[float] = 1.0 - 1e-3,
) -> bool:
    """Validate actual outputs against reference outputs.

    ``reference_outputs`` and ``actual_outputs`` may be lists/tuples,
    dictionaries with matching keys, or single output values. Tensor outputs
    are checked for shape equality, closeness, mismatch percentage, and cosine
    similarity. Non-floating tensors are checked exactly unless
    ``max_mismatch_pct`` allows mismatches. Non-tensor outputs use Python
    equality.
    """
    if isinstance(reference_outputs, dict) or isinstance(actual_outputs, dict):
        if not isinstance(reference_outputs, dict) or not isinstance(actual_outputs, dict):
            return False
        if reference_outputs.keys() != actual_outputs.keys():
            return False
        refs = [reference_outputs[key] for key in reference_outputs]
        actuals = [actual_outputs[key] for key in reference_outputs]
    else:
        refs = _as_output_list(reference_outputs)
        actuals = _as_output_list(actual_outputs)

    if len(refs) != len(actuals):
        return False

    for ref, actual in zip(refs, actuals, strict=True):
        if isinstance(ref, torch.Tensor) != isinstance(actual, torch.Tensor):
            return False
        if not isinstance(ref, torch.Tensor):
            if ref != actual:
                return False
            continue

        if ref.shape != actual.shape:
            return False
        if ref.numel() == 0:
            continue

        ref_is_float = torch.is_floating_point(ref)
        actual_is_float = torch.is_floating_point(actual)
        if ref_is_float or actual_is_float:
            tolerance_dtype = actual.dtype if actual_is_float else ref.dtype
            default_rtol, default_atol = default_tolerances(tolerance_dtype)
            rtol_value = default_rtol if rtol is None else rtol
            atol_value = default_atol if atol is None else atol
            close = torch.isclose(
                actual.to(torch.float32),
                ref.to(torch.float32),
                rtol=rtol_value,
                atol=atol_value,
            )
            mismatch_pct = 100.0 * (~close).sum().item() / close.numel()
            if mismatch_pct > max_mismatch_pct:
                return False
            if min_cos_sim is not None:
                cos_sim = _cosine_similarity(actual, ref)
                if cos_sim < min_cos_sim:
                    return False
        else:
            matches = actual == ref
            mismatch_pct = 100.0 * (~matches).sum().item() / matches.numel()
            if mismatch_pct > max_mismatch_pct:
                return False

    return True


def standard_check(
    reference_outputs: Any,
    actual_outputs: Any,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    max_mismatch_pct: float = 0.0,
    min_cos_sim: Optional[float] = 1.0 - 1e-3,
) -> bool:
    """The check of a schema that sets none."""
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=rtol,
        atol=atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )
