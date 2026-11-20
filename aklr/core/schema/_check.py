###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
#
#
###############################################################################

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch


def default_tolerances(dtype: Union[torch.dtype, str]) -> Tuple[float, float]:
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
