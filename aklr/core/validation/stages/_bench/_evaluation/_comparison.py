# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract import semantics

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "find_nondeterminism", "is_reference_comparable",
]  # fmt: skip


def is_reference_comparable(
    output_dtypes: Mapping[str, str],
    reference_outputs: Mapping[str, Tensor],
    compared: Callable[[Mapping[str, Tensor]], Mapping[str, semantics.CompareResult]],
) -> bool:
    rounded = {o: r.to(semantics.torch_dtypes(output_dtypes[o])[0]) for o, r in reference_outputs.items()}
    is_finite_where_reference_is = all(
        (rounded[o].double().isfinite() | ~r.isfinite()).all() for o, r in reference_outputs.items()
    )
    return is_finite_where_reference_is and all(r.is_passed for r in compared(rounded).values())


def find_nondeterminism(
    schema_json: Mapping[str, Any],
    first_outputs: Mapping[str, Tensor],
    second_outputs: Mapping[str, Tensor],
) -> str | None:
    """``None`` when two runs agree under the schema's determinism level, else the detail."""
    runs = {
        o: [first_outputs[o], second_outputs[o]] for o in semantics.mutable_tensor_declarations(schema_json)
    }
    determinism = schema_json["compare"]["determinism"]
    if all(semantics.runs_agree(determinism, r) for r in runs.values()):
        return None
    fraction = max(semantics.run_to_run_fraction(*r) for r in runs.values())
    return f"{fraction:.3g} of an output's elements differ from numeric/0's"
