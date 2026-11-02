###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.utils import dtype_str_to_torch_dtype

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["is_reference_comparable"]


def is_reference_comparable(
    output_dtypes: Mapping[str, str],
    reference_outputs: Mapping[str, Tensor],
    check: Callable[..., Any],
) -> bool:
    """Whether the reference rounded to the output dtypes stays finite where it was and passes ``check``."""
    rounded = {o: r.to(dtype_str_to_torch_dtype(output_dtypes[o])) for o, r in reference_outputs.items()}
    is_finite_where_reference_is = all(
        (rounded[o].double().isfinite() | ~r.isfinite()).all() for o, r in reference_outputs.items()
    )
    return is_finite_where_reference_is and bool(
        check(list(reference_outputs.values()), list(rounded.values()))
    )
