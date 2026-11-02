###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer) and
# FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import Field, NonNegativeInt

from aklr.core._frozen_model import Excluded, NonEmptyStr, frozen_dataclass


class DType(str, Enum):
    """Element types, each named as its torch dtype."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT8_E4M3FNUZ = "float8_e4m3fnuz"
    FLOAT8_E5M2FNUZ = "float8_e5m2fnuz"
    FLOAT8_E8M0FNU = "float8_e8m0fnu"
    FLOAT4_E2M1FN_X2 = "float4_e2m1fn_x2"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"


@frozen_dataclass
class Tensor:
    """An op's tensor argument: its symbolic shape, where the call passes it and how its dtype resolves."""

    dim_names: List[NonEmptyStr] = Field(alias="shape")
    dtype: Optional[DType] = None
    description: Optional[str] = None

    param: Excluded[Optional[NonEmptyStr]] = None
    tuple_idx: Excluded[Optional[NonNegativeInt]] = None
    dtype_from: Excluded[Optional[NonEmptyStr]] = None
    dtype_from_scalar: Excluded[Optional[NonEmptyStr]] = None
    optional: Excluded[bool] = False


@frozen_dataclass
class Scalar:
    dtype: str
    description: Optional[str] = None
    # always None: in the json a scalar is a tensor spec without a shape
    shape: None = Field(default=None, kw_only=True, repr=False)

    param: Excluded[Optional[NonEmptyStr]] = None
    optional: Excluded[bool] = False


@frozen_dataclass
class Var:
    """A runtime-variable axis, such as batch_size or seq_len."""

    description: Optional[str] = None
    type: Literal["var"] = Field(default="var", kw_only=True, repr=False)


@frozen_dataclass
class Const:
    """A compile-time-constant axis, such as hidden_size or num_heads."""

    description: Optional[str] = None
    abbrev: Excluded[Optional[str]] = None  # its prefix in schema names: "" omits the axis, None: its name
    value: Optional[NonNegativeInt] = None
    type: Literal["const"] = Field(default="const", kw_only=True, repr=False)

    extract: Excluded[Optional[Callable[[Dict[str, Any]], Optional[int]]]] = None
