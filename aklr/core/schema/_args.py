# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import Field, NonNegativeInt

from aklr.core._frozen_model import Excluded, NonEmptyStr, frozen_dataclass


class DType(str, Enum):
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
    """Specification of a tensor argument of an op: its symbolic shape, where the
    call passes it, and how its dtype is resolved."""

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
    """Specification of a scalar (non-tensor) argument of an op."""

    dtype: str
    description: Optional[str] = None
    # Always null: a scalar is a tensor spec without a shape in the json.
    shape: None = Field(default=None, kw_only=True, repr=False)

    param: Excluded[Optional[NonEmptyStr]] = None
    optional: Excluded[bool] = False


@frozen_dataclass
class Var:
    """Runtime-variable axis (e.g. ``batch_size``, ``seq_len``)."""

    description: Optional[str] = None
    type: Literal["var"] = Field(default="var", kw_only=True, repr=False)


@frozen_dataclass
class Const:
    """Compile-time-constant axis (e.g. ``hidden_size``, ``num_heads``)."""

    description: Optional[str] = None
    abbrev: Excluded[Optional[str]] = None
    value: Optional[NonNegativeInt] = None
    type: Literal["const"] = Field(default="const", kw_only=True, repr=False)

    extract: Excluded[Optional[Callable[[Dict[str, Any]], Optional[int]]]] = None
