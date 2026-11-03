###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import Annotated, TypeVar

from pydantic import ConfigDict, Field, StringConstraints
from pydantic.dataclasses import dataclass
from typing_extensions import dataclass_transform

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

T = TypeVar("T")
Excluded = Annotated[T, Field(exclude=True)]


@dataclass_transform(frozen_default=True, field_specifiers=(Field,))
def frozen_dataclass(cls: type[T]) -> type[T]:
    """A frozen pydantic dataclass: unknown fields rejected, enums kept as plain strings, dumped by alias."""
    # the ignore: pydantic's stub types dataclass()'s result as PydanticDataclass, not type[T]
    return dataclass(  # pyright: ignore[reportReturnType]
        cls,
        frozen=True,
        config=ConfigDict(
            extra="forbid",
            use_enum_values=True,
            validate_by_name=True,
            validate_by_alias=True,
            serialize_by_alias=True,
        ),
    )
