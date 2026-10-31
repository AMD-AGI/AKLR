# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Annotated, TypeVar

from pydantic import ConfigDict, Field, StringConstraints
from pydantic.dataclasses import dataclass
from typing_extensions import dataclass_transform

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
"""Type alias for non-empty strings with minimum length of 1."""

T = TypeVar("T")
Excluded = Annotated[T, Field(exclude=True)]
"""A field the code reads but the json dump leaves out."""


# Decorator of every frozen record: immutable, unknown fields rejected, enums kept as
# their plain strings (the form the exported json holds). Dumped by field alias, so a
# field can be named for the code and aliased for the json.
# A decorated function, not a partial: type checkers need dataclass_transform.
@dataclass_transform(frozen_default=True, field_specifiers=(Field,))
def frozen_dataclass(cls: type[T]) -> type[T]:
    return dataclass(  # pyright: ignore[reportReturnType]  # stub types it as PydanticDataclass
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
