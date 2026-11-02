###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = ["ConfigRegistry"]

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class ConfigRegistry(BaseModel, Generic[ConfigT]):
    default: ConfigT | None = None
    per_schema: dict[str, ConfigT] = Field(default_factory=dict)

    def get(self, schema_name: str) -> ConfigT | None:
        return self.per_schema.get(schema_name, self.default)

    def register(self, schema_name: str, config: ConfigT, *, override: bool = False) -> Self:
        if not override and schema_name in self.per_schema:
            raise ValueError(f"Schema '{schema_name}' already exists. Use override=True to replace it.")
        self.per_schema[schema_name] = config
        return self

    def register_many(self, configs: dict[str, ConfigT], *, override: bool = False) -> Self:
        if not override:
            conflicts = self.per_schema.keys() & configs.keys()
            if conflicts:
                raise ValueError(f"Cannot register. The following schemas already exist: {list(conflicts)}")
        self.per_schema.update(configs)
        return self
