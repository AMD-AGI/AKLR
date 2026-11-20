###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from aklr.core.runtime.dispatch import KernelFunction
from aklr.core.schema import Schema

__all__ = ["BuiltKernel"]


class BuiltKernel:
    def __init__(self, schema: Schema, kernel: KernelFunction) -> None:
        self.schema = schema
        self.kernel = kernel

    @classmethod
    def build(cls, schema: Schema) -> BuiltKernel:
        kernel = KernelFunction(schema)
        kernel.runnable  # noqa: B018
        return cls(schema, kernel)

    def bind(self, inputs: Mapping[str, Any], axes: Mapping[str, int]) -> Callable[[], Any]:
        call = {name: value for name, value in inputs.items() if name not in self.schema.outputs}
        return lambda: self.kernel(call, {**self.schema.const_axes, **axes})
