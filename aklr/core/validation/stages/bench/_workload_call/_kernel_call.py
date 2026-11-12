###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A built kernel's call, made the way dispatch makes it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from aklr.core.runtime.dispatch._adapt import adapt_and_call
from aklr.core.schema import Schema

__all__ = ["KernelCall"]


class KernelCall:
    def __init__(self, schema: Schema, runnable: Any) -> None:
        self.schema = schema
        self.runnable = runnable

    @classmethod
    def build(cls, schema: Schema) -> KernelCall:
        from aklr.core.runtime.builder import BuilderRegistry

        return cls(schema, BuilderRegistry.get_instance().build(schema))

    def bind(self, namespace: Mapping[str, Any], axes: Mapping[str, int]) -> Callable[[], Any]:
        """The kernel on namespace's inputs through the dispatch adapter, into fresh output buffers."""
        return lambda: adapt_and_call(
            template=self.schema,
            fn=self.runnable,
            namespace=dict(namespace),
            dps=self.runnable.metadata.destination_passing_style,
            is_inplace=False,
            dests={},
            axes={**self.schema.const_axes, **axes},
        )
