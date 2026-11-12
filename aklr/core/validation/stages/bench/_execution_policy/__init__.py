###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""An op type's execution policy: where and in what processes its tasks run, never what they check."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aklr.core.common import Registrable

if TYPE_CHECKING:
    from aklr.core.schema import Schema

__all__ = ["ExecutionPolicy"]


class ExecutionPolicy(Registrable):
    """The default: one rank, a reused process, no setup; an op that differs registers a subclass in <op_type>.py."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    @staticmethod
    def for_schema(schema: Schema) -> ExecutionPolicy:
        """The policy registered under schema.op_type, else this default."""
        import importlib
        import pkgutil

        for module in pkgutil.iter_modules(__path__):
            if not module.name.startswith("_"):
                importlib.import_module(f"{__name__}.{module.name}")
        try:
            kind = ExecutionPolicy.resolve(schema.op_type)
        except ValueError:  # no policy registered under this op type
            kind = ExecutionPolicy
        return kind(schema)

    @property
    def ranks_per_task(self) -> int:
        """Ranks that run a task together, one device each."""
        return 1

    @property
    def isolate(self) -> bool:
        """Whether each task runs in new processes, as --isolate makes it."""
        return self.ranks_per_task > 1

    def setup(self) -> None:
        """Prepare each rank's process before the build; idempotent unless isolate; no global numeric setting."""
