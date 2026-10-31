###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import Any, ClassVar

from aklr.core.validation.stages._bench._execution._model import (
    ExecutionContext,
    ExecutionTask,
)


class ExecutionEngine:
    # what the runner builds and hands a call; subclasses may widen it
    """Call what the task carries, on the resource that was chosen for it."""

    context_type: ClassVar[type[ExecutionContext]] = ExecutionContext

    def execute(self, task: ExecutionTask, context: ExecutionContext) -> Any:
        return task.call(context, **task.kwargs)
