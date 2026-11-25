###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
#


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aklr.core.schema import Schema, Tensor, extract_axes
from aklr.core.schema._workload import Workload


@dataclass(frozen=True)
class DispatchKey:
    consts: tuple[tuple[str, int], ...] = ()
    vars: tuple[tuple[str, int], ...] = ()
    dtypes: tuple[tuple[str, str | None], ...] = ()

    @classmethod
    def from_call(cls, schema: Schema, call: dict[str, Any]) -> DispatchKey:
        consts = extract_axes(schema.const_axis_extractors, call)
        var_axes = schema.get_axes_values_from_inputs(schema.ordered_inputs(call))
        return cls(
            tuple(sorted(consts.items())),
            tuple(sorted(var_axes.items())),
            tuple(sorted(schema.call_dtypes(call).items())),
        )

    @classmethod
    def from_workload(cls, schema: Schema, workload: Workload) -> DispatchKey:
        tensors = schema.inputs | schema.outputs
        return cls(
            tuple(sorted(schema.const_axes.items())),
            tuple(sorted(workload.axes.items())),
            tuple(sorted((name, t.dtype) for name, t in tensors.items() if isinstance(t, Tensor))),
        )
