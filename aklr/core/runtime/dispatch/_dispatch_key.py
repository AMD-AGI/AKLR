###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).

"""A dispatch key: what a call or a workload is looked up by."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aklr.core.schema import Schema, Tensor, extract_axes
from aklr.core.schema._workload import Workload


@dataclass(frozen=True)
class DispatchKey:
    """A call's or workload's Const and Var axis values and its tensors' dtypes, each as sorted (name, value)
    pairs. A dtype is None for an optional tensor the call left out."""

    consts: tuple[tuple[str, int], ...] = ()
    vars: tuple[tuple[str, int], ...] = ()
    dtypes: tuple[tuple[str, str | None], ...] = ()

    @classmethod
    def from_call(cls, schema: Schema, call: dict[str, Any]) -> DispatchKey:
        """The key of call, the api's bound arguments; raises ValueError if two inputs disagree on a Var axis."""
        consts = extract_axes(schema.const_axis_extractors, call)
        var_axes = schema.get_axes_values_from_inputs(schema.ordered_inputs(call))
        return cls(
            tuple(sorted(consts.items())),
            tuple(sorted(var_axes.items())),
            tuple(sorted(schema.call_dtypes(call).items())),
        )

    @classmethod
    def from_workload(cls, schema: Schema, workload: Workload) -> DispatchKey:
        """The key of workload, one of schema's: the dtypes are the ones schema was recorded with."""
        tensors = schema.inputs | schema.outputs
        return cls(
            tuple(sorted(schema.const_axes.items())),
            tuple(sorted(workload.axes.items())),
            tuple(sorted((name, t.dtype) for name, t in tensors.items() if isinstance(t, Tensor))),
        )
