# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A toy group op for the bench tests: each rank's out is the sum of every rank's x; never touches a GPU on import."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from aklr.core.schema import Const, Kernel, Schema, Standard, Tensor, Var, Workload
from aklr.core.validation import ExecutionPolicy


class ToyGroupExecutionPolicy(ExecutionPolicy, name="toy_group"):
    @property
    def ranks_per_task(self) -> int:
        return self.schema.const_axes["world"]


def _init(n, device, seed):
    torch.manual_seed(seed)
    return {"x": torch.randn(n, device=device)}


def _reference(x, group_calls):
    return sum(call["x"] for call in group_calls)


def _check(reference_outputs, actual_outputs, **_unused):
    return torch.allclose(reference_outputs[0], actual_outputs[0], rtol=1e-5, atol=1e-5)


def toy_group_schema(
    kernel: Kernel | None = None, workloads: Sequence[Workload] = (), world: int = 2
) -> Schema:
    """The toy group op's schema on world ranks."""
    return Schema(
        op_type="toy_group",
        axes={"n": Var("Elements."), "world": Const("Ranks of the group.", value=world)},
        inputs={"x": Tensor(["n"], dtype="float32")},
        outputs={"out": Tensor(["n"], dtype="float32")},
        standard=Standard(init=_init, reference=_reference, check=_check),
        kernel=kernel,
        workloads=list(workloads),
    )


# the toy op's template, where the bench finds the standard of a toy op's json
Schema.declare_template(toy_group_schema())
