# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A toy group op for the bench tests: each rank's out is the sum of every rank's x; never touches a GPU on import."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from aklr.core.runtime.dispatch import aklr_api
from aklr.core.schema import Const, Kernel, Schema, Standard, Tensor, Var, Workload
from aklr.core.validation import OpBench


class ToyGroupBench(OpBench, name="toy_group"):
    @classmethod
    def ranks(cls, schema: Schema) -> int:
        return schema.const_axes["world"]


class ToyClockBench(ToyGroupBench, name="toy_clock"):
    """A bench with its own clock: every call reads CLOCK_MS."""

    CLOCK_MS = 2.5

    def clock(self, fn) -> float:
        fn()
        return self.CLOCK_MS


def _init(n, device, seed):
    torch.manual_seed(seed)
    return {"x": torch.randn(n, device=device)}


def _reference(x, group_calls):
    return sum(call["x"] for call in group_calls)


def _check(reference_outputs, actual_outputs, **_unused):
    return torch.allclose(reference_outputs[0], actual_outputs[0], rtol=1e-5, atol=1e-5)


def toy_group_schema(
    kernel: Kernel | None = None,
    workloads: Sequence[Workload] = (),
    world: int = 2,
    op_type: str = "toy_group",
) -> Schema:
    """The toy group op's schema on world ranks; op_type toy_clock is the same op benched by ToyClockBench."""
    return Schema(
        op_type=op_type,
        axes={"n": Var("Elements."), "world": Const("Ranks of the group.", value=world)},
        inputs={"x": Tensor(["n"], dtype="float32")},
        outputs={"out": Tensor(["n"], dtype="float32")},
        standard=Standard(init=_init, reference=_reference, check=_check),
        kernel=kernel,
        workloads=list(workloads),
    )


# the toy op's api, where the bench finds the standard of a toy op's json
@aklr_api(toy_group_schema())
def toy_group(x, out=None): ...


@aklr_api(toy_group_schema(op_type="toy_clock"))
def toy_clock(x, out=None): ...
