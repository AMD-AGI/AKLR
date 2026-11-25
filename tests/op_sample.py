# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import aklr.plugins.aiter
from aklr.core.platform.device import DeviceTarget
from aklr.core.runtime.dispatch import Dispatcher
from aklr.core.schema import (
    Build,
    Evaluation,
    Kernel,
    Schema,
    SchemaCollection,
    SourceFile,
    Workload,
)
from aklr.core.validation.stages.bench import InputSampler
from aklr.core.validation.stages.workload_key import WorkloadKey

GEMM_KERNEL = "def run(a, b, bias, out):\n    out.copy_(a @ b.mT)\n"


def dispatcher(*schemas: Schema) -> Dispatcher:
    return Dispatcher(SchemaCollection(schemas={s.name: [s] for s in schemas}))


@dataclass(frozen=True)
class OpSample:
    """One op variant at small axes: the schema, calls and reference the bench would draw for it."""

    api: Callable[..., Any]
    name: str
    axes: Mapping[str, Any]

    @property
    def template(self) -> Schema:
        return next(s for s in self.api.variants if s.schema_name({}) == self.name)

    @property
    def key(self) -> WorkloadKey:
        """Its workload's key: the case's Var axes."""
        return WorkloadKey({a: v for a, v in self.axes.items() if a in self.template.var_axes})

    def call(self, device: str = "cpu", trial: int = 0) -> dict[str, Any]:
        """Rank 0's call of trial, as the bench draws it for this workload: a single rank's too."""
        return InputSampler(self.schema(), self.key.axes).draw(trial, 0, device)

    def schema(self, kernel: Kernel | None = None, workloads: Sequence[Workload] = ()) -> Schema:
        """The template at the case's axes, as for_call reads it off a call: one its init draws on the meta
        device, which has shapes and dtypes and no data."""
        return replace(
            self.template.for_call(InputSampler(self.template, dict(self.axes)).draw(0, 0, "meta")),
            kernel=kernel,
            workloads=list(workloads),
        )

    def served(
        self,
        source: str,
        *,
        dps: bool = True,
        evaluations: Sequence[Evaluation] | None = None,
    ) -> Schema:
        target = DeviceTarget.current()
        if evaluations is None:
            evaluations = [Evaluation(hardware=str(target), accuracy="passed")]
        kernel = Kernel(
            name=f"{self.name}_{'dps' if dps else 'returning'}",
            author="tests",
            build=Build(
                language="triton",
                target_hardware=[target.arch],
                entry_point="kernel.py::run",
                requires={"triton": ">=0"},
                destination_passing_style=dps,
            ),
            sources=[SourceFile(path="kernel.py", content=source)],
        )
        return self.schema(kernel, [Workload(axes=self.key.axes, evaluations=evaluations)])


GEMM_SAMPLE = OpSample(aklr.plugins.aiter.gemm_bf16_nt, "gemm_bf16_nt", {"M": 4, "N": 32, "K": 64})
