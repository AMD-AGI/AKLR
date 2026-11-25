###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from aklr.core.schema import Schema, Tensor
from aklr.core.utils import dtype_str_to_torch_dtype
from aklr.core.validation._execution import RunnerCrashed, TaskTimeout
from aklr.core.validation.stages.bench._measure._built_kernel import BuiltKernel
from aklr.core.validation.stages.bench._measure._op_bench import OpBench
from aklr.core.validation.stages.bench._measure._profile import MeasurementProfile
from aklr.core.validation.stages.bench._measure._result import WorkloadResult
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised, Outcome
from aklr.core.validation.stages.report._markdown import schema_short_name

if TYPE_CHECKING:
    from aklr.core.validation._execution import ExecutionContext, ExecutionFuture
    from aklr.core.validation.stages.bench._plan._plan import WorkloadPlan

__all__ = ["BenchTask"]


@dataclass(frozen=True)
class BenchTask:
    schema_path: str
    schema: Schema
    workloads: tuple[WorkloadPlan, ...]
    profile: MeasurementProfile
    bench_type: type[OpBench]

    @property
    def ranks(self) -> int:
        return self.bench_type.ranks(self.schema)

    @property
    def cost(self) -> float:
        schema = self.schema
        total = 0.0
        for workload in self.workloads:
            try:
                shapes = schema.get_input_shapes({**schema.const_axes, **workload.key.axes})
            except ValueError:
                shapes = [None] * len(schema.inputs)
            size = sum(
                math.prod(shape) * dtype_str_to_torch_dtype(spec.dtype).itemsize
                for spec, shape in zip(schema.inputs.values(), shapes, strict=True)
                if shape is not None
                and None not in shape
                and isinstance(spec, Tensor)
                and spec.dtype is not None
            )
            total += 1 + size / 2**30
        return total

    def failed(self, code: FindingCode, detail: str) -> list[WorkloadResult]:
        return [WorkloadResult.failed(self.schema_path, w.key, code, detail) for w in self.workloads]

    def results(self, future: ExecutionFuture) -> list[WorkloadResult]:
        try:
            return TypeAdapter(list[WorkloadResult]).validate_json(future.result())
        except TaskTimeout as error:
            return self.failed(FindingCode.TASK_TIMEOUT, str(error))
        except RunnerCrashed as error:
            return self.failed(FindingCode.TASK_CRASHED, str(error))
        except ValueError as error:
            return self.failed(FindingCode.PLATFORM_FAULT, f"unparsable task result: {error}")
        except Exception as error:  # noqa: BLE001
            return self.failed(FindingCode.PLATFORM_FAULT, f"the task failed: {error}")

    def summary(self, results: list[WorkloadResult]) -> str:
        outcomes = [{FindingCode(f.code).outcome for f in r.findings} for r in results]
        failed = sum(Outcome.ACCURACY_FAILED in o for o in outcomes)
        unmeasured = sum(Outcome.NOT_MEASURED in o for o in outcomes)
        compared = (r.comparison(self.profile) for r in results)
        speedups = [c.speedup for c in compared if c is not None]
        timed = f"speedup {min(speedups):.3f}-{max(speedups):.3f}" if speedups else "none timed"
        line = f"{schema_short_name(self.schema_path)}: {len(results)} workloads, {timed},"
        line += f" accuracy {len(results) - failed - unmeasured}/{len(results)} passed"
        return line + (f", {unmeasured} not measured" if unmeasured else "")

    def run(self, context: ExecutionContext) -> str:
        import torch
        import torch.distributed as dist

        from aklr.core.validation.stages.stage import Progress

        if dist.get_world_size() != self.ranks:
            raise RuntimeError(
                f"{self.schema_path} runs on {self.ranks} ranks, its task on {dist.get_world_size()}"
            )
        count = len(self.workloads)
        device = torch.device("cuda", torch.cuda.current_device())
        Progress("bench").write(f"running {self.schema_path} ({count} workloads) on {device}")
        with torch.no_grad():
            try:
                with FindingCode.PLATFORM_FAULT.failing_every_rank("the op's setup"):
                    self.bench_type.setup(self.schema)
                with FindingCode.BUILD_FAILED.failing_every_rank("building the Candidate"):
                    candidate = BuiltKernel.build(self.schema)
            except FindingRaised as failure:
                results = self.failed(failure.code, failure.detail)
                return TypeAdapter(list[WorkloadResult]).dump_json(results).decode()
            bench = self.bench_type(self.schema_path, candidate, self.profile, device)
            results = []
            for plan in self.workloads:
                try:
                    results.append(bench.measure(plan))
                except FindingRaised as failure:
                    results.append(
                        WorkloadResult.failed(self.schema_path, plan.key, failure.code, failure.detail)
                    )
        return TypeAdapter(list[WorkloadResult]).dump_json(results).decode()
