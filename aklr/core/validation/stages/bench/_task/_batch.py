###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A batch: the workloads of one schema that one task checks and times, building their kernels once."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from aklr.core.schema import Schema, Tensor
from aklr.core.utils import dtype_str_to_torch_dtype
from aklr.core.validation._execution import RunnerCrashed, TaskTimeout
from aklr.core.validation.stages.bench._execution_policy import ExecutionPolicy
from aklr.core.validation.stages.bench._measurement._measurement_profile import MeasurementProfile
from aklr.core.validation.stages.bench._measurement._workload_result import WorkloadResult
from aklr.core.validation.stages.bench._task._batch_run import BatchRun
from aklr.core.validation.stages.bench._workload_call._kernel_call import KernelCall
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised, Outcome
from aklr.core.validation.stages.report._markdown import schema_short_name

if TYPE_CHECKING:
    from aklr.core.validation._execution import ExecutionContext, ExecutionFuture
    from aklr.core.validation.stages.bench._admission._plan import WorkloadPlan

__all__ = ["Batch"]


@dataclass(frozen=True)
class Batch:
    schema_path: str
    schema_text: str
    workloads: tuple[WorkloadPlan, ...]
    profile: MeasurementProfile
    execution_policy: ExecutionPolicy

    @cached_property
    def schema(self) -> Schema:
        return TypeAdapter(Schema).validate_json(self.schema_text)

    @property
    def cost(self) -> float:
        """About 1 per workload plus 1 per GiB of one of its input sets."""
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
                # a Const axis a call sets (no value in the schema) leaves its size unknown: not counted
                if shape is not None
                and None not in shape
                and isinstance(spec, Tensor)
                and spec.dtype is not None
            )
            total += 1 + size / 2**30
        return total

    def failed(self, code: FindingCode, detail: str) -> list[WorkloadResult]:
        """The batch's failure, standing for each of its workloads."""
        return [WorkloadResult.failed(self.schema_path, w.key, code, detail) for w in self.workloads]

    def results(self, future: ExecutionFuture) -> list[WorkloadResult]:
        """Its task's results; a task that crashed or timed out is the Candidate's for the whole batch."""
        try:
            return TypeAdapter(list[WorkloadResult]).validate_json(future.result())
        except TaskTimeout as error:
            return self.failed(FindingCode.TASK_TIMEOUT, str(error))
        except RunnerCrashed as error:
            return self.failed(FindingCode.TASK_CRASHED, str(error))
        except ValueError as error:  # pydantic's ValidationError among them
            return self.failed(FindingCode.PLATFORM_FAULT, f"unparsable task result: {error}")
        except Exception as error:  # noqa: BLE001 -- the task's own code raised: aklr's fault
            return self.failed(FindingCode.PLATFORM_FAULT, f"the task failed: {error}")

    def summary(self, results: list[WorkloadResult]) -> str:
        """One progress line: schema, workload count, speedup range and accuracy."""
        outcomes = [{FindingCode(f.code).outcome for f in r.findings} for r in results]
        failed = sum(Outcome.ACCURACY_FAILED in o for o in outcomes)
        unmeasured = sum(Outcome.NOT_MEASURED in o for o in outcomes)
        speedups = [r.comparison.speedup for r in results if r.comparison is not None]
        timed = f"speedup {min(speedups):.3f}-{max(speedups):.3f}" if speedups else "none timed"
        line = f"{schema_short_name(self.schema_path)}: {len(results)} workloads, {timed},"
        line += f" accuracy {len(results) - failed - unmeasured}/{len(results)} passed"
        return line + (f", {unmeasured} not measured" if unmeasured else "")

    def run(self, context: ExecutionContext) -> str:
        """The task, on every rank of its group: set up the op's processes, build, check and time every
        workload, each step agreed on by every rank; the JSON of their results, the same on every rank."""
        import torch
        import torch.distributed as dist

        from aklr.core.validation.stages.stage import Progress

        ranks = self.execution_policy.ranks_per_task
        if dist.get_world_size() != ranks:
            raise RuntimeError(
                f"{self.schema_path} runs on {ranks} ranks, its task on {dist.get_world_size()}"
            )
        count = len(self.workloads)
        device = torch.device("cuda", torch.cuda.current_device())  # the runner set this rank's device
        Progress("bench").write(f"running {self.schema_path} ({count} workloads) on {device}")
        with torch.no_grad():
            try:
                with FindingCode.PLATFORM_FAULT.failing_every_rank("the op's setup"):  # no kernel ran yet
                    self.execution_policy.setup()
                with FindingCode.BUILD_FAILED.failing_every_rank("building the Candidate"):
                    candidate = KernelCall.build(self.schema)
            except FindingRaised as failure:
                results = self.failed(failure.code, failure.detail)
                return TypeAdapter(list[WorkloadResult]).dump_json(results).decode()
            batch_run = BatchRun(self.schema_path, candidate, self.profile, device)
            results = [batch_run.result(plan) for plan in self.workloads]
        return TypeAdapter(list[WorkloadResult]).dump_json(results).decode()
