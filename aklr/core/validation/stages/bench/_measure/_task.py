###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A bench task: the workloads of one schema checked and timed in one run, their kernels built once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from aklr.core.contract.op import OpContract as Schema
from aklr.core.validation._execution import RunnerCrashed, TaskTimeout
from aklr.core.validation.stages.bench._measure._built_kernel import BuiltKernel
from aklr.core.validation.stages.bench._measure._contract_workload import ContractWorkload
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
    # the class, not its op type: pickled by reference, so that the task's processes import the module defining it
    bench_type: type[OpBench]

    @property
    def ranks(self) -> int:
        """Ranks that run it together, one device each."""
        return self.bench_type.ranks(self.schema)

    @property
    def cost(self) -> float:
        """About 1 per workload plus 1 per GiB of one of its input sets."""
        return sum(
            1 + ContractWorkload.of(self.schema, w.key.axes).input_bytes / 2**30 for w in self.workloads
        )

    def failed(self, code: FindingCode, detail: str) -> list[WorkloadResult]:
        """The task's failure, standing for each of its workloads."""
        return [WorkloadResult.failed(self.schema_path, w.key, code, detail) for w in self.workloads]

    def results(self, future: ExecutionFuture) -> list[WorkloadResult]:
        """Its run's results; a run that crashed or timed out is the Candidate's for every workload."""
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
        compared = (r.comparison(self.profile) for r in results)
        speedups = [c.speedup for c in compared if c is not None]
        timed = f"speedup {min(speedups):.3f}-{max(speedups):.3f}" if speedups else "none timed"
        line = f"{schema_short_name(self.schema_path)}: {len(results)} workloads, {timed},"
        line += f" accuracy {len(results) - failed - unmeasured}/{len(results)} passed"
        return line + (f", {unmeasured} not measured" if unmeasured else "")

    def run(self, context: ExecutionContext) -> str:
        """The run, on every rank of its group: set up the op's processes, build the Candidate, then measure every
        workload by the op's bench, each step agreed on by every rank; the JSON of their results, the same on
        every rank. What fails a step is never the bench's to decide: the setup is the platform's, the build
        the Candidate's, and a workload's finding its result."""
        import torch
        import torch.distributed as dist

        from aklr.core.validation.stages.stage import Progress

        if dist.get_world_size() != self.ranks:
            raise RuntimeError(
                f"{self.schema_path} runs on {self.ranks} ranks, its task on {dist.get_world_size()}"
            )
        count = len(self.workloads)
        device = torch.device("cuda", torch.cuda.current_device())  # the runner set this rank's device
        Progress("bench").write(f"running {self.schema_path} ({count} workloads) on {device}")
        with torch.no_grad():
            try:
                with FindingCode.PLATFORM_FAULT.failing_every_rank("the op's setup"):  # no kernel ran yet
                    self.bench_type.setup(self.schema)
                with FindingCode.BUILD_FAILED.failing_every_rank("building the Candidate"):
                    candidate = BuiltKernel.build(self.schema, device)
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
