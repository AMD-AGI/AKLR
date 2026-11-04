# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, NamedTuple

from aklr.core.contract import semantics
from aklr.core.contract.op import OpContract
from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages import _measure
from aklr.core.validation.stages._bench._evaluation._trials import Step, TrialRequest
from aklr.core.validation.stages._bench._execution import (
    ExecutionContext,
    TaskExecutionTimeout,
    caused_by_the_task,
)
from aklr.core.validation.stages._report._finding import Finding, FindingCode

if TYPE_CHECKING:
    from torch import Tensor

    from aklr.core.runtime.dispatch import KernelDispatcher

__all__ = [
    "MAX_FIRST_CALL_BYTES", "CandidateResult", "CandidateWorkloadResult",
    "StepFailed", "run_candidate",
]  # fmt: skip

# the first calls' peak, outputs included, and what stays allocated after them
MAX_FIRST_CALL_BYTES: Final = 16 << 30

_BASELINE_FAILED_DETAIL: Final = (
    "the fallback does not serve this workload and a rerun will not help: remove "
    "the workload or fix the contract's domain"
)


class CandidateWorkloadResult(FrozenModel):
    axes_text: str
    findings: tuple[Finding, ...]
    rounds: tuple[_measure.RoundTimes, ...]
    # the lent tensors it filled (Step.output_key), in step order
    output_keys: tuple[str, ...]

    @property
    def is_declined(self) -> bool:
        """Whether the baseline declined the workload: its finding alone stands, nothing is
        evaluated."""
        return any(f.code == FindingCode.BASELINE_FAILED for f in self.findings)


class CandidateResult(FrozenModel):
    findings: tuple[Finding, ...]
    workload_results: tuple[CandidateWorkloadResult, ...]

    @classmethod
    def from_task(cls, value: Any, axes_texts: Sequence[str]) -> CandidateResult:
        detail = "the candidate task's result is not one for the batch"
        try:
            result = cls.model_validate(value)
        except ValueError as e:
            raise ValueError(detail) from e
        if sorted(r.axes_text for r in result.workload_results) != sorted(axes_texts):
            raise ValueError(detail)
        return result

    @classmethod
    def of_failure(cls, error: BaseException) -> CandidateResult:
        """The batch finding of a candidate task that gave no result: the last line of
        ``error``, whose type says whether the candidate or the platform failed."""
        detail = str(error).strip().split("\n")[-1]
        if isinstance(error, TaskExecutionTimeout):
            code = FindingCode.CANDIDATE_TIMEOUT
        elif caused_by_the_task(error):
            # a tracer fault crashes the timing by itself; only the last line is
            # aklr's text, the log and traceback above it are the candidate's
            is_tracer = ": timing: " in detail and _measure.is_tracer_fault(detail)
            code = FindingCode.PLATFORM_FAILURE if is_tracer else FindingCode.CANDIDATE_CRASH
        elif isinstance(error, ValueError):
            code = FindingCode.CANDIDATE_CRASH
        else:
            code = FindingCode.PLATFORM_FAILURE
        return cls(findings=(Finding.unlocated(code, detail),), workload_results=())


class StepFailed(Exception):
    code: FindingCode

    def __init__(self, code: FindingCode, message: str) -> None:
        super().__init__(message)
        self.code = code

    @classmethod
    @contextlib.contextmanager
    def failing_as(cls, code: FindingCode, description: str = "") -> Iterator[None]:
        try:
            yield
        except StepFailed:
            raise
        except Exception as e:
            is_io = isinstance(e, OSError) or isinstance(e.__cause__, OSError)
            ended_as = FindingCode.PLATFORM_FAILURE if is_io else code
            raise cls(ended_as, f"{description}{type(e).__name__}: {e}") from e


def run_candidate(
    context: ExecutionContext,
    *,
    requests: Sequence[TrialRequest],
    outputs: Mapping[str, Mapping[str, Tensor]],
) -> dict:
    """Check and time the batch on ``context.device``, each workload's outputs copied
    into the tensors the trial task lent (``outputs[axes_text]``,
    ``TrialRequest.empty_outputs``); returns a ``CandidateResult`` json."""
    import torch

    from aklr.core.platform.device import DeviceTarget
    from aklr.core.platform.toolchain import unmet_requirement
    from aklr.core.utils.testing import flush_l2_cache

    schema = requests[0].schema
    batch_findings: tuple[Finding, ...] = ()
    stopped: dict[str, Finding] = {}
    rounds: dict[str, tuple[_measure.RoundTimes, ...]] = {}
    dispatcher = None
    incumbent_dispatchers: dict[str, KernelDispatcher] = {}
    target = DeviceTarget.current(context.device)
    try:
        if not target.is_complete:
            detail = f"{context.device}: unknown arch or hardware ({target})"
            raise StepFailed(FindingCode.PLATFORM_FAILURE, detail)
        if not DeviceTarget.matches_target(target, schema.kernel.targets):
            detail = f"{target} matches no target of kernel.targets"
            raise StepFailed(FindingCode.NO_MATCHING_DEVICE, detail)
        if (unmet := unmet_requirement(schema.kernel.requires)) is not None:
            detail = f"kernel.requires: {unmet}"
            raise StepFailed(FindingCode.NO_MATCHING_TOOLCHAIN, detail)
        with StepFailed.failing_as(FindingCode.PLATFORM_FAILURE, "compile tripwire: "):
            tripwire = _CompileTripwire.resolve()
        run = CandidateRun(context.device, tripwire, _measure.bind_clock())
        flush_l2_cache()
        run.clock.synchronize()
        for request in requests:
            try:
                with StepFailed.failing_as(FindingCode.PLATFORM_FAILURE, "baseline: "):
                    try:
                        run.run_baseline(request)
                    except NotImplementedError as e:
                        raise StepFailed(FindingCode.BASELINE_FAILED, _BASELINE_FAILED_DETAIL) from e
            except StepFailed as e:
                stopped[request.axes_text] = Finding.unlocated(e.code, str(e))
        run.clock.synchronize()
        live = [r for r in requests if r.axes_text not in stopped]
        if live:
            before = torch.cuda.memory_allocated()
            with StepFailed.failing_as(FindingCode.BUILD_FAILED):
                dispatcher, first_call_peak_bytes = run.make_warmed_dispatcher(schema, live)
            resident_bytes = torch.cuda.memory_allocated() - before
            if max(first_call_peak_bytes, resident_bytes) > MAX_FIRST_CALL_BYTES:
                detail = f"peak {first_call_peak_bytes} or resident {resident_bytes} bytes: too many"
                raise StepFailed(FindingCode.BUILD_FAILED, detail)
            # one dispatcher per incumbent kernel; the incumbents passed their review
            with StepFailed.failing_as(FindingCode.PLATFORM_FAILURE, "incumbent: "):
                requests_by_hash: dict[str, list[TrialRequest]] = {}
                for request in live:
                    if request.incumbent is not None:
                        content_hash = request.incumbent.kernel.content_hash
                        requests_by_hash.setdefault(content_hash, []).append(request)
                for requests_of_hash in requests_by_hash.values():
                    incumbent_dispatcher, _ = run.make_warmed_dispatcher(
                        requests_of_hash[0].incumbent, requests_of_hash
                    )
                    incumbent_dispatchers |= dict.fromkeys(
                        (r.axes_text for r in requests_of_hash), incumbent_dispatcher
                    )
    except StepFailed as e:
        batch_findings = (Finding.unlocated(e.code, str(e)),)
    output_keys: dict[str, tuple[str, ...]] = {}
    for request in requests:
        axes_text = request.axes_text
        filled: list[str] = []
        if not batch_findings and axes_text not in stopped:
            try:
                rounds[axes_text] = run.check_and_time(
                    request, dispatcher, incumbent_dispatchers.get(axes_text), outputs[axes_text], filled
                )
            except StepFailed as e:
                stopped[axes_text] = Finding.unlocated(e.code, str(e))
        output_keys[axes_text] = tuple(filled)
    if context.device != "cpu":
        torch.cuda.synchronize()  # the lent tensors are filled before the result says so
    workload_results = [
        CandidateWorkloadResult(
            axes_text=r.axes_text,
            findings=(stopped[r.axes_text],) if r.axes_text in stopped else (),
            rounds=rounds.get(r.axes_text, ()),
            output_keys=output_keys[r.axes_text],
        )
        for r in requests
    ]
    value = CandidateResult(findings=batch_findings, workload_results=workload_results)
    return value.model_dump(mode="json")


def _bytes(tensor: Tensor) -> Tensor:
    """``tensor``'s values as bytes: NaN payloads and -0 compare too."""
    import torch

    return tensor.contiguous().reshape(-1).view(torch.uint8)


class _CompileTripwire(NamedTuple):
    hooks: tuple[tuple[object, str], ...]

    @classmethod
    def resolve(cls) -> _CompileTripwire:
        import flydsl.compiler.jit_function as flydsl_jit
        import triton.runtime.jit as triton_jit

        hooks = (
            (triton_jit.JITFunction, "_do_compile"),
            (flydsl_jit, "_build_call_state"),
        )
        for owner, name in hooks:
            getattr(owner, name)
        return cls(hooks)

    @contextlib.contextmanager
    def armed(self) -> Iterator[list[str]]:
        originals = [getattr(owner, name) for owner, name in self.hooks]
        compile_calls: list[str] = []
        for owner, name in self.hooks:
            # a partial, not a function: no method binding on a class attribute
            trip = functools.partial(
                lambda hook_name, *args, **kwargs: (
                    compile_calls.append(hook_name),
                    (_ for _ in ()).throw(RuntimeError(f"{hook_name}: compiled in the capture")),
                ),
                name,
            )
            setattr(owner, name, trip)
        try:
            yield compile_calls
        finally:
            for (owner, name), original in zip(self.hooks, originals, strict=True):
                setattr(owner, name, original)


class CandidateRun:
    """The candidate task's run on its device, shared by every workload of the batch."""

    device: str
    tripwire: _CompileTripwire
    clock: _measure.Clock

    def __init__(self, device: str, tripwire: _CompileTripwire, clock: _measure.Clock) -> None:
        self.device = device
        self.tripwire = tripwire
        self.clock = clock

    def run_baseline(self, request: TrialRequest) -> None:
        baseline = request.contract.baseline()
        if baseline is None:
            return  # the new case is its own baseline: nothing to run
        row_major_args = None
        if request.has_workload_set:
            workload_step = next(s for s in request.steps if s.input_set == "workload")
            with contextlib.suppress(ValueError):
                row_major_args = request.compose_inputs(workload_step, self.device)
        if row_major_args is None:
            row_major_args = request.compose_inputs(None, self.device)
        baseline(**request.assemble_call(row_major_args, self.device))

    def make_warmed_dispatcher(
        self, kernel_schema: OpContract, requests: Sequence[TrialRequest]
    ) -> tuple[KernelDispatcher, int]:
        import torch

        from aklr.core.runtime.builder import BuilderRegistry
        from aklr.core.runtime.dispatch import (
            DispatchTable,
            FallbackReason,
            KernelDispatcher,
        )

        # the keys come from the candidate's schema: same-named files share them
        schema, contract = requests[0].schema, requests[0].contract
        runnable = BuilderRegistry.get_instance().build(kernel_schema)
        unmeasured = {"speedup": None, "accuracy": None}
        workloads = [r.workload.model_copy(update=unmeasured) for r in requests]
        dispatcher = KernelDispatcher(contract)
        dispatcher.set_table(
            DispatchTable.from_schemas(
                contract,
                torch.device(self.device),
                [(schema, runnable, workloads)],
                min_speedup=0,
            )
        )
        allocated = dict.fromkeys(contract.allocated_outputs)
        peak_bytes = 0
        for request in requests:
            # the workload's own scalars too: a kernel may compile per scalar value,
            # and the workload step runs in a capture
            steps = [None, *[s for s in request.steps if s.input_set == "workload"]]
            for step in steps:
                try:
                    row_major_args = request.compose_inputs(step, self.device)
                except ValueError:  # as in produce_outputs: the step is skipped
                    if step is None:
                        raise
                    continue
                call = request.assemble_call(row_major_args, self.device)
                torch.cuda.synchronize()
                allocated_before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                # the first call: it may compile
                first_result = dispatcher.dispatch(call | allocated)
                if isinstance(first_result, FallbackReason):
                    raise RuntimeError(  # noqa: TRY004 -- BUILD_FAILED, not a type error
                        f"{request.axes_text}: the first call fell back: {first_result.value}"
                    )
                torch.cuda.synchronize()
                peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated() - allocated_before)
        return dispatcher, peak_bytes

    def check_and_time(
        self,
        request: TrialRequest,
        dispatcher: KernelDispatcher,
        incumbent_dispatcher: KernelDispatcher | None,
        lent: Mapping[str, Tensor],
        filled: list[str],
    ) -> tuple[_measure.RoundTimes, ...]:
        """Run the workload's check steps, then its timing rounds; return the rounds."""
        from aklr.core.runtime.dispatch import FallbackReason

        contract, steps = request.contract, request.steps
        outputs_json = request.schema_json["outputs"]
        with self._poisoned_alloc_outputs(dispatcher, outputs_json):
            self.produce_outputs(
                request,
                dispatcher,
                [s for s in steps if s.check != "post_timing"],
                lent,
                filled,
            )
        call = request.assemble_call(request.compose_inputs(None, self.device), self.device)
        allocated = dict.fromkeys(contract.allocated_outputs)
        # a new dict per call: the dispatcher allocates into the one it gets
        fns = {"candidate": lambda: dispatcher.dispatch(call | allocated)}
        if incumbent_dispatcher is not None:
            fns["incumbent"] = lambda: incumbent_dispatcher.dispatch(call | allocated)
            # called once in make_warmed_dispatcher(); still serves the call?
            if isinstance(fns["incumbent"](), FallbackReason):
                detail = "internal: the incumbent's dispatch fell back"
                raise StepFailed(FindingCode.PLATFORM_FAILURE, detail)
        where = f"{request.schema_path}:{request.axes_text}"
        try:
            baseline = contract.baseline()
            if baseline is not None:
                fns["baseline"] = lambda: baseline(**call)
            elif "incumbent" in fns:  # no defined baseline: the incumbent is it
                fns["baseline"] = fns.pop("incumbent")
            rounds = ()
            if "baseline" in fns:  # none: the new case is its own baseline, untimed
                root_hash = semantics.workload_hash(request.schema_json["name"], request.workload_json)
                seed = semantics.derive_seed(root_hash, "timing")
                rounds = self.clock.time_rounds(
                    fns, seed=seed, thresholds=request.thresholds, min_speedup=request.min_speedup
                )
        except Exception as e:
            text = f"{where}: timing: {type(e).__name__}: {e}"
            raise RuntimeError(text) from e
        with self._poisoned_alloc_outputs(dispatcher, outputs_json):
            self.produce_outputs(
                request,
                dispatcher,
                [s for s in steps if s.check == "post_timing"],
                lent,
                filled,
                timing_call=call,
            )
        return rounds

    def produce_outputs(
        self,
        request: TrialRequest,
        dispatcher: KernelDispatcher,
        steps: Sequence[Step],
        lent: Mapping[str, Tensor],
        filled: list[str],
        timing_call: Mapping[str, Any] | None = None,
    ) -> None:
        """Run ``steps``, each step's mutable tensors copied into the lent tensors of its
        ``Step.output_key`` as it ends, the key then appended to ``filled``."""
        import torch

        from aklr.core.runtime.dispatch import FallbackReason

        contract = request.contract
        outputs_json = request.schema_json["outputs"]
        output_shapes = {o: contract.storage_shape_of(o, request.point) for o in contract.outputs}
        unset_outputs = dict.fromkeys(contract.allocated_outputs, None)
        live = result = step_outputs = graph = None
        for step in steps:
            label = step.label
            try:
                values = request.assemble_call(request.compose_inputs(step, self.device), self.device)
            except ValueError:
                if step.input_set != "workload":
                    raise
                continue
            if step.buffer_source == "new":
                live = values
            elif step.buffer_source == "timing":
                live = dict(timing_call)
            held = {
                name: v for name, v in live.items() if torch.is_tensor(v) and name not in contract.outputs
            }
            if step.buffer_source != "new":
                for name, value in held.items():
                    # Model weight reloads that preserve both address and _version.
                    value.data.copy_(values[name])
            # a reused step hands every output back as the last step left it
            reused = step_outputs if step.is_out_reused else unset_outputs
            call = live | reused
            outputs_to_poison = () if step.is_out_reused else contract.required_outputs
            # on the device: a CPU copy of a multi-GB pool costs more than the call
            read_only = {n: v for n, v in held.items() if n not in contract.mutable_inputs}
            snapshots = {name: v.detach().clone() for name, v in read_only.items()}
            if step.mode == "capture":
                description = f"{label}: capture: "
                try:
                    with StepFailed.failing_as(FindingCode.CAPTURE_UNSAFE, description):
                        graph = torch.cuda.CUDAGraph()
                        with (
                            self.tripwire.armed() as compile_calls,
                            torch.cuda.graph(graph, capture_error_mode="global"),
                        ):
                            # the poisoning is captured too
                            self._poison_outputs(call, outputs_to_poison, outputs_json)
                            result = dispatcher.dispatch(call)
                        if compile_calls:
                            raise RuntimeError(f"{compile_calls[0]}: compiled in the capture")
                except Exception:
                    raise
                except BaseException as e:
                    text = f"{description}{type(e).__name__}: {e}"
                    raise StepFailed(FindingCode.CAPTURE_UNSAFE, text) from e
            if isinstance(result, FallbackReason):
                detail = f"{label}: internal: dispatch fell back ({result.value})"
                raise StepFailed(FindingCode.PLATFORM_FAILURE, detail)
            description = f"{label}: replay: "
            try:
                with StepFailed.failing_as(FindingCode.CAPTURE_UNSAFE, description):
                    graph.replay()
            except Exception:
                raise
            except BaseException as e:
                text = f"{description}{type(e).__name__}: {e}"
                raise StepFailed(FindingCode.CAPTURE_UNSAFE, text) from e
            step_outputs = contract.outputs_by_name(result)
            for o, out in step_outputs.items():
                shape, dtype = output_shapes[o], outputs_json[o]["dtype"]["any_of"][0]
                is_declared = torch.is_tensor(out) and tuple(out.shape) == shape
                if not is_declared or out.dtype not in semantics.torch_dtypes(dtype):
                    detail = f"{label}: the output {o} is not a {shape} {dtype} tensor"
                    raise StepFailed(FindingCode.CANDIDATE_CRASH, detail)
            is_unchanged = all(call[name] is v for name, v in held.items()) and all(
                torch.equal(_bytes(v.detach()), _bytes(snapshots[name])) for name, v in read_only.items()
            )
            if not is_unchanged:
                detail = f"{label}: the call changed its inputs"
                raise StepFailed(FindingCode.INPUT_MUTATION, detail)
            for o, out in contract.call_mutable_tensors(result, call).items():
                key = step.output_key(o)
                # another storage dtype of the output holds the same bytes; copy_ would broadcast
                value = out.view(lent[key].dtype)
                if value.shape != lent[key].shape:
                    detail = f"{label}: {o} is {tuple(value.shape)}, not {tuple(lent[key].shape)}"
                    raise StepFailed(FindingCode.CANDIDATE_CRASH, detail)
                lent[key].copy_(value)
                filled.append(key)

    @staticmethod
    @contextlib.contextmanager
    def _poisoned_alloc_outputs(
        dispatcher: KernelDispatcher, outputs_json: Mapping[str, Any]
    ) -> Iterator[None]:
        table = dispatcher.table
        dispatcher.set_table(
            dataclasses.replace(
                table,
                # the outputs left None are known before the allocation fills them
                alloc_outputs=lambda key, call: (
                    outputs_to_poison := [o for o in outputs_json if call[o] is None],
                    table.alloc_outputs(key, call),
                    CandidateRun._poison_outputs(call, outputs_to_poison, outputs_json),
                )[1],
            )
        )
        try:
            yield
        finally:
            dispatcher.set_table(table)

    @staticmethod
    def _poison_outputs(
        call: Mapping[str, Any],
        output_names: Iterable[str],
        outputs_json: Mapping[str, Any],
    ) -> None:
        import torch

        for o in output_names:
            traits = semantics.DTYPE_TRAITS[outputs_json[o]["dtype"]["any_of"][0]]
            if traits.is_integral or traits.bits < 8:
                call[o].view(torch.uint8).fill_(0xFF)
            else:
                call[o].fill_(math.nan)
