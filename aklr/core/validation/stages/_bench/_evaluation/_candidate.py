###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, NamedTuple

from pydantic import TypeAdapter

from aklr.core._frozen_model import frozen_dataclass
from aklr.core.schema import Schema
from aklr.core.utils import dtype_str_to_torch_dtype
from aklr.core.validation.stages import _measure
from aklr.core.validation.stages._bench._evaluation._safetensors import (
    outputs_file_name,
)
from aklr.core.validation.stages._bench._evaluation._trials import Step, TrialRequest
from aklr.core.validation.stages._bench._execution import ExecutionContext
from aklr.core.validation.stages._measure import derive_seed
from aklr.core.validation.stages._report._finding import Finding, FindingCode

if TYPE_CHECKING:
    from torch import Tensor

    from aklr.core.runtime.builder import Runnable

__all__ = [
    "MAX_FIRST_CALL_BYTES", "CandidateResult", "CandidateWorkloadResult",
    "StepFailed", "run_candidate",
]  # fmt: skip

# bounds both the first calls' peak, outputs included, and what stays allocated after them
MAX_FIRST_CALL_BYTES: Final = 16 << 30

_BASELINE_FAILED_DETAIL: Final = (
    "no adapter of the api binds this workload and a rerun will not help: remove "
    "the workload or fix the schema"
)


@frozen_dataclass
class CandidateWorkloadResult:
    axes_text: str
    findings: tuple[Finding, ...]
    rounds: tuple[_measure.RoundTimes, ...]
    baseline: str | None
    backend: str | None


@frozen_dataclass
class CandidateResult:
    findings: tuple[Finding, ...]
    workload_results: tuple[CandidateWorkloadResult, ...]


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


def run_candidate(context: ExecutionContext, *, requests: Sequence[TrialRequest]) -> dict:
    """Check and time the batch on ``context.device``; returns a ``CandidateResult`` json."""
    import torch
    from safetensors.torch import save_file

    from aklr.core.platform.device import DeviceTarget
    from aklr.core.utils.testing import flush_l2_cache

    schema = requests[0].schema
    batch_findings: tuple[Finding, ...] = ()
    stopped: dict[str, Finding] = {}
    rounds: dict[str, tuple[_measure.RoundTimes, ...]] = {}
    baselines: dict[str, tuple[str, str] | None] = {}
    runnable = None
    incumbent_runnables: dict[str, Runnable] = {}
    target = DeviceTarget.current(context.device)
    try:
        if not target.is_complete:
            detail = f"{context.device}: unknown arch or hardware ({target})"
            raise StepFailed(FindingCode.PLATFORM_FAILURE, detail)
        if not schema.kernel.build.runs_on(target):
            detail = f"{target} matches no kernel.build.target_hardware"
            raise StepFailed(FindingCode.NO_MATCHING_DEVICE, detail)
        with StepFailed.failing_as(FindingCode.PLATFORM_FAILURE, "compile tripwire: "):
            tripwire = _CompileTripwire.resolve()
        run = CandidateRun(context.device, tripwire, _measure.bind_clock())
        flush_l2_cache()
        run.clock.synchronize()
        for request in requests:
            try:
                with StepFailed.failing_as(FindingCode.PLATFORM_FAILURE, "baseline: "):
                    run.run_baseline(request)
            except StepFailed as e:
                stopped[request.axes_text] = Finding.unlocated(e.code, str(e))
        run.clock.synchronize()
        live = [r for r in requests if r.axes_text not in stopped]
        if live:
            before = torch.cuda.memory_allocated()
            with StepFailed.failing_as(FindingCode.BUILD_FAILED):
                runnable, first_call_peak_bytes = run.make_warmed_runnable(schema, live)
            resident_bytes = torch.cuda.memory_allocated() - before
            if max(first_call_peak_bytes, resident_bytes) > MAX_FIRST_CALL_BYTES:
                detail = f"peak {first_call_peak_bytes} or resident {resident_bytes} bytes: too many"
                raise StepFailed(FindingCode.BUILD_FAILED, detail)
            # the incumbents passed their review: a failure here is the platform's fault
            with StepFailed.failing_as(FindingCode.PLATFORM_FAILURE, "incumbent: "):
                requests_by_hash: dict[str, list[TrialRequest]] = {}
                for request in live:
                    if request.incumbent is not None:
                        content_hash = request.incumbent.kernel.hash()
                        requests_by_hash.setdefault(content_hash, []).append(request)
                for requests_of_hash in requests_by_hash.values():
                    incumbent_runnable, _ = run.make_warmed_runnable(
                        requests_of_hash[0].incumbent, requests_of_hash
                    )
                    incumbent_runnables |= dict.fromkeys(
                        (r.axes_text for r in requests_of_hash), incumbent_runnable
                    )
    except StepFailed as e:
        batch_findings = (Finding.unlocated(e.code, str(e)),)
    for request in requests:
        axes_text = request.axes_text
        outputs: dict[str, Tensor] = {}
        if not batch_findings and axes_text not in stopped:
            try:
                rounds[axes_text], baselines[axes_text] = run.check_and_time(
                    request, runnable, incumbent_runnables.get(axes_text), outputs
                )
            except StepFailed as e:
                stopped[axes_text] = Finding.unlocated(e.code, str(e))
        # even with no tensor: a missing file is then always the candidate's fault
        save_file(outputs, context.directory / outputs_file_name(axes_text))
    workload_results = []
    for r in requests:
        baseline, backend = baselines.get(r.axes_text) or (None, None)
        workload_results.append(
            CandidateWorkloadResult(
                axes_text=r.axes_text,
                findings=(stopped[r.axes_text],) if r.axes_text in stopped else (),
                rounds=rounds.get(r.axes_text, ()),
                baseline=baseline,
                backend=backend,
            )
        )
    value = CandidateResult(findings=batch_findings, workload_results=workload_results)
    return TypeAdapter(CandidateResult).dump_python(value, mode="json")


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

        # private compile entry points of triton and FlyDSL: a release that renames one fails here
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
    device: str
    tripwire: _CompileTripwire
    clock: _measure.Clock

    def __init__(self, device: str, tripwire: _CompileTripwire, clock: _measure.Clock) -> None:
        self.device = device
        self.tripwire = tripwire
        self.clock = clock

    def bind_baseline(
        self, request: TrialRequest, call: Mapping[str, Any]
    ) -> tuple[Callable[[], Any], str, str] | None:
        import aklr.seam  # noqa: F401 -- declares every adapter
        from aklr.core.runtime.seam import SeamRegistry

        adapters = SeamRegistry.default().adapters(request.schema.op_type)
        if not adapters:
            return None
        failure: Exception | None = None
        for adapter in adapters:
            try:
                adapter.original()
            except Exception as e:  # noqa: BLE001
                failure = failure or e
                continue
            backend_call = adapter.baseline(call)
            if backend_call is not None:
                return backend_call, adapter.entry, adapter.backend
        if failure is not None:
            raise failure
        raise StepFailed(FindingCode.BASELINE_FAILED, _BASELINE_FAILED_DETAIL)

    def run_baseline(self, request: TrialRequest) -> None:
        bound = self.bind_baseline(request, request.compose_call(None, self.device))
        if bound is not None:
            bound[0]()

    @staticmethod
    def call_kernel(request: TrialRequest, runnable: Runnable, call: Mapping[str, Any]) -> Any:
        from aklr.core.runtime.dispatch._adapt import adapt_and_call

        return adapt_and_call(
            template=request.schema,
            fn=runnable,
            namespace=dict(call),
            dps=runnable.metadata.destination_passing_style,
            is_inplace=False,
            dests={o: o for o in request.schema.outputs},
            axes=request.workload.axes,
        )

    def make_warmed_runnable(
        self, kernel_schema: Schema, requests: Sequence[TrialRequest]
    ) -> tuple[Runnable, int]:
        import torch

        from aklr.core.runtime.builder import BuilderRegistry

        runnable = BuilderRegistry.get_instance().build(kernel_schema)
        peak_bytes = 0
        for request in requests:
            call = request.compose_call(None, self.device)
            torch.cuda.synchronize()
            allocated_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            self.call_kernel(request, runnable, call)
            torch.cuda.synchronize()
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated() - allocated_before)
        return runnable, peak_bytes

    def check_and_time(
        self,
        request: TrialRequest,
        runnable: Runnable,
        incumbent_runnable: Runnable | None,
        outputs: dict[str, Tensor],
    ) -> tuple[tuple[_measure.RoundTimes, ...], tuple[str, str] | None]:
        steps = request.steps
        self.produce_outputs(request, runnable, [s for s in steps if s.check != "post_timing"], outputs)
        call = request.compose_call(None, self.device)
        fns = {"candidate": lambda: self.call_kernel(request, runnable, call)}
        if incumbent_runnable is not None:
            fns["incumbent"] = lambda: self.call_kernel(request, incumbent_runnable, call)
        where = f"{request.schema_path}:{request.axes_text}"
        try:
            bound = self.bind_baseline(request, call)
            if bound is None:
                if "incumbent" in fns:
                    fns["baseline"] = fns.pop("incumbent")
            else:
                fns["baseline"] = bound[0]
            rounds = ()
            if "baseline" in fns:
                seed = derive_seed(request.workload_hash, "timing")
                rounds = self.clock.time_rounds(fns, seed=seed)
        except StepFailed:
            raise
        except Exception as e:
            text = f"{where}: timing: {type(e).__name__}: {e}"
            raise RuntimeError(text) from e
        self.produce_outputs(
            request,
            runnable,
            [s for s in steps if s.check == "post_timing"],
            outputs,
            timing_call=call,
        )
        return rounds, None if bound is None else (bound[1], bound[2])

    def produce_outputs(
        self,
        request: TrialRequest,
        runnable: Runnable,
        steps: Sequence[Step],
        outputs: dict[str, Tensor],
        timing_call: Mapping[str, Any] | None = None,
    ) -> None:
        import torch

        schema = request.schema
        output_names = tuple(schema.outputs)
        output_shapes = request.output_shapes
        live = step_outputs = graph = None
        for step in steps:
            label = step.label
            values = request.compose_call(step, self.device)
            if step.buffer_source == "new":
                live = values
            elif step.buffer_source == "timing":
                live = dict(timing_call)
            held = {name: v for name, v in live.items() if torch.is_tensor(v) and name not in output_names}
            if step.buffer_source != "new":
                for name, value in held.items():
                    # like a model's weight reload: new values at the same address and _version
                    value.data.copy_(values[name])
            call = live | (step_outputs if step.is_out_reused else {o: values[o] for o in output_names})
            outputs_to_poison = () if step.is_out_reused else output_names
            # on the device: a CPU copy of a multi-GB pool costs more than the call
            snapshots = {name: v.detach().clone() for name, v in held.items()}
            if step.mode == "capture":
                description = f"{label}: capture: "
                try:
                    with StepFailed.failing_as(FindingCode.CAPTURE_UNSAFE, description):
                        graph = torch.cuda.CUDAGraph()
                        with (
                            self.tripwire.armed() as compile_calls,
                            torch.cuda.graph(graph, capture_error_mode="global"),
                        ):
                            # captured too: each replay poisons the outputs first
                            self._poison_outputs(call, outputs_to_poison)
                            self.call_kernel(request, runnable, call)
                        if compile_calls:
                            raise RuntimeError(f"{compile_calls[0]}: compiled in the capture")
                except Exception:
                    raise
                except BaseException as e:
                    text = f"{description}{type(e).__name__}: {e}"
                    raise StepFailed(FindingCode.CAPTURE_UNSAFE, text) from e
            description = f"{label}: replay: "
            try:
                with StepFailed.failing_as(FindingCode.CAPTURE_UNSAFE, description):
                    graph.replay()
            except Exception:
                raise
            except BaseException as e:
                text = f"{description}{type(e).__name__}: {e}"
                raise StepFailed(FindingCode.CAPTURE_UNSAFE, text) from e
            step_outputs = {o: call[o] for o in output_names}
            for o, out in step_outputs.items():
                shape, dtype = output_shapes[o], schema.outputs[o].dtype
                is_declared = torch.is_tensor(out) and tuple(out.shape) == shape
                if not is_declared or out.dtype != dtype_str_to_torch_dtype(dtype):
                    detail = f"{label}: the output {o} is not a {shape} {dtype} tensor"
                    raise StepFailed(FindingCode.CANDIDATE_CRASH, detail)
            is_unchanged = all(call[name] is v for name, v in held.items()) and all(
                torch.equal(_bytes(v.detach()), _bytes(snapshots[name])) for name, v in held.items()
            )
            if not is_unchanged:
                detail = f"{label}: the call changed its inputs"
                raise StepFailed(FindingCode.INPUT_MUTATION, detail)
            for o, out in step_outputs.items():
                outputs[step.output_key(o)] = out.to("cpu", copy=True).contiguous()

    @staticmethod
    def _poison_outputs(call: Mapping[str, Any], output_names: Iterable[str]) -> None:
        import torch

        for o in output_names:
            out = call[o]
            if out.dtype.is_floating_point and out.dtype.itemsize > 1:
                out.fill_(math.nan)
            else:
                out.view(torch.uint8).fill_(0xFF)
