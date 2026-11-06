# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The bench of an op that ranks call together: devices leased as a group, the
group's timing, its child task a rank, and a group the devices cannot hold."""

import itertools
import math
import multiprocessing
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from weakref import WeakSet

import pytest

from aklr.core.contract.op import OpContract
from aklr.core.validation import BenchStage, ValidationScope
from aklr.core.validation.stages._bench._evaluation import (
    CandidateResult,
    CandidateWorkloadResult,
    TrialRequest,
)
from aklr.core.validation.stages._bench._evaluation._reference import (
    ReferenceWorkloadResult,
    WorkloadEvaluation,
)
from aklr.core.validation.stages._bench._evaluation._trial import TrialResult, run_trial
from aklr.core.validation.stages._bench._execution import (
    ExecutionContext,
    Executor,
    TaskExecutionFailed,
)
from aklr.core.validation.stages._bench._execution._resources import (
    LocalAcceleratorCatalog,
)
from aklr.core.validation.stages._measure import GRAPH_BUDGET_MILLISECONDS, Clock, RoundTimes
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from tests.op_case import OpCase

_SCHEMA = "aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/09c4acd65577d824.json"
_TRACER_FAULT = "x: timing: rocprofiler-sdk dropped 1 records"


def _time_collective(rank: int, rendezvous: str) -> None:
    import datetime

    import torch
    import torch.distributed as dist

    from aklr.core.utils.testing import graph_kernel_timing_environment
    from aklr.core.validation.stages._measure import bind_clock

    with graph_kernel_timing_environment():
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            init_method=rendezvous,
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=120),
        )
        try:
            group = dist.group.WORLD
            clock = bind_clock(group)
            value = torch.zeros(16, device="cuda")
            executions = torch.zeros(1, dtype=torch.int64, device="cuda")
            captured = 0

            def collective():
                nonlocal captured
                if torch.cuda.is_current_stream_capturing():
                    captured += 1
                executions.add_(1)
                value.fill_(rank + 1)
                dist.all_reduce(value, group=group)

            # Every rank must capture the same number of calls, whatever its own estimate.
            ms = clock.graph_reading(collective)
            assert math.isfinite(ms) and ms > 0
            assert captured == 1
            assert executions.item() == 16  # six eager calls and ten replays
            torch.testing.assert_close(value, torch.full_like(value, 3))
        finally:
            dist.destroy_process_group()


@pytest.mark.gpu
def test_group_clock_captures_one_collective_per_rank(tmp_path: Path) -> None:
    import torch

    if torch.version.hip is None or torch.cuda.device_count() < 2:
        pytest.skip("requires two ROCm GPUs")
    rendezvous = (tmp_path / "store").as_uri()
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        jobs = [pool.apply_async(_time_collective, (rank, rendezvous)) for rank in range(2)]
        for job in jobs:
            job.get(timeout=300)


def test_catalog_leases_devices_in_groups() -> None:
    devices = [f"cuda:{i}" for i in range(5)]
    groups = lambda n: [r.devices for r in LocalAcceleratorCatalog(devices, n).resources()]
    assert groups(1) == [(d,) for d in devices]
    assert groups(2) == [("cuda:0", "cuda:1"), ("cuda:2", "cuda:3")]
    with pytest.raises(ValueError, match="6 devices a resource"):
        LocalAcceleratorCatalog(devices, 6)


def _rank_result(
    axes_text: str, milliseconds: list[dict[str, float]], keys: tuple[str, ...] = ()
) -> CandidateResult:
    rounds = tuple(
        RoundTimes(mode="graph", round_index=i, order=("baseline", "candidate"), milliseconds=m)
        for i, m in enumerate(milliseconds)
    )
    workload = CandidateWorkloadResult(axes_text=axes_text, findings=(), rounds=rounds, output_keys=keys)
    return CandidateResult(findings=(), workload_results=(workload,))


def test_group_round_is_its_slowest_rank() -> None:
    """A group's call ends when its last rank's does: each round's time of a
    participant is the largest over the ranks."""
    rank0 = _rank_result("m=1", [{"baseline": 2.0, "candidate": 1.0}, {"baseline": 2.0, "candidate": 1.5}])
    rank1 = _rank_result("m=1", [{"baseline": 3.0, "candidate": 0.5}, {"baseline": 1.0, "candidate": 1.0}])
    merged = CandidateResult.of_ranks([rank0, rank1])
    assert [r.milliseconds for r in merged.workload_results[0].rounds] == [
        {"baseline": 3.0, "candidate": 1.0},
        {"baseline": 2.0, "candidate": 1.5},
    ]


def test_group_larger_than_the_devices_is_no_matching_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every workload of a schema whose group outnumbers the devices: a located
    NO_MATCHING_DEVICE, nothing run."""
    monkeypatch.setattr(TrialRequest, "n_ranks", property(lambda self: 2))
    monkeypatch.setattr(Executor, "for_devices", lambda *a, **k: pytest.fail("ran a task"))
    scope = replace(ValidationScope(Path.cwd(), "", "HEAD", (), ()), bench_paths=(Path(_SCHEMA),))
    report = BenchStage(["cpu"]).run(scope, None)
    assert report.findings
    assert {(f.code, f.schema_path) for f in report.findings} == {(FindingCode.NO_MATCHING_DEVICE, _SCHEMA)}
    assert all(f.axes is not None for f in report.findings)


class _ScriptedRanks:
    """An executor whose child task of rank ``r`` does what the attempt's ``scripts[r]``
    names: ``"ok"``, ``"finding"``, ``"hang"`` (until killed) or an error's text."""

    def __init__(self, request: TrialRequest, *attempts: list[str]) -> None:
        self.request = request
        self.attempts = iter(attempts)
        self.submits: list[dict[str, Any]] = []
        self.shutdowns = 0

    def for_devices(self, devices: list[str], *, devices_per_task: int, **kwargs: Any) -> nullcontext:
        self.script = next(self.attempts)
        self.hanging: list[Future] = []
        # a task a rank, each leasing the whole group (its tensors are lent on any of its devices)
        n = len(self.script)
        assert devices_per_task == n and devices == devices[:n] * n
        return nullcontext(self)

    def submit(self, fn: Any, **kwargs: Any) -> Future:
        self.submits.append(kwargs)
        action, future = self.script[kwargs.get("rank", 0)], Future()
        if action == "hang":
            self.hanging.append(future)
        elif action in ("ok", "finding", "invalid_timing"):
            result = _rank_result(self.request.axes_text, [{"baseline": 2.0, "candidate": 1.0}])
            if action == "finding":
                result = result.model_copy(
                    update={"findings": (Finding.unlocated(FindingCode.BUILD_FAILED, "no build"),)}
                )
            elif action == "invalid_timing":
                result = _rank_result(self.request.axes_text, [{"baseline": 2.0, "candidate": 0.0}])
            future.set_result(result.model_dump(mode="json"))
        else:
            future.set_exception(TaskExecutionFailed(action))
        return future

    def shutdown(self, wait: bool = True) -> None:
        self.shutdowns += 1
        for future in self.hanging:
            future.set_exception(TaskExecutionFailed("killed"))


def _run_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *attempts: list[str], device: str = "cpu"
) -> tuple[TrialResult, _ScriptedRanks]:
    n_ranks = len(attempts[0])
    monkeypatch.setattr(TrialRequest, "n_ranks", property(lambda self: n_ranks))
    case = OpCase.of("fp8_paged_mqa_logits@1.0")
    contract = OpContract.model_validate(case.contract.model_dump() | {"ranks": "page_size"})
    request = replace(case.request, contract=contract)
    ranks = _ScriptedRanks(request, *attempts)
    evaluated = ReferenceWorkloadResult(
        axes_text=request.axes_text, findings=(), worst_violation_fraction=0.0, is_complete=True
    )
    monkeypatch.setattr(WorkloadEvaluation, "evaluate", lambda self: evaluated)
    context = ExecutionContext(device=device, log_path=tmp_path / "log", n_devices=n_ranks)
    with patch.object(Executor, "for_devices", ranks.for_devices):
        value = run_trial(context, requests=[request], candidate_timeout_seconds=1.0, n_cpu_shares=1)
    return TrialResult.from_task(value), ranks


def test_group_runs_a_child_task_a_rank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every rank its child task, the same rendezvous, its own outputs; the group's
    result merged and evaluated."""
    trial, ranks = _run_group(monkeypatch, tmp_path, ["ok", "ok"])
    assert [s["rank"] for s in ranks.submits] == [0, 1]
    assert len({s["rendezvous"] for s in ranks.submits}) == 1
    assert ranks.submits[0]["outputs"] is not ranks.submits[1]["outputs"]
    assert not trial.candidate.findings and trial.reference is not None


def test_group_ends_at_a_rank_finding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A rank's finding ends the group at once, the rank waiting on it killed."""
    trial, ranks = _run_group(monkeypatch, tmp_path, ["finding", "hang"])
    assert ranks.shutdowns == 1
    assert [(f.code, f.detail) for f in trial.candidate.findings] == [
        (FindingCode.BUILD_FAILED, "rank 0: no build")
    ]
    assert trial.reference is None


def test_group_reruns_whole_at_a_tracer_fault(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A rank's tracer fault reruns the whole group once, at a fresh rendezvous."""
    trial, ranks = _run_group(monkeypatch, tmp_path, ["hang", _TRACER_FAULT], ["ok", "ok"])
    assert len(ranks.submits) == 4
    assert ranks.submits[0]["rendezvous"] != ranks.submits[2]["rendezvous"]
    assert not trial.candidate.findings


def test_a_ranks_contract_of_one_rank_still_runs_as_a_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A contract with ``ranks`` at a group of one still gets its process group and
    ``group_init``: its calls may take them for granted."""
    _, ranks = _run_group(monkeypatch, tmp_path, ["ok"])
    assert [s["rank"] for s in ranks.submits] == [0] and "rendezvous" in ranks.submits[0]


def test_a_group_rank_never_retries_a_tracer_fault_alone() -> None:
    """Alone, a tracer fault is read again; in a group the fault fails the task (the
    bench reruns it whole), else the rank would repeat a call its peers moved past."""
    calls = []

    def flaky(call, rep):
        calls.append(rep)
        if len(calls) == 1:
            raise RuntimeError("rocprofiler-sdk dropped 1 records")
        return 1.0

    assert Clock(lambda: None, flaky).graph_reading(lambda: None) == 1.0
    assert calls == [GRAPH_BUDGET_MILLISECONDS, GRAPH_BUDGET_MILLISECONDS]
    calls.clear()
    group = object()
    with pytest.raises(RuntimeError, match="rocprofiler-sdk"):
        Clock(lambda: None, flaky, group).graph_reading(lambda: None)
    assert calls == [0]


@pytest.mark.parametrize("milliseconds", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_group_keeps_an_unusable_time_from_any_rank(milliseconds: float) -> None:
    """Whichever rank's it is, an unusable time is the group's: compare_performance reports it."""
    rank0 = _rank_result("m=1", [{"baseline": 2.0, "candidate": 1.0}] * 16)
    rank1 = _rank_result("m=1", [{"baseline": 2.0, "candidate": milliseconds}] * 16)
    for results in ([rank0, rank1], [rank1, rank0]):
        merged = CandidateResult.of_ranks(results)
        assert not merged.findings
        kept = {r.milliseconds["candidate"] for r in merged.workload_results[0].rounds}
        assert len(kept) == 1 and not (math.isfinite(kept.pop()) and milliseconds > 0)


def test_group_matches_workloads_by_axes() -> None:
    results = []
    for rank in range(2):
        workloads = tuple(
            _rank_result(f"m={m}", [{"baseline": 2.0 * m, "candidate": (rank + 1.0) * m}]).workload_results[0]
            for m in (1, 100)
        )
        result = CandidateResult(findings=(), workload_results=workloads[:: -1 if rank else 1])
        results.append(CandidateResult.from_task(result.model_dump(mode="json"), ["m=1", "m=100"]))
    merged = CandidateResult.of_ranks(results)
    assert [(w.axes_text, w.rounds[0].milliseconds["candidate"]) for w in merged.workload_results] == [
        ("m=1", 2.0),
        ("m=100", 200.0),
    ]


@pytest.mark.parametrize(
    "changed", [{"round_index": 3}, {"order": ("candidate", "baseline")}, {"milliseconds": {"baseline": 2.0}}]
)
def test_group_rejects_incompatible_rounds(changed: dict[str, Any]) -> None:
    rank0 = _rank_result("m=1", [{"baseline": 2.0, "candidate": 1.0}])
    workload = rank0.workload_results[0]
    rank1 = rank0.model_copy(
        update={
            "workload_results": (
                workload.model_copy(update={"rounds": (workload.rounds[0].model_copy(update=changed),)}),
            )
        }
    )
    merged = CandidateResult.of_ranks([rank0, rank1])
    assert [f.code for f in merged.findings] == [FindingCode.CANDIDATE_CRASH]
    assert merged.findings[0].detail.startswith("rank 1: m=1:")


def test_a_ranks_unusable_time_reaches_the_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A rank's unusable time stays in the group's rounds, for compare_performance to report."""
    trial, _ = _run_group(monkeypatch, tmp_path, ["ok", "invalid_timing"])
    assert not trial.candidate.findings
    assert trial.candidate.workload_results[0].rounds[0].milliseconds["candidate"] == 0.0


@pytest.mark.parametrize("failed", [False, True], ids=["success", "finding"])
def test_rank_synchronizes_outputs_only_when_they_will_be_evaluated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failed: bool
) -> None:
    import torch

    from aklr.core.platform.device import DeviceTarget
    from aklr.core.utils import testing
    from aklr.core.validation.stages._bench._evaluation import _candidate
    from tests.op_case import KERNEL_TARGET

    request = OpCase.of("gemm@1.0").request
    target = DeviceTarget.from_names(**KERNEL_TARGET)
    monkeypatch.setattr(DeviceTarget, "current", lambda device: target)
    monkeypatch.setattr(_candidate._CompileTripwire, "resolve", lambda: None)
    monkeypatch.setattr(_candidate._measure, "bind_clock", lambda group: Clock(lambda: None, None, group))
    monkeypatch.setattr(testing, "flush_l2_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(_candidate.CandidateRun, "run_baseline", lambda *a: None)
    monkeypatch.setattr(_candidate.CandidateRun, "make_warmed_dispatcher", lambda *a: (None, 0))
    finding = _candidate.StepFailed(FindingCode.CAPTURE_UNSAFE, "collective capture failed")
    monkeypatch.setattr(
        _candidate.CandidateRun,
        "check_and_time",
        Mock(side_effect=finding if failed else None, return_value=()),
    )
    synchronize = Mock(side_effect=AssertionError("waiting on an unfinished collective") if failed else None)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    result = _candidate.run_candidate(
        ExecutionContext(device="cuda:0", log_path=tmp_path / "log", n_devices=2),
        requests=[request],
        outputs={request.axes_text: {}},
        rank=1,
        rendezvous=str(tmp_path / "store"),
    )
    candidate = CandidateResult.from_task(result, [request.axes_text])
    assert selected == ["cuda:1"]
    assert synchronize.call_count == (0 if failed else 1)
    assert [f.code for f in candidate.findings] == ([FindingCode.CAPTURE_UNSAFE] if failed else [])
    if failed:
        assert candidate.findings[0].detail.endswith("collective capture failed")


@pytest.mark.parametrize(
    "attempts", [(["ok", "ok"],), (["finding", "hang"],), (["hang", _TRACER_FAULT], ["ok", "ok"])]
)
def test_group_removes_each_rendezvous(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, attempts: tuple[list[str], ...]
) -> None:
    submit = _ScriptedRanks.submit

    def check_directory(self, fn, **kwargs):
        directory = Path(kwargs["rendezvous"]).parent
        assert directory.is_dir() and directory.parent == tmp_path
        return submit(self, fn, **kwargs)

    monkeypatch.setattr(_ScriptedRanks, "submit", check_directory)
    _, ranks = _run_group(monkeypatch, tmp_path, *attempts)
    assert all(not Path(s["rendezvous"]).parent.exists() for s in ranks.submits)


def test_trial_owns_each_output_pool_on_its_device_until_children_finish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    current, allocated, alive = ["cuda:0"], [], WeakSet()

    class Pool:
        def __init__(self):
            self.device = current[0]
            alive.add(self)

    @contextmanager
    def device_context(device):
        previous, current[0] = current[0], device
        try:
            yield
        finally:
            current[0] = previous

    @contextmanager
    def use_pool(pool, device=None):
        assert pool.device == (device or current[0])
        allocated.append(pool.device)
        yield

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "MemPool", Pool)
    monkeypatch.setattr(torch.cuda, "use_mem_pool", use_pool)
    empty_outputs = TrialRequest.empty_outputs
    monkeypatch.setattr(TrialRequest, "empty_outputs", lambda self, device: empty_outputs(self, "cpu"))
    submit = _ScriptedRanks.submit

    def check_pools(self, fn, **kwargs):
        assert {pool.device for pool in alive} == {"cuda:0", "cuda:1"}
        return submit(self, fn, **kwargs)

    monkeypatch.setattr(_ScriptedRanks, "submit", check_pools)
    _run_group(monkeypatch, tmp_path, ["ok", "ok"], device="cuda:0")
    assert allocated == ["cuda:0", "cuda:1"] and current == ["cuda:0"]


def test_group_rejects_different_round_counts() -> None:
    rank0 = _rank_result("m=1", [{"baseline": 2.0, "candidate": 1.0}] * 16)
    rank1 = _rank_result("m=1", [{"baseline": 2.0, "candidate": 1.0}] * 32)
    merged = CandidateResult.of_ranks([rank0, rank1])
    assert [f.code for f in merged.findings] == [FindingCode.CANDIDATE_CRASH]
    assert merged.findings[0].detail == "rank 1: m=1: timing rounds differ across ranks"


def test_group_early_stopping_uses_every_ranks_times(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    group, reductions = object(), []

    def slowest(times, op, *, group):
        assert op == torch.distributed.ReduceOp.MAX
        reductions.append(group)
        other = torch.tensor([[1.02, 0.99 if i % 2 == 0 else 1.01] for i in range(len(times))])
        times.copy_(torch.maximum(times, other.to(times.dtype)))

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", slowest)
    lengths = []
    for collective in (None, group):
        for fraction in (0.5, 1.0):
            jitter = itertools.cycle([1.01, 0.99])
            fns = {
                "baseline": lambda: 1.02,
                "candidate": lambda fraction=fraction, jitter=jitter: fraction * next(jitter),
            }
            clock = Clock(lambda: None, lambda fn, **kwargs: fn(), collective)
            rounds = clock.time_rounds(fns, seed=0, thresholds={"baseline": "entry"}, min_speedup=0)
            lengths.append(len(rounds))
    assert lengths == [16, 32, 32, 32]
    assert reductions == [group, group]
