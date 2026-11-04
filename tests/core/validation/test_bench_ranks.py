# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The bench of an op that ranks call together: devices leased as a group, the
group's timing, its child task a rank, and a group the devices cannot hold."""

from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

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
from aklr.core.validation.stages._measure import Clock, RoundTimes
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from tests.op_case import OpCase

_SCHEMA = "aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/09c4acd65577d824.json"
_TRACER_FAULT = "x: timing: Kineto recorded no kernels for a single call"


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
        elif action in ("ok", "finding"):
            result = _rank_result(self.request.axes_text, [{"baseline": 2.0, "candidate": 1.0}])
            if action == "finding":
                result = result.model_copy(
                    update={"findings": (Finding.unlocated(FindingCode.BUILD_FAILED, "no build"),)}
                )
            future.set_result(result.model_dump(mode="json"))
        else:
            future.set_exception(TaskExecutionFailed(action))
        return future

    def shutdown(self, wait: bool = True) -> None:
        self.shutdowns += 1
        for future in self.hanging:
            future.set_exception(TaskExecutionFailed("killed"))


def _run_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *attempts: list[str]
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
    context = ExecutionContext(device="cpu", log_path=tmp_path / "log", n_devices=n_ranks)
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

    def flaky(call, rep, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("Kineto recorded no kernels for a single call")
        return 1.0

    assert Clock(lambda: None, flaky).graph_reading(lambda: None) == 1.0
    assert calls == [{}, {}]
    calls.clear()
    group = object()
    with pytest.raises(RuntimeError, match="Kineto"):
        Clock(lambda: None, flaky, group).graph_reading(lambda: None)
    assert calls == [{"group": group}]
