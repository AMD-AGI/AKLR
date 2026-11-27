# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.


import json
import subprocess
from concurrent.futures import Future
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from aklr.core.contract.op import OpContract
from aklr.core.contract.spec import WorkloadSpec
from aklr.core.platform import toolchain
from aklr.core.platform.device import DeviceTarget
from aklr.core.validation import ValidationScope
from aklr.core.validation.report import PlatformRecord
from aklr.core.validation.validators.bench import BenchValidator, OpBench
from aklr.core.validation.validators.bench._measure._input_sampler import InputSampler
from tests.op_case import GEMM_KERNEL_SOURCE, OpCase

_TARGET = DeviceTarget.from_names("gfx950", "MI355X")
_PLATFORM = PlatformRecord("gfx950", "MI355X", 1, "7", "2", "3", "0.3")
_OTHER_KERNEL = GEMM_KERNEL_SOURCE + "# another kernel\n"
_HARDWARE = {"gfx950": "MI355X", "gfx942": "MI300X"}
_BASELINE = "aklr.plugins.aiter.provider.gemm.baseline.GemmBaseline"


@pytest.fixture(autouse=True)
def _triton_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The schemas here require triton, which a CPU test machine may lack."""
    monkeypatch.setattr(toolchain, "toolchain_version", lambda name: "3.8.0")


def _kernel(source: str | None = GEMM_KERNEL_SOURCE, arch: str = "gfx950", requires: str = ">=0") -> dict:
    """A triton kernel of one source; None: a definition's, without sources."""
    return {
        "language": "triton",
        "targets": [{"arch": arch, "hardware": _HARDWARE[arch]}],
        "requires": {"triton": requires},
        "entry_point": "kernel.py::run",
        "sources": [] if source is None else [{"path": "kernel.py", "content": source}],
    }


def _schema(
    source: str | None = GEMM_KERNEL_SOURCE,
    speedup: float | None = None,
    n: int = 1,
    arch: str = "gfx950",
    requires: str = ">=0",
    failed: tuple[int, ...] = (),
) -> OpContract:
    """A gemm schema of the workloads m=1..n, each stored passed at speedup (None: unmeasured), those in failed
    stored failed."""
    case = OpCase.of("gemm@1.0")
    workloads = []
    for m in range(1, n + 1):
        point = case.point.model_copy(update={"axes": case.point.axes | {"m": m}})
        workload = case.contract.workload_for(point, {})
        if m in failed:
            workload = workload.model_copy(update={"accuracy": "failed"})
        elif speedup is not None:
            workload = workload.model_copy(update={"accuracy": "passed", "speedup": speedup})
        workloads.append(workload)
    return case.contract.narrow(case.point, workloads=workloads, kernel=_kernel(source, arch, requires))


def _norm_schema(
    epsilon: float, source: str = GEMM_KERNEL_SOURCE, speedup: float | None = None
) -> OpContract:
    """An rmsnorm schema of one workload at the scalar epsilon, stored passed at speedup (None: unmeasured)."""
    case = OpCase.of("rmsnorm@1.0")
    workload = case.contract.workload_for(case.point, {"epsilon": epsilon})
    if speedup is not None:
        workload = workload.model_copy(update={"accuracy": "passed", "speedup": speedup})
    return case.contract.narrow(case.point, workloads=[workload], kernel=_kernel(source))


def _group_schema(ep_size: int) -> OpContract:
    """A mega_moe schema of one workload, on an expert-parallel group of ep_size ranks."""
    case = OpCase.of("mega_moe@1.0")
    point = case.point.model_copy(update={"attrs": case.point.attrs | {"ep_size": ep_size}})
    return case.contract.narrow(point, workloads=[case.contract.workload_for(point, {})], kernel=_kernel())


def _timed(ratio: float) -> dict:
    """A task's result for a workload whose opponent took ratio times its Candidate's time on one trial."""
    return {"candidate_ms": [1.0], "opponent_ms": [ratio]}


class _Tasks:
    """The bench's task processes, mocked: a task answers for each of its workloads with the outcome scripted
    for it, by its axes' first value: a result's fields, a finding's code, or None for no result."""

    def __init__(self, outcomes: dict[int, dict | str | None] | None = None, error: Exception | None = None):
        self.outcomes = outcomes or {}
        self.error = error
        self.submitted: list[
            tuple[str, int, dict]
        ] = []  # each task's schema path, workload count and options

    def for_devices(self, *_args, **_kwargs) -> "_Tasks":
        return self

    def __enter__(self) -> "_Tasks":
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def submit(self, run, **options) -> Future:
        task = run.__self__  # the task submitted: its schema path and workloads are what a real task reads
        self.submitted.append((task.schema_path, len(task.workloads), options))
        future: Future = Future()
        if self.error is not None:
            future.set_exception(self.error)
            return future
        results = []
        for workload in task.workloads:
            axes = workload.key.axes
            outcome = self.outcomes.get(next(iter(axes.values())), {})
            if isinstance(outcome, str):
                finding = {"code": outcome, "schema_path": task.schema_path, "workload": {"axes": axes}}
                outcome = {"findings": [finding | {"detail": "scripted"}]}
            if outcome is not None:
                results.append({"key": {"axes": axes}} | outcome)
        future.set_result(json.dumps(results))
        return future


class _Repo:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.monkeypatch = monkeypatch
        self._git("init", "-q")

    def _git(self, *args: str) -> str:
        argv = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(self.root), *args]
        return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()

    def write(self, schema: OpContract, uuid: str, name: str | None = None) -> Path:
        path = (
            self.root
            / f"aklr/plugins/aiter/provider/{schema.op_type}/impls/triton/{name or schema.name}/{uuid:0>16}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(schema.schema_text(), encoding="utf-8")
        return path

    def commit(self) -> str:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", "c")
        return self._git("rev-parse", "HEAD")

    def _scope(self, base: str | None, paths: tuple[Path, ...]) -> ValidationScope:
        return ValidationScope(self.root, self.commit(), base, paths, paths)

    def plans(self, base: str | None, *paths: Path, validator: BenchValidator | None = None) -> list:
        return (validator or BenchValidator(["cuda:0"])).plan(self._scope(base, paths), _TARGET)

    def run(self, tasks: _Tasks, base: str | None, *paths: Path, validator: BenchValidator | None = None):
        self.monkeypatch.setattr("aklr.core.validation.validators.bench.validator.Executor", tasks)
        return (validator or BenchValidator(["cuda:0"])).run(self._scope(base, paths), _PLATFORM)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Repo:
    return _Repo(tmp_path, monkeypatch)


def _only(plans: list):
    assert len(plans) == 1
    assert len(plans[0].workloads) == 1
    return plans[0].workloads[0]


def _baseline(workload) -> tuple:
    """A planned workload timed against the Baseline: its opponent's kind, its threshold and the Baseline's name."""
    return workload.opponent.label, workload.threshold, workload.opponent.baseline


def _written(report) -> dict:
    """The workloads the report writes back for its only schema, by key."""
    (evaluations,) = report.evaluations.values()
    return evaluations


def _codes(findings) -> list[str]:
    return [f.code for f in findings]


class TestPlan:
    def test_a_kernel_less_schema_is_not_benched(self, repo: _Repo) -> None:
        assert repo.plans(None, repo.write(_schema(source=None), "1")) == []

    @pytest.mark.parametrize("is_released", [False, True], ids=["first_release", "released"])
    def test_without_an_incumbent_the_baseline_is_timed(self, repo: _Repo, is_released: bool) -> None:
        if is_released:
            repo.write(_schema(), "1", name="another_schema")
        base = repo.commit()
        workload = _only(repo.plans(base, repo.write(_schema(), "2")))
        assert _baseline(workload) == ("baseline", "entry" if is_released else "retain", _BASELINE)

    def test_without_a_base_the_baseline_is_timed_at_retain(self, repo: _Repo) -> None:
        workload = _only(repo.plans(None, repo.write(_schema(), "2")))
        assert _baseline(workload) == ("baseline", "retain", _BASELINE)

    def test_a_candidate_that_is_its_own_incumbent_retains_against_the_baseline(self, repo: _Repo) -> None:
        path = repo.write(_schema(speedup=1.5), "1")
        base = repo.commit()
        assert _baseline(_only(repo.plans(base, path))) == ("baseline", "retain", _BASELINE)

    def test_an_incumbent_without_a_stored_speedup_leaves_the_baseline_to_be_timed(
        self, repo: _Repo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a speedup over it could not be carried to one over the Baseline, which a passed workload must have
        monkeypatch.setenv("AKLR_MIN_SPEEDUP", "0")  # an unmeasured base kernel serves
        repo.write(_schema(_OTHER_KERNEL), "1")
        base = repo.commit()
        workload = _only(repo.plans(base, repo.write(_schema(), "2")))
        assert _baseline(workload) == ("baseline", "entry", _BASELINE)

    def test_another_incumbent_is_timed_at_entry_with_its_stored_speedup(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=2.0), "1")
        base = repo.commit()
        workload = _only(repo.plans(base, repo.write(_schema(), "2")))
        opponent = workload.opponent
        assert (opponent.label, workload.threshold, opponent.stored_speedup, opponent.baseline) == (
            "incumbent",
            "entry",
            2.0,
            _BASELINE,
        )
        assert opponent.path.endswith("0000000000000001.json")

    def test_an_incumbent_measured_at_other_scalars_leaves_the_baseline_to_be_timed(
        self, repo: _Repo
    ) -> None:
        # its stored speedup is of another call: carried over, it would write a wrong one against the Baseline
        repo.write(_norm_schema(1e-3, _OTHER_KERNEL, speedup=2.0), "1")
        base = repo.commit()
        workload = _only(repo.plans(base, repo.write(_norm_schema(1e-4), "2")))
        assert workload.opponent.label == "baseline"

    def test_an_incumbent_measured_at_the_same_scalars_is_timed(self, repo: _Repo) -> None:
        repo.write(_norm_schema(1e-3, _OTHER_KERNEL, speedup=2.0), "1")
        base = repo.commit()
        workload = _only(repo.plans(base, repo.write(_norm_schema(1e-3), "2")))
        assert (workload.opponent.label, workload.opponent.stored_speedup) == ("incumbent", 2.0)

    def test_a_schema_that_does_not_parse_is_its_own_finding_and_the_others_are_planned(
        self, repo: _Repo
    ) -> None:
        broken = repo.root / f"aklr/plugins/aiter/provider/gemm/impls/triton/broken/{'1':0>16}.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{not json", encoding="utf-8")
        broken_plan, plan = repo.plans(None, broken, repo.write(_schema(), "2"))
        assert (broken_plan.workloads, _codes(broken_plan.findings)) == ((), ["schema_invalid"])
        assert (len(plan.workloads), plan.findings) == (1, ())

    def test_the_incumbent_with_the_higher_stored_speedup_is_chosen(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=1.5), "1")
        repo.write(_schema(_OTHER_KERNEL + "# third\n", speedup=2.5), "3")
        base = repo.commit()
        assert _only(repo.plans(base, repo.write(_schema(), "2"))).opponent.stored_speedup == 2.5

    def test_among_equal_stored_speedups_the_incumbent_is_the_one_dispatch_serves(self, repo: _Repo) -> None:
        schemas = [_schema(_OTHER_KERNEL + f"# {i}\n", speedup=2.0) for i in "13"]
        for name, schema in zip("13", schemas, strict=True):
            repo.write(schema, name)
        base = repo.commit()
        served = max(schemas, key=lambda s: s.rank_key(s.workloads[0]))
        opponent = _only(repo.plans(base, repo.write(_schema(), "2"))).opponent
        assert opponent.schema.kernel.content_hash == served.kernel.content_hash

    def test_the_own_kernel_is_preferred_to_a_faster_incumbent(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=3.0), "1")
        path = repo.write(_schema(speedup=1.1), "2")
        base = repo.commit()
        assert _only(repo.plans(base, path)).opponent.label == "baseline"

    def test_a_base_schema_that_does_not_qualify_is_no_incumbent(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=1.0), "1")  # below AKLR_MIN_SPEEDUP's 1.05
        base = repo.commit()
        workload = _only(repo.plans(base, repo.write(_schema(), "2")))
        assert _baseline(workload) == ("baseline", "entry", _BASELINE)

    @pytest.mark.parametrize(("arch", "label"), [("gfx950", "incumbent"), ("gfx942", "baseline")])
    def test_only_a_base_kernel_that_runs_on_the_target_is_an_incumbent(
        self, repo: _Repo, arch: str, label: str
    ) -> None:
        repo.write(_schema(_OTHER_KERNEL, arch=arch, speedup=2.0), "1")
        base = repo.commit()
        candidate = repo.write(_schema(), "2")
        assert _only(repo.plans(base, candidate)).opponent.label == label

    def test_a_base_file_that_is_not_json_is_the_platforms(self, repo: _Repo) -> None:
        broken = repo.root / f"aklr/plugins/aiter/provider/gemm/impls/triton/{_schema().name}/{'1':0>16}.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{not json", encoding="utf-8")
        base = repo.commit()
        (plan,) = repo.plans(base, repo.write(_schema(), "2"))
        assert plan.workloads == ()
        assert _codes(plan.findings) == ["platform_fault"]

    def test_a_base_file_of_another_format_is_ignored(self, repo: _Repo) -> None:
        other = repo.root / f"aklr/plugins/aiter/provider/gemm/impls/triton/{_schema().name}/{'1':0>16}.json"
        other.parent.mkdir(parents=True)
        other.write_text('{"contract_id": "gemm@1.0"}', encoding="utf-8")
        base = repo.commit()
        (plan,) = repo.plans(base, repo.write(_schema(), "2"))
        assert plan.findings == ()
        assert plan.workloads[0].opponent.label == "baseline"

    def test_a_kernel_for_other_hardware_is_the_platforms(self, repo: _Repo) -> None:
        (plan,) = repo.plans(None, repo.write(_schema(arch="gfx942"), "2"))
        assert plan.workloads == ()
        assert _codes(plan.findings) == ["platform_fault"]

    def test_a_kernel_whose_requires_this_environment_misses_is_skipped(self, repo: _Repo) -> None:
        (plan,) = repo.plans(None, repo.write(_schema(requires=">=99"), "2"))
        assert plan.workloads == ()
        assert plan.findings == ()
        assert plan.skipped.startswith("triton ")

    def test_untimed_the_workloads_not_recorded_failed_are_checked_against_nothing(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=2.0), "1")
        base = repo.commit()
        path = repo.write(_schema(n=2, failed=(1,)), "2")
        (plan,) = repo.plans(base, path, validator=BenchValidator(["cuda:0"], timed=False))
        assert [(str(w.key), w.opponent.label) for w in plan.workloads] == [("m=2", "nothing")]


class TestTasks:
    def test_a_long_schema_is_split_into_even_tasks_of_at_most_sixteen_workloads(self, repo: _Repo) -> None:
        tasks = _Tasks()
        repo.run(tasks, None, repo.write(_schema(n=40), "2"))
        assert sorted(count for _, count, _ in tasks.submitted) == [13, 13, 14]

    def test_the_costliest_tasks_start_first(self, repo: _Repo) -> None:
        small = repo.write(_schema(n=2), "1", name="small_schema")
        large = repo.write(_schema(n=40), "2")
        tasks = _Tasks()
        repo.run(tasks, None, small, large)
        assert [count for _, count, _ in tasks.submitted] == [14, 13, 13, 2]

    def test_the_widest_tasks_start_first_and_keep_the_cost_order(self, repo: _Repo) -> None:
        paths = (
            repo.write(_schema(n=2), "1", name="small_schema"),
            repo.write(_group_schema(2), "2", name="narrow"),
            repo.write(_schema(n=40), "3"),
            repo.write(_group_schema(8), "4", name="wide"),
        )
        tasks = _Tasks()
        repo.run(tasks, None, *paths)
        widths = [(options["resource_request"].count, count) for _, count, options in tasks.submitted]
        assert widths == [(8, 1), (2, 1), (1, 14), (1, 13), (1, 13), (1, 2)]

    def test_an_op_runs_on_one_rank_in_a_reused_process(self, repo: _Repo) -> None:
        tasks = _Tasks()
        repo.run(tasks, None, repo.write(_schema(), "2"))
        ((_, _, options),) = tasks.submitted
        assert (options["resource_request"].count, options["runner"]) == (1, "reuse")

    def test_an_isolated_op_type_runs_in_a_fresh_process(self, repo: _Repo) -> None:
        tasks = _Tasks()
        repo.run(
            tasks, None, repo.write(_schema(), "2"), validator=BenchValidator(["cuda:0"], isolate={"gemm"})
        )
        assert tasks.submitted[0][2]["runner"] == "disposable"

    def test_a_group_op_runs_on_its_benchs_ranks_in_fresh_processes(self, repo: _Repo) -> None:
        tasks = _Tasks()
        repo.run(tasks, None, repo.write(_group_schema(4), "2"))
        ((_, _, options),) = tasks.submitted
        assert (options["resource_request"].count, options["runner"]) == (4, "disposable")

    def test_a_task_has_its_build_and_per_workload_budgets(self, repo: _Repo) -> None:
        tasks = _Tasks()
        repo.run(tasks, None, repo.write(_schema(n=3), "2"))
        assert tasks.submitted[0][2]["timeout_secs"] == 900 + 3 * 300


class TestVerdict:
    def test_a_faster_candidate_is_admitted_with_its_speedup_over_the_baseline(self, repo: _Repo) -> None:
        path = repo.write(_schema(), "2")
        report = repo.run(_Tasks({1: _timed(1.5)}), None, path)
        assert report.status == "passed"
        assert report.findings == ()
        assert report.demotions == ()
        assert [r.verdict for r in report.records] == ["passed"]
        ((evaluation,),) = [list(e.values()) for e in report.evaluations.values()]
        assert (evaluation.accuracy, evaluation.speedup) == ("passed", 1.5)

    def test_an_inaccurate_workload_is_demoted_without_a_speedup(self, repo: _Repo) -> None:
        path = repo.write(_schema(n=2), "2")
        report = repo.run(_Tasks({1: "numeric_mismatch", 2: _timed(1.5)}), None, path)
        assert report.findings == ()
        assert _codes(report.demotions) == ["numeric_mismatch"]
        assert report.records[0].is_demoted
        assert report.records[0].verdict == "numeric_mismatch"
        assert _written(report)["m=1"] == WorkloadSpec(axes={"m": 1}, accuracy="failed")

    @pytest.mark.parametrize(
        ("is_released", "ratio", "code"),
        [
            pytest.param(False, 0.98, "regressed", id="retain"),
            pytest.param(True, 1.05, "not_faster", id="entry"),
        ],
    )
    def test_a_ratio_at_or_below_its_threshold_is_a_performance_demotion(
        self, repo: _Repo, is_released: bool, ratio: float, code: str
    ) -> None:
        if is_released:
            repo.write(_schema(), "1", name="another_schema")
        base = repo.commit() if is_released else None
        path = repo.write(_schema(n=2), "2")
        report = repo.run(_Tasks({1: _timed(ratio), 2: _timed(1.5)}), base, path)
        assert _codes(report.demotions) == [code]
        evaluation = _written(report)["m=1"]
        assert (evaluation.accuracy, evaluation.speedup) == ("passed", ratio)

    @pytest.mark.parametrize(("speedup", "codes"), [(1.049, ["no_qualifying_workload"]), (1.05, [])])
    def test_a_workload_admits_at_a_published_speedup_of_five_percent(
        self, repo: _Repo, speedup: float, codes: list[str]
    ) -> None:
        """Admission's bar is dispatch's: the speedup written back, compared with >= as dispatch compares it."""
        report = repo.run(_Tasks({1: _timed(speedup)}), None, repo.write(_schema(), "2"))
        assert _codes(report.findings) == codes

    def test_passing_retain_without_clearing_entry_admits_nothing(self, repo: _Repo) -> None:
        report = repo.run(_Tasks({1: _timed(1.0)}), None, repo.write(_schema(), "2"))
        assert report.demotions == ()
        assert _codes(report.findings) == ["no_qualifying_workload"]
        assert report.evaluations != {}  # its Demotions are still written back

    def test_a_speedup_over_an_incumbent_is_carried_through_its_stored_speedup(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=2.0), "1")
        base = repo.commit()
        path = repo.write(_schema(), "2")
        report = repo.run(_Tasks({1: _timed(1.5)}), base, path)
        assert _written(report)["m=1"].speedup == 3.0

    @pytest.mark.parametrize(
        ("code", "status"), [("platform_fault", "aborted"), ("baseline_unbound", "failed")]
    )
    def test_a_platform_or_schema_finding_decides_the_validator_and_writes_nothing(
        self, repo: _Repo, code: str, status: str
    ) -> None:
        report = repo.run(_Tasks({1: code, 2: _timed(1.5)}), None, repo.write(_schema(n=2), "2"))
        assert _codes(report.findings) == [code]
        assert report.evaluations == {}
        assert report.status == status

    def test_a_baseline_timed_workload_writes_back_its_kernels_median_time(self, repo: _Repo) -> None:
        timings = {"candidate_ms": [0.0123456, 0.02, 0.01], "opponent_ms": [0.03, 0.04, 0.02]}
        report = repo.run(_Tasks({1: timings}), None, repo.write(_schema(), "2"))
        assert _written(report)["m=1"].microseconds == 12.3

    def test_a_workload_timed_against_an_incumbent_writes_back_no_time(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=2.0), "1")
        base = repo.commit()
        report = repo.run(_Tasks({1: _timed(1.5)}), base, repo.write(_schema(), "2"))
        assert _written(report)["m=1"].microseconds is None

    def test_a_skipped_schema_is_listed_and_decides_nothing(self, repo: _Repo) -> None:
        report = repo.run(_Tasks(), None, repo.write(_schema(requires=">=99"), "2"))
        assert (report.status, report.findings, report.records, report.evaluations) == ("passed", (), (), {})
        ((path, why),) = report.skipped_schemas.items()
        assert path.endswith("0000000000000002.json")
        assert why.startswith("triton ")
        assert "1 schemas skipped: kernel.requires" in report.console_lines()

    @pytest.mark.parametrize(
        ("outcome", "codes", "status"),
        [pytest.param({}, [], "passed", id="accurate"), ("numeric_mismatch", ["numeric_mismatch"], "failed")],
    )
    def test_untimed_a_wrong_output_fails_the_validator_and_nothing_is_written_back(
        self, repo: _Repo, outcome: dict | str, codes: list[str], status: str
    ) -> None:
        validator = BenchValidator(["cuda:0"], timed=False)
        report = repo.run(_Tasks({1: outcome}), None, repo.write(_schema(), "2"), validator=validator)
        assert (_codes(report.findings), report.status) == (codes, status)
        assert (report.demotions, report.evaluations) == ((), {})

    def test_a_task_that_raises_is_the_platforms_for_each_of_its_workloads(self, repo: _Repo) -> None:
        path = repo.write(_schema(n=2), "2")
        report = repo.run(_Tasks(error=RuntimeError("the task raised")), None, path)
        assert _codes(report.findings) == ["platform_fault", "platform_fault"]
        assert report.status == "aborted"
        assert report.evaluations == {}

    def test_an_opposed_workload_that_comes_back_untimed_is_the_platforms(self, repo: _Repo) -> None:
        report = repo.run(_Tasks({1: {}}), None, repo.write(_schema(), "2"))
        assert _codes(report.findings) == ["platform_fault"]
        assert report.evaluations == {}

    def test_an_unopposed_workload_that_comes_back_timed_is_the_platforms(self, repo: _Repo) -> None:
        validator = BenchValidator(["cuda:0"], timed=False)
        report = repo.run(_Tasks({1: _timed(1.5)}), None, repo.write(_schema(), "2"), validator=validator)
        assert _codes(report.findings) == ["platform_fault"]
        assert report.evaluations == {}

    def test_a_record_names_what_its_workload_was_timed_against(self, repo: _Repo) -> None:
        report = repo.run(_Tasks({1: _timed(1.5)}), None, repo.write(_schema(), "2"))
        (record,) = report.records
        assert (record.versus, record.versus_name) == ("baseline", _BASELINE)
        assert (record.comparison.ratio, record.comparison.speedup) == (pytest.approx(1.5), 1.5)

    def test_the_report_records_the_thresholds_it_judged_by(self, repo: _Repo) -> None:
        report = repo.run(_Tasks({1: _timed(1.5)}), None, repo.write(_schema(), "2"))
        assert report.bars == {"entry": 1.05, "retain": 0.99}

    def test_a_workload_without_a_result_is_the_platforms(self, repo: _Repo) -> None:
        report = repo.run(_Tasks({1: None}), None, repo.write(_schema(), "2"))
        assert _codes(report.findings) == ["platform_fault"]

    def test_demotions_alone_pass_the_validator(self, repo: _Repo) -> None:
        path = repo.write(_schema(n=2), "2")
        report = repo.run(_Tasks({1: "candidate_raised", 2: _timed(1.5)}), None, path)
        assert report.status == "passed"
        assert _codes(report.demotions) == ["candidate_raised"]
        assert set(_written(report)) == {"m=1", "m=2"}

    def test_the_report_names_each_records_and_demotions_workload(self, repo: _Repo) -> None:
        path = repo.write(_schema(n=2), "2")
        report = repo.run(_Tasks({1: "candidate_raised", 2: _timed(1.5)}), None, path)
        written = TypeAdapter(type(report)).dump_python(report, mode="json")
        assert [r["workload"] for r in written["records"]] == [{"axes": {"m": 1}}, {"axes": {"m": 2}}]
        assert [f["workload"] for f in written["demotions"]] == [{"axes": {"m": 1}}]
        assert all("axes" not in r for r in written["records"] + written["demotions"])

    def test_the_protocol_is_named_by_its_profile(self, repo: _Repo) -> None:
        path = repo.write(_schema(), "2")
        three = repo.run(_Tasks({1: _timed(1.5)}), None, path).protocol
        assert repo.run(_Tasks({1: _timed(1.5)}), None, path).protocol == three
        assert (
            repo.run(
                _Tasks({1: _timed(1.5)}), None, path, validator=BenchValidator(["cuda:0"], num_trials=4)
            ).protocol
            != three
        )

    @pytest.mark.parametrize("num_trials", [0, -1])
    def test_a_bench_of_no_trials_is_refused(self, num_trials: int) -> None:
        with pytest.raises(ValueError, match="num_trials"):
            BenchValidator(["cuda:0"], num_trials=num_trials)

    def test_a_schema_that_does_not_parse_fails_the_bench_and_the_others_are_measured(
        self, repo: _Repo
    ) -> None:
        broken = repo.root / f"aklr/plugins/aiter/provider/gemm/impls/triton/broken/{'1':0>16}.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{not json", encoding="utf-8")
        report = repo.run(_Tasks({1: _timed(1.5)}), None, broken, repo.write(_schema(), "2"))
        assert (report.status, _codes(report.findings)) == ("failed", ["schema_invalid"])
        assert [r.verdict for r in report.records] == ["passed"]
        assert len(report.evaluations) == 1

    def test_a_report_read_back_shows_the_protocol_it_was_measured_under(self, repo: _Repo) -> None:
        report = repo.run(_Tasks({1: _timed(1.5)}), None, repo.write(_schema(), "2"))
        adapter = TypeAdapter(type(report))
        written = adapter.dump_python(report, mode="json") | {"protocol": "0123456789abcdef"}
        rendered = [line for b in adapter.validate_python(written).markdown_blocks() for line in b.lines()]
        assert any("protocol 0123456789abcdef" in line for line in rendered)


@pytest.mark.parametrize("ep_size", [1, 2])
def test_a_mega_moe_task_runs_on_ep_size_ranks_in_fresh_processes(ep_size: int) -> None:
    schema = _group_schema(ep_size)
    bench = OpBench.for_op(schema.op_type)
    assert (bench.ranks(schema), bench.isolate(schema)) == (ep_size, True)


def test_a_trial_draws_its_workloads_own_scalars() -> None:
    case = OpCase.of("rmsnorm@1.0")
    workload = case.contract.workload_for(case.point, {"epsilon": 1e-3})
    schema = case.contract.narrow(case.point, workloads=[workload], kernel=_kernel())
    inputs = InputSampler(schema, workload.axes).draw(0, 0, "cpu")
    assert inputs["epsilon"] == workload.scalars["epsilon"] != 1e-6
