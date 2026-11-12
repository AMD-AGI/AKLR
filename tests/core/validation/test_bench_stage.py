# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The bench stage through its public API, on the CPU: what it plans each workload against, how it batches the
tasks, and what it decides from their results. The task processes, the only part that needs a GPU, are mocked:
each task answers with the results scripted for its workloads, in the JSON a real task sends back."""

import json
import subprocess
from concurrent.futures import Future
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from aklr.core.platform.device import DeviceTarget
from aklr.core.schema import Build, Evaluation, Kernel, Schema, SourceFile, Speedup, Workload, save_json_file
from aklr.core.validation import PipelineReport, ValidationScope
from aklr.core.validation.pipeline import PlatformRecord
from aklr.core.validation.stages.bench import BenchStage
from tests.core.validation.toy_group import toy_group_schema
from tests.op_case import GEMM_KERNEL, OpCase

_TARGET = DeviceTarget.from_names("gfx950", "MI355X")
_PLATFORM = PlatformRecord("gfx950", "MI355X", 1, "7", "2", "3", "0.3")
_OTHER_KERNEL = GEMM_KERNEL + "# another kernel\n"


def _kernel(source: str = GEMM_KERNEL, arch: str = "gfx950") -> Kernel:
    return Kernel(
        name="gemm",
        author="tests",
        build=Build(language="triton", target_hardware=[arch], entry_point="kernel.py::run"),
        sources=[SourceFile(path="kernel.py", content=source)],
    )


def _schema(
    source: str = GEMM_KERNEL, speedup: float | None = None, n: int = 1, arch: str = "gfx950"
) -> Schema:
    evaluations = []
    if speedup is not None:
        measured = {"aiter": Speedup("e" * 40, speedup)}
        evaluations = [Evaluation(hardware=str(_TARGET), accuracy="passed", speedup=measured)]
    workloads = [Workload(axes={"M": m}, evaluations=evaluations) for m in range(1, n + 1)]
    return OpCase.from_id("gemm_bf16_nt").schema(_kernel(source, arch), workloads)


def _timed(ratio: float, versus: str = "baseline", name: str = "aiter") -> dict:
    """A task's result for a workload timed at ratio against its opponent."""
    comparison = {"versus": versus, "versus_name": name, "candidate_ms": [1.0], "opponent_ms": [ratio]}
    return {"comparison": comparison | {"ratio": ratio, "speedup": ratio}}


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

    def write(self, schema: Schema, uuid: str, name: str | None = None) -> Path:
        op = "toy_group" if schema.op_type == "toy_group" else "gemm"
        path = self.root / f"aklr/ops/{op}/impls/triton/{name or schema.name}/{uuid:0>16}.json"
        save_json_file(schema, path)
        return path

    def commit(self) -> str:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", "c")
        return self._git("rev-parse", "HEAD")

    def _scope(self, base: str | None, paths: tuple[Path, ...]) -> ValidationScope:
        return ValidationScope(self.root, self.commit(), base, paths, paths)

    def plans(self, base: str | None, *paths: Path) -> list:
        return BenchStage(["cuda:0"]).plan(self._scope(base, paths), _TARGET)

    def run(self, tasks: _Tasks, base: str | None, *paths: Path, stage: BenchStage | None = None):
        """The bench stage's report on paths, its tasks answered by tasks."""
        self.monkeypatch.setattr("aklr.core.validation.stages.bench.stage.Executor", tasks)
        scope = self._scope(base, paths)
        earlier = PipelineReport(
            head_sha=scope.head_sha,
            base_sha=base,
            platform=_PLATFORM,
            schema_paths=(),
            bench_paths=(),
            stages=(),
        )
        return (stage or BenchStage(["cuda:0"])).run(scope, earlier)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Repo:
    return _Repo(tmp_path, monkeypatch)


def _only(plans: list):
    assert len(plans) == 1 and len(plans[0].workloads) == 1
    return plans[0].workloads[0]


def _baseline(opponent) -> tuple:
    return opponent.label, opponent.threshold, opponent.backend


def _written(report) -> dict:
    """The Evaluations the report writes back for its only schema, by workload."""
    (evaluations,) = report.evaluations.values()
    return evaluations


def _codes(findings) -> list[str]:
    return [f.code for f in findings]


class TestPlan:
    def test_a_kernel_less_schema_is_not_benched(self, repo: _Repo) -> None:
        path = repo.write(OpCase.from_id("gemm_bf16_nt").schema(), "1")
        assert repo.plans(None, path) == []

    @pytest.mark.parametrize("is_released", [False, True], ids=["first_release", "released"])
    def test_without_an_incumbent_the_baseline_is_timed(self, repo: _Repo, is_released: bool) -> None:
        if is_released:
            repo.write(_schema(), "1", name="another_schema")
        base = repo.commit()
        opponent = _only(repo.plans(base, repo.write(_schema(), "2"))).opponent
        assert _baseline(opponent) == ("baseline", "entry" if is_released else "retain", "aiter")

    def test_without_a_base_the_baseline_is_timed_at_retain(self, repo: _Repo) -> None:
        opponent = _only(repo.plans(None, repo.write(_schema(), "2"))).opponent
        assert _baseline(opponent) == ("baseline", "retain", "aiter")

    def test_a_candidate_that_is_its_own_incumbent_retains_against_the_baseline(self, repo: _Repo) -> None:
        path = repo.write(_schema(speedup=1.5), "1")
        base = repo.commit()
        assert _baseline(_only(repo.plans(base, path)).opponent) == ("baseline", "retain", "aiter")

    def test_another_incumbent_is_timed_at_entry_with_its_stored_speedup(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=2.0), "1")
        base = repo.commit()
        opponent = _only(repo.plans(base, repo.write(_schema(), "2"))).opponent
        assert (opponent.label, opponent.threshold, opponent.stored_speedup, opponent.baseline) == (
            "incumbent",
            "entry",
            2.0,
            "aiter",
        )
        assert opponent.path.endswith("0000000000000001.json")

    def test_the_incumbent_with_the_higher_stored_speedup_is_chosen(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=1.5), "1")
        repo.write(_schema(_OTHER_KERNEL + "# third\n", speedup=2.5), "3")
        base = repo.commit()
        assert _only(repo.plans(base, repo.write(_schema(), "2"))).opponent.stored_speedup == 2.5

    def test_the_own_kernel_is_preferred_to_a_faster_incumbent(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=3.0), "1")
        path = repo.write(_schema(speedup=1.1), "2")
        base = repo.commit()
        assert _only(repo.plans(base, path)).opponent.label == "baseline"

    def test_a_base_schema_that_does_not_qualify_is_no_incumbent(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=1.0), "1")  # below AKLR_MIN_SPEEDUP's 1.05
        base = repo.commit()
        opponent = _only(repo.plans(base, repo.write(_schema(), "2"))).opponent
        assert _baseline(opponent) == ("baseline", "entry", "aiter")

    @pytest.mark.parametrize(("arch", "label"), [("gfx950", "incumbent"), ("gfx942", "nothing")])
    def test_only_a_base_kernel_that_runs_on_the_target_is_an_incumbent(
        self, repo: _Repo, arch: str, label: str
    ) -> None:
        # toy_group has no Baseline: a base kernel needs no measurement on this target to qualify
        workloads = [Workload(axes={"n": 4})]
        repo.write(toy_group_schema(_kernel(_OTHER_KERNEL, arch), workloads), "1")
        base = repo.commit()
        candidate = repo.write(toy_group_schema(_kernel(), workloads), "2")
        assert _only(repo.plans(base, candidate)).opponent.label == label

    def test_a_base_file_that_is_not_json_is_the_platforms(self, repo: _Repo) -> None:
        broken = repo.root / f"aklr/ops/gemm/impls/triton/{_schema().name}/{'1':0>16}.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{not json", encoding="utf-8")
        base = repo.commit()
        (plan,) = repo.plans(base, repo.write(_schema(), "2"))
        assert plan.workloads == ()
        assert _codes(plan.findings) == ["platform_fault"]

    def test_a_base_file_of_another_format_is_ignored(self, repo: _Repo) -> None:
        other = repo.root / f"aklr/ops/gemm/impls/triton/{_schema().name}/{'1':0>16}.json"
        other.parent.mkdir(parents=True)
        other.write_text('{"contract_id": "gemm@1.0"}', encoding="utf-8")
        base = repo.commit()
        (plan,) = repo.plans(base, repo.write(_schema(), "2"))
        assert plan.findings == () and plan.workloads[0].opponent.label == "baseline"

    def test_a_kernel_for_other_hardware_is_the_platforms(self, repo: _Repo) -> None:
        (plan,) = repo.plans(None, repo.write(_schema(arch="gfx942"), "2"))
        assert plan.workloads == ()
        assert _codes(plan.findings) == ["platform_fault"]


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
            repo.write(toy_group_schema(_kernel(), [Workload(axes={"n": 4})], world=2), "2", name="narrow"),
            repo.write(_schema(n=40), "3"),
            repo.write(toy_group_schema(_kernel(), [Workload(axes={"n": 4})], world=8), "4", name="wide"),
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
            tasks, None, repo.write(_schema(), "2"), stage=BenchStage(["cuda:0"], isolate={"gemm_bf16_nt"})
        )
        assert tasks.submitted[0][2]["runner"] == "disposable"

    def test_a_group_op_runs_on_its_policys_ranks_in_fresh_processes(self, repo: _Repo) -> None:
        tasks = _Tasks()
        schema = toy_group_schema(_kernel(), [Workload(axes={"n": 4})], world=4)
        repo.run(tasks, None, repo.write(schema, "2"))
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
        assert report.status == "passed" and report.findings == () and report.demotions == ()
        assert [r.verdict for r in report.records] == ["passed"]
        ((evaluation,),) = [list(e.values()) for e in report.evaluations.values()]
        assert (evaluation.accuracy, evaluation.speedup["aiter"].value) == ("passed", 1.5)

    def test_an_inaccurate_workload_is_demoted_without_a_speedup(self, repo: _Repo) -> None:
        path = repo.write(_schema(n=2), "2")
        report = repo.run(_Tasks({1: "outputs_mismatch", 2: _timed(1.5)}), None, path)
        assert report.findings == ()
        assert _codes(report.demotions) == ["outputs_mismatch"]
        assert report.records[0].is_demoted and report.records[0].verdict == "outputs_mismatch"
        assert _written(report)["M=1"] == Evaluation(hardware="gfx950/MI355X", accuracy="failed")

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
        evaluation = _written(report)["M=1"]
        assert (evaluation.accuracy, evaluation.speedup["aiter"].value) == ("passed", ratio)

    @pytest.mark.parametrize(("speedup", "codes"), [(1.049, ["no_admitted_workload"]), (1.05, [])])
    def test_a_workload_admits_at_a_published_speedup_of_five_percent(
        self, repo: _Repo, speedup: float, codes: list[str]
    ) -> None:
        """Admission's bar is dispatch's: the speedup written back, compared with >= as dispatch compares it."""
        report = repo.run(_Tasks({1: _timed(speedup)}), None, repo.write(_schema(), "2"))
        assert _codes(report.findings) == codes

    def test_passing_retain_without_clearing_entry_admits_nothing(self, repo: _Repo) -> None:
        report = repo.run(_Tasks({1: _timed(1.0)}), None, repo.write(_schema(), "2"))
        assert report.demotions == ()
        assert _codes(report.findings) == ["no_admitted_workload"]
        assert report.evaluations != {}  # its Demotions are still written back

    def test_where_nothing_is_timed_accuracy_admits(self, repo: _Repo) -> None:
        path = repo.write(toy_group_schema(_kernel(), [Workload(axes={"n": 4})]), "2")
        report = repo.run(_Tasks(), None, path)
        assert report.findings == ()
        assert _written(report) == {"n=4": Evaluation(hardware="gfx950/MI355X", accuracy="passed")}

    def test_a_speedup_over_an_incumbent_is_carried_through_its_stored_speedup(self, repo: _Repo) -> None:
        repo.write(_schema(_OTHER_KERNEL, speedup=2.0), "1")
        base = repo.commit()
        path = repo.write(_schema(), "2")
        report = repo.run(_Tasks({1: _timed(1.5, "incumbent", "base.json")}), base, path)
        assert _written(report)["M=1"].speedup["aiter"].value == 3.0

    @pytest.mark.parametrize(
        ("code", "status"), [("platform_fault", "aborted"), ("baseline_unbound", "failed")]
    )
    def test_a_platform_or_schema_finding_decides_the_stage_and_writes_nothing(
        self, repo: _Repo, code: str, status: str
    ) -> None:
        report = repo.run(_Tasks({1: code, 2: _timed(1.5)}), None, repo.write(_schema(n=2), "2"))
        assert _codes(report.findings) == [code]
        assert report.evaluations == {}
        assert report.status == status

    def test_a_task_that_raises_is_the_platforms_for_each_of_its_workloads(self, repo: _Repo) -> None:
        path = repo.write(_schema(n=2), "2")
        report = repo.run(_Tasks(error=RuntimeError("the task raised")), None, path)
        assert _codes(report.findings) == ["platform_fault", "platform_fault"]
        assert report.status == "aborted" and report.evaluations == {}

    def test_a_workload_without_a_result_is_the_platforms(self, repo: _Repo) -> None:
        report = repo.run(_Tasks({1: None}), None, repo.write(_schema(), "2"))
        assert _codes(report.findings) == ["platform_fault"]

    def test_demotions_alone_pass_the_stage(self, repo: _Repo) -> None:
        path = repo.write(_schema(n=2), "2")
        report = repo.run(_Tasks({1: "candidate_raised", 2: _timed(1.5)}), None, path)
        assert report.status == "passed"
        assert _codes(report.demotions) == ["candidate_raised"]
        assert set(_written(report)) == {"M=1", "M=2"}

    def test_the_report_names_each_records_and_demotions_workload(self, repo: _Repo) -> None:
        path = repo.write(_schema(n=2), "2")
        report = repo.run(_Tasks({1: "candidate_raised", 2: _timed(1.5)}), None, path)
        written = TypeAdapter(type(report)).dump_python(report, mode="json")
        assert [r["workload"] for r in written["records"]] == [{"axes": {"M": 1}}, {"axes": {"M": 2}}]
        assert [f["workload"] for f in written["demotions"]] == [{"axes": {"M": 1}}]
        assert all("axes" not in r for r in written["records"] + written["demotions"])

    def test_the_protocol_is_named_by_its_profile(self, repo: _Repo) -> None:
        path = repo.write(_schema(), "2")
        three = repo.run(_Tasks({1: _timed(1.5)}), None, path).protocol
        assert repo.run(_Tasks({1: _timed(1.5)}), None, path).protocol == three
        assert (
            repo.run(
                _Tasks({1: _timed(1.5)}), None, path, stage=BenchStage(["cuda:0"], num_trials=4)
            ).protocol
            != three
        )

    def test_a_report_read_back_shows_the_protocol_it_was_measured_under(self, repo: _Repo) -> None:
        report = repo.run(_Tasks({1: _timed(1.5)}), None, repo.write(_schema(), "2"))
        adapter = TypeAdapter(type(report))
        written = adapter.dump_python(report, mode="json") | {"protocol": "0123456789abcdef"}
        rendered = [line for b in adapter.validate_python(written).markdown_blocks() for line in b.lines()]
        assert any("protocol 0123456789abcdef" in line for line in rendered)
