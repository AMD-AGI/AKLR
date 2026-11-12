# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The driver's output is read by KernelForge's own parsers, so assert with those: a line they skip
loses its case silently."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fake_aklr import CASES, SCHEMA_NAME, install_aklr, schema_payload, write_schema
from forge_task import workspace
from forge_task.schema import SchemaError, find, load

STARTING_KERNEL = '"""Start."""\n\nimport torch\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    torch.matmul(a, b.t(), out=out)\n'
CANDIDATE_KERNEL = STARTING_KERNEL + "\n# edited: torch.matmul still passes\n"
FAILING_KERNEL = '"""Wrong."""\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    pass\n'


# A kernel of two modules: kernel.py imports the operation from the file beside it.
SPLIT_KERNEL = {
    "kernel.py": '"""Start."""\n\nfrom .gemm import gemm\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    gemm(a, b, out)\n',
    "gemm.py": '"""The product."""\n\nimport torch\n\n\ndef gemm(a, b, out):\n    torch.matmul(a, b.t(), out=out)\n',
}


def _materialize(aklr_root: Path, tmp_path: Path, kernel: dict | None = None) -> workspace.Run:
    return workspace.materialize(
        load(find(aklr_root, SCHEMA_NAME)),
        tmp_path / "run",
        aklr_root=aklr_root,
        language="flydsl",
        starting_kernel=kernel or {"kernel.py": STARTING_KERNEL},
        anchor="baseline",
    )


@pytest.fixture
def run(aklr_root: Path, tmp_path: Path):
    return _materialize(aklr_root, tmp_path)


def _run_driver(run, *args: str, expect_rc: int = 0) -> str:
    completed = subprocess.run(
        [sys.executable, str(run.driver), *args],
        cwd=run.workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == expect_rc, completed.stdout + completed.stderr
    return completed.stdout


def _import_driver(run):
    spec = importlib.util.spec_from_file_location("aklr_forge_driver", run.driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_calls(aklr_root: Path) -> int:
    calls = aklr_root / "validate_calls"
    return int(calls.read_text()) if calls.exists() else 0


@pytest.fixture
def real_run(tmp_path: Path, monkeypatch):
    """A task on a kernel-less schema of this checkout, whose driver imports this checkout's AKLR."""
    pytest.importorskip("torch")
    monkeypatch.setattr(sys, "path", list(sys.path))
    root = Path(__file__).resolve().parents[3]
    try:
        schema = load(find(root, "bmm_n128_k512"))
    except SchemaError:
        pytest.skip("this checkout holds no kernel-less bmm_n128_k512")
    run = workspace.materialize(
        schema,
        tmp_path / "run",
        aklr_root=root,
        language="flydsl",
        starting_kernel=workspace.torch_starting_kernel(schema, root),
        anchor="baseline",
    )
    return run, schema


def test_the_driver_builds_validates_calls_with_this_checkouts_aklr(real_run):
    """The driver times with AKLR's private timer and trial code, which a change in AKLR breaks in the GPU modes
    alone: here this checkout's AKLR builds every timed call on the CPU, and the starting kernel runs on each.
    """
    run, schema = real_run
    driver = _import_driver(run)
    budget, bind_clock, is_tracer_fault = driver._aklr_timer()
    assert budget > 0 and callable(bind_clock) and callable(is_tracer_fault)
    calls = driver._calls("cpu")
    assert set(calls) == {workload.case_id for workload in schema.workloads}
    registry, *_ = driver._aklr()
    parsed = registry.parse_schema(driver._candidate_text()[0].encode(), "schema")
    candidate = driver._load_candidate()
    for call in calls.values():
        assert set(call) <= set(parsed.kernel_call_keys)
        candidate.run(**call)


def test_the_driver_reads_aklrs_own_report_of_a_wrong_workload(real_run):
    """The report a stand-in writes can drift from AKLR's: this one is written by this checkout's own models, where
    a workload computed wrong is demoted with the fraction it compared, and the schema's others still pass.
    """
    run, schema = real_run
    driver = _import_driver(run)
    driver._aklr()
    from aklr.core.validation.stages._bench._report import BenchReport, WorkloadRecord
    from aklr.core.validation.stages._report._finding import Finding, FindingCode

    wrong = schema.workloads[0]
    records = tuple(
        WorkloadRecord(
            schema_path="candidate.json",
            axes=workload.axes,
            kernel_content_hash="",
            baseline=None,
            is_timed=False,
            worst_violation_fraction=0.5 if workload is wrong else 0.0,
            timings=(),
            batch_stopped=False,
        )
        for workload in schema.workloads
    )
    mismatch = Finding(
        code=FindingCode.NUMERIC_MISMATCH,
        schema_path="candidate.json",
        axes=wrong.axes,
        detail="numeric/0: out",
    )
    bench = BenchReport.from_findings([mismatch], workload_records=records)
    outcomes = driver._outcomes({"stages": [bench.model_dump(mode="json")]})
    assert outcomes[wrong.case_id]["incorrect"] == ["numeric_mismatch"]
    assert all(not outcomes[w.case_id]["incorrect"] for w in schema.workloads[1:])


def test_bench_output_parses_into_every_case(run, monkeypatch, capsys):
    """KernelForge's `parse_case_timings` must find every scored workload, none unscored."""
    runner = pytest.importorskip("kernelforge.evaluation.benchmark.runner")
    driver = _import_driver(run)
    monkeypatch.setattr(driver, "_measure_candidate", lambda warmup, rounds: {"m=8": 0.01, "m=16": 0.02})
    assert driver._run_bench(1, 2) == 0
    stdout = capsys.readouterr().out
    case_times, unscored, duplicates = runner.parse_case_timings(stdout)
    assert set(case_times) == set(CASES)
    assert not unscored and not duplicates
    assert re.search(r"^median_ms: 0\.030000$", stdout, re.MULTILINE)


def test_bench_times_the_candidate_without_validating_it(run, aklr_root: Path, monkeypatch):
    """A validate run times the baseline too, past KernelForge's limit on one benchmark."""
    driver = _import_driver(run)
    monkeypatch.setattr(driver, "_measure_candidate", lambda warmup, rounds: {"m=8": 0.01, "m=16": 0.02})
    driver._run_bench(1, 2)
    assert _validate_calls(aklr_root) == 0


def test_correctness_is_reported_as_allclose(run):
    """AKLR grades a violation fraction, not an SNR; KernelForge reads allclose instead."""
    stdout = _run_driver(run)
    assert "allclose: True" in stdout
    assert "SNR:" not in stdout


def test_a_workload_aklr_finds_wrong_fails_the_candidate(run):
    run.kernel.write_text(FAILING_KERNEL, encoding="utf-8")
    stdout = _run_driver(run)
    for case_id in CASES:
        assert f"aklr_findings: {case_id} numeric_mismatch" in stdout
    assert "allclose: False" in stdout


def test_the_baseline_is_what_kernelforge_reads_as_its_anchor(run, tmp_path: Path):
    path = tmp_path / "baseline.json"
    _run_driver(run, "--baseline-json", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["case_times"]) == set(CASES)
    assert payload["wall_ms"] == pytest.approx(sum(payload["case_times"].values()))
    loop = pytest.importorskip("kernelforge.cli.forge_loop_command")
    assert loop._load_external_baseline(str(path))[1] == payload["case_times"]
    assert loop._load_anchor_presentation(str(path)) == ("AKLR baseline", 1.05)


def test_without_a_serving_bar_the_anchor_marks_none(tmp_path: Path):
    """AKLR_MIN_SPEEDUP at or below zero serves every workload; KernelForge refuses a bar that is not positive."""
    aklr_root = install_aklr(tmp_path / "aklr_checkout", min_speedup=0.0)
    write_schema(aklr_root, schema_payload())
    path = tmp_path / "baseline.json"
    _run_driver(_materialize(aklr_root, tmp_path), "--baseline-json", str(path))
    assert "serving_speedup" not in json.loads(path.read_text(encoding="utf-8"))


def test_one_validate_run_serves_correctness_and_the_baseline_of_a_kernel(
    run, aklr_root: Path, tmp_path: Path
):
    _run_driver(run)
    _run_driver(run, "--baseline-json", str(tmp_path / "baseline.json"))
    assert _validate_calls(aklr_root) == 1
    run.kernel.write_text(FAILING_KERNEL, encoding="utf-8")
    _run_driver(run)
    assert _validate_calls(aklr_root) == 2


def test_a_schema_aklr_times_no_baseline_for_has_no_anchor(tmp_path: Path):
    aklr_root = install_aklr(tmp_path / "aklr_checkout", timed_baseline=False)
    write_schema(aklr_root, schema_payload())
    run = _materialize(aklr_root, tmp_path)
    _run_driver(run, "--baseline-json", str(tmp_path / "b.json"), expect_rc=2)
    assert not (tmp_path / "b.json").exists()


def test_the_published_candidate_leaves_the_checkout(run, aklr_root: Path):
    """A candidate file left in the AKLR checkout would be served beside the next one."""
    _run_driver(run)
    published = [
        path
        for path in aklr_root.glob("aklr/ops/*/impls/*/*/*.json")
        if json.loads(path.read_text())["kernel"]["sources"]
    ]
    assert published == []


def test_publish_keeps_the_kernel_in_the_checkout_to_commit(run, aklr_root: Path, capsys):
    from forge_task import cli

    run.kernel.write_text(CANDIDATE_KERNEL, encoding="utf-8")
    assert cli.main(["publish", str(run.root)]) == 0
    out = capsys.readouterr().out
    published = aklr_root / out.split("published: ")[1].splitlines()[0]
    sources = json.loads(published.read_text())["kernel"]["sources"]
    assert sources == [{"path": "kernel.py", "content": CANDIDATE_KERNEL}]
    assert "python -m aklr export check" in out


def test_finding_kinds_split_into_wrong_and_slow(run):
    """A speed finding grades a workload AKLR still serves the baseline for; only a correctness one fails it."""
    driver = _import_driver(run)
    fast, slow = CASES
    report = {
        "stages": [
            {"stage_name": "audit", "findings": []},
            {
                "stage_name": "bench",
                "findings": [],
                "demotions": [
                    {"code": "not_faster", "axes": {"m": 8}},
                    {"code": "nondeterministic", "axes": {"m": 16}},
                ],
                "workload_records": [
                    {"axes": {"m": m}, "worst_violation_fraction": 0.0, "timings": []} for m in (8, 16)
                ],
            },
        ]
    }
    outcomes = driver._outcomes(report)
    assert outcomes[fast]["incorrect"] == []
    assert outcomes[slow]["incorrect"] == ["nondeterministic"]


def test_a_workload_aklr_did_not_compare_is_not_correct(run):
    """A tracer fault aborts the run with no wrong-output finding; nothing was compared, so nothing passes."""
    driver = _import_driver(run)
    report = {
        "stages": [
            {"stage_name": "audit", "findings": []},
            {
                "stage_name": "bench",
                "findings": [{"code": "platform_failure", "axes": {"m": m}} for m in (8, 16)],
                "workload_records": [
                    {"axes": {"m": m}, "worst_violation_fraction": None, "timings": []} for m in (8, 16)
                ],
            },
        ]
    }
    assert all(outcome["incorrect"] == ["not_compared"] for outcome in driver._outcomes(report).values())


def test_a_platform_failure_is_validated_once_more(tmp_path: Path):
    aklr_root = install_aklr(tmp_path / "aklr_checkout", platform_failures=1)
    write_schema(aklr_root, schema_payload())
    run = _materialize(aklr_root, tmp_path)
    assert "allclose: True" in _run_driver(run)
    _run_driver(run)
    assert _validate_calls(aklr_root) == 2


def test_a_platform_failure_twice_fails_every_workload(tmp_path: Path):
    aklr_root = install_aklr(tmp_path / "aklr_checkout", platform_failures=2)
    write_schema(aklr_root, schema_payload())
    stdout = _run_driver(_materialize(aklr_root, tmp_path))
    assert all(f"aklr_findings: {case_id} not_compared" in stdout for case_id in CASES)
    assert "allclose: False" in stdout


def test_an_audit_finding_rejects_every_workload(run):
    driver = _import_driver(run)
    report = {
        "stages": [
            {
                "stage_name": "audit",
                "findings": [{"code": "source_rejected", "axes": None}],
            }
        ]
    }
    assert all(outcome["incorrect"] == ["source_rejected"] for outcome in driver._outcomes(report).values())


def test_a_kernel_of_several_files_is_published_and_loaded_whole(aklr_root: Path, tmp_path: Path):
    """kernel.py and the modules it imports relatively are the kernel; a probe beside them is not."""
    run = _materialize(aklr_root, tmp_path, SPLIT_KERNEL)
    (run.workspace / "probe.py").write_text("raise SystemExit\n", encoding="utf-8")
    driver = _import_driver(run)
    assert list(driver.kernel_sources()) == ["gemm.py", "kernel.py"]
    published = json.loads(driver._candidate_text()[0])["kernel"]["sources"]
    assert [source["path"] for source in published] == ["gemm.py", "kernel.py"]
    assert callable(driver._load_candidate().run)
    assert "allclose: True" in _run_driver(run)


def test_an_edit_to_any_of_the_kernels_files_is_validated_anew(aklr_root: Path, tmp_path: Path):
    run = _materialize(aklr_root, tmp_path, SPLIT_KERNEL)
    _run_driver(run)
    (run.workspace / "probe.py").write_text("# not the kernel\n", encoding="utf-8")
    _run_driver(run)
    assert _validate_calls(aklr_root) == 1
    gemm = run.workspace / "gemm.py"
    gemm.write_text(gemm.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    _run_driver(run)
    assert _validate_calls(aklr_root) == 2


def test_driver_is_importable_without_a_gpu(run):
    """KernelForge hashes the driver before it ever runs; importing must not need ROCm or AKLR."""
    assert _import_driver(run).SCHEMA_JSON == run.schema_json


def test_a_reading_the_profiler_dropped_is_taken_again_once(run):
    """A trace that lost replays says nothing about the kernel; a second loss, or any other error, still fails."""
    driver = _import_driver(run)
    dropped = RuntimeError("rocprofiler-sdk recorded 9 of 10 graph replays")

    def bench_failing(*failures):
        queue = list(failures)

        def bench(call, rep):
            if queue:
                raise queue.pop(0)
            return 0.01

        return bench

    def is_fault(text):
        return "graph replays" in text

    assert driver.graph_reading(bench_failing(dropped), None, 20.0, is_fault) == 0.01
    with pytest.raises(RuntimeError, match="9 of 10"):
        driver.graph_reading(bench_failing(dropped, dropped), None, 20.0, is_fault)
    with pytest.raises(RuntimeError, match="HIP error"):
        driver.graph_reading(
            bench_failing(RuntimeError("HIP error: illegal address")),
            None,
            20.0,
            is_fault,
        )


def test_stress_passes_one_answer_and_names_the_workload_that_differs(real_run, monkeypatch, capsys):
    """--stress compares the outputs of every call under contention with the candidate's own result alone, bit for
    bit: here on the CPU, with stand-in conditions in place of the CU-limited and concurrent streams."""
    run, schema = real_run
    driver = _import_driver(run)
    calls = driver._calls("cpu")
    monkeypatch.setattr(driver, "_calls", lambda device: calls)

    def again(candidate, call, outputs):
        candidate.run(**call)
        return [call]

    def drifting(candidate, call, outputs):
        candidate.run(**call)
        for name in outputs:
            call[name].add_(1)
        return [call]

    monkeypatch.setattr(driver, "_stress_conditions", lambda: {"again": again})
    assert driver._run_stress("cpu") == 0
    assert capsys.readouterr().out.strip().endswith("stress: PASS")
    monkeypatch.setattr(driver, "_stress_conditions", lambda: {"drifting": drifting})
    assert driver._run_stress("cpu") == 1
    out = capsys.readouterr().out
    assert f"differs from the call alone: {schema.workloads[0].case_id} drifting" in out
    assert out.strip().endswith("stress: FAIL")


def test_stress_compares_as_the_schemas_determinism_asks(run):
    """Bitwise wants every bit; bounded lets atomics reorder the last bits but not a new NaN or a wrong value."""
    torch = pytest.importorskip("torch")
    driver = _import_driver(run)
    alone = torch.linspace(1, 64, 4096, dtype=torch.bfloat16)
    one_ulp = alone.clone()
    one_ulp.view(torch.int16)[::50] += 1  # the next bf16 value up: one ulp
    assert not driver._agrees(one_ulp, alone, "bitwise", 0.0)
    assert driver._agrees(one_ulp, alone, "bounded", 0.001)
    nan = one_ulp.clone()
    nan[7] = float("nan")
    assert not driver._agrees(nan, alone, "bounded", 0.001)
    wrong = alone.clone()
    wrong[:100] += 8
    assert not driver._agrees(wrong, alone, "bounded", 0.001)
    ids = torch.arange(16)
    assert driver._agrees(ids, ids.clone(), "bounded", 0.001)


def test_stress_reports_a_call_that_never_finishes_as_hung(real_run, monkeypatch, capsys):
    """A kernel whose workgroups wait for each other may never finish under stress: the check names the case and fails
    instead of waiting with it."""
    run, schema = real_run
    driver = _import_driver(run)
    calls = driver._calls("cpu")
    monkeypatch.setattr(driver, "_calls", lambda device: calls)

    def hangs(candidate, call, outputs):
        raise driver.StressHang

    monkeypatch.setattr(driver, "_stress_conditions", lambda: {"on 16 CUs": hangs})
    assert driver._run_stress("cpu") == 1
    out = capsys.readouterr().out
    assert f"hung, no result within 60 s: {schema.workloads[0].case_id} on 16 CUs" in out
    assert out.strip().endswith("stress: FAIL")
