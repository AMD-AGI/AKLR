# SPDX-License-Identifier: MIT
"""The kernels section of the Pages site: benchmarks/kernels/kernels_report."""

from __future__ import annotations

import json

import pytest
from kernels_report.data_collect import (
    Context,
    Measurement,
    baseline_summary,
    collect,
    parse_measurements,
    summarize,
)
from kernels_report.report_render import write_section

from aklr.core.contract import locate
from aklr.core.contract.baseline import OpBaseline
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import WorkloadSpec

COMMIT = "a" * 40
MEASURED = "b" * 40
MEASUREMENT = Measurement(
    run_id="44", measured_at="2026-11-19T09:18:14+00:00", commit=MEASURED, hardware="MI355X"
)


@pytest.fixture
def gemm_schemas():
    """Two committed gemm@1.0 kernels of one schema name: candidates for the same Points."""
    registry = ContractRegistry.default()
    schemas, errors = registry.read_schemas("aklr.ops.gemm")
    assert not errors
    by_name = {}
    for path, schema in schemas:
        if schema.contract_id == "gemm@1.0" and schema.kernel.sources:
            by_name.setdefault(schema.name, []).append((str(path), schema))
    pair = next(found for found in by_name.values() if len(found) >= 2)[:2]
    return registry.get("gemm@1.0"), pair


def measured(schema, *workloads):
    return schema.model_copy(update={"workloads": tuple(WorkloadSpec(**w) for w in workloads)})


def point(m, accuracy=None, speedup=None, microseconds=None):
    return {"axes": {"m": m}, "accuracy": accuracy, "speedup": speedup, "microseconds": microseconds}


def test_each_point_reports_what_dispatch_serves(gemm_schemas):
    contract, ((path_a, a), (path_b, b)) = gemm_schemas
    schemas = [
        (
            path_a,
            measured(
                a,
                point(1, "passed", 1.3, 2.0),
                point(2, "passed", 1.02, 2.0),
                point(4, "failed", 1.5, 2.0),
                point(8),
            ),
        ),
        (path_b, measured(b, point(1, "passed", 1.1, 2.4), point(2, "failed", 1.4, 1.0))),
    ]
    report = summarize(contract, schemas, min_speedup=1.05, baseline="aiter's gemm", measured=None)

    by_m = {p.dims["m"]: p for p in report.points}
    served = by_m["1"]
    assert (served.reason, served.path, served.speedup, served.microseconds) == ("served", path_a, 1.3, 2.0)
    assert (served.kernel, served.candidates) == (a.name, 2)
    assert served.requires == dict(a.kernel.requires)
    # not qualified outranks inaccurate, as dispatch's fallback reason does
    assert (by_m["2"].reason, by_m["2"].kernel, by_m["2"].speedup, by_m["2"].requires) == (
        "not_qualified",
        None,
        None,
        None,
    )
    assert (by_m["4"].reason, by_m["4"].speedup) == ("inaccurate", None)
    assert (by_m["8"].reason, by_m["8"].speedup) == ("not_measured", None)
    assert (report.served, len(report.points), report.status) == (1, 4, "partial")
    assert report.geomean == pytest.approx(1.3)
    # dims every point shares move out of the point labels
    assert "n" in report.fixed and "n" not in served.dims


def test_without_a_baseline_every_point_not_failed_is_served(gemm_schemas):
    contract, ((path_a, a), _) = gemm_schemas
    schemas = [(path_a, measured(a, point(1, "passed", 0.8, 2.0), point(2), point(4, "failed", 2.0, 1.0)))]
    report = summarize(contract, schemas, min_speedup=0, baseline=None, measured=None)

    reasons = {p.dims["m"]: p.reason for p in report.points}
    assert reasons == {"1": "served", "2": "served", "4": "inaccurate"}
    # validation's speedup without a baseline is no baseline speedup: the report shows none
    assert all(p.speedup is None for p in report.points)
    assert report.geomean is None


@pytest.mark.parametrize(
    ("env", "argv", "expected"), [("1.2", [], 1.2), ("1.2", ["--min-speedup", "1.1"], 1.1)]
)
def test_threshold_defaults_to_aklr_min_speedup(monkeypatch, tmp_path, env, argv, expected):
    import kernels_report.__main__ as cli

    seen = {}

    def collect_at(root, *, min_speedup, measurements):
        seen["t"] = min_speedup
        return []

    monkeypatch.setenv("AKLR_MIN_SPEEDUP", env)
    monkeypatch.setattr(cli, "read_measurements", lambda root: {})
    monkeypatch.setattr(cli, "collect", collect_at)
    monkeypatch.setattr(cli, "write_section", lambda *args: {"status_counts": {}})

    assert cli.main([*argv, str(tmp_path / "kernels")]) == 0
    assert seen["t"] == expected


class _DocumentedBaseline(OpBaseline):
    def baseline(self, *, x, **_):
        """Runs aiter's ``op`` on x; casts after.

        More."""
        return {"out": x}


class _UndocumentedBaseline(OpBaseline):
    def baseline(self, *, x, **_):
        return {"out": x}


@pytest.mark.parametrize(
    ("baseline", "expected"),
    [(_DocumentedBaseline(), "Runs aiter's op on x"), (_UndocumentedBaseline(), "_UndocumentedBaseline")],
    ids=["docstring", "no-docstring"],
)
def test_baseline_summary_is_the_first_sentence_of_the_baseline_docstring(baseline, expected):
    assert baseline_summary(baseline) == expected


def test_collect_reads_every_committed_kernel_with_any_size_axes_as_wildcards():
    root = locate.op_dir("aklr.ops.gemm").parents[2]
    reports = collect(root, min_speedup=1.05, measurements={})

    registry = ContractRegistry.default()
    assert reports and all(r.points for r in reports)
    for report in reports:
        contract = registry.get(report.contract_id)
        for name, axis in contract.axes.items():
            labels = [p.dims[name] for p in report.points if name in p.dims] + [report.fixed.get(name)]
            if axis.any_size:
                assert set(labels) <= {"*", None}


def test_measurements_take_each_schema_file_from_its_newest_bot_commit():
    log = (
        f"{'1' * 40}\x1f2026-11-19T09:18:14+00:00\x1fperf: measurements on MI355X (run 44) [skip ci]\x1f"
        f"perf: measurements on MI355X (run 44)\n\n{MEASURED} aklr/ops/gemm/impls/triton/x/1.json\n\x00"
        f"{'2' * 40}\x1f2026-11-18T09:18:14+00:00\x1fperf: measurements on MI355X (run 43) [skip ci]\x1f"
        f"{'c' * 40} aklr/ops/gemm/impls/triton/x/1.json\n{'c' * 40} aklr/ops/gemm/impls/triton/y/2.json\n\x00"
    )

    found = parse_measurements(log)

    assert found["aklr/ops/gemm/impls/triton/x/1.json"] == MEASUREMENT
    assert found["aklr/ops/gemm/impls/triton/y/2.json"].run_id == "43"


def test_section_pages_show_a_dash_for_points_not_served(tmp_path, gemm_schemas):
    contract, ((path_a, a), _) = gemm_schemas
    schemas = [(path_a, measured(a, point(1, "passed", 1.3, 2.0), point(2, "passed", 1.021, 2.0)))]
    report = summarize(contract, schemas, min_speedup=1.05, baseline="aiter's gemm", measured=MEASUREMENT)
    out = tmp_path / "kernels"
    (out / "contracts").mkdir(parents=True)
    (out / "contracts" / "withdrawn-1.0.html").write_text("stale")
    context = Context(
        commit=COMMIT,
        generated_at="2026-11-22T08:00:00Z",
        run_id="45",
        run_url="https://github.com/o/r/actions/runs/45",
        repo_url="https://github.com/o/r",
        ref="refs/heads/main",
        event="push",
    )

    write_section(out, [report], context)

    section = json.loads((out / "section.json").read_text())
    assert (section["id"], section["path"], section["status_counts"]) == (
        "kernels",
        "index.html",
        {"partial": 1},
    )
    assert section["last_run"]["commit"] == COMMIT
    assert not (out / "contracts" / "withdrawn-1.0.html").exists()
    page = (out / "contracts" / "gemm-1.0.html").read_text()
    assert "1.300" in page and "1.021" not in page
    # speedup times kernel µs is no baseline latency: no such column
    assert "Baseline µs" not in page
    assert "kernel.requires" in page
    index = (out / "index.html").read_text()
    assert "contracts/gemm-1.0.html" in index and "1/2" in index
    contracts = json.loads((out / "data.json").read_text())["contracts"]
    assert contracts[0]["points"][0]["requires"] == dict(a.kernel.requires)
