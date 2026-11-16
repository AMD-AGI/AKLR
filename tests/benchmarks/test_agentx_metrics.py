# SPDX-License-Identifier: MIT
"""InferenceX CSV metric compatibility and offline report migration."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest import mock

import pytest

from tests.benchmarks.agentx_results import (
    E2E,
    FULL_RESPONSE,
    REPO,
    aggregate,
    analysis,
    legacy_results,
    recipe_resolve,
    report,
    report_all,
    runtime,
    utils,
)


def test_runtime_matches_csv_reciprocal_instead_of_other_interactivity_fields():
    value = runtime.metrics_of(aggregate())
    assert value[FULL_RESPONSE] == 1 / 0.00351
    assert value[FULL_RESPONSE] != 285.10342
    assert value[FULL_RESPONSE] != 84.72682
    assert value["tput_per_gpu"] == 3390.56736
    assert value["num_requests_successful"] == 73


@pytest.mark.parametrize(
    "itl",
    [
        0,
        -1,
        True,
        False,
        None,
        "0.00351",
        float("nan"),
        float("inf"),
        5e-324,
        pytest.param(10**400, id="10**400"),
    ],
)
def test_bad_itl_never_falls_back_to_other_metrics(itl):
    with pytest.raises(ValueError, match="Invalid full-response P90"):
        runtime.metrics_of(aggregate(itl))


def test_missing_itl_never_falls_back_to_other_metrics():
    missing = aggregate()
    del missing["request_metrics"]["latency"]["full_response_itl"]
    with pytest.raises(ValueError, match="Missing full-response P90 ITL"):
        runtime.metrics_of(missing)


def test_legacy_modes_recompute_without_changing_raw_data_or_source_identity():
    source = legacy_results()
    source["points"][0][FULL_RESPONSE] = 999
    failed = {"arm": "tp1", "conc": 2, "status": "failed", "error": "server failed before ready"}
    source["points"].append(failed)
    original = copy.deepcopy(source)
    migrated = utils.validated_results(source)
    assert source == original
    assert migrated["run"] == original["run"]
    assert migrated["points"][0][FULL_RESPONSE] == 1 / 0.00351
    assert migrated["control"]["points"][0][FULL_RESPONSE] == 1 / 0.00359
    for before, after in ((source, migrated), (source["control"], migrated["control"])):
        assert after["points"][0]["raw_data"] == before["points"][0]["raw_data"]
        assert after["points"][0][E2E] == before["points"][0][E2E]
        assert after["points"][0]["tput_per_gpu"] == before["points"][0]["tput_per_gpu"]
    assert migrated["points"][1] == failed
    assert utils.validated_results(migrated) == migrated


@pytest.mark.parametrize("invalid_raw", [None, aggregate(0)], ids=["missing", "zero-itl"])
def test_missing_raw_e2e_only_points_and_invalid_raw_with_cached_values_are_unavailable(invalid_raw):
    source = legacy_results()
    point = source["points"][0]
    if invalid_raw is None:
        del point["raw_data"]
    else:
        point.update(raw_data=invalid_raw, p90_full_response_intvty=999)
    migrated = utils.validated_results(source)
    assert migrated["points"][0]["status"] == "failed"
    assert FULL_RESPONSE not in migrated["points"][0]
    assert "full-response P90 ITL" in migrated["points"][0]["error"]
    assert analysis.measured(migrated) == []


def test_cached_interactivity_does_not_replace_raw_itl_evidence():
    source = utils.validated_results(legacy_results())
    for mode, cached in ((source, 999), (source["control"], 1)):
        point = mode["points"][0]
        point.pop("raw_data")
        point[FULL_RESPONSE] = cached
    original = copy.deepcopy(source)

    normalized = utils.validated_results(source)

    assert source == original
    for mode in (normalized, normalized["control"]):
        point = mode["points"][0]
        assert point["status"] == "failed"
        assert "Missing full-response P90 ITL" in point["error"]
        assert FULL_RESPONSE not in point
        assert analysis.measured(mode) == []
    assert analysis.compare_aklr(normalized)["status"] == "unavailable"
    assert not report_all.eligible(normalized)
    assert not report_all.eligible(normalized, historical=True)


def test_pareto_report_axis_and_published_upstream_selection_axis_are_independent():
    fast = {FULL_RESPONSE: 300, E2E: 10, "tput_per_gpu": 100}
    slow = {FULL_RESPONSE: 200, E2E: 20, "tput_per_gpu": 90}
    assert analysis.frontier([fast, slow]) == [fast]
    assert recipe_resolve.X == "p90_e2e_norm_intvty"
    assert recipe_resolve.frontier([fast, slow], x_axis=recipe_resolve.X) == [fast, slow]


def test_paired_and_weekly_ratios_use_full_response_after_migration():
    current = utils.validated_results(legacy_results())
    previous = legacy_results("previous-run", itl=0.00702)
    previous["points"][0][E2E] *= 2
    previous = utils.validated_results(previous)
    paired = analysis.compare_aklr(current)
    assert paired["status"] == "ok"
    assert paired["points"][0]["intvty_ratio"] == pytest.approx(0.00359 / 0.00351)
    weekly = analysis.compare(current, previous)
    assert weekly["status"] == "ok"
    assert weekly["arms"]["tp1"]["points"][0]["intvty_ratio"] == pytest.approx(2)


class ReportMetricMigrationCase:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.relative = utils.target_path("fixture", "fixture-gpu")
        self.config = self.root / "config.yaml"
        self.config.write_text("report:\n  repo_url: https://example.invalid/repo\n")
        # Git history is independent of report metrics; keep this suite offline.
        monkeypatch.setattr(report, "landed", mock.Mock(return_value=None))

    def build_with_baseline(self, baseline, *, formal=False, current=None, expected_status="ok"):
        current = copy.deepcopy(current) if current is not None else legacy_results()
        if not formal:
            for result in (current, current["control"]):
                result["run"]["duration_override_s"] = 600
        plan = {
            "schema": 3,
            "run_id": current["run"]["id"],
            "targets": [
                {
                    "model": current["model"],
                    "target": current["target"],
                    "status": "resolved",
                    "serving_sha256": current["run"]["serving_sha256"],
                    "point_ids": ["tp1:1"],
                }
            ],
        }
        plan_path = utils.write_json(self.root / "plan.json", plan)
        results_dir = self.root / "results"
        utils.write_json(results_dir / self.relative / "results.json", current)
        prior = self.root / "previous"
        utils.write_json(prior / "manifest.json", plan)
        baseline_path = utils.write_json(prior / self.relative / "baseline-live.json", baseline)
        original_bytes = baseline_path.read_bytes()
        out = self.root / "site"
        manifest = report_all.build(
            plan_path, results_dir, out, config=self.config, git_dir=REPO, baseline_dir=prior
        )
        assert baseline_path.read_bytes() == original_bytes
        assert (out / "index.html").is_file()
        assert manifest["targets"][0]["failure_stage"] is None
        assert manifest["targets"][0]["status"] == expected_status
        return out, manifest["targets"][0]


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReportMetricMigrationCase:
    return ReportMetricMigrationCase(tmp_path, monkeypatch)


def test_report_cli_redraws_old_results_with_new_coordinates_and_labels(case):
    source = legacy_results()
    source_path = utils.write_json(case.root / "results.json", source)
    original_bytes = source_path.read_bytes()
    out = case.root / "site"
    code = report.main(["--results", str(source_path), "--out", str(out), "--config", str(case.config)])
    assert code == 0
    assert source_path.read_bytes() == original_bytes
    result = utils.read_json(out / "results.json")
    summary = utils.read_json(out / "report.json")
    assert result == source
    assert summary["axes"] == {"x": FULL_RESPONSE, "y": "tput_per_gpu"}
    assert summary["frontiers"]["combined"][0][FULL_RESPONSE] == 1 / 0.00351
    page = (out / "index.html").read_text()
    assert "P90 Interactivity (tok/s/user): 284.9" in page
    assert "Throughput/Chip (tok/s): 3390.6" in page
    assert "1 / P90(full-response ITL)" in page
    assert "P90 E2E interactivity" not in page
    assert "Off P90 E2E tok/s/user" not in page


def test_historical_baseline_with_raw_is_normalized_for_comparison_without_rewriting_archive(case):
    baseline = legacy_results("previous-run", itl=0.00702)
    original = copy.deepcopy(baseline)
    out, target = case.build_with_baseline(baseline)
    stored = utils.read_json(out / case.relative / "baseline-live.json")
    summary = utils.read_json(out / case.relative / "report.json")
    assert baseline == original
    assert stored == original
    assert summary["comparison"]["status"] == "ok"
    assert summary["comparison"]["arms"]["tp1"]["points"][0]["intvty_ratio"] == pytest.approx(2)
    assert "live" in target["baselines"]
    assert not target["baseline_unavailable"]


def test_legacy_baseline_without_raw_is_preserved_without_mixing_axes(case):
    baseline = legacy_results("previous-run")
    for result in (baseline, baseline["control"]):
        del result["points"][0]["raw_data"]
    out, target = case.build_with_baseline(baseline)
    assert utils.read_json(out / case.relative / "baseline-live.json") == baseline
    summary = utils.read_json(out / case.relative / "report.json")
    assert summary["comparison"]["status"] != "ok"
    assert summary["comparison"]["arms"] == {}
    assert "live" not in target["baselines"]
    assert "live" in target["baseline_unavailable"]
    assert "full-response P90 unavailable" in (out / "index.html").read_text()
    assert "historical baseline is preserved" in (out / case.relative / "index.html").read_text()


def test_new_formal_run_can_replace_an_unconvertible_historical_baseline(case):
    baseline = legacy_results("previous-run")
    for result in (baseline, baseline["control"]):
        del result["points"][0]["raw_data"]
    out, target = case.build_with_baseline(baseline, formal=True)
    stored = utils.read_json(out / case.relative / "baseline-live.json")
    assert stored["run"]["id"] == "fixture-run"
    assert stored == legacy_results()
    summary = utils.read_json(out / case.relative / "report.json")
    assert summary["frontiers"]["combined"][0][FULL_RESPONSE] == 1 / 0.00351
    assert summary["disabled_frontiers"]["combined"][0][FULL_RESPONSE] == 1 / 0.00359
    assert "live" in target["baselines"]
    assert not target["baseline_unavailable"]


def test_unverifiable_cached_points_neither_compare_nor_replace_the_baseline(case):
    current = utils.validated_results(legacy_results())
    for mode, cached in ((current, 999), (current["control"], 1)):
        mode["points"][0].pop("raw_data")
        mode["points"][0][FULL_RESPONSE] = cached
    baseline = legacy_results("previous-run")

    out, target = case.build_with_baseline(baseline, current=current, formal=True, expected_status="failed")

    assert (target["points_ok"], target["points_total"]) == (0, 2)
    assert utils.read_json(out / case.relative / "baseline-live.json") == baseline
    assert utils.read_json(out / case.relative / "results.json") == current
    summary = utils.read_json(out / case.relative / "report.json")
    assert summary["comparison"]["arms"] == {}
    assert summary["aklr_comparison"]["status"] == "unavailable"
