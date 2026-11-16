# SPDX-License-Identifier: MIT
"""Partial measurements can publish a baseline and support later point comparisons."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest import mock

import pytest

POINT_IDS = ["tp1:1", "tp1:2", "tp1:3"]


from tests.benchmarks.agentx_results import (
    REPO,
    analysis,
    fail,
    formal_results,
    recipe_resolve,
    report,
    report_all,
    utils,
)


class BaselineCase:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.config = self.root / "config.yaml"
        self.config.write_text("report:\n  repo_url: https://example.invalid/repo\n")
        self.relative = utils.target_path("fixture", "fixture-gpu")
        self.site_count = 0
        # Git history is independent of baseline eligibility and metric comparisons.
        monkeypatch.setattr(report, "landed", mock.Mock(return_value=None))

    def publish(self, results, previous=None):
        self.site_count += 1
        run_dir = self.root / f"run-{self.site_count}"
        policy = results["run"].get("point_policy", "all")
        target = {
            "model": results["model"],
            "target": results["target"],
            "status": "resolved",
            "serving_sha256": results["run"]["serving_sha256"],
            "point_policy": policy,
            "point_ids": POINT_IDS,
        }
        if policy == recipe_resolve.POLICY:
            target["selection_manifest_sha256"] = results["run"]["selection"]["manifest_sha256"]
        plan_path = utils.write_json(
            run_dir / "plan.json",
            {"schema": 3, "run_id": results["run"]["id"], "targets": [target]},
        )
        results_dir = run_dir / "results"
        result_path = utils.write_json(results_dir / self.relative / "results.json", results)
        original = result_path.read_bytes()
        site = run_dir / "site"
        manifest = report_all.build(
            plan_path,
            results_dir,
            site,
            config=self.config,
            git_dir=REPO,
            baseline_dir=previous,
        )
        assert result_path.read_bytes() == original
        assert (site / "index.html").is_file()
        assert utils.read_json(site / "manifest.json") == manifest
        return site, manifest["targets"][0]

    def baseline(self, site, kind="live"):
        return utils.read_json(site / self.relative / f"baseline-{kind}.json")


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BaselineCase:
    return BaselineCase(tmp_path, monkeypatch)


@pytest.mark.parametrize("success", ["enabled", "disabled", "disabled-with-empty-enabled", "disjoint"])
def test_any_successful_mode_promotes_without_requiring_a_successful_pair(case, success):
    current = formal_results(success)
    for mode in (current, current["control"]):
        for point in mode["points"]:
            fail(point)
    if success in ("enabled", "disjoint"):
        current["points"][0].update(status="ok", error=None)
    if success != "enabled":
        current["control"]["points"][1].update(status="ok", error=None)
    if success == "disabled-with-empty-enabled":
        current["points"] = []
        current["run"]["completed"] = False
        current["serving"]["arms"]["tp1"].pop("arm_measurement_sha256")
    site, target = case.publish(current)
    assert target["status"] == "partial"
    assert target["points_ok"] == (2 if success == "disjoint" else 1)
    assert target["points_total"] == 6
    assert target["failure_stage"] is None
    assert target["baselines"]["live"] == current["run"]["finished_at"]
    stored = case.baseline(site)
    assert stored == utils.validated_results(current)
    assert not utils.complete_measurement(stored)
    assert analysis.compare_aklr(stored)["status"] != "ok"


@pytest.mark.parametrize("outcome", ["failed", "empty", "invalid-throughput", "invalid-latency"])
def test_no_valid_success_preserves_the_previous_baseline(case, outcome):
    previous, _ = case.publish(formal_results("previous-run"))
    expected = case.baseline(previous)
    current = formal_results(outcome)
    for mode in (current, current["control"]):
        if outcome == "empty":
            mode["points"] = []
            mode["run"]["completed"] = False
        for point in mode["points"]:
            if outcome == "failed":
                fail(point)
            elif outcome == "invalid-throughput":
                point["tput_per_gpu"] = 0
            elif outcome == "invalid-latency":
                point["raw_data"]["request_metrics"]["latency"]["full_response_itl"]["p90"] = 0
    site, target = case.publish(current, previous)
    assert target["status"] == "failed"
    assert target["points_ok"] == 0
    assert target["failure_stage"] is None
    assert case.baseline(site) == expected
    assert case.baseline(previous) == expected


def test_all_failed_first_run_does_not_create_a_baseline(case):
    current = formal_results()
    for mode in (current, current["control"]):
        for point in mode["points"]:
            fail(point)
    site, target = case.publish(current)
    assert target["baselines"] == {}
    assert not (site / case.relative / "baseline-live.json").exists()


def test_partial_baseline_round_trips_and_compares_only_successful_points(case):
    prior = formal_results("previous-run")
    for mode in (prior, prior["control"]):
        fail(mode["points"][1])
    previous, _ = case.publish(prior)
    expected_previous = case.baseline(previous)
    current = formal_results("current-run")
    for mode in (current, current["control"]):
        fail(mode["points"][2])
    current["points"][0]["tput_per_gpu"] *= 1.25
    current["points"][0]["raw_data"]["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"] *= 1.25
    current["points"][0]["raw_data"]["request_metrics"]["latency"]["full_response_itl"]["p90"] /= 2
    site, target = case.publish(current, previous)
    assert target["status"] == "partial"
    assert target["failure_stage"] is None
    assert case.baseline(site)["run"]["id"] == "current-run"
    assert case.baseline(previous) == expected_previous
    stored = case.baseline(site)
    assert stored["points"][2]["status"] == "failed"
    assert stored["points"][2]["error"] == "server failed before ready"
    comparison = utils.read_json(site / case.relative / "report.json")["comparison"]
    assert comparison["status"] == "ok"
    assert comparison["baseline_run"] == "previous-run"
    arm = comparison["arms"]["tp1"]
    rows = {row["conc"]: row for row in arm["points"]}
    assert rows[1]["tput_ratio"] == pytest.approx(1.25)
    assert rows[1]["intvty_ratio"] == pytest.approx(2)
    assert arm["tput_geomean_ratio"] == pytest.approx(1.25)
    for conc in (2, 3):
        assert rows[conc].get("tput_ratio") is None
        assert rows[conc].get("intvty_ratio") is None
    assert "week-over-week comparison disabled" not in (site / case.relative / "index.html").read_text()


def test_interrupted_baseline_compares_matching_point_measurements(case):
    prior = formal_results("interrupted-run")
    for mode in (prior, prior["control"]):
        mode["points"] = mode["points"][:2]
        fail(mode["points"][1])
        mode["run"]["completed"] = False
        mode["serving"]["arms"]["tp1"].pop("arm_measurement_sha256")
    previous, _ = case.publish(prior)
    current = formal_results("current-run")
    current["points"][0]["tput_per_gpu"] *= 1.2
    site, _ = case.publish(current, previous)
    comparison = utils.read_json(site / case.relative / "report.json")["comparison"]
    assert comparison["status"] == "ok"
    rows = {row["conc"]: row for row in comparison["arms"]["tp1"]["points"]}
    assert rows[1]["tput_ratio"] == pytest.approx(1.2)
    assert rows[2].get("tput_ratio") is None
    assert rows[3]["baseline"] is None
    assert rows[3].get("tput_ratio") is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration_override_s", 600),
        ("model_path_override", "/fixture/model"),
        ("arms_requested", "tp1"),
        ("points_requested", "tp1:1"),
        ("source_dirty", True),
        ("source_dirty", None),
        ("execution", {}),
    ],
    ids=str,
)
def test_smoke_dirty_and_unverified_runs_do_not_replace_the_baseline(case, field, value):
    previous, _ = case.publish(formal_results("previous-run"))
    expected = case.baseline(previous)
    current = formal_results("unqualified-run")
    for mode in (current, current["control"]):
        mode["run"][field] = value
    site, target = case.publish(current, previous)
    assert target["failure_stage"] is None
    assert case.baseline(site) == expected


def test_interrupted_baseline_does_not_compare_changed_point_settings(case):
    prior = formal_results("interrupted-run")
    for mode in (prior, prior["control"]):
        mode["points"] = mode["points"][:1]
        mode["run"]["completed"] = False
        mode["serving"]["arms"]["tp1"].pop("arm_measurement_sha256")
    previous, _ = case.publish(prior)
    current = formal_results("changed-settings")
    for mode in (current, current["control"]):
        point = mode["points"][0]
        point["measurement"]["client_env"]["DURATION"] = "30"
        point["measurement_sha256"] = utils.sha256_json(point["measurement"])
        mode["serving"]["arms"]["tp1"]["arm_measurement_sha256"] = utils.sha256_json(
            {p["conc"]: p["measurement"] for p in mode["points"]}
        )
    site, _ = case.publish(current, previous)
    comparison = utils.read_json(site / case.relative / "report.json")["comparison"]
    assert comparison["status"] == "ok"
    arm = comparison["arms"]["tp1"]
    assert arm.get("tput_geomean_ratio") is None
    assert not any(row.get("tput_ratio") for row in arm["points"])


@pytest.mark.parametrize("mismatch", ["source", "image", "mode-switch"])
def test_mismatched_pair_provenance_does_not_replace_the_baseline(case, mismatch):
    previous, _ = case.publish(formal_results("previous-run"))
    expected = case.baseline(previous)
    current = formal_results("mismatched-run")
    control = current["control"]
    if mismatch == "source":
        control["run"]["aklr_sha"] = "4" * 40
    elif mismatch == "image":
        control["run"]["execution"]["image_id"] = "sha256:" + "4" * 64
    else:
        control["points"][0]["measurement"]["launch"]["env"]["AKLR_DISABLE"] = "0"
    site, _ = case.publish(current, previous)
    assert case.baseline(site) == expected


@pytest.mark.parametrize("invalid", ["unplanned", "duplicate", "expected-count"])
def test_control_points_must_belong_to_the_plan_without_duplicates(case, invalid):
    previous, _ = case.publish(formal_results("previous-run"))
    expected = case.baseline(previous)
    current = formal_results("invalid-control")
    control = current["control"]
    if invalid == "unplanned":
        control["points"][1].update(conc=4, exp_name="tp1-c4")
    elif invalid == "duplicate":
        control["points"][1] = copy.deepcopy(control["points"][0])
    else:
        control["run"]["points_expected"] = 4
    site, target = case.publish(current, previous)
    assert target["status"] == "failed"
    assert target["failure_stage"] == "report"
    assert case.baseline(site) == expected


def test_pareto_subset_updates_only_the_pareto_baseline(case):
    previous, _ = case.publish(formal_results("full-grid-run"))
    live = case.baseline(previous)
    current = formal_results("pareto-run")
    selection = {
        "schema": 1,
        "policy": recipe_resolve.POLICY,
        "algorithm": recipe_resolve.ALGORITHM,
        "scope": {
            "model": "fixture",
            "hardware": "fixture-gpu",
            "framework": "sglang",
            "precision": "fp4",
        },
        "curve_date": "2026-11-09",
        "curve_workflow_run_id": "fixture-curve",
        "candidate_count": 3,
        "points": POINT_IDS,
        "unselected": [],
        "errors": [],
        "selection_sha256": "5" * 64,
    }
    selection["manifest_sha256"] = utils.sha256_json(selection)
    for mode in (current, current["control"]):
        mode["run"].update(
            point_policy=recipe_resolve.POLICY, selection=copy.deepcopy(selection), completed=False
        )
        mode["points"] = mode["points"][:1]
        mode["serving"]["arms"]["tp1"].pop("arm_measurement_sha256")
    site, target = case.publish(current, previous)
    assert target["status"] == "partial"
    assert set(target["baselines"]) == {"live", "pareto"}
    assert case.baseline(site) == live
    assert case.baseline(site, "pareto")["run"]["id"] == "pareto-run"
    next_site, _ = case.publish(formal_results("next-full-grid-run"), site)
    assert case.baseline(next_site, "pareto")["run"]["id"] == "pareto-run"
