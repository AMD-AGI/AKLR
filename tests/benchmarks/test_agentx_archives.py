# SPDX-License-Identifier: MIT
"""Keep measurement evidence intact while rendering verified report views."""

from __future__ import annotations

import copy
from html import escape
from pathlib import Path
from unittest import mock

import pytest

from tests.benchmarks.agentx_results import (
    REPO,
    formal_results,
    legacy_results,
    report,
    report_all,
    utils,
)


class ArchiveCase:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.relative = utils.target_path("fixture", "fixture-gpu", "sglang")
        self.serial = 0
        monkeypatch.setattr(report, "landed", mock.Mock(return_value=None))

    def directory(self, name):
        self.serial += 1
        return self.root / f"{name}-{self.serial}"

    def descriptor(self, results):
        return {
            "model": results["model"],
            "target": results["target"],
            "framework": results["framework"],
            "status": "resolved",
            "serving_sha256": results["run"]["serving_sha256"],
            "point_ids": [f"{point['arm']}:{point['conc']}" for point in results["points"]],
            "point_policy": results["run"]["point_policy"],
            "selection_manifest_sha256": results["run"]["selection"]["manifest_sha256"],
        }

    def previous(self, results, *, baseline=None):
        directory = self.directory("previous")
        utils.write_json(
            directory / "manifest.json",
            {"schema": 4, "run_id": results["run"]["id"], "targets": [self.descriptor(results)]},
        )
        utils.write_json(directory / self.relative / "results.json", results)
        if baseline is not None:
            utils.write_json(directory / self.relative / "baseline-pareto.json", baseline)
        return directory

    def publish(self, *, current=None, previous=None, selected=True, error=None):
        directory = self.directory("publish")
        results_dir = directory / "results"
        planned = self.descriptor(current or legacy_results("current-run"))
        if error:
            planned.update(status="resolve-failed", error=error)
        plan = {
            "schema": 4,
            "run_id": current["run"]["id"] if current else "current-run",
            "targets": [planned] if selected else [],
        }
        if current is not None:
            utils.write_json(results_dir / self.relative / "results.json", current)
        out = directory / "site"
        manifest = report_all.collect(
            plan,
            results_dir,
            out,
            report_all.PreviousSite(previous, None),
            "https://example.invalid/repo",
            REPO,
        )
        return out, manifest["targets"][0]


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ArchiveCase:
    return ArchiveCase(tmp_path, monkeypatch)


@pytest.mark.parametrize("missing", [(1, 2, 3), (2,)], ids=str)
def test_historical_results_and_baselines_survive_failed_or_partial_normalization(case, missing):
    original = formal_results("previous-run")
    for mode in (original, original["control"]):
        for point in mode["points"]:
            point.pop("p90_full_response_intvty")
            if point["conc"] in missing:
                point.pop("raw_data")
    previous = case.previous(original, baseline=original)
    before = (previous / case.relative / "results.json").read_bytes()

    for _ in range(2):
        site, target = case.publish(previous=previous, selected=False)
        assert utils.read_json(site / case.relative / "results.json") == original
        assert utils.read_json(site / case.relative / "baseline-pareto.json") == original
        summary = utils.read_json(site / case.relative / "report.json")
        pairs = summary["aklr_comparison"]["points"]
        assert [row["conc"] for row in pairs] == [c for c in (1, 2, 3) if c not in missing]
        assert target["points_ok"] == 2 * (3 - len(missing))
        assert ("pareto" in target["baseline_unavailable"]) == (len(missing) == 3)
        assert (previous / case.relative / "results.json").read_bytes() == before
        previous = site


def test_unverified_full_response_only_baseline_is_preserved_but_unavailable(case):
    original = formal_results("previous-run")
    for mode in (original, original["control"]):
        for point in mode["points"]:
            point.pop("raw_data")
            point.pop("p90_e2e_norm_intvty")
    previous = case.previous(original, baseline=original)

    site, target = case.publish(previous=previous, selected=False)

    assert utils.read_json(site / case.relative / "baseline-pareto.json") == original
    assert utils.read_json(site / case.relative / "results.json") == original
    assert "pareto" in target["baseline_unavailable"]
    assert "pareto" not in target["baselines"]
    assert target["points_ok"] == 0
    summary = utils.read_json(site / case.relative / "report.json")
    assert summary["frontiers"]["combined"] == []
    assert summary["aklr_comparison"]["status"] == "unavailable"


def test_current_results_archive_and_baseline_retain_unverifiable_point_evidence(case):
    current = formal_results("current-run")
    for mode in (current, current["control"]):
        mode["points"][1].pop("raw_data")
    original = copy.deepcopy(current)

    site, target = case.publish(current=current)

    assert current == original
    assert utils.read_json(site / case.relative / "results.json") == original
    assert utils.read_json(site / case.relative / "baseline-pareto.json") == original
    assert target["status"] == "partial"
    assert target["points_ok"] == 4
    summary = utils.read_json(site / case.relative / "report.json")
    assert [row["conc"] for row in summary["aklr_comparison"]["points"]] == [1, 3]


def test_single_result_entry_can_rerender_without_rewriting_the_archive(case):
    original = legacy_results("previous-run")
    source = utils.write_json(case.root / "input.json", original)
    config = case.root / "config.yaml"
    config.write_text("report:\n  repo_url: https://example.invalid/repo\n")

    for _ in range(2):
        out = case.directory("single")
        assert report.main(["--results", str(source), "--out", str(out), "--config", str(config)]) == 0
        assert utils.read_json(out / "results.json") == original
        summary = utils.read_json(out / "report.json")
        point = summary["frontiers"]["combined"][0]
        assert point["p90_full_response_intvty"] == pytest.approx(1 / 0.00351)
        source = out / "results.json"


def test_historical_single_mode_results_remain_readable(case):
    original = legacy_results("previous-run")
    original.pop("control")
    previous = case.previous(original, baseline=original)

    site, target = case.publish(previous=previous, selected=False)

    assert target["report"]
    assert target["failure_stage"] is None
    assert target["points_ok"] == 1
    assert utils.read_json(site / case.relative / "results.json") == original
    assert utils.read_json(site / case.relative / "baseline-pareto.json") == original


def test_current_failure_is_visible_when_rendering_a_historical_target(case):
    previous = case.previous(legacy_results("previous-run"))
    error = "recipe <weights> could not resolve & no measurement ran"

    site, target = case.publish(previous=previous, error=error)

    assert target["failure_stage"] == "plan"
    assert target["report"]
    for path in (site / "index.html", site / case.relative / "index.html"):
        html = path.read_text()
        assert escape(error) in html
        assert error not in html
