# SPDX-License-Identifier: MIT
"""Report failures reach the actual workflow check without GPU or network access."""

from __future__ import annotations

import copy
import inspect
import json
import os
import re
import shutil
import subprocess
from html import escape
from pathlib import Path

import pytest
import yaml
from agentx import pipeline, report_all, serving_config, utils

from tests.benchmarks.agentx_results import REPO


class ReportOutcomeCase:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.results_dir = self.root / "results"
        self.relative = utils.target_path("fixture", "fixture-gpu", "sglang")
        selection = serving_config.seal_selection(
            {
                "points": ["tp8:1", "tp8:2"],
                "policy": serving_config.POLICY,
                "duration": None,
                "points_requested": None,
            }
        )
        self.results_path = self.results_dir / self.relative / "results.json"
        self.results = {
            "schema": 4,
            "model": {"id": "fixture", "display_name": "Fixture"},
            "target": {"id": "fixture-gpu"},
            "framework": {"id": "sglang"},
            "run": {
                "id": "fixture-run",
                "mode": "live",
                "aklr_mode": "enabled",
                "completed": True,
                "points_expected": 2,
                "serving_sha256": "a" * 64,
                "attempt_id": "fixture-attempt",
                "finished_at": "2026-01-01T00:00:00Z",
                "point_policy": serving_config.POLICY,
                "selection": selection,
            },
            "points": [
                {
                    "arm": "tp8",
                    "conc": conc,
                    "status": "ok",
                    "tput_per_gpu": 100,
                    "p90_full_response_intvty": 20,
                    "raw_data": {
                        "request_metrics": {
                            "latency": {"full_response_itl": {"p90": 0.05}},
                        }
                    },
                }
                for conc in (1, 2)
            ],
        }
        self.results["control"] = copy.deepcopy(self.results)
        self.results["control"]["run"]["aklr_mode"] = "disabled"
        utils.write_json(self.results_path, self.results)
        self.plan = {
            "schema": 4,
            "run_id": "fixture-run",
            "targets": [
                {
                    "model": self.results["model"],
                    "target": self.results["target"],
                    "framework": self.results["framework"],
                    "status": "resolved",
                    "serving_sha256": "a" * 64,
                    "point_ids": ["tp8:1", "tp8:2"],
                    "point_policy": serving_config.POLICY,
                    "selection_manifest_sha256": selection["manifest_sha256"],
                }
            ],
        }
        self.workflow = yaml.safe_load((REPO / ".github/workflows/agentx-ci.yaml").read_text())
        self.check_step = self.workflow["jobs"]["report"]["steps"][-1]
        self.site_count = 0
        # Rendering the target page is independent of plan/result validation and CI outcomes.
        monkeypatch.setattr(report_all, "write_report", lambda *_args, **_kwargs: None)

    def render(self, previous=None):
        self.site_count += 1
        site = self.root / f"run-{self.site_count}" / "site"
        manifest = report_all.collect(
            self.plan,
            self.results_dir,
            site,
            previous or report_all.PreviousSite(None, None),
            "https://example.invalid/repo",
            REPO,
        )
        assert (site / "index.html").is_file()
        assert utils.read_json(site / "manifest.json") == manifest
        return site, manifest["targets"][0]

    def check(self, site, *, download_outcome="success"):
        if not shutil.which("bash") or not shutil.which("jq"):
            pytest.skip("the workflow outcome check requires bash and jq")
        return subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", self.check_step["run"]],
            env={
                **os.environ,
                "RUNNER_TEMP": str(site.parent),
                "RESULTS_DOWNLOAD_OUTCOME": download_outcome,
            },
            capture_output=True,
            text=True,
        ).returncode

    def previous_site(self):
        previous = self.root / "previous"
        manifest = copy.deepcopy(self.plan)
        manifest["targets"][0].update(status="failed", failure_stage="report")
        utils.write_json(previous / "manifest.json", manifest)
        utils.write_json(previous / self.relative / "results.json", self.results)
        return report_all.PreviousSite(previous, None)


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReportOutcomeCase:
    return ReportOutcomeCase(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("statuses", "status", "ok"),
    [
        (["ok"] * 4, "ok", 4),
        (["ok", "failed", "failed", "failed"], "partial", 1),
        (["failed"] * 4, "failed", 0),
    ],
    ids=["ok", "partial", "failed"],
)
def test_measurement_outcomes_do_not_fail_the_report_check(case, statuses, status, ok):
    current = copy.deepcopy(case.results)
    points = current["control"]["points"] + current["points"]
    for point, outcome in zip(points, statuses):
        point.update(status=outcome, error="measurement failed" if outcome == "failed" else None)
    utils.write_json(case.results_path, current)
    site, target = case.render()
    assert target["status"] == status
    assert (target["points_ok"], target["points_total"]) == (ok, 4)
    assert target["failure_stage"] is None
    assert case.check(site) == 0


def test_resolution_failure_fails_after_saving_the_report(case):
    error = "upstream <duration> changed & no recipe was selected"
    case.plan["targets"][0].update(status="resolve-failed", error=error)
    site, target = case.render()
    assert target["status"] == "resolve-failed"
    assert target["failure_stage"] == "plan"
    assert not target["report"]
    assert target["error"] == error
    html = (site / "index.html").read_text()
    assert escape(error) not in html
    assert error not in html
    assert "../index.html" not in html
    assert case.check(site, download_outcome="skipped") == 1


def test_missing_results_fail_after_saving_the_report(case):
    case.results_path.unlink()
    site, target = case.render()
    assert target["failure_stage"] == "report"
    assert case.check(site) == 1


@pytest.mark.parametrize("is_json", [False, True], ids=["invalid-json", "another-run"])
def test_invalid_results_fail_after_saving_the_report(case, is_json):
    invalid = copy.deepcopy(case.results)
    invalid["run"]["id"] = "another-run"
    case.results_path.write_text(json.dumps(invalid) if is_json else "{invalid JSON")
    site, target = case.render()
    assert target["failure_stage"] == "report"
    assert case.check(site) == 1


@pytest.mark.parametrize(
    "invalid", ["missing-control", "empty-control", "enabled-label", "disabled-label", "mode"]
)
def test_missing_or_invalid_modes_fail_the_report_check(case, invalid):
    current = copy.deepcopy(case.results)
    if invalid == "missing-control":
        current.pop("control")
    elif invalid == "empty-control":
        current["control"] = None
    elif invalid == "enabled-label":
        current["run"]["aklr_mode"] = "disabled"
    elif invalid == "disabled-label":
        current["control"]["run"]["aklr_mode"] = "enabled"
    else:
        current["control"]["run"]["mode"] = "mock"
    utils.write_json(case.results_path, current)
    site, target = case.render()
    assert target["status"] == "failed"
    assert target["failure_stage"] == "report"
    assert case.check(site) == 1


@pytest.mark.parametrize("kept", [0, 1])
def test_an_empty_or_partial_mode_keeps_successful_measurements(case, kept):
    current = copy.deepcopy(case.results)
    current["control"]["points"] = current["control"]["points"][:kept]
    current["control"]["run"]["completed"] = False
    utils.write_json(case.results_path, current)
    site, target = case.render()
    assert target["status"] == "partial"
    assert (target["points_ok"], target["points_total"]) == (2 + kept, 4)
    assert target["failure_stage"] is None
    assert case.check(site) == 0


def unreadable(_path):
    raise PermissionError("cannot read results")


def test_unreadable_results_are_saved_as_a_report_failure(case):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(report_all, "read_json", unreadable)
        site, target = case.render()
    assert target["error"] == "cannot read results"
    assert target["failure_stage"] == "report"
    assert case.check(site) == 1


def test_failed_download_is_not_hidden_by_usable_results(case):
    site, target = case.render()
    assert target["status"] == "ok"
    assert case.check(site, download_outcome="failure") == 1
    download = next(
        step for step in case.workflow["jobs"]["report"]["steps"] if step.get("id") == "download_results"
    )
    assert download["continue-on-error"]
    assert case.check_step["env"]["RESULTS_DOWNLOAD_OUTCOME"] == "${{ steps.download_results.outcome }}"


@pytest.mark.parametrize(("status", "stage"), [("resolve-failed", "plan"), ("resolved", "report")])
def test_historical_results_do_not_hide_current_plan_or_report_errors(case, status, stage):
    previous = case.previous_site()
    case.results_path.unlink()
    case.plan["targets"][0]["status"] = status
    site, target = case.render(previous)
    assert target["report"]
    assert target["points_ok"] == 4
    assert target["failure_stage"] == stage
    assert case.check(site) == 1


def test_unselected_history_does_not_repeat_an_old_failure(case):
    previous = case.previous_site()
    case.plan["targets"] = []
    site, target = case.render(previous)
    assert target["status"] == "not-run"
    assert target["failure_stage"] is None
    assert case.check(site, download_outcome="skipped") == 0


def test_outcome_check_follows_the_upload_and_publish_accepts_a_failed_report_job(case):
    steps = case.workflow["jobs"]["report"]["steps"]
    upload = next(i for i, step in enumerate(steps) if step.get("id") == "site_artifact")
    assert upload < len(steps) - 1
    publish_condition = case.workflow["jobs"]["publish"]["if"]
    assert "!cancelled()" in publish_condition
    assert "needs.report.outputs.site_ready == 'true'" in publish_condition
    assert "success()" not in publish_condition
    assert "needs.report.result" not in publish_condition


def test_the_local_pipeline_passes_the_report_builder_only_what_it_takes():
    """pipeline's report step calls report_all.build by keyword: one it does not take fails every local pipeline run
    after its sweep (it passed config after the builder dropped it)."""
    source = inspect.getsource(pipeline)
    call = source[source.index("report_all.build(") :]
    call = call[: call.index("\n        )\n")]
    passed = re.findall(r"^\s+(\w+)=", call, re.M)
    assert passed and set(passed) <= set(inspect.signature(report_all.build).parameters)
