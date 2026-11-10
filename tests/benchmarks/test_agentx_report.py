# SPDX-License-Identifier: MIT
"""Report failures reach the actual workflow check without GPU or network access."""

from __future__ import annotations

import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parents[2]
with mock.patch.object(sys, "path", [str(REPO / "benchmarks/agentx"), *sys.path]):
    report_all = importlib.import_module("agentx_ci.report_all")
    utils = importlib.import_module("agentx_ci.utils")


class AgentXReportOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.results_dir = self.root / "results"
        self.relative = utils.target_path("fixture", "fixture-gpu")
        self.results_path = self.results_dir / self.relative / "results.json"
        self.results = {
            "schema": 3,
            "model": {"id": "fixture", "display_name": "Fixture"},
            "target": {"id": "fixture-gpu"},
            "run": {
                "id": "fixture-run",
                "mode": "live",
                "completed": True,
                "points_expected": 2,
                "serving_sha256": "a" * 64,
                "attempt_id": "fixture-attempt",
                "finished_at": "2026-01-01T00:00:00Z",
            },
            "points": [
                {
                    "arm": "tp8",
                    "conc": conc,
                    "status": "ok",
                    "tput_per_gpu": 100,
                    "p90_full_response_intvty": 20,
                }
                for conc in (1, 2)
            ],
        }
        self.results["control"] = copy.deepcopy(self.results)
        utils.write_json(self.results_path, self.results)
        self.plan = {
            "schema": 3,
            "run_id": "fixture-run",
            "targets": [
                {
                    "model": self.results["model"],
                    "target": self.results["target"],
                    "status": "resolved",
                    "serving_sha256": "a" * 64,
                    "point_ids": ["tp8:1", "tp8:2"],
                }
            ],
        }
        self.workflow = yaml.safe_load((REPO / ".github/workflows/agentx-ci.yaml").read_text())
        self.check_step = self.workflow["jobs"]["report"]["steps"][-1]
        self.site_count = 0
        # Rendering the target page is independent of plan/result validation and CI outcomes.
        self.enterContext(mock.patch.object(report_all, "write_report"))

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
        self.assertTrue((site / "index.html").is_file())
        self.assertEqual(utils.read_json(site / "manifest.json"), manifest)
        return site, manifest["targets"][0]

    def check(self, site, *, download_outcome="success"):
        if not shutil.which("bash") or not shutil.which("jq"):
            self.skipTest("the workflow outcome check requires bash and jq")
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

    def test_measurement_outcomes_do_not_fail_the_report_check(self):
        for statuses, status, ok in (
            (["ok"] * 4, "ok", 4),
            (["ok", "failed", "failed", "failed"], "partial", 1),
            (["failed"] * 4, "failed", 0),
        ):
            with self.subTest(status=status):
                current = copy.deepcopy(self.results)
                points = current["control"]["points"] + current["points"]
                for point, outcome in zip(points, statuses):
                    point.update(status=outcome, error="measurement failed" if outcome == "failed" else None)
                utils.write_json(self.results_path, current)
                site, target = self.render()
                self.assertEqual(target["status"], status)
                self.assertEqual((target["points_ok"], target["points_total"]), (ok, 4))
                self.assertIsNone(target["failure_stage"])
                self.assertEqual(self.check(site), 0)

    def test_resolution_failure_fails_after_saving_the_report(self):
        self.plan["targets"][0].update(status="resolve-failed", error="upstream duration changed")
        site, target = self.render()
        self.assertEqual(target["status"], "resolve-failed")
        self.assertEqual(target["failure_stage"], "plan")
        self.assertEqual(self.check(site, download_outcome="skipped"), 1)

    def test_missing_results_fail_after_saving_the_report(self):
        self.results_path.unlink()
        site, target = self.render()
        self.assertEqual(target["failure_stage"], "report")
        self.assertEqual(self.check(site), 1)

    def test_invalid_results_fail_after_saving_the_report(self):
        invalid = copy.deepcopy(self.results)
        invalid["run"]["id"] = "another-run"
        for contents in ("{invalid JSON", json.dumps(invalid)):
            with self.subTest(contents=contents[:30]):
                self.results_path.write_text(contents)
                site, target = self.render()
                self.assertEqual(target["failure_stage"], "report")
                self.assertEqual(self.check(site), 1)

    def test_unreadable_results_are_saved_as_a_report_failure(self):
        with mock.patch.object(report_all, "read_json", side_effect=PermissionError("cannot read results")):
            site, target = self.render()
        self.assertEqual(target["error"], "cannot read results")
        self.assertEqual(target["failure_stage"], "report")
        self.assertEqual(self.check(site), 1)

    def test_failed_download_is_not_hidden_by_usable_results(self):
        site, target = self.render()
        self.assertEqual(target["status"], "ok")
        self.assertEqual(self.check(site, download_outcome="failure"), 1)
        download = next(
            step for step in self.workflow["jobs"]["report"]["steps"] if step.get("id") == "download_results"
        )
        self.assertTrue(download["continue-on-error"])
        self.assertEqual(
            self.check_step["env"]["RESULTS_DOWNLOAD_OUTCOME"], "${{ steps.download_results.outcome }}"
        )

    def test_historical_results_do_not_hide_current_plan_or_report_errors(self):
        previous = self.previous_site()
        self.results_path.unlink()
        for status, stage in (("resolve-failed", "plan"), ("resolved", "report")):
            with self.subTest(stage=stage):
                self.plan["targets"][0]["status"] = status
                site, target = self.render(previous)
                self.assertTrue(target["report"])
                self.assertEqual(target["points_ok"], 4)
                self.assertEqual(target["failure_stage"], stage)
                self.assertEqual(self.check(site), 1)

    def test_unselected_history_does_not_repeat_an_old_failure(self):
        previous = self.previous_site()
        self.plan["targets"] = []
        site, target = self.render(previous)
        self.assertEqual(target["status"], "not-run")
        self.assertIsNone(target["failure_stage"])
        self.assertEqual(self.check(site, download_outcome="skipped"), 0)

    def test_outcome_check_follows_uploads_and_pages_accepts_a_failed_report_job(self):
        steps = self.workflow["jobs"]["report"]["steps"]
        for action in ("actions/upload-artifact@v4", "actions/upload-pages-artifact@v3"):
            upload = next(i for i, step in enumerate(steps) if step.get("uses") == action)
            self.assertLess(upload, len(steps) - 1)
        publish_condition = self.workflow["jobs"]["pages"]["if"]
        self.assertIn("!cancelled()", publish_condition)
        self.assertIn("needs.report.outputs.publish_ready == 'true'", publish_condition)
        self.assertNotIn("success()", publish_condition)
        self.assertNotIn("needs.report.result", publish_condition)


if __name__ == "__main__":
    unittest.main()
