# SPDX-License-Identifier: MIT
"""Partial measurements can publish a baseline and support later point comparisons."""

from __future__ import annotations

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tests.benchmarks.test_agentx_metrics import (
    REPO,
    analysis,
    legacy_results,
    report,
    report_all,
    sealed_selection,
    utils,
)

POINT_IDS = ["tp1:1", "tp1:2", "tp1:3"]


def formal_results(run_id="fixture-run"):
    results = utils.validated_results(legacy_results(run_id))
    for mode in (results, results["control"]):
        mode["run"].update(points_expected=3, attempt_id=f"{run_id}-attempt")
        template = mode["points"][0]
        points = []
        for conc in (1, 2, 3):
            point = copy.deepcopy(template)
            point.update(
                conc=conc,
                exp_name=f"tp1-c{conc}",
                serve_spec_sha256=utils.sha256_json({"arm": "tp1", "conc": conc}),
            )
            point["measurement"]["client_env"] = {"FRAMEWORK": "sglang", "CONC": str(conc)}
            point["measurement_sha256"] = utils.sha256_json(point["measurement"])
            points.append(point)
        mode["points"] = points
        mode["serving"]["arms"]["tp1"].update(
            concs=[1, 2, 3],
            arm_measurement_sha256=utils.sha256_json({p["conc"]: p["measurement"] for p in points}),
        )
    return use_selection(results, pareto_selection())


def pareto_selection():
    return sealed_selection(POINT_IDS)


def use_selection(results, selection):
    for mode in (results, results["control"]):
        mode["run"].update(point_policy=selection["policy"], selection=copy.deepcopy(selection))
    return results


def fail(point):
    point.update(status="failed", error="server failed before ready")


class AgentXBaselineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = self.root / "config.yaml"
        self.config.write_text("report:\n  repo_url: https://example.invalid/repo\n")
        self.relative = utils.target_path("fixture", "fixture-gpu", "sglang")
        self.site_count = 0
        # Git history is independent of baseline eligibility and metric comparisons.
        self.enterContext(mock.patch.object(report, "landed", return_value=None))
        self.enterContext(redirect_stdout(io.StringIO()))

    def publish(self, results, previous=None):
        self.site_count += 1
        run_dir = self.root / f"run-{self.site_count}"
        policy = results["run"]["point_policy"]
        target = {
            "model": results["model"],
            "target": results["target"],
            "framework": results["framework"],
            "status": "resolved",
            "serving_sha256": results["run"]["serving_sha256"],
            "point_policy": policy,
            "point_ids": POINT_IDS,
            "selection_manifest_sha256": results["run"]["selection"]["manifest_sha256"],
        }
        plan_path = utils.write_json(
            run_dir / "plan.json",
            {"schema": 4, "run_id": results["run"]["id"], "targets": [target]},
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
        self.assertEqual(result_path.read_bytes(), original)
        self.assertTrue((site / "index.html").is_file())
        self.assertEqual(utils.read_json(site / "manifest.json"), manifest)
        return site, manifest["targets"][0]

    def baseline(self, site, kind="pareto"):
        return utils.read_json(site / self.relative / f"baseline-{kind}.json")

    def test_any_successful_mode_promotes_without_requiring_a_successful_pair(self):
        for success in ("enabled", "disabled", "disabled-with-empty-enabled", "disjoint"):
            with self.subTest(success=success):
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
                site, target = self.publish(current)
                self.assertEqual(target["status"], "partial")
                self.assertEqual(target["points_ok"], 2 if success == "disjoint" else 1)
                self.assertEqual(target["points_total"], 6)
                self.assertIsNone(target["failure_stage"])
                self.assertEqual(target["baselines"]["pareto"], current["run"]["finished_at"])
                stored = self.baseline(site)
                self.assertEqual(stored, utils.validated_results(current))
                self.assertFalse(utils.complete_measurement(stored))
                self.assertNotEqual(analysis.compare_aklr(stored)["status"], "ok")

    def test_sites_from_before_frameworks_start_fresh_baselines(self):
        previous, _ = self.publish(formal_results("previous-run"))
        manifest = utils.read_json(previous / "manifest.json")
        utils.write_json(previous / "manifest.json", {**manifest, "schema": 3})
        current = formal_results("current-run")
        site, target = self.publish(current, previous)
        self.assertEqual(target["baselines"], {"pareto": current["run"]["finished_at"]})
        self.assertEqual(
            utils.read_json(site / self.relative / "report.json")["comparison"]["status"], "no-baseline"
        )

    def test_no_valid_success_preserves_the_previous_baseline(self):
        previous, _ = self.publish(formal_results("previous-run"))
        expected = self.baseline(previous)
        for outcome in ("failed", "empty", "invalid-throughput", "invalid-latency"):
            with self.subTest(outcome=outcome):
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
                site, target = self.publish(current, previous)
                self.assertEqual(target["status"], "failed")
                self.assertEqual(target["points_ok"], 0)
                self.assertIsNone(target["failure_stage"])
                self.assertEqual(self.baseline(site), expected)
        self.assertEqual(self.baseline(previous), expected)

    def test_all_failed_first_run_does_not_create_a_baseline(self):
        current = formal_results()
        for mode in (current, current["control"]):
            for point in mode["points"]:
                fail(point)
        site, target = self.publish(current)
        self.assertEqual(target["baselines"], {})
        self.assertFalse((site / self.relative / "baseline-pareto.json").exists())

    def test_partial_baseline_round_trips_and_compares_only_successful_points(self):
        prior = formal_results("previous-run")
        for mode in (prior, prior["control"]):
            fail(mode["points"][1])
        previous, _ = self.publish(prior)
        expected_previous = self.baseline(previous)
        current = formal_results("current-run")
        for mode in (current, current["control"]):
            fail(mode["points"][2])
        current["points"][0]["tput_per_gpu"] *= 1.25
        current["points"][0]["raw_data"]["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"] *= 1.25
        current["points"][0]["raw_data"]["request_metrics"]["latency"]["full_response_itl"]["p90"] /= 2
        site, target = self.publish(current, previous)
        self.assertEqual(target["status"], "partial")
        self.assertIsNone(target["failure_stage"])
        self.assertEqual(self.baseline(site)["run"]["id"], "current-run")
        self.assertEqual(self.baseline(previous), expected_previous)
        stored = self.baseline(site)
        self.assertEqual(stored["points"][2]["status"], "failed")
        self.assertEqual(stored["points"][2]["error"], "server failed before ready")
        comparison = utils.read_json(site / self.relative / "report.json")["comparison"]
        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["baseline_run"], "previous-run")
        arm = comparison["arms"]["tp1"]
        rows = {row["conc"]: row for row in arm["points"]}
        self.assertAlmostEqual(rows[1]["tput_ratio"], 1.25)
        self.assertAlmostEqual(rows[1]["intvty_ratio"], 2)
        self.assertAlmostEqual(arm["tput_geomean_ratio"], 1.25)
        for conc in (2, 3):
            self.assertIsNone(rows[conc].get("tput_ratio"))
            self.assertIsNone(rows[conc].get("intvty_ratio"))
        self.assertNotIn(
            "week-over-week comparison disabled", (site / self.relative / "index.html").read_text()
        )

    def test_interrupted_baseline_compares_matching_point_measurements(self):
        prior = formal_results("interrupted-run")
        for mode in (prior, prior["control"]):
            mode["points"] = mode["points"][:2]
            fail(mode["points"][1])
            mode["run"]["completed"] = False
            mode["serving"]["arms"]["tp1"].pop("arm_measurement_sha256")
        previous, _ = self.publish(prior)
        current = formal_results("current-run")
        current["points"][0]["tput_per_gpu"] *= 1.2
        site, _ = self.publish(current, previous)
        comparison = utils.read_json(site / self.relative / "report.json")["comparison"]
        self.assertEqual(comparison["status"], "ok")
        rows = {row["conc"]: row for row in comparison["arms"]["tp1"]["points"]}
        self.assertAlmostEqual(rows[1]["tput_ratio"], 1.2)
        self.assertIsNone(rows[2].get("tput_ratio"))
        self.assertIsNone(rows[3]["baseline"])
        self.assertIsNone(rows[3].get("tput_ratio"))

    def test_smoke_dirty_and_unverified_runs_do_not_replace_the_baseline(self):
        previous, _ = self.publish(formal_results("previous-run"))
        expected = self.baseline(previous)
        for field, value in (
            ("duration_override_s", 600),
            ("model_path_override", "/fixture/model"),
            ("arms_requested", "tp1"),
            ("points_requested", "tp1:1"),
            ("source_dirty", True),
            ("source_dirty", None),
            ("execution", {}),
        ):
            with self.subTest(field=field, value=value):
                current = formal_results("unqualified-run")
                for mode in (current, current["control"]):
                    mode["run"][field] = value
                site, target = self.publish(current, previous)
                self.assertIsNone(target["failure_stage"])
                self.assertEqual(self.baseline(site), expected)

    def test_changed_point_settings_compare_frontiers_instead_of_points(self):
        prior = formal_results("interrupted-run")
        for mode in (prior, prior["control"]):
            mode["points"] = mode["points"][:1]
            mode["run"]["completed"] = False
            mode["serving"]["arms"]["tp1"].pop("arm_measurement_sha256")
        previous, _ = self.publish(prior)
        current = formal_results("changed-settings")
        for mode in (current, current["control"]):
            point = mode["points"][0]
            point["measurement"]["client_env"]["DURATION"] = "30"
            point["measurement_sha256"] = utils.sha256_json(point["measurement"])
            mode["serving"]["arms"]["tp1"]["arm_measurement_sha256"] = utils.sha256_json(
                {p["conc"]: p["measurement"] for p in mode["points"]}
            )
        site, _ = self.publish(current, previous)
        comparison = utils.read_json(site / self.relative / "report.json")["comparison"]
        self.assertEqual(comparison["status"], "ok")
        arm = comparison["arms"]["tp1"]
        self.assertEqual(arm["status"], "curve")
        self.assertEqual(arm["changed_concs"], [1])
        self.assertEqual(arm["points"], [])
        self.assertAlmostEqual(arm["tput_geomean_ratio"], 1)

    def test_mismatched_pair_provenance_does_not_replace_the_baseline(self):
        previous, _ = self.publish(formal_results("previous-run"))
        expected = self.baseline(previous)
        for mismatch in ("source", "image", "mode-switch"):
            with self.subTest(mismatch=mismatch):
                current = formal_results("mismatched-run")
                control = current["control"]
                if mismatch == "source":
                    control["run"]["aklr_sha"] = "4" * 40
                elif mismatch == "image":
                    control["run"]["execution"]["image_id"] = "sha256:" + "4" * 64
                else:
                    control["points"][0]["measurement"]["launch"]["env"]["AKLR_DISABLE"] = "0"
                site, _ = self.publish(current, previous)
                self.assertEqual(self.baseline(site), expected)

    def test_control_points_must_belong_to_the_plan_without_duplicates(self):
        previous, _ = self.publish(formal_results("previous-run"))
        expected = self.baseline(previous)
        for invalid in ("unplanned", "duplicate", "expected-count"):
            with self.subTest(invalid=invalid):
                current = formal_results("invalid-control")
                control = current["control"]
                if invalid == "unplanned":
                    control["points"][1].update(conc=4, exp_name="tp1-c4")
                elif invalid == "duplicate":
                    control["points"][1] = copy.deepcopy(control["points"][0])
                else:
                    control["run"]["points_expected"] = 4
                site, target = self.publish(current, previous)
                self.assertEqual(target["status"], "failed")
                self.assertEqual(target["failure_stage"], "report")
                self.assertEqual(self.baseline(site), expected)
