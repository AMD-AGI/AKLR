# SPDX-License-Identifier: MIT
"""InferenceX CSV metric compatibility and offline report migration."""

from __future__ import annotations

import copy
import importlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
with mock.patch.object(sys, "path", [str(REPO / "benchmarks/agentx"), *sys.path]):
    utils = importlib.import_module("agentx_ci.utils")
    runtime = importlib.import_module("agentx_ci.runtime")
    analysis = importlib.import_module("agentx_ci.performance_analysis")
    report = importlib.import_module("agentx_ci.report")
    report_all = importlib.import_module("agentx_ci.report_all")
    recipe_resolve = importlib.import_module("agentx_ci.recipe_resolve")

FULL_RESPONSE = "p90_full_response_intvty"
E2E = "p90_e2e_norm_intvty"


def aggregate(itl=0.00351, *, e2e=84.72682, throughput=3390.56736):
    return {
        "num_requests_successful": 73,
        "request_metrics": {
            "latency": {
                "full_response_itl": {"p90": itl},
                "full_response_intvty": {"p90": 285.10342},
                "e2e_norm_intvty": {"p90": e2e},
            },
            "throughput": {"per_gpu": {"total_tput_tps": throughput}},
        },
    }


def legacy_results(run_id="fixture-run", *, itl=0.00351):
    """A complete old-format run with verified, paired execution identities."""
    image = "example.invalid/server@sha256:" + "a" * 64
    result = {
        "schema": 3,
        "model": {"id": "fixture", "display_name": "Fixture"},
        "target": {"id": "fixture-gpu"},
        "run": {
            "id": run_id,
            "attempt_id": "fixture-attempt",
            "mode": "live",
            "aklr_mode": "enabled",
            "aklr_sha": "b" * 40,
            "source_dirty": False,
            "completed": True,
            "points_expected": 1,
            "serving_sha256": "c" * 64,
            "finished_at": "2026-11-09T19:35:00Z",
            "host": "fixture-host",
            "arms_requested": "all",
            "point_policy": "all",
            "duration_override_s": None,
            "image": {"run": image, "inferencex": image},
            "execution": {"image_reference": image, "image_id": "sha256:" + "d" * 64},
            "expected_hardware": {"gpu_count": 1, "gpu_name_pattern": "Fixture", "gpu_arch": "gfx950"},
            "hardware": {
                "devices": [{"name": "Fixture", "arch": "gfx950", "total_memory": 1024}],
                "peer_access": [[True]],
            },
            "client_packages": {"agentx-harness": "1.0.13"},
            "server_packages": {"sglang": "0.5.22"},
            "inferencex": {"repo": "example/source", "sha": "e" * 40, "config_key": "fixture"},
        },
        "serving": {
            "grid_sha256": "f" * 64,
            "arms": {"tp1": {"concs": [1], "arm_serve_sha256": "1" * 64}},
        },
        "points": [
            {
                "arm": "tp1",
                "conc": 1,
                "exp_name": "tp1-c1",
                "serve_spec_sha256": "2" * 64,
                "status": "ok",
                "error": None,
                "tput_per_gpu": 3390.56736,
                E2E: 84.72682,
                "raw_data": aggregate(itl),
                "measurement": {
                    "trace": {"repo": "example/traces", "loader": "fixture", "revision": "3" * 40},
                    "launch": {"env": {"AKLR_SGLANG_PLUGIN": "1", "AKLR_DISABLE": "0"}},
                },
            }
        ],
    }
    control = copy.deepcopy(result)
    control["run"]["aklr_mode"] = "disabled"
    point = control["points"][0]
    point.update(
        tput_per_gpu=3372.91456,
        p90_e2e_norm_intvty=155.05189,
        raw_data=aggregate(0.00359, e2e=155.05189, throughput=3372.91456),
    )
    point["measurement"]["launch"]["env"] = {"AKLR_SGLANG_PLUGIN": "0", "AKLR_DISABLE": "1"}
    result["control"] = control
    return result


class AgentXMetricTests(unittest.TestCase):
    def test_runtime_matches_csv_reciprocal_instead_of_other_interactivity_fields(self):
        value = runtime.metrics_of(aggregate())
        self.assertEqual(value[FULL_RESPONSE], 1 / 0.00351)
        self.assertNotEqual(value[FULL_RESPONSE], 285.10342)
        self.assertNotEqual(value[FULL_RESPONSE], 84.72682)
        self.assertEqual(value["tput_per_gpu"], 3390.56736)
        self.assertEqual(value["num_requests_successful"], 73)

    def test_bad_or_missing_itl_never_falls_back_to_other_metrics(self):
        for itl in (0, -1, True, False, None, "0.00351", float("nan"), float("inf"), 5e-324, 10**400):
            with self.subTest(itl=itl), self.assertRaises(ValueError):
                runtime.metrics_of(aggregate(itl))
        missing = aggregate()
        del missing["request_metrics"]["latency"]["full_response_itl"]
        with self.assertRaisesRegex(ValueError, "Missing full-response P90 ITL"):
            runtime.metrics_of(missing)

    def test_legacy_modes_recompute_without_changing_raw_data_or_source_identity(self):
        source = legacy_results()
        source["points"][0][FULL_RESPONSE] = 999
        failed = {"arm": "tp1", "conc": 2, "status": "failed", "error": "server failed before ready"}
        source["points"].append(failed)
        original = copy.deepcopy(source)
        migrated = utils.validated_results(source)
        self.assertEqual(source, original)
        self.assertEqual(migrated["run"], original["run"])
        self.assertEqual(migrated["points"][0][FULL_RESPONSE], 1 / 0.00351)
        self.assertEqual(migrated["control"]["points"][0][FULL_RESPONSE], 1 / 0.00359)
        for before, after in ((source, migrated), (source["control"], migrated["control"])):
            self.assertEqual(after["points"][0]["raw_data"], before["points"][0]["raw_data"])
            self.assertEqual(after["points"][0][E2E], before["points"][0][E2E])
            self.assertEqual(after["points"][0]["tput_per_gpu"], before["points"][0]["tput_per_gpu"])
        self.assertEqual(migrated["points"][1], failed)
        self.assertEqual(utils.validated_results(migrated), migrated)

    def test_missing_raw_e2e_only_points_and_invalid_raw_with_cached_values_are_unavailable(self):
        for invalid_raw in (None, aggregate(0)):
            with self.subTest(invalid_raw=invalid_raw):
                source = legacy_results()
                point = source["points"][0]
                if invalid_raw is None:
                    del point["raw_data"]
                else:
                    point.update(raw_data=invalid_raw, p90_full_response_intvty=999)
                migrated = utils.validated_results(source)
                self.assertEqual(migrated["points"][0]["status"], "failed")
                self.assertNotIn(FULL_RESPONSE, migrated["points"][0])
                self.assertIn("full-response P90 ITL", migrated["points"][0]["error"])
                self.assertEqual(analysis.measured(migrated), [])

    def test_cached_interactivity_does_not_replace_raw_itl_evidence(self):
        source = utils.validated_results(legacy_results())
        for mode, cached in ((source, 999), (source["control"], 1)):
            point = mode["points"][0]
            point.pop("raw_data")
            point[FULL_RESPONSE] = cached
        original = copy.deepcopy(source)

        normalized = utils.validated_results(source)

        self.assertEqual(source, original)
        for mode in (normalized, normalized["control"]):
            point = mode["points"][0]
            self.assertEqual(point["status"], "failed")
            self.assertIn("Missing full-response P90 ITL", point["error"])
            self.assertNotIn(FULL_RESPONSE, point)
            self.assertEqual(analysis.measured(mode), [])
        self.assertEqual(analysis.compare_aklr(normalized)["status"], "unavailable")
        self.assertFalse(report_all.eligible(normalized))
        self.assertFalse(report_all.eligible(normalized, historical=True))

    def test_pareto_report_axis_and_published_upstream_selection_axis_are_independent(self):
        fast = {FULL_RESPONSE: 300, E2E: 10, "tput_per_gpu": 100}
        slow = {FULL_RESPONSE: 200, E2E: 20, "tput_per_gpu": 90}
        self.assertEqual(analysis.frontier([fast, slow]), [fast])
        self.assertEqual(recipe_resolve.X, "p90_e2e_norm_intvty")
        self.assertEqual(recipe_resolve.frontier([fast, slow], x_axis=recipe_resolve.X), [fast, slow])

    def test_paired_and_weekly_ratios_use_full_response_after_migration(self):
        current = utils.validated_results(legacy_results())
        previous = legacy_results("previous-run", itl=0.00702)
        previous["points"][0][E2E] *= 2
        previous = utils.validated_results(previous)
        paired = analysis.compare_aklr(current)
        self.assertEqual(paired["status"], "ok")
        self.assertAlmostEqual(paired["points"][0]["intvty_ratio"], 0.00359 / 0.00351)
        weekly = analysis.compare(current, previous)
        self.assertEqual(weekly["status"], "ok")
        self.assertAlmostEqual(weekly["arms"]["tp1"]["points"][0]["intvty_ratio"], 2)


class AgentXReportMetricMigrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.relative = utils.target_path("fixture", "fixture-gpu")
        self.config = self.root / "config.yaml"
        self.config.write_text("report:\n  repo_url: https://example.invalid/repo\n")
        # Git history is independent of report metrics; keep this suite offline.
        self.enterContext(mock.patch.object(report, "landed", return_value=None))
        self.enterContext(redirect_stdout(io.StringIO()))

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
        self.assertEqual(baseline_path.read_bytes(), original_bytes)
        self.assertTrue((out / "index.html").is_file())
        self.assertIsNone(manifest["targets"][0]["failure_stage"])
        self.assertEqual(manifest["targets"][0]["status"], expected_status)
        return out, manifest["targets"][0]

    def test_report_cli_redraws_old_results_with_new_coordinates_and_labels(self):
        source = legacy_results()
        source_path = utils.write_json(self.root / "results.json", source)
        original_bytes = source_path.read_bytes()
        out = self.root / "site"
        code = report.main(["--results", str(source_path), "--out", str(out), "--config", str(self.config)])
        self.assertEqual(code, 0)
        self.assertEqual(source_path.read_bytes(), original_bytes)
        result = utils.read_json(out / "results.json")
        summary = utils.read_json(out / "report.json")
        self.assertEqual(result, source)
        self.assertEqual(summary["axes"], {"x": FULL_RESPONSE, "y": "tput_per_gpu"})
        self.assertEqual(summary["frontiers"]["combined"][0][FULL_RESPONSE], 1 / 0.00351)
        page = (out / "index.html").read_text()
        self.assertIn("P90 Interactivity (tok/s/user): 284.9", page)
        self.assertIn("Throughput/Chip (tok/s): 3390.6", page)
        self.assertIn("1 / P90(full-response ITL)", page)
        self.assertNotIn("P90 E2E interactivity", page)
        self.assertNotIn("Off P90 E2E tok/s/user", page)

    def test_historical_baseline_with_raw_is_normalized_for_comparison_without_rewriting_archive(self):
        baseline = legacy_results("previous-run", itl=0.00702)
        original = copy.deepcopy(baseline)
        out, target = self.build_with_baseline(baseline)
        stored = utils.read_json(out / self.relative / "baseline-live.json")
        summary = utils.read_json(out / self.relative / "report.json")
        self.assertEqual(baseline, original)
        self.assertEqual(stored, original)
        self.assertEqual(summary["comparison"]["status"], "ok")
        self.assertAlmostEqual(summary["comparison"]["arms"]["tp1"]["points"][0]["intvty_ratio"], 2)
        self.assertIn("live", target["baselines"])
        self.assertFalse(target["baseline_unavailable"])

    def test_legacy_baseline_without_raw_is_preserved_without_mixing_axes(self):
        baseline = legacy_results("previous-run")
        for result in (baseline, baseline["control"]):
            del result["points"][0]["raw_data"]
        out, target = self.build_with_baseline(baseline)
        self.assertEqual(utils.read_json(out / self.relative / "baseline-live.json"), baseline)
        summary = utils.read_json(out / self.relative / "report.json")
        self.assertNotEqual(summary["comparison"]["status"], "ok")
        self.assertEqual(summary["comparison"]["arms"], {})
        self.assertNotIn("live", target["baselines"])
        self.assertIn("live", target["baseline_unavailable"])
        self.assertIn("full-response P90 unavailable", (out / "index.html").read_text())
        self.assertIn("historical baseline is preserved", (out / self.relative / "index.html").read_text())

    def test_new_formal_run_can_replace_an_unconvertible_historical_baseline(self):
        baseline = legacy_results("previous-run")
        for result in (baseline, baseline["control"]):
            del result["points"][0]["raw_data"]
        out, target = self.build_with_baseline(baseline, formal=True)
        stored = utils.read_json(out / self.relative / "baseline-live.json")
        self.assertEqual(stored["run"]["id"], "fixture-run")
        self.assertEqual(stored, legacy_results())
        summary = utils.read_json(out / self.relative / "report.json")
        self.assertEqual(summary["frontiers"]["combined"][0][FULL_RESPONSE], 1 / 0.00351)
        self.assertEqual(summary["disabled_frontiers"]["combined"][0][FULL_RESPONSE], 1 / 0.00359)
        self.assertIn("live", target["baselines"])
        self.assertFalse(target["baseline_unavailable"])

    def test_unverifiable_cached_points_neither_compare_nor_replace_the_baseline(self):
        current = utils.validated_results(legacy_results())
        for mode, cached in ((current, 999), (current["control"], 1)):
            mode["points"][0].pop("raw_data")
            mode["points"][0][FULL_RESPONSE] = cached
        baseline = legacy_results("previous-run")

        out, target = self.build_with_baseline(
            baseline, current=current, formal=True, expected_status="failed"
        )

        self.assertEqual((target["points_ok"], target["points_total"]), (0, 2))
        self.assertEqual(utils.read_json(out / self.relative / "baseline-live.json"), baseline)
        self.assertEqual(utils.read_json(out / self.relative / "results.json"), current)
        summary = utils.read_json(out / self.relative / "report.json")
        self.assertEqual(summary["comparison"]["arms"], {})
        self.assertEqual(summary["aklr_comparison"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
