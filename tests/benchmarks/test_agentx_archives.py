# SPDX-License-Identifier: MIT
"""Keep measurement evidence intact while rendering verified report views."""

from __future__ import annotations

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from html import escape
from pathlib import Path
from unittest import mock

from tests.benchmarks.test_agentx_baselines import formal_results
from tests.benchmarks.test_agentx_metrics import REPO, legacy_results, report, report_all, utils


class AgentXArchiveTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.relative = utils.target_path("fixture", "fixture-gpu", "sglang")
        self.serial = 0
        self.enterContext(mock.patch.object(report, "landed", return_value=None))
        self.enterContext(redirect_stdout(io.StringIO()))

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

    def test_historical_results_and_baselines_survive_failed_or_partial_normalization(self):
        for missing in ((1, 2, 3), (2,)):
            with self.subTest(missing=missing):
                original = formal_results("previous-run")
                for mode in (original, original["control"]):
                    for point in mode["points"]:
                        point.pop("p90_full_response_intvty")
                        if point["conc"] in missing:
                            point.pop("raw_data")
                previous = self.previous(original, baseline=original)
                before = (previous / self.relative / "results.json").read_bytes()

                for _ in range(2):
                    site, target = self.publish(previous=previous, selected=False)
                    self.assertEqual(utils.read_json(site / self.relative / "results.json"), original)
                    self.assertEqual(utils.read_json(site / self.relative / "baseline-pareto.json"), original)
                    summary = utils.read_json(site / self.relative / "report.json")
                    pairs = summary["aklr_comparison"]["points"]
                    self.assertEqual(
                        [row["conc"] for row in pairs], [c for c in (1, 2, 3) if c not in missing]
                    )
                    self.assertEqual(target["points_ok"], 2 * (3 - len(missing)))
                    self.assertEqual("pareto" in target["baseline_unavailable"], len(missing) == 3)
                    self.assertEqual((previous / self.relative / "results.json").read_bytes(), before)
                    previous = site

    def test_unverified_full_response_only_baseline_is_preserved_but_unavailable(self):
        original = formal_results("previous-run")
        for mode in (original, original["control"]):
            for point in mode["points"]:
                point.pop("raw_data")
                point.pop("p90_e2e_norm_intvty")
        previous = self.previous(original, baseline=original)

        site, target = self.publish(previous=previous, selected=False)

        self.assertEqual(utils.read_json(site / self.relative / "baseline-pareto.json"), original)
        self.assertEqual(utils.read_json(site / self.relative / "results.json"), original)
        self.assertIn("pareto", target["baseline_unavailable"])
        self.assertNotIn("pareto", target["baselines"])
        self.assertEqual(target["points_ok"], 0)
        summary = utils.read_json(site / self.relative / "report.json")
        self.assertEqual(summary["frontiers"]["combined"], [])
        self.assertEqual(summary["aklr_comparison"]["status"], "unavailable")

    def test_current_results_archive_and_baseline_retain_unverifiable_point_evidence(self):
        current = formal_results("current-run")
        for mode in (current, current["control"]):
            mode["points"][1].pop("raw_data")
        original = copy.deepcopy(current)

        site, target = self.publish(current=current)

        self.assertEqual(current, original)
        self.assertEqual(utils.read_json(site / self.relative / "results.json"), original)
        self.assertEqual(utils.read_json(site / self.relative / "baseline-pareto.json"), original)
        self.assertEqual(target["status"], "partial")
        self.assertEqual(target["points_ok"], 4)
        summary = utils.read_json(site / self.relative / "report.json")
        self.assertEqual([row["conc"] for row in summary["aklr_comparison"]["points"]], [1, 3])

    def test_single_result_entry_can_rerender_without_rewriting_the_archive(self):
        original = legacy_results("previous-run")
        source = utils.write_json(self.root / "input.json", original)
        config = self.root / "config.yaml"
        config.write_text("report:\n  repo_url: https://example.invalid/repo\n")

        for _ in range(2):
            out = self.directory("single")
            self.assertEqual(
                report.main(["--results", str(source), "--out", str(out), "--config", str(config)]),
                0,
            )
            self.assertEqual(utils.read_json(out / "results.json"), original)
            summary = utils.read_json(out / "report.json")
            point = summary["frontiers"]["combined"][0]
            self.assertAlmostEqual(point["p90_full_response_intvty"], 1 / 0.00351)
            source = out / "results.json"

    def test_historical_single_mode_results_remain_readable(self):
        original = legacy_results("previous-run")
        original.pop("control")
        previous = self.previous(original, baseline=original)

        site, target = self.publish(previous=previous, selected=False)

        self.assertTrue(target["report"])
        self.assertIsNone(target["failure_stage"])
        self.assertEqual(target["points_ok"], 1)
        self.assertEqual(utils.read_json(site / self.relative / "results.json"), original)
        self.assertEqual(utils.read_json(site / self.relative / "baseline-pareto.json"), original)

    def test_current_failure_is_visible_when_rendering_a_historical_target(self):
        previous = self.previous(legacy_results("previous-run"))
        error = "recipe <weights> could not resolve & no measurement ran"

        site, target = self.publish(previous=previous, error=error)

        self.assertEqual(target["failure_stage"], "plan")
        self.assertTrue(target["report"])
        for path in (site / "index.html", site / self.relative / "index.html"):
            html = path.read_text()
            self.assertIn(escape(error), html)
            self.assertNotIn(error, html)


if __name__ == "__main__":
    unittest.main()
