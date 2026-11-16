# SPDX-License-Identifier: MIT
"""Compare successful matching points without requiring an entirely successful run."""

from __future__ import annotations

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tests.benchmarks.test_agentx_baselines import fail, formal_results
from tests.benchmarks.test_agentx_metrics import REPO, analysis, report, report_all, utils


def set_metrics(point, throughput, interactivity=20):
    point.update(tput_per_gpu=throughput, p90_full_response_intvty=interactivity)
    metrics = point["raw_data"]["request_metrics"]
    metrics["throughput"]["per_gpu"]["total_tput_tps"] = throughput
    metrics["latency"]["full_response_itl"]["p90"] = 1 / interactivity


def change_settings(point, change):
    if change == "missing-measurement":
        point.pop("measurement")
    elif change == "serve-spec":
        point["serve_spec_sha256"] = "9" * 64
    elif change == "duration":
        point["measurement"]["client_env"]["DURATION"] = "30"
    elif change == "environment":
        point["measurement"]["launch"]["env"]["SGLANG_USE_AITER"] = "0"
    elif change == "trace":
        point["measurement"]["trace"]["revision"] = "9" * 40
    elif change == "invalid-trace":
        point["measurement"]["trace"]["revision"] = "unverified"
    elif change == "mode-switch":
        point["measurement"]["launch"]["env"]["AKLR_DISABLE"] = "unexpected"
    else:
        raise AssertionError(f"unknown fixture change: {change}")


def refresh_hashes(results):
    for point in results["points"]:
        point["measurement_sha256"] = utils.sha256_json(point.get("measurement", {}))
    for arm, spec in results["serving"]["arms"].items():
        points = [point for point in results["points"] if point["arm"] == arm]
        spec["arm_serve_sha256"] = utils.sha256_json(
            {point["conc"]: point["serve_spec_sha256"] for point in points}
        )
        spec["arm_measurement_sha256"] = utils.sha256_json(
            {point["conc"]: point.get("measurement", {}) for point in points}
        )


class AgentXPartialPairTests(unittest.TestCase):
    def test_failed_points_on_different_sides_leave_the_successful_intersection(self):
        results = formal_results()
        set_metrics(results["points"][0], 200, 40)
        set_metrics(results["control"]["points"][0], 100, 20)
        fail(results["points"][1])
        results["points"][1].pop("measurement")
        fail(results["control"]["points"][2])
        results["control"]["points"][2].pop("measurement")
        original = copy.deepcopy(results)

        comparison = analysis.compare_aklr(results)

        self.assertEqual(results, original)
        self.assertEqual(comparison["status"], "ok")
        self.assertEqual([row["conc"] for row in comparison["points"]], [1])
        self.assertEqual(comparison["points"][0]["tput_ratio"], 2)
        self.assertEqual(comparison["points"][0]["intvty_ratio"], 2)
        self.assertAlmostEqual(comparison["tput_geomean_ratio"], 2)

    def test_unfinished_modes_with_different_recorded_points_can_compare(self):
        results = formal_results()
        results["points"] = results["points"][:2]
        results["control"]["points"] = results["control"]["points"][::2]
        for mode in (results, results["control"]):
            mode["run"]["completed"] = False
            mode["serving"]["arms"]["tp1"].pop("arm_measurement_sha256")

        comparison = analysis.compare_aklr(results)

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual([row["conc"] for row in comparison["points"]], [1])

    def test_point_settings_failures_exclude_only_the_affected_pair(self):
        for change in (
            "duration",
            "environment",
            "trace",
            "invalid-trace",
            "missing-measurement",
            "serve-spec",
            "mode-switch",
        ):
            with self.subTest(change=change):
                results = formal_results()
                for mode in (results, results["control"]):
                    for point in mode["points"]:
                        set_metrics(point, 100)
                set_metrics(results["points"][0], 200)
                set_metrics(results["points"][1], 100_000)
                set_metrics(results["points"][2], 800)
                change_settings(results["control"]["points"][1], change)

                comparison = analysis.compare_aklr(results)

                self.assertEqual(comparison["status"], "ok")
                self.assertEqual([row["conc"] for row in comparison["points"]], [1, 3])
                self.assertAlmostEqual(comparison["tput_geomean_ratio"], 4)

    def test_invalid_metrics_are_excluded_even_when_marked_successful(self):
        for metric, value in (
            (analysis.Y, 0),
            (analysis.Y, float("nan")),
            (analysis.X, True),
        ):
            with self.subTest(metric=metric, value=value):
                results = formal_results()
                results["points"][1][metric] = value
                comparison = analysis.compare_aklr(results)
                self.assertEqual(comparison["status"], "ok")
                self.assertEqual([row["conc"] for row in comparison["points"]], [1, 3])

    def test_disjoint_successful_points_have_no_paired_gain(self):
        results = formal_results()
        fail(results["points"][1])
        fail(results["points"][2])
        fail(results["control"]["points"][0])
        comparison = analysis.compare_aklr(results)
        self.assertEqual(comparison["status"], "unavailable")
        self.assertEqual(comparison["points"], [])
        self.assertIsNone(comparison["tput_geomean_ratio"])

    def test_equal_concurrency_in_different_arms_does_not_form_a_pair(self):
        results = formal_results()
        for mode in (results, results["control"]):
            mode["run"].update(points_expected=4, completed=False)
            mode["serving"]["arms"]["other-arm"] = {
                "concs": [1],
                "arm_serve_sha256": "9" * 64,
            }
        results["control"]["points"][0].update(arm="other-arm", exp_name="other-arm-c1")
        comparison = analysis.compare_aklr(results)
        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(
            [(row["arm"], row["conc"]) for row in comparison["points"]], [("tp1", 2), ("tp1", 3)]
        )

    def test_invalid_record_counts_and_duplicates_do_not_become_partial_pairs(self):
        for invalid in ("duplicate", "too-many", "invalid-expected", "different-expected"):
            with self.subTest(invalid=invalid):
                results = formal_results()
                control = results["control"]
                if invalid == "duplicate":
                    control["points"][1] = copy.deepcopy(control["points"][0])
                elif invalid == "too-many":
                    control["run"]["points_expected"] = 2
                elif invalid == "invalid-expected":
                    control["run"]["points_expected"] = True
                else:
                    control["run"]["points_expected"] = 4
                comparison = analysis.compare_aklr(results)
                self.assertEqual(comparison["status"], "unavailable")
                self.assertEqual(comparison["points"], [])

    def test_partial_pairing_keeps_run_identity_guards(self):
        for path, value in (
            (("model", "id"), "another-model"),
            (("target", "id"), "another-target"),
            (("run", "id"), "another-run"),
            (("run", "attempt_id"), "another-attempt"),
            (("run", "aklr_sha"), "9" * 40),
            (("run", "serving_sha256"), "9" * 64),
            (("run", "execution", "image_id"), "sha256:" + "9" * 64),
            (("run", "client_packages", "agentx-harness"), "different-version"),
            (("run", "inferencex", "sha"), "9" * 40),
            (("run", "aklr_mode"), "enabled"),
        ):
            with self.subTest(path=path):
                results = formal_results()
                fail(results["points"][2])
                target = results["control"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                comparison = analysis.compare_aklr(results)
                self.assertEqual(comparison["status"], "unavailable")
                self.assertEqual(comparison["points"], [])

    def test_report_pair_filter_does_not_relax_baseline_promotion(self):
        results = formal_results()
        self.assertTrue(report_all.eligible(results))
        change_settings(results["control"]["points"][1], "duration")
        self.assertEqual(analysis.compare_aklr(results)["status"], "ok")
        self.assertFalse(report_all.eligible(results))


class AgentXPartialWeeklyTests(unittest.TestCase):
    def test_failed_record_metadata_does_not_block_other_successful_points(self):
        current = formal_results("current")
        baseline = formal_results("previous")
        set_metrics(current["points"][0], 150, 40)
        set_metrics(baseline["points"][0], 100, 20)
        fail(current["points"][1])
        current["points"][1].pop("measurement")
        fail(baseline["points"][2])
        change_settings(baseline["points"][2], "invalid-trace")

        comparison = analysis.compare(current, baseline)

        self.assertEqual(comparison["status"], "ok")
        arm = comparison["arms"]["tp1"]
        self.assertEqual(arm["status"], "comparable")
        rows = {row["conc"]: row for row in arm["points"]}
        self.assertEqual(rows[1]["tput_ratio"], 1.5)
        self.assertEqual(rows[1]["intvty_ratio"], 2)
        self.assertAlmostEqual(arm["tput_geomean_ratio"], 1.5)
        for conc in (2, 3):
            self.assertIsNone(rows[conc].get("tput_ratio"))

    def test_one_changed_point_does_not_disable_the_whole_arm(self):
        for change in (
            "duration",
            "environment",
            "trace",
            "invalid-trace",
            "missing-measurement",
            "serve-spec",
        ):
            with self.subTest(change=change):
                current = formal_results("current")
                baseline = formal_results("previous")
                for point in baseline["points"]:
                    set_metrics(point, 100)
                set_metrics(current["points"][0], 150)
                set_metrics(current["points"][1], 100_000)
                set_metrics(current["points"][2], 600)
                change_settings(current["points"][1], change)
                refresh_hashes(current)
                refresh_hashes(baseline)
                original = copy.deepcopy(current)

                comparison = analysis.compare(current, baseline)

                self.assertEqual(current, original)
                self.assertEqual(comparison["status"], "ok")
                arm = comparison["arms"]["tp1"]
                self.assertEqual(arm["status"], "comparable")
                rows = {row["conc"]: row for row in arm["points"]}
                self.assertEqual(rows[1]["tput_ratio"], 1.5)
                self.assertIsNone(rows[2].get("tput_ratio"))
                self.assertTrue(rows[2].get("reason"))
                self.assertEqual(rows[3]["tput_ratio"], 6)
                self.assertAlmostEqual(arm["tput_geomean_ratio"], 3)

    def test_changed_grid_still_compares_unchanged_shared_points(self):
        current = formal_results("current")
        baseline = formal_results("previous")
        for mode in (baseline, baseline["control"]):
            mode["run"]["points_expected"] = 2
            mode["points"] = mode["points"][:2]
            mode["serving"]["arms"]["tp1"]["concs"] = [1, 2]
            mode["serving"]["grid_sha256"] = "9" * 64
            refresh_hashes(mode)
        refresh_hashes(current)

        comparison = analysis.compare(current, baseline)

        self.assertEqual(comparison["status"], "ok")
        arm = comparison["arms"]["tp1"]
        self.assertEqual(arm["status"], "comparable")
        rows = {row["conc"]: row for row in arm["points"]}
        self.assertEqual(rows[1]["tput_ratio"], 1)
        self.assertEqual(rows[2]["tput_ratio"], 1)
        self.assertIsNone(rows[3].get("tput_ratio"))

    def test_global_identity_changes_still_disable_weekly_comparison(self):
        for path, value in (
            (("run", "source_dirty"), True),
            (("run", "execution", "image_id"), "sha256:" + "9" * 64),
            (("run", "server_packages", "sglang"), "different-version"),
            (("run", "inferencex", "sha"), "9" * 40),
        ):
            with self.subTest(path=path):
                current = formal_results("current")
                baseline = formal_results("previous")
                target = current
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                comparison = analysis.compare(current, baseline)
                self.assertNotEqual(comparison["status"], "ok")
                self.assertEqual(comparison["arms"], {})


class AgentXPartialComparisonReportTests(unittest.TestCase):
    def test_previous_frontier_contains_only_matching_successful_points(self):
        current = formal_results("current")
        baseline = formal_results("previous")
        fail(current["points"][1])
        change_settings(current["points"][2], "duration")
        comparison = analysis.compare(current, baseline)

        with mock.patch.object(report, "svg_chart", return_value="") as chart:
            report.charts(current, baseline, comparison, "en")

        series = chart.call_args.args[1]
        previous = [item for item in series if item["color"] == report.BASELINE_COLOR]
        self.assertEqual(len(previous), 1)
        self.assertEqual(previous[0]["frontier"], [baseline["points"][0]])
        self.assertEqual(series[0]["points"], current["control"]["points"])
        self.assertEqual(series[1]["points"], [current["points"][0], current["points"][2]])

        fail(current["points"][0])
        comparison = analysis.compare(current, baseline)
        with mock.patch.object(report, "svg_chart", return_value="") as chart:
            report.charts(current, baseline, comparison, "en")
        self.assertFalse(any(item["color"] == report.BASELINE_COLOR for item in chart.call_args.args[1]))

    def test_partial_report_exports_gains_and_drops_the_all_success_requirement(self):
        current = formal_results("current")
        baseline = formal_results("previous")
        fail(current["points"][1])
        fail(current["control"]["points"][2])
        fail(baseline["points"][2])
        set_metrics(current["points"][0], 150)
        set_metrics(current["control"]["points"][0], 100)
        set_metrics(baseline["points"][0], 100)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(report, "landed", return_value=None),
            redirect_stdout(io.StringIO()),
        ):
            output = Path(directory)
            report.write_report(current, baseline, output, "https://example.invalid/repo", REPO)
            summary = utils.read_json(output / "report.json")
            html = (output / "index.html").read_text()

        pairing = summary["aklr_comparison"]
        self.assertEqual(pairing["status"], "ok")
        self.assertEqual(len(pairing["points"]), 1)
        self.assertAlmostEqual(pairing["tput_geomean_ratio"], 1.5)
        self.assertAlmostEqual(summary["comparison"]["arms"]["tp1"]["tput_geomean_ratio"], 1.5)
        self.assertIn("+50.0%", html)
        self.assertIn("1 matched pairs", html)
        self.assertIn("1 对匹配测量点", html)
        for obsolete in (
            "both modes must complete with matching settings",
            "Both AKLR modes must finish every selected point successfully",
            "week-over-week comparison disabled",
            "两种模式需完整成功且配置匹配",
            "两种 AKLR 模式都需成功完成全部选中数据点",
            "测量未完成或有失败点；不计算周环比",
        ):
            self.assertNotIn(obsolete, html)


if __name__ == "__main__":
    unittest.main()
