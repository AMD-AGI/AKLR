# SPDX-License-Identifier: MIT
"""Compare successful matching points without requiring an entirely successful run."""

from __future__ import annotations

import copy

import pytest
from agentx_ci import performance_analysis as analysis
from agentx_ci import report, report_all, utils

from tests.benchmarks.agentx_results import REPO, fail, formal_results


def set_metrics(point, throughput, interactivity=20):
    point.update(tput_per_gpu=throughput, p90_full_response_intvty=interactivity)
    metrics = point["raw_data"]["request_metrics"]
    metrics["throughput"]["per_gpu"]["total_tput_tps"] = throughput
    metrics["latency"]["full_response_itl"]["p90"] = 1 / interactivity


@pytest.fixture
def charted(monkeypatch: pytest.MonkeyPatch) -> list[list[dict]]:
    """The series of each chart report draws, in order."""
    drawn = []
    monkeypatch.setattr(
        report, "svg_chart", lambda _title, series, *_args, **_kwargs: drawn.append(series) or ""
    )
    return drawn


def slim(points):
    return [analysis.frontier_point(point) for point in analysis.frontier(points)]


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


def test_failed_points_on_different_sides_leave_the_successful_intersection():
    results = formal_results()
    set_metrics(results["points"][0], 200, 40)
    set_metrics(results["control"]["points"][0], 100, 20)
    fail(results["points"][1])
    results["points"][1].pop("measurement")
    fail(results["control"]["points"][2])
    results["control"]["points"][2].pop("measurement")
    original = copy.deepcopy(results)

    comparison = analysis.compare_aklr(results)

    assert results == original
    assert comparison["status"] == "ok"
    assert [row["conc"] for row in comparison["points"]] == [1]
    assert comparison["points"][0]["tput_ratio"] == 2
    assert comparison["points"][0]["intvty_ratio"] == 2
    assert comparison["tput_geomean_ratio"] == pytest.approx(2, abs=5e-8)


def test_unfinished_modes_with_different_recorded_points_can_compare():
    results = formal_results()
    results["points"] = results["points"][:2]
    results["control"]["points"] = results["control"]["points"][::2]
    for mode in (results, results["control"]):
        mode["run"]["completed"] = False
        mode["serving"]["arms"]["tp1"].pop("arm_measurement_sha256")

    comparison = analysis.compare_aklr(results)

    assert comparison["status"] == "ok"
    assert [row["conc"] for row in comparison["points"]] == [1]


@pytest.mark.parametrize(
    "change",
    [
        "duration",
        "environment",
        "trace",
        "invalid-trace",
        "missing-measurement",
        "serve-spec",
        "mode-switch",
    ],
)
def test_point_settings_failures_exclude_only_the_affected_pair(change):
    results = formal_results()
    for mode in (results, results["control"]):
        for point in mode["points"]:
            set_metrics(point, 100)
    set_metrics(results["points"][0], 200)
    set_metrics(results["points"][1], 100_000)
    set_metrics(results["points"][2], 800)
    change_settings(results["control"]["points"][1], change)

    comparison = analysis.compare_aklr(results)

    assert comparison["status"] == "ok"
    assert [row["conc"] for row in comparison["points"]] == [1, 3]
    assert comparison["tput_geomean_ratio"] == pytest.approx(4, abs=5e-8)


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        (analysis.Y, 0),
        (analysis.Y, float("nan")),
        (analysis.X, True),
    ],
    ids=["zero-throughput", "nan-throughput", "bool-interactivity"],
)
def test_invalid_metrics_are_excluded_even_when_marked_successful(metric, value):
    results = formal_results()
    results["points"][1][metric] = value
    comparison = analysis.compare_aklr(results)
    assert comparison["status"] == "ok"
    assert [row["conc"] for row in comparison["points"]] == [1, 3]


def test_disjoint_successful_points_have_no_paired_gain():
    results = formal_results()
    fail(results["points"][1])
    fail(results["points"][2])
    fail(results["control"]["points"][0])
    comparison = analysis.compare_aklr(results)
    assert comparison["status"] == "unavailable"
    assert comparison["points"] == []
    assert comparison["tput_geomean_ratio"] is None


def test_equal_concurrency_in_different_arms_does_not_form_a_pair():
    results = formal_results()
    for mode in (results, results["control"]):
        mode["run"].update(points_expected=4, completed=False)
        mode["serving"]["arms"]["other-arm"] = {
            "concs": [1],
            "arm_serve_sha256": "9" * 64,
        }
    results["control"]["points"][0].update(arm="other-arm", exp_name="other-arm-c1")
    comparison = analysis.compare_aklr(results)
    assert comparison["status"] == "ok"
    assert [(row["arm"], row["conc"]) for row in comparison["points"]] == [("tp1", 2), ("tp1", 3)]


@pytest.mark.parametrize("invalid", ["duplicate", "too-many", "invalid-expected", "different-expected"])
def test_invalid_record_counts_and_duplicates_do_not_become_partial_pairs(invalid):
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
    assert comparison["status"] == "unavailable"
    assert comparison["points"] == []


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("model", "id"), "another-model", id="model.id"),
        pytest.param(("target", "id"), "another-target", id="target.id"),
        pytest.param(("framework", "id"), "vllm", id="framework.id"),
        pytest.param(("run", "id"), "another-run", id="run.id"),
        pytest.param(("run", "attempt_id"), "another-attempt", id="run.attempt_id"),
        pytest.param(("run", "aklr_sha"), "9" * 40, id="run.aklr_sha"),
        pytest.param(("run", "serving_sha256"), "9" * 64, id="run.serving_sha256"),
        pytest.param(("run", "execution", "image_id"), "sha256:" + "9" * 64, id="run.execution.image_id"),
        pytest.param(
            ("run", "client_packages", "agentx-harness"),
            "different-version",
            id="run.client_packages.agentx-harness",
        ),
        pytest.param(("run", "client", "sha"), "9" * 40, id="run.client.sha"),
        pytest.param(("run", "aklr_mode"), "enabled", id="run.aklr_mode"),
    ],
)
def test_partial_pairing_keeps_run_identity_guards(path, value):
    results = formal_results()
    fail(results["points"][2])
    target = results["control"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    comparison = analysis.compare_aklr(results)
    assert comparison["status"] == "unavailable"
    assert comparison["points"] == []


def test_report_pair_filter_does_not_relax_baseline_promotion():
    results = formal_results()
    assert report_all.eligible(results)
    change_settings(results["control"]["points"][1], "duration")
    assert analysis.compare_aklr(results)["status"] == "ok"
    assert not report_all.eligible(results)


def test_failed_record_metadata_does_not_block_other_successful_points():
    current = formal_results("current")
    baseline = formal_results("previous")
    set_metrics(current["points"][0], 150, 40)
    set_metrics(baseline["points"][0], 100, 20)
    fail(current["points"][1])
    current["points"][1].pop("measurement")
    fail(baseline["points"][2])
    change_settings(baseline["points"][2], "invalid-trace")

    comparison = analysis.compare(current, baseline)

    assert comparison["status"] == "ok"
    arm = comparison["arms"]["tp1"]
    assert arm["status"] == "comparable"
    rows = {row["conc"]: row for row in arm["points"]}
    assert rows[1]["tput_ratio"] == 1.5
    assert rows[1]["intvty_ratio"] == 2
    assert arm["tput_geomean_ratio"] == pytest.approx(1.5, abs=5e-8)
    for conc in (2, 3):
        assert rows[conc].get("tput_ratio") is None


@pytest.mark.parametrize(
    "change",
    [
        "duration",
        "environment",
        "trace",
        "invalid-trace",
        "missing-measurement",
        "serve-spec",
    ],
)
def test_one_changed_point_does_not_disable_the_whole_arm(change):
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

    assert current == original
    assert comparison["status"] == "ok"
    arm = comparison["arms"]["tp1"]
    assert arm["status"] == "comparable"
    rows = {row["conc"]: row for row in arm["points"]}
    assert rows[1]["tput_ratio"] == 1.5
    assert rows[2].get("tput_ratio") is None
    assert rows[2].get("reason")
    assert rows[3]["tput_ratio"] == 6
    assert arm["tput_geomean_ratio"] == pytest.approx(3, abs=5e-8)


def test_changed_grid_still_compares_unchanged_shared_points():
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

    assert comparison["status"] == "ok"
    arm = comparison["arms"]["tp1"]
    assert arm["status"] == "comparable"
    rows = {row["conc"]: row for row in arm["points"]}
    assert rows[1]["tput_ratio"] == 1
    assert rows[2]["tput_ratio"] == 1
    assert rows[3].get("tput_ratio") is None


def test_dirty_source_still_disables_weekly_comparison():
    current = formal_results("current")
    current["run"]["source_dirty"] = True
    comparison = analysis.compare(current, formal_results("previous"))
    assert comparison["status"] == "no-baseline"
    assert comparison["arms"] == {}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("run", "execution", "image_id"), "sha256:" + "9" * 64, id="run.execution.image_id"),
        pytest.param(
            ("run", "server_packages", "sglang"), "different-version", id="run.server_packages.sglang"
        ),
        pytest.param(
            ("run", "client_packages", "agentx-harness"),
            "different-version",
            id="run.client_packages.agentx-harness",
        ),
    ],
)
def test_execution_changes_compare_frontiers(path, value):
    current = formal_results("current")
    baseline = formal_results("previous")
    target = current
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    comparison = analysis.compare(current, baseline)
    assert (comparison["status"], comparison["method"]) == ("ok", "curve")
    assert "execution image or dependencies changed" in comparison["notes"]
    arm = comparison["arms"]["tp1"]
    assert arm["status"] == "curve"
    assert arm["points"] == []
    assert arm["tput_geomean_ratio"] == pytest.approx(1, abs=5e-8)


@pytest.mark.parametrize("change", ["other_trace", "other_gpu", "other_client"])
def test_other_hardware_client_or_traces_have_no_baseline(change):
    current = formal_results("current")
    if change == "other_trace":
        for point in current["points"]:
            point["measurement"]["trace"]["revision"] = "9" * 40
    elif change == "other_gpu":
        current["run"]["hardware"]["devices"][0]["total_memory"] *= 2
    else:
        current["run"]["client"]["sha"] = "9" * 40
    comparison = analysis.compare(current, formal_results("previous"))
    assert comparison["status"] == "no-baseline"
    assert comparison["reason"] == "hardware, AgentX client source or trace dataset changed"
    assert comparison["arms"] == {}


def test_curves_use_only_points_with_verified_measurements():
    current = formal_results("current")
    baseline = formal_results("previous")
    current["run"]["execution"]["image_id"] = "sha256:" + "9" * 64
    for point in current["points"]:
        point.pop("measurement")
    comparison = analysis.compare(current, baseline)
    assert comparison["arms"]["tp1"]["status"] == "curve-no-overlap"
    assert comparison["arms"]["tp1"]["frontier_points"]["current"] == 0
    assert comparison["combined"]["tput_geomean_ratio"] is None


def test_settings_changed_on_every_point_reports_a_curve_method(charted):
    current = formal_results("current")
    baseline = formal_results("previous")
    for point in current["points"]:
        change_settings(point, "duration")
    comparison = analysis.compare(current, baseline)
    assert comparison["arms"]["tp1"]["status"] == "curve"
    assert comparison["method"] == "curve"
    report.charts(current, baseline, comparison, "en")
    previous = [item for item in charted[-1] if item["color"] == report.BASELINE_COLOR]
    assert previous[0]["line"] == slim(analysis.verified(baseline, "tp1"))


def test_method_is_mixed_when_arms_compare_differently():
    arms = {"a": {"status": "comparable"}, "b": {"status": "curve"}, "c": {"status": "new-arm"}}
    assert analysis.comparison_method(arms) == "mixed"
    assert analysis.comparison_method({"c": {"status": "new-arm"}}) == "point"


@pytest.mark.parametrize("change", ["duration", "trace", "environment"])
def test_changed_points_stay_out_of_the_combined_frontier(change):
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
    comparison = analysis.compare(current, baseline)
    arm = comparison["arms"]["tp1"]["tput_geomean_ratio"]
    combined = comparison["combined"]["tput_geomean_ratio"]
    assert combined < 10
    assert combined <= max(arm, 6)


def test_renamed_arms_under_a_run_level_change_keep_the_previous_frontier(charted):
    current = formal_results("current")
    baseline = formal_results("previous")
    for mode in (current, current["control"]):
        mode["run"].update(point_policy="pareto", selection={"selection_sha256": "1" * 64})
    for point in current["points"]:
        point["arm"] = "renamed"
    current["serving"]["arms"] = {"renamed": current["serving"]["arms"].pop("tp1")}
    refresh_hashes(current)
    comparison = analysis.compare(current, baseline)
    assert comparison["method"] == "curve"
    assert comparison["combined"]["status"] == "curve"
    assert comparison["combined"]["baseline_frontier"] == slim(analysis.measured(baseline))
    report.charts(current, baseline, comparison, "en")
    previous = [item for item in charted[-1] if item["color"] == report.BASELINE_COLOR]
    assert previous[0]["line"] == slim(analysis.measured(baseline))


def points(*pairs):
    return [{"status": "ok", analysis.X: x, analysis.Y: y} for x, y in pairs]


def test_frontiers_interpolate_in_log_interactivity_over_the_shared_range():
    current = points((10, 100), (40, 50), (20, 10))
    baseline = points((20, 60), (80, 5))
    curve = analysis.compare_curve(current, baseline, "fixture")
    assert curve["status"] == "curve"
    assert curve["overlap"] == [20, 40]
    assert [row["intvty"] for row in curve["positions"]] == [20, 40]
    first, last = curve["positions"]
    assert first["current"] == pytest.approx(75, abs=5e-8)
    assert first["tput_ratio"] == pytest.approx(1.25, abs=5e-8)
    assert last["baseline"] == pytest.approx(60 - 55 / 2, abs=5e-8)
    assert curve["tput_ratio_min"] == pytest.approx(1.25, abs=5e-8)
    assert curve["tput_ratio_max"] == pytest.approx(50 / 32.5, abs=5e-8)
    assert curve["tput_geomean_ratio"] == pytest.approx((1.25 * 50 / 32.5) ** 0.5, abs=5e-8)


def test_frontiers_without_a_shared_range_only_chart():
    curve = analysis.compare_curve(points((10, 100)), points((20, 60), (30, 50)), "fixture")
    assert curve["status"] == "curve-no-overlap"
    assert curve["tput_geomean_ratio"] is None
    assert analysis.compare_curve([], points((1, 1)), "fixture")["status"] == "curve-no-overlap"


def test_curves_compare_only_shared_traces():
    def traced(points, revision):
        for point in points:
            point["measurement"] = {"trace": {"repo": "owner/traces", "revision": revision}}
        return points

    current = traced(points((10, 100), (40, 50)), "a" * 40)
    current += traced(points((10, 9999), (40, 9999)), "b" * 40)
    baseline = traced(points((10, 100), (40, 50)), "a" * 40)
    curve = analysis.compare_curve(current, baseline, "fixture")
    assert curve["tput_geomean_ratio"] == pytest.approx(1, abs=5e-8)
    other = traced(points((10, 100), (40, 50)), "c" * 40)
    assert analysis.compare_curve(current[2:], other, "fixture")["status"] == "curve-no-overlap"


def test_changed_selection_compares_frontiers(charted):
    current = formal_results("current")
    baseline = formal_results("previous")
    for mode in (current, current["control"]):
        mode["run"].update(point_policy="pareto", selection={"selection_sha256": "1" * 64})
    set_metrics(current["points"][0], 150, 40)
    comparison = analysis.compare(current, baseline)
    assert (comparison["status"], comparison["method"]) == ("ok", "curve")
    assert "point selection policy or selected recipes changed" in comparison["notes"]
    assert comparison["arms"]["tp1"]["status"] == "curve"

    report.charts(current, baseline, comparison, "en")
    previous = [item for item in charted[-1] if item["color"] == report.BASELINE_COLOR]
    assert previous[0]["line"] == slim(analysis.measured(baseline))
    html = report.comparison_tables(comparison, "zh")
    assert "曲线比较" in html
    assert "合并前沿" in html


def test_previous_frontier_contains_only_matching_successful_points(charted):
    current = formal_results("current")
    baseline = formal_results("previous")
    fail(current["points"][1])
    change_settings(current["points"][2], "duration")
    comparison = analysis.compare(current, baseline)

    report.charts(current, baseline, comparison, "en")

    series = charted[-1]
    previous = [item for item in series if item["color"] == report.BASELINE_COLOR]
    assert len(previous) == 1
    assert previous[0]["line"] == slim([baseline["points"][0]])
    assert series[0]["points"] == current["control"]["points"]
    assert series[1]["points"] == [current["points"][0], current["points"][2]]

    fail(current["points"][0])
    comparison = analysis.compare(current, baseline)
    assert comparison["method"] == "curve"
    report.charts(current, baseline, comparison, "en")
    previous = [item for item in charted[-1] if item["color"] == report.BASELINE_COLOR]
    assert previous[0]["line"] == slim(analysis.verified(baseline, "tp1"))


def test_chart_lines_connect_every_successful_point(charted):
    current = formal_results("current")
    set_metrics(current["points"][0], 100, 40)
    set_metrics(current["points"][1], 50, 20)
    set_metrics(current["points"][2], 10, 30)
    fail(current["control"]["points"][1])
    report.charts(current, None, {"status": "none"}, "en")
    disabled, enabled = charted[-1]
    assert enabled["line"] == [current["points"][i] for i in (1, 2, 0)]
    assert len(analysis.frontier(enabled["line"])) == 1
    assert disabled["line"] == report.through_all(analysis.measured(current["control"]))
    assert len(disabled["line"]) == len(current["control"]["points"]) - 1


def test_partial_report_exports_gains_and_drops_the_all_success_requirement(tmp_path, monkeypatch):
    current = formal_results("current")
    baseline = formal_results("previous")
    fail(current["points"][1])
    fail(current["control"]["points"][2])
    fail(baseline["points"][2])
    set_metrics(current["points"][0], 150)
    set_metrics(current["control"]["points"][0], 100)
    set_metrics(baseline["points"][0], 100)
    monkeypatch.setattr(report, "landed", lambda _base, _head, _cwd, _repo_url: None)
    report.write_report(current, baseline, tmp_path, "https://example.invalid/repo", REPO)
    summary = utils.read_json(tmp_path / "report.json")
    html = (tmp_path / "index.html").read_text()

    pairing = summary["aklr_comparison"]
    assert pairing["status"] == "ok"
    assert len(pairing["points"]) == 1
    assert pairing["tput_geomean_ratio"] == pytest.approx(1.5, abs=5e-8)
    assert summary["comparison"]["arms"]["tp1"]["tput_geomean_ratio"] == pytest.approx(1.5, abs=5e-8)
    assert "+50.0%" in html
    assert "1 matched pairs" in html
    assert "1 对匹配测量点" in html
    for obsolete in (
        "both modes must complete with matching settings",
        "Both AKLR modes must finish every selected point successfully",
        "week-over-week comparison disabled",
        "两种模式需完整成功且配置匹配",
        "两种 AKLR 模式都需成功完成全部选中数据点",
        "测量未完成或有失败点；不计算周环比",
        "Week over week",
        "周环比",
        "AKLR changes since the previous run",
        "上次运行以来的 AKLR 改动",
        "observed GPUs",
        "实际 GPU",
    ):
        assert obsolete not in html
