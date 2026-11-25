# SPDX-License-Identifier: MIT
"""AKLR on / off pairing, the index and group pages, and publishing to site-data's e2e/ section."""

from __future__ import annotations

import math

import pytest
from e2e_perf import publish, report
from e2e_perf.utils import geomean, pair, read_json, write_json

from tests.benchmarks.e2e_results import CONCS, WORKLOAD, point, results

GROUP = "models/glm52-fp4/targets/mi355x/frameworks/sglang"
GROUP_KEY = "glm52-fp4/mi355x/sglang"


def test_complete_runs_pair_every_concurrency():
    comparison = pair(results())
    assert comparison["status"] == "ok" and comparison["complete"] is True
    assert comparison["matched"] == comparison["expected"] == 7
    assert math.isclose(comparison["tput_geomean_ratio"], 1.1)
    assert math.isclose(comparison["rows"][0]["intvty_ratio"], 1.1)


def test_failed_points_are_unmatched_and_excluded_from_the_geomean():
    enabled = [point(conc, 120.0 if conc == 1 else 110.0) for conc in CONCS]
    enabled[6] = point(64, status="failed")
    comparison = pair(results(enabled=enabled))
    assert comparison["status"] == "partial" and comparison["matched"] == 6
    assert comparison["rows"][6]["tput_ratio"] is None
    assert math.isclose(comparison["max_ratio"], 1.2)
    assert math.isclose(comparison["tput_geomean_ratio"], geomean([1.2] + [1.1] * 5))


def test_points_pair_by_workload_and_concurrency():
    data = results(
        disabled=[point(1), point(1, 50.0, workload="100k1k"), point(8, workload="100k1k")],
        enabled=[point(1, 110.0), point(1, 60.0, workload="100k1k")],
    )
    data["run"]["workloads"] = [
        {"id": WORKLOAD, "concurrency": [1]},
        {"id": "100k1k", "concurrency": [1, 8]},
    ]
    comparison = pair(data)
    assert [(row["workload"], row["conc"]) for row in comparison["rows"]] == [
        (WORKLOAD, 1),
        ("100k1k", 1),
        ("100k1k", 8),
    ]
    assert [row["tput_ratio"] for row in comparison["rows"]] == [pytest.approx(1.1), pytest.approx(1.2), None]
    assert comparison["status"] == "partial" and comparison["matched"] == 2 and comparison["expected"] == 3


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (1.03, "delta good"),
        (0.97, "delta bad"),
        (1.0296, "delta good"),
        (0.9704, "delta bad"),
        (1.0294, ""),
        (1.029, ""),
        (0.971, ""),
        (None, ""),
    ],
)
def test_changes_beyond_the_threshold_are_green_for_gains_and_red_for_regressions(ratio, expected):
    assert report.delta_class(ratio) == expected


def test_runs_with_no_matched_pair_fail_and_unfinished_runs_are_incomplete():
    assert pair(results(enabled=[]))["status"] == "failed"
    unfinished = pair(results(completed=False))
    assert unfinished["status"] == "partial" and unfinished["complete"] is False
    empty = results(disabled=[], enabled=[])
    empty["run"]["workloads"] = []
    assert pair(empty)["complete"] is False and pair(empty)["status"] == "failed"


def site_from(tmp_path, groups, *, run_id="101", name="site"):
    """Report one run: groups maps a group path to results, or None for a group without results."""
    manifest = {"schema": 1, "run_id": run_id, "groups": []}
    for path, value in groups.items():
        parts = path.split("/")
        manifest["groups"].append(
            {
                "model": {"id": parts[1], "display_name": parts[1].upper()},
                "target": {"id": parts[3], "runner_label": "conductor-mi355x"},
                "framework": {"id": parts[5]},
                "status": "planned",
                "run_key": f"key-{path}-{run_id}",
                "gpus": 8,
                "error": None,
            }
        )
        if value is not None:
            value["run"].update(id=run_id, run_key=manifest["groups"][-1]["run_key"])
            write_json(tmp_path / f"results-{name}" / path / "results.json", value)
    plan = write_json(tmp_path / f"plan-{name}.json", manifest)
    url = f"https://example.invalid/runs/{run_id}"
    return report.build(plan, tmp_path / f"results-{name}", tmp_path / name, url, report.DEFAULT_CONFIG)


def test_report_writes_group_pages_and_an_index(tmp_path):
    site = site_from(tmp_path, {GROUP: results()})
    row = site["groups"][0]
    assert row["status"] == "ok" and row["complete"] is True and row["matched"] == 7
    assert row["gpu"] == "AMD Instinct MI355X" and row["run_id"] == "101"
    page = (tmp_path / "site" / GROUP / "index.html").read_text()
    assert "1.100× (+10.0%)" in page and "7/7 matched pairs" in page and "交互速度变化" in page
    assert 'href="../../../../../../index.html?lang=zh"' in page
    assert (tmp_path / "site" / GROUP / "results.json").is_file()
    index = (tmp_path / "site/index.html").read_text()
    assert "← AKLR Dashboard" in index and "← AKLR 看板" in index
    assert "mi355x · sglang · TP8" in index and "1.100×" in index
    assert '<td class="delta good">1.100×</td>' in index and "8k1k (ISL 8192 / OSL 1024)" in index
    assert "<td>8k1k</td><td>1</td>" in page
    assert page.count('<td class="delta good">+10.0%</td>') == 2 * 2 * 7
    assert "<svg" not in index + page


def test_failed_cells_show_their_status_and_error(tmp_path):
    enabled = [point(conc, 110.0) for conc in CONCS]
    enabled[2] = {
        "workload": WORKLOAD,
        "conc": 4,
        "status": "not-completed",
        "error": "server exited with code -9",
    }
    site_from(tmp_path, {GROUP: results(enabled=enabled)})
    page = (tmp_path / "site" / GROUP / "index.html").read_text()
    assert 'title="server exited with code -9">Not completed' in page and "6/7 matched pairs" in page


def test_a_group_without_results_is_listed_as_failed_without_a_page(tmp_path):
    site = site_from(tmp_path, {GROUP: None})
    assert site["groups"][0]["status"] == "failed" and site["groups"][0]["report"] is False
    assert "the sweep saved no results" in site["groups"][0]["error"]
    assert not (tmp_path / "site" / GROUP).exists()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda r: r.update(framework={"id": "vllm"}), "framework 'vllm'"),
        (lambda r: r["run"].update(id="100"), "from run '100'"),
        (lambda r: r["run"].update(run_key="old"), "another run key"),
    ],
)
def test_results_of_another_group_run_or_plan_are_rejected(tmp_path, change, reason):
    site_from(tmp_path, {GROUP: results()})
    path = tmp_path / "results-site" / GROUP / "results.json"
    stale = read_json(path)
    change(stale)
    write_json(path, stale)
    plan, found = tmp_path / "plan-site.json", tmp_path / "results-site"
    site = report.build(plan, found, tmp_path / "out", None, report.DEFAULT_CONFIG)
    row = site["groups"][0]
    assert row["report"] is False and row["status"] == "failed" and reason in row["error"]
    assert not (tmp_path / "out" / GROUP).exists()


def publish_run(tmp_path, site, run_id, commit="c" * 40, newer=()):
    source = {"run_id": run_id, "run_url": f"https://example.invalid/runs/{run_id}", "commit": commit}
    return publish.merge(
        site,
        tmp_path / "section",
        {**source, "ref": "refs/heads/main", "event": "schedule"},
        descends=lambda child, parent: (child, parent) in newer,
    )


def test_publish_replaces_only_the_measured_groups_and_rebuilds_the_index(tmp_path):
    other = GROUP.replace("glm52-fp4", "glm52-fp8")
    first = site_from(tmp_path, {GROUP: results(), other: results(model="glm52-fp8")}, run_id="1", name="a")
    publish_run(tmp_path, tmp_path / "a", "1")
    second = site_from(tmp_path, {other: results(model="glm52-fp8", run_id="2")}, run_id="2", name="b")
    latest = publish_run(tmp_path, tmp_path / "b", "2")
    section = tmp_path / "section"
    assert list(latest["groups"]) == ["glm52-fp4/mi355x/sglang", "glm52-fp8/mi355x/sglang"]
    assert latest["groups"]["glm52-fp4/mi355x/sglang"]["run_key"] == first["groups"][0]["run_key"]
    assert latest["groups"]["glm52-fp8/mi355x/sglang"]["run_key"] == second["groups"][0]["run_key"]
    assert latest["groups"]["glm52-fp8/mi355x/sglang"]["source"]["run_id"] == "2"
    assert (section / "runs/1" / GROUP / "results.json").is_file()
    assert (section / "runs/2" / other / "results.json").is_file()
    assert (section / GROUP / "index.html").is_file()
    summary = read_json(section / "section.json")
    assert summary["id"] == "e2e" and summary["title"]["zh"] == "端到端性能"
    assert summary["targets"] == 2 and summary["status_counts"] == {"ok": 2}
    assert "runs/2" in (section / "index.html").read_text()


def test_publish_drops_groups_that_are_no_longer_configured(tmp_path):
    stale = tmp_path / "section/models/old/targets/mi355x/frameworks/sglang"
    stale.mkdir(parents=True)
    write_json(
        tmp_path / "section/latest.json", {"schema": 1, "groups": {"old/mi355x/sglang": {"status": "ok"}}}
    )
    site_from(tmp_path, {GROUP: results()})
    latest = publish_run(tmp_path, tmp_path / "site", "101")
    assert list(latest["groups"]) == ["glm52-fp4/mi355x/sglang"]
    assert not (tmp_path / "section/models/old").exists()


def test_publish_rejects_unsafe_run_ids(tmp_path):
    site_from(tmp_path, {GROUP: results()})
    with pytest.raises(ValueError, match="invalid run id"):
        publish_run(tmp_path, tmp_path / "site", "../x")


def test_publish_keeps_groups_of_a_newer_commit(tmp_path):
    other = GROUP.replace("glm52-fp4", "glm52-fp8")
    site_from(tmp_path, {GROUP: results()}, run_id="2", name="new")
    publish_run(tmp_path, tmp_path / "new", "2", commit="n" * 40)
    later = tmp_path / "section/models/new/targets/mi355x/frameworks/sglang"
    later.mkdir(parents=True)
    latest = read_json(tmp_path / "section/latest.json")
    latest["groups"]["new/mi355x/sglang"] = {**latest["groups"][GROUP_KEY], "key": "new/mi355x/sglang"}
    write_json(tmp_path / "section/latest.json", latest)
    old = results(enabled=[point(conc, 90.0) for conc in CONCS])
    site_from(tmp_path, {GROUP: old, other: results(model="glm52-fp8")}, run_id="1", name="old")
    merged = publish_run(tmp_path, tmp_path / "old", "1", commit="o" * 40, newer={("n" * 40, "o" * 40)})
    assert merged["groups"][GROUP_KEY]["source"]["run_id"] == "2"
    assert merged["groups"]["glm52-fp8/mi355x/sglang"]["source"]["run_id"] == "1"
    assert list(merged["groups"])[-1] == "new/mi355x/sglang" and later.is_dir()
    assert "1.100×" in (tmp_path / "section" / GROUP / "index.html").read_text()


def test_publish_replaces_a_group_with_its_latest_failed_attempt(tmp_path):
    site_from(tmp_path, {GROUP: results()}, run_id="1", name="a")
    publish_run(tmp_path, tmp_path / "a", "1")
    site_from(tmp_path, {GROUP: None}, run_id="2", name="b")
    latest = publish_run(tmp_path, tmp_path / "b", "2", commit="d" * 40)
    assert latest["groups"][GROUP_KEY]["status"] == "failed"
    assert not (tmp_path / "section" / GROUP).exists()
