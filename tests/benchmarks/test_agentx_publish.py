# SPDX-License-Identifier: MIT
"""Publishing a run replaces only its targets in site-data/agentx, and only main publishes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from agentx_ci import publish, utils
from agentx_ci.model_config import ConfigError

from tests.benchmarks.agentx_results import REPO

RUNS = [("alpha", "mi355x", "sglang"), ("alpha", "mi355x", "vllm"), ("beta", "mi355x", "sglang")]


def write_config(path: Path, runs: list[tuple[str, str, str]]) -> Path:
    models: dict = {}
    for model_id, target_id, framework_id in runs:
        targets = models.setdefault(model_id, {"targets": {}})["targets"]
        targets.setdefault(target_id, {"frameworks": {}})["frameworks"][framework_id] = {}
    path.write_text(yaml.safe_dump({"models": models}))
    return path


def entry(key: tuple[str, str, str], status: str = "ok") -> dict:
    model_id, target_id, framework_id = key
    return {
        "model": {"id": model_id, "display_name": model_id.title()},
        "target": {"id": target_id},
        "framework": {"id": framework_id},
        "status": status,
        "report": status == "ok",
    }


def write_site(root: Path, run_id: str, items: list[dict]) -> Path:
    for item in items:
        if item["report"]:
            key = (item["model"]["id"], item["target"]["id"], item["framework"]["id"])
            page = root / utils.target_path(*key) / "index.html"
            page.parent.mkdir(parents=True)
            page.write_text(run_id)
    utils.write_json(
        root / "manifest.json",
        {"schema": utils.SCHEMA, "run_id": run_id, "generated_at": "now", "targets": items},
    )
    return root


def source(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "run_url": f"https://github.com/o/r/actions/runs/{run_id}",
        "commit": run_id * 8,
        "ref": "refs/heads/main",
        "event": "schedule",
    }


def page(section: Path, key: tuple[str, str, str]) -> str | None:
    path = section / utils.target_path(*key) / "index.html"
    return path.read_text() if path.is_file() else None


@pytest.fixture
def published(tmp_path: Path) -> tuple[Path, Path]:
    """A section that a full run "1" published, and the config enabling RUNS."""
    config = write_config(tmp_path / "agentx.yaml", RUNS)
    section = tmp_path / "site-data" / "agentx"
    site = write_site(tmp_path / "run-1", "1", [entry(key) for key in RUNS])
    publish.merge(site, section, source("1"), config=config)
    return section, config


def test_a_run_replaces_only_its_targets(tmp_path, published):
    section, config = published
    site = write_site(tmp_path / "run-2", "2", [entry(RUNS[1])])
    manifest = publish.merge(site, section, source("2"), config=config)
    assert [page(section, key) for key in RUNS] == ["1", "2", "1"]
    assert [item["source"]["run_id"] for item in manifest["targets"]] == ["1", "2", "1"]
    assert utils.read_json(section / "manifest.json") == manifest
    summary = utils.read_json(section / "section.json")
    assert (summary["id"], summary["path"], summary["last_run"]) == ("agentx", "index.html", source("2"))
    assert (summary["targets"], summary["status_counts"]) == (3, {"ok": 3})
    index = (section / "index.html").read_text()
    assert 'href="https://github.com/o/r/actions/runs/2"><code>22222222</code>' in index
    assert index.count('href="../index.html"') == 2


def test_section_lists_the_main_commits_of_its_targets(tmp_path, published):
    section, config = published
    assert utils.read_json(section / "section.json")["commits"] == ["11111111"]
    site = write_site(tmp_path / "run-2", "2", [entry(RUNS[1])])
    publish.merge(site, section, source("2"), config=config)
    assert utils.read_json(section / "section.json")["commits"] == ["11111111", "22222222"]


def test_a_failed_target_replaces_its_previous_report(tmp_path, published):
    section, config = published
    site = write_site(tmp_path / "run-2", "2", [entry(RUNS[0], status="failed")])
    manifest = publish.merge(site, section, source("2"), config=config)
    assert page(section, RUNS[0]) is None
    assert [item["status"] for item in manifest["targets"]] == ["failed", "ok", "ok"]
    assert utils.read_json(section / "section.json")["status_counts"] == {"failed": 1, "ok": 2}


def test_targets_no_longer_enabled_are_removed(tmp_path, published):
    section, _ = published
    config = write_config(tmp_path / "fewer.yaml", RUNS[:2])
    site = write_site(tmp_path / "run-2", "2", [entry(RUNS[0])])
    manifest = publish.merge(site, section, source("2"), config=config)
    keys = [(i["model"]["id"], i["target"]["id"], i["framework"]["id"]) for i in manifest["targets"]]
    assert keys == RUNS[:2]
    assert not (section / "models" / "beta").exists()


def test_a_previous_schema_starts_a_fresh_section(tmp_path, published):
    section, config = published
    stale = utils.read_json(section / "manifest.json") | {"schema": utils.SCHEMA - 1}
    utils.write_json(section / "manifest.json", stale)
    site = write_site(tmp_path / "run-2", "2", [entry(RUNS[2])])
    manifest = publish.merge(site, section, source("2"), config=config)
    assert len(manifest["targets"]) == 1
    assert [page(section, key) for key in RUNS] == [None, None, "2"]


def test_a_target_outside_the_config_is_refused(tmp_path, published):
    section, config = published
    before = (section / "manifest.json").read_text()
    site = write_site(tmp_path / "run-2", "2", [entry(("gamma", "mi355x", "sglang"))])
    with pytest.raises(ConfigError, match="gamma/mi355x/sglang"):
        publish.merge(site, section, source("2"), config=config)
    assert (section / "manifest.json").read_text() == before


def test_cli_records_the_run_source(tmp_path, published, capsys):
    section, config = published
    site = write_site(tmp_path / "run-2", "2", [entry(RUNS[0])])
    argv = ["--site", str(site), "--config", str(config), "--run-id", "2"]
    argv += ["--run-url", source("2")["run_url"], "--commit", "2" * 8, "--ref", "refs/heads/main"]
    argv += ["--event", "schedule", "--section", str(section)]
    assert publish.main(argv) == 0
    assert "alpha/mi355x/sglang: ok from run 2" in capsys.readouterr().out
    assert json.loads((section / "section.json").read_text())["last_run"] == source("2")


def workflow(name: str) -> dict:
    return yaml.safe_load((REPO / ".github/workflows" / name).read_text())


def test_only_main_schedules_and_manual_runs_publish():
    jobs = workflow("agentx-ci.yaml")["jobs"]
    assert "pages" not in jobs
    assert all("pages" not in job.get("permissions", {}) for job in jobs.values())
    rendered = next(step for step in jobs["report"]["steps"] if "agentx_bench report" in step.get("run", ""))
    assert "--baseline" not in rendered["run"]
    job = jobs["publish"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "pull_request" not in job["if"]
    assert job["environment"] == "agentx-bot"
    token = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/create-github-app-token")
    )
    assert token["with"]["app-id"] == "${{ secrets.AKLR_AGENTX_BOT_APP_ID }}"
    assert token["with"]["private-key"] == "${{ secrets.AKLR_AGENTX_BOT_PRIVATE_KEY }}"
    publish = next(step for step in job["steps"] if "commit_site_data.sh agentx" in step.get("run", ""))
    assert publish["env"]["BOT_LOGIN"] == "${{ vars.AKLR_AGENTX_BOT_LOGIN }}"
    assert job["concurrency"] == {"group": "site-data", "cancel-in-progress": False}
    runs = [step.get("run", "") for step in job["steps"]]
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout"))
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    check = runs.index('git merge-base --is-ancestor "$GITHUB_SHA" origin/main')
    assert check < next(i for i, run in enumerate(runs) if "commit_site_data.sh agentx" in run)


def test_the_site_workflow_deploys_only_after_main_runs():
    site = workflow("site.yaml")
    run = site[True]["workflow_run"]
    assert (run["workflows"], run["types"], run["branches"]) == (
        ["AKLR-agentx-ci", "AKLR-CI"],
        ["completed"],
        ["main"],
    )
    assert workflow("agentx-ci.yaml")["name"] == "AKLR-agentx-ci"
    assert workflow("ci.yaml")["name"] == "AKLR-CI"
    assert site["jobs"]["deploy"]["permissions"]["pages"] == "write"
    assert "github.event.workflow_run.event" in site["jobs"]["deploy"]["if"]
    data = next(step for step in site["jobs"]["deploy"]["steps"] if step.get("id") == "data")
    assert "[[ $status == 2 ]]" in data["run"]
    assert 'exit "$status"' in data["run"]
