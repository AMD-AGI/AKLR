# SPDX-License-Identifier: MIT
"""Plan skips a group only when its run key matches a complete published result."""

from __future__ import annotations

import json

import pytest
import yaml
from e2e_perf import plan
from e2e_perf.model_config import DEFAULT_CONFIG, load_config
from e2e_perf.utils import read_json
from perf_utils.gh import ConfigError

COMMIT = "c" * 40


def run_plan(tmp_path, *extra, latest=None, out="plan"):
    argv = ["--commit", COMMIT, "--out", str(tmp_path / out), "--run-id", "7", *extra]
    if latest is not None:
        path = tmp_path / "latest.json"
        path.write_text(json.dumps(latest))
        argv += ["--latest", str(path)]
    assert plan.main(argv) == 0
    return read_json(tmp_path / out / "manifest.json")


def published(manifest, complete=True):
    groups = {
        f"{g['model']['id']}/{g['target']['id']}/{g['framework']['id']}": {
            "run_key": g["run_key"],
            "complete": complete,
        }
        for g in manifest["groups"]
    }
    return {"schema": 1, "groups": groups}


def test_without_published_results_every_group_is_planned(tmp_path):
    manifest = run_plan(tmp_path)
    assert [g["status"] for g in manifest["groups"]] == ["planned"] * 2
    spec = read_json(tmp_path / "plan/models/glm52-fp8/targets/mi355x/frameworks/sglang/plan.json")
    assert spec["model"]["id"] == "glm52-fp8" and spec["image"]["run"].split("@")[1].startswith("sha256:")
    assert spec["aklr_sha"] == COMMIT and spec["run_key"] == manifest["groups"][1]["run_key"]


def test_unchanged_complete_groups_are_skipped_and_force_runs_them(tmp_path):
    first = run_plan(tmp_path, out="first")
    again = run_plan(tmp_path, latest=published(first), out="again")
    assert [g["status"] for g in again["groups"]] == ["skipped"] * 2
    assert not (tmp_path / "again/models").exists()
    forced = run_plan(tmp_path, "--force", latest=published(first), out="forced")
    assert [g["status"] for g in forced["groups"]] == ["planned"] * 2


def test_an_incomplete_published_result_runs_again(tmp_path):
    first = run_plan(tmp_path, out="first")
    again = run_plan(tmp_path, latest=published(first, complete=False), out="again")
    assert [g["status"] for g in again["groups"]] == ["planned"] * 2


def test_only_the_changed_group_runs(tmp_path):
    first = run_plan(tmp_path, out="first")
    latest = published(first)
    latest["groups"]["glm52-fp8/mi355x/sglang"]["run_key"] = "stale"
    again = run_plan(tmp_path, latest=latest, out="again")
    assert [g["status"] for g in again["groups"]] == ["skipped", "planned"]


@pytest.mark.parametrize(
    "change",
    [
        {"aklr_sha": "d" * 40},
        {"image": "example.invalid/x@sha256:" + "e" * 64},
        {"recipe": {"launch": {"command": ["other"]}}},
        {"workloads": [{"id": "8k1k", "isl": 4096}]},
        {"client": {"sha": "f" * 40}},
        {"weights": {"revision": "0" * 40}},
    ],
    ids=lambda change: next(iter(change)),
)
def test_run_key_changes_with_every_input(change):
    base = {
        "aklr_sha": COMMIT,
        "image": "example.invalid/x@sha256:" + "a" * 64,
        "recipe": {"launch": {"command": ["sglang"]}},
        "workloads": [{"id": "8k1k", "isl": 8192}],
        "client": {"sha": "1" * 40},
        "weights": {"revision": "2" * 40},
    }
    assert plan.run_key(**base) != plan.run_key(**{**base, **change})


def test_a_missing_recipe_fails_only_its_group(tmp_path, monkeypatch):
    real = plan.load_recipe

    def load(group):
        if group["model"]["id"] == "glm52-fp8":
            raise FileNotFoundError("no recipe")
        return real(group)

    monkeypatch.setattr(plan, "load_recipe", load)
    manifest = run_plan(tmp_path)
    assert [g["status"] for g in manifest["groups"]] == ["planned", "resolve-failed"]
    assert "no recipe" in manifest["groups"][1]["error"]


def test_manual_runs_need_an_allowlisted_reviewer(tmp_path):
    allowlist = tmp_path / "allowlist.yaml"
    allowlist.write_text("reviewers: [alice]\n")
    argv = ["--commit", COMMIT, "--out", str(tmp_path / "plan"), "--allowlist", str(allowlist)]
    assert plan.main([*argv, "--manual-actor", "mallory", "--manual-ref", "refs/heads/main"]) == 1
    assert plan.main([*argv, "--manual-actor", "alice", "--manual-ref", "refs/heads/main"]) == 0


@pytest.mark.parametrize(
    "workloads",
    [
        {"id": "8k1k", "isl": 8192, "osl": 1024, "num_prompts_per_conc": 10, "concurrency": [1]},
        [],
        [{"id": "8k1k", "isl": 8192, "osl": 1024, "num_prompts_per_conc": 10, "concurrency": [1, 1]}],
        [{"id": "8k1k", "isl": 0, "osl": 1024, "num_prompts_per_conc": 10, "concurrency": [1]}],
        [
            {"id": "8k1k", "isl": 8192, "osl": 1024, "num_prompts_per_conc": 10, "concurrency": [1]},
            {"id": "8k1k", "isl": 102400, "osl": 1024, "num_prompts_per_conc": 10, "concurrency": [1]},
        ],
    ],
    ids=["mapping", "empty", "repeated-conc", "zero-isl", "repeated-id"],
)
def test_workloads_must_be_a_list_of_valid_distinct_workloads(tmp_path, workloads):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config["workloads"] = workloads
    path = tmp_path / "e2e.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigError):
        load_config(path)
