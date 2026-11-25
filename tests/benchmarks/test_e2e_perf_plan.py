# SPDX-License-Identifier: MIT
"""Plan skips a group only when its run key matches a complete published result."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from e2e_perf import plan
from e2e_perf.__main__ import main as e2e_main
from e2e_perf.model_config import DEFAULT_CONFIG, group_path, load_config
from e2e_perf.utils import read_json
from perf_utils import image
from perf_utils.gh import ConfigError, load_reviewers

COMMIT = "c" * 40


def run_plan(tmp_path, *extra, latest=None, out="plan"):
    argv = ["--commit", COMMIT, "--out", str(tmp_path / out), "--run-id", "7", *extra]
    if latest is not None:
        path = tmp_path / "latest.json"
        path.write_text(json.dumps(latest))
        argv += ["--latest", str(path)]
    assert plan.main(argv) == 0
    return read_json(tmp_path / out / "manifest.json")


def group_dir(tmp_path, item, out="plan"):
    return tmp_path / out / group_path(item["model"]["id"], item["target"]["id"], item["framework"]["id"])


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


def run_pr(tmp_path, comments, out="plan"):
    allowlist = tmp_path / "allowlist.yaml"
    allowlist.write_text("reviewers: [alice, bob]\n")
    path = tmp_path / f"{out}-comments.json"
    path.write_text(json.dumps(comments))
    argv = ["--commit", COMMIT, "--out", str(tmp_path / out), "--run-id", "7", "--force"]
    return plan.main([*argv, "--pr-comments", str(path), "--allowlist", str(allowlist)])


def test_the_latest_reviewer_comment_selects_the_pr_models(tmp_path):
    comments = [
        {"user": "alice", "body": "/e2e-perf-mi355x all"},
        {"user": "BOB", "body": "/e2e-perf-mi355x GLM52-FP8, missing"},
        {"user": "mallory", "body": "/e2e-perf-mi355x glm52-fp4"},
        {"user": "alice", "body": "/agentx-mi355x glm52-fp4"},
    ]
    assert run_pr(tmp_path, comments) == 0
    manifest = read_json(tmp_path / "plan/manifest.json")
    assert [g["model"]["id"] for g in manifest["groups"]] == ["glm52-fp8"]


@pytest.mark.parametrize(
    "comments",
    [
        [],
        [{"user": "mallory", "body": "/e2e-perf-mi355x all"}],
        [{"user": "alice", "body": "/agentx-mi355x all"}],
    ],
    ids=["none", "outsider", "agentx-command"],
)
def test_a_pr_run_needs_an_allowlisted_reviewers_request(tmp_path, capsys, comments):
    assert run_pr(tmp_path, comments) == 1
    assert not (tmp_path / "plan").exists()
    assert "No e2e perf reviewer has authorized this PR" in capsys.readouterr().err


def test_a_request_that_selects_no_model_skips_the_pr_run(tmp_path, monkeypatch):
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert run_pr(tmp_path, [{"user": "alice", "body": "/e2e-perf-mi355x missing"}]) == 0
    assert not (tmp_path / "plan").exists()
    assert "count=0" in output.read_text() and "skipped=true" in output.read_text()


def e2e_workflow():
    return yaml.safe_load(
        (Path(DEFAULT_CONFIG).parents[3] / ".github/workflows/e2e-perf-ci.yaml").read_text()
    )


def test_a_labeled_pr_from_this_repository_starts_the_workflow():
    workflow = e2e_workflow()
    assert workflow[True]["pull_request"] == {
        "branches": ["main"],
        "types": ["opened", "synchronize", "reopened", "labeled"],
    }
    gate = workflow["jobs"]["plan"]["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in gate
    assert gate.count("'ci:e2e-perf-mi355x'") == 2
    assert workflow["jobs"]["plan"]["permissions"] == {"contents": "read", "pull-requests": "read"}
    steps = workflow["jobs"]["plan"]["steps"]
    comments = next(
        step for step in steps if step.get("name") == "Read the PR comments that may authorize e2e perf"
    )
    assert "issues/$PR_NUMBER/comments" in comments["run"] and "--paginate" in comments["run"]
    allowlist = next(step for step in steps if step.get("name") == "Read main's allowlist")
    assert allowlist["if"] == "github.event_name != 'schedule'"
    resolve = next(step for step in steps if step.get("id") == "resolve")
    assert '--pr-comments "$RUNNER_TEMP/$PR_COMMENTS"' in resolve["run"]
    assert '--allowlist "$RUNNER_TEMP/$ALLOWLIST"' in resolve["run"]
    publish = workflow["jobs"]["publish"]["if"]
    assert "github.ref == 'refs/heads/main'" in publish and "pull_request" not in publish


def test_workflow_queue_allowlist_matches_the_allowlist_file():
    concurrency = e2e_workflow()["concurrency"]
    lists = re.findall(r"fromJSON\('(\[[^']*\])'\),\s*github\.triggering_actor\)", concurrency["group"])
    assert len(lists) == 1
    assert json.loads(lists[0]) == load_reviewers()
    assert "format('e2e-perf-pr-{0}', github.event.pull_request.number)" in concurrency["group"]
    assert "format('e2e-perf-refused-{0}', github.run_id)" in concurrency["group"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_safe_images_are_read_from_the_registry_as_docker_inspects_them(tmp_path, monkeypatch):
    config = json.dumps(
        {"architecture": "amd64", "config": {"Env": ["PATH=/usr/bin", "HIP_FORCE=1"]}}
    ).encode()
    calls = []

    def skopeo(argv, **_):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=config, stderr=b"")

    monkeypatch.setattr(image.subprocess, "run", skopeo)
    manifest = run_plan(tmp_path, "--force")
    assert e2e_main(["inspect-images", "--plan", str(tmp_path / "plan")]) == 0
    planned = [item for item in manifest["groups"] if item["status"] == "planned"]
    assert planned and len(calls) == len(
        {read_json(group_dir(tmp_path, item) / "plan.json")["image"]["run"] for item in planned}
    )
    for item in planned:
        pinned = read_json(group_dir(tmp_path, item) / "plan.json")["image"]["run"]
        inspection = read_json(group_dir(tmp_path, item) / "execution.json")
        assert (
            image.verify_image(pinned, inspection)["image_id"]
            == "sha256:" + hashlib.sha256(config).hexdigest()
        )
        assert image.image_environment(inspection) == {"PATH": "/usr/bin", "HIP_FORCE": "1"}
    name, digest = pinned.rsplit("@", 1)
    assert calls[0][-1] == f"docker://{name.rsplit(':', 1)[0]}@{digest}"


def test_a_registry_failure_fails_the_inspection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        image.subprocess, "run", lambda argv, **_: subprocess.CompletedProcess(argv, 1, b"", b"unauthorized")
    )
    run_plan(tmp_path, "--force")
    assert e2e_main(["inspect-images", "--plan", str(tmp_path / "plan")]) == 1
    with pytest.raises(ValueError, match="pinned by digest"):
        image.registry_inspection("lmsysorg/sglang:latest")


def test_a_safe_sweep_is_summarized_from_its_results(tmp_path, capsys):
    assert e2e_main(["sweep", "--summarize", "0", "--out", str(tmp_path)]) == 1
    assert "No usable results were saved." in capsys.readouterr().out
    with pytest.raises(SystemExit):
        e2e_main(["sweep", "--out", str(tmp_path)])


def test_the_safe_backend_runs_each_group_in_its_own_workload():
    workflow = e2e_workflow()
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["gpu_runner"]["options"] == ["aklr-gpu", "aklr-safe"]
    assert inputs["gpu_runner"]["default"] == "aklr-gpu"
    plan_steps = workflow["jobs"]["plan"]["steps"]
    inspect = next(step for step in plan_steps if "inspect-images" in step.get("run", ""))
    assert inspect["if"] == "env.E2E_GPU_RUNNER == 'aklr-safe' && steps.resolve.outputs.count != '0'"
    sweep = workflow["jobs"]["sweep"]
    assert (
        sweep["runs-on"]
        == "${{ needs.plan.outputs.gpu_runner == 'aklr-safe' && 'aklr-safe' || matrix.runner }}"
    )
    steps = {step.get("name"): step for step in sweep["steps"]}
    safe = steps["Measure one group on SaFE, AKLR off then on"]
    assert safe["shell"] == "bash .github/scripts/gpu_job.sh {0}"
    assert safe["env"]["SAFE_WORKSPACE_STORAGE"] == "readonly"
    delete = steps["Delete the SaFE workload"]
    assert delete["env"]["SAFE_JOB_NAME"] == safe["env"]["SAFE_JOB_NAME"] == "sweep-${{ matrix.job_index }}"
    assert delete["if"].startswith("always()")
    passed = set(safe["env"]["PASS_ENV"].split())
    used = set(re.findall(r"\$\{?(E2E_[A-Z_]+|FRAMEWORK_ID|GITHUB_[A-Z_]+)", safe["run"]))
    assert used <= passed
    assert (
        steps["Measure one group, AKLR off then on"]["if"] == "needs.plan.outputs.gpu_runner != 'aklr-safe'"
    )
