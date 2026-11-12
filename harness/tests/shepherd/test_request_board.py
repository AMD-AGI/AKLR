"""Exercise issue requests through the CLI against a fake `gh` that replays GitHub replies and logs writes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "automations/shepherd/skills/shepherd/scripts/request_board.py"
SCHEMA = "aklr/ops/gemm/impls/flydsl/gemm_a8w8_n512/0123abcd.json"
# Replays `GitHub` from github.json and appends every call to calls.jsonl beside it.
FAKE_GH = """
import json, sys
from pathlib import Path
here = Path(__file__).parent
github = json.loads((here / "github.json").read_text())
args = sys.argv[1:]
with open(here / "calls.jsonl", "a") as log:
    log.write(json.dumps(args) + "\\n")
fields = dict(arg.split("=", 1) for arg in args if "=" in arg and not arg.startswith("query"))
query = next((arg for arg in args if arg.startswith("query=")), "")
if "search(" in query:
    print(json.dumps({"data": {"search": {"nodes": github["search"]}}}))
elif "issues(labels" in query:
    numbers = [{"number": int(n)} for n, i in github["issues"].items() if i["state"] == "OPEN"
               and "shepherd" in [l["name"] for l in i["labels"]["nodes"]]]
    print(json.dumps({"data": {"repository": {"issues": {"nodes": numbers}}}}))
elif "issue(number" in query:
    print(json.dumps({"data": {"repository": {"issue": github["issues"][fields["number"]]}}}))
elif args[-3:-1] == ["--jq", ".[].login"] or args[-1] == ".[].login":
    print("\\n".join(github["members"]))
elif "comments" in args[-3] and "-X" not in args:
    for comment in github["comments"]:
        print(json.dumps(comment))
"""


def issue(body, labeler="owner", author="owner", editor=None, edited=None, state="OPEN", labeled=True):
    return {
        "state": state,
        "body": body,
        "lastEditedAt": edited,
        "author": {"login": author},
        "editor": {"login": editor} if editor else None,
        "labels": {"nodes": [{"name": "shepherd"}] if labeled else []},
        "timelineItems": {
            "nodes": [
                {
                    "createdAt": "2026-11-12T01:00:00Z",
                    "actor": {"__typename": "Bot" if labeler == "aklr-shepherd" else "User", "login": labeler},
                    "label": {"name": "shepherd"},
                }
            ]
        },
    }


def request_body(spec, notes="Try split-K below M=64."):
    return f"### Request\n\n```yaml\n{spec}\n```\n\n### Notes\n\n{notes}\n"


@pytest.fixture
def github(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!{sys.executable}\n{FAKE_GH}")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    config = tmp_path / "configuration.yaml"
    config.write_text(
        "repository: AMD-AGI/AKLR\npolicy: {allowed_ops: [gemm]}\n"
        "requests: {label: shepherd, team: shepherd-requesters, bot: 'aklr-shepherd[bot]'}\n"
    )
    root = tmp_path / "state"
    root.mkdir()
    replies = {"members": ["owner", "peer"], "search": [], "issues": {}, "comments": []}

    def run(*args, state=None, **changes):
        replies.update(changes)
        (bin_dir / "github.json").write_text(json.dumps(replies))
        (root / "state.json").write_text(json.dumps({"revision": 0, "state": state or {"campaigns": [], "requests": []}}))
        (bin_dir / "calls.jsonl").unlink(missing_ok=True)
        result = subprocess.run(
            [sys.executable, "-I", str(SCRIPT), "--config", str(config), *args]
            + ([] if args[0] == "label" else ["--root", str(root)]),
            capture_output=True,
            text=True,
            check=False,
        )
        log = bin_dir / "calls.jsonl"
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


def test_title_prefix_from_a_team_member_adds_the_label(github):
    result, calls = github(
        "label",
        search=[
            {"number": 1, "title": "shepherd: speed up gemm", "author": {"login": "owner"}, "labels": {"nodes": []}},
            {"number": 2, "title": "Shepherd: from outside", "author": {"login": "stranger"}, "labels": {"nodes": []}},
            {"number": 3, "title": "shepherd: done", "author": {"login": "owner"}, "labels": {"nodes": [{"name": "shepherd"}]}},
            {"number": 4, "title": "about shepherd: a question", "author": {"login": "owner"}, "labels": {"nodes": []}},
            {"number": 5, "title": "shepherd: edited", "author": {"login": "owner"}, "editor": {"login": "stranger"},
             "labels": {"nodes": []}},
        ],
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"labeled": [1]}
    assert [c for c in calls if "POST" in c] == [
        ["api", "-X", "POST", "repos/AMD-AGI/AKLR/issues/1/labels", "-f", "labels[]=shepherd"]
    ]


def test_observe_approves_only_team_labels_and_edits(github):
    good = request_body(f"schemas: [{SCHEMA}]\nmax_prs: 1")
    result, calls = github(
        "observe",
        state={"campaigns": [], "requests": [{"issue": 7}, {"issue": 8}]},
        issues={
            "1": issue(good),
            "2": issue(good, labeler="aklr-shepherd", author="peer"),
            "3": issue(good, labeler="stranger"),
            "4": issue(good, editor="stranger", edited="2026-11-12T02:00:00Z"),
            "5": issue(good, editor="stranger", edited="2026-11-12T00:00:00Z"),
            "6": issue(request_body("schemas: [aklr/ops/fused_moe/impls/triton/x/1.json]")),
            "7": issue(good, state="CLOSED"),
            "8": issue(good, labeled=False),
            "9": issue(good, labeler="aklr-shepherd", author="stranger"),
            "10": issue(good, labeler="aklr-shepherd", editor="stranger", edited="2026-11-12T00:00:00Z"),
            "11": {**issue(good), "timelineItems": {"nodes": [
                {"createdAt": "2026-11-12T01:00:00Z", "actor": {"__typename": "User", "login": "aklr-shepherd"},
                 "label": {"name": "shepherd"}}]}},
        },
    )
    assert result.returncode == 0, result.stderr
    found = {r["issue"]: r for r in json.loads(result.stdout)["requests"]}
    assert {n: (r["status"], r["reason"]) for n, r in found.items()} == {
        1: ("active", None),
        2: ("active", None),
        4: ("paused", "editor_not_in_team"),
        5: ("active", None),
        6: ("paused", "schema_not_allowed:aklr/ops/fused_moe/impls/triton/x/1.json"),
        7: ("closed", None),
        8: ("unlabeled", None),
        10: ("paused", "editor_not_in_team"),
    }
    assert found[1]["schemas"] == [SCHEMA] and found[1]["max_prs"] == 1 and found[1]["max_campaigns"] == 1
    assert found[1]["prompt"] == "### Request\n\n\n\n### Notes\n\nTry split-K below M=64."
    assert not [c for c in calls if "-X" in c]


@pytest.mark.parametrize(
    "spec,reason",
    [
        ("schemas: [", "invalid_yaml"),
        ("- a", "invalid_yaml"),
        (f"schemas: [{SCHEMA}]\nbudget: 100", "unknown_keys:budget"),
        ("schemas: []", "schemas_not_a_list_of_unique_paths"),
        (f"schemas: [{SCHEMA}, {SCHEMA}]", "schemas_not_a_list_of_unique_paths"),
        (f"schemas: [{SCHEMA}]\nmax_prs: 0", "max_prs_not_a_positive_integer"),
        (f"schemas: [{SCHEMA}]\nmax_campaigns: true", "max_campaigns_not_a_positive_integer"),
    ],
)
def test_a_bad_request_block_pauses_with_its_reason(github, spec, reason):
    result, _ = github("observe", issues={"1": issue(request_body(spec))})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["requests"][0]["reason"] == reason


def test_progress_comment_is_written_once_and_edited_in_place(github):
    entry = {"issue": 1, "status": "active", "version": 2, "schemas": [SCHEMA], "max_prs": 1, "max_campaigns": 1}
    state = {
        "requests": [entry],
        "campaigns": [{"id": "c1", "request": 1, "schema": SCHEMA, "phase": "pr_open", "pr": 321}, {"id": "c2"}],
    }
    spoof = {"id": 5, "body": "<!-- shepherd-progress -->\nfake", "login": "aklr-shepherd[bot]", "type": "User"}
    result, calls = github("progress", "--issue", "1", state=state, comments=[spoof])
    assert json.loads(result.stdout) == {"comment": "created"}
    body = next(c for c in calls if "POST" in c)[-1].removeprefix("body=")
    assert "| `" + SCHEMA + "` | c1 | pr_open | #321 |" in body and "PRs 1 of 1, campaigns 1 of 1." in body
    mine = {"id": 9, "body": body, "login": "aklr-shepherd[bot]", "type": "Bot"}
    result, calls = github("progress", "--issue", "1", state=state, comments=[spoof, mine])
    assert json.loads(result.stdout) == {"comment": "unchanged"}
    assert not [c for c in calls if "-X" in c]
    result, calls = github("progress", "--issue", "1", state={**state, "requests": [{**entry, "status": "done"}]})
    assert json.loads(result.stdout) == {"comment": "updated"}
    assert next(c for c in calls if "PATCH" in c)[3] == "repos/AMD-AGI/AKLR/issues/comments/9"


def test_github_failure_exits_non_zero(github, tmp_path):
    (tmp_path / "bin" / "gh").write_text("#!/bin/sh\necho boom >&2\nexit 1\n")
    result, _ = github("observe")
    assert result.returncode == 1 and "boom" in json.loads(result.stderr)["error"]
