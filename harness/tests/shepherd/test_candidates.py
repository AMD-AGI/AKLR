"""Exercise candidate listing through its CLI on a scratch Git repository."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "automations/shepherd/skills/optimize-and-deliver-kernels/scripts/candidate.py"
)
NOT_A_SCHEMA = "not a schema: a JSON object whose kernel is an object"
CONFIG = """
policy:
  strategies:
    - type: manual
      schemas: [aklr/ops/gemm/impls/flydsl/manual/a.json, aklr/ops/gemm/impls/flydsl/gone/a.json]
    - type: kernel-less
  allowed_ops: [gemm, moe]
  backends: [flydsl, triton]
"""
SCHEMAS = {
    "aklr/ops/gemm/impls/flydsl/manual/a.json": {"kernel": {"sources": ["k.py"]}},
    "aklr/ops/gemm/impls/triton/empty/a.json": {"kernel": {"language": "triton", "sources": []}},
    "aklr/ops/gemm/impls/hip/empty/a.json": {"kernel": None},
    "aklr/ops/gemm/impls/flydsl/served/a.json": {"kernel": {"sources": ["k.py"]}},
    "aklr/ops/norm/impls/flydsl/empty/a.json": {"kernel": None},
}


def listed(tmp_path, commit, campaigns, **more):
    (tmp_path / "ledger.json").write_text(json.dumps({"state": {"campaigns": campaigns, **more}}))
    command = ["list", "--config", "config.yaml", "--commit", commit, "--ledger", "ledger.json"]
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_candidates_follow_strategies_filters_and_the_ledger(tmp_path):
    for path, schema in SCHEMAS.items():
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).write_text(json.dumps(schema))
    # One unreadable file on main must not hide the other candidates.
    for name, text in (("broken", "{broken"), ("list", "[]"), ("text", '{"kernel": "x"}')):
        (tmp_path / f"aklr/ops/gemm/impls/flydsl/{name}").mkdir()
        (tmp_path / f"aklr/ops/gemm/impls/flydsl/{name}/a.json").write_text(text)
    (tmp_path / "config.yaml").write_text(CONFIG)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "aklr"], check=True)
    subprocess.run([*git, "commit", "-qm", "schemas"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    first = listed(tmp_path, head, [])
    assert [(c["path"], c["strategy"]) for c in first["candidates"]] == [
        ("aklr/ops/gemm/impls/flydsl/manual/a.json", "manual"),
        ("aklr/ops/gemm/impls/triton/empty/a.json", "kernel-less"),
    ]
    assert {s["path"]: s["reason"] for s in first["skipped"]} == {
        "aklr/ops/gemm/impls/flydsl/broken/a.json": NOT_A_SCHEMA,
        "aklr/ops/gemm/impls/flydsl/gone/a.json": "missing at this commit",
        "aklr/ops/gemm/impls/flydsl/list/a.json": NOT_A_SCHEMA,
        "aklr/ops/gemm/impls/flydsl/text/a.json": NOT_A_SCHEMA,
        "aklr/ops/gemm/impls/hip/empty/a.json": "language not in policy.backends",
        "aklr/ops/norm/impls/flydsl/empty/a.json": "op not in policy.allowed_ops",
    }

    manual, empty = first["candidates"]

    def reasons(*campaigns):
        result = listed(tmp_path, head, [{"id": "c", "schema_path": manual["path"], **c} for c in campaigns])
        skipped = {s["path"]: s["reason"] for s in result["skipped"]}
        kept = {c["path"]: c.get("skip_regimes") for c in result["candidates"]}
        return skipped.get(manual["path"]), kept

    ended = {"phase": "ended", "input_sha256": manual["input_sha256"]}
    assert reasons({"phase": "pr_open", "regime": "small"})[0] == "campaign c is undelivered"
    assert reasons({**ended, "outcome": "merged"})[0] == "already delivered by campaign c"
    # The input changed since: a candidate again. A cooldown that passed lifts; an unreadable one does not.
    past = {**ended, "outcome": "failed", "cooldown_until": "2000-01-01T00:00:00+00:00"}
    for again in ({**ended, "outcome": "merged", "input_sha256": "other"}, past):
        assert reasons(again) == (None, {manual["path"]: None, empty["path"]: None}), again
    for cooldown in ("2100-01-01T00:00:00+00:00", "soon"):
        assert reasons({**past, "cooldown_until": cooldown})[0].startswith("cooldown until"), cooldown
    # A campaign of one regime closes that regime only.
    reason, kept = reasons({**ended, "outcome": "merged", "regime": "small"})
    assert reason is None and kept[manual["path"]][0]["regime"] == "small"
    # A blocked contract blocks the whole op until its key changes.
    block = {"contract": "gemm@1.0", "op": "gemm", "lifts_when_changed": manual["contract_key"]}
    assert listed(tmp_path, head, [], blocked_contracts=[block])["candidates"] == []
    block["lifts_when_changed"] = {**manual["contract_key"], "runtime_revision": "older"}
    assert len(listed(tmp_path, head, [], blocked_contracts=[block])["candidates"]) == 2
    # A measurement update under impls/ does not lift a block; a change of the op's code does.
    (tmp_path / "aklr/ops/gemm/impls/flydsl/served/a.json").write_text("{}")
    (tmp_path / "aklr/ops/gemm/op.py").write_text("x")
    for added, same in (("aklr/ops/gemm/impls", True), ("aklr/ops/gemm/op.py", False)):
        subprocess.run([*git, "add", added], check=True)
        subprocess.run([*git, "commit", "-qm", added], check=True)
        key = listed(tmp_path, "HEAD", [])["candidates"][0]["contract_key"]
        assert (key == manual["contract_key"]) is same, added
    assert (
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(SCRIPT),
                "list",
                "--config",
                "config.yaml",
                "--commit",
                "nope",
                "--ledger",
                "ledger.json",
            ],
            cwd=tmp_path,
        ).returncode
        == 2
    )


def test_a_schema_whose_workloads_all_have_a_sibling_kernel_is_not_kernel_less(tmp_path):
    one, two = {"axes": {"m": 1}}, {"axes": {"m": 2}}
    schemas = {
        "kernel": {
            "kernel": {"sources": ["k.py"]},
            "workloads": [{**one, "speedup": 2.0, "accuracy": "passed"}],
        },
        "served": {"kernel": None, "workloads": [one]},
        "partly": {"kernel": None, "workloads": [one, two]},
    }
    directory = tmp_path / "aklr/ops/gemm/impls/flydsl/name"
    directory.mkdir(parents=True)
    for name, schema in schemas.items():
        (directory / f"{name}.json").write_text(json.dumps(schema))
    (tmp_path / "config.yaml").write_text(CONFIG)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "aklr"], check=True)
    subprocess.run([*git, "commit", "-qm", "schemas"], check=True)
    result = listed(tmp_path, "HEAD", [])
    kernel_less = [c["path"] for c in result["candidates"] if c["strategy"] == "kernel-less"]
    assert kernel_less == ["aklr/ops/gemm/impls/flydsl/name/partly.json"]


def test_a_manual_schema_is_kept_wherever_its_strategy_stands(tmp_path):
    path = "aklr/ops/gemm/impls/flydsl/name/a.json"
    (tmp_path / path).parent.mkdir(parents=True)
    (tmp_path / path).write_text(json.dumps({"kernel": {"sources": ["k.py"]}}))
    (tmp_path / "config.yaml").write_text(
        f"policy:\n  strategies:\n    - type: kernel-less\n    - type: manual\n      schemas: [{path}]\n"
        "  allowed_ops: [all]\n  backends: [flydsl]\n"
    )
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "aklr"], check=True)
    subprocess.run([*git, "commit", "-qm", "schemas"], check=True)
    assert [(c["path"], c["strategy"]) for c in listed(tmp_path, "HEAD", [])["candidates"]] == [
        (path, "manual")
    ]


def test_an_open_issue_request_lists_its_schemas_with_the_issue(tmp_path):
    paths = [f"aklr/ops/gemm/impls/flydsl/{name}/a.json" for name in ("open", "stopped", "closed")]
    for path in paths:
        (tmp_path / path).parent.mkdir(parents=True)
        # A request names its schema as a manual strategy does: a kernel does not drop it.
        (tmp_path / path).write_text(json.dumps({"kernel": {"sources": ["k.py"]}}))
    (tmp_path / "state/requests").mkdir(parents=True)
    for issue, (path, more) in enumerate(
        zip(
            paths,
            ({}, {"stopped_at": "2026-01-01T00:00:00+00:00"}, {"closed_at": "2026-01-01T00:00:00+00:00"}),
        ),
        start=7,
    ):
        request = {"issue": issue, "schemas": [path], **more}
        (tmp_path / f"state/requests/{issue}.json").write_text(json.dumps(request))
    (tmp_path / "config.yaml").write_text(
        f"execution: {{state_dir: {tmp_path}/state}}\n"
        "policy: {strategies: [{type: issue}], allowed_ops: [all], backends: [flydsl]}\n"
    )
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "aklr"], check=True)
    subprocess.run([*git, "commit", "-qm", "schemas"], check=True)
    (candidate,) = listed(tmp_path, "HEAD", [])["candidates"]
    assert (candidate["path"], candidate["strategy"], candidate["issue"]) == (paths[0], "issue", 7)
