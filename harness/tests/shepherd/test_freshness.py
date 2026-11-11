"""Exercise the evidence freshness rule through its CLI on a scratch Git repository."""

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "automations/shepherd/skills/shepherd/scripts/freshness.py"
SCHEMA = "aklr/ops/gemm/impls/flydsl/gemm_a8w8_n512/0123abcd.json"


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def commit(repo, *paths):
    for path in paths:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(uuid.uuid4().hex)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", "change")
    return git(repo, "rev-parse", "HEAD")


def check(repo, base, main, schema=SCHEMA):
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--schema", schema, "--base", base, "--main", main],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, json.loads(result.stdout or result.stderr)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "test")
    commit(tmp_path, SCHEMA, "aklr/core/x.py", "aklr/ops/gemm/reference.py", "pyproject.toml")
    return tmp_path


@pytest.mark.parametrize(
    "path",
    [
        "aklr/ops/gemm/impls/flydsl/gemm_bf16_n512/ffff.json",
        "aklr/ops/gemm/impls/triton/gemm_bf16_n512/kernel.py",
        "aklr/ops/fused_moe/impls/triton/moe_x/kernel.py",
        "aklr/plugins/sglang/models/x/impls/triton/y/1.json",
        "aklr/core/README.md",
        "harness/skills/x/SKILL.md",
        "docs/guide.txt",
        ".github/workflows/lint.yaml",
        ".gitignore",
    ],
)
def test_unrelated_change_keeps_evidence_fresh(repo, path):
    base = git(repo, "rev-parse", "HEAD")
    main = commit(repo, path)
    assert check(repo, base, main) == (0, {"base": base, "main": main, "relevant": []})


@pytest.mark.parametrize(
    "path",
    [
        SCHEMA,
        "aklr/ops/gemm/impls/triton/gemm_a8w8_n512/kernel.py",
        "aklr/ops/gemm/reference.py",
        "aklr/ops/__init__.py",
        # Another op's code can be imported by this one's reference or baseline (mega_moe uses fused_moe's).
        "aklr/ops/fused_moe/reference.py",
        # The CI workflow pins the baseline every Point is measured against.
        ".github/workflows/ci.yaml",
        "aklr/core/x.py",
        "aklr/__init__.py",
        "pyproject.toml",
    ],
)
def test_relevant_change_makes_evidence_stale(repo, path):
    base = git(repo, "rev-parse", "HEAD")
    main = commit(repo, path, "docs/unrelated.md")
    assert check(repo, base, main) == (1, {"base": base, "main": main, "relevant": [path]})


def test_new_submodule_is_relevant(repo, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    git(other, "init", "-q")
    git(other, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    base = git(repo, "rev-parse", "HEAD")
    git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{git(other, 'rev-parse', 'HEAD')},third_party/lib",
    )
    git(repo, "commit", "-q", "-m", "submodule")
    code, report = check(repo, base, git(repo, "rev-parse", "HEAD"))
    assert (code, report["relevant"]) == (1, ["third_party/lib"])


def test_rewritten_history_is_stale(repo):
    base = commit(repo, "aklr/core/y.py")
    git(repo, "reset", "-q", "--hard", "HEAD~1")
    main = commit(repo, "docs/z.md")
    code, report = check(repo, base, main)
    assert code == 1 and "stale" in report


def test_bad_input_is_an_error_not_fresh(repo):
    head = git(repo, "rev-parse", "HEAD")
    assert check(repo, head, head, schema="aklr/ops/gemm/reference.py")[0] == 2
    assert check(repo, "0" * 40, head)[0] == 2


def test_removed_submodule_is_relevant(repo, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    git(other, "init", "-q")
    git(other, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{git(other, 'rev-parse', 'HEAD')},third_party/lib",
    )
    git(repo, "commit", "-q", "-m", "submodule")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "rm", "-q", "--cached", "third_party/lib")
    git(repo, "commit", "-q", "-m", "drop submodule")
    code, report = check(repo, base, git(repo, "rev-parse", "HEAD"))
    assert (code, report["relevant"]) == (1, ["third_party/lib"])
