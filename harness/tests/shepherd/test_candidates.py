"""Exercise candidate listing through its CLI on a scratch Git repository."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "automations/shepherd/skills/optimize-and-deliver-kernels/scripts/candidate.py"
)
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


def test_candidates_follow_strategies_and_filters(tmp_path):
    for path, schema in SCHEMAS.items():
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).write_text(json.dumps(schema))
    (tmp_path / "config.yaml").write_text(CONFIG)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "aklr"], check=True)
    subprocess.run([*git, "commit", "-qm", "schemas"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "list", "--config", "config.yaml", "--commit", head],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    listed = json.loads(result.stdout)
    assert [(c["path"], c["strategy"]) for c in listed["candidates"]] == [
        ("aklr/ops/gemm/impls/flydsl/manual/a.json", "manual"),
        ("aklr/ops/gemm/impls/triton/empty/a.json", "kernel-less"),
    ]
    assert {s["path"]: s["reason"] for s in listed["skipped"]} == {
        "aklr/ops/gemm/impls/flydsl/gone/a.json": "missing at this commit",
        "aklr/ops/gemm/impls/hip/empty/a.json": "language not in policy.backends",
        "aklr/ops/norm/impls/flydsl/empty/a.json": "op not in policy.allowed_ops",
    }
