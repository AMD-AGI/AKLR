"""The committed call tables against the committed schemas: every schema is called by some model."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "harness/automations/shepherd/skills/optimize-and-deliver-kernels/scripts/model_calls.py"


def test_every_schema_is_in_a_models_table():
    """A schema no table holds is never a candidate: the PR adding a schema adds its Points to the table of each
    model whose serving calls them (calls 1 until measured)."""
    read = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in ROOT.glob("harness/automations/shepherd/models/*.json")
    ]
    tables = [table["calls"] for table in read]
    assert tables
    # Counts per different units do not add up over the target models.
    assert len({table["per"] for table in read}) == 1
    assert all(calls > 0 for table in tables for points in table.values() for calls in points.values())
    # shortcut: rows of schemas no longer committed are not refused, since this branch's tree is older than the
    # main the tables were seeded from; refuse them once it is rebased
    unlisted = []
    for path in sorted(ROOT.glob("aklr/**/impls/*/*/*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        points = {",".join(f"{a}={v}" for a, v in w["axes"].items()) for w in schema["workloads"]}
        if not any(points & table.get(schema["name"], {}).keys() for table in tables):
            unlisted.append(path.relative_to(ROOT).as_posix())
    assert not unlisted


def test_record_writes_a_run_s_calls_and_keeps_the_points_it_did_not_reach(tmp_path):
    table = tmp_path / "harness/automations/shepherd/models/kimi-k3.json"
    table.parent.mkdir(parents=True)
    table.write_text(
        json.dumps(
            {"model": "kimi-k3", "per": "forward pass", "calls": {"a": {"m=1": 1, "m=2": 1}, "b": {"m=1": 1}}}
        )
    )
    export = (
        "served   gemm/impls/flydsl/a/1.json   1 workloads  m=1:122   (target mismatch: called on x, which falls back)\n"
        "added    gemm/impls/flydsl/c/2.json   2 workloads  m=4,n=8:7 :6   (sources empty: fill kernel.sources)\n"
        "5 calls had no point (fallback only)\n"
    )
    unit = ["--per", "forward pass", "--units", "2"]
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "record", "--model", "kimi-k3", "--source", "run", *unit],
        cwd=tmp_path,
        input=export,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"path": str(table.relative_to(tmp_path)), "measured": 3, "kept": 2}
    assert json.loads(table.read_text()) == {
        "model": "kimi-k3",
        "source": "run; the Points it did not reach keep their earlier value",
        "per": "forward pass",
        "calls": {"a": {"m=1": 61, "m=2": 1}, "b": {"m=1": 1}, "c": {"m=4,n=8": 3.5, "": 3}},
    }
    empty = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "record", "--model", "kimi-k3", "--source", "run", *unit],
        cwd=tmp_path,
        input="no calls.*.jsonl in d\n",
        capture_output=True,
        text=True,
    )
    assert empty.returncode == 2
    other = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "record", "--model", "kimi-k3", "--source", "run"]
        + ["--per", "layer", "--units", "1"],
        cwd=tmp_path,
        input=export,
        capture_output=True,
        text=True,
    )
    assert other.returncode == 2 and "counts per 'forward pass', not per 'layer'" in other.stderr
