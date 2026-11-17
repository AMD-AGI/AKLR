"""The committed call tables against the committed schemas: every schema is called by some model."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_every_schema_is_in_a_models_table():
    """A schema no table holds is never a candidate: the PR adding a schema adds its Points to the table of each
    model whose serving calls them (calls 1 until measured)."""
    tables = [
        json.loads(path.read_text(encoding="utf-8"))["calls"]
        for path in ROOT.glob("harness/automations/shepherd/models/*.json")
    ]
    assert tables
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
