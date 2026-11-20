# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Recordings, traffic/schemas.json, and the recorded calls bench measures workloads on."""

import json
from pathlib import Path

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.validation import traffic
from aklr.core.validation.stages._measure import RoundTimes, summed_rounds

ROOT = Path(__file__).resolve().parents[3]
TOPK = (
    ROOT
    / "aklr/plugins/sglang/layers/dsa_backend/impls/hip/dsa_topk_transform_topk2048/1a5994b58e0a84ae.json"
)


def load(path):
    registry = ContractRegistry.default()
    schema = registry.parse_schema(path.read_bytes(), str(path))
    return schema, registry.get(schema.contract_id)


def test_a_recording_reads_back_as_written(tmp_path):
    (tmp_path / "traffic").mkdir()
    steps = (("decode", ((6, 100), (6, 7))), ("prefill", ((50, 50),)))
    traffic.write(tmp_path / "traffic" / "x.traffic.jsonl.xz", {"draft_tokens": 6}, steps)
    assert traffic.read(tmp_path, "x") == ({"draft_tokens": 6}, steps)


def test_every_listed_recording_makes_calls_of_its_schemas_op_inside_the_contract():
    names = {json.loads(p.read_text())["name"]: p for p in ROOT.glob("aklr/**/impls/*/*/*.json")}
    for schema_name, listing in traffic.schemas(ROOT).items():
        schema, contract = load(names[schema_name])
        op = traffic.op_name(contract)
        assert op in traffic.OPS, f"{schema_name}: no recorded calls of {op}"
        for name in listing.recordings:
            calls = traffic.calls(ROOT, name, op)
            assert calls, f"{schema_name}: {name} has no call of {op}"
            for axes in {tuple(sorted(c["axes"].items())) for c in calls}:
                point = schema.point_with_contract_id(
                    schema.workloads[0].axes | dict(axes), contract.contract_id
                )
                assert contract.domain.contains(point), f"{schema_name}: {name} calls at {dict(axes)}"


def test_a_recording_whose_table_is_narrower_than_a_request_is_refused(tmp_path):
    (tmp_path / "traffic").mkdir()
    header = {"draft_tokens": 6, "page_size": 64, "table_pages": 2}
    traffic.write(tmp_path / "traffic" / "x.traffic.jsonl.xz", header, (("decode", ((6, 200),)),))
    with pytest.raises(ValueError, match="a request of 200 keys, the page table 128 wide"):
        traffic.calls(tmp_path, "x", "dsa_topk_transform")


def test_a_verifys_rows_share_their_requests_table_and_see_one_more_key_each():
    call = {"axes": {"n_slots": 1000}, "rows_per_request": 3, "kv": [130, 70]}
    _, contract = load(TOPK)
    inputs = traffic.tensors(contract, call, torch.Generator().manual_seed(0))
    assert inputs["lengths"].tolist() == traffic.row_kv(call) == [128, 129, 130, 68, 69, 70]
    assert inputs["cu_seqlens_q"].tolist() == [0, 3, 6]
    table = inputs["page_table_size_1"]
    assert table.shape == (2, 1000) and all(len(set(row.tolist())) == 1000 for row in table)


def listed(root, samples, override=None):
    """A repository at ``root`` whose top-k schema is measured on ten verifies, of 10 to 100 keys."""
    (root / "traffic").mkdir()
    header = {"draft_tokens": 6, "page_size": 64, "table_pages": 2}
    steps = [("decode", ((6, kv),)) for kv in (70, 10, 100, 40, 20, 90, 30, 60, 50, 80)]
    traffic.write(root / "traffic" / "x.traffic.jsonl.xz", header, steps)
    entry = {"recordings": ["x"]} | ({} if override is None else {"samples": override})
    listing = {"samples": samples, "schemas": {"dsa_topk_transform_topk2048": entry}}
    (root / "traffic" / "schemas.json").write_text(json.dumps(listing))


@pytest.mark.parametrize(
    ("samples", "override", "kv"),
    [
        (1, None, [60]),
        (3, None, [20, 50, 90]),
        (3, 2, [30, 80]),
        (20, None, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    ],
    ids=["median", "three parts", "the schema's own", "more parts than calls"],
)
def test_a_workload_takes_each_parts_median_of_its_keys_calls_by_work(tmp_path, samples, override, kv):
    listed(tmp_path, samples, override)
    schema, contract = load(TOPK)
    six_rows = next(w.axes for w in schema.workloads if w.axes["n_tokens"] == 6)
    calls = traffic.recorded_calls(tmp_path, schema, contract, six_rows)
    assert [c["kv"] for c in calls] == [[k] for k in kv]


def test_a_workload_no_recorded_call_has_takes_none(tmp_path):
    listed(tmp_path, 3)
    schema, contract = load(TOPK)
    twelve_rows = next(w.axes for w in schema.workloads if w.axes["n_tokens"] == 12)
    assert traffic.recorded_calls(tmp_path, schema, contract, twelve_rows) == ()


def test_a_schema_not_listed_is_drawn():
    schema, contract = load(TOPK)
    unlisted = schema.model_copy(update={"name": "not_listed"})
    assert traffic.recorded_calls(ROOT, unlisted, contract, schema.workloads[-1].axes) == ()


def test_samples_rounds_sum_at_the_rounds_every_sample_reached():
    def rounds(*ms):
        return tuple(
            RoundTimes(
                mode="graph",
                round_index=i,
                order=("baseline", "candidate"),
                milliseconds={"baseline": b, "candidate": c},
            )
            for i, (b, c) in enumerate(ms)
        )

    summed = summed_rounds([rounds((1.0, 2.0), (1.0, 2.0), (1.0, 2.0)), rounds((3.0, 1.0), (5.0, 1.0))])
    assert [r.milliseconds for r in summed] == [
        {"baseline": 4.0, "candidate": 3.0},
        {"baseline": 6.0, "candidate": 3.0},
    ]
    assert summed_rounds([rounds((1.0, 2.0))]) == rounds((1.0, 2.0))


def test_a_call_is_described_by_its_requests_and_the_axes_it_changed():
    call = {"axes": {"n_tokens": 12, "n_seqs": 2, "n_slots": 1000}, "rows_per_request": 6, "kv": [130, 70]}
    workload = {"n_tokens": 12, "n_seqs": 12, "n_slots": 300}
    assert traffic.describe(call, workload) == "2 requests x 6 rows, kv 130 70; n_seqs=2 n_slots=1000"
