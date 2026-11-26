# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3.1's DSA indexer prefill on vLLM to AKLR's kimi_k3_fp4_indexer_prefill: what the route takes and how it hands
each query tile over, at K3.1's shapes, to the committed schemas' keys (CPU); the reference's quantization and scores
against the contract's text (CPU); the baseline's copy of the fork's quantizer against the fork's (GPU)."""

import struct
from types import SimpleNamespace

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.dispatch import DispatchTable, FallbackReason
from aklr.plugins.vllm.models.kimi_k3.fp4_indexer import api
from aklr.plugins.vllm.models.kimi_k3.fp4_indexer.reference import mxfp4_decoded
from tests.op_case import OpCase

HEADS, TOPK = 32, 2048
# Kimi-K3.1 c1 under DCP8: 128k cached + 8k new tokens, the 8k in two tiles of 7710 and 482 rows (vLLM's 4 GiB
# budget), each rank's share 964 and 61 rows, against all 139264 keys
K31_PAGES = 2176
K31_TILES = ((0, 964), (964, 1025))


@pytest.fixture
def op(monkeypatch):
    """The op replaced: each call recorded and its tile written with 7."""
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        args[6].fill_(7)
        return args[6]

    monkeypatch.setattr(api, "kimi_k3_fp4_indexer_prefill", fake)
    return calls


def chunk_call(tiles=K31_TILES, pages=K31_PAGES, world=8):
    """prefill_indexer's arguments as prefill_chunks passes them for one chunk: this rank's rows first..first +
    per_rank in query tiles, the chunk's staged pages."""
    first, last = tiles[0][0], tiles[-1][1]
    rows = last + 3
    queries = [
        SimpleNamespace(
            start=start,
            end=end,
            row_starts=torch.zeros(end - start, dtype=torch.int32),
            row_ends=torch.full((end - start,), pages * 64, dtype=torch.int32),
        )
        for start, end in tiles
    ]
    return dict(
        q=torch.zeros(rows, HEADS, 128, dtype=torch.bfloat16),
        weights=torch.zeros(rows, HEADS, dtype=torch.bfloat16),
        cache=None,
        out=torch.full((rows, TOPK), -1, dtype=torch.int32),
        chunk=SimpleNamespace(token_start=0, token_end=rows),
        world=world,
        interleave=1,
        weight_scale=32**-0.5 * 128**-0.5,
        staged_keys=(
            torch.zeros(pages, 1, 4, 64, 16, dtype=torch.uint8),
            torch.zeros(pages, 1, 4, 64, dtype=torch.uint8),
        ),
        pending=[],
        plan=SimpleNamespace(first=first, per_rank=last - first + 2, queries=queries),
    )


@pytest.mark.parametrize(
    "changed",
    [{"staged_keys": None}, {"pending": None}, {"plan": None}, {"world": 1}],
    ids=["stages_its_keys", "gathers_its_rows", "plans_its_tiles", "no_dcp"],
)
def test_what_no_contract_states_goes_back_to_vllm(op, changed):
    call = chunk_call() | changed
    pending = call["pending"]
    assert api._prefill_indexer(**call) is FallbackReason.OUT_OF_CONTRACT
    assert op == [] and pending in (None, [])


def test_each_tile_is_one_call_and_the_rows_go_to_prefill_chunks_gather(op):
    call = chunk_call()
    assert api._prefill_indexer(**call) is None
    payload, scales = call["staged_keys"]
    ((token_start, token_end, local_out),) = call["pending"]
    assert (token_start, token_end) == (0, call["chunk"].token_end)
    plan = call["plan"]
    assert local_out.shape == (plan.per_rank, TOPK) and local_out.dtype == torch.int32
    for (args, kwargs), query in zip(op, plan.queries, strict=True):
        q, weights, key_values, key_scales, starts, ends, indices = args
        assert (
            torch.equal(q, call["q"][query.start : query.end])
            and q.data_ptr() == call["q"][query.start].data_ptr()
        )
        assert weights.data_ptr() == call["weights"][query.start].data_ptr() and weights.shape == (
            q.shape[0],
            HEADS,
        )
        assert key_values.shape == (K31_PAGES, 4, 64, 16) and key_values.data_ptr() == payload.data_ptr()
        assert key_scales.shape == (K31_PAGES, 4, 64) and key_scales.data_ptr() == scales.data_ptr()
        assert starts is query.row_starts and ends is query.row_ends  # aligned: handed over as they are
        assert indices.data_ptr() == local_out[query.start - plan.first].data_ptr()
        assert kwargs == {"weight_scale": call["weight_scale"]}
    # the tiles' rows hold what the op wrote; the rank's rows past them stay -1, as vLLM leaves them
    assert (local_out[: K31_TILES[-1][1]] == 7).all() and (local_out[K31_TILES[-1][1] :] == -1).all()


def test_row_bounds_off_16_bytes_are_handed_over_aligned(op):
    """A rank's rows of a 482-row tile start at row 61 x rank of the chunk's bounds: not 16-byte aligned, which AKLR's
    dispatch declines; the route copies them."""
    call = chunk_call(tiles=((0, 61),))
    (query,) = call["plan"].queries
    bounds = torch.zeros(2, 64, dtype=torch.int32)
    bounds[0, 1:62], bounds[1, 1:62] = 0, K31_PAGES * 64
    query.row_starts, query.row_ends = bounds[0, 1:62], bounds[1, 1:62]
    assert api._prefill_indexer(**call) is None
    ((args, _),) = op
    for given, handed in ((query.row_starts, args[4]), (query.row_ends, args[5])):
        assert handed.data_ptr() % 16 == 0 and handed is not given and torch.equal(handed, given)


def test_a_tile_the_op_does_not_serve_sends_the_chunk_back(monkeypatch):
    served = iter([None, FallbackReason.NOT_MEASURED])

    def fake(*args, **kwargs):
        return next(served) or args[6]

    monkeypatch.setattr(api, "kimi_k3_fp4_indexer_prefill", fake)
    call = chunk_call()
    assert api._prefill_indexer(**call) is FallbackReason.NOT_MEASURED
    assert call["pending"] == []


def _served_table(monkeypatch):
    """The committed schemas of the op as AKLR serves them, every workload measured accurate at 2x."""
    registry = ContractRegistry.default()
    contract = registry.get("kimi_k3_fp4_indexer_prefill@1.0")
    schemas, errors = registry.read_schemas(contract.op_package)
    assert errors == [] and schemas
    runnable = SimpleNamespace(metadata=SimpleNamespace(misc={}))
    measured = [
        (
            schema,
            runnable,
            [w.model_copy(update={"accuracy": "passed", "speedup": 2.0}) for w in schema.workloads],
        )
        for _, schema in schemas
    ]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return (
        contract,
        runnable,
        DispatchTable.from_schemas(contract, torch.device("cpu"), measured, min_speedup=1.0),
    )


@pytest.mark.parametrize(
    ("tiles", "pages", "served"),
    [
        (K31_TILES, K31_PAGES, True),  # 128k + 8k: 964 and 61 rows
        (((0, 2048),), 256, True),  # a 64k prompt's first 16k chunk (its context unmeasured)
        (((0, 1500),), 1024, False),  # rows no workload holds
        (((0, 3000),), 1024, False),  # past the most rows the schemas measure (2048)
    ],
    ids=["128k_8k", "first_16k_chunk", "1500_rows_64k", "3000_rows"],
)
def test_the_routes_tiles_at_k31_shapes_key_to_the_committed_schemas(monkeypatch, tiles, pages, served):
    contract, runnable, table = _served_table(monkeypatch)
    keyed = []

    def lookup(*args, **kwargs):
        call = dict(zip((p.name for p in contract.api_params), args, strict=False)) | kwargs
        keyed.append(table.lookup(call)[2])
        return args[6]

    monkeypatch.setattr(api, "kimi_k3_fp4_indexer_prefill", lookup)
    assert api._prefill_indexer(**chunk_call(tiles, pages)) is None
    assert keyed and all((k is runnable) is served for k in keyed)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _vllm_group(values):
    """One group of 32 through vLLM's _quantize, in scalar float32 steps, decoded."""
    scale = max(_f32(_f32(max(abs(v) for v in values)) / 6.0), _f32(1e-4))
    bits = struct.unpack("i", struct.pack("f", scale))[0]
    exponent = min(((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0), 254)
    power = struct.unpack("f", struct.pack("i", exponent << 23))[0]
    out = []
    for v in values:
        x = _f32(v / power)
        a = abs(x)
        code = (a > 0.25) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0)
        out.append((-1.0 if x < 0 and code else 1.0) * _E2M1[code] * power)
    return out


def test_the_reference_quantizes_the_queries_as_vllm():
    """Every midpoint ties to the even code; an all-zero group, tiny and huge groups, a power-of-two amax / 6."""
    x = torch.randn(16, 128, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16).float()
    midpoints = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    x[0, :32] = torch.tensor([6.0, *midpoints, *(-m for m in midpoints), *[0.0] * 17])
    x[1, :32] = 0.0
    x[2, :32] *= 1e-7
    x[3, :32] *= 1e30
    x[4, :32] = torch.tensor([3.0 * 2**-5] + [0.1] * 31)
    decoded = mxfp4_decoded(x)
    for row in range(x.shape[0]):
        for group in range(4):
            columns = slice(32 * group, 32 * group + 32)
            assert decoded[row, columns].tolist() == _vllm_group(x[row, columns].tolist()), (row, group)


def _scores_from_the_text(args):
    """Each row's float64 score of each key, from the contract's text: page layout, scales, relu, weights."""
    q = mxfp4_decoded(args["q"].float()).double()
    values = args["key_values"].view(torch.uint8).long()
    nibbles = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2)
    table = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=torch.float64)
    scales = 2.0 ** (args["key_scales"].view(torch.uint8).double() - 127)
    # key t: page t // 64, slot s = t % 64; group g's values at [page, g, s], its scale at [page, g, (s % 16) * 4 + s // 16]
    t = torch.arange(values.shape[0] * 64)
    page, slot = t // 64, t % 64
    group_values = table[nibbles[page, :, slot]]  # [keys, 4, 32]
    group_scales = scales[page, :, (slot % 16) * 4 + slot // 16]  # [keys, 4]
    keys = (group_values * group_scales[..., None]).flatten(1)
    dots = torch.einsum("rhd,td->rht", q, keys).relu()
    return torch.einsum("rht,rh->rt", dots, args["weights"].double()) * args["weight_scale"]


def test_the_reference_selects_the_keys_the_contract_text_scores_highest():
    case = OpCase.of("kimi_k3_fp4_indexer_prefill@1.0")
    args = case.inputs()
    scores = _scores_from_the_text(args)
    ids = []
    for row, (start, end) in enumerate(
        zip(args["row_starts"].tolist(), args["row_ends"].tolist(), strict=True)
    ):
        ranked = scores[row, start:end].argsort(descending=True, stable=True)[:TOPK]
        ids.append(ranked)
    expected = torch.stack(ids).to(torch.int32)
    assert case.compare({"indices": expected}, args) == {}


def test_a_short_rows_keys_come_in_order():
    """A row of fewer keys than it selects holds all of them, ascending, then -1."""
    case = OpCase.of("kimi_k3_fp4_indexer_prefill@1.0")
    args = case.inputs()
    args["row_starts"] = torch.tensor([0, 5, 100, 0], dtype=torch.int32)
    args["row_ends"] = torch.tensor([1, 2054, 2100, 2048], dtype=torch.int32)
    out = case.reference_outputs(args)["indices"]
    assert out[0, 0] == 0 and (out[0, 1:] == -1).all()
    assert out[2, :2000].tolist() == list(range(2000)) and (out[2, 2000:] == -1).all()
    assert out[3].tolist() == list(range(2048))  # exactly topk keys: all of them
    # one key more than it selects: by score, one key left out
    assert len(set(out[1].tolist())) == 2048 and 0 <= out[1].min() and out[1].max() <= 2048


def test_a_rows_keys_are_compared_in_any_order():
    """vLLM's top-k writes a row's keys in any order: a full row's keys shuffled, with two whose scores round to one
    float16 swapped, and a short row's keys shuffled among its -1s pass; a full row's key replaced by one outside its
    top-k, or a short row's key by -1, does not."""
    case = OpCase.of("kimi_k3_fp4_indexer_prefill@1.0")
    args = case.inputs()
    args["row_ends"][1] = args["row_starts"][1] + 100
    expected = case.reference_outputs(args)
    ref, scores = expected["indices"], expected["indices_scores"]
    generator = torch.Generator().manual_seed(0)
    shuffled = ref.clone()
    for row in range(ref.shape[0]):
        shuffled[row] = ref[row, torch.randperm(ref.shape[1], generator=generator)]
    assert case.compare({"indices": shuffled}, args) == {}
    row0 = scores[0, ref[0].long()]
    # two neighbours in row 0's order whose scores round to one float16
    tie = int((row0[1:] == row0[:-1]).nonzero()[0, 0])
    swapped = ref.clone()
    swapped[0, tie], swapped[0, tie + 1] = ref[0, tie + 1], ref[0, tie]
    assert case.compare({"indices": swapped}, args) == {}
    # row 0's best key for the worst key it left out
    left_out = scores[0].float().clone()
    left_out[ref[0].long()] = torch.inf
    wrong = ref.clone()
    wrong[0, 0] = int(left_out.argmin())
    assert set(case.compare({"indices": wrong}, args)) == {"indices"}
    missing = ref.clone()
    missing[1, 0] = -1
    assert set(case.compare({"indices": missing}, args)) == {"indices"}


@pytest.mark.gfx950
@pytest.mark.parametrize("rows", [1, 7, 300])
def test_the_baselines_quantizer_is_the_forks(rows):
    """The copy of the fork's Triton quantizer the baseline runs writes the fork's bytes (where the fork is
    installed: Kimi-K3.1's vLLM image)."""
    fork = pytest.importorskip("vllm.models.kimi_k3.amd.ops.fp4_indexer")
    from aklr.plugins.vllm.models.kimi_k3.fp4_indexer._triton import quantize_queries

    q = torch.randn(rows, HEADS, 128, generator=torch.Generator().manual_seed(rows)).to(torch.bfloat16).cuda()
    ours, theirs = quantize_queries(q), fork.quantize_queries(q)
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1], theirs[1])
