# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's FlyDSL GEMM kernels for SGLang's decode rows (128 to 384) at n x k = 6144 x 7168 and 7168 x 3584, whose
launches fold their K splits on scratch words compared with the launch's own tag: the scratch a call hands them (CPU,
through the public builder), and on gfx950 every workload on a poisoned allocator and on K partials that cancel."""

import ast
import inspect
import json
import sys

import pytest
import torch

from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.plugins.aiter.provider.gemm.contract import GEMM

# told apart from main's kernels of these shapes by their sources: the 6144 x 7168 one hands its launches a _scratch
# of its own, the 7168 x 3584 one runs band.py
_MARKS = {
    "gemm_n6144_k7168": ("kernel.py", "def _scratch("),
    "gemm_n7168_k3584": ("band.py", "def run_band("),
}


def _schemas():
    picked = []
    for path in committed_schema_paths(op_dir(GEMM.op_package)):
        if path.parent.name not in _MARKS:
            continue
        sources = {s["path"]: s["content"] for s in json.loads(path.read_text())["kernel"]["sources"]}
        name, mark = _MARKS[path.parent.name]
        if mark in sources.get(name, ""):
            picked.append(path)
    return picked


def _module(builder_registry, path, name="kernel"):
    """The kernel's entry module (or its sibling ``name``), as dispatch imports it."""
    schema = ContractRegistry.default().parse_schema(path.read_bytes(), str(path))
    entry = sys.modules[builder_registry.build(schema).metadata.misc["module_name"]]
    return entry if name == "kernel" else sys.modules[entry.__name__.rpartition(".")[0] + "." + name]


def _by_shape(paths):
    return {p.parent.name: p for p in paths}


def test_both_kernels_are_committed():
    paths = _schemas()
    assert sorted(p.parent.name for p in paths) == sorted(_MARKS)


def _allocations(source):
    """The torch.empty and torch.zeros calls of a function's source."""
    tree = ast.parse(inspect.cleandoc("\n" + source) if source.startswith(" ") else source)
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return calls & {"torch.empty", "torch.zeros"}


def test_a_6144_launch_compares_its_tag_only_with_counters_the_call_zeroed(builder_registry, monkeypatch):
    """The splits of a tile find which of them folds by swapping the launch's tag into a counter word: a word an
    earlier tensor left there could hold the tag. Every call zeroes the counters of the scratch it hands a launch; the
    partial planes before them each launch writes first."""
    pytest.importorskip("flydsl.expr")
    module = _module(builder_registry, _by_shape(_schemas())["gemm_n6144_k7168"])
    empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *a, **k: empty(*a, **k).fill_(-1)
    )  # memory as an earlier tensor left it
    ws = module._scratch(96, 64, "cpu")
    assert ws.shape == (96,) and (ws[:64] == -1).all() and (ws[64:] == 0).all()
    assert _allocations(inspect.getsource(module.run)) == set()


def test_a_7168_launch_takes_its_partners_plane_only_after_its_tag_came_back(builder_registry, monkeypatch):
    """A split-K class's lane swaps the launch's tag into its exchange word and reads its partner's plane only when the
    swap returns the tag: a word an earlier tensor left there could hold it. Every call sets the exchange words to all
    ones, which no tag (its low 8 bits zero) is; the partial planes before them each launch writes first."""
    pytest.importorskip("flydsl.expr")
    path = _by_shape(_schemas())["gemm_n7168_k3584"]
    kernel, band = _module(builder_registry, path), _module(builder_registry, path, "band")
    cfg = kernel._band_class(128)
    assert cfg[-1] == 2  # the class of 128 rows splits K
    handed = []
    # the launch stands in for the binary: no GPU, its arguments as run_band hands them
    monkeypatch.setattr(
        band, "compiled", lambda n, k, cfg, device: lambda out, a, b, ws, m, stream: handed.append(ws)
    )
    monkeypatch.setattr(band, "_ptr", lambda tensor, dtype: tensor)
    monkeypatch.setattr(band.fx, "Stream", lambda stream: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *a, **k: empty(*a, **k).fill_(0)
    )  # zero: a word that could hold a tag
    a = torch.zeros(128, 3584, dtype=torch.bfloat16)
    b = torch.zeros(7168, 3584, dtype=torch.bfloat16)
    band.run_band(a, b, torch.zeros(128, 7168, dtype=torch.bfloat16), cfg)
    (ws,) = handed
    planes = band.partial_plane_bytes(7168, 3584, *cfg) // 4
    assert ws.numel() * 4 == band.ws_bytes(7168, 3584, *cfg) and 0 < planes < ws.numel()
    words = ws.view(torch.int32)
    assert (words[:planes] == 0).all() and (words[planes:] == -1).all()


def _workloads():
    return [(path, w["axes"]["m"]) for path in _schemas() for w in json.loads(path.read_text())["workloads"]]


_WORKLOADS = pytest.mark.parametrize(
    "path, m", _workloads(), ids=lambda v: v.parent.name if hasattr(v, "parent") else str(v)
)


def _case_and_kernel(builder_registry, path, m):
    from tests.op_case import OpCase

    data = json.loads(path.read_text())
    contract = ContractRegistry.default().for_schema_json(data)
    case = OpCase(contract.point_from_json(data, {"m": m}))
    schema = contract.narrow(
        case.point, workloads=[contract.workload_for(case.point, {})], kernel=data["kernel"]
    )
    return case, builder_registry.build(schema)


@pytest.mark.gfx950
@_WORKLOADS
def test_a_kernel_matches_the_reference_on_a_poisoned_allocator(builder_registry, path, m):
    """At each workload, the call's scratch taken from blocks an earlier tensor filled with ones in every bit, then
    with zeros: the out within the contract's bound of the reference each time."""
    case, runnable = _case_and_kernel(builder_registry, path, m)
    args = case.inputs()
    call = dict(case.call({n: t.cuda() if torch.is_tensor(t) else t for n, t in args.items()}, "cuda"))
    call["out"] = torch.empty(
        case.contract.storage_shape_of("out", case.point), dtype=torch.bfloat16, device="cuda"
    )
    for byte in (255, 0):
        torch.empty(256 << 20, dtype=torch.uint8, device="cuda").fill_(byte)  # freed, kept by the allocator
        call["out"].fill_(float("nan"))
        runnable(**call)
        torch.cuda.synchronize()
        assert case.compare({"out": call["out"]}, args) == {}


@pytest.mark.gfx950
@_WORKLOADS
def test_a_kernel_keeps_k_partials_that_cancel_in_fp32(builder_registry, path, m):
    """Every row of a (256, 1, -256) and of b (256, 1, 256) at K 0, 64 and 1792, zeros elsewhere: each output is 1,
    while the K splits' partials, 65537 and -65536, cancel. A partial rounded below fp32 (65538 at 24 bits) would give
    2: each output is exactly 1."""
    case, runnable = _case_and_kernel(builder_registry, path, m)
    args = case.inputs()
    call = dict(case.call({n: t.cuda() if torch.is_tensor(t) else t for n, t in args.items()}, "cuda"))
    for name in ("a", "b"):
        call[name] = torch.zeros_like(call[name])
    for col, (x, y) in {0: (256.0, 256.0), 64: (1.0, 1.0), 1792: (-256.0, 256.0)}.items():
        call["a"][:, col], call["b"][:, col] = x, y
    call["out"] = torch.full(
        case.contract.storage_shape_of("out", case.point), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    runnable(**call)
    torch.cuda.synchronize()
    assert bool((call["out"] == 1).all()), call["out"].unique()[:8].tolist()
