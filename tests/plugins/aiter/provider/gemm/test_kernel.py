# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's FlyDSL GEMM for n 2112 and 6288 at k 7168, whose launches fold their K splits on scratch words compared
with the launch's own tag: the scratch a call hands them (CPU, through the public builder)."""

import ast
import inspect
import json
import sys

import pytest
import torch

from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.plugins.aiter.provider.gemm.contract import GEMM

# the K3 kernel of these shapes: its tiles' fold counters (kernel.py, tiled.py) and skw.py's tag slots, on the
# scratch its entry hands each launch (_scratch). main's sibling of it has skw.py too and no scratch of its own; it
# is told apart by that, not by its sizes (a schema may hold any m)
_NAMES = ("gemm_n2112_k7168", "gemm_n6288_k7168")


def _schemas():
    picked = []
    for path in committed_schema_paths(op_dir(GEMM.op_package)):
        data = json.loads(path.read_text())
        sources = {s["path"]: s["content"] for s in data["kernel"]["sources"]}
        if (
            path.parent.name in _NAMES
            and "skw.py" in sources
            and any("def _scratch(" in text for text in sources.values())
        ):
            picked.append(path)
    return picked


def _module(builder_registry, path):
    """The kernel's entry module, as dispatch imports it."""
    schema = ContractRegistry.default().parse_schema(path.read_bytes(), str(path))
    return sys.modules[builder_registry.build(schema).metadata.misc["module_name"]]


def test_the_k3_kernel_is_committed():
    assert {p.parent.name for p in _schemas()} == set(_NAMES)


@pytest.mark.parametrize("path", _schemas(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_a_launch_compares_its_tag_only_with_words_the_call_zeroed(builder_registry, monkeypatch, path):
    """The splits of a tile find which of them folds by swapping the launch's tag (its dispatch id and queue) into a
    counter word: a word an earlier tensor left there could hold the tag, and a split would fold the planes before the
    others wrote theirs. Every call zeroes the counters of the scratch it hands a launch (aklr/plugins/aiter/README.md: scratch
    is written before it is read); the partial planes before them each launch writes first."""
    pytest.importorskip("flydsl.expr")
    module = _module(builder_registry, path)
    empty = torch.empty
    # memory as an earlier tensor left it
    monkeypatch.setattr(torch, "empty", lambda *a, **k: empty(*a, **k).fill_(-1))
    ws = module._scratch(96, 64, "cpu")
    assert ws.shape == (96,) and (ws[:64] == -1).all() and (ws[64:] == 0).all()
    # every launch's scratch in run comes from _scratch, skw.py's zeroed whole
    assert _allocations(inspect.getsource(module.run)) == set()
    skw = sys.modules[module.__name__.rpartition(".")[0] + ".skw"]
    assert _allocations(inspect.getsource(skw.Launcher.__call__)) == {"torch.zeros"}


def _allocations(source):
    """The torch.empty and torch.zeros calls of a function's source."""
    tree = ast.parse(inspect.cleandoc("\n" + source) if source.startswith(" ") else source)
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return calls & {"torch.empty", "torch.zeros"}


def _workloads():
    return [(path, w["axes"]["m"]) for path in _schemas() for w in json.loads(path.read_text())["workloads"]]


@pytest.mark.gfx950
@pytest.mark.parametrize(
    "path, m", _workloads(), ids=lambda v: v.parent.name if hasattr(v, "parent") else str(v)
)
def test_the_k3_kernel_matches_the_reference_on_a_poisoned_allocator(builder_registry, path, m):
    """At each workload, the call's scratch taken from blocks an earlier tensor filled with ones in every bit: the
    out within the contract's bound of the reference, twice (eager, then the same call again)."""
    from tests.op_case import OpCase

    data = json.loads(path.read_text())
    contract = ContractRegistry.default().for_schema_json(data)
    case = OpCase(contract.point_from_json(data, {"m": m}))
    schema = contract.narrow(
        case.point, workloads=[contract.workload_for(case.point, {})], kernel=data["kernel"]
    )
    runnable = builder_registry.build(schema)
    args = case.inputs()
    call = dict(case.call({n: t.cuda() if torch.is_tensor(t) else t for n, t in args.items()}, "cuda"))
    call["out"] = torch.empty(
        contract.storage_shape_of("out", case.point), dtype=torch.bfloat16, device="cuda"
    )
    for _ in range(2):
        torch.empty(256 << 20, dtype=torch.uint8, device="cuda").fill_(255)  # freed, kept by the allocator
        call["out"].fill_(float("nan"))
        runnable(**call)
        torch.cuda.synchronize()
        assert case.compare({"out": call["out"]}, args) == {}
