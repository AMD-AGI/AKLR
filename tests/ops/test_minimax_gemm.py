# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical regressions for the embedded MiniMax hidden-state GEMM kernels."""

import ast
import functools
import json
import operator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry

_IMPLS = Path(__file__).parents[2] / "aklr/ops/gemm/impls/flydsl/gemm_n6144_k30720"


def _schema(first_m):
    matches = [
        path
        for path in _IMPLS.glob("*.json")
        if json.loads(path.read_text())["workloads"][0]["axes"]["m"] == first_m
    ]
    (path,) = matches
    return ContractRegistry.default().parse_schema(path.read_bytes(), str(path))


def _exec(nodes, scope):
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "kernel.py", "exec"),
        scope,
    )


def _numeric_scope():
    """CPU equivalents of the emitted buffer and vector operations, not of GEMM/MFMA.

    The integer ops also run the old FP24 pack/unpack bodies, so the regression
    produces their NaN instead of merely checking for a particular source format.
    """

    def integer_op(fn):
        def apply(a, b):
            value = fn(a.to(torch.int64) & 0xFFFFFFFF, b.to(torch.int64) & 0xFFFFFFFF)
            return SimpleNamespace(result=(value & 0xFFFFFFFF).to(torch.int32))

        return apply

    def bitcast(dtype, value):
        return value.view(dtype[1])

    def store(value, resource, offset, scalar_offset, **kwargs):
        data, size = resource
        start = offset + scalar_offset
        raw = value.view(torch.uint8)
        assert 0 <= start <= start + raw.numel() <= size <= data.numel()
        data[start : start + raw.numel()].copy_(raw)

    def load(dtype, resource, offset, scalar_offset, **kwargs):
        count, scalar_type = dtype
        data, size = resource
        start = offset + scalar_offset
        end = start + count * torch.empty((), dtype=scalar_type).element_size()
        assert 0 <= start <= end <= size <= data.numel()
        return data[start:end].view(scalar_type)

    return {
        "T": SimpleNamespace(
            i32=torch.int32, f32=torch.float32, bf16=torch.bfloat16, vec=lambda n, t: (n, t)
        ),
        "vector": SimpleNamespace(bitcast=bitcast, shuffle=lambda a, b, indices: torch.cat((a, b))[indices]),
        "arith": SimpleNamespace(unwrap=lambda x: x, addf=torch.add, truncf=lambda t, x: x.to(t[1])),
        "std_arith": SimpleNamespace(
            AddIOp=integer_op(operator.add),
            AndIOp=integer_op(operator.and_),
            OrIOp=integer_op(operator.or_),
            ShRUIOp=integer_op(operator.rshift),
            ShLIOp=integer_op(operator.lshift),
        ),
        "_vconst": lambda values: torch.tensor(values, dtype=torch.int64).to(torch.int32),
        "_rsrc": lambda data, size: (data.view(torch.uint8).flatten(), size),
        "rocdl": SimpleNamespace(raw_ptr_buffer_store=store, raw_ptr_buffer_load=load),
        "ir": SimpleNamespace(IntegerAttr=SimpleNamespace(get=lambda *args: None)),
        "fx": SimpleNamespace(Int32=int, block_idx=SimpleNamespace(x=0), thread_idx=SimpleNamespace(x=0)),
        "range_constexpr": range,
    }


def test_finite_split_partials_survive_storage_and_cancellation():
    """Execute the actual producer epilogue and fold on finite +/- FLT_MAX.

    Column 0 cancels to zero; column 1 retains one FP32 ulp at FLT_MAX. Full FP32
    storage preserves both. FP24 overflow yields NaN; saturating loses column 1.
    """
    tree = ast.parse(_schema(268).kernel.sources[0].content)
    scope = _numeric_scope()
    _exec(
        [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and all(isinstance(target, ast.Name) for target in node.targets)
                and isinstance(node.value, ast.Constant)
            )
            or (isinstance(node, ast.FunctionDef) and node.name in {"_pack24", "_unpack24"})
        ],
        scope,
    )
    producer = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.endswith("_splitk")
    )
    first_store = next(
        i
        for i, node in enumerate(producer.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "wr" for t in node.targets)
    )
    # One lane's four columns suffice to exercise packing and byte offsets. The
    # full m=268, n=6144 kernel is tested below on gfx950.
    scope.update(
        W=torch.empty(8 * 4 * 4, dtype=torch.uint8),
        SPLIT=8,
        m=1,
        n=4,
        block_m_idx=0,
        BLOCK_M=16,
        warp_m_idx=0,
        block_n_idx=0,
        BLOCK_N=16,
        warp_n_idx=0,
        w_tid=0,
        WMMA_N=16,
        WMMA_M=16,
        WMMA_C_FRAG_VALUES=4,
        WARP_M_STEPS=1,
        WARP_N_STEPS=1,
    )
    maximum = torch.tensor(torch.finfo(torch.float32).max)
    previous = torch.nextafter(maximum, torch.tensor(0.0))
    partials = torch.zeros((8, 4), dtype=torch.float32)
    partials[0] = maximum
    partials[1] = torch.stack((-maximum, -previous, -maximum, -previous))
    for q in range(8):
        scope.update(split_idx=q, c_frags=[partials[q]])
        _exec(producer.body[first_store:], scope)

    fold = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "gemm_n6144_k30720_fold"
    )
    scope.update(PLANE=4, FOLD_VEC=4, FOLD_THREADS=256, C=torch.empty(4, dtype=torch.bfloat16))
    _exec(fold.body, scope)
    expected = torch.tensor([0.0, 2.0**104, 0.0, 2.0**104], dtype=torch.bfloat16)
    torch.testing.assert_close(scope["C"], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("first_m", "torch_rows"),
    [(1, {209, 210, 212, 216, 224}), (268, {2856, 3672, 12896, 19992, 19996, 32768})],
)
def test_documented_torch_workloads_use_the_fallback(first_m, torch_rows):
    """Check the entry point's backend choice over every declared workload."""
    schema = _schema(first_m)
    tree = ast.parse(schema.kernel.sources[0].content)
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.For))
        or isinstance(node, ast.FunctionDef)
        and node.name in {"_launcher", "run"}
    ]
    events = []
    scope = {
        "functools": functools,
        "torch": SimpleNamespace(
            matmul=lambda *args, **kwargs: events.append("torch"),
            cuda=SimpleNamespace(current_stream=lambda device: None),
        ),
        "fx": SimpleNamespace(Stream=lambda stream: stream),
    }
    _exec(body, scope)
    launch = lambda *args, **kwargs: events.append("flydsl")
    # Keep the real table and entry point; stub compilation and launching only.
    scope["_build_tiled"] = lambda *args: launch
    scope["_build_fold"] = lambda *args: launch
    scope["splitk_launcher"] = lambda *args: launch
    if first_m == 268:
        scope["torch"].empty = lambda *args, **kwargs: None
        scope["torch"].float32 = torch.float32
    observed = set()
    for workload in schema.workloads:
        m = workload.axes["m"]
        events.clear()
        scope["run"](
            a=SimpleNamespace(shape=(m, 30720), device="cpu"),
            b=SimpleNamespace(t=lambda: None),
            trans_b=True,
            bias=None,
            out_dtype=torch.bfloat16,
            out=None,
        )
        if events == ["torch"]:
            observed.add(m)
        else:
            assert events and set(events) == {"flydsl"}
    assert observed == torch_rows


@pytest.mark.gfx950
def test_gemm_finite_extreme_split_cancellation_on_gpu(builder_registry):
    """Run the actual m=268 kernel with finite BF16 inputs generating +/- FLT_MAX."""
    schema = _schema(268)
    runnable = builder_registry.build(schema)
    m, n, k, span = 268, 6144, 30720, 3840
    a = torch.ones((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.zeros((n, k), dtype=torch.bfloat16, device="cuda")
    powers = torch.tensor([2.0**i for i in range(104, 128)], dtype=torch.bfloat16, device="cuda")
    b[:, :24] = powers
    b[:, span : span + 24] = -powers
    b[1::2, span] = 0
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    runnable(a=a, b=b, trans_b=True, bias=None, out_dtype=None, out=out)
    expected = torch.zeros_like(out)
    expected[:, 1::2] = 2.0**104
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
