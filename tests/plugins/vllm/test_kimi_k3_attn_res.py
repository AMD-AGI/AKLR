# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's attention residual on vLLM to AKLR's kimi_k3_attn_res and kimi_k3_attn_res_fp8: what the route takes
and how it hands it over (CPU), the baseline's copy of vLLM's Triton kernel against vLLM's own (GPU), and the committed
kernels: the scratch they hand their launches (CPU) and their softmax and mix at the edges of float32 (gfx950)."""

import ast
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.models.kimi_k3.attn_res import api
from aklr.plugins.vllm.models.kimi_k3.attn_res.contract import KIMI_K3_ATTN_RES
from tests.op_case import OpCase

TOKENS, HIDDEN, ROWS = 4, 256, 4


@pytest.fixture
def op(monkeypatch):
    calls = []

    def fake(name):
        def call(*args, **kwargs):
            calls.append((args, kwargs))
            return name

        return call

    monkeypatch.setattr(api, "kimi_k3_attn_res", fake("aklr"))
    monkeypatch.setattr(api, "kimi_k3_attn_res_fp8", fake("aklr_fp8"))
    return calls


def operands():
    bf16 = dict(dtype=torch.bfloat16)
    return dict(
        prefix=torch.zeros(TOKENS, HIDDEN, **bf16),
        delta=torch.zeros(TOKENS, HIDDEN, **bf16),
        blocks=torch.zeros(TOKENS, ROWS, HIDDEN, **bf16),
        norm_weight=torch.ones(HIDDEN, **bf16),
        qk_weight=torch.ones(HIDDEN, **bf16),
        output_norm_weight=torch.ones(HIDDEN, **bf16),
    )


@pytest.mark.parametrize(
    ("changed", "write_idx"),
    [({"quant_dtype": torch.float8_e4m3fnuz}, 2), ({}, 1)],
    ids=["fnuz_fp8_output", "write_before_the_last_row"],
)
def test_what_no_contract_states_goes_back_to_vllm(op, changed, write_idx):
    x = operands()
    result = api._attn_res(
        x["prefix"], x["delta"], x["blocks"], x["norm_weight"], x["qk_weight"], x["output_norm_weight"], 2,
        write_idx, 1e-6, 1e-5, **changed,
    )  # fmt: skip
    assert result is FallbackReason.OUT_OF_CONTRACT and op == []


@pytest.mark.parametrize(
    ("quant_dtype", "to"), [(None, "aklr"), (torch.float8_e4m3fn, "aklr_fp8")], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("write_idx", [-1, 2], ids=["no_write", "write_row_n"])
@pytest.mark.parametrize("with_optionals", [True, False], ids=["addend_and_norm", "neither"])
def test_a_supported_call_hands_every_operand_over(op, write_idx, with_optionals, quant_dtype, to):
    x = operands()
    delta = x["delta"] if with_optionals else None
    norm = x["output_norm_weight"] if with_optionals else None
    result = api._attn_res(
        x["prefix"], delta, x["blocks"], x["norm_weight"], x["qk_weight"], norm, 2, write_idx, 1e-6, 1e-5,
        quant_dtype=quant_dtype,
    )  # fmt: skip
    assert result == to
    ((args, kwargs),) = op
    assert args[0] is x["prefix"] and args[1] is x["blocks"]
    assert args[2] is x["norm_weight"] and args[3] is x["qk_weight"]
    assert kwargs == dict(
        n_valid_rows=2,
        write_bank=write_idx >= 0,
        addend=delta,
        norm_weight=norm,
        score_eps=1e-6,
        norm_eps=1e-5,
    )


@pytest.mark.gfx950
@pytest.mark.parametrize("n_tokens", [1, 255, 256, 300])
@pytest.mark.parametrize("n_rows", [0, 1, 3])
@pytest.mark.parametrize("variant", ["addend_write_norm", "plain"])
def test_the_baselines_kernel_is_vllms(n_tokens, n_rows, variant):
    """The copy of vLLM's Triton kernel the baseline runs writes vLLM's bits: output, updated prefix and bank."""
    linear = pytest.importorskip("vllm.models.kimi_k3.amd.linear")
    from aklr.plugins.vllm.models.kimi_k3.attn_res._triton import attn_res

    torch.manual_seed(0)
    full = variant == "addend_write_norm"

    def make():
        torch.manual_seed(1)
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        return [
            torch.randn(n_tokens, 7168, **bf16),
            torch.randn(n_tokens, 7168, **bf16) if full else None,
            torch.randn(n_tokens, 8, 7168, **bf16),
            torch.randn(7168, **bf16),
            torch.randn(7168, **bf16),
            torch.randn(7168, **bf16) if full else None,
        ]

    write = n_rows if full else -1
    ours, theirs = make(), make()
    out = attn_res(*ours, n_rows, write, 1e-6, 1e-5)
    expected = linear.attn_res(*theirs, n_rows, write, 1e-6, 1e-5)
    assert torch.equal(out, expected)
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[2], theirs[2])


# ---------------------------------------------------------------- the committed kernels


def _kernel_schemas() -> list[Path]:
    """Every committed kimi_k3_attn_res and kimi_k3_attn_res_fp8 schema holding a kernel."""
    paths = committed_schema_paths(op_dir(KIMI_K3_ATTN_RES.op_package))
    return [p for p in paths if json.loads(p.read_text())["kernel"]["sources"]]


def _schema_id(path: Path) -> str:
    """``fp8_rows1/<uuid>`` for ``impls/flydsl/kimi_k3_attn_res_fp8_rows1_hidden_size7168_n_bank_rows8/<uuid>.json``."""
    name = path.parent.name.removeprefix("kimi_k3_attn_res").removesuffix("_hidden_size7168_n_bank_rows8")
    return f"{name.lstrip('_') or 'rows8'}/{path.stem}"


# the dispatch packet's id and queue: what a kernel would tag its launch with
_LAUNCH_IDS = frozenset({"llvm.amdgcn.dispatch.id", "llvm.amdgcn.queue.ptr"})


@pytest.mark.parametrize("path", _kernel_schemas(), ids=_schema_id)
def test_a_committed_kernel_tells_its_writes_from_leftover_memory_without_a_launch_tag(path):
    """The workgroups coordinate through scratch the call allocates: the counter a walk claims tokens from, the tags
    under which column shares publish their partials. A tag taken from the launch (its dispatch id and queue) and
    compared with the scratch as allocated can match words an earlier tensor left there, and then rows are skipped or
    partials read unwritten; the call zeroes those words instead (aklr/ops/README.md: scratch is written before it is
    read), so no kernel reads its launch's ids."""
    for source in json.loads(path.read_text())["kernel"]["sources"]:
        strings = {n.value for n in ast.walk(ast.parse(source["content"])) if isinstance(n, ast.Constant)}
        assert not strings & _LAUNCH_IDS, source["path"]


def _representative_points() -> list[tuple[Path, int]]:
    """(schema, n_tokens): for each kernel, contract and norm (the softmax and the mix are one code over the other
    attrs), the first schema with bank rows at its smallest workload, and at its smallest past 512 tokens (two
    workgroups a CU on MI355X), past which the builds walk the tokens, claim them or split them into column shares."""
    seen, points = set(), []
    for path in _kernel_schemas():
        data = json.loads(path.read_text())
        key = (json.dumps(data["kernel"]["sources"]), data["contract_id"], data["attrs"]["has_norm"])
        if data["attrs"]["n_valid_rows"] == 0 or key in seen:
            continue
        seen.add(key)
        counts = sorted(w["axes"]["n_tokens"] for w in data["workloads"])
        points += [(path, n) for n in sorted({counts[0], *[c for c in counts if c > 512][:1]})]
    return points


_POINTS = pytest.mark.parametrize(
    ("path", "n_tokens"),
    _representative_points(),
    ids=lambda v: _schema_id(v) if isinstance(v, Path) else str(v),
)


def _fill_rows(args: dict, value: float) -> None:
    """Every bank row and p ``value`` (the prefix, an addend of 0); a norm weight of 1."""
    args["prefix"].fill_(value)
    args["bank"].fill_(value)
    if args.get("addend") is not None:
        args["addend"].zero_()
    if args.get("norm_weight") is not None:
        args["norm_weight"].fill_(1.0)


def _run_committed(builder_registry, path: Path, n_tokens: int, fill: Callable[[dict, bool], None]) -> None:
    """The kernel committed at ``path``, at ``n_tokens`` on the inputs ``fill`` sets (told whether a norm weight is
    passed): its outputs and mutable inputs are finite and pass the contract's compare against the reference."""
    data = json.loads(path.read_text())
    contract = ContractRegistry.default().for_schema_json(data)
    case = OpCase(contract.point_from_json(data, {"n_tokens": n_tokens}))
    args = case.inputs()
    fill(args, case.point.attrs["has_norm"])
    schema = contract.narrow(
        case.point, workloads=[contract.workload_for(case.point, {})], kernel=data["kernel"]
    )
    runnable = builder_registry.build(schema)
    call = dict(case.call(args, "cuda"))
    for name in contract.outputs:
        shape = contract.dtype_storage_shape_of(name, case.point)
        call[name] = torch.empty(shape, dtype=getattr(torch, case.point.dtypes[name]), device="cuda")
    runnable(**call)
    torch.cuda.synchronize()
    outputs = {n: call[n] for n in contract.mutable_tensors}
    for name, tensor in outputs.items():
        assert tensor.float().isfinite().all(), name
    assert case.compare(outputs, args) == {}


@pytest.mark.gfx950
@_POINTS
def test_a_committed_kernel_masks_the_lanes_past_the_rows_with_minus_inf(builder_registry, path, n_tokens):
    """Rows of 1 and a score weight of -1.125 * 2**115 (the norm's) times 1 (the projection's): every row scores
    7168 * -1.125 * 2**115 * rsqrt(1 + eps), about -3.35e38. A lane past the rows scored with any finite value above
    that, such as -3e38, would win the softmax's max and leave every row a weight of 0, and the out NaN."""

    def fill(args, has_norm):
        _fill_rows(args, 1.0)
        args["score_norm_weight"].fill_(-1.125 * 2.0**115)
        args["score_proj_weight"].fill_(1.0)

    _run_committed(builder_registry, path, n_tokens, fill)


@pytest.mark.gfx950
@_POINTS
def test_a_committed_kernel_mixes_rows_past_float32s_squares(builder_registry, path, n_tokens):
    """Equal rows and score weights of 0: the softmax is uniform and the mix the rows. With a norm weight, rows of
    2**57: their mean square 2**114 is a float32, Z**2 times it is not (Z, the sum of the unnormalized weights, is
    the row count, 2 to 9). Without one, rows of 2**127: Z times them is not a float32 either. The weights are
    normalized before the mix."""

    def fill(args, has_norm):
        _fill_rows(args, 2.0**57 if has_norm else 2.0**127)
        args["score_norm_weight"].zero_()
        args["score_proj_weight"].fill_(1.0)

    _run_committed(builder_registry, path, n_tokens, fill)


def _split_points() -> list[tuple[Path, int]]:
    """(schema, n_tokens) of every committed workload past 512 tokens of a kernel with a split build (its column
    shares exchange partials through scratch): the tokens past 512 are the split's."""
    points = []
    for path in _kernel_schemas():
        data = json.loads(path.read_text())
        if any("def _make_split" in s["content"] for s in data["kernel"]["sources"]):
            points += [
                (path, w["axes"]["n_tokens"]) for w in data["workloads"] if 512 < w["axes"]["n_tokens"] <= 560
            ]
    return points


@pytest.mark.gfx950
@pytest.mark.parametrize(
    ("path", "n_tokens"), _split_points(), ids=lambda v: _schema_id(v) if isinstance(v, Path) else str(v)
)
def test_a_split_build_normalizes_a_mix_whose_rows_cancel(builder_registry, path, n_tokens):
    """Bank rows 0 and 1 are x and -x with |x| near 1024, the other rows and p near 1, score weights 0 (every row
    weighted alike): the mix is the small rows' and its sum of squares a 2**-20 part of the rows'. A sum of squares
    taken from the rows' Gram partials (x.x + (-x).(-x) + 2 x.(-x)) cancels to float32's rounding of the large terms,
    13% off at 1024 and negative, the out NaN, at 4096; the mix's own sum of squares is exact to float32."""

    def fill(args, has_norm):
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(args["prefix"].shape, generator=gen) * 1024
        args["bank"][:, 0] = x.to(torch.bfloat16)
        args["bank"][:, 1] = (-x).to(torch.bfloat16)
        args["bank"][:, 2:].copy_(torch.randn(args["bank"][:, 2:].shape, generator=gen))
        args["prefix"].copy_(torch.randn(args["prefix"].shape, generator=gen))
        if args.get("addend") is not None:
            args["addend"].copy_(torch.randn(args["addend"].shape, generator=gen))
        args["score_norm_weight"].zero_()
        args["score_proj_weight"].zero_()
        if args.get("norm_weight") is not None:
            args["norm_weight"].fill_(1.0)

    _run_committed(builder_registry, path, n_tokens, fill)
