# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The committed GLM-5.2 attention-DP8 (64 heads a rank) sparse MLA kernels on top-k rows the bench does not
draw (gfx950): slots in any order and none at all, which the contract allows; the kernels' outputs pass the
contract's compare against the reference."""

import json
from pathlib import Path

import pytest
import torch

from aklr.core.contract.spec import PointSpec
from tests.op_case import OpCase

_DSA = Path(__file__).resolve().parents[3] / "aklr/plugins/sglang/layers/dsa_backend/impls/triton"
_DECODE = "dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048"
_PREFILL = "dsa_sparse_mla_prefill_n_heads64_kv_dim576_topk2048"
_SLOTS = 4096


def _committed_kernel(directory: str, n_tokens: int | None = None) -> dict:
    """The kernel of the file in ``directory`` serving ``n_tokens``, or of its one file."""
    files = [json.loads(path.read_text()) for path in sorted((_DSA / directory).glob("*.json"))]
    if n_tokens is not None:
        files = [f for f in files if any(w["axes"]["n_tokens"] == n_tokens for w in f["workloads"])]
    (kernel,) = (f["kernel"] for f in files)
    return kernel


def _case(contract_id: str, n_tokens: int) -> OpCase:
    return OpCase(
        PointSpec(
            contract_id=contract_id,
            axes={"n_tokens": n_tokens, "n_heads": 64, "kv_dim": 576, "topk": 2048, "n_slots": _SLOTS},
            dtypes={
                "q": "float8_e4m3fn",
                "kv_cache": "float8_e4m3fn",
                "topk_indices": "int32",
                "output": "bfloat16",
            },
            attrs={},
            layout_variant_name="row_major",
        )
    )


def _scattered_rows(n_tokens: int) -> torch.Tensor:
    """Distinct slots in a random order; row 0 names none, row 1 one, later rows end in a run of -1."""
    generator = torch.Generator().manual_seed(n_tokens)
    rows = torch.stack([torch.randperm(_SLOTS, generator=generator)[:2048] for _ in range(n_tokens)])
    rows[0] = -1
    if n_tokens > 1:
        rows[1, 1:] = -1
    for row in range(2, n_tokens):
        rows[row, 2048 - 97 * row % 2048 :] = -1
    return rows.to(torch.int32)


def _run(builder_registry, case: OpCase, kernel: dict) -> None:
    """The kernel on ``case``'s inputs, top-k rows from ``_scattered_rows``, passes ``case``'s compare."""
    args = case.inputs()
    args["topk_indices"] = _scattered_rows(case.point.axes["n_tokens"])
    schema = case.contract.narrow(
        case.point, workloads=[case.contract.workload_for(case.point, {})], kernel=kernel
    )
    runnable = builder_registry.build(schema)
    call = dict(case.call(args, "cuda"))
    shape = case.contract.dtype_storage_shape_of("output", case.point)
    call["output"] = torch.full(shape, float("nan"), dtype=torch.bfloat16, device="cuda")
    runnable(**call)
    torch.cuda.synchronize()
    output = call["output"]
    assert output.float().isfinite().all()
    assert (output[0] == 0).all()
    failed = {o: r.detail for o, r in case.compare({"output": output}, args).items() if not r.is_passed}
    assert failed == {}


@pytest.mark.gfx950
@pytest.mark.parametrize("n_tokens", [1, 2, 4, 8])
def test_dp8_sparse_mla_decode_attends_scattered_and_empty_rows(builder_registry, n_tokens):
    _run(builder_registry, _case("dsa_sparse_mla_decode@1.0", n_tokens), _committed_kernel(_DECODE, n_tokens))


@pytest.mark.gfx950
def test_dp8_sparse_mla_prefill_attends_scattered_and_empty_rows(builder_registry):
    """64 tokens: the wide path the 1024 and 2048 token Points take, on rows no new sequence has."""
    _run(builder_registry, _case("dsa_sparse_mla_prefill@1.0", 64), _committed_kernel(_PREFILL))
