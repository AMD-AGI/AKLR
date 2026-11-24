# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The committed Kimi-K3 KDA kernels of sglang's Triton backend (kimi_k3_sgl_kda_prefill, _prefill_track and
_decode) on calls validation does not draw (gfx950): each case is a committed workload drawn as validation draws it,
with the slots or values that matter set by hand, and the kernel's outputs and written inputs pass the contract's
compare against the reference, computed on the GPU."""

import json
from pathlib import Path

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from tests.op_case import OpCase

_IMPLS = Path(__file__).resolve().parents[3] / "aklr/plugins/sglang/layers/kda_triton/impls/flydsl"


def _schemas(prefix: str) -> list[Path]:
    """The committed schemas, kernels and all, of the scenario names starting ``prefix`` (``kimi_k3_sgl_kda_prefill``:
    the snapshot variant's too)."""
    paths = _IMPLS.glob(f"{prefix}*/*.json")
    return sorted(p for p in paths if json.loads(p.read_text())["kernel"]["sources"])


def _schema_id(path: Path) -> str:
    return f"{path.parent.name.removeprefix('kimi_k3_sgl_kda_').split('_strided_beta')[0]}/{path.stem}"


def _case(path: Path, axes: dict) -> tuple[OpCase, dict]:
    """The case of the committed schema at ``path`` at its workload ``axes``, and the schema's kernel."""
    data = json.loads(path.read_text())
    assert axes in [w["axes"] for w in data["workloads"]], axes
    contract = ContractRegistry.default().for_schema_json(data)
    return OpCase(contract.point_from_json(data, axes)), data["kernel"]


def _run(builder_registry, case: OpCase, kernel: dict, args: dict) -> dict:
    """The kernel's call on ``args`` (row-major, on the CPU): its outputs and mutable inputs as it leaves them, which
    are finite and pass the contract's compare against the reference of ``args``."""
    contract = case.contract
    schema = contract.narrow(case.point, workloads=[contract.workload_for(case.point, {})], kernel=kernel)
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
    on_gpu = {n: v.cuda() if torch.is_tensor(v) else v for n, v in args.items()}
    expected = contract.reference.reference(**case.request.reference_call(on_gpu, "cuda"))
    assert contract.reference.check(outputs, expected, case.request.schema) == {}
    return outputs


def _untouched(args: dict, outputs: dict, slots: list[int]) -> None:
    """The pool's slots no live row names hold their bits."""
    unnamed = sorted(set(range(args["state_cache"].shape[0])) - {s for s in slots if s >= 0})
    assert torch.equal(outputs["state_cache"][unnamed].cpu(), args["state_cache"][unnamed])


def _padded_slots(n: int, n_slots: int) -> list[int]:
    """Slot 0 first, -1 on every third row, distinct slots from 1 up elsewhere."""
    others = iter(torch.randperm(n_slots - 1, generator=torch.Generator().manual_seed(n))[:n].add(1).tolist())
    return [0 if i == 0 else -1 if i % 3 == 1 else next(others) for i in range(n)]


@pytest.mark.gfx950
@pytest.mark.parametrize("path", _schemas("kimi_k3_sgl_kda_decode"), ids=_schema_id)
@pytest.mark.parametrize("n_tokens", [8, 48, 128], ids=["narrow", "mid", "wide"])
def test_a_committed_decode_kernel_steps_padded_rows_from_a_zero_state(builder_registry, path, n_tokens):
    """Slot 0 stepped beside rows of -1 (sglang's padded graph rows, several a batch): a -1 row gives the out of a
    zero state and writes no slot, slot 0 takes its token's step, the slots no row names keep their bits. 8, 48 and
    128 tokens take the kernel's three band builds."""
    case, kernel = _case(path, {"n_tokens": n_tokens, "n_slots": n_tokens + 1})
    args = case.inputs()
    slots = _padded_slots(n_tokens, n_tokens + 1)
    args["state_indices"] = torch.tensor(slots, dtype=torch.int32)
    outputs = _run(builder_registry, case, kernel, args)
    _untouched(args, outputs, slots)


def _prefill_points() -> list[tuple[Path, dict]]:
    """Each committed prefill and snapshot schema at its workload of the most sequences."""
    points = []
    for path in _schemas("kimi_k3_sgl_kda_prefill"):
        workloads = [w["axes"] for w in json.loads(path.read_text())["workloads"]]
        points.append((path, max(workloads, key=lambda axes: (axes["n_seqs"], axes["n_tokens"]))))
    return points


_PREFILL_POINTS = pytest.mark.parametrize(
    ("path", "axes"),
    _prefill_points(),
    ids=lambda v: _schema_id(v) if isinstance(v, Path) else str(v["n_tokens"]),
)


@pytest.mark.gfx950
@_PREFILL_POINTS
def test_a_committed_prefill_kernel_runs_padded_sequences_from_a_zero_state(builder_registry, path, axes):
    """Slot 0 beside two sequences of slot -1 (sglang's padding): a -1 sequence runs from a zero state and writes no
    slot, slot 0 takes its sequence's last state, the slots no sequence names keep their bits; with a snapshot, every
    sequence's, the padded ones' too, is written where asked."""
    case, kernel = _case(path, axes)
    args = case.inputs()
    slots = [0, -1, 5, -1][: axes["n_seqs"]]
    args["state_indices"] = torch.tensor(slots, dtype=torch.int32)
    if "track_chunk_idx" in args:
        args["track_chunk_idx"] = torch.tensor([1, 3, -1, 0][: axes["n_seqs"]], dtype=torch.int32)
    outputs = _run(builder_registry, case, kernel, args)
    _untouched(args, outputs, slots)
