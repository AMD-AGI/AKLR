# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The committed Kimi-K3 KDA kernels of sglang's Triton backend (kimi_k3_sgl_kda_prefill and _prefill_track) on calls
validation does not draw (gfx950): each case is a committed workload drawn as validation draws it,
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


def _case(path: Path, axes: dict, n_slots: int | None = None) -> tuple[OpCase, dict]:
    """The case of the committed schema at ``path`` at its workload ``axes``, its pool of ``n_slots`` (an any_size
    axis: any pool serves) where given, and the schema's kernel."""
    data = json.loads(path.read_text())
    assert axes in [w["axes"] for w in data["workloads"]], axes
    contract = ContractRegistry.default().for_schema_json(data)
    if n_slots is not None:
        axes = {**axes, "n_slots": n_slots}
    return OpCase(contract.point_from_json(data, axes)), data["kernel"]


def _kernel_and_reference(builder_registry, case: OpCase, kernel: dict, args: dict) -> tuple[dict, dict]:
    """The kernel's call on ``args`` (row-major, on the CPU): its outputs and mutable inputs as it leaves them, all
    finite, and the reference's for ``args``."""
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
    expected = contract.reference.reference(**case.contract_workload.reference_call(on_gpu, "cuda"))
    return outputs, expected


def _run(builder_registry, case: OpCase, kernel: dict, args: dict) -> dict:
    """The kernel's outputs and mutable inputs on ``args``, which are finite and pass the contract's compare against
    the reference of ``args``."""
    outputs, expected = _kernel_and_reference(builder_registry, case, kernel, args)
    assert case.contract.reference.check(outputs, expected, case.contract_workload.schema) == {}
    return outputs


def _bound_multiples(case: OpCase, outputs: dict, expected: dict) -> dict[str, float]:
    """Each bounded tensor's largest error against the reference in multiples of its bound, atol + rtol |y| (past 1,
    the contract's compare fails)."""
    multiples = {}
    for name, spec in case.contract_workload.schema.tensors().items():
        if not spec.error_bounds:
            continue
        bound = spec.error_bounds[0][1]
        y = expected[name].double()
        error = (outputs[name].double().reshape(y.shape) - y).abs()
        limit = bound.absolute_tolerance + bound.relative_tolerance * y.abs()
        multiples[name] = float((error / limit).max())
    return multiples


def _untouched(args: dict, outputs: dict, slots: list[int]) -> None:
    """The pool's slots no live row names hold their bits."""
    unnamed = sorted(set(range(args["state_cache"].shape[0])) - {s for s in slots if s >= 0})
    assert torch.equal(outputs["state_cache"][unnamed].cpu(), args["state_cache"][unnamed])


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


def _smallest_prefill_points() -> list[tuple[Path, dict]]:
    """Each committed prefill and snapshot schema at its first workload (one sequence, its fewest tokens)."""
    return [
        (path, json.loads(path.read_text())["workloads"][0]["axes"])
        for path in _schemas("kimi_k3_sgl_kda_prefill")
    ]


_ALL_PREFILL_POINTS = pytest.mark.parametrize(
    ("path", "axes"),
    _smallest_prefill_points() + _prefill_points(),
    ids=lambda v: _schema_id(v) if isinstance(v, Path) else f"{v['n_tokens']}x{v['n_seqs']}",
)
# the value channels whose v is 2**18: whole 16-channel slabs (a walk wave's columns), or every other channel, each
# wave's large columns beside small ones
_LARGE_CHANNELS = {"slabs_0_3": slice(0, 64), "every_other": slice(0, 128, 2)}


@pytest.mark.gfx950
@_ALL_PREFILL_POINTS
@pytest.mark.parametrize("channels", list(_LARGE_CHANNELS))
def test_a_committed_prefill_kernel_keeps_a_delta_past_fp16s_range(builder_registry, path, axes, channels):
    """q = k = e0 at every token, raw beta 0 (beta 1/2), zero states, and v of 2**18 at each sequence's first token on
    some value channels: the first delta, beta v = 2**17, and the next ones, beta (v - S k), pass fp16's largest,
    65504, while the kernel's u = beta (v - Kd h) goes through fp16, which a wave holding such a u takes scaled, each
    of its columns by its own power of two. The other channels keep their drawn v: in whole slabs of their own, or
    every other channel, beside the large ones in each wave.

    Every output, state and snapshot stays finite. Past that the contract's bounds are not what these inputs can be
    held to: the later outputs on the large channels are differences of 2**17-sized terms, which bf16 operands do not
    hold to atol 2e-2 (on MI355X sglang's chunk_kda is 5 to 20000 bounds from the float32 reference here, the kernel
    2 to 14000). So each tensor is within its bound or no further from the reference than sglang's own chunk_kda."""
    pytest.importorskip("sglang")
    case, kernel = _case(path, axes)
    args = case.inputs()
    q, k, v = args["qkv"].unflatten(1, (3, case.point.axes["n_heads"])).unbind(1)
    for x in (q, k):
        x.zero_()
        x[..., 0] = 1
    v[args["cu_seqlens"][:-1].long(), :, _LARGE_CHANNELS[channels]] = 2.0**18
    args["raw_beta"].zero_()
    args["state_cache"].zero_()
    outputs, expected = _kernel_and_reference(builder_registry, case, kernel, args)
    contract = case.contract
    call = dict(case.call(args, "cuda"))
    sglangs = dict(contract.baseline.baseline(**call)) | {n: call[n] for n in contract.mutable_inputs}
    ours, theirs = _bound_multiples(case, outputs, expected), _bound_multiples(case, sglangs, expected)
    assert {n: x for n, x in ours.items() if x > max(1.0, theirs[n])} == {}, (ours, theirs)


# a slot of 12 heads' 128 x 128 float32 states is 786432 bytes: slot 5462 starts past 2**32 bytes
_FAR_SLOT = 5462


@pytest.mark.gfx950
@_ALL_PREFILL_POINTS
def test_a_committed_prefill_kernel_reaches_slots_past_4_gib(builder_registry, path, axes):
    """A pool of 5470 slots (4.3 GB), the sequences' slots past 2**32 bytes into it (the last one, and slot 5462)
    beside slot 0 and a -1: the states are read from and written back to those slots and no other slot changes, as
    a 32-bit offset would not."""
    case, kernel = _case(path, axes, n_slots=_FAR_SLOT + 8)
    args = case.inputs()
    slots = [_FAR_SLOT + 7, 0, _FAR_SLOT, -1][: axes["n_seqs"]]
    args["state_indices"] = torch.tensor(slots, dtype=torch.int32)
    outputs = _run(builder_registry, case, kernel, args)
    _untouched(args, outputs, slots)
