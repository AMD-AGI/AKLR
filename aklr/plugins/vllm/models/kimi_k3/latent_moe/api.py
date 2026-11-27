# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The latent-MoE tail's api and its vLLM route.

The op's kernel waits inside itself on every other rank of the TP group, so a rank that serves a call while another
declines it leaves the first spinning until its wait gives up. Every decision here therefore reads only what is the
same on every rank: the step's rows, vLLM's config, whether a graph is being captured, and verdicts the ranks agree
on over the TP group (the dispatcher's, the peer memory's setup, the first-call check against vLLM's tail). What the
dispatcher reads that a rank has alone, its operands' addresses, is made alike first: an operand off the
dispatcher's alignment goes over as an aligned copy, and the shared partial, which the op writes in place, as a copy
where it shares bytes with another operand.

The kernel keeps its group's state in its own sources (their ``runtime.py``): the route sets it up through the
module of the kernel the dispatcher serves the call with, before that kernel's first call, and watches its FAILED
word after every step (``failstop``). A rank whose in-kernel wait gives up (after ``runtime.WAIT_SECONDS``, 10 s)
fails the whole group, and the step's check stops vLLM before that step's output leaves it; a TP rank that reaches
the layer more than 10 s after the others (a long host stall) is such a failure. ``AKLR_DISABLE=
kimi_k3_latent_moe_tail@1.0`` hands every call to vLLM's tail."""

from __future__ import annotations

import functools
import importlib
import json
import logging

import torch
import torch.distributed as dist

from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import _original, vllm_route
from aklr.plugins.vllm.models.kimi_k3.latent_moe import failstop
from aklr.plugins.vllm.models.kimi_k3.latent_moe.contract import KIMI_K3_LATENT_MOE_TAIL

__all__ = ["kimi_k3_latent_moe_tail"]

_LOGGER = logging.getLogger(__name__)
# the first eager call of each row bucket is checked against vLLM's tail (``_checked_call``)
_CHECK = True
_SEEN: set[str] = set()
# the group state by (device, the serving kernel's package), None where the ranks agreed it cannot be set up
_GROUPS: dict[tuple[torch.device, str], object | None] = {}
# the first-call check's agreed verdict by (device, row bucket)
_CHECKED: dict[tuple[torch.device, int], bool] = {}
# the dispatcher's agreed verdict by (device, rows, eps, capturing): (None, the serving kernel's package) where it
# serves the call, else (the reason, None)
_VERDICTS: dict[tuple, tuple[FallbackReason | None, str | None]] = {}
# the dispatcher declines an operand whose address is not a multiple of this (its fast key's MIN_PTR_ALIGN_BYTES)
_ALIGN = 16


@api(KIMI_K3_LATENT_MOE_TAIL)
def kimi_k3_latent_moe_tail(
    routed,
    shared,
    up_weight,
    norm_weight,
    tp_size=8,
    eps=1e-5,
    out_dtype=None,
    out=None,
): ...


def _once(reason: str) -> None:
    """Log each distinct reason the route hands a call back to vLLM, once per process."""
    if reason not in _SEEN:
        _SEEN.add(reason)
        _LOGGER.warning("aklr latent-MoE tail route falls back: %s", reason)


@functools.cache
def _committed_rows() -> int:
    """The most rows any committed schema of the op serves: a call of more goes back to vLLM before the dispatcher's
    verdict, which every rank agrees on with a collective at each new row count (prefill's would be many)."""
    paths = committed_schema_paths(op_dir(KIMI_K3_LATENT_MOE_TAIL.op_package))
    return max(
        (w["axes"]["n_tokens"] for path in paths for w in json.loads(path.read_text())["workloads"]),
        default=0,
    )


def _tp() -> tuple[int, int]:
    """(vLLM's TP world size, this rank's TP rank)."""
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()


def _stages() -> int:
    """vLLM's pipeline stages."""
    from vllm.distributed import get_pp_group

    return get_pp_group().world_size


def _up_shard(runner, rank: int) -> torch.Tensor:
    """This rank's rows of the up-projection, as _shard_up_proj_tail takes them."""
    weight = runner.routed_output_transform.up_proj.weight
    if runner._up_proj_preshard:
        return weight
    shard = runner._up_proj_shard_size
    return weight.narrow(0, rank * shard, shard)


def _out_of_contract(runner, fused_output, shared_output, trunc_size) -> list[str]:
    """What keeps vLLM's call from being the op's, if anything; shapes, dtypes and layouts are the op's to judge."""
    transform = getattr(runner, "routed_output_transform", None)
    if (
        transform is None
        or getattr(transform, "norm", None) is None
        or getattr(transform, "up_proj", None) is None
    ):
        return ["no latent norm or up-projection"]
    config = getattr(runner, "moe_config", None)
    checks = (
        # vLLM 0.34.0 hands a latent MoE with shared experts the shared experts' width whether or not it padded, and
        # its tail keeps states[..., :trunc_size]: only a smaller size strips padded columns
        ("a padded hidden dim", trunc_size not in (None, shared_output.shape[-1])),
        ("the routed output comes reduced", bool(getattr(runner, "_fused_output_is_reduced", False))),
        # vLLM's tail then leaves the output unreduced; the op always reduces it
        ("the final all-reduce is skipped", bool(getattr(config, "skip_final_all_reduce", False))),
        # only the last stage's output leaves its workers, through the fail-stop's check (failstop.after_step)
        ("pipeline parallel", _stages() != 1),
    )
    return [name for name, bad in checks if bad]


def _group(device: torch.device, package: str):
    """The group state on ``device`` of the kernel in ``package``, set up through that kernel's runtime over vLLM's TP
    group by the first call that gets here (every rank's, outside a capture), its fail-stop watched after every step;
    None where the ranks agreed it cannot be."""
    key = (device, package)
    if key not in _GROUPS:
        from vllm.distributed import get_tp_group

        runtime = importlib.import_module(f"{package}.runtime")
        tp = get_tp_group()
        group = runtime.setup(tp.cpu_group, tp.rank_in_group, tp.world_size, device)
        if group is not None:
            failstop.watch(group)
        _GROUPS[key] = group
    return _GROUPS[key]


def _all_ranks(ok: bool, cpu_group) -> bool:
    """Whether every rank of ``cpu_group`` (None: this rank alone) passed ``ok``; every rank calls it at the same
    point."""
    if cpu_group is None:
        return ok
    flags = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(flags, ok, group=cpu_group)
    return all(flags)


def _agrees(ours: torch.Tensor, theirs: torch.Tensor) -> bool:
    """Whether the kernel's out is vLLM's within bf16 roundings: 2**-7 of each value plus 2**-5 of its row's rms (a
    missing or doubled reduction is off by whole rows)."""
    ours, theirs = ours.float(), theirs.float()
    rms = theirs.square().mean(-1, keepdim=True).sqrt()
    bound = 2.0**-7 * theirs.abs() + 2.0**-5 * rms
    return bool(((ours - theirs).abs() <= bound).all())


@vllm_route(
    "vllm.models.kimi_k3.amd.latent_moe_runner.ROCmLatentMoERunner._fused_forward",
    "vllm.models.kimi_k3.amd.latent_moe_runner.ROCmLatentMoERunner._shard_up_proj_tail",
    versions=("0.34.0",),
)
def _shard_up_proj_tail(self, fused_output, shared_output, trunc_size):
    """Kimi-K3's latent-MoE tail under TP: the routed partial [n, latent] and the shared partial [n, hidden] of a
    decode step, all-reduced, normalized and up-projected in one kernel that does the TP reductions over peer memory.
    Returns what vLLM's does: the reduced MoE output [n, hidden], the same on every rank (a fresh tensor; vLLM's own
    writes its up-projection into shared_output, which nothing reads after)."""
    failed = _out_of_contract(self, fused_output, shared_output, trunc_size)
    if failed:
        _once(f"out of contract ({', '.join(failed)})")
        return FallbackReason.OUT_OF_CONTRACT
    device, n = fused_output.device, fused_output.shape[0]
    if not 1 <= n <= _committed_rows():
        _once(f"rows past the committed schemas' 1 to {_committed_rows()}")
        return FallbackReason.NO_WORKLOAD
    world, rank = _tp()
    norm = self.routed_output_transform.norm
    args = _alike(fused_output, shared_output, _up_shard(self, rank), norm.weight)
    kwargs = {"tp_size": world, "eps": float(norm.variance_epsilon)}
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and all(d != device for d, _ in _GROUPS):
        _once("first reached inside a graph capture: the peer memory is set up outside one")
        return FallbackReason.NOT_WARMED
    from vllm.distributed import get_tp_group

    reason, package = _agreed(args, kwargs, (device, n, kwargs["eps"], capturing), get_tp_group().cpu_group)
    if reason is not None:
        _once(f"the dispatcher's verdict: {reason.value}")
        return reason
    group = _group(device, package)
    if group is None:
        return FallbackReason.OUT_OF_CONTRACT
    key = (device, n.bit_length())
    if _CHECK and not capturing and key not in _CHECKED:
        return _checked_call(self, group, key, args, kwargs, fused_output, shared_output, trunc_size)
    if not _CHECKED.get(key, True):
        return FallbackReason.INACCURATE
    return _served(args, kwargs)


def _alike(routed, shared, up_weight, norm_weight) -> tuple[torch.Tensor, ...]:
    """The operands at addresses the dispatcher takes on every rank: each on its alignment, else a copy (aligned, as
    every allocation); the shared partial, which the op writes, apart from the others, else a copy (vLLM reads its
    shared output no more)."""
    routed, shared, up_weight, norm_weight = (
        t if t.data_ptr() % _ALIGN == 0 else t.clone() for t in (routed, shared, up_weight, norm_weight)
    )
    if any(_overlap(shared, t) for t in (routed, up_weight, norm_weight)):
        shared = shared.clone()
    return routed, shared, up_weight, norm_weight


def _overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.data_ptr() < b.data_ptr() + b.nbytes and b.data_ptr() < a.data_ptr() + a.nbytes


def _agreed(args, kwargs, key, cpu_group) -> tuple[FallbackReason | None, str | None]:
    """The dispatcher's verdict on the call, the same on every rank of ``cpu_group``: (None, the package of the kernel
    it serves the call with) where every rank's serves it, else (the reason of the first rank, in rank order, whose
    declines it, None). Agreed at a key's first call, which every rank reaches at the same point, and kept for the
    key's later calls (their operands as alike, ``_alike``), but for NOT_WARMED: a Point warms at its first eager
    call."""
    if key in _VERDICTS:
        return _VERDICTS[key]
    served = kimi_k3_latent_moe_tail.serving(*args, **kwargs)
    local = served.value if isinstance(served, FallbackReason) else None
    reasons = [local]
    if cpu_group is not None:
        reasons = [None] * dist.get_world_size(cpu_group)
        dist.all_gather_object(reasons, local, group=cpu_group)
    verdict = next((FallbackReason(r) for r in reasons if r is not None), None)
    # the kernel's module (aklr's Python builders' metadata): its package holds the kernel's other sources
    package = None if verdict is not None else served.misc["module_name"].rpartition(".")[0]
    if verdict is not FallbackReason.NOT_WARMED:
        _VERDICTS[key] = (verdict, package)
    return verdict, package


def _served(args, kwargs) -> torch.Tensor:
    """The op's out on a call every rank's dispatcher agreed to serve: this rank declining it now would leave the
    others waiting in the kernel, so it raises instead."""
    out = kimi_k3_latent_moe_tail(*args, **kwargs)
    if isinstance(out, FallbackReason):
        raise RuntimeError(
            f"aklr latent-MoE tail: this rank's dispatcher declines ({out.value}) a call every TP rank's agreed to "
            "serve; the other ranks have launched it (AKLR_DISABLE=kimi_k3_latent_moe_tail@1.0 runs vLLM's tail)"
        )
    return out


def _checked_call(runner, group, key, args, kwargs, fused_output, shared_output, trunc_size):
    """The op's out, checked against vLLM's tail on the same operands; the ranks agree on the verdict, and where any
    rank's differs the bucket stays with vLLM (this call returns vLLM's out)."""
    given = shared_output.clone()  # both tails write it in place
    ours = _served(args, kwargs)
    theirs = _original(_shard_up_proj_tail)(runner, fused_output, given, trunc_size)
    agreed = _all_ranks(_agrees(ours, theirs), group.cpu_group)
    _CHECKED[key] = agreed
    if not agreed:
        _LOGGER.warning(
            "aklr latent-MoE tail: the kernel's out for %d rows differs from vLLM's on some TP rank; rows of %d bits "
            "stay with vLLM",
            fused_output.shape[0],
            key[1],
        )
        return theirs
    return ours
