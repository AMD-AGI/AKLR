# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The latent-MoE tail's api and its vLLM route.

The op's kernel waits inside itself on every other rank of the TP group, so a rank that serves a call while another
declines it leaves the first spinning until its wait gives up. Every decision here therefore reads only what is the
same on every rank: the step's rows, the layer's shapes, dtypes and layouts, vLLM's config, the environment, whether
a graph is being captured, and verdicts the ranks agree on over the TP group (the peer memory's setup, the first-call
check against vLLM's tail)."""

from __future__ import annotations

import logging
import os

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import _original, vllm_route
from aklr.plugins.vllm.models.kimi_k3.latent_moe import runtime
from aklr.plugins.vllm.models.kimi_k3.latent_moe.contract import KIMI_K3_LATENT_MOE_TAIL

__all__ = ["kimi_k3_latent_moe_tail"]

_LOGGER = logging.getLogger(__name__)
# AKLR_VLLM_MOE_TAIL=0 hands every call straight back to vLLM; AKLR_VLLM_MOE_TAIL_CHECK=0 skips the first-call check
_ENABLED = os.environ.get("AKLR_VLLM_MOE_TAIL", "1") != "0"
_CHECK = os.environ.get("AKLR_VLLM_MOE_TAIL_CHECK", "1") != "0"
TP, HIDDEN, LATENT, MAX_ROWS = 8, 7168, 3584, 128
_SEEN: set[str] = set()
# the group state by device, None where the ranks agreed it cannot be set up
_GROUPS: dict[torch.device, runtime.TailGroup | None] = {}
# the first-call check's agreed verdict by (device, row bucket)
_CHECKED: dict[tuple[torch.device, int], bool] = {}


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


def _tp() -> tuple[int, int]:
    """(vLLM's TP world size, this rank's TP rank)."""
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()


def _up_shard(runner, rank: int) -> torch.Tensor:
    """This rank's rows of the up-projection, as _shard_up_proj_tail takes them."""
    weight = runner.routed_output_transform.up_proj.weight
    if runner._up_proj_preshard:
        return weight
    shard = runner._up_proj_shard_size
    return weight.narrow(0, rank * shard, shard)


def _out_of_contract(runner, fused_output, shared_output, trunc_size) -> list[str]:
    """What keeps the call from the op, if anything."""
    transform = getattr(runner, "routed_output_transform", None)
    norm = getattr(transform, "norm", None)
    if transform is None or norm is None or getattr(transform, "up_proj", None) is None:
        return ["no latent norm or up-projection"]
    world, rank = _tp()
    n = fused_output.shape[0] if fused_output.dim() == 2 else 0
    up = _up_shard(runner, rank)

    # the devices are the dispatcher's to check: its key takes no tensor off its device
    def dense_bf16(t, shape):
        return tuple(t.shape) == shape and t.dtype == torch.bfloat16 and t.is_contiguous()

    checks = (
        ("a padded hidden dim", trunc_size is not None),
        ("the routed output comes reduced", bool(getattr(runner, "_fused_output_is_reduced", False))),
        (f"TP {world}", world != TP),
        (f"{n} rows", not 1 <= n <= MAX_ROWS),
        ("fused_output", not dense_bf16(fused_output, (n, LATENT))),
        ("shared_output", not dense_bf16(shared_output, (n, HIDDEN))),
        ("up_proj shard", not dense_bf16(up, (HIDDEN // TP, LATENT))),
        ("latent norm weight", not dense_bf16(norm.weight, (LATENT,))),
    )
    return [name for name, bad in checks if bad]


def _group(device: torch.device) -> runtime.TailGroup | None:
    """The kernel's group state on ``device``, set up over vLLM's TP group by the first call that gets here (every
    rank's, outside a capture); None where the ranks agreed it cannot be."""
    if device not in _GROUPS:
        from vllm.distributed import get_tp_group

        tp = get_tp_group()
        _GROUPS[device] = runtime.setup(tp.cpu_group, tp.rank_in_group, tp.world_size, device)
    return _GROUPS[device]


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
    versions=("0.34.0", "0.32.1rc1.dev276"),
)
def _shard_up_proj_tail(self, fused_output, shared_output, trunc_size):
    """Kimi-K3's latent-MoE tail under TP: the routed partial [n, latent] and the shared partial [n, hidden] of a
    decode step, all-reduced, normalized and up-projected in one kernel that does the TP reductions over peer memory.
    Returns what vLLM's does: the reduced MoE output [n, hidden], the same on every rank (a fresh tensor; vLLM's own
    writes its up-projection into shared_output, which nothing reads after)."""
    if not _ENABLED:
        return FallbackReason.OUT_OF_CONTRACT
    failed = _out_of_contract(self, fused_output, shared_output, trunc_size)
    if failed:
        _once(f"out of contract ({', '.join(failed)}): fused_output {tuple(fused_output.shape)}/"
              f"{fused_output.stride()} {fused_output.dtype}, shared_output {tuple(shared_output.shape)}/"
              f"{shared_output.stride()} {shared_output.dtype}")  # fmt: skip
        return FallbackReason.OUT_OF_CONTRACT
    device = fused_output.device
    norm = self.routed_output_transform.norm
    args = (fused_output, shared_output, _up_shard(self, _tp()[1]), norm.weight)
    kwargs = {"tp_size": TP, "eps": float(norm.variance_epsilon)}
    reason = kimi_k3_latent_moe_tail.fallback_reason(*args, **kwargs)
    if reason is not None:
        _once(f"{reason.value}: {fused_output.shape[0]} rows")
        return reason
    capturing = torch.cuda.is_current_stream_capturing()
    if device not in _GROUPS and capturing:
        _once("first reached inside a graph capture: the peer memory is set up outside one")
        return FallbackReason.NOT_WARMED
    group = _group(device)
    if group is None:
        return FallbackReason.OUT_OF_CONTRACT
    key = (device, fused_output.shape[0].bit_length())
    if _CHECK and not capturing and key not in _CHECKED:
        return _checked_call(self, group, key, args, kwargs, fused_output, shared_output, trunc_size)
    if not _CHECKED.get(key, True):
        return FallbackReason.INACCURATE
    return kimi_k3_latent_moe_tail(*args, **kwargs)


def _checked_call(runner, group, key, args, kwargs, fused_output, shared_output, trunc_size):
    """The op's out, checked against vLLM's tail on the same operands; the ranks agree on the verdict, and where any
    rank's differs the bucket stays with vLLM (this call returns vLLM's out)."""
    ours = kimi_k3_latent_moe_tail(*args, **kwargs)
    if isinstance(ours, FallbackReason):
        return ours
    # stream-ordered after the kernel has read shared_output, which vLLM's tail writes
    theirs = _original(_shard_up_proj_tail)(runner, fused_output, shared_output, trunc_size)
    agreed = runtime.tp_agree(_agrees(ours, theirs), group.cpu_group)
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
