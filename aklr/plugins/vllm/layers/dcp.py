# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The query gather of vLLM's MLA decode context parallelism (``MLADCPManager._gather_query``), handed to the TP
group where the two groups have the same ranks.

vLLM gives aiter's all-gather to the group named "tp" alone (``CudaCommunicator``: its ``aiter_ar_comm`` and
``use_aiter_tp_all_gather``), so the DCP group's ``all_gather`` along the head dim runs on RCCL through
torch.distributed. With the TP group's ranks, the same gather is one the TP group can do: the route asks vLLM's TP
group for it (``GroupCoordinator.all_gather``, which on ROCm is ``CudaCommunicator._maybe_aiter_all_gather``), the
same bytes in the same order. The route calls nothing of aiter's: inside a CUDA graph capture as outside one, what
runs is the path vLLM's own TP all-gathers take.

The first eager call is compared byte for byte with the DCP group's own gather, one verdict for all ranks; a
difference turns the route off. ``AKLR_DISABLE=vllm_dcp_query_gather@1.0`` turns it off too (the route has no
contract: the id names it for that alone). Run on vLLM 0.34.0's image (aiter 0.1.28.post1), MI355X, TP8 with DCP8."""

from __future__ import annotations

import functools
import logging

import torch

from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []
_LOGGER = logging.getLogger(__name__)
# what AKLR_DISABLE names the route by
ROUTE_ID = "vllm_dcp_query_gather@1.0"
# empty until the first eager call compares the TP group's gather with the DCP group's own byte for byte; then
# whether every rank found them equal (one decision for the group: a gather needs every rank on one path)
_CHECKED: list[bool] = []


@functools.cache
def _enabled() -> bool:
    """Whether AKLR_DISABLE leaves the route on, read once."""
    return ROUTE_ID not in RuntimeEnv().disabled_contracts


@vllm_route(
    "vllm.v1.attention.ops.dcp.MLADCPManager._gather_query",
    "vllm.distributed.parallel_state.GroupCoordinator.all_gather",
    versions=("0.34.0", "0.32.1rc1.dev276"),
)
def _dcp_query_all_gather(self, input_, dim=-1):
    from vllm.distributed.parallel_state import get_dcp_group, get_tp_group

    if not _enabled():
        return FallbackReason.DISABLED
    if _CHECKED and not _CHECKED[0]:
        return FallbackReason.INACCURATE
    if dim != 1 or input_.dim() != 3 or self.world_size == 1 or self is not get_dcp_group():
        return FallbackReason.OUT_OF_CONTRACT
    tp = get_tp_group()
    # the TP group's gather is aiter's only where vLLM turned it on for that group: elsewhere it is the RCCL gather
    # the DCP group runs already
    if (
        tp is self
        or list(tp.ranks) != list(self.ranks)
        or not getattr(tp.device_communicator, "use_aiter_tp_all_gather", False)
    ):
        return FallbackReason.OUT_OF_CONTRACT
    x = input_.contiguous()
    # vLLM's TP gather hands aiter 16- and 32-bit floats: a 1-byte query (fp8 under the a8w8 MLA decode) goes as bf16
    # pairs of its bytes, the same bytes
    if x.element_size() == 1:
        if x.shape[-1] % 2:
            return FallbackReason.OUT_OF_CONTRACT
        x = x.view(torch.bfloat16)
    elif x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return FallbackReason.OUT_OF_CONTRACT
    out = tp.all_gather(x, 1)
    # rank r's [m, h, d] is block r along the head dim, as the DCP group's own gather lays it out
    if out.dtype != x.dtype or tuple(out.shape) != (x.shape[0], self.world_size * x.shape[1], x.shape[2]):
        _LOGGER.warning(
            "aklr dcp query all-gather: the TP group's gather returned %s %s for %s; the route stays with vLLM",
            out.dtype,
            tuple(out.shape),
            tuple(x.shape),
        )
        return FallbackReason.OUT_OF_CONTRACT
    out = out.contiguous().view(input_.dtype)
    if not _CHECKED and not torch.cuda.is_current_stream_capturing():
        ref = self.device_communicator.all_gather(input_, dim)  # the DCP group's own path, once, eagerly
        same = torch.equal(ref.contiguous().view(torch.uint8), out.view(torch.uint8))
        flag = torch.tensor([1 if same else 0], device=input_.device, dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=self.device_group)
        _CHECKED.append(bool(flag.item()))
        _LOGGER.warning(
            "aklr dcp query all-gather: the TP group's gather vs the DCP group's %s (%s)",
            "equal" if _CHECKED[0] else "DIFFER",
            tuple(input_.shape),
        )
        if not _CHECKED[0]:
            return ref
    return out
