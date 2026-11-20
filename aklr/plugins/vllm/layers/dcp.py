# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The query gather of vLLM's MLA decode context parallelism (``MLADCPManager._gather_query``): the DCP group's
``all_gather`` along the head dim runs on RCCL (vLLM's communicator all-gathers with pynccl only). When the DCP group
has the TP group's ranks, the TP group's aiter custom all-gather does it instead (1.4-2x faster than RCCL from 16 KB
to 8 MB a rank, MI355X TP8): the same bytes in the same order."""

from __future__ import annotations

import logging

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []
_LOGGER = logging.getLogger(__name__)
# empty until the first eager call compares the custom gather with vLLM's pynccl one byte for byte; then whether
# every rank found them equal (one decision for the group: the custom gather needs every rank)
_CHECKED: list[bool] = []


@vllm_route(
    "vllm.v1.attention.ops.dcp.MLADCPManager._gather_query",
    "vllm.distributed.parallel_state.GroupCoordinator.all_gather",
    versions=("0.34.0", "0.32.1rc1.dev276"),
)
def _dcp_query_all_gather(self, input_, dim=-1):
    from vllm.distributed.parallel_state import get_dcp_group, get_tp_group

    if _CHECKED and not _CHECKED[0]:
        return FallbackReason.INACCURATE
    if dim != 1 or input_.dim() != 3 or self.world_size == 1 or self is not get_dcp_group():
        return FallbackReason.OUT_OF_CONTRACT
    tp = get_tp_group()
    if list(tp.ranks) != list(self.ranks):
        return FallbackReason.OUT_OF_CONTRACT
    ca = getattr(tp.device_communicator, "aiter_ar_comm", None)
    x = input_.contiguous()
    # aiter's custom gather moves 16- and 32-bit floats: a 1-byte query (fp8 under the a8w8 MLA decode) goes as bf16
    # pairs of its bytes, the same bytes
    if x.element_size() == 1:
        if x.shape[-1] % 2:
            return FallbackReason.OUT_OF_CONTRACT
        x = x.view(torch.bfloat16)
    elif x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return FallbackReason.OUT_OF_CONTRACT
    if ca is None or ca.disabled or not ca.should_custom_ag(x):
        return FallbackReason.OUT_OF_CONTRACT
    out = ca.custom_all_gather(x, dim=0)
    if out is None:
        return FallbackReason.OUT_OF_CONTRACT
    out = out.view(input_.dtype)
    # rank r's [m, h, d] is block r along the head dim, as vLLM's pynccl path lays it out
    w, (m, h, d) = self.world_size, input_.shape
    out = out.view(w, m, h, d).movedim(0, 1).reshape(m, w * h, d)
    if not _CHECKED and not torch.cuda.is_current_stream_capturing():
        ref = self.device_communicator.all_gather(input_, dim)  # vLLM's own path, once, eagerly
        same = torch.equal(ref.contiguous().view(torch.uint8), out.contiguous().view(torch.uint8))
        flag = torch.tensor([1 if same else 0], device=input_.device, dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=self.device_group)
        _CHECKED.append(bool(flag.item()))
        _LOGGER.warning(
            "aklr dcp query all-gather: custom vs pynccl %s (%s)",
            "equal" if _CHECKED[0] else "DIFFER",
            tuple(input_.shape),
        )
        if not _CHECKED[0]:
            return ref
    return out
