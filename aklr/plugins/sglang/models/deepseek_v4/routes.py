# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DeepSeek-V4.1 forwards (sglang's monokernel runtime runs the served steps: whatever
reaches them is a fallback, named and counted) and the target's ``__init__``, where a process
learns whether it serves DeepSeek-V4.1."""

from __future__ import annotations

import logging

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import RouteFallback, sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import MODEL_TYPE, SERVED_TP, VERSIONS, serving

__all__ = ["DRAFT_FORWARD", "TARGET_FORWARD", "TARGET_INIT", "VERSIONS"]

_LOGGER = logging.getLogger(__name__)

TARGET_INIT = "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.__init__"
TARGET_FORWARD = "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.forward"
DRAFT_FORWARD = "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.forward"


def _why(target: str, model, forward_batch) -> RouteFallback:
    # imported here: every route module imports this one for VERSIONS, compat needs sglang
    from aklr.plugins.sglang.models.deepseek_v4 import compat

    mode = forward_batch.forward_mode
    backend = compat.backend()
    if backend != "aklr":
        return RouteFallback(f"monokernel backend {backend}")
    if not (mode.is_decode() or mode.is_target_verify()):
        return RouteFallback(mode.name.lower())
    served = compat.served_keys(model)
    key = compat.shape_key(forward_batch)
    rows = int(forward_batch.input_ids.shape[0])
    reason = "no spec built" if not served else "shape not served"
    counter = (
        stats.device_counter(f"{target}:{reason}", forward_batch.input_ids.device) if serving() else None
    )
    if counter is not None:
        counter.add_(1)
    return RouteFallback(reason, f"{key} ({rows} rows)")


@sglang_route(TARGET_FORWARD, versions=VERSIONS)
def _target_forward(self, input_ids, positions, forward_batch, *args, **kwargs):
    return _why(TARGET_FORWARD, self, forward_batch)


@sglang_route(DRAFT_FORWARD, versions=VERSIONS)
def _draft_forward(self, input_ids, positions, forward_batch, *args, **kwargs):
    return _why(DRAFT_FORWARD, self, forward_batch)


def _not_served() -> list[str]:
    """Why this process's GPU or TP size is not one the step routes were tested on."""
    import torch

    reasons = []
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        if torch.version.hip is None or not torch.cuda.is_available():
            reasons.append("no ROCm GPU")
        else:
            device = torch.cuda.current_device()
            isa = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            if isa != "gfx950":
                reasons.append(f"{isa}, not gfx950")
        tp = get_tp_group().world_size
        if tp not in SERVED_TP:
            reasons.append(f"TP {tp}, not one of {list(SERVED_TP)}")
    except Exception as e:  # noqa: BLE001
        reasons.append(f"checking the setup: {type(e).__name__}: {e}")
    return reasons


@sglang_route(TARGET_INIT, versions=VERSIONS, around=True)
def _target_init(orig, self, config, *args, **kwargs):
    import aklr.plugins.sglang.models.deepseek_v4 as dsv4

    if getattr(config, "model_type", None) == MODEL_TYPE and not serving():
        reasons = _not_served()
        if reasons:
            stats.set_info("step_routes", reasons)
            _LOGGER.warning(
                "DeepSeek-V4.1: AKLR's routes around the monokernels are off (%s)",
                "; ".join(reasons),
            )
        else:
            from aklr.plugins.sglang.models.deepseek_v4 import jit_routes

            dsv4._SERVING = True
            stats.set_info("step_routes", "on")
            jit_routes.activate()
    return orig(self, config, *args, **kwargs)
