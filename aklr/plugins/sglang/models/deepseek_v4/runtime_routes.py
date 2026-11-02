# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""On sglang's monokernel runtime: a setup AKLR's spec refuses (or ``AKLR_DISABLE`` names)
keeps sglang's forward instead of stopping the server. A build that fails is not caught:
the ranks may stand in different collectives."""

from __future__ import annotations

import logging

from aklr.core.platform.env import RuntimeEnv
from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import (
    BACKEND,
    DRAFT,
    SPEC_NAMES,
    TARGET,
    VERSIONS,
    compat,
)

__all__ = ["built_models"]

_LOGGER = logging.getLogger(__name__)

RUNTIME = "sglang.srt.monokernels.runtime"
SGLANG_READS = {
    "srt/monokernels/runtime.py": (
        "install_monokernel_forward(model)",
        "install_monokernel_forward(hf_config)",
        "install_monokernel_forward(capture_tail_hooks)",
        "_backend",
    ),
    "srt/model_executor/model_runner.py": ("install_monokernel_forward",),
}

# the model classes whose monokernels this process built
_BUILT: set[str] = set()


def built_models() -> frozenset[str]:
    return frozenset(_BUILT)


def _tp_rank() -> int:
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().tp_rank


def _keep_sglang(model_class: str, reasons: list[str]) -> None:
    name = SPEC_NAMES[model_class]
    stats.set_info(f"{name}:refused", reasons)
    if _tp_rank() == 0:
        _LOGGER.warning(
            "%s not installed; sglang's forward runs every step of %s:\n  - %s",
            name,
            model_class,
            "\n  - ".join(reasons),
        )


@sglang_route(f"{RUNTIME}.install_monokernel_forward", versions=VERSIONS, around=True)
def _install(orig, *, model, hf_config, capture_tail_hooks):
    from sglang.srt.monokernels.runtime import _backend

    model_class = type(model).__name__
    if _backend() != BACKEND or model_class not in SPEC_NAMES:
        return orig(model=model, hf_config=hf_config, capture_tail_hooks=capture_tail_hooks)
    name = SPEC_NAMES[model_class]
    if name in RuntimeEnv().disabled_names:
        return _keep_sglang(model_class, [f"AKLR_DISABLE names {name}"])
    if model_class == DRAFT and TARGET not in _BUILT:
        return _keep_sglang(model_class, ["the target model runs sglang's forward"])
    refusal = f"--monokernel-backend {BACKEND}: "
    try:
        orig(model=model, hf_config=hf_config, capture_tail_hooks=capture_tail_hooks)
    except (ValueError, RuntimeError) as e:
        if not str(e).startswith(refusal):
            raise
        return _keep_sglang(model_class, str(e).removeprefix(refusal).split("; "))
    if compat.served_keys(model):
        _BUILT.add(model_class)
    return None
