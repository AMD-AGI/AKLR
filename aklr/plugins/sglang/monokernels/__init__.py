# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR's monokernels for SGLang's --enable-amd-monokernel, in place of sgl-amd-mono's: one package
per model, its spec (``spec.py``) running each decoder layer as one call of a layer op."""

from __future__ import annotations

import os
from importlib.metadata import entry_points

__all__ = ["register_specs"]

# the backend SGLang's --enable-amd-monokernel looks specs up under
BACKEND = "amd"
# each spec by HF architecture
SPECS = {
    "GlmMoeDsaForCausalLM": "aklr.plugins.sglang.monokernels.glm53.spec:SPEC",
}


def register_specs() -> None:
    """Register every spec with SGLang, unless SGLang loads sgl-amd-mono too: both register under
    the same backend, where a second spec for an architecture is an error."""
    if _loads_sgl_amd_mono():
        return
    from sglang.kernels.plugin_kernels.amd.spec import register

    for architecture, path in SPECS.items():
        register(backend=BACKEND, architecture=architecture, path=path)


def _loads_sgl_amd_mono() -> bool:
    """Whether SGLang's load_plugins runs sgl-amd-mono's entry point: installed and not left out
    by SGLANG_PLUGINS."""
    allowed = {n.strip() for n in os.environ.get("SGLANG_PLUGINS", "").split(",") if n.strip()}
    if allowed and "sgl_amd_mono" not in allowed:
        return False
    return any(ep.name == "sgl_amd_mono" for ep in entry_points(group="sglang.srt.plugins"))
