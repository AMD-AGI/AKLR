# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR's monokernels for SGLang's --enable-amd-monokernel, in place of sgl-amd-mono's: one package
per model, its spec (``spec.py``) running each decoder layer as one call of a layer op. A spec is
registered only on the SGLang releases it was tested on, as a route is hooked."""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points

from aklr.plugins.sglang import hook

__all__ = ["register_specs"]

_LOGGER = logging.getLogger(__name__)

# the backend SGLang's --enable-amd-monokernel looks specs up under
BACKEND = "amd"
# each spec by HF architecture, and the SGLang releases it was tested on
SPECS = {
    "GlmMoeDsaForCausalLM": ("aklr.plugins.sglang.monokernels.glm53.spec:SPEC", ("0.5.23",)),
    "GlmMoeDsaForCausalLMNextN": ("aklr.plugins.sglang.monokernels.glm53.draft:SPEC", ("0.5.23",)),
}


def register_specs() -> None:
    """Register with SGLang every spec tested on the installed release; on another, none, and no spec module
    is imported (a release before the monokernel interface lacks it). Beside sgl-amd-mono, none either: SGLang
    takes one spec an architecture (``_ask_for_one_plugin``)."""
    installed = hook.sglang_version()
    tested = {arch: path for arch, (path, versions) in SPECS.items() if installed in versions}
    if untested := [arch for arch in SPECS if arch not in tested]:
        # a warning, as hook.install's: the line shows at any sglang log level
        _LOGGER.warning(
            "aklr registers no monokernel spec for %s on sglang %s, untested on it",
            ", ".join(untested),
            installed,
        )
    if not tested:
        return
    if _loads_sgl_amd_mono():
        _ask_for_one_plugin(tested)
        return
    from sglang.kernels.plugin_kernels.amd.spec import register

    for architecture, path in tested.items():
        register(backend=BACKEND, architecture=architecture, path=path)


def _ask_for_one_plugin(tested: dict) -> None:
    """Beside sgl-amd-mono: --enable-amd-monokernel on a model aklr has a spec for stops start-up and asks for
    one plugin."""
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

    @register_resolution_hook("handle_amd_monokernel")
    def one_plugin(server_args, previous):
        # as SGLang's own step reads them
        from sglang.srt.arg_groups.overrides import model_config_of, resolving_view

        if resolving_view(server_args).enable_amd_monokernel:
            architectures = model_config_of(server_args).hf_config.architectures or ()
            if served := [arch for arch in architectures if arch in tested]:
                raise ValueError(
                    f"--enable-amd-monokernel: aklr (AKLR_SGLANG_PLUGIN=1, a spec for {', '.join(served)}) and "
                    "sgl_amd_mono are both loaded, and SGLang takes one. Choose with SGLANG_PLUGINS=aklr or "
                    "SGLANG_PLUGINS=sgl_amd_mono (or AKLR_SGLANG_PLUGIN=0)"
                )
        previous(server_args)


def _loads_sgl_amd_mono() -> bool:
    """Whether SGLang's load_plugins runs sgl-amd-mono's entry point: installed and not left out
    by SGLANG_PLUGINS."""
    allowed = {n.strip() for n in os.environ.get("SGLANG_PLUGINS", "").split(",") if n.strip()}
    if allowed and "sgl_amd_mono" not in allowed:
        return False
    return any(ep.name == "sgl_amd_mono" for ep in entry_points(group="sglang.srt.plugins"))
