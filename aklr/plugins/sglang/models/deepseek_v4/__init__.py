# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1-Flash decode on gfx950 through sglang's monokernel plugin API: ``register()``
registers the specs as monokernel backend ``aklr`` and selects it when the setup passes their
checks; an sglang without the plugin API runs the model itself."""

from __future__ import annotations

import logging
import os
from importlib.metadata import version

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from aklr.core.platform.env import RuntimeEnv
from aklr.plugins.sglang import stats

__all__ = [
    "BACKEND",
    "DRAFT",
    "ENV_MONO",
    "MODEL_TYPE",
    "SERVED_TP",
    "SPECS",
    "SPEC_NAMES",
    "TARGET",
    "VERSIONS",
    "register",
    "serving",
]

_LOGGER = logging.getLogger(__name__)

BACKEND = "aklr"
ENV_MONO = "AKLR_DSV41_MONO"
# "0": an explicit --monokernel-backend amd keeps sglang's own monokernels
ENV_TAKEOVER = "AKLR_DSV41_TAKEOVER"
# the backends AKLR's stands in for when the setup passes its checks (AKLR vendors
# sgl-amd-mono's kernels' tuned successors)
TAKEOVER_FROM = frozenset({"amd"})
# the sglang releases the routes and specs were tested on (a dev or rc build: its release)
VERSIONS = ">=0.5.22,<0.5.23"
TARGET = "DeepseekV4ForCausalLM"
DRAFT = "DeepseekV4ForCausalLMDSpark"
# a spec module imports only when a server runs this backend
SPECS = {
    TARGET: f"{__name__}.spec:SPEC",
    DRAFT: f"{__name__}.draft:SPEC",
}
SPEC_NAMES = {TARGET: "aklr_deepseek_v41", DRAFT: "aklr_deepseek_v41_dspark_draft"}
# hf_config.model_type of DeepSeek-V4.1: DeepSeek-V4 is served by the same classes
MODEL_TYPE = "deepseek_v41"
# the TP sizes the routes of the steps around the monokernels were tested on
SERVED_TP = (2, 4)
# set by the target model's __init__ (routes.py)
_SERVING = False


def serving() -> bool:
    """Whether this process builds a DeepSeek-V4.1 target on gfx950 at a TP size of
    ``SERVED_TP``. The routes around the monokernels act only then: another model on the
    classes they hook, another GPU or TP size keeps sglang's code."""
    return _SERVING


def register() -> None:
    mono = os.environ.get(ENV_MONO) or "1"
    if mono not in ("0", "1"):
        raise ValueError(f"{ENV_MONO}={mono!r}: expected '1' or '0'")
    if mono == "0":
        _LOGGER.info("DeepSeek-V4.1 monokernels off (%s=0)", ENV_MONO)
        return
    running = version("sglang")
    if not SpecifierSet(VERSIONS).contains(Version(running).base_version):
        _LOGGER.warning(
            "DeepSeek-V4.1: sglang %s is not in %s, which AKLR's monokernels were "
            "tested on; sglang runs the model",
            running,
            VERSIONS,
        )
        return
    try:
        from sglang.srt.monokernels.spec import register as register_spec
    except ImportError:
        _LOGGER.warning(
            "DeepSeek-V4.1: this sglang has no monokernel plugin API (sglang.srt.monokernels); "
            "sglang runs the model"
        )
        return
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

    from aklr.plugins.sglang.models.deepseek_v4 import (  # noqa: F401
        accept,
        compat,
        compressor_routes,
        dp_routes,
        engram_routes,
        inject,
        jit_routes,
        metadata_routes,
        proposal,
        radix_routes,
        routes,
        runtime_routes,
        tokenize_routes,
    )

    for architecture, path in SPECS.items():
        register_spec(backend=BACKEND, architecture=architecture, path=path)
    register_resolution_hook("handle_monokernel")(_select_backend)


class _WithBackend:
    """The resolving view with ``monokernel_backend`` set, for sglang's own checks."""

    def __init__(self, view, backend: str):
        self._view, self.monokernel_backend = view, backend

    def __getattr__(self, name):
        return getattr(self._view, name)


def _takeover() -> bool:
    value = os.environ.get(ENV_TAKEOVER) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV_TAKEOVER}={value!r}: expected '1' or '0'")
    return value == "1"


def _select_backend(server_args, previous) -> None:
    """``handle_monokernel`` with backend ``aklr`` on a DeepSeek-V4.1 server that names
    no backend, or sgl-amd-mono's (``TAKEOVER_FROM``), and passes the spec's startup
    checks; runs in the launching process."""
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        model_config_of,
        resolving_view,
    )
    from sglang.srt.monokernels.guards import device_isa, monokernel_disable_reasons

    cfg = resolving_view(server_args)
    requested = cfg.monokernel_backend
    if requested is None or (requested in TAKEOVER_FROM and _takeover()):
        model_config = model_config_of(server_args)
        if TARGET in (model_config.hf_config.architectures or ()):
            if SPEC_NAMES[TARGET] in RuntimeEnv().disabled_names:
                reasons = [f"AKLR_DISABLE names {SPEC_NAMES[TARGET]}"]
            else:
                try:
                    _, reasons = monokernel_disable_reasons(
                        cfg=_WithBackend(cfg, BACKEND),
                        model_config=model_config,
                        isa=device_isa(),
                    )
                except Exception as e:  # noqa: BLE001
                    reasons = [f"checking the setup: {type(e).__name__}: {e}"]
            if reasons:
                _LOGGER.warning(
                    "DeepSeek-V4.1 monokernels not selected; %s runs every step:\n  - %s",
                    ("sglang's forward" if requested is None else f"backend {requested!r}"),
                    "\n  - ".join(reasons),
                )
                stats.set_info("backend_refused", reasons)
            else:
                declare_resolution(server_args, "aklr.plugins.sglang", monokernel_backend=BACKEND)
                _LOGGER.info(
                    "DeepSeek-V4.1: monokernel backend %r selected%s",
                    BACKEND,
                    "" if requested is None else f" in place of {requested!r}",
                )
                stats.set_info("backend_selected", {"backend": BACKEND, "requested": requested})
    elif cfg.monokernel_backend != BACKEND:
        _LOGGER.info(
            "DeepSeek-V4.1: --monokernel-backend %s given; AKLR's backend is %r",
            cfg.monokernel_backend,
            BACKEND,
        )
    previous(server_args)
