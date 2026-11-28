# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""``AKLR_AITER_PLUGIN=capture``: AKLR serves aiter's routed ops only in sglang's decode CUDA graphs.

With AKLR among aiter's enabled providers, aiter routes every call of its routed ops through its plugin
interface, ten-odd microseconds of Python a call that an eager forward pays each time (a prefill forward
makes about 1600 such calls), while a graph replay pays none. In this mode the sglang plugin keeps AKLR
registered but not enabled, and this route enables it for each decode graph capture: the capture's eager
warm-ups (which build and warm AKLR's kernels) and the capture itself. Imported by the sglang plugin in
that mode only."""

from __future__ import annotations

from aklr.plugins.aiter.provider import disable, enable
from aklr.plugins.sglang.hook import sglang_route

__all__ = ["SOURCE"]

# the provenance aiter records with its routing decisions
SOURCE = "AKLR sglang plugin, AKLR_AITER_PLUGIN=capture"


@sglang_route(
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_one_shape",
    versions=(">=0.5.23.dev20261120",),
    wraps=True,
)
def _capture_one_shape(capture_one_shape, self, *args, **kwargs):
    """sglang's capture of one decode graph shape, its warm-ups included, with aiter asking AKLR
    first; AKLR is out of aiter's providers again afterwards, the capture done or raised."""
    enable(SOURCE)
    try:
        return capture_one_shape(self, *args, **kwargs)
    finally:
        disable(SOURCE)
