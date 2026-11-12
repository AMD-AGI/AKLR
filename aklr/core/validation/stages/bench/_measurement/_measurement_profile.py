###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The measurement profile: how the bench checks and times a workload, recorded in its report."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from pydantic import Field, TypeAdapter

from aklr.core.platform.env import DEFAULT_MIN_SPEEDUP
from aklr.core.schema._frozen_model import frozen_dataclass

__all__ = ["MeasurementProfile"]


@frozen_dataclass
class MeasurementProfile:
    num_trials: int = 3
    graph_budget_ms: float = 20.0
    max_log_ratio: float = Field(default_factory=lambda: math.log(2))
    entry: float = DEFAULT_MIN_SPEEDUP  # dispatch's default bar: a kernel validation admits is one it serves
    retain: float = 0.99

    @property
    def digest(self) -> str:
        """The protocol's name: this profile, the timer's sources, the comparison's statistics and those drawing
        the inputs (their seeds among them), so a change in any renames it."""
        import sys

        import aklr.core.utils as utils
        import aklr.core.validation.stages.bench._measurement._workload_result as comparison
        import aklr.core.validation.stages.bench._task._graph_timer as graph_timer
        import aklr.core.validation.stages.bench._workload_call as workload_call  # imports every sampler it holds
        import aklr.core.validation.stages.workload_key as workload_key

        # every sampler of the bench's, found by subclassing: one defined elsewhere (a test's) names no protocol
        samplers, pending = set(), [workload_call.InputSampler]
        while pending:
            sampler = pending.pop()
            samplers.add(sampler)
            pending += sampler.__subclasses__()
        sources = {
            sys.modules[s.__module__].__file__
            for s in samplers
            if s.__module__.startswith(workload_call.__name__)
        }
        digest = hashlib.sha256(TypeAdapter(MeasurementProfile).dump_json(self))
        for name in ("testing.py", "_kernel_profiler.py", "_kernel_tracer.cpp"):
            digest.update((Path(utils.__file__).parent / name).read_bytes())
        for path in [graph_timer.__file__, comparison.__file__, *sorted(sources), workload_key.__file__]:
            digest.update(Path(path).read_bytes())
        return digest.hexdigest()[:16]
