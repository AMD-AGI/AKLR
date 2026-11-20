###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import hashlib
import math
from pathlib import Path

from pydantic import Field, TypeAdapter

from aklr.core.schema._frozen_model import frozen_dataclass

__all__ = ["MeasurementProfile"]


@frozen_dataclass
class MeasurementProfile:
    num_trials: int = 3
    graph_budget_ms: float = 20.0
    max_log_ratio: float = Field(default_factory=lambda: math.log(2))

    @property
    def digest(self) -> str:
        import aklr.core.utils as utils

        bench = Path(__file__).parents[1]
        measure = ("_op_bench.py", "_built_kernel.py", "_input_sampler.py", "_output_check.py", "_result.py")
        sources = [
            *(
                Path(utils.__file__).parent / name
                for name in ("testing.py", "_kernel_profiler.py", "_kernel_tracer.cpp")
            ),
            *(bench / "_measure" / name for name in measure),
            *sorted((bench / "_ops").glob("*.py")),
            bench / "_plan" / "_threshold.py",
            bench / "_plan" / "_opponent.py",
            bench.parent / "workload_key.py",
        ]
        digest = hashlib.sha256(TypeAdapter(MeasurementProfile).dump_json(self))
        for path in sources:
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]
