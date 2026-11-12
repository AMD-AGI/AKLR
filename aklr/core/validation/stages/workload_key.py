###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A workload's identity: what keys its result, its Evaluation, its report row and findings, and the seed of
its timing order."""

from __future__ import annotations

import hashlib

from aklr.core.schema import Workload
from aklr.core.schema._frozen_model import frozen_dataclass

__all__ = ["WorkloadKey"]


@frozen_dataclass
class WorkloadKey:
    axes: dict[str, int]  # equal keys ignore the order of the axes

    @classmethod
    def from_workload(cls, workload: Workload) -> WorkloadKey:
        return cls(axes=workload.axes)

    def __str__(self) -> str:
        """Its name, as `M=4`: the axes in the workload's order."""
        return ",".join(f"{a}={v}" for a, v in self.axes.items())

    def timing_seed(self, schema_name: str, trial: int) -> int:
        """The seed of a trial's timing order, the same in every process."""
        key = f"{schema_name}|{sorted(self.axes.items())}|timing {trial}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
