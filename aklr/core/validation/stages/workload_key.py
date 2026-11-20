###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A workload's identity: what keys its result, what it writes back, its report row and findings, and its seeds."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from aklr.core.contract.spec import WorkloadSpec as Workload
from aklr.core.validation._frozen_model import frozen_dataclass

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

    @staticmethod
    def seed(schema_name: str, axes: Mapping[str, Any], salt: str) -> int:
        """A seed of the workload at axes of the schema named, the same in every process; salt tells its uses
        apart, as a trial's timing order."""
        key = f"{schema_name}|{sorted(axes.items())}|{salt}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
