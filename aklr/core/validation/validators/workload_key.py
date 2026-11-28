###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from aklr.core.common.frozen_model import frozen_dataclass
from aklr.core.contract.spec import WorkloadSpec as Workload

__all__ = ["WorkloadKey"]


@frozen_dataclass
class WorkloadKey:
    axes: dict[str, int]

    @classmethod
    def from_workload(cls, workload: Workload) -> WorkloadKey:
        return cls(axes=workload.axes)

    def __str__(self) -> str:
        return ",".join(f"{a}={v}" for a, v in self.axes.items())

    @staticmethod
    def seed(schema_name: str, axes: Mapping[str, Any], salt: str) -> int:
        key = f"{schema_name}|{sorted(axes.items())}|{salt}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
