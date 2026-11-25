###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
#
###############################################################################

from dataclasses import dataclass
from typing import Any


@dataclass
class WorkloadEntry:
    schema_name: str

    axes: dict[str, int]

    inputs_to_dump: dict[str, Any]

    order: int

    cuda_graph_snapshot: dict[str, Any] | None = None
