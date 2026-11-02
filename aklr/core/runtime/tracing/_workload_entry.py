###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
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
