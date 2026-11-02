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

from __future__ import annotations

from collections.abc import Hashable
from functools import cache
from typing import Any

import torch

from aklr.core.runtime.tracing._config import TracingConfig, TracingConfigRegistry
from aklr.core.runtime.tracing._policy import PolicyRegistry
from aklr.core.runtime.tracing._workload_entry import WorkloadEntry
from aklr.core.utils import is_dtype_integer


@PolicyRegistry.register_input_dump_policy("dump_all")
class DumpAllPolicy:
    def dump(self, inputs: dict[str, Any]) -> list[str]:
        return [name for name, val in inputs.items() if isinstance(val, torch.Tensor)]


@PolicyRegistry.register_input_dump_policy("dump_none")
class DumpNonePolicy:
    def dump(self, inputs: dict[str, Any]) -> list[str]:
        return []


@PolicyRegistry.register_input_dump_policy("dump_int")
class DumpIntPolicy:
    def dump(self, inputs: dict[str, Any]) -> list[str]:

        return [
            name
            for name, val in inputs.items()
            if isinstance(val, torch.Tensor) and is_dtype_integer(val.dtype)
        ]


@PolicyRegistry.register_filter_policy("keep_all")
class KeepAllPolicy:
    def __init__(self):
        self.entries: list[WorkloadEntry] = []

    def submit(self, entry: WorkloadEntry) -> None:
        self.entries.append(entry)

    def drain(self) -> list[WorkloadEntry]:
        result = self.entries
        self.entries = []
        return result

    def reset(self) -> None:
        self.entries.clear()


@PolicyRegistry.register_filter_policy("keep_first_k")
class KeepFirstKPolicy:
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.entries: list[WorkloadEntry] = []

    def submit(self, entry: WorkloadEntry) -> None:
        if len(self.entries) < self.k:
            self.entries.append(entry)

    def drain(self) -> list[WorkloadEntry]:
        result = self.entries
        self.entries = []
        return result

    def reset(self) -> None:
        self.entries.clear()


@PolicyRegistry.register_filter_policy("keep_first")
class KeepFirstPolicy(KeepFirstKPolicy):
    def __init__(self):
        super().__init__(k=1)


@PolicyRegistry.register_filter_policy("keep_none")
class KeepNonePolicy:
    def __init__(self):
        pass

    def submit(self, entry: WorkloadEntry) -> None:
        pass

    def drain(self) -> list[WorkloadEntry]:
        return []

    def reset(self) -> None:
        pass


@PolicyRegistry.register_filter_policy("keep_first_k_by_axes")
class KeepFirstKByAxesPolicy:
    def __init__(self, k: int = 1):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.seen_counts: dict[Hashable, int] = {}
        self.entries: list[WorkloadEntry] = []

    def submit(self, entry: WorkloadEntry) -> None:
        key = tuple(sorted(entry.axes.items()))
        count = self.seen_counts.get(key, 0)
        if count < self.k:
            self.seen_counts[key] = count + 1
            self.entries.append(entry)

    def drain(self) -> list[WorkloadEntry]:
        result = self.entries
        self.entries = []
        return result

    def reset(self) -> None:
        self.seen_counts.clear()
        self.entries.clear()


@cache
def get_default_configs() -> TracingConfigRegistry:
    return TracingConfigRegistry(default=TracingConfig())
