###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).

"""An operator entry: one variant's kernels, ranked for dispatch."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import ClassVar

from aklr.core.platform.device import DeviceTarget
from aklr.core.runtime.dispatch._config import DispatchConfig
from aklr.core.runtime.dispatch._dispatch_key import DispatchKey
from aklr.core.runtime.dispatch._kernel_function import KernelFunction
from aklr.core.schema import Schema


class OperatorEntry:
    """The kernels of one variant's schemas, ranked per baseline and dispatch key by their measured workloads."""

    UNMEASURED: ClassVar[str] = ""
    """The slot of a direct call, or of a baseline no workload was measured against."""

    def __init__(
        self,
        schemas: list[Schema],
        config: DispatchConfig,
        target: DeviceTarget,
        min_speedup: float,
        baselines: Iterable[str] = (),
    ) -> None:
        """Rank the kernels of schemas, the dataset's files of one variant, against each of baselines (those any
        workload of the dataset was measured against) and UNMEASURED."""
        self.config = config
        self._ranked: dict[str, dict[DispatchKey, tuple[KernelFunction, ...]]] = {}
        self._best: dict[str, dict[tuple[tuple[str, int], ...], KernelFunction]] = {}
        kernels = {s.kernel.name: KernelFunction(s) for s in schemas if s.kernel is not None}

        for baseline in sorted({self.UNMEASURED, *baselines}):
            speedups: dict[DispatchKey, dict[str, dict[str, float]]] = {}
            for schema in schemas:
                for workload in schema.workloads:
                    if not schema.serves(workload, target, min_speedup, baseline):
                        continue
                    measured = speedups.setdefault(
                        DispatchKey.from_workload(schema, workload), {}
                    ).setdefault(schema.kernel.name, {})
                    evaluation = workload.evaluation(target)
                    for name, speedup in evaluation.speedup.items() if evaluation else ():
                        measured[name] = max(measured.get(name, 0.0), speedup.value)

            ranked = self._ranked.setdefault(baseline, {})
            wins: dict[tuple[tuple[str, int], ...], Counter[str]] = {}
            for key, by_kernel in speedups.items():
                rank_by = baseline
                # UNMEASURED ranks by the baseline most of the key's kernels were measured against
                if baseline == self.UNMEASURED:
                    common = Counter(b for measured in by_kernel.values() for b in measured)
                    rank_by = min(common, key=lambda b: (-common[b], b), default=self.UNMEASURED)
                names = sorted(by_kernel, key=lambda k: (-by_kernel[k].get(rank_by, 0.0), k))
                ranked[key] = tuple(kernels[name] for name in names)
                wins.setdefault(key.consts, Counter())[names[0]] += 1
            # a kernel is specialized to its Const axes: the best one is the best of those axes' keys
            self._best[baseline] = {consts: kernels[won.most_common(1)[0][0]] for consts, won in wins.items()}

    def slot(self, baseline: str | None) -> str:
        return baseline if baseline in self._ranked else self.UNMEASURED

    def candidates(self, key: DispatchKey, baseline: str | None) -> tuple[KernelFunction, ...]:
        """The kernels serving key against baseline (None: a direct call), best first."""
        return self._ranked[self.slot(baseline)].get(key, ())

    def lookup(self, key: DispatchKey, baseline: str | None) -> KernelFunction | None:
        """The best kernel for key against baseline; on a miss the config's on_miss_policy, None to fall back."""
        candidates = self.candidates(key, baseline)
        if candidates:
            return candidates[0]
        if self.config.on_miss_policy == "use_best_kernel":
            return self._best[self.slot(baseline)].get(key.consts)
        return None

    def prewarm(self) -> None:
        """Build ahead of the first call: per Const-axes point the config's aot_ratio of its key-winning kernels,
        and its best kernel when misses run it."""
        for baseline, ranked in self._ranked.items():
            if self.config.aot_ratio > 0.0:
                wins: dict[tuple[tuple[str, int], ...], Counter[KernelFunction]] = {}
                for key, kernels in ranked.items():
                    wins.setdefault(key.consts, Counter())[kernels[0]] += 1
                for won in wins.values():
                    for kernel, _ in won.most_common(max(1, int(len(won) * self.config.aot_ratio))):
                        kernel.runnable  # noqa: B018 -- builds it
            if self.config.on_miss_policy == "use_best_kernel":
                for kernel in self._best[baseline].values():
                    kernel.runnable  # noqa: B018 -- builds it
