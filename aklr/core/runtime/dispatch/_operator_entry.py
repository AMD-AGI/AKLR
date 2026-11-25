###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
#


from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import ClassVar

from aklr.core.platform.device import DeviceTarget
from aklr.core.runtime.dispatch._config import DispatchConfig
from aklr.core.runtime.dispatch._dispatch_key import DispatchKey
from aklr.core.runtime.dispatch._kernel_function import KernelFunction
from aklr.core.schema import Schema


class OperatorEntry:
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
        self.config = config
        self._ranked: dict[str, dict[DispatchKey, tuple[KernelFunction, ...]]] = {}
        self._best: dict[str, dict[DispatchKey, KernelFunction]] = {}
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
            wins: dict[DispatchKey, Counter[str]] = {}
            for key, by_kernel in speedups.items():
                rank_by = baseline
                if baseline == self.UNMEASURED:
                    common = Counter(b for measured in by_kernel.values() for b in measured)
                    rank_by = min(common, key=lambda b: (-common[b], b), default=self.UNMEASURED)
                names = sorted(by_kernel, key=lambda k: (-by_kernel[k].get(rank_by, 0.0), k))
                ranked[key] = tuple(kernels[name] for name in names)
                wins.setdefault(replace(key, vars=()), Counter())[names[0]] += 1
            self._best[baseline] = {fixed: kernels[won.most_common(1)[0][0]] for fixed, won in wins.items()}

    def slot(self, baseline: str | None) -> str:
        return baseline if baseline in self._ranked else self.UNMEASURED

    def candidates(self, key: DispatchKey, baseline: str | None) -> tuple[KernelFunction, ...]:
        return self._ranked[self.slot(baseline)].get(key, ())

    def lookup(self, key: DispatchKey, baseline: str | None) -> KernelFunction | None:
        candidates = self.candidates(key, baseline)
        if candidates:
            return candidates[0]
        if self.config.on_miss_policy == "use_best_kernel":
            return self._best[self.slot(baseline)].get(replace(key, vars=()))
        return None

    def prewarm(self) -> None:
        for baseline, ranked in self._ranked.items():
            if self.config.aot_ratio > 0.0:
                wins: dict[tuple[tuple[str, int], ...], Counter[KernelFunction]] = {}
                for key, kernels in ranked.items():
                    wins.setdefault(key.consts, Counter())[kernels[0]] += 1
                for won in wins.values():
                    for kernel, _ in won.most_common(max(1, int(len(won) * self.config.aot_ratio))):
                        kernel.runnable  # noqa: B018
            if self.config.on_miss_policy == "use_best_kernel":
                for kernel in self._best[baseline].values():
                    kernel.runnable  # noqa: B018
