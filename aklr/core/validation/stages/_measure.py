# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import dataclasses
import gc
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, get_args

from aklr.core.contract.semantics import derive_seed
from aklr.core.contract.spec import FrozenModel
from aklr.core.utils import testing
from aklr.core.validation.stages._report._finding import Finding, FindingCode

__all__ = [
    "ALPHA",
    "GRAPH_BUDGET_MILLISECONDS",
    "MAX_ABS_LOG_RATIO",
    "MAX_REGRESS_FRACTION",
    "MEASUREMENT_PROFILE",
    "MIN_GAIN_FRACTION",
    "N_ROUNDS",
    "N_WARMUP_ROUNDS",
    "THRESHOLD_RATIOS",
    "Clock",
    "Mode",
    "Participant",
    "RoundTimes",
    "Threshold",
    "TimingRecord",
    "Versus",
    "bind_clock",
    "compare_performance",
    "gate",
    "ratio_lower_bound",
    "timing_records",
]

MIN_GAIN_FRACTION = 0.02
MAX_REGRESS_FRACTION = 0.01
THRESHOLD_RATIOS: Mapping[Threshold, float] = MappingProxyType(
    {"entry": 1 + MIN_GAIN_FRACTION, "retain": 1 - MAX_REGRESS_FRACTION}
)
ALPHA = 0.05
MAX_ABS_LOG_RATIO = math.log(2)
N_ROUNDS = 32
N_WARMUP_ROUNDS = 1
GRAPH_BUDGET_MILLISECONDS = 20.0

MEASUREMENT_PROFILE: Mapping[str, Any] = MappingProxyType(
    {
        "version": 18,
        "timer": "bench_cudagraph",
        **dataclasses.asdict(testing.CudaGraphTimer()),
        "sample_count_policy": "rep_ms_estimate_capped_by_kernel_record_budget",
        "estimation_timer": "cuda_event_eager_with_cache_flush",
        "execution_mode": "cuda_graph",
        "metric": "gpu_duration_sum",
        "profiler": "rocprofiler_sdk_kernel_dispatch",
        "calls_per_graph": "adaptive",
        "cache_policy": "flush_before_each_call_in_graph",
        "cache_flush_access": "read",
        "cache_flush_kernel": "_flush_l2_cache_kernel",
        "profiler_warmup": "one_unrecorded_baseline_bench_cudagraph",
        "sample_aggregation": "mean_within_replay",
        "replay_aggregation": "median",
        "graph_budget_milliseconds": GRAPH_BUDGET_MILLISECONDS,
        "round_order": "seeded_shuffle_per_round",
        "rounds": N_ROUNDS,
        "warmup_rounds": N_WARMUP_ROUNDS,
        "max_abs_log_ratio": MAX_ABS_LOG_RATIO,
        "max_abs_log_ratio_scope": "gate_lower_bound_only",
        "trial_aggregation": "student_t_lower_bound_of_mean_log_ratio",
        "alpha": ALPHA,
        "concurrent_timing": "one_task_per_device, reference_tasks_may_run_on_neighbors",
    }
)

Participant = Literal["baseline", "candidate", "incumbent"]


Mode = Literal["graph"]
Versus = Literal["baseline", "incumbent"]
Threshold = Literal["entry", "retain"]


class TimingRecord(FrozenModel):
    """A measurement, never a verdict: ``gate`` decides it."""

    mode: Mode
    versus: Versus
    threshold: Threshold
    n_rounds: int
    log_ratio_mean: float
    ratio_lower_bound: float
    min_ratio: float
    versus_milliseconds: tuple[float, ...]
    candidate_milliseconds: tuple[float, ...]


class RoundTimes(FrozenModel):
    mode: Mode
    round_index: int
    order: tuple[Participant, ...]
    milliseconds: dict[Participant, float]


def ratio_lower_bound(log_ratios: Sequence[float], *, alpha: float = ALPHA) -> float:
    """One-sided Student-t lower bound of the mean speedup ratio."""
    n = len(log_ratios)
    if n < 2:
        raise ValueError(f"ratio_lower_bound needs at least 2 rounds, got {n}")
    # the 1 - alpha quantile of Student-t with n - 1 degrees of freedom, by bisection
    df = n - 1
    lo, hi = 0.0, 1.0
    for _ in range(100):
        x = (lo + hi) / 2
        lo, hi = (x, hi) if _beta_regularized(df / 2, 0.5, x) < 2 * alpha else (lo, x)
    t_quantile = math.sqrt(df * (1 - lo) / lo)
    half_width = t_quantile * statistics.stdev(log_ratios) / math.sqrt(n)
    return math.exp(statistics.fmean(log_ratios) - half_width)


def _beta_regularized(a: float, b: float, x: float) -> float:
    if x <= 0 or x >= 1:
        return float(x >= 1)
    if x > (a + 1) / (a + b + 2):
        return 1 - _beta_regularized(b, a, 1 - x)
    tiny = 1e-300
    c, d = 1.0, 1 / (1 - (a + b) * x / (a + 1))
    h = d
    for m in range(1, 400):
        even = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        for numerator in (even, odd):
            d = 1 + numerator * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + numerator / c
            c = c if abs(c) > tiny else tiny
            h *= c * d
        if abs(c * d - 1) < 1e-15:
            break
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return math.exp(log_front + a * math.log(x) + b * math.log1p(-x)) * h / a


def timing_records(
    rounds: Sequence[RoundTimes], thresholds: Mapping[Versus, Threshold]
) -> tuple[TimingRecord, ...]:
    """One record per mode and per ``versus`` in ``thresholds``."""
    records = []
    for mode in get_args(Mode):
        mode_rounds = [r for r in rounds if r.mode == mode]
        for versus, threshold in thresholds.items():
            if not all(versus in r.milliseconds for r in mode_rounds):
                raise ValueError(f"{mode} rounds without {versus} times")
            versus_milliseconds = tuple(r.milliseconds[versus] for r in mode_rounds)
            candidate_milliseconds = tuple(r.milliseconds["candidate"] for r in mode_rounds)
            if not all(map(math.isfinite, versus_milliseconds + candidate_milliseconds)):
                raise ValueError(f"{mode} rounds with a non-finite time")
            raw = [math.log(v / c) for v, c in zip(versus_milliseconds, candidate_milliseconds)]
            # clamped only for the gate's lower bound, against an outlier round
            log_ratios = [min(max(x, -MAX_ABS_LOG_RATIO), MAX_ABS_LOG_RATIO) for x in raw]
            records.append(
                TimingRecord(
                    mode=mode,
                    versus=versus,
                    threshold=threshold,
                    n_rounds=len(log_ratios),
                    log_ratio_mean=statistics.fmean(raw),
                    ratio_lower_bound=ratio_lower_bound(log_ratios),
                    min_ratio=math.exp(min(raw)),
                    versus_milliseconds=versus_milliseconds,
                    candidate_milliseconds=candidate_milliseconds,
                )
            )
    return tuple(records)


def gate(records: Sequence[TimingRecord]) -> FindingCode | None:
    """A finding when a ratio lower bound is below its threshold, else ``None``."""
    failed = {r.threshold for r in records if r.ratio_lower_bound < THRESHOLD_RATIOS[r.threshold]}
    if "retain" in failed:
        return FindingCode.REGRESSED
    return FindingCode.NOT_FASTER if failed else None


def compare_performance(
    rounds: Sequence[RoundTimes], thresholds: Mapping[Versus, Threshold]
) -> tuple[tuple[TimingRecord, ...], Finding | None]:
    """The timing records of ``rounds`` and the gate's finding."""
    try:
        records = timing_records(rounds, thresholds)
        code = gate(records)
    except (ValueError, ArithmeticError) as e:
        detail = f"unusable timing rounds: {type(e).__name__}: {e}"
        return (), Finding.unlocated(FindingCode.CANDIDATE_CRASH, detail)
    if code is None:
        return records, None
    detail = "; ".join(
        f"{t.mode} vs {t.versus} ratio_lower_bound {t.ratio_lower_bound:.4f} < "
        f"{THRESHOLD_RATIOS[t.threshold]:g}"
        for t in records
        if t.ratio_lower_bound < THRESHOLD_RATIOS[t.threshold]
    )
    return records, Finding.unlocated(code, detail)


class Clock(NamedTuple):
    synchronize: Callable
    bench_cudagraph: Callable

    def time_rounds(
        self,
        fns: Mapping[Participant, Callable[[], object]],
        *,
        seed: int,
    ) -> tuple[RoundTimes, ...]:
        """Every recorded round, after one unrecorded warm-up profile."""
        participants = list(fns)
        rounds = []
        is_gc_enabled = gc.isenabled()
        gc.disable()
        try:
            self.bench_cudagraph(fns["baseline"], rep=GRAPH_BUDGET_MILLISECONDS)
            for mode in get_args(Mode):
                for round_index in range(-N_WARMUP_ROUNDS, N_ROUNDS):
                    order_seed = derive_seed(seed, "order", mode, round_index)
                    order = random.Random(order_seed).sample(participants, k=len(participants))
                    milliseconds = {}
                    for p in order:
                        milliseconds[p] = self.bench_cudagraph(fns[p], rep=GRAPH_BUDGET_MILLISECONDS)
                    if round_index >= 0:
                        rounds.append(
                            RoundTimes(
                                mode=mode,
                                round_index=round_index,
                                order=tuple(order),
                                milliseconds=milliseconds,
                            )
                        )
        finally:
            if is_gc_enabled:
                gc.enable()
        return tuple(rounds)


def bind_clock() -> Clock:
    import torch

    return Clock(
        synchronize=torch.cuda.synchronize,
        bench_cudagraph=testing.bench_cudagraph,
    )
