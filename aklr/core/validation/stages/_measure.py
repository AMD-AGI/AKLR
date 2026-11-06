# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import gc
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, get_args

from aklr.core.contract.semantics import derive_seed
from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._report._finding import Finding, FindingCode

__all__ = [
    "ALPHA",
    "GRAPH_BUDGET_MILLISECONDS",
    "MAX_ABS_LOG_RATIO",
    "MAX_REGRESS_FRACTION",
    "MEASUREMENT_PROFILE",
    "MIN_GAIN_FRACTION",
    "LOOK_ALPHA",
    "N_WARMUP_ROUNDS",
    "ROUND_LOOKS",
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
    "is_tracer_fault",
    "is_tracer_refusal",
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
# the round counts at which the bounds are read; timing stops at the first that decides them all.
# None at 8: eight rounds can all miss a slow mode a quarter of the rounds take, and the t bound
# then passes a kernel slower than the threshold far more often than alpha
ROUND_LOOKS = (16, 32)
# alpha split evenly over the looks, so that stopping early keeps the false-pass rate within ALPHA
LOOK_ALPHA = ALPHA / len(ROUND_LOOKS)
N_WARMUP_ROUNDS = 1
GRAPH_BUDGET_MILLISECONDS = 20.0

MEASUREMENT_PROFILE: Mapping[str, Any] = MappingProxyType(
    {
        "version": 19,
        "timer": "bench_cudagraph",
        "sample_count_policy": "rep_ms_estimate",
        "estimation_samples": 5,
        "estimation_timer": "cuda_event_eager_with_cache_flush",
        "capture_warmup_runs": 1,
        "execution_mode": "cuda_graph",
        "metric": "gpu_duration_sum",
        "profiler": "aklr_kernel_tracer",
        "calls_per_graph": "adaptive",
        "cache_policy": "flush_before_each_call_in_graph",
        "cache_flush_bytes": 1 << 30,
        "cache_flush_access": "read",
        "cache_flush_kernel": "_flush_l2_cache_kernel",
        "profiler_warmup": "one_unrecorded_baseline_bench_cudagraph",
        "tracer_fault_policy": "measure_again_once_in_place_then_rerun_task_once",
        "measurement_replays": 10,
        "sample_aggregation": "mean_over_replays_per_call",
        "call_aggregation": "median",
        "graph_budget_milliseconds": GRAPH_BUDGET_MILLISECONDS,
        "round_order": "seeded_shuffle_per_round",
        "round_looks": ROUND_LOOKS,
        "round_stop": "gate_baseline_entry_and_min_speedup_decided_at_a_look",
        "warmup_rounds": N_WARMUP_ROUNDS,
        "max_abs_log_ratio": MAX_ABS_LOG_RATIO,
        "max_abs_log_ratio_scope": "gate_bounds_only",
        "trial_aggregation": "student_t_lower_bound_of_mean_log_ratio",
        "alpha": ALPHA,
        "alpha_per_look": LOOK_ALPHA,
        "inputs_drawn_on": "task_device",
        "concurrent_timing": "one_trial_task_per_device",
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
    ratio_upper_bound: float
    min_ratio: float
    versus_milliseconds: tuple[float, ...]
    candidate_milliseconds: tuple[float, ...]

    @property
    def speedup_bounds(self) -> tuple[float, float]:
        """The lower and upper bounds of the mean ratio over the unclamped rounds, which a 3x
        kernel must not lose to the clamp, as published: to 3 decimals, floored at 0.001."""
        raw = [math.log(v / c) for v, c in zip(self.versus_milliseconds, self.candidate_milliseconds)]
        lower, upper = _ratio_bounds(raw, alpha=LOOK_ALPHA)
        return max(round(lower, 3), 0.001), max(round(upper, 3), 0.001)

    def is_decided(self, min_speedup: float) -> bool:
        """Whether the bounds settle every verdict the record feeds, each threshold under the
        lower bound or over the upper: the gate, and against the baseline whether the workload
        qualifies (``entry``) and whether dispatch serves its published speedup (``min_speedup``)."""
        bounds = (self.ratio_lower_bound, self.ratio_upper_bound)
        settled = {(*bounds, THRESHOLD_RATIOS[self.threshold])}
        if self.versus == "baseline":
            settled |= {(*bounds, THRESHOLD_RATIOS["entry"]), (*self.speedup_bounds, min_speedup)}
        return all(lower >= t or upper < t for lower, upper, t in settled)


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


def _ratio_bounds(log_ratios: Sequence[float], *, alpha: float) -> tuple[float, float]:
    """One-sided Student-t lower and upper bounds of the mean speedup ratio, each at ``alpha``."""
    return ratio_lower_bound(log_ratios, alpha=alpha), 1 / ratio_lower_bound(
        [-x for x in log_ratios], alpha=alpha
    )


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
            # clamped for the gate's bounds only, against an outlier round
            log_ratios = [min(max(x, -MAX_ABS_LOG_RATIO), MAX_ABS_LOG_RATIO) for x in raw]
            lower, upper = _ratio_bounds(log_ratios, alpha=LOOK_ALPHA)
            records.append(
                TimingRecord(
                    mode=mode,
                    versus=versus,
                    threshold=threshold,
                    n_rounds=len(log_ratios),
                    log_ratio_mean=statistics.fmean(raw),
                    ratio_lower_bound=lower,
                    ratio_upper_bound=upper,
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
    # the torch.distributed group whose ranks time each call together; None: this process alone
    group: Any = None

    def graph_reading(self, call: Callable[[], object]) -> float:
        if self.group is not None:
            # no rank retries alone, its peers would go on to the next call: a tracer fault
            # fails the group's task, which the bench reruns whole
            return self.bench_cudagraph(call, rep=GRAPH_BUDGET_MILLISECONDS, group=self.group)
        try:
            return self.bench_cudagraph(call, rep=GRAPH_BUDGET_MILLISECONDS)
        except Exception as e:
            if not is_tracer_fault(str(e)):
                raise
        return self.bench_cudagraph(call, rep=GRAPH_BUDGET_MILLISECONDS)

    def time_rounds(
        self,
        fns: Mapping[Participant, Callable[[], object]],
        *,
        seed: int,
        thresholds: Mapping[Versus, Threshold],
        min_speedup: float,
    ) -> tuple[RoundTimes, ...]:
        """The recorded rounds, after one unrecorded warm-up profile, up to the first of
        ``ROUND_LOOKS`` at which every timing record of ``thresholds`` is decided."""
        participants = list(fns)
        rounds = []
        is_gc_enabled = gc.isenabled()
        gc.disable()
        try:
            self.graph_reading(fns["baseline"])
            for mode in get_args(Mode):
                for round_index in range(-N_WARMUP_ROUNDS, ROUND_LOOKS[-1]):
                    order_seed = derive_seed(seed, "order", mode, round_index)
                    order = random.Random(order_seed).sample(participants, k=len(participants))
                    milliseconds = {}
                    for p in order:
                        milliseconds[p] = self.graph_reading(fns[p])
                    if round_index >= 0:
                        rounds.append(
                            RoundTimes(
                                mode=mode,
                                round_index=round_index,
                                order=tuple(order),
                                milliseconds=milliseconds,
                            )
                        )
                    if round_index + 1 in ROUND_LOOKS[:-1]:  # the last look ends the rounds anyway
                        looked = rounds
                        if self.group is not None:
                            # every rank decides on the group's rounds, the slowest rank's time of
                            # each (CandidateResult.of_ranks), so that all stop at the same look
                            import torch

                            slowest = torch.tensor(
                                [[r.milliseconds[p] for p in participants] for r in rounds],
                                dtype=torch.float64,
                                device=torch.cuda.current_device(),
                            )
                            torch.distributed.all_reduce(
                                slowest, torch.distributed.ReduceOp.MAX, group=self.group
                            )
                            looked = [
                                r.model_copy(
                                    update={"milliseconds": dict(zip(participants, row, strict=True))}
                                )
                                for r, row in zip(rounds, slowest.tolist(), strict=True)
                            ]
                        times = [ms for r in looked for ms in r.milliseconds.values()]
                        if not all(math.isfinite(ms) and ms > 0 for ms in times):
                            break  # rounds unusable now stay so: compare_performance reports them
                        if all(r.is_decided(min_speedup) for r in timing_records(looked, thresholds)):
                            break
        finally:
            if is_gc_enabled:
                gc.enable()
        return tuple(rounds)


def bind_clock(group: Any = None) -> Clock:
    import torch

    from aklr.core.utils.testing import bench_cudagraph

    if group is not None:
        from aklr.core.validation.stages._group_timing import bench_cudagraph

    return Clock(
        synchronize=torch.cuda.synchronize,
        bench_cudagraph=bench_cudagraph,
        group=group,
    )


_TRACER_FAULT_MESSAGES = (
    "rocprofiler-sdk",
    "graph replays",
    "the kernel tracer failed",
    "calibration could not restore",
)


def is_tracer_fault(text: str) -> bool:
    """A failure of the kernel tracer's reading that another reading, or a fresh process, may not repeat."""
    return any(message in text for message in _TRACER_FAULT_MESSAGES)


def is_tracer_refusal(text: str) -> bool:
    """The kernel tracer cannot bind the call's graph to its nodes: the platform's failure, every reading's."""
    return "cannot attribute the replays" in text
