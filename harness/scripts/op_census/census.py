"""A point's census: every GPU activity of every captured window, by phase, operator class and kernel.

Times are per rank: a sum over ranks divided by the number of ranks, so a TP point reads as one GPU. Shares are of
summed activity time (two kernels on concurrent streams both count), not of wall time; ``busy_us`` is the wall-time
union per rank, for the share of the window the GPU was busy at all.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable

from .classify import op_class_at
from .trace import RankTrace

NO_STEP = "(no step)"
UNMATCHED = "(no launch record)"


def _pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else 0.0


# A single call this much longer than its kernel's median, and longer than STALL_MIN_US, is the GPU stalled (one Qwen3.8
# call recorded 27 s in a 15 s window while host memory was under pressure), not work: it is listed under "stalls"
# and left out of every time and share.
STALL_FACTOR = 50.0
STALL_MIN_US = 1_000_000.0


def load_bucket(phase: str, bs: int, tokens: int) -> tuple[str, int] | None:
    """The load a step ran at: a decode step's batch size, a prefill or mixed step's token count rounded up to a power
    of two (512 to 16384). An agentic client's load varies, so two runs of a point capture different mixes of these;
    within one bucket the operator mix should not change from run to run."""
    if phase == "decode":
        return f"bs {bs}", bs
    if phase in ("prefill", "mixed"):
        hi = 512
        while hi < tokens and hi < 16384:
            hi *= 2
        return (f"tokens <= {hi}", hi) if tokens <= hi else ("tokens > 16384", 1 << 30)
    return None


HOST_COPY = re.compile(r"DtoH|HtoD|Device -> Host|Host -> Device")


def move_cache_traffic(steps: list, cls_of) -> float:
    """Host-cache traffic launched after a step's span and before a prefill step - the new request's cached prefix
    loaded back to the GPU, pages written back to the host to make room for it - is that prefill's work, not the
    step whose span it follows (a decode step, which it made look 3.8% heavier at Qwen3.8 c80). It moves to the
    prefill step as ``before``; returns the activity time moved."""
    moved_us = 0.0
    for s, nxt in zip(steps, steps[1:]):
        if nxt.phase not in ("prefill", "mixed"):
            continue
        keep, move = [], []
        for a in s.activities:
            kind, name, _ts, _dur, how = a
            traffic = cls_of(kind, name) == "kv_cache" or (kind == "memcpy" and HOST_COPY.search(name))
            (move if how == "tail" and traffic else keep).append(a)
        if move:
            s.activities = keep
            nxt.activities = nxt.activities + [(k, n, ts, d, "before") for k, n, ts, d, _ in move]
            moved_us += sum(a[3] for a in move)
    return moved_us


def _medians(windows) -> dict:
    durs = defaultdict(list)
    for _, ranks in windows:
        for rt in ranks:
            for s in rt.steps:
                for kind, name, _ts, dur, _how in s.activities:
                    durs[(kind, name)].append(dur)
            for kind, name, _ts, dur in (*rt.no_step, *rt.unmatched):
                durs[(kind, name)].append(dur)
    return {k: statistics.median(v) for k, v in durs.items()}


def build(point: dict, windows: Iterable[tuple[str, list[RankTrace]]]) -> dict:
    """``windows``: (window name, its ranks' traces). Returns the census dict ``report`` renders."""
    windows = list(windows)
    median = _medians(windows)
    pid = point.get("id") or ""

    def op_class(kind, name):
        return op_class_at(pid, kind, name)

    stalls = []

    def stalled(kind, name, dur, where):
        if dur > STALL_MIN_US and dur > STALL_FACTOR * median[(kind, name)]:
            stalls.append(
                {"where": where, "kind": kind, "name": name, "us": dur, "median_us": median[(kind, name)]}
            )
            return True
        return False

    kern = defaultdict(
        lambda: {"calls": 0, "us": 0.0, "durs": [], "ctx": Counter()}
    )  # (phase, cls, kind, name)
    by_stage = defaultdict(lambda: {"calls": 0, "us": 0.0})  # (phase, stage, cls)
    stage_steps = Counter()  # (phase, stage) -> steps on rank 0
    by_load = defaultdict(lambda: {"calls": 0, "us": 0.0})  # (phase, bucket, bucket key, cls)
    load_steps = Counter()  # (phase, bucket) -> steps on rank 0
    step_rows = []  # rank 0's steps, in order
    phase_steps = Counter()
    phase_stage = defaultdict(Counter)  # phase -> stage -> steps
    phase_tokens = Counter()
    phase_bs = defaultdict(list)
    window_rows = []
    tail_us = 0.0
    moved_us = 0.0
    nranks_all = 0
    empty_steps = Counter()
    # (rank, phase) -> [activity us, communication us]: a rank that waits less in collectives is the critical path.
    rank_phase = defaultdict(lambda: [0.0, 0.0])
    for wname, ranks in windows:
        nranks = len(ranks)
        nranks_all = max(nranks_all, nranks)
        rows = []
        for r, rt in enumerate(ranks):
            act_us = 0.0
            moved_us += move_cache_traffic(rt.steps, op_class)
            steps_by_phase = Counter()
            last_prefill = None
            for s in rt.steps:
                steps_by_phase[s.phase] += 1
                bucket = load_bucket(s.phase, s.bs, s.tokens)
                # A draft model's extend after a prefill carries no token count: it belongs to that prefill's bucket.
                if s.phase in ("prefill", "mixed"):
                    if s.tokens > 0:
                        last_prefill = bucket
                    elif last_prefill:
                        bucket = last_prefill
                if r == 0:
                    if bucket:
                        load_steps[(s.phase, bucket[0])] += 1
                    phase_steps[s.phase] += 1
                    phase_stage[s.phase][s.stage] += 1
                    phase_tokens[s.phase] += s.tokens
                    phase_bs[s.phase].append(s.bs)
                    stage_steps[(s.phase, s.stage)] += 1
                    step_rows.append(
                        {
                            "window": wname,
                            "stage": s.stage,
                            "phase": s.phase,
                            "bs": s.bs,
                            "tokens": s.tokens,
                            "context_tokens": s.context_tokens,
                            "generation_tokens": s.generation_tokens,
                            "activities": len(s.activities),
                            "activity_us": sum(a[3] for a in s.activities),
                        }
                    )
                had_work = bool(s.activities)
                # A kernel no rule names (often a generic name several kernels share, e.g. FlyDSL's kernel_kernel) is
                # told apart by its neighbours in the step's GPU order: a graph replay records no grid or stack.
                s.activities = [
                    a
                    for a in s.activities
                    if not stalled(a[0], a[1], a[3], f"{wname}/rank{r}/step{s.index} {s.stage}")
                ]
                if not had_work and s.phase in ("decode", "prefill", "mixed"):
                    empty_steps[s.phase] += 1
                acts = sorted(s.activities, key=lambda a: a[2])
                for j, (kind, name, _ts, _dur, _how) in enumerate(acts):
                    if op_class(kind, name) == "other":
                        prev = acts[j - 1][1][:70] if j else "(step start)"
                        nxt = acts[j + 1][1][:70] if j + 1 < len(acts) else "(step end)"
                        kern[(s.phase, "other", kind, name)]["ctx"][f"{prev} -> * -> {nxt}"] += 1
                for kind, name, _ts, dur, how in s.activities:
                    cls = op_class(kind, name)
                    rp = rank_phase[(r, s.phase)]
                    rp[0] += dur
                    if cls == "communication":
                        rp[1] += dur
                    # Work launched after the step's span (vLLM's sample_tokens, a fused draft after verify) is
                    # shown as the stage's tail, so a stage row never hides it.
                    st = by_stage[
                        (s.phase, s.stage + {"tail": "/tail", "before": "/before"}.get(how, ""), cls)
                    ]
                    st["calls"] += 1
                    st["us"] += dur
                    if bucket:
                        ld = by_load[(s.phase, *bucket, cls)]
                        ld["calls"] += 1
                        ld["us"] += dur
                    rec = kern[(s.phase, op_class(kind, name), kind, name)]
                    rec["calls"] += 1
                    rec["us"] += dur
                    rec["durs"].append(dur)
                    act_us += dur
                    if how == "tail":
                        tail_us += dur
            for label, acts in ((NO_STEP, rt.no_step), (UNMATCHED, rt.unmatched)):
                for kind, name, _ts, dur in acts:
                    if stalled(kind, name, dur, f"{wname}/rank{r} {label}"):
                        continue
                    rec = kern[(label, op_class(kind, name), kind, name)]
                    rec["calls"] += 1
                    rec["us"] += dur
                    rec["durs"].append(dur)
                    act_us += dur
            rows.append(
                {
                    "rank": r,
                    "trace": rt.path,
                    "window_us": rt.window_us,
                    "busy_us": rt.busy_us,
                    "activity_us": act_us,
                    "steps": dict(steps_by_phase),
                    "no_step_us": sum(a[3] for a in rt.no_step),
                    "unmatched_us": sum(a[3] for a in rt.unmatched),
                }
            )
        window_rows.append({"name": wname, "ranks": rows})
    n = max(nranks_all, 1)
    phase_us = Counter()
    for (phase, *_), rec in kern.items():
        phase_us[phase] += rec["us"]
    all_us = sum(phase_us.values()) or 1.0
    kernels = []
    ops = defaultdict(lambda: {"calls": 0, "us": 0.0, "kernels": 0})
    for (phase, cls, kind, name), rec in kern.items():
        steps = phase_steps.get(phase) or 0
        kernels.append(
            {
                "phase": phase,
                "op_class": cls,
                "kind": kind,
                "name": name,
                "calls_per_rank": rec["calls"] / n,
                "us_per_rank": rec["us"] / n,
                "calls_per_step": rec["calls"] / n / steps if steps else None,
                "us_per_step": rec["us"] / n / steps if steps else None,
                "p10_us": _pct(rec["durs"], 0.1),
                "p50_us": _pct(rec["durs"], 0.5),
                "p90_us": _pct(rec["durs"], 0.9),
                "share_of_phase": rec["us"] / phase_us[phase] if phase_us[phase] else 0.0,
                "share_of_all": rec["us"] / all_us,
                **({"context": rec["ctx"].most_common(3)} if rec["ctx"] else {}),
            }
        )
        o = ops[(phase, cls)]
        o["calls"] += rec["calls"]
        o["us"] += rec["us"]
        o["kernels"] += 1
    kernels.sort(key=lambda k: (k["phase"], -k["us_per_rank"]))
    op_rows = []
    for (phase, cls), o in ops.items():
        steps = phase_steps.get(phase) or 0
        op_rows.append(
            {
                "phase": phase,
                "op_class": cls,
                "distinct_kernels": o["kernels"],
                "calls_per_rank": o["calls"] / n,
                "us_per_rank": o["us"] / n,
                "calls_per_step": o["calls"] / n / steps if steps else None,
                "us_per_step": o["us"] / n / steps if steps else None,
                "share_of_phase": o["us"] / phase_us[phase] if phase_us[phase] else 0.0,
                "share_of_all": o["us"] / all_us,
            }
        )
    op_rows.sort(key=lambda o: (o["phase"], -o["us_per_rank"]))
    phases = {}
    for phase, us in phase_us.items():
        bs = phase_bs.get(phase) or []
        phases[phase] = {
            "steps": phase_steps.get(phase, 0),
            "stages": dict(phase_stage.get(phase, {})),
            "us_per_rank": us / n,
            "share_of_all": us / all_us,
            "us_per_step": us / n / phase_steps[phase] if phase_steps.get(phase) else None,
            "tokens": phase_tokens.get(phase, 0),
            "bs_median": statistics.median(bs) if bs else None,
            "bs_max": max(bs) if bs else None,
        }
    # Spread of activity time across the ranks of one window (windows differ in load, ranks should not).
    spreads = []
    for w in window_rows:
        acts = [r["activity_us"] for r in w["ranks"]]
        if len(acts) > 1 and min(acts) > 0:
            spreads.append(max(acts) / min(acts) - 1)
    checks = {
        "ranks": n,
        "unmatched_share": phase_us.get(UNMATCHED, 0.0) / all_us,
        "no_step_share": phase_us.get(NO_STEP, 0.0) / all_us,
        "tail_share": tail_us / all_us,
        "cache_traffic_moved_share": moved_us / all_us,
        "rank_activity_spread": max(spreads) if spreads else None,
        "steps_without_gpu_work": dict(empty_steps),
        "stalls": len(stalls),
        "stall_us_per_rank": sum(x["us"] for x in stalls) / n,
        "busy_share": (
            sum(r["busy_us"] for w in window_rows for r in w["ranks"])
            / (sum(r["window_us"] for w in window_rows for r in w["ranks"]) or 1.0)
        ),
    }
    per_rank = defaultdict(list)
    for (r, phase), (us, comm) in sorted(rank_phase.items()):
        per_rank[phase].append(
            {"rank": r, "us": us, "communication_us": comm, "without_communication_us": us - comm}
        )
    stage_rows = []
    for (phase, stage, cls), v in by_stage.items():
        steps = stage_steps.get((phase, stage.split("/")[0])) or 0
        stage_rows.append(
            {
                "phase": phase,
                "stage": stage,
                "op_class": cls,
                "steps": steps,
                "us_per_rank": v["us"] / n,
                "us_per_step": v["us"] / n / steps if steps else None,
                "calls_per_step": v["calls"] / n / steps if steps else None,
                "share_of_phase": v["us"] / phase_us[phase] if phase_us[phase] else 0.0,
            }
        )
    stage_rows.sort(key=lambda o: (o["phase"], o["stage"], -o["us_per_rank"]))
    bucket_us = Counter()
    for (phase, bucket, _key, _cls), v in by_load.items():
        bucket_us[(phase, bucket)] += v["us"]
    load_rows = []
    for (phase, bucket, key, cls), v in by_load.items():
        steps = load_steps.get((phase, bucket)) or 0
        load_rows.append(
            {
                "phase": phase,
                "bucket": bucket,
                "bucket_key": key,
                "op_class": cls,
                "steps": steps,
                "us_per_rank": v["us"] / n,
                "us_per_step": v["us"] / n / steps if steps else None,
                "calls_per_step": v["calls"] / n / steps if steps else None,
                "share_of_bucket": v["us"] / bucket_us[(phase, bucket)]
                if bucket_us[(phase, bucket)]
                else 0.0,
                "bucket_share_of_phase": (
                    bucket_us[(phase, bucket)] / phase_us[phase] if phase_us[phase] else 0.0
                ),
            }
        )
    load_rows.sort(key=lambda o: (o["phase"], o["bucket_key"], -o["us_per_rank"]))
    return {
        "point": point,
        "windows": window_rows,
        "phases": phases,
        "ops": op_rows,
        "ops_by_stage": stage_rows,
        "ops_by_load": load_rows,
        "kernels": kernels,
        "steps": step_rows,
        "per_rank": dict(per_rank),
        "checks": checks,
        "stalls": sorted(stalls, key=lambda x: -x["us"]),
    }
