"""One rank's torch-profiler trace as forward steps and the GPU work each step launched.

Both servers name every forward step with a CPU span while the profiler runs:

- SGLang: ``step[<STAGE> bs=<n> toks=<t> ...]`` from ``build_step_span_name`` (EXTEND, DECODE, VERIFY, DRAFT, MIXED,
  IDLE; a draft model's steps are all DRAFT), and ``step[DRAFT_LOOP ...]`` in a fused draft loop.
- vLLM: ``execute_context_<n>(<t>)_generation_<m>(<u>)`` around ``Worker.execute_model`` (context = requests that
  have not produced a token yet, chunked prefill included; generation = the rest).

A GPU activity (kernel, memcpy, memset) is tied to the runtime call that launched it by ``args.correlation`` and that
call to the step span enclosing it on the same thread. This also holds for a CUDA-graph replay (one runtime call
launches the graph's kernels) and for an overlap scheduler that prepares step i+1 while step i runs on the GPU.
A call outside every span but after a step on its thread and before the next one (vLLM samples and drafts in
``sample_tokens``, after ``execute_model`` returns) belongs to that step as its tail; a call before a thread's first
step is ``(no step)``. Nothing is dropped: every activity lands in a step or in ``(no step)``.
"""

from __future__ import annotations

import bisect
import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

SGLANG_STEP = re.compile(r"^step\[(?P<stage>[A-Z_]+)(?: raw_bs=\d+)? bs=(?P<bs>\d+)(?: toks=(?P<toks>\d+))?")
VLLM_STEP = re.compile(r"^execute_context_(?P<cr>\d+)\((?P<ct>\d+)\)_generation_(?P<gr>\d+)\((?P<gt>\d+)\)")
RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}
GPU_CATS = {"kernel": "kernel", "gpu_memcpy": "memcpy", "gpu_memset": "memset"}


def load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        raw = f.read()
    try:
        import orjson

        data = orjson.loads(raw)
    except ImportError:  # pragma: no cover
        data = json.loads(raw)
    return data["traceEvents"] if isinstance(data, dict) else data


@dataclass
class Step:
    index: int
    name: str
    framework: str
    stage: str  # EXTEND / DECODE / VERIFY / DRAFT / ... for SGLang; CONTEXT / GENERATION / MIXED for vLLM
    phase: str = ""  # prefill / decode / mixed / idle / unknown
    shape: str = ""  # SGLang detailed annotations: context (c_) or generation (g_)
    bs: int = 0
    tokens: int = 0
    context_requests: int = 0
    context_tokens: int = 0
    generation_requests: int = 0
    generation_tokens: int = 0
    cpu_start: float = 0.0
    cpu_end: float = 0.0
    activities: list = field(default_factory=list)  # (kind, name, gpu_ts, dur_us, how) how = in / tail


def parse_step(name: str) -> dict | None:
    m = SGLANG_STEP.match(name)
    if m:
        # Detailed annotations prefix context/extend-shaped aggregates with c_ and single-query ones with g_.
        shape = "context" if " c_" in name else "generation" if " g_" in name else ""
        return {
            "framework": "sglang",
            "stage": m["stage"],
            "bs": int(m["bs"]),
            "tokens": int(m["toks"] or 0),
            "shape": shape,
        }
    m = VLLM_STEP.match(name)
    if m:
        cr, ct, gr, gt = int(m["cr"]), int(m["ct"]), int(m["gr"]), int(m["gt"])
        stage = "MIXED" if cr and gr else "CONTEXT" if cr else "GENERATION" if gr else "EMPTY"
        return {
            "framework": "vllm",
            "stage": stage,
            "bs": cr + gr,
            "tokens": ct + gt,
            "context_requests": cr,
            "context_tokens": ct,
            "generation_requests": gr,
            "generation_tokens": gt,
        }
    return None


SGLANG_PHASE = {
    "EXTEND": "prefill",
    "DECODE": "decode",
    "VERIFY": "decode",
    "TARGET_VERIFY": "decode",
    "DRAFT_LOOP": "decode",
    "MIXED": "mixed",
    "IDLE": "idle",
    "SPLIT_PREFILL": "prefill",
    "DRAFT_EXTEND": "prefill",
    "DLLM_EXTEND": "prefill",
}
VLLM_PHASE = {"CONTEXT": "prefill", "GENERATION": "decode", "MIXED": "mixed", "EMPTY": "idle"}


def assign_phases(steps: list[Step]) -> None:
    """Phase of each step.

    A draft model's step (SGLang ``DRAFT``) is either a draft extend (context-shaped: it runs the draft model over
    tokens the target just processed) or a draft decode (single-query: it proposes tokens). A draft decode is
    decoding. A draft extend has the phase of the target step it follows: after EXTEND it is the draft side of a
    prefill, after VERIFY it is the draft side of a decode round. Without detailed annotations the shape is unknown;
    then only the first DRAFT after an EXTEND is that prefill's draft extend.
    """
    last_target = None
    drafts_since_target = 0
    for s in steps:
        if s.framework == "vllm":
            s.phase = VLLM_PHASE.get(s.stage, "unknown")
            continue
        if s.stage == "DRAFT":
            by_target = {"EXTEND": "prefill", "MIXED": "mixed"}.get(last_target, "decode")
            if s.shape == "generation":
                s.phase = "decode"
            elif s.shape == "context":
                s.phase = by_target
            else:
                s.phase = by_target if drafts_since_target == 0 else "decode"
            drafts_since_target += 1
            continue
        s.phase = SGLANG_PHASE.get(s.stage, "unknown")
        if s.stage != "IDLE":
            last_target = s.stage
            drafts_since_target = 0


@dataclass
class RankTrace:
    path: str
    steps: list[Step]
    no_step: list  # activities launched outside every step
    unmatched: list  # activities whose correlation has no runtime call in the trace
    window_us: float  # first activity start to last activity end
    busy_us: float  # union of activity intervals
    threads_with_steps: int


def analyse(events: list[dict], path: str = "") -> RankTrace:
    spans = defaultdict(list)  # (pid, tid) -> [(ts, end, name)]
    runtime = {}  # correlation -> (pid, tid, ts)
    gpu = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in GPU_CATS:
            gpu.append(e)
        elif cat in RUNTIME_CATS:
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                runtime[corr] = (e.get("pid"), e.get("tid"), e["ts"])
        elif cat in ("user_annotation", "cpu_op", "python_function") or not cat:
            name = e.get("name", "")
            if name.startswith(("step[", "execute_context_")) and parse_step(name):
                spans[(e.get("pid"), e.get("tid"))].append((e["ts"], e["ts"] + e.get("dur", 0), name))
    steps: list[Step] = []
    index = {}  # (thread, i) -> Step
    starts = {}
    order = sorted((s[0], key, i) for key, v in spans.items() for i, s in enumerate(sorted(v)))
    for key in spans:
        spans[key].sort()
        starts[key] = [s[0] for s in spans[key]]
    for _, key, i in order:
        ts, end, name = spans[key][i]
        info = parse_step(name)
        step = Step(index=len(steps), name=name, cpu_start=ts, cpu_end=end, **info)
        steps.append(step)
        index[(key, i)] = step
    assign_phases(steps)
    no_step, unmatched = [], []
    intervals = []
    for e in gpu:
        kind = GPU_CATS[e["cat"]]
        act = (kind, e.get("name", ""), e["ts"], float(e.get("dur", 0)))
        intervals.append((e["ts"], e["ts"] + float(e.get("dur", 0))))
        r = runtime.get((e.get("args") or {}).get("correlation"))
        if r is None:
            unmatched.append(act)
            continue
        key = (r[0], r[1])
        i = bisect.bisect_right(starts.get(key, []), r[2]) - 1
        if i < 0:
            no_step.append(act)
            continue
        ts, end, _ = spans[key][i]
        index[(key, i)].activities.append((*act, "in" if r[2] <= end else "tail"))
    intervals.sort()
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    window = (max(e for _, e in intervals) - intervals[0][0]) if intervals else 0.0
    return RankTrace(
        path=path,
        steps=steps,
        no_step=no_step,
        unmatched=unmatched,
        window_us=window,
        busy_us=busy,
        threads_with_steps=len(spans),
    )
