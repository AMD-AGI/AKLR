"""Does this image's torch profiler record the kernels a CUDA graph replays?

Decode runs in CUDA graphs. Some torch/ROCm builds record the graph launch but none of the kernels it replays, so a
decode step looks empty (torch 2.9.1 + ROCm 7.2 recorded none). This captures a graph of 10 matmuls, replays it 5
times under the profiler and counts the matmul kernels: 50 means the image's profiler can be trusted for decode.
Also checks that a kernel is tied to its launch by ``correlation``, which ``trace.analyse`` relies on.

    python3 -m op_census probe --out <dir>     # inside the image, on one GPU
"""

from __future__ import annotations

import json
import os
from collections import Counter


def main(out: str) -> int:
    import torch

    from .trace import analyse, load_events

    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda", 0)
    a = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
    b = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
    c = torch.empty_like(a)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            torch.matmul(a, b, out=c)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    path = os.path.join(out, "probe.trace.json")
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        for i in range(5):
            with torch.profiler.record_function(f"step[DECODE bs={i + 1}]"):
                g.replay()
        with torch.profiler.record_function("step[EXTEND bs=1 toks=512]"):
            for _ in range(4):
                torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
    prof.export_chrome_trace(path)
    rt = analyse(load_events(path), path)
    per_step = Counter()
    for st in rt.steps:
        per_step[st.stage] += len(st.activities)
    result = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "graph_kernels_recorded": per_step.get("DECODE", 0),
        "graph_kernels_expected": 50,
        "eager_kernels_recorded": per_step.get("EXTEND", 0),
        "eager_kernels_expected": 4,
        "no_step": len(rt.no_step),
        "unmatched": len(rt.unmatched),
    }
    # Eager matmuls may launch a helper kernel besides the GEMM; the graph count is the one that matters.
    result["ok"] = (
        result["graph_kernels_recorded"] == 50 and result["eager_kernels_recorded"] >= 4 and not rt.unmatched
    )
    print(json.dumps(result))
    with open(os.path.join(out, "probe.json"), "w") as f:
        json.dump(result, f, indent=1)
    return 0 if result["ok"] else 1
