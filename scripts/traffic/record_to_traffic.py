"""record_to_traffic.py - A recorded serving run as an AKLR recording (traffic/<name>.traffic.jsonl.xz).

    PYTHONPATH=. python scripts/traffic/record_to_traffic.py <run dir> <name> --page-size 64 --table-pages 16386

<run dir>: a run directory with the server's launch command, its log, and each step's requests.
Its steps inside the profiling window, health checks dropped, every decode and prefill step in order: a
decode (a verify of q draft tokens) attends its prefix + q + 1 keys, a prefill its prefix + q. The header
takes the model and draft tokens from the launch command and the KV pool from the server log; the page
size and the page table's width (pages, as the attention backend passes it in CUDA-graph replay) the run
does not record, so they are given.
"""

import argparse
import gzip
import json
import re
import shlex
from pathlib import Path

from aklr.core.validation import traffic

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("name")
    parser.add_argument("--page-size", type=int, required=True)
    parser.add_argument("--table-pages", type=int, required=True)
    args = parser.parse_args()
    run = args.run

    launch = shlex.split(json.loads((run / "trace" / "server_args.json").read_text())["_launch_command"])
    # a flag's value is the word after it, unless that is another flag
    flags = {
        w: None if v.startswith("--") else v
        for w, v in zip(launch, [*launch[1:], "--"], strict=True)
        if w.startswith("--")
    }
    log = (run / "server.log").read_text()
    header = {
        "model": flags["--model-path"],
        "draft_tokens": int(flags.get("--speculative-num-draft-tokens", 1)),
        "pool_tokens": int(re.search(r"max_total_num_tokens=(\d+)", log)[1]),
        "page_size": args.page_size,
        "table_pages": args.table_pages,
    }

    lo, hi = json.loads((run / "run.json").read_text())["profiling_window_ns"]
    path = run / "trace" / "steps.jsonl"
    steps = []
    with gzip.open(f"{path}.gz", "rt") if Path(f"{path}.gz").exists() else path.open() as lines:
        for line in lines:
            s = json.loads(line)
            if not lo <= s["t_ns"] <= hi or s["mode"] not in ("decode", "extend"):
                continue
            if any(r.startswith("HEALTH_CHECK_") for r in s["rid"]):
                continue
            decode = s["mode"] == "decode"
            seqs = tuple((q, p + q + decode) for q, p in zip(s["q"], s["prefix"], strict=True))
            steps.append(("decode" if decode else "prefill", seqs))

    out = ROOT / "traffic" / f"{args.name}.traffic.jsonl.xz"
    traffic.write(out, header, steps)
    assert traffic.read(ROOT, args.name) == (header, tuple(steps))
    n = sum(p == "decode" for p, _ in steps)
    print(f"{out}: {len(steps)} steps ({n} decode), {out.stat().st_size // 1024} KB; {header}")


if __name__ == "__main__":
    main()
