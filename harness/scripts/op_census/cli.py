"""op_census: every GPU kernel an InferenceX AgentX point really runs, by phase and operator class.

plan    InferenceX at a ref -> plan.json: every single-node SGLang/vLLM AgentX point of a target, bound to the
        exact recipe InferenceX submits (synthetic acceptance included).
show    one point's server command, server environment and client environment.
run     on the GPU host: serve each selected point, replay its AgentX traces, capture profiler windows.
parse   a run directory -> census.json (steps, phases, operator classes, every kernel).
report  census.json files -> Markdown tables.
compare two runs of one point: do their censuses agree (within load buckets when the load mix differs)?
inventory  every point of each deployment side by side, and every kernel any point ran (Markdown, CSV).
probe   in an image: does its torch profiler record the kernels a CUDA graph replays?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _plan(a) -> int:
    from . import inferencex, serving

    sha = inferencex.fetch(a.ref, Path(a.checkout))
    keys = a.keys.split(",") if a.keys else inferencex.config_keys(Path(a.checkout), a.target)
    points = inferencex.bind(Path(a.checkout), keys)
    seen = {}
    for p in points:
        short = p["config_key"].replace(f"-{a.target}", "")
        pid = f"{short}.{p['row']['exp-name']}"
        n = seen.get(pid, 0)
        seen[pid] = n + 1
        p["id"] = pid if not n else f"{pid}.{n}"
        p["gpus"] = p["recipe"]["roles"]["agg"]["gpus"]
        p["repositories"] = serving.repositories(p["recipe"])
    plan = {
        "inferencex": {"repo": inferencex.REPO, "ref": a.ref, "commit": sha},
        "target": a.target,
        "points": points,
    }
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"InferenceX {sha[:12]}: {len(keys)} configs, {len(points)} points -> {a.out}")
    return 0


def _select(plan: dict, pattern: str | None) -> list[dict]:
    pts = plan["points"]
    return [p for p in pts if re.search(pattern, p["id"])] if pattern else pts


def _show(a) -> int:
    from . import serving

    plan = json.loads(Path(a.plan).read_text())
    paths = json.loads(Path(a.paths).read_text()) if a.paths else {}
    for p in _select(plan, a.points):
        local = {r: paths.get(r, f"<{r}>") for r in p["repositories"]}
        print(f"## {p['id']}  ({p['gpus']} GPU, image {p['row']['image']})")
        print("server:", " ".join(serving.server_command(p["recipe"], local, "127.0.0.1", 30000)))
        print("server env:", json.dumps(serving.server_env(p["recipe"])))
        print("acceptance length:", p.get("acceptance_length"))
        print()
    return 0


def _run(a) -> int:
    from . import runner

    plan = json.loads(Path(a.plan).read_text())
    host_cfg = json.loads(Path(a.host).read_text())
    host = runner.Host(
        gpus=[int(g) for g in a.gpus.split(",")],
        out=Path(a.out),
        images=host_cfg["images"],
        paths=host_cfg["paths"],
        infx=Path(host_cfg["infx"]),
        agentx_runtime=Path(host_cfg["agentx_runtime"]),
        hf_home=Path(host_cfg["hf_home"]),
        base_port=host_cfg.get("base_port", 31000),
        deviations=host_cfg.get("deviations", []),
        binds=host_cfg.get("binds", ["/shared_nfs:/shared_nfs:ro"]),
    )
    cap = runner.Capture(
        windows=a.windows,
        window_s=a.window_s,
        gap_s=a.gap_s,
        settle_s=a.settle_s,
        fallback_s=a.fallback_s,
        deadline=a.deadline,
    )
    points = _select(plan, a.points)
    if a.repeat > 1:
        points = [dict(p, id=f"{p['id']}.r{r}") for p in points for r in range(a.repeat)]
    host.out.mkdir(parents=True, exist_ok=True)
    # The profiling phase must outlast the last window; InferenceX's own minimum for a scenario is 900 s.
    duration = a.duration or max(1200, int(cap.settle_s + cap.windows * (cap.window_s + cap.gap_s) + 120))
    for r in runner.run_points(host, cap, points, duration, a.max_parallel):
        print(json.dumps(r))
    return 0


def _analyse_file(path: str):
    from . import trace

    return trace.analyse(trace.load_events(path), path)


def _parse(a) -> int:
    from concurrent.futures import ProcessPoolExecutor

    from . import census

    run = Path(a.run)
    point = json.loads((run / "point.json").read_text())
    layout = []
    for wdir in sorted((run / "traces").glob("w*")):
        files = sorted(str(f) for f in wdir.iterdir() if re.search(r"\.json(\.gz)?$", f.name))
        layout.append((wdir.name, files))
    # One rank's trace of one window per process: a TP8 point has 64 of them, hours in one process.
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        done = dict(
            zip(
                [f for _, fs in layout for f in fs],
                pool.map(_analyse_file, [f for _, fs in layout for f in fs]),
            )
        )
    windows = [(name, [done[f] for f in files]) for name, files in layout]
    meta = {k: point.get(k) for k in ("id", "config_key", "variant", "acceptance_length", "gpus")}
    dev = run / "deviations.json"
    meta["deviations"] = json.loads(dev.read_text()) if dev.exists() else []
    events = (
        [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
        if (run / "events.jsonl").exists()
        else []
    )
    meta["window_load"] = [
        {k: e.get(k) for k in ("event", "window", "load")}
        for e in events
        if e.get("event") in ("window_start", "window_stop")
    ]
    meta.update(
        image=point["row"]["image"],
        model=point["row"]["model"],
        framework=point["row"]["framework"],
        conc=point["row"]["conc"],
        exp_name=point["row"]["exp-name"],
    )
    out = census.build(meta, windows)
    (run / "census.json").write_text(json.dumps(out, indent=1))
    print(f"{run / 'census.json'}: {sum(len(r) for _, r in windows)} rank traces in {len(windows)} windows")
    return 0


def _report(a) -> int:
    from . import report

    censuses = [json.loads(Path(f).read_text()) for f in a.census]
    if a.summary:
        print(report.summary(censuses))
        return 0
    for c in censuses:
        print(report.markdown(c, top=a.top))
    return 0


def _compare(a) -> int:
    from . import compare

    r = compare.compare(
        json.loads(Path(a.a).read_text()),
        json.loads(Path(a.b).read_text()),
        class_tol=a.class_tol,
        kernel_share=a.kernel_share,
    )
    print(compare.markdown(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    # 0 agree, 2 the load mix differs only, 3 too few steps to judge that, 1 disagree.
    return {"agree": 0, "load_mix": 2, "few_steps": 3}.get(r["verdict"], 1)


def _inventory(a) -> int:
    from . import inventory

    censuses = [json.loads(Path(f).read_text()) for f in a.census]
    md = inventory.markdown(censuses, top_kernels=a.top)
    if a.csv:
        Path(a.csv).write_text(inventory.to_csv(censuses))
    if a.out:
        Path(a.out).write_text(md)
    else:
        print(md)
    return 0


def _probe(a) -> int:
    from . import probe

    return probe.main(a.out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="op_census", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--ref", default="main")
    p.add_argument("--checkout", default="/tmp/op_census/inferencex")
    p.add_argument("--target", default="mi355x")
    p.add_argument("--keys", help="comma-separated config keys (default: every SGLang/vLLM AgentX config)")
    p.add_argument("--out", default="plan.json")
    p.set_defaults(fn=_plan)
    s = sub.add_parser("show")
    s.add_argument("--plan", default="plan.json")
    s.add_argument("--paths", help="JSON: repository id -> local directory")
    s.add_argument("--points", help="regex on point ids")
    s.set_defaults(fn=_show)
    r = sub.add_parser("run")
    r.add_argument("--plan", default="plan.json")
    r.add_argument("--host", required=True, help="JSON: images, paths, infx, agentx_runtime, hf_home")
    r.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    r.add_argument("--points", help="regex on point ids")
    r.add_argument("--out", required=True)
    r.add_argument("--windows", type=int, default=8)
    r.add_argument("--window-s", type=float, default=15.0)
    r.add_argument("--gap-s", type=float, default=105.0)
    r.add_argument("--settle-s", type=float, default=120.0)
    r.add_argument(
        "--fallback-s",
        type=float,
        default=7200.0,
        help="start the windows this long after the client if it never logs its profiling phase (a high-concurrency warmup can take over 30 min)",
    )
    r.add_argument("--duration", type=int, help="client DURATION (default: just past the last window)")
    r.add_argument("--repeat", type=int, default=1, help="run each point this many times (separate servers)")
    r.add_argument("--deadline", type=float, default=0.0, help="unix time: stop whatever still runs then")
    r.add_argument("--max-parallel", type=int, default=4, help="points running at once")
    r.set_defaults(fn=_run)
    q = sub.add_parser("parse")
    q.add_argument("run")
    q.add_argument("--jobs", type=int, default=16, help="trace files analysed in parallel")
    q.set_defaults(fn=_parse)
    t = sub.add_parser("report")
    t.add_argument("census", nargs="+")
    t.add_argument("--top", type=int, default=30)
    t.add_argument("--summary", action="store_true", help="one row per point instead of each point's tables")
    t.set_defaults(fn=_report)
    c = sub.add_parser(
        "compare",
        help="two runs of one point: do their censuses agree (exit 0; 2 load mix only, "
        "3 too few steps to tell, 1 disagree)",
    )
    c.add_argument("a", help="census.json of the first run")
    c.add_argument("b", help="census.json of the second run")
    c.add_argument(
        "--class-tol", type=float, default=0.03, help="largest difference of a class's share of a phase"
    )
    c.add_argument(
        "--kernel-share",
        type=float,
        default=0.005,
        help="share of a phase above which a kernel must be in both",
    )
    c.add_argument("--json", help="also write the comparison here")
    c.set_defaults(fn=_compare)
    v = sub.add_parser(
        "inventory",
        help="every point of each deployment side by side: classes by concurrency, "
        "every kernel any point ran",
    )
    v.add_argument("census", nargs="+")
    v.add_argument("--top", type=int, default=40, help="kernels per phase in the Markdown (the CSV has all)")
    v.add_argument("--out", help="write the Markdown here instead of printing it")
    v.add_argument("--csv", help="also write every (point, phase, kernel) row here")
    v.set_defaults(fn=_inventory)
    b = sub.add_parser("probe")
    b.add_argument("--out", default="probe")
    b.set_defaults(fn=_probe)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
