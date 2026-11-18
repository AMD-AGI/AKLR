"""A census as Markdown: what was captured, the checks, then per phase the operator classes and every kernel."""

from __future__ import annotations

PHASE_ORDER = ("prefill", "decode", "mixed", "idle", "unknown", "(no step)", "(no launch record)")


def summary(censuses: list[dict], top: int = 5) -> str:
    """One row per point: phase shares, the top operator classes of prefill and decode, and the checks."""
    out = [
        "| point | conc | prefill | decode | mixed | prefill: top classes | decode: top classes | other | checks |",
        "|---|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for c in sorted(censuses, key=lambda c: (c["point"].get("config_key", ""), c["point"].get("conc", 0))):
        p, ph, ch = c["point"], c["phases"], c["checks"]

        def tops(phase, c=c):
            ops = [o for o in c["ops"] if o["phase"] == phase][:top]
            return ", ".join(f"{o['op_class']} {100 * o['share_of_phase']:.0f}%" for o in ops)

        other = sum(o["share_of_all"] for o in c["ops"] if o["op_class"] == "other")
        flags = []
        if ch["no_step_share"] > 0.005 or ch["unmatched_share"] > 0.005:
            flags.append(f"unattributed {100 * (ch['no_step_share'] + ch['unmatched_share']):.1f}%")
        if ch.get("stalls"):
            flags.append(f"stalls {ch['stalls']} ({ch['stall_us_per_rank'] / 1e3:.0f} ms/rank left out)")
        if ch["steps_without_gpu_work"]:
            flags.append(f"empty steps {ch['steps_without_gpu_work']}")
        if ch["rank_activity_spread"] is not None and ch["rank_activity_spread"] > 0.15:
            flags.append(f"rank spread {100 * ch['rank_activity_spread']:.0f}%")
        if p.get("deviations"):
            flags.append("deviation: " + ", ".join(k for d in p["deviations"] for k in d.get("env", {})))
        share = {k: f"{100 * v['share_of_all']:.1f}%" for k, v in ph.items()}
        out.append(
            f"| {p.get('id')} | {p.get('conc')} | {share.get('prefill', '')} | {share.get('decode', '')} | "
            f"{share.get('mixed', '')} | {tops('prefill')} | {tops('decode')} | {100 * other:.1f}% | "
            f"{'; '.join(flags) or 'ok'} |"
        )
    return "\n".join(out) + "\n"


def _f(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def markdown(c: dict, top: int = 30) -> str:
    p = c["point"]
    out = [
        f"# {p.get('id')}",
        "",
        f"{p['model']} on {p['framework']}, concurrency {p['conc']}, {p.get('gpus')} GPU; image `{p['image']}`; "
        f"recipe variant `{p.get('variant')}`; synthetic acceptance length {p.get('acceptance_length')}.",
        "",
    ]
    for d in p.get("deviations") or []:
        out += [f"**Deviation from the cookbook**: {d.get('env')} — {d.get('reason')}", ""]
    loads = [w for w in p.get("window_load") or [] if w.get("load")]
    if loads:
        out += [
            "Server load at the windows' edges (running / queued requests): "
            + "; ".join(f"w{w['window']} {w['event'].split('_')[1]} {w['load']}" for w in loads),
            "",
        ]
    ch = c["checks"]
    out += [
        "## Capture and checks",
        "",
        f"- windows: {len(c['windows'])}; ranks per window: {ch['ranks']}; "
        f"GPU busy {100 * ch['busy_share']:.1f}% of the captured wall time",
        f"- activity outside every step: {100 * ch['no_step_share']:.2f}%; without a launch record: "
        f"{100 * ch['unmatched_share']:.2f}%; launched after a step's span (its tail): {100 * ch['tail_share']:.2f}%; "
        f"host-cache traffic before a prefill moved to it: {100 * ch.get('cache_traffic_moved_share', 0):.2f}%",
        "- rank spread of activity time within a window: "
        + (
            f"{100 * ch['rank_activity_spread']:.1f}%"
            if ch["rank_activity_spread"] is not None
            else "one rank"
        ),
        f"- steps with no GPU work recorded: {ch['steps_without_gpu_work'] or 'none'}",
        f"- GPU stalls left out (a call > 50x its kernel's median and > 1 s): {ch.get('stalls', 0)}"
        + (
            f", {ch['stall_us_per_rank'] / 1e3:.0f} ms per rank: "
            + "; ".join(
                f"{x['name'][:40]} {x['us'] / 1e3:.0f} ms ({x['where']})" for x in c.get("stalls", [])[:5]
            )
            if ch.get("stalls")
            else ""
        ),
        "",
    ]
    out += [
        "## Phases",
        "",
        "| phase | steps | stages | median bs | tokens | ms per rank | ms per step | share |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    phases = sorted(
        c["phases"].items(), key=lambda kv: PHASE_ORDER.index(kv[0]) if kv[0] in PHASE_ORDER else 99
    )
    for ph, v in phases:
        st = ", ".join(f"{k} {n}" for k, n in sorted(v["stages"].items()))
        out.append(
            f"| {ph} | {v['steps']} | {st} | {_f(v['bs_median'], 0)} | {v['tokens']} | "
            f"{v['us_per_rank'] / 1e3:.2f} | {_f(v['us_per_step'] / 1e3 if v['us_per_step'] else None, 3)} | "
            f"{100 * v['share_of_all']:.1f}% |"
        )
    for ph, _ in phases:
        ops = [o for o in c["ops"] if o["phase"] == ph]
        out += [
            "",
            f"## {ph}: operator classes",
            "",
            "| class | kernels | calls/step | us/step | ms per rank | share of phase | share of all |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for o in ops:
            out.append(
                f"| {o['op_class']} | {o['distinct_kernels']} | {_f(o['calls_per_step'])} | "
                f"{_f(o['us_per_step'])} | {o['us_per_rank'] / 1e3:.2f} | {100 * o['share_of_phase']:.2f}% | "
                f"{100 * o['share_of_all']:.2f}% |"
            )
        st = [o for o in c.get("ops_by_stage", []) if o["phase"] == ph]
        if st:
            out += [
                "",
                f"### {ph}: by stage (`/tail`: launched after the step's span; `/before`: host-cache traffic for this prefill, launched after the previous step's span)",
                "",
                "| stage | class | steps | calls/step | us/step | share of phase |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for o in st:
                out.append(
                    f"| {o['stage']} | {o['op_class']} | {o['steps']} | {_f(o['calls_per_step'])} | "
                    f"{_f(o['us_per_step'])} | {100 * o['share_of_phase']:.2f}% |"
                )
        ld = [o for o in c.get("ops_by_load", []) if o["phase"] == ph]
        if ld:
            out += [
                "",
                f"### {ph}: by load (decode batch size, prefill tokens rounded up to a power of two)",
                "",
                "| load | steps | us/step | share of phase | top classes (share of the bucket) |",
                "|---|---:|---:|---:|---|",
            ]
            buckets = {}
            for o in ld:
                buckets.setdefault((o["bucket_key"], o["bucket"]), []).append(o)
            for (_, name), rows in sorted(buckets.items()):
                us = sum(o["us_per_rank"] for o in rows)
                steps = rows[0]["steps"]
                tops = ", ".join(
                    f"{o['op_class']} {100 * o['share_of_bucket']:.0f}%"
                    for o in sorted(rows, key=lambda o: -o["us_per_rank"])[:4]
                )
                out.append(
                    f"| {name} | {steps} | {_f(us / steps if steps else None, 0)} | "
                    f"{100 * rows[0]['bucket_share_of_phase']:.1f}% | {tops} |"
                )
        ks = [k for k in c["kernels"] if k["phase"] == ph]
        out += [
            "",
            f"### {ph}: kernels ({len(ks)} distinct; top {min(top, len(ks))} by time, all in census.json)",
            "",
            "| share | class | calls/step | us/step | p50 us | kernel |",
            "|---:|---|---:|---:|---:|---|",
        ]
        for k in ks[:top]:
            ctx = (
                ""
                if not k.get("context")
                else " (between: " + "; ".join(f"{c} ×{n}" for c, n in k["context"][:2]) + ")"
            )
            out.append(
                f"| {100 * k['share_of_phase']:.2f}% | {k['op_class']} | {_f(k['calls_per_step'])} | "
                f"{_f(k['us_per_step'])} | {_f(k['p50_us'])} | `{k['name'][:150]}`{ctx} |"
            )
    return "\n".join(out) + "\n"
