"""Two censuses of one point, each from its own server run: do they agree?

A census is a sample (a few windows of one run), so a single run is not a result until a second run of the same
point gives the same picture. Agreement, per phase both runs have:
  - each operator class's share of the phase within ``class_tol`` (absolute, default 3 points);
  - the same kernels above ``kernel_share`` of the phase: a kernel at or above it in one run must be at or above it
    in the other, unless both runs are near it (under twice the threshold in one, at least half of it in the
    other): such a kernel may sit on either side and is only listed.
Phase shares and decode per-step time are shown, not judged: they follow the load, which an agentic client varies
from run to run.

The load mix varies too: one run of a concurrency-4 point decoded 44% of its steps at batch size 1, the other 57%,
and a batch-1 step runs a fused decoder-layer kernel a larger batch does not; one caught three 16384-token prefill
chunks, the other none. So a phase that disagrees is compared again within each load bucket (a decode batch size,
a prefill token range) that both runs sampled enough: if every such bucket agrees and they hold most of the phase's
time, the runs differ in load mix only (``load_mix``); if too little of the phase can be judged, ``few_steps``.
"""

from __future__ import annotations

PHASES = ("prefill", "decode", "mixed")
MIN_STEPS = {"decode": 200, "prefill": 5, "mixed": 5}  # rank-0 steps a bucket needs in both runs to be judged
MIN_COVER = 0.5  # judged buckets must hold this much of the phase's time in both runs


def _by(rows: list[dict], phase: str, key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in rows:
        if r["phase"] == phase:
            out[r[key]] = out.get(r[key], 0.0) + r["share_of_phase"]
    return out


def _buckets(c: dict, phase: str) -> dict:
    out = {}
    for r in c.get("ops_by_load") or []:
        if r["phase"] == phase:
            b = out.setdefault(
                r["bucket"],
                {
                    "key": r["bucket_key"],
                    "steps": r["steps"],
                    "share": r["bucket_share_of_phase"],
                    "classes": {},
                },
            )
            b["classes"][r["op_class"]] = b["classes"].get(r["op_class"], 0.0) + r["share_of_bucket"]
    return out


def by_load(a: dict, b: dict, phase: str, class_tol: float) -> dict:
    """The phase compared within each load bucket both runs sampled enough; ``status``: load_mix or few_steps when
    every judged bucket agrees, disagree when one does not."""
    ba, bb = _buckets(a, phase), _buckets(b, phase)
    rows, judged = [], []
    for name in sorted(set(ba) | set(bb), key=lambda n: (ba.get(n) or bb.get(n))["key"]):
        x, y = ba.get(name), bb.get(name)
        row = {
            "bucket": name,
            "steps": [x and x["steps"], y and y["steps"]],
            "share_of_phase": [x["share"] if x else 0.0, y["share"] if y else 0.0],
        }
        if x and y and min(x["steps"], y["steps"]) >= MIN_STEPS.get(phase, 5):
            diffs = {
                c: y["classes"].get(c, 0.0) - x["classes"].get(c, 0.0)
                for c in set(x["classes"]) | set(y["classes"])
            }
            worst = max(diffs, key=lambda c: abs(diffs[c]))
            row.update(
                judged=True, worst_class=worst, worst_diff=diffs[worst], ok=abs(diffs[worst]) <= class_tol
            )
            judged.append(row)
        rows.append(row)
    cover = [sum(r["share_of_phase"][i] for r in judged) for i in (0, 1)]
    if any(not r["ok"] for r in judged):
        status = "disagree"
    elif judged and min(cover) >= MIN_COVER:
        status = "load_mix"
    else:
        status = "few_steps"
    return {"status": status, "judged_cover": cover, "buckets": rows}


def compare(a: dict, b: dict, class_tol: float = 0.03, kernel_share: float = 0.005) -> dict:
    """Per phase: class share differences, kernels above ``kernel_share`` in one run only, and a verdict.

    ``agree`` is True when every phase agrees as a whole; ``verdict`` is agree, load_mix (each phase that disagrees
    agrees within its load buckets), few_steps (some phase cannot be judged that way) or disagree."""
    empty = [k for k, c in (("a", a), ("b", b)) if not c.get("windows") or not c.get("phases")]
    if empty:
        # A run that captured nothing (it failed, or its traces are missing) cannot vouch for the other.
        return {
            "a": a["point"].get("id"),
            "b": b["point"].get("id"),
            "class_tol": class_tol,
            "kernel_share": kernel_share,
            "phases": [],
            "agree": False,
            "reason": f"no capture in run {' and '.join(empty)}",
        }
    phases = []
    for phase in PHASES:
        pa, pb = a["phases"].get(phase), b["phases"].get(phase)
        if not pa and not pb:
            continue
        row = {
            "phase": phase,
            "share_of_all": [pa and pa["share_of_all"], pb and pb["share_of_all"]],
            "us_per_step": [pa and pa.get("us_per_step"), pb and pb.get("us_per_step")],
            "steps": [pa and pa["steps"], pb and pb["steps"]],
        }
        if not pa or not pb:
            # A phase seen in one run only: fine if it is a sliver there (a few mixed steps), not if it is real.
            seen = pa or pb
            row["only_in"] = "a" if pa else "b"
            row["agree"] = seen["share_of_all"] < kernel_share
            row["classes"], row["kernels_missing"], row["kernels_near"] = [], [], []
            phases.append(row)
            continue
        ca, cb = _by(a["ops"], phase, "op_class"), _by(b["ops"], phase, "op_class")
        classes = []
        for cls in sorted(set(ca) | set(cb), key=lambda c: -max(ca.get(c, 0.0), cb.get(c, 0.0))):
            d = cb.get(cls, 0.0) - ca.get(cls, 0.0)
            classes.append(
                {
                    "op_class": cls,
                    "a": ca.get(cls, 0.0),
                    "b": cb.get(cls, 0.0),
                    "diff": d,
                    "ok": abs(d) <= class_tol,
                }
            )
        ka, kb = _by(a["kernels"], phase, "name"), _by(b["kernels"], phase, "name")
        missing, near = [], []
        for name in sorted(set(ka) | set(kb)):
            x, y = ka.get(name, 0.0), kb.get(name, 0.0)
            hi, lo = max(x, y), min(x, y)
            if hi < kernel_share:
                continue
            rec = {"name": name, "a": x, "b": y}
            if lo >= kernel_share:
                continue
            (near if lo >= kernel_share / 2 and hi < 2 * kernel_share else missing).append(rec)
        row.update(
            classes=classes,
            kernels_missing=missing,
            kernels_near=near,
            agree=all(c["ok"] for c in classes) and not missing,
        )
        if not row["agree"]:
            row["by_load"] = by_load(a, b, phase, class_tol)
        phases.append(row)
    status = [
        "agree" if p["agree"] else p["by_load"]["status"] if "by_load" in p else "disagree" for p in phases
    ]
    verdict = (
        "agree"
        if all(s == "agree" for s in status)
        else "disagree"
        if "disagree" in status
        else "few_steps"
        if "few_steps" in status
        else "load_mix"
    )
    return {
        "a": a["point"].get("id"),
        "b": b["point"].get("id"),
        "class_tol": class_tol,
        "kernel_share": kernel_share,
        "phases": phases,
        "agree": verdict == "agree",
        "verdict": verdict,
    }


def _pct(x) -> str:
    return "" if x is None else f"{100 * x:.1f}%"


def markdown(r: dict) -> str:
    head = {
        "agree": "agree",
        "load_mix": "load mix differs, agree within load buckets",
        "few_steps": "DISAGREE as a whole, too few steps to judge by load",
        "disagree": "DISAGREE",
    }
    out = [
        f"# {r['a']} vs {r['b']}: {head[r.get('verdict', 'agree' if r['agree'] else 'disagree')]}",
        "",
        f"Class shares within {100 * r['class_tol']:.0f} points; kernels at {100 * r['kernel_share']:.1f}% of a "
        "phase or more in both runs, or near that in both.",
        "",
    ]
    if r.get("reason"):
        out += [r["reason"], ""]
    for p in r["phases"]:
        us = ["" if u is None else f"{u:.0f} us" for u in p["us_per_step"]]
        out.append(
            f"## {p['phase']}: {'agree' if p['agree'] else 'DISAGREE'} (share of all {_pct(p['share_of_all'][0])}"
            f" / {_pct(p['share_of_all'][1])}; per step {us[0]} / {us[1]}; steps {p['steps'][0]} / "
            f"{p['steps'][1]})"
        )
        if "only_in" in p:
            out.append(f"\nonly in run {p['only_in']}\n")
            continue
        out += ["", "| class | a | b | diff |", "|---|---:|---:|---:|"]
        out += [
            f"| {c['op_class']} | {_pct(c['a'])} | {_pct(c['b'])} | {100 * c['diff']:+.1f}{'' if c['ok'] else ' ✗'} |"
            for c in p["classes"]
        ]
        for title, rows in (
            ("above the threshold in one run only", p["kernels_missing"]),
            ("near the threshold", p["kernels_near"]),
        ):
            if rows:
                out += ["", f"Kernels {title}:", ""]
                out += [f"- `{k['name'][:140]}`: {_pct(k['a'])} / {_pct(k['b'])}" for k in rows]
        bl = p.get("by_load")
        if bl:
            out += [
                "",
                f"By load ({bl['status']}; judged buckets hold {_pct(bl['judged_cover'][0])} / "
                f"{_pct(bl['judged_cover'][1])} of the phase):",
                "",
                "| bucket | steps | share of phase | largest class difference |",
                "|---|---:|---:|---|",
            ]
            for b in bl["buckets"]:
                d = (
                    f"{b['worst_class']} {100 * b['worst_diff']:+.1f}{'' if b['ok'] else ' ✗'}"
                    if b.get("judged")
                    else "not judged (too few steps)"
                )
                out.append(
                    f"| {b['bucket']} | {b['steps'][0] or 0} / {b['steps'][1] or 0} | "
                    f"{_pct(b['share_of_phase'][0])} / {_pct(b['share_of_phase'][1])} | {d} |"
                )
        out.append("")
    return "\n".join(out) + "\n"
