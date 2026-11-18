"""Every point of one deployment (config) side by side: operator classes by concurrency, and every kernel any point ran.

A census is one point; choosing what to optimize needs the deployment's whole curve - which classes grow with
concurrency, which kernels only appear at some loads - and a list in which no kernel that ran is missing.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict

PHASES = ("prefill", "decode", "mixed")


def _deployments(censuses: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for c in censuses:
        out[c["point"].get("config_key") or c["point"].get("id", "").split(".")[0]].append(c)
    for cs in out.values():
        cs.sort(key=lambda c: (c["point"].get("conc") or 0, c["point"].get("id", "")))
    return dict(out)


def _label(c: dict) -> str:
    """A column: the concurrency, plus what else tells two points of one config apart (TP, host KV offload)."""
    exp = c["point"].get("exp_name") or c["point"].get("id", "")
    tp = next((p for p in exp.split("_") if p.startswith("tp")), "")
    kv = "kvdram" if "kvdram" in exp else ""
    return "c" + str(c["point"].get("conc")) + "".join(f" {x}" for x in (tp, kv) if x)


def class_table(cs: list[dict], phase: str) -> tuple[list[str], list[tuple[str, list[float | None]]]]:
    """Columns (points) and rows (class, its share of ``phase`` at each point; None where the phase is absent)."""
    cols = [_label(c) for c in cs]
    shares = defaultdict(lambda: [None] * len(cs))
    for i, c in enumerate(cs):
        for o in c["ops"]:
            if o["phase"] == phase:
                shares[o["op_class"]][i] = o["share_of_phase"]
    for i, c in enumerate(cs):  # a class absent from a point that has the phase ran 0% of it there
        if phase in c["phases"]:
            for row in shares.values():
                if row[i] is None:
                    row[i] = 0.0
    rows = sorted(shares.items(), key=lambda kv: -max(x or 0.0 for x in kv[1]))
    return cols, rows


def kernel_rows(cs: list[dict]) -> list[dict]:
    """Every kernel any point ran (each phase it ran in): class, and per point its share of the phase and its
    calls and time per step of that phase."""
    rows = {}
    for i, c in enumerate(cs):
        for k in c["kernels"]:
            if k["phase"] not in PHASES:
                continue
            r = rows.setdefault(
                (k["phase"], k["name"]),
                {
                    "phase": k["phase"],
                    "op_class": k["op_class"],
                    "name": k["name"],
                    "points": [None] * len(cs),
                },
            )
            r["points"][i] = {
                "share_of_phase": k["share_of_phase"],
                "share_of_all": k["share_of_all"],
                "calls_per_step": k["calls_per_step"],
                "us_per_step": k["us_per_step"],
                "p50_us": k["p50_us"],
            }
    out = list(rows.values())
    for r in out:
        r["max_share_of_phase"] = max(p["share_of_phase"] for p in r["points"] if p)
        r["points_seen"] = sum(1 for p in r["points"] if p)
    out.sort(key=lambda r: (PHASES.index(r["phase"]), -r["max_share_of_phase"]))
    return out


def _pct(x) -> str:
    return "" if x is None else f"{100 * x:.1f}%"


def markdown(censuses: list[dict], top_kernels: int = 40) -> str:
    out = []
    for key, cs in sorted(_deployments(censuses).items()):
        p = cs[0]["point"]
        out += [
            f"# {key}",
            "",
            f"{p.get('model')} on {p.get('framework')}; {len(cs)} points; image `{p.get('image')}`.",
            "",
        ]
        cols = [_label(c) for c in cs]
        out += [
            "## Phase shares of GPU time",
            "",
            "| phase | " + " | ".join(cols) + " |",
            "|---|" + "---:|" * len(cols),
        ]
        for ph in PHASES:
            if any(ph in c["phases"] for c in cs):
                out.append(
                    f"| {ph} | "
                    + " | ".join(_pct(c["phases"].get(ph, {}).get("share_of_all")) for c in cs)
                    + " |"
                )
        for ph in PHASES:
            if not any(ph in c["phases"] for c in cs):
                continue
            cols, rows = class_table(cs, ph)
            out += [
                "",
                f"## {ph}: operator classes (share of the phase)",
                "",
                "| class | " + " | ".join(cols) + " |",
                "|---|" + "---:|" * len(cols),
            ]
            out += [f"| {cls} | " + " | ".join(_pct(x) for x in xs) + " |" for cls, xs in rows]
        ks = kernel_rows(cs)
        out += [
            "",
            f"## Kernels: {len(ks)} (kernel, phase) pairs over the points; the top {top_kernels} of each phase "
            "by their largest share (all of them in the CSV)",
            "",
        ]
        for ph in PHASES:
            sel = [r for r in ks if r["phase"] == ph][:top_kernels]
            if not sel:
                continue
            out += [
                f"### {ph}",
                "",
                "| class | " + " | ".join(cols) + " | kernel |",
                "|---|" + "---:|" * len(cols) + "---|",
            ]
            out += [
                f"| {r['op_class']} | "
                + " | ".join(_pct(x and x["share_of_phase"]) for x in r["points"])
                + f" | `{r['name'][:110]}` |"
                for r in sel
            ]
            out.append("")
    return "\n".join(out) + "\n"


def to_csv(censuses: list[dict]) -> str:
    """One row per (deployment, point, phase, kernel): nothing any point ran is left out."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "config_key",
            "point",
            "conc",
            "phase",
            "op_class",
            "kernel",
            "share_of_phase",
            "share_of_all",
            "calls_per_step",
            "us_per_step",
            "p50_us",
        ]
    )
    for key, cs in sorted(_deployments(censuses).items()):
        for c in cs:
            for k in c["kernels"]:
                w.writerow(
                    [
                        key,
                        c["point"].get("id"),
                        c["point"].get("conc"),
                        k["phase"],
                        k["op_class"],
                        k["name"],
                        f"{k['share_of_phase']:.6f}",
                        f"{k['share_of_all']:.6f}",
                        "" if k["calls_per_step"] is None else f"{k['calls_per_step']:.3f}",
                        "" if k["us_per_step"] is None else f"{k['us_per_step']:.2f}",
                        f"{k['p50_us']:.2f}",
                    ]
                )
    return buf.getvalue()
