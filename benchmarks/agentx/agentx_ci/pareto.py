# SPDX-License-Identifier: MIT
"""Pareto frontiers and the week-over-week comparison.

Axes: x = P90 E2E-normalized interactivity (tok/s/user), y = throughput per GPU (tok/s/GPU);
both higher is better, so the frontier is the upper-right hull of the measured points.

A baseline arm is comparable only when its ``arm_serve_sha256`` equals this run's: any change
to any point's serving configuration starts a new segment for the whole arm, and the
per-point hashes only say which points changed.
"""

from __future__ import annotations

import math

X, Y = "p90_e2e_norm_intvty", "tput_per_gpu"


def measured(results: dict, arm: str | None = None) -> list[dict]:
    return [
        p
        for p in results["points"]
        if p["status"] == "ok" and (arm is None or p["arm"] == arm)
    ]


def frontier(points: list[dict]) -> list[dict]:
    """Points no other point beats on both axes, ordered by rising interactivity."""
    best_y, keep = -math.inf, []
    for point in sorted(points, key=lambda p: (-p[X], -p[Y])):
        if point[Y] > best_y:
            keep.append(point)
            best_y = point[Y]
    return keep[::-1]


def frontiers(results: dict) -> dict:
    arms = sorted({p["arm"] for p in results["points"]})
    return {
        "arms": {arm: frontier(measured(results, arm)) for arm in arms},
        "combined": frontier(measured(results)),
    }


def ratio(new: float, old: float) -> float | None:
    return new / old if old else None


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def compare_arm(arm: str, current: dict, baseline: dict) -> dict:
    now = current["serving"]["arms"][arm]["arm_serve_sha256"]
    before_arm = baseline["serving"]["arms"].get(arm)
    if before_arm is None:
        return {
            "status": "new-arm",
            "reason": "arm absent from the baseline",
            "points": [],
        }
    before = before_arm["arm_serve_sha256"]
    base_points = {p["conc"]: p for p in baseline["points"] if p["arm"] == arm}
    if now != before:
        changed = sorted(
            p["conc"]
            for p in current["points"]
            if p["arm"] == arm
            and base_points.get(p["conc"], {}).get("serve_spec_sha256")
            != p["serve_spec_sha256"]
        )
        return {
            "status": "new-segment",
            "reason": f"serving config changed (arm {before[:12]} → {now[:12]})",
            "changed_concs": changed,
            "points": [],
        }
    rows, tput_ratios = [], []
    for point in sorted(
        (p for p in current["points"] if p["arm"] == arm), key=lambda p: p["conc"]
    ):
        old = base_points.get(point["conc"])
        row = {"conc": point["conc"], "current": point, "baseline": old}
        if point["status"] == "ok" and old and old["status"] == "ok":
            row["tput_ratio"] = ratio(point[Y], old[Y])
            row["intvty_ratio"] = ratio(point[X], old[X])
            if row["tput_ratio"]:
                tput_ratios.append(row["tput_ratio"])
        rows.append(row)
    return {
        "status": "comparable",
        "points": rows,
        "tput_geomean_ratio": geomean(tput_ratios),
    }


def compare(current: dict, baseline: dict | None) -> dict:
    """Week-over-week view of ``current`` against ``baseline`` (the previous published run)."""
    if baseline is None:
        return {
            "status": "no-baseline",
            "reason": "no previous run",
            "arms": {},
            "notes": [],
        }
    if baseline["run"]["mode"] != current["run"]["mode"]:
        reason = f"previous run is {baseline['run']['mode']}, this one is {current['run']['mode']}"
        return {"status": "no-baseline", "reason": reason, "arms": {}, "notes": []}
    notes = []
    if baseline["run"]["image"]["run"] != current["run"]["image"]["run"]:
        notes.append("run image changed")
    if baseline["run"]["inferencex"]["sha"] != current["run"]["inferencex"]["sha"]:
        notes.append(
            "InferenceX moved "
            f"{baseline['run']['inferencex']['sha'][:10]} → {current['run']['inferencex']['sha'][:10]}"
        )
    if baseline["serving"]["grid_sha256"] != current["serving"]["grid_sha256"]:
        notes.append("concurrency grid changed")
    arms = {
        arm: compare_arm(arm, current, baseline) for arm in current["serving"]["arms"]
    }
    return {
        "status": "ok",
        "baseline_run": baseline["run"]["id"],
        "baseline_aklr_sha": baseline["run"]["aklr_sha"],
        "arms": arms,
        "notes": notes,
    }
