# SPDX-License-Identifier: MIT
"""Compute Pareto frontiers and compare runs."""

from __future__ import annotations

import copy
import math

from agentx_ci.preflight import comparison_identity
from agentx_ci.utils import complete_measurement, valid_metrics

X, Y = "p90_e2e_norm_intvty", "tput_per_gpu"


def measured(results: dict, arm: str | None = None) -> list[dict]:
    return [
        p
        for p in results["points"]
        if p["status"] == "ok" and valid_metrics(p) and (arm is None or p["arm"] == arm)
    ]


def frontier(points: list[dict]) -> list[dict]:
    """Return non-dominated points by increasing interactivity."""
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
    current_measurement = current["serving"]["arms"][arm].get("arm_measurement_sha256")
    previous_measurement = before_arm.get("arm_measurement_sha256")
    if now == before and current_measurement != previous_measurement:
        return {
            "status": "new-segment",
            "reason": "measurement settings changed (weights, launch, workload or selected points)",
            "changed_concs": [],
            "points": [],
        }
    base_points = {p["conc"]: p for p in baseline["points"] if p["arm"] == arm}
    if now != before:
        changed = sorted(
            p["conc"]
            for p in current["points"]
            if p["arm"] == arm
            and base_points.get(p["conc"], {}).get("serve_spec_sha256") != p["serve_spec_sha256"]
        )
        return {
            "status": "new-segment",
            "reason": f"serving config changed (arm {before[:12]} → {now[:12]})",
            "changed_concs": changed,
            "points": [],
        }
    rows, tput_ratios = [], []
    for point in sorted((p for p in current["points"] if p["arm"] == arm), key=lambda p: p["conc"]):
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
    """Compare a run with its previous published baseline."""
    if not complete_measurement(current) or (baseline is not None and not complete_measurement(baseline)):
        return {
            "status": "incomplete",
            "reason": "measurement incomplete or points failed",
            "arms": {},
            "notes": [],
        }
    if current["run"].get("source_dirty") is not False or (
        baseline is not None and baseline["run"].get("source_dirty") is not False
    ):
        return {
            "status": "no-baseline",
            "reason": "source is dirty or its state is unknown",
            "arms": {},
            "notes": [],
        }
    if baseline is None:
        return {
            "status": "no-baseline",
            "reason": "no previous run",
            "arms": {},
            "notes": [],
        }
    current_policy = current["run"].get("point_policy", "all")
    baseline_policy = baseline["run"].get("point_policy", "all")
    if current_policy != baseline_policy or (
        current_policy == "inferencex-pareto"
        and current["run"].get("selection", {}).get("selection_sha256")
        != baseline["run"].get("selection", {}).get("selection_sha256")
    ):
        return {
            "status": "no-baseline",
            "reason": "point selection policy or selected recipes changed",
            "arms": {},
            "notes": [],
        }
    for kind in ("model", "target"):
        identity = current.get(kind, {}).get("id")
        if not identity or identity != baseline.get(kind, {}).get("id"):
            return {
                "status": "no-baseline",
                "reason": f"{kind} identity differs or is missing",
                "arms": {},
                "notes": [],
            }
    if baseline["run"]["mode"] != current["run"]["mode"]:
        reason = f"previous run is {baseline['run']['mode']}, this one is {current['run']['mode']}"
        return {"status": "no-baseline", "reason": reason, "arms": {}, "notes": []}
    current_identity, baseline_identity = comparison_identity(current), comparison_identity(baseline)
    if current_identity is None or baseline_identity is None:
        return {
            "status": "no-baseline",
            "reason": "execution identity is missing or unverified",
            "arms": {},
            "notes": [],
        }
    if current_identity != baseline_identity:
        return {
            "status": "no-baseline",
            "reason": "execution image, InferenceX source, dependencies, hardware or traces changed",
            "arms": {},
            "notes": [],
        }
    notes = []
    if baseline["run"]["image"]["run"] != current["run"]["image"]["run"]:
        notes.append("run image changed")
    if baseline["serving"]["grid_sha256"] != current["serving"]["grid_sha256"]:
        notes.append("concurrency grid changed")
    arms = {arm: compare_arm(arm, current, baseline) for arm in current["serving"]["arms"]}
    return {
        "status": "ok",
        "baseline_run": baseline["run"]["id"],
        "baseline_aklr_sha": baseline["run"]["aklr_sha"],
        "arms": arms,
        "notes": notes,
    }


def compare_aklr(results: dict) -> dict:
    control = results.get("control")
    unavailable = {"status": "unavailable", "points": [], "tput_geomean_ratio": None}
    if not control or not complete_measurement(results):
        return {**unavailable, "reason": "Both AKLR modes must finish every selected point successfully."}
    for kind in ("model", "target"):
        identity = results.get(kind, {}).get("id")
        if not identity or identity != control.get(kind, {}).get("id"):
            return {**unavailable, "reason": f"AKLR modes have different or missing {kind} identities."}
    identity = comparison_identity(results)
    if identity is None or identity != comparison_identity(control):
        return {**unavailable, "reason": "AKLR modes have different or unverified execution identities."}
    for key in ("id", "attempt_id", "aklr_sha", "serving_sha256", "source_dirty"):
        if results["run"].get(key) is None or results["run"].get(key) != control["run"].get(key):
            return {
                **unavailable,
                "reason": "AKLR modes do not belong to the same source and execution attempt.",
            }
    if results["run"].get("aklr_mode") != "enabled" or control["run"].get("aklr_mode") != "disabled":
        return {**unavailable, "reason": "AKLR mode labels are missing or invalid."}
    before = {(p["arm"], p["conc"]): p for p in control["points"]}
    if set(before) != {(p["arm"], p["conc"]) for p in results["points"]}:
        return {**unavailable, "reason": "AKLR modes selected different points."}
    rows = []
    for point in results["points"]:
        old = before[(point["arm"], point["conc"])]
        settings = []
        for record, plugin, disable in ((point, "1", "0"), (old, "0", "1")):
            measurement = copy.deepcopy(record.get("measurement", {}))
            env = measurement.get("launch", {}).get("env", {})
            if env.pop("AKLR_SGLANG_PLUGIN", None) != plugin or env.pop("AKLR_DISABLE", None) != disable:
                return {**unavailable, "reason": "AKLR mode switches do not match their labels."}
            settings.append(measurement)
        if settings[0] != settings[1]:
            return {**unavailable, "reason": "Settings other than the AKLR switches differ between modes."}
        rows.append(
            {
                "arm": point["arm"],
                "conc": point["conc"],
                "enabled": point,
                "disabled": old,
                "tput_ratio": ratio(point[Y], old[Y]),
                "intvty_ratio": ratio(point[X], old[X]),
            }
        )
    return {
        "status": "ok",
        "points": rows,
        "tput_geomean_ratio": geomean([row["tput_ratio"] for row in rows]),
    }
