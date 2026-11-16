# SPDX-License-Identifier: MIT
"""Compute Pareto frontiers and compare runs."""

from __future__ import annotations

import copy
import math

from agentx_ci.preflight import comparison_identity
from agentx_ci.utils import METRIC_AXES, valid_measurement_points, valid_metrics

X, Y = METRIC_AXES


def _same_execution(first: dict, second: dict) -> bool:
    """Partial runs may observe fewer traces; shared traces must still match."""
    return {key: value for key, value in first.items() if key != "traces"} == {
        key: value for key, value in second.items() if key != "traces"
    } and all(trace == second["traces"].get(loader, trace) for loader, trace in first["traces"].items())


def _run_identity(results: dict) -> dict | None:
    """Verify shared execution metadata; each compared point verifies its own trace."""
    return comparison_identity({**results, "points": []}, allow_empty_points=True)


def _point_settings(results: dict, point: dict, *, aklr_mode: str | None = None) -> tuple[str, dict] | None:
    """Return verified settings for a successful point, normalizing only AKLR's switches."""
    spec = point.get("serve_spec_sha256")
    if (
        point.get("status") != "ok"
        or not valid_metrics(point)
        or not isinstance(spec, str)
        or not spec
        or comparison_identity({**results, "points": [point]}) is None
    ):
        return None
    measurement = point["measurement"]
    if aklr_mode is not None:
        measurement = copy.deepcopy(measurement)
        launch = measurement.get("launch")
        env = launch.get("env") if isinstance(launch, dict) else None
        expected = ("1", "0") if aklr_mode == "enabled" else ("0", "1")
        if (
            not isinstance(env, dict)
            or (env.pop("AKLR_SGLANG_PLUGIN", None), env.pop("AKLR_DISABLE", None)) != expected
        ):
            return None
    return spec, measurement


def measured(results: dict, arm: str | None = None) -> list[dict]:
    return [
        p
        for p in results["points"]
        if p["status"] == "ok" and valid_metrics(p) and (arm is None or p["arm"] == arm)
    ]


def frontier(points: list[dict], *, x_axis: str = X) -> list[dict]:
    """Return non-dominated points by increasing interactivity."""
    best_y, keep = -math.inf, []
    for point in sorted(points, key=lambda p: (-p[x_axis], -p[Y])):
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
    before_arm = baseline["serving"]["arms"].get(arm)
    if before_arm is None:
        return {
            "status": "new-arm",
            "reason": "arm absent from the baseline",
            "points": [],
        }
    base_points = {p["conc"]: p for p in baseline["points"] if p["arm"] == arm}
    rows, tput_ratios, changed_concs = [], [], []
    for point in sorted((p for p in current["points"] if p["arm"] == arm), key=lambda p: p["conc"]):
        old = base_points.get(point["conc"])
        row = {"conc": point["conc"], "current": point, "baseline": old}
        if (
            point["status"] == "ok"
            and valid_metrics(point)
            and old
            and old["status"] == "ok"
            and valid_metrics(old)
        ):
            current_settings = _point_settings(current, point)
            previous_settings = _point_settings(baseline, old)
            if current_settings is None or previous_settings is None:
                row["reason"] = "point measurement settings are missing or unverified"
            elif current_settings != previous_settings:
                row["reason"] = "point measurement settings differ"
                changed_concs.append(point["conc"])
            else:
                row["tput_ratio"] = ratio(point[Y], old[Y])
                row["intvty_ratio"] = ratio(point[X], old[X])
                tput_ratios.append(row["tput_ratio"])
        rows.append(row)
    if changed_concs and not tput_ratios:
        return {
            "status": "new-segment",
            "reason": "measurement settings changed (weights, launch, workload or selected points)",
            "changed_concs": changed_concs,
            "points": rows,
            "tput_geomean_ratio": None,
        }
    return {
        "status": "comparable",
        "points": rows,
        "tput_geomean_ratio": geomean(tput_ratios),
    }


def compare(current: dict, baseline: dict | None) -> dict:
    """Compare a run with its previous published baseline."""
    if (
        not valid_measurement_points(current)
        or not measured(current)
        or (baseline is not None and (not valid_measurement_points(baseline) or not measured(baseline)))
    ):
        return {
            "status": "incomplete",
            "reason": "no valid successful measurement points to compare",
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
    current_identity, baseline_identity = _run_identity(current), _run_identity(baseline)
    if current_identity is None or baseline_identity is None:
        return {
            "status": "no-baseline",
            "reason": "execution identity is missing or unverified",
            "arms": {},
            "notes": [],
        }
    if not _same_execution(current_identity, baseline_identity):
        return {
            "status": "no-baseline",
            "reason": "execution image, InferenceX source, dependencies or hardware changed",
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


def aklr_pairing_error(results: dict) -> str | None:
    """Verify both modes' provenance and recorded settings without requiring successful pairs."""
    control = results.get("control")
    if not control:
        return "AKLR off/on execution records are required."
    for kind in ("model", "target"):
        identity = results.get(kind, {}).get("id")
        if not identity or identity != control.get(kind, {}).get("id"):
            return f"AKLR modes have different or missing {kind} identities."
    identity = comparison_identity(results, allow_empty_points=True)
    control_identity = comparison_identity(control, allow_empty_points=True)
    if identity is None or control_identity is None or not _same_execution(identity, control_identity):
        return "AKLR modes have different or unverified execution identities."
    for key in ("id", "attempt_id", "aklr_sha", "serving_sha256", "source_dirty"):
        if results["run"].get(key) is None or results["run"].get(key) != control["run"].get(key):
            return "AKLR modes do not belong to the same source and execution attempt."
    if results["run"].get("aklr_mode") != "enabled" or control["run"].get("aklr_mode") != "disabled":
        return "AKLR mode labels are missing or invalid."
    if results["run"].get("points_expected") != control["run"].get("points_expected"):
        return "AKLR modes selected different points."
    settings = {}
    for mode, plugin, disable in ((results, "1", "0"), (control, "0", "1")):
        for record in mode["points"]:
            key = (record["arm"], record["conc"])
            measurement = copy.deepcopy(record.get("measurement", {}))
            env = measurement.get("launch", {}).get("env", {})
            if env.pop("AKLR_SGLANG_PLUGIN", None) != plugin or env.pop("AKLR_DISABLE", None) != disable:
                return "AKLR mode switches do not match their labels."
            if key in settings and settings[key] != measurement:
                return "Settings other than the AKLR switches differ between modes."
            settings[key] = measurement
    return None


def compare_aklr(results: dict) -> dict:
    control = results.get("control")
    unavailable = {"status": "unavailable", "points": [], "tput_geomean_ratio": None}
    if not control:
        return {**unavailable, "reason": "AKLR off/on execution records are required."}
    if not valid_measurement_points(results) or not valid_measurement_points(control):
        return {**unavailable, "reason": "AKLR measurement point records are invalid or duplicated."}
    # Check run provenance first, then qualify each point independently. Baseline
    # promotion still uses aklr_pairing_error with every recorded point intact.
    error = aklr_pairing_error({**results, "points": [], "control": {**control, "points": []}})
    if error:
        return {**unavailable, "reason": error}
    before = {(p["arm"], p["conc"]): p for p in control["points"]}
    rows = []
    for point in results["points"]:
        old = before.get((point["arm"], point["conc"]))
        if old is None:
            continue
        enabled_settings = _point_settings(results, point, aklr_mode="enabled")
        disabled_settings = _point_settings(control, old, aklr_mode="disabled")
        if enabled_settings is None or enabled_settings != disabled_settings:
            continue
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
    if not rows:
        return {**unavailable, "reason": "No successful points with matching settings in both AKLR modes."}
    return {
        "status": "ok",
        "points": rows,
        "tput_geomean_ratio": geomean([row["tput_ratio"] for row in rows]),
    }
