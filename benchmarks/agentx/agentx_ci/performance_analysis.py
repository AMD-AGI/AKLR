# SPDX-License-Identifier: MIT
"""Compute Pareto frontiers and compare runs."""

from __future__ import annotations

import copy
import json
import math

from agentx_ci.preflight import comparison_identity
from agentx_ci.utils import (
    METRIC_AXES,
    PLUGIN_SWITCHES,
    plugin_switch,
    valid_measurement_points,
    valid_metrics,
)

X, Y = METRIC_AXES


# Measurements on other hardware, another AgentX client or other traces are not comparable at all.
INCOMPARABLE_EXECUTION = ("model_id", "target_id", "framework_id", "client", "hardware")


def _same_traces(first: dict, second: dict) -> bool:
    """Partial runs may observe fewer traces; shared traces must still match."""
    return all(trace == second["traces"].get(loader, trace) for loader, trace in first["traces"].items())


def _comparable_execution(first: dict, second: dict) -> bool:
    return all(first[key] == second[key] for key in INCOMPARABLE_EXECUTION)


def _measured_traces(results: dict) -> list[dict]:
    """Trace datasets of verified points; a point measured on another one is excluded on its own."""
    return [p["measurement"]["trace"] for p in verified(results)]


def _trace_key(point: dict) -> str:
    return json.dumps((point.get("measurement") or {}).get("trace"), sort_keys=True)


def frontier_point(point: dict) -> dict:
    """The fields a chart needs from a frontier point."""
    return {key: point[key] for key in ("arm", "conc", X, Y) if key in point}


def _shared_traces(current: list[dict], baseline: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep only points measured on a trace dataset that both sides measured."""
    shared = {_trace_key(p) for p in current} & {_trace_key(p) for p in baseline}
    return [p for p in current if _trace_key(p) in shared], [p for p in baseline if _trace_key(p) in shared]


def _same_execution(first: dict, second: dict) -> bool:
    return _same_traces(first, second) and {
        key: value for key, value in first.items() if key != "traces"
    } == {key: value for key, value in second.items() if key != "traces"}


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
        expected = ("1", "0") if aklr_mode == "enabled" else ("0", "1")
        if _pop_switches(measurement) != expected:
            return None
    return spec, measurement


def _pop_switches(measurement: dict) -> tuple | None:
    """Remove and return the framework's AKLR plugin switch and AKLR_DISABLE."""
    launch = measurement.get("launch")
    env = launch.get("env") if isinstance(launch, dict) else None
    client_env = measurement.get("client_env")
    framework = client_env.get("FRAMEWORK") if isinstance(client_env, dict) else None
    if not isinstance(env, dict) or framework not in PLUGIN_SWITCHES:
        return None
    return env.pop(plugin_switch(framework), None), env.pop("AKLR_DISABLE", None)


def measured(results: dict, arm: str | None = None) -> list[dict]:
    return [
        p
        for p in results["points"]
        if p["status"] == "ok" and valid_metrics(p) and (arm is None or p["arm"] == arm)
    ]


def verified(results: dict, arm: str | None = None) -> list[dict]:
    """Successful points whose recorded measurement settings and provenance verify."""
    return [p for p in measured(results, arm) if _point_settings(results, p) is not None]


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


def _interpolate(line: list[dict], position: float) -> float | None:
    """Throughput at an interactivity on a frontier, linear in log(interactivity)."""
    for left, right in zip(line, line[1:]):
        if left[X] <= position <= right[X]:
            if right[X] == left[X]:
                return max(left[Y], right[Y])
            weight = (math.log(position) - math.log(left[X])) / (math.log(right[X]) - math.log(left[X]))
            return left[Y] + weight * (right[Y] - left[Y])
    return line[0][Y] if len(line) == 1 and line[0][X] == position else None


def compare_curve(current: list[dict], baseline: list[dict], reason: str) -> dict:
    """Compare two Pareto frontiers over their shared interactivity range and trace datasets."""
    current, baseline = _shared_traces(current, baseline)
    now, before = frontier(current), frontier(baseline)
    result = {
        "status": "curve-no-overlap",
        "reason": reason,
        "points": [],
        "frontier_points": {"current": len(now), "baseline": len(before)},
        "baseline_frontier": [frontier_point(p) for p in before],
        "tput_geomean_ratio": None,
    }
    if not now or not before:
        return result
    low, high = max(now[0][X], before[0][X]), min(now[-1][X], before[-1][X])
    if low > high:
        return result
    positions = []
    for position in sorted({p[X] for p in now + before if low <= p[X] <= high}):
        new, old = _interpolate(now, position), _interpolate(before, position)
        if new is not None and old:
            positions.append(
                {"intvty": position, "current": new, "baseline": old, "tput_ratio": ratio(new, old)}
            )
    if not positions:
        return result
    ratios = [row["tput_ratio"] for row in positions]
    return {
        **result,
        "status": "curve",
        "overlap": [low, high],
        "positions": positions,
        "tput_geomean_ratio": geomean(ratios),
        "tput_ratio_min": min(ratios),
        "tput_ratio_max": max(ratios),
    }


def compare_arm(arm: str, current: dict, baseline: dict, curve_reason: str | None = None) -> dict:
    before_arm = baseline["serving"]["arms"].get(arm)
    if before_arm is None:
        return {
            "status": "new-arm",
            "reason": "arm absent from the baseline",
            "points": [],
        }
    if curve_reason is not None:
        return compare_curve(verified(current, arm), verified(baseline, arm), curve_reason)
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
        curve = compare_curve(
            verified(current, arm),
            verified(baseline, arm),
            "measurement settings changed (weights, launch, workload or selected points)",
        )
        return {**curve, "changed_concs": changed_concs, "point_rows": rows}
    return {
        "status": "comparable",
        "points": rows,
        "tput_geomean_ratio": geomean(tput_ratios),
    }


def _arm_inputs(arm: str, view: dict, current: dict, baseline: dict) -> tuple[list[dict], list[dict]]:
    """The points an arm's comparison used: matched pairs, or its verified points for a curve."""
    if is_curve(view):
        return verified(current, arm), verified(baseline, arm)
    pairs = [(row["current"], row["baseline"]) for row in view["points"] if row.get("tput_ratio") is not None]
    return [new for new, _ in pairs], [old for _, old in pairs]


def compare(current: dict, baseline: dict | None) -> dict:
    """Compare a run with its previous published baseline.

    Matching selections and execution compare point by point; a changed selection, image or
    package set compares Pareto frontiers instead. Other hardware, another AgentX client or other
    traces have no baseline. The combined frontier uses the points the arms compared, or every
    verified point when the whole run compares by curve; curves use only shared trace datasets.
    """
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
    for kind in ("model", "target", "framework"):
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
    current_traces, baseline_traces = _measured_traces(current), _measured_traces(baseline)
    if not _comparable_execution(current_identity, baseline_identity) or (
        current_traces and baseline_traces and not any(trace in baseline_traces for trace in current_traces)
    ):
        return {
            "status": "no-baseline",
            "reason": "hardware, AgentX client source or trace dataset changed",
            "arms": {},
            "notes": [],
        }
    notes, curve_reasons = [], []
    if current["run"].get("point_policy") != baseline["run"].get("point_policy") or current["run"].get(
        "selection", {}
    ).get("selection_sha256") != baseline["run"].get("selection", {}).get("selection_sha256"):
        curve_reasons.append("point selection policy or selected recipes changed")
    if not _same_execution(current_identity, baseline_identity):
        curve_reasons.append("execution image or dependencies changed")
    if baseline["run"]["image"]["run"] != current["run"]["image"]["run"]:
        notes.append("run image changed")
    if baseline["serving"]["grid_sha256"] != current["serving"]["grid_sha256"]:
        notes.append("concurrency grid changed")
    curve_reason = "; ".join(curve_reasons) or None
    arms = {arm: compare_arm(arm, current, baseline, curve_reason) for arm in current["serving"]["arms"]}
    if curve_reason is not None:
        combined = compare_curve(verified(current), verified(baseline), curve_reason)
    else:
        inputs = [
            _arm_inputs(arm, view, current, baseline)
            for arm, view in arms.items()
            if view["status"] != "new-arm"
        ]
        combined = compare_curve(
            [p for new, _ in inputs for p in new], [p for _, old in inputs for p in old], "combined frontier"
        )
    return {
        "status": "ok",
        "method": "curve" if curve_reason is not None else comparison_method(arms),
        "baseline_run": baseline["run"]["id"],
        "baseline_aklr_sha": baseline["run"]["aklr_sha"],
        "arms": arms,
        "combined": combined,
        "notes": curve_reasons + notes,
    }


def is_curve(view: dict) -> bool:
    return view.get("status") in ("curve", "curve-no-overlap")


def comparison_method(arms: dict) -> str:
    """`point`, `curve` or `mixed`, from how the compared arms were actually compared."""
    methods = {
        "curve" if is_curve(view) else "point" for view in arms.values() if view["status"] != "new-arm"
    }
    return methods.pop() if len(methods) == 1 else "mixed" if methods else "point"


def aklr_pairing_error(results: dict) -> str | None:
    """Verify both modes' provenance and recorded settings without requiring successful pairs."""
    control = results.get("control")
    if not control:
        return "AKLR off/on execution records are required."
    for kind in ("model", "target", "framework"):
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
            if _pop_switches(measurement) != (plugin, disable):
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
