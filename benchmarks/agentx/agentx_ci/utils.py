# SPDX-License-Identifier: MIT
"""Shared configuration, result and JSON helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import uuid
from pathlib import Path

import yaml

from agentx_ci.model_config import validate_model_id, validate_target_id

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parents[1]
DEFAULT_CONFIG = APP_DIR / "agentx.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def target_path(model_id: str, target_id: str) -> Path:
    return Path("models") / validate_model_id(model_id) / "targets" / validate_target_id(target_id)


def sha256_json(value) -> str:
    """Hash canonical JSON."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return path


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def begin_attempt(out: Path) -> str:
    """Invalidate previous results before setup."""
    attempt = uuid.uuid4().hex
    write_json(
        out / "results.json",
        {
            "schema": 3,
            "model": {},
            "target": {},
            "run": {"attempt_id": attempt, "completed": False},
            "points": [],
        },
    )
    return attempt


def pending_results(serving: dict, points: list[dict], run_fields: dict) -> dict:
    """Keep plan identity available if setup fails."""
    if serving.get("schema") != 3 or not isinstance(serving.get("target"), dict):
        raise ValueError("serving plan has no target identity; regenerate it with plan or resolve")
    target_path(serving["model"]["id"], serving["target"]["id"])
    selection = serving.get("selection", {})
    policy = selection.get("policy", "all")
    results = {
        "schema": 3,
        "model": serving["model"],
        "target": serving["target"],
        "run": {
            "mode": "live",
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "completed": False,
            "serving_sha256": sha256_json(serving),
            "image": serving["image"],
            "expected_hardware": serving["hardware"],
            "aklr_mode": "enabled",
            "inferencex": serving["provenance"]["inferencex"],
            "weights": serving["weights"],
            "points_expected": len(points),
            "point_policy": policy,
            "arms_requested": selection.get("arms", "all"),
            "points_requested": selection.get("points") if policy == "all" else None,
            "duration_override_s": selection.get("duration"),
            "model_path_override": None,
            **({"selection": selection} if policy != "all" else {}),
            **run_fields,
        },
        "serving": {
            "grid_sha256": serving["grid_sha256"],
            "arms": {
                arm: dict(spec)
                for arm, spec in serving["arms"].items()
                if any(point["arm"] == arm for point in points)
            },
        },
        "points": [],
    }
    results["control"] = copy.deepcopy(results)
    results["control"]["run"]["aklr_mode"] = "disabled"
    return results


INTERACTIVITY_METRIC = "p90_full_response_intvty"
METRIC_AXES = (INTERACTIVITY_METRIC, "tput_per_gpu")
LEGACY_METRIC_AXES = ("p90_e2e_norm_intvty", "tput_per_gpu")


def positive_metric(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def full_response_interactivity(aggregate: dict) -> float:
    """Match the InferenceX CSV's reciprocal of its saved P90 ITL in seconds."""
    try:
        p90_itl = aggregate["request_metrics"]["latency"]["full_response_itl"]["p90"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Missing full-response P90 ITL; cannot use E2E interactivity for this axis"
        ) from error
    if not positive_metric(p90_itl):
        raise ValueError("Invalid full-response P90 ITL: expected finite positive seconds")
    # Use the saved ITL, not full_response_intvty.p90, which was calculated before
    # upstream rounded the ITL to five decimal places for its CSV export.
    interactivity = 1 / p90_itl
    if not positive_metric(interactivity):
        raise ValueError("Invalid full-response P90 interactivity: reciprocal must be finite")
    return interactivity


def valid_metrics(point: dict, *, axes: tuple[str, str] = METRIC_AXES) -> bool:
    return all(positive_metric(point.get(name)) for name in axes)


def validated_results(results: dict) -> dict:
    """Recompute the report axis from raw data, including older E2E-based runs."""
    points = []
    for original in results["points"]:
        point = dict(original)
        if point.get("status") == "ok" and ("raw_data" in point or INTERACTIVITY_METRIC not in point):
            try:
                point[INTERACTIVITY_METRIC] = full_response_interactivity(point.get("raw_data"))
            except ValueError as error:
                point.update(status="failed", error=str(error))
                point.pop(INTERACTIVITY_METRIC, None)
        if not valid_metrics(point):
            if point.get("status") == "ok":
                point.update(status="failed", error="Invalid metrics: axes must be finite positive numbers")
            for name in METRIC_AXES:
                point.pop(name, None)
        points.append(point)
    return {
        **results,
        "points": points,
        **({"control": validated_results(results["control"])} if "control" in results else {}),
    }


def complete_measurement(results: dict, *, axes: tuple[str, str] = METRIC_AXES) -> bool:
    points, run = results["points"], results["run"]
    identities = [(p["arm"], p["conc"]) for p in points]
    return bool(
        run.get("completed") is True
        and points
        and type(run.get("points_expected")) is int
        and len(points) == run.get("points_expected")
        and len(identities) == len(set(identities))
        and all(p.get("status") == "ok" and valid_metrics(p, axes=axes) for p in points)
        and ("control" not in results or complete_measurement(results["control"], axes=axes))
    )
