# SPDX-License-Identifier: MIT
"""Shared helpers: the run definition, canonical hashes and JSON files."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from pathlib import Path

import yaml

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parents[1]
DEFAULT_CONFIG = APP_DIR / "agentx.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def sha256_json(value) -> str:
    """Hash of ``value``'s canonical JSON: sorted keys, no whitespace."""
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
    """Invalidate an earlier result before parsing a plan or preparing the runtime."""
    attempt = uuid.uuid4().hex
    write_json(
        out / "results.json",
        {
            "model": {},
            "run": {"attempt_id": attempt, "completed": False},
            "points": [],
        },
    )
    return attempt


METRIC_AXES = ("p90_e2e_norm_intvty", "tput_per_gpu")


def valid_metrics(point: dict) -> bool:
    for name in METRIC_AXES:
        value = point.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            if not math.isfinite(value) or value <= 0:
                return False
        except OverflowError:
            return False
    return True


def validated_results(results: dict) -> dict:
    """Treat invalid external/historical measurements as failed points before rendering."""
    points = []
    for original in results["points"]:
        point = dict(original)
        if not valid_metrics(point):
            if point.get("status") == "ok":
                point.update(status="failed", error="Invalid metrics: axes must be finite positive numbers")
            for name in METRIC_AXES:
                point.pop(name, None)
        points.append(point)
    return {**results, "points": points}


def complete_measurement(results: dict) -> bool:
    points, run = results["points"], results["run"]
    identities = [(p["arm"], p["conc"]) for p in points]
    return bool(
        run.get("completed") is True
        and points
        and type(run.get("points_expected")) is int
        and len(points) == run.get("points_expected")
        and len(identities) == len(set(identities))
        and all(p.get("status") == "ok" and valid_metrics(p) for p in points)
    )
