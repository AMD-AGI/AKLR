# SPDX-License-Identifier: MIT
"""JSON files, hashes and timestamps; the pinned weight snapshot; metrics and AKLR-off / on pairing of one run."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from pathlib import Path

SCHEMA = 1
THROUGHPUT = "tput_per_gpu"
INTERACTIVITY = "p90_full_response_intvty"


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return path


def sha256_json(value) -> str:
    """Hash canonical JSON."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def begin_attempt(out: Path) -> str:
    """Replace any earlier results before setup, so a crash cannot leave them standing."""
    attempt = uuid.uuid4().hex
    write_json(
        out / "results.json",
        {"schema": SCHEMA, "run": {"attempt_id": attempt, "completed": False}, "modes": {}},
    )
    return attempt


def snapshot(weights: dict, hub: Path) -> Path:
    """The snapshot directory of the pinned revision; the server loads it and the client tokenizes with it."""
    path = hub / ("models--" + weights["repo"].replace("/", "--")) / "snapshots" / weights["revision"]
    if not path.is_dir():
        raise RuntimeError(f"model snapshot {path} not found; prefill the HF hub cache")
    return path


def positive(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


def metrics_of(result: dict, gpus: int, requests: int) -> dict:
    """The two AgentX axes from a bench_serving result, as InferenceX's fixed-sequence results compute them.

    Total tok/s/GPU counts input and output tokens; P90 tok/s/user is 1 / P90 TPOT (per request).
    bench_serving passes with up to 5% of requests failed; an on / off comparison needs all of them.
    """
    completed = result.get("completed")
    if completed != requests:
        raise ValueError(f"{completed} of {requests} requests completed")
    throughput, tpot = result.get("total_token_throughput"), result.get("p90_tpot_ms")
    if not positive(throughput) or not positive(tpot):
        raise ValueError("result needs positive total_token_throughput and p90_tpot_ms")
    return {THROUGHPUT: throughput / gpus, INTERACTIVITY: 1000 / tpot}


def ratio(enabled, disabled) -> float | None:
    return enabled / disabled if positive(enabled) and positive(disabled) else None


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else None


def usable(point: dict | None) -> bool:
    return bool(point and point.get("status") == "ok" and positive(point.get(THROUGHPUT)))


def pair(results: dict) -> dict:
    """Rows per workload and concurrency; a row is matched when both modes measured it.

    Status is ok when the run completed with every point matched, partial with some matched, else failed.
    """
    keys = expected_points(results)
    modes = {
        mode: {
            (point.get("workload"), point["conc"]): point
            for point in results.get("modes", {}).get(mode, {}).get("points", [])
        }
        for mode in ("disabled", "enabled")
    }
    rows = []
    for workload, conc in keys:
        off, on = modes["disabled"].get((workload, conc)), modes["enabled"].get((workload, conc))
        matched = usable(off) and usable(on)
        rows.append(
            {
                "workload": workload,
                "conc": conc,
                "disabled": off,
                "enabled": on,
                "tput_ratio": ratio(on[THROUGHPUT], off[THROUGHPUT]) if matched else None,
                "intvty_ratio": ratio(on.get(INTERACTIVITY), off.get(INTERACTIVITY)) if matched else None,
            }
        )
    ratios = [row["tput_ratio"] for row in rows if row["tput_ratio"] is not None]
    complete = results["run"].get("completed") is True and bool(keys) and len(ratios) == len(keys)
    return {
        "rows": rows,
        "matched": len(ratios),
        "expected": len(keys),
        "status": "ok" if complete else "partial" if ratios else "failed",
        "complete": complete,
        "tput_geomean_ratio": geomean(ratios),
        "min_ratio": min(ratios, default=None),
        "max_ratio": max(ratios, default=None),
    }


def expected_points(results: dict) -> list[tuple[str, int]]:
    """Every (workload id, concurrency) the run planned, in run order."""
    return [
        (workload["id"], conc) for workload in results["run"]["workloads"] for conc in workload["concurrency"]
    ]
