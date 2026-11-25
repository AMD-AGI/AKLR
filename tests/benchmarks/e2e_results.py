# SPDX-License-Identifier: MIT
"""Build daily end-to-end performance results for tests."""

from __future__ import annotations

CONCS = [1, 2, 4, 8, 16, 32, 64]
WORKLOAD = "8k1k"


def point(
    conc: int,
    tput: float | None = 100.0,
    intvty: float = 50.0,
    status: str = "ok",
    workload: str = WORKLOAD,
) -> dict:
    if status != "ok" or tput is None:
        return {
            "workload": workload,
            "conc": conc,
            "status": status if status != "ok" else "failed",
            "error": "boom",
        }
    return {
        "workload": workload,
        "conc": conc,
        "status": "ok",
        "error": None,
        "tput_per_gpu": tput,
        "p90_full_response_intvty": intvty,
    }


def results(
    disabled: list[dict] | None = None,
    enabled: list[dict] | None = None,
    *,
    completed: bool = True,
    model: str = "glm52-fp4",
    framework: str = "sglang",
    run_id: str = "101",
) -> dict:
    disabled = [point(conc) for conc in CONCS] if disabled is None else disabled
    enabled = [point(conc, 110.0, 55.0) for conc in CONCS] if enabled is None else enabled
    return {
        "schema": 1,
        "model": {"id": model, "display_name": "GLM-5.2 MXFP4"},
        "target": {"id": "mi355x", "runner_label": "agentx-mi355x"},
        "framework": {"id": framework},
        "run": {
            "id": run_id,
            "run_key": "k" * 64,
            "completed": completed,
            "error": None,
            "finished_at": "2026-11-23T08:00:00Z",
            "aklr_sha": "a" * 40,
            "host": "node-1",
            "image": {"run": "example.invalid/sglang:v1@sha256:" + "b" * 64},
            "hardware": {"devices": [{"name": "AMD Instinct MI355X"}]},
            "workloads": [{"id": WORKLOAD, "isl": 8192, "osl": 1024, "concurrency": CONCS}],
            "gpus": 8,
        },
        "modes": {"disabled": {"points": disabled}, "enabled": {"points": enabled}},
    }
