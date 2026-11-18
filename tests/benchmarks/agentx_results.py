# SPDX-License-Identifier: MIT
"""AgentX result payloads and the agentx_ci modules the AgentX tests share."""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
with mock.patch.object(sys, "path", [str(REPO / "benchmarks/agentx"), *sys.path]):
    utils = importlib.import_module("agentx_ci.utils")
    runtime = importlib.import_module("agentx_ci.runtime")
    analysis = importlib.import_module("agentx_ci.performance_analysis")
    report = importlib.import_module("agentx_ci.report")
    report_all = importlib.import_module("agentx_ci.report_all")
    serving_config = importlib.import_module("agentx_ci.serving_config")

FULL_RESPONSE = "p90_full_response_intvty"
E2E = "p90_e2e_norm_intvty"


def aggregate(itl=0.00351, *, e2e=84.72682, throughput=3390.56736):
    return {
        "num_gpus": 1,
        "num_requests_successful": 73,
        "request_metrics": {
            "latency": {
                "full_response_itl": {"p90": itl},
                "full_response_intvty": {"p90": 285.10342},
                "e2e_norm_intvty": {"p90": e2e},
            },
            "throughput": {"per_gpu": {"total_tput_tps": throughput}},
        },
    }


def sealed_selection(points):
    selection = {
        "points": list(points),
        "policy": serving_config.POLICY,
        "duration": None,
        "points_requested": None,
    }
    return serving_config.seal_selection(selection)


def legacy_results(run_id="fixture-run", *, itl=0.00351):
    """A complete run with cached E2E metrics and verified, paired execution identities."""
    image = "example.invalid/server@sha256:" + "a" * 64
    result = {
        "schema": 4,
        "model": {"id": "fixture", "display_name": "Fixture"},
        "target": {"id": "fixture-gpu"},
        "framework": {"id": "sglang"},
        "run": {
            "id": run_id,
            "attempt_id": "fixture-attempt",
            "mode": "live",
            "aklr_mode": "enabled",
            "aklr_sha": "b" * 40,
            "source_dirty": False,
            "completed": True,
            "points_expected": 1,
            "serving_sha256": "c" * 64,
            "finished_at": "2026-11-09T19:35:00Z",
            "host": "fixture-host",
            "arms_requested": "all",
            "point_policy": serving_config.POLICY,
            "selection": sealed_selection(["tp1:1"]),
            "duration_override_s": None,
            "image": {"run": image},
            "execution": {"image_reference": image, "image_id": "sha256:" + "d" * 64},
            "expected_hardware": {"gpu_count": 1, "gpu_name_pattern": "Fixture", "gpu_arch": "gfx950"},
            "hardware": {
                "devices": [{"name": "Fixture", "arch": "gfx950", "total_memory": 1024}],
                "peer_access": [[True]],
            },
            "client_packages": {"agentx-harness": "1.0.13"},
            "server_packages": {"sglang": "0.5.22"},
            "source": {"kind": "inferencex", "repo": "example/source", "sha": "e" * 40},
            "client": {"repo": "example/source", "sha": "e" * 40},
        },
        "serving": {
            "grid_sha256": "f" * 64,
            "arms": {"tp1": {"concs": [1], "arm_serve_sha256": "1" * 64}},
        },
        "points": [
            {
                "arm": "tp1",
                "conc": 1,
                "exp_name": "tp1-c1",
                "serve_spec_sha256": "2" * 64,
                "status": "ok",
                "error": None,
                "tput_per_gpu": 3390.56736,
                E2E: 84.72682,
                "raw_data": aggregate(itl),
                "measurement": {
                    "trace": {"repo": "example/traces", "loader": "fixture", "revision": "3" * 40},
                    "launch": {"env": {"AKLR_SGLANG_PLUGIN": "1", "AKLR_DISABLE": "0"}},
                    "client_env": {"FRAMEWORK": "sglang"},
                },
            }
        ],
    }
    control = copy.deepcopy(result)
    control["run"]["aklr_mode"] = "disabled"
    point = control["points"][0]
    point.update(
        tput_per_gpu=3372.91456,
        p90_e2e_norm_intvty=155.05189,
        raw_data=aggregate(0.00359, e2e=155.05189, throughput=3372.91456),
    )
    point["measurement"]["launch"]["env"] = {"AKLR_SGLANG_PLUGIN": "0", "AKLR_DISABLE": "1"}
    result["control"] = control
    return result


POINT_IDS = ["tp1:1", "tp1:2", "tp1:3"]


def formal_results(run_id="fixture-run"):
    results = utils.validated_results(legacy_results(run_id))
    for mode in (results, results["control"]):
        mode["run"].update(points_expected=3, attempt_id=f"{run_id}-attempt")
        template = mode["points"][0]
        points = []
        for conc in (1, 2, 3):
            point = copy.deepcopy(template)
            point.update(
                conc=conc,
                exp_name=f"tp1-c{conc}",
                serve_spec_sha256=utils.sha256_json({"arm": "tp1", "conc": conc}),
            )
            point["measurement"]["client_env"] = {"FRAMEWORK": "sglang", "CONC": str(conc)}
            point["measurement_sha256"] = utils.sha256_json(point["measurement"])
            points.append(point)
        mode["points"] = points
        mode["serving"]["arms"]["tp1"].update(
            concs=[1, 2, 3],
            arm_measurement_sha256=utils.sha256_json({p["conc"]: p["measurement"] for p in points}),
        )
    return use_selection(results, pareto_selection())


def pareto_selection():
    return sealed_selection(POINT_IDS)


def use_selection(results, selection):
    for mode in (results, results["control"]):
        mode["run"].update(point_policy=selection["policy"], selection=copy.deepcopy(selection))
    return results


def fail(point):
    point.update(status="failed", error="server failed before ready")
