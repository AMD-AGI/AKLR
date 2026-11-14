# SPDX-License-Identifier: MIT
"""Check reviewed recipe serving.json files and derive the run-layer fields that plan adds."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

from agentx_ci.model_config import RECIPE_SOURCE_KINDS
from agentx_ci.preflight import pinned_image
from agentx_ci.utils import SCHEMA, sha256_json

POLICY = "pareto"
SOURCE_KINDS = RECIPE_SOURCE_KINDS
# Identity owned by agentx.yaml; plan copies it in.
AGENTX_TOP = ("model", "target", "framework", "weights", "hardware")
# Fields owned by plan; a recipe file must not contain them.
RUN_TOP = ("plan_id", "grid_sha256", "arms", "selection")
RUN_POINT = ("hashes", "trace")
RUN_CLIENT_ENV = ("WEKA_LOADER_OVERRIDE",)
# The AgentX client derives num_gpus, and so throughput per GPU, from these.
TOPOLOGY_ENV = ("TP", "PP_SIZE", "PCP_SIZE", "DCP_SIZE", "EP_SIZE")
# Plan derives device placement from each point's gpus.
DEVICE_ENV = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
RUNTIME_LIMITS = ("ready_timeout_s", "point_timeout_s", "poll_s")
RUN_FIELDS = ("selection", "plan_id")
SHA1 = re.compile(r"[0-9a-f]{40}")
# exp_name names a result directory, so it must stay a single safe path component.
EXP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
GITHUB_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class RecipeError(ValueError):
    pass


class SelectionError(ValueError):
    pass


def recipe_name(model_id: str) -> str:
    return f"serving.{model_id}.json"


def recipe_path(config: str | Path, definition: dict) -> Path:
    """recipes/<family>/<target-id>/<framework>/serving.<model-id>.json beside agentx.yaml."""
    return (
        Path(config).resolve().parent
        / "recipes"
        / definition["family"]
        / definition["target"]["id"]
        / definition["framework"]["id"]
        / recipe_name(definition["model"]["id"])
    )


def point_id(point: dict) -> str:
    return f"{point['arm']}:{point['conc']}"


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RecipeError(message)


def _strings(value) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


def _check_topology(label: str, env: dict, gpus: int) -> None:
    for name in TOPOLOGY_ENV:
        value = env.get(name)
        _expect(
            isinstance(value, str) and value.isdigit() and int(value) > 0,
            f"point {label} client_env.{name} must be a positive integer string",
        )
    _expect(
        int(env["TP"]) * int(env["PP_SIZE"]) * int(env["PCP_SIZE"]) == gpus,
        f"point {label} client_env TP x PP_SIZE x PCP_SIZE must equal gpus ({gpus})",
    )
    _expect(
        env.get("DP_ATTENTION") in ("true", "false"),
        f"point {label} client_env.DP_ATTENTION must be 'true' or 'false'",
    )


def check_point_runtime(label: str, point: dict) -> None:
    """Check the runtime and client settings sweep needs, so a bad recipe fails before a GPU job."""
    runtime = point.get("runtime")
    _expect(isinstance(runtime, dict), f"point {label} needs runtime settings")
    for field in RUNTIME_LIMITS:
        value = runtime.get(field)
        _expect(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0,
            f"point {label} runtime.{field} needs a finite positive value",
        )
    _expect("port" in runtime, f"point {label} runtime.port is required; use null for a free port")
    port = runtime["port"]
    _expect(
        port is None or (type(port) is int and 0 < port < 65536),
        f"point {label} runtime.port must be null or a TCP port",
    )
    env = point.get("client_env") or {}
    dram = env.get("TOTAL_CPU_DRAM_GB")
    _expect(
        isinstance(dram, str) and dram.isdigit(), f"point {label} client_env.TOTAL_CPU_DRAM_GB must be digits"
    )
    _expect(
        env.get("CONC") == str(point.get("conc")),
        f"point {label} client_env.CONC must equal its conc, which labels the result",
    )
    _expect(
        env.get("AIPERF_EXPERIMENTAL_FAST") == "0",
        f"point {label} client_env.AIPERF_EXPERIMENTAL_FAST must be '0'; fast mode replaces DURATION",
    )


def _check_source(source: dict, configured: dict) -> None:
    """Require the recipe to come from the source agentx.yaml names for this run."""
    kind = configured["kind"]
    _expect(
        source.get("kind") == kind, f"recipe source.kind must be {kind!r}, as agentx.yaml recipe_source says"
    )
    keys = ("repo", "config_key") if kind == "inferencex" else ("url",)
    for key in keys:
        _expect(
            source.get(key) == configured.get(key),
            f"recipe source.{key} must be {configured.get(key)!r}, as agentx.yaml recipe_source says",
        )
    expect = configured.get("expect") or {}
    expected = [expect[key] for key in ("config_file", "recipe") if key in expect]
    missing = [path for path in expected if path not in (source.get("files") or {})]
    _expect(not missing, f"recipe source.files is missing the agentx.yaml recipe_source files: {missing}")


def check_recipe(recipe: dict, definition: dict) -> None:
    """Validate a recipe file against the schema 4 contract and agentx.yaml."""
    _expect(isinstance(recipe, dict) and recipe.get("schema") == SCHEMA, f"recipe schema must be {SCHEMA}")
    leaked = [key for key in AGENTX_TOP if key in recipe]
    _expect(not leaked, f"recipe contains agentx.yaml-owned fields: {leaked}")
    leaked = [key for key in RUN_TOP if key in recipe]
    _expect(not leaked, f"recipe contains plan-owned fields: {leaked}")
    for key in ("image", "source", "client", "trace"):
        _expect(isinstance(recipe.get(key), dict), f"recipe.{key} must be an object")
    _expect(pinned_image(recipe["image"].get("run", "")), "recipe image.run must be pinned by digest")
    source = recipe["source"]
    _expect(source.get("kind") in SOURCE_KINDS, f"recipe source.kind must be one of {SOURCE_KINDS}")
    _expect(
        isinstance(source.get("repo") or source.get("url"), str), "recipe source needs a repo or url string"
    )
    _check_source(source, definition["recipe_source"])
    client = recipe["client"]
    _expect(
        isinstance(client.get("repo"), str)
        and GITHUB_REPO.fullmatch(client["repo"])
        and SHA1.fullmatch(str(client.get("sha"))),
        "recipe client needs an owner/name repo and a 40-character sha",
    )
    trace = recipe["trace"]
    _expect(
        all(isinstance(trace.get(key), str) and trace[key] for key in ("loader", "repo"))
        and SHA1.fullmatch(str(trace.get("revision"))),
        "recipe trace needs loader, repo and a pinned revision",
    )
    points = recipe.get("points")
    _expect(isinstance(points, list) and points, "recipe.points must be a nonempty list")
    ids, exp_names = [], []
    for point in points:
        _expect(
            isinstance(point, dict) and isinstance(point.get("arm"), str) and point["arm"],
            "every point needs an arm name",
        )
        _expect(type(point.get("conc")) is int and point["conc"] > 0, "point conc must be a positive integer")
        label = point_id(point)
        ids.append(label)
        leaked = [key for key in RUN_POINT if key in point]
        _expect(not leaked, f"point {label} contains plan-owned fields: {leaked}")
        for key in ("exp_name", "served_model_name"):
            _expect(isinstance(point.get(key), str) and point[key], f"point {label} needs {key}")
        _expect(
            EXP_NAME.fullmatch(point["exp_name"]),
            f"point {label} exp_name must be a single path component of letters, digits, '.', '_' or '-'",
        )
        exp_names.append(point["exp_name"])
        _expect(
            type(point.get("gpus")) is int and 0 < point["gpus"] <= definition["hardware"]["gpu_count"],
            f"point {label} gpus must fit the target hardware",
        )
        launch = point.get("launch")
        _expect(
            isinstance(launch, dict)
            and isinstance(launch.get("command"), list)
            and launch["command"]
            and all(isinstance(arg, str) for arg in launch["command"])
            and _strings(launch.get("env")),
            f"point {label} launch needs a command list and string env",
        )
        leaked = sorted(set(DEVICE_ENV) & set(launch["env"]))
        _expect(not leaked, f"point {label} launch.env must not set devices: {leaked}")
        _expect(_strings(point.get("client_env")), f"point {label} client_env must map strings")
        framework = definition["framework"]["id"]
        _expect(
            point["client_env"].get("FRAMEWORK") == framework,
            f"point {label} client_env.FRAMEWORK must be {framework!r}, the framework this recipe is filed under",
        )
        leaked = sorted(set(RUN_CLIENT_ENV) & set(point["client_env"]))
        _expect(not leaked, f"point {label} client_env contains plan-owned keys: {leaked}")
        _check_topology(label, point["client_env"], point["gpus"])
        check_point_runtime(label, point)
    _expect(len(ids) == len(set(ids)), "recipe points must have unique ARM:CONC ids")
    _expect(len(exp_names) == len(set(exp_names)), "recipe points must have unique exp_name values")


def spec_sha256(spec: dict) -> str:
    """Hash the serving configuration without per-run fields."""
    return sha256_json({k: v for k, v in spec.items() if k not in RUN_FIELDS})


def seal_selection(selection: dict) -> dict:
    """Seal which points are selected; their launch settings are compared point by point."""
    selection["selection_sha256"] = sha256_json(
        {"points": sorted(selection["points"]), "points_requested": selection.get("points_requested")}
    )
    selection["manifest_sha256"] = sha256_json({k: v for k, v in selection.items() if k != "manifest_sha256"})
    return selection


def build_run(recipe: dict, definition: dict) -> dict:
    """Add agentx.yaml identity and plan-owned fields: env, trace, arms, hashes and a sealed selection."""
    check_recipe(recipe, definition)
    spec = {"schema": SCHEMA, **{key: copy.deepcopy(definition[key]) for key in AGENTX_TOP}}
    spec.update(copy.deepcopy(recipe))
    trace = spec.pop("trace")
    advisory = definition.get("advisory_env") or {}
    for point in spec["points"]:
        launch = point["launch"]
        clash = sorted(set(advisory) & set(launch["env"]))
        if clash:
            raise RecipeError(f"advisory env {clash} would override what recipe point {point_id(point)} sets")
        point["hashes"] = {
            "serve_spec_sha256": sha256_json({"command": launch["command"], "env": launch["env"]})
        }
        devices = {"HIP_VISIBLE_DEVICES": ",".join(str(index) for index in range(point["gpus"]))}
        launch["env"] = dict(sorted({**launch["env"], **advisory, **devices}.items()))
        point["client_env"] = dict(
            sorted({**point["client_env"], "WEKA_LOADER_OVERRIDE": trace["loader"]}.items())
        )
        point["trace"] = copy.deepcopy(trace)
    spec["arms"] = {}
    for name in dict.fromkeys(point["arm"] for point in spec["points"]):
        mine = [point for point in spec["points"] if point["arm"] == name]
        spec["arms"][name] = {
            "concs": [point["conc"] for point in mine],
            "arm_serve_sha256": sha256_json({p["conc"]: p["hashes"]["serve_spec_sha256"] for p in mine}),
        }
    spec["grid_sha256"] = sha256_json([[p["arm"], p["conc"]] for p in spec["points"]])
    spec["selection"] = full_selection(spec)
    return spec


def full_selection(spec: dict) -> dict:
    """Seal a selection of every point in a serving plan whose points carry their hashes."""
    selection = {
        "points": [point_id(p) for p in spec["points"]],
        "policy": POLICY,
        "duration": None,
        "points_requested": None,
        "serve_sha256": sha256_json({point_id(p): p["hashes"]["serve_spec_sha256"] for p in spec["points"]}),
    }
    selection["spec_sha256"] = spec_sha256(spec)
    return seal_selection(selection)


def narrow_selection(selection: dict, points: str | None, duration: int | None) -> dict:
    """Restrict a selection to smoke points and an optional duration override."""
    result = copy.deepcopy(selection)
    if points:
        wanted = [value.strip() for value in points.split(",")]
        if len(set(wanted)) != len(wanted) or not all(wanted):
            raise SelectionError("--points must list distinct ARM:CONC ids")
        missing = sorted(set(wanted) - set(selection["points"]))
        if missing:
            raise SelectionError(f"points are not in the recipe selection: {missing}")
        result["points"] = [p for p in selection["points"] if p in wanted]
        result["points_requested"] = ",".join(result["points"])
    result["duration"] = duration
    return seal_selection(result)


def validate_manifest(selection: dict) -> None:
    """Check a sealed selection."""
    if selection.get("policy") != POLICY:
        raise SelectionError("Unsupported point selection; regenerate the plan")
    if selection.get("manifest_sha256") != sha256_json(
        {k: v for k, v in selection.items() if k != "manifest_sha256"}
    ):
        raise SelectionError("Selection manifest was modified; regenerate the plan")
    if not selection.get("points"):
        raise SelectionError("Selection manifest is empty")


def selected_points(spec: dict) -> list[dict]:
    selection = spec.get("selection", {})
    validate_manifest(selection)
    if selection.get("spec_sha256") != spec_sha256(spec):
        raise SelectionError("Serving configuration changed after selection; regenerate the plan")
    wanted = set(selection["points"])
    points = [p for p in spec["points"] if point_id(p) in wanted]
    if len(points) != len(wanted):
        raise SelectionError("Selected points no longer exist in the serving plan")
    return points
