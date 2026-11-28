# SPDX-License-Identifier: MIT
"""Read e2e.yaml and the e2e recipes, one per model, target and serving framework."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from perf_utils.gh import ConfigError

from e2e_perf.utils import read_json

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[2]
DEFAULT_CONFIG = APP_DIR / "e2e.yaml"
PLUGIN_SWITCHES = {"sglang": "AKLR_SGLANG_PLUGIN", "vllm": "AKLR_VLLM_PLUGIN"}
TP_FLAGS = ("--tp", "--tp-size", "--tensor-parallel-size", "-tp")
WORKLOAD_INTS = ("isl", "osl", "num_prompts_per_conc")


def positive_int(value) -> bool:
    return type(value) is int and value > 0


def valid_concurrency(concurrency) -> bool:
    return (
        isinstance(concurrency, list)
        and bool(concurrency)
        and all(positive_int(conc) for conc in concurrency)
        and len(set(concurrency)) == len(concurrency)
    )


def check_workloads(workloads, path) -> None:
    if not isinstance(workloads, list) or not workloads:
        raise ConfigError(f"{path} needs workloads as a non-empty list")
    ids = []
    for workload in workloads:
        if not isinstance(workload, dict):
            raise ConfigError(f"{path}: every workload must be a mapping")
        ids.append(validate_id(workload.get("id"), "workload"))
        if not all(positive_int(workload.get(key)) for key in WORKLOAD_INTS):
            raise ConfigError(f"{path}: workload {ids[-1]} needs positive ints {', '.join(WORKLOAD_INTS)}")
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{path}: workload ids must be distinct")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or not isinstance(config.get("models"), dict):
        raise ConfigError(f"{path} needs a models mapping")
    if not isinstance(config.get("client"), dict):
        raise ConfigError(f"{path} needs a client mapping")
    check_workloads(config.get("workloads"), path)
    for model_id, model in config["models"].items():
        if not valid_concurrency(model.get("concurrency")):
            raise ConfigError(
                f"{path}: model {model_id} needs concurrency as a non-empty list of distinct positive ints"
            )
    return config


def validate_id(value, kind: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise ConfigError(f"invalid {kind} id {value!r}; use 1–64 lowercase letters, digits, '_' or '-'")
    return value


def group_key(model: str, target: str, framework: str) -> str:
    return f"{model}/{target}/{framework}"


def group_path(model: str, target: str, framework: str) -> Path:
    if framework not in PLUGIN_SWITCHES:
        raise ConfigError(f"invalid framework {framework!r}; use one of {', '.join(PLUGIN_SWITCHES)}")
    return (
        Path("models")
        / validate_id(model, "model")
        / "targets"
        / validate_id(target, "target")
        / "frameworks"
        / framework
    )


def selection(value: str, known: list[str], kind: str) -> set[str]:
    if value == "all":
        return set(known)
    chosen = {item.strip() for item in value.split(",") if item.strip()}
    if unknown := sorted(chosen - set(known)):
        raise ConfigError(f"unknown {kind}: {', '.join(unknown)}; choose from {', '.join(known)} or all")
    return chosen


def model_workloads(config: dict, model: dict) -> list[dict]:
    return [{**workload, "concurrency": model["concurrency"]} for workload in config["workloads"]]


def groups(config: dict, models: str = "all", frameworks: str = "all") -> list[dict]:
    """Every selected model × target × framework group, in e2e.yaml order."""
    model_ids = selection(models, list(config["models"]), "model")
    framework_ids = selection(frameworks, list(PLUGIN_SWITCHES), "framework")
    out = []
    for model_id, model in config["models"].items():
        validate_id(model_id, "model")
        if model_id not in model_ids:
            continue
        for target_id, target in model["targets"].items():
            for framework in target["frameworks"]:
                group_path(model_id, target_id, framework)
                if framework not in framework_ids:
                    continue
                out.append(
                    {
                        "model": {"id": model_id, "display_name": model["display_name"]},
                        "target": {"id": target_id, "runner_label": target["runner_label"]},
                        "framework": {"id": framework},
                        "family": model["family"],
                        "weights": dict(model["weights"]),
                        "hardware": dict(target["hardware"]),
                        "workloads": model_workloads(config, model),
                    }
                )
    return out


def recipe_path(group: dict, root: Path = APP_DIR / "recipes") -> Path:
    return (
        root
        / group["family"]
        / group["target"]["id"]
        / group["framework"]["id"]
        / f"serving.{group['model']['id']}.json"
    )


def load_recipe(group: dict, root: Path = APP_DIR / "recipes") -> dict:
    path = recipe_path(group, root)
    if not path.is_file():
        raise ConfigError(f"missing recipe {path}")
    recipe = read_json(path)
    command, gpus = recipe["launch"]["command"], recipe.get("gpus")
    flags = [index for index, arg in enumerate(command[:-1]) if arg in TP_FLAGS]
    if len(flags) != 1 or not positive_int(gpus) or command[flags[0] + 1] != str(gpus):
        raise ConfigError(f"{path}: gpus must equal the one tensor-parallel size in the launch command")
    if gpus > group["hardware"]["gpu_count"]:
        raise ConfigError(f"{path}: gpus {gpus} exceeds the target's {group['hardware']['gpu_count']}")
    setup = recipe.get("setup", [])
    if not isinstance(setup, list) or not all(isinstance(line, str) and line.strip() for line in setup):
        raise ConfigError(f"{path}: setup must be a list of shell commands")
    return recipe
