# SPDX-License-Identifier: MIT
"""Select model/target/framework runs and merge shared settings."""

from __future__ import annotations

import copy
import re

RECIPE_SOURCE_KINDS = ("inferencex", "cookbook")
# AKLR's plugin switch for each supported serving framework; framework ids are its keys.
PLUGIN_SWITCHES = {"sglang": "AKLR_SGLANG_PLUGIN", "vllm": "AKLR_VLLM_PLUGIN"}


class ConfigError(ValueError):
    pass


def validate_env(values: dict) -> dict:
    if not isinstance(values, dict) or any(
        not isinstance(key, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
        or not isinstance(value, str)
        or "\0" in value
        for key, value in values.items()
    ):
        raise ConfigError("advisory_env must map valid environment names to strings; quote YAML values")
    return values


def hardware_config(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ConfigError("each target needs hardware: gpu_name_pattern, gpu_arch and gpu_count")
    if type(value.get("gpu_count")) is not int or value["gpu_count"] <= 0:
        raise ConfigError("hardware.gpu_count must be a positive integer")
    for key in ("gpu_name_pattern", "gpu_arch"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ConfigError(f"hardware.{key} must be a nonempty string")
    try:
        re.compile(value["gpu_name_pattern"])
    except re.error as error:
        raise ConfigError(f"invalid hardware.gpu_name_pattern: {error}") from error
    return value


def validate_id(value: str, kind: str) -> str:
    if value == "all":
        raise ConfigError(f"{kind} id 'all' is reserved for selecting all enabled {kind}s")
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise ConfigError(f"invalid {kind} id {value!r}; use 1–64 lowercase letters, digits, '_' or '-'")
    return value


def validate_model_id(model_id: str) -> str:
    return validate_id(model_id, "model")


def validate_target_id(target_id: str) -> str:
    return validate_id(target_id, "target")


def validate_framework_id(framework_id: str) -> str:
    if framework_id not in PLUGIN_SWITCHES:
        raise ConfigError(f"invalid framework id {framework_id!r}; use one of {', '.join(PLUGIN_SWITCHES)}")
    return framework_id


def selected_ids(selection: str, kind: str) -> list[str]:
    if not isinstance(selection, str):
        raise ConfigError(f"{kind} selection must be a comma-separated string or all")
    check = validate_framework_id if kind == "framework" else lambda value: validate_id(value, kind)
    selected = [check(value.strip()) for value in selection.split(",")]
    if len(set(selected)) != len(selected):
        raise ConfigError(f"select {kind}s without duplicates")
    return selected


def select_models(config: dict, selection: str = "all") -> list[str]:
    if not isinstance(config.get("models"), dict):
        raise ConfigError("agentx.yaml needs a models mapping")
    for model_id, entry in config["models"].items():
        validate_model_id(model_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("enabled", True), bool):
            raise ConfigError(f"models.{model_id} needs a mapping and a boolean enabled flag")
    enabled = [key for key, value in config["models"].items() if value.get("enabled", True)]
    selected = enabled if selection == "all" else selected_ids(selection, "model")
    if not selected:
        raise ConfigError("select at least one enabled model")
    for key in selected:
        if key not in enabled:
            raise ConfigError(f"unknown or disabled model {key!r}; enabled models: {', '.join(enabled)}")
    return selected


def select_targets(config: dict, model_id: str, selection: str = "all") -> list[str]:
    select_models(config, model_id)
    targets = config["models"][model_id].get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ConfigError(f"models.{model_id}.targets must define hardware, runner_label and frameworks")
    for target_id, entry in targets.items():
        validate_target_id(target_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("enabled", True), bool):
            raise ConfigError(
                f"models.{model_id}.targets.{target_id} needs a mapping and boolean enabled flag"
            )
    enabled = [key for key, value in targets.items() if value.get("enabled", True)]
    selected = enabled if selection == "all" else selected_ids(selection, "target")
    for key in selected:
        if key not in enabled:
            raise ConfigError(f"unknown or disabled target {model_id}/{key}; enabled: {', '.join(enabled)}")
    return selected


def select_frameworks(config: dict, model_id: str, target_id: str, selection: str = "all") -> list[str]:
    select_targets(config, model_id, target_id)
    frameworks = config["models"][model_id]["targets"][target_id].get("frameworks")
    location = f"models.{model_id}.targets.{target_id}"
    if not isinstance(frameworks, dict) or not frameworks:
        raise ConfigError(f"{location}.frameworks must map sglang or vllm to its recipe_source")
    for framework_id, entry in frameworks.items():
        validate_framework_id(framework_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("enabled", True), bool):
            raise ConfigError(
                f"{location}.frameworks.{framework_id} needs a mapping and boolean enabled flag"
            )
    enabled = [key for key, value in frameworks.items() if value.get("enabled", True)]
    selected = enabled if selection == "all" else selected_ids(selection, "framework")
    for key in selected:
        if key not in enabled:
            raise ConfigError(
                f"unknown or disabled framework {model_id}/{target_id}/{key}; enabled: {', '.join(enabled)}"
            )
    return selected


def _filter_runs(runs: list[tuple[str, ...]], selection: str, kind: str, index: int) -> list[tuple[str, ...]]:
    if selection == "all":
        return runs
    requested = selected_ids(selection, kind)
    missing = set(requested) - {run[index] for run in runs}
    if missing:
        raise ConfigError(f"unknown or disabled {kind}s in the selection: {', '.join(sorted(missing))}")
    return [run for run in runs if run[index] in requested]


def select_runs(
    config: dict, models: str = "all", targets: str = "all", frameworks: str = "all"
) -> list[tuple[str, str, str]]:
    runs = [
        (model_id, target_id, framework_id)
        for model_id in select_models(config, models)
        for target_id in select_targets(config, model_id)
        for framework_id in select_frameworks(config, model_id, target_id)
    ]
    runs = _filter_runs(_filter_runs(runs, targets, "target", 1), frameworks, "framework", 2)
    if not runs:
        raise ConfigError("select at least one enabled model/target/framework run")
    return runs


def _only(selected: list[str], kind: str, flag: str, scope: str) -> str:
    if len(selected) != 1:
        raise ConfigError(
            f"select exactly one {kind} for {scope} with {flag}; enabled: {', '.join(selected)}"
        )
    return selected[0]


def model_config(
    config: dict, model_id: str | None = None, target_id: str | None = None, framework_id: str | None = None
) -> dict:
    """Merge model settings with one target and one serving framework."""
    selected = select_models(config, model_id if model_id is not None else "all")
    if len(selected) != 1:
        raise ConfigError("select exactly one model with --model")
    model_id = selected[0]
    entry = config["models"][model_id]
    targets = select_targets(config, model_id, target_id if target_id is not None else "all")
    target_id = _only(targets, "target", "--target", model_id)
    target = entry["targets"][target_id]
    frameworks = select_frameworks(
        config, model_id, target_id, framework_id if framework_id is not None else "all"
    )
    framework_id = _only(frameworks, "framework", "--framework", f"{model_id}/{target_id}")
    serving = target["frameworks"][framework_id]
    location = f"models.{model_id}.targets.{target_id}.frameworks.{framework_id}"
    try:
        source = serving.get("recipe_source")
        if not isinstance(source, dict) or source.get("kind") not in RECIPE_SOURCE_KINDS:
            raise ConfigError(
                f"{location}.recipe_source.kind must be one of {', '.join(RECIPE_SOURCE_KINDS)}"
            )
        precision = entry.get("precision")
        if not isinstance(precision, str) or not precision.strip():
            raise ConfigError(f"models.{model_id}.precision must be a nonempty string")
        family = entry.get("family")
        if not isinstance(family, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", family):
            raise ConfigError(
                f"models.{model_id}.family must name its recipes/ directory: 1–64 letters, digits, '_' or '-'"
            )
        for field in ("repo", "revision"):
            if not entry["weights"].get(field):
                raise ConfigError(f"models.{model_id}.weights.{field} is required")
        runner = target.get("runner_label")
        if not isinstance(runner, str) or not runner.strip() or any(c in runner for c in "\0\n\r"):
            raise ConfigError(
                f"models.{model_id}.targets.{target_id}.runner_label must be a nonempty runner label"
            )
        return copy.deepcopy(
            {
                "model": {
                    "id": model_id,
                    "display_name": str(entry.get("display_name", model_id)),
                    "precision": precision.strip(),
                },
                "target": {"id": target_id, "runner_label": runner.strip()},
                "framework": {"id": framework_id},
                "family": family,
                "recipe_source": source,
                "weights": entry["weights"],
                "hardware": hardware_config(target.get("hardware")),
                "advisory_env": {
                    **validate_env(config.get("advisory_env", {})),
                    **validate_env(entry.get("advisory_env", {})),
                    **validate_env(target.get("advisory_env", {})),
                    **validate_env(serving.get("advisory_env", {})),
                },
                "report": config["report"],
            }
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ConfigError(
            f"incomplete configuration for {model_id}/{target_id}/{framework_id}: {error}"
        ) from error
