# SPDX-License-Identifier: MIT
"""Select targets and merge shared settings."""

from __future__ import annotations

import copy
import re


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


def selected_ids(selection: str, kind: str) -> list[str]:
    if not isinstance(selection, str):
        raise ConfigError(f"{kind} selection must be a comma-separated string or all")
    selected = [validate_id(value.strip(), kind) for value in selection.split(",")]
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
        raise ConfigError(f"models.{model_id}.targets must define hardware, runner_label and inferencex")
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


def select_runs(config: dict, models: str = "all", targets: str = "all") -> list[tuple[str, str]]:
    pairs = [
        (model_id, target_id)
        for model_id in select_models(config, models)
        for target_id in select_targets(config, model_id)
    ]
    if targets != "all":
        requested = selected_ids(targets, "target")
        available = {target_id for _, target_id in pairs}
        missing = set(requested) - available
        if missing:
            raise ConfigError(f"unknown or disabled targets in selected models: {', '.join(sorted(missing))}")
        pairs = [pair for pair in pairs if pair[1] in requested]
    if not pairs:
        raise ConfigError("select at least one enabled model target")
    return pairs


def model_config(config: dict, model_id: str | None = None, target_id: str | None = None) -> dict:
    """Merge model settings with one target."""
    selected = select_models(config, model_id if model_id is not None else "all")
    if len(selected) != 1:
        raise ConfigError("select exactly one model with --model")
    model_id = selected[0]
    entry = config["models"][model_id]
    targets = select_targets(config, model_id, target_id if target_id is not None else "all")
    if len(targets) != 1:
        raise ConfigError(
            f"select exactly one target for {model_id} with --target; enabled: {', '.join(targets)}"
        )
    target_id = targets[0]
    target = entry["targets"][target_id]
    location = f"models.{model_id}.targets.{target_id}"
    try:
        source = {**config["inferencex"], **target["inferencex"]}
        for field in ("repo", "ref", "config_key", "expect"):
            if not source.get(field):
                raise ConfigError(f"{location}.inferencex.{field} is required")
        for field in ("repo", "revision"):
            if not entry["weights"].get(field):
                raise ConfigError(f"models.{model_id}.weights.{field} is required")
        runner = target.get("runner_label")
        if not isinstance(runner, str) or not runner.strip() or any(c in runner for c in "\0\n\r"):
            raise ConfigError(f"{location}.runner_label must be a nonempty runner label")
        image = target.get("run_image")
        if image is not None and (
            not isinstance(image, str)
            or not image.strip()
            or any(c.isspace() for c in image)
            or "\0" in image
        ):
            raise ConfigError(f"{location}.run_image must be a nonempty image reference without whitespace")
        return copy.deepcopy(
            {
                "model": {
                    "id": model_id,
                    "display_name": str(entry.get("display_name", model_id)),
                    "precision": source["expect"]["precision"],
                },
                "target": {"id": target_id, "runner_label": runner.strip()},
                "inferencex": source,
                "weights": entry["weights"],
                "hardware": hardware_config(target.get("hardware")),
                "run_image": image,
                "advisory_env": {
                    **validate_env(config.get("advisory_env", {})),
                    **validate_env(entry.get("advisory_env", {})),
                    **validate_env(target.get("advisory_env", {})),
                },
                "report": config["report"],
            }
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ConfigError(f"incomplete configuration for target {model_id}/{target_id}: {error}") from error
