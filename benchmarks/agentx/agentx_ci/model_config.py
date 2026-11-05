# SPDX-License-Identifier: MIT
"""Model selection and AKLR-specific environment overrides."""

from __future__ import annotations

import copy
import re


class ConfigError(ValueError):
    pass


def validate_model_id(model_id: str) -> str:
    if model_id == "all":
        raise ConfigError("model id 'all' is reserved for selecting all enabled models")
    if not isinstance(model_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", model_id):
        raise ConfigError(f"invalid model id {model_id!r}; use 1–64 lowercase letters, digits, '_' or '-'")
    return model_id


def select_models(config: dict, selection: str = "all") -> list[str]:
    if not isinstance(config.get("models"), dict):
        raise ConfigError("agentx.yaml needs a models mapping")
    for model_id, entry in config["models"].items():
        validate_model_id(model_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("enabled", True), bool):
            raise ConfigError(f"models.{model_id} needs a mapping and a boolean enabled flag")
    enabled = [key for key, value in config["models"].items() if value.get("enabled", True)]
    selected = enabled if selection == "all" else [s.strip() for s in selection.split(",")]
    if not selected or len(set(selected)) != len(selected):
        raise ConfigError("select at least one model, without duplicates")
    for key in selected:
        if key not in enabled:
            raise ConfigError(f"unknown or disabled model {key!r}; enabled models: {', '.join(enabled)}")
    return selected


def model_config(config: dict, model_id: str | None = None) -> dict:
    """Return one complete definition; model identity and weights are never inherited."""
    selected = select_models(config, model_id if model_id is not None else "all")
    if len(selected) != 1:
        raise ConfigError("select exactly one model with --model")
    model_id = selected[0]
    entry = config["models"][model_id]
    try:
        source = {**config["inferencex"], **entry["inferencex"]}
        for field in ("repo", "ref", "config_key", "expect"):
            if not source.get(field):
                raise ConfigError(f"models.{model_id}.inferencex.{field} is required")
        for field in ("repo", "revision"):
            if not entry["weights"].get(field):
                raise ConfigError(f"models.{model_id}.weights.{field} is required")
        return copy.deepcopy(
            {
                "model": {
                    "id": model_id,
                    "display_name": str(entry.get("display_name", model_id)),
                    "precision": source["expect"]["precision"],
                },
                "inferencex": source,
                "weights": entry["weights"],
                "advisory_env": {**config.get("advisory_env", {}), **entry.get("advisory_env", {})},
                "report": config["report"],
            }
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ConfigError(f"incomplete configuration for model {model_id}: {error}") from error
