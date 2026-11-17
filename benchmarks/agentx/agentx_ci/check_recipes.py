# SPDX-License-Identifier: MIT
"""Check every recipe under recipes/ against serving.schema.json and agentx.yaml, without Docker or a registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema
import yaml

from agentx_ci import ci_image
from agentx_ci import serving_config as sc
from agentx_ci.model_config import ConfigError, model_config
from agentx_ci.utils import DEFAULT_CONFIG, load_config

LAYOUT = "<family>/<target>/<framework>/serving.<model>.json"


def recipe_errors(
    path: Path, config_path: Path, config: dict, validator: jsonschema.Validator, images: dict[str, str]
) -> list[str]:
    """Return the rule violations of one file under recipes/; empty when plan would accept it."""
    parts = path.relative_to(config_path.parent / "recipes").parts
    if len(parts) != 4 or not (parts[3].startswith("serving.") and parts[3].endswith(".json")):
        return [f"recipes/ holds only {LAYOUT} files"]
    _, target_id, framework_id, name = parts
    model_id = name.removeprefix("serving.").removesuffix(".json")
    try:
        definition = model_config(config, model_id, target_id, framework_id)
    except ConfigError as error:
        return [f"no enabled agentx.yaml run {model_id}/{target_id}/{framework_id}: {error}"]
    expected = sc.recipe_path(config_path, definition)
    if expected != path:
        return [f"agentx.yaml files this run's recipe at {expected.relative_to(config_path.parent)}"]
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"invalid JSON: {error}"]
    errors = [
        f"schema: {'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in sorted(
            validator.iter_errors(recipe), key=lambda error: list(map(str, error.absolute_path))
        )
    ]
    try:
        sc.build_run(recipe, definition, images[framework_id])
    except sc.RecipeError as error:
        errors.append(f"plan: {error}")
    except (TypeError, KeyError, AttributeError):
        if not errors:
            raise
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentx_bench check-recipes", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="agentx.yaml beside recipes/")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    try:
        schema = json.loads(sc.RECIPE_SCHEMA.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        sc.template()
        config = load_config(config_path)
        images = ci_image.ci_images()
    except (OSError, ValueError, yaml.YAMLError, jsonschema.SchemaError) as error:
        print(f"AgentX recipe check cannot start: {error}", file=sys.stderr)
        return 1
    validator = jsonschema.Draft202012Validator(schema)
    recipes = config_path.parent / "recipes"
    paths = sorted(
        path for path in recipes.rglob("*") if path.is_file() and path != recipes / sc.RECIPE_SCHEMA.name
    )
    failed = 0
    for path in paths:
        errors = recipe_errors(path, config_path, config, validator, images)
        failed += bool(errors)
        for error in errors:
            print(f"{path.relative_to(config_path.parent)}: {error}", file=sys.stderr)
    print(f"AgentX recipes: {len(paths) - failed} of {len(paths)} files pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
