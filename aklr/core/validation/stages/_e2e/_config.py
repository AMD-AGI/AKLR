# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The e2e configuration file: the cookbook recipes each scenario serves, and how."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import Field, PositiveInt

from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._e2e._cookbook import COOKBOOK_REPO, MatchValue

__all__ = ["DEFAULT_CONFIG_PATH", "PD_ROLES", "UNIFIED_ROLE", "ArgValue", "E2eConfig", "Scenario"]

DEFAULT_CONFIG_PATH: Final = Path(__file__).with_name("scenarios.yaml")
UNIFIED_ROLE: Final = "unified"
PD_ROLES: Final = ("prefill", "decode")

# a server flag's override: a value, a switch (true), or removed (null or false)
ArgValue = str | int | float | bool | None


class Scenario(FrozenModel):
    """One deployment shape of one model, each role from its own cookbook cell."""

    name: str
    model: str
    display_name: str
    enabled: bool
    cookbook: str
    # ``unified``, or ``prefill`` then ``decode``: the cell match of each role
    roles: dict[str, dict[str, MatchValue]]
    model_path: str
    revision: str
    nnodes: dict[str, PositiveInt]  # by role; a role not named has its cell's
    env: dict[str, dict[str, str]]  # by role
    args: dict[str, dict[str, ArgValue]]  # by role


class _Weights(FrozenModel):
    repo: str
    # required: the launcher always sets --revision, so a cell's cannot choose the weights' code
    revision: str = Field(min_length=1)


class _CookbookEntry(FrozenModel):
    config: str
    matches: tuple[dict[str, Any], ...] = Field(min_length=1)


class _ModelEntry(FrozenModel):
    enabled: bool = True
    display_name: str = ""
    cookbook: _CookbookEntry
    weights: _Weights
    # a cell may leave it out (GLM-5.2's multi-2 cells): one count, or one per PD role
    nnodes: PositiveInt | dict[str, PositiveInt] | None = None
    env: dict[str, Any] = {}  # noqa: RUF012
    args: dict[str, Any] = {}  # noqa: RUF012


class _CookbookSource(FrozenModel):
    repo: str = COOKBOOK_REPO
    ref: str = "main"


class _ConfigFile(FrozenModel):
    cookbook: _CookbookSource = _CookbookSource()
    models: dict[str, _ModelEntry] = Field(min_length=1)


class E2eConfig(FrozenModel):
    """The cookbook to read and the scenarios of an e2e configuration file, in its order."""

    cookbook_repo: str
    cookbook_ref: str
    scenarios: tuple[Scenario, ...]

    @classmethod
    def read(cls, path: Path = DEFAULT_CONFIG_PATH) -> E2eConfig:
        """Read the YAML file at ``path``, one scenario per item of a model's ``matches``."""
        import yaml
        from pydantic import ValidationError

        try:
            document = _ConfigFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (OSError, yaml.YAMLError, ValidationError) as error:
            raise ValueError(f"{path}: {error}") from error
        scenarios: list[Scenario] = []
        for model, entry in document.models.items():
            for index, match in enumerate(entry.cookbook.matches):
                where = f"{path}: {model}: match {index}"
                is_pd = set(match) == set(PD_ROLES) and all(isinstance(m, dict) for m in match.values())
                roles = {role: match[role] for role in PD_ROLES} if is_pd else {UNIFIED_ROLE: match}
                nnodes = (
                    entry.nnodes if isinstance(entry.nnodes, dict) else dict.fromkeys(roles, entry.nnodes)
                )
                if entry.nnodes is None:
                    nnodes = {}
                if unknown := set(nnodes) - set(roles):
                    raise ValueError(f"{where}: nnodes of roles it has not: {sorted(unknown)}")
                values = [str(v) for role_match in roles.values() for v in role_match.values()]
                try:
                    scenario = Scenario(
                        name="-".join([model, *values]),
                        model=model,
                        display_name=entry.display_name or model,
                        enabled=entry.enabled,
                        cookbook=entry.cookbook.config,
                        roles=roles,
                        model_path=entry.weights.repo,
                        revision=entry.weights.revision,
                        nnodes=nnodes,
                        env={
                            role: {k: str(v) for k, v in env.items()}
                            for role, env in cls._by_role(entry.env, tuple(roles)).items()
                        },
                        args=cls._by_role(entry.args, tuple(roles)),
                    )
                except (ValueError, ValidationError) as error:
                    raise ValueError(f"{where}: {error}") from error
                if any(s.name == scenario.name for s in scenarios):
                    raise ValueError(f"{where}: the scenario name {scenario.name} is repeated")
                scenarios.append(scenario)
        return cls(
            cookbook_repo=document.cookbook.repo,
            cookbook_ref=document.cookbook.ref,
            scenarios=tuple(scenarios),
        )

    @staticmethod
    def _by_role(value: Mapping[str, Any], roles: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """``env`` or ``args`` per role: written per PD role, or one mapping for every role."""
        is_per_role = (
            bool(value) and set(value) <= set(PD_ROLES) and all(isinstance(v, dict) for v in value.values())
        )
        if not is_per_role:
            return dict.fromkeys(roles, dict(value))
        if roles != PD_ROLES:
            raise ValueError(f"{sorted(value)}: per-role settings of a scenario with no PD roles")
        return {role: dict(value.get(role, {})) for role in roles}

    def selected(self, names: Sequence[str]) -> tuple[Scenario, ...]:
        """The scenarios named, or of the models named; with no name, every enabled one."""
        if unknown := [n for n in names if not any(n in (s.name, s.model) for s in self.scenarios)]:
            raise ValueError(f"no e2e scenario or model named {unknown}")
        if not names:
            return tuple(s for s in self.scenarios if s.enabled)
        return tuple(s for s in self.scenarios if s.name in names or s.model in names)
