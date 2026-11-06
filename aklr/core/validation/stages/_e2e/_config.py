# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The e2e configuration file: the cookbook recipes each scenario serves, and how."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, PositiveInt

from aklr.core.contract.spec import FrozenModel
from aklr.core.validation.stages._e2e._cookbook import MatchValue

__all__ = ["E2eConfig", "Scenario"]


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
    args: dict[str, dict[str, str | int | float | bool | None]]  # by role


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
    repo: str = "sgl-project/sglang"
    source: str = "main"


class _ConfigFile(FrozenModel):
    cookbook: _CookbookSource = _CookbookSource()
    hosts: tuple[str, ...] = Field(default=("127.0.0.1",), min_length=1)
    work_dir: Path | None = None
    models: dict[str, _ModelEntry] = Field(min_length=1)


class E2eConfig(FrozenModel):
    """The cookbook to read and the scenarios of an e2e configuration file, in its order."""

    cookbook_repo: str
    cookbook_source: str
    hosts: tuple[str, ...] = ("127.0.0.1",)
    work_dir: Path | None = None
    scenarios: tuple[Scenario, ...]

    @classmethod
    def read(cls, path: Path | None = None) -> E2eConfig:
        """Read the YAML file at ``path``, one scenario per item of a model's ``matches``."""
        import yaml
        from pydantic import ValidationError

        path = path or Path(__file__).with_name("scenarios.yaml")
        pd_roles = ("prefill", "decode")
        try:
            document = _ConfigFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (OSError, yaml.YAMLError, ValidationError) as error:
            raise ValueError(f"{path}: {error}") from error
        scenarios: list[Scenario] = []
        for model, entry in document.models.items():
            for index, match in enumerate(entry.cookbook.matches):
                where = f"{path}: {model}: match {index}"
                is_pd = set(match) == set(pd_roles) and all(isinstance(m, dict) for m in match.values())
                roles = {role: match[role] for role in pd_roles} if is_pd else {"unified": match}
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
                            for role, env in cls._expand_server_settings(entry.env, tuple(roles)).items()
                        },
                        args=cls._expand_server_settings(entry.args, tuple(roles)),
                    )
                except (ValueError, ValidationError) as error:
                    raise ValueError(f"{where}: {error}") from error
                if any(s.name == scenario.name for s in scenarios):
                    raise ValueError(f"{where}: the scenario name {scenario.name} is repeated")
                scenarios.append(scenario)
        return cls(
            cookbook_repo=document.cookbook.repo,
            cookbook_source=document.cookbook.source,
            hosts=document.hosts,
            work_dir=document.work_dir,
            scenarios=tuple(scenarios),
        )

    @staticmethod
    def _expand_server_settings(
        value: Mapping[str, Any], roles: tuple[str, ...]
    ) -> dict[str, dict[str, Any]]:
        """Expand shared or separate prefill/decode settings into each server's env or args."""
        pd_roles = ("prefill", "decode")
        is_per_role = (
            bool(value) and set(value) <= set(pd_roles) and all(isinstance(v, dict) for v in value.values())
        )
        if not is_per_role:
            return dict.fromkeys(roles, dict(value))
        if roles != pd_roles:
            raise ValueError(f"{sorted(value)}: per-role settings of a scenario with no PD roles")
        return {role: dict(value.get(role, {})) for role in roles}

    @property
    def enabled_scenarios(self) -> tuple[Scenario, ...]:
        """Every scenario whose model is enabled in scenarios.yaml."""
        return tuple(s for s in self.scenarios if s.enabled)
