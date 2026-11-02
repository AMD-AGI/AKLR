###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, TypeAdapter

from aklr.core._frozen_model import frozen_dataclass

__all__ = [
    "DEFAULT_CONFIG_PATH", "SCENARIO_DIR", "E2eConfig", "E2eSettings", "Scenario",
]  # fmt: skip

DEFAULT_CONFIG_PATH: Final = Path(__file__).with_name("scenarios.yaml")
# the scripts must import this checkout's aklr, hence a path from the package's location
SCENARIO_DIR: Final = Path(__file__).resolve().parents[4] / "benchmarks" / "models"


@frozen_dataclass
class E2eSettings:
    n_runs_per_side: int = Field(ge=2)  # ratio_lower_bound needs two rounds
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    min_gain_fraction: float = Field(ge=0, allow_inf_nan=False)  # e2e accepts no regression

    @property
    def min_ratio(self) -> float:
        """The on/off throughput ratio's lower bound a scenario must reach."""
        return 1 + self.min_gain_fraction


@frozen_dataclass
class Scenario:
    name: str
    model: str
    phase: Literal["decode", "prefill"]
    settings: E2eSettings
    model_path: str | None = None
    tp: int | None = None
    dp: int | None = None
    ep: int | None = None
    input_length: int | None = None
    concurrency: int | None = None
    extra_args: dict[str, str | int | float | bool] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)

    def argv(self, result_json_path: Path) -> list[str]:
        argv: list[str] = [str(SCENARIO_DIR / "main.py"), self.model, self.phase]
        for flag, value in (
            ("--model-path", self.model_path),
            ("--tp-size", self.tp),
            ("--dp-size", self.dp),
            ("--ep-size", self.ep),
            ("--ISL", self.input_length),
            ("--CONC", self.concurrency),
        ):
            if value is not None:
                argv += [flag, str(value)]
        for name, value in self.extra_args.items():
            if value is True:
                argv.append(f"--{name}")
            elif value is not False:
                argv += [f"--{name}", str(value)]
        return [*argv, "--result-json", str(result_json_path)]


@frozen_dataclass
class E2eConfig:
    scenarios: tuple[Scenario, ...]

    @classmethod
    def read(cls, path: Path = DEFAULT_CONFIG_PATH) -> E2eConfig:
        """The YAML file's scenarios, defaults applied; any defect in the file is a ValueError."""
        import yaml
        from pydantic import ValidationError

        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"{path}: {error}") from error
        if not isinstance(document, dict) or set(document) != {
            "defaults",
            "scenarios",
        }:
            raise ValueError(f"{path}: not a mapping of 'defaults' and 'scenarios'")
        defaults, entries = document["defaults"], document["scenarios"]
        if not isinstance(defaults, dict) or not isinstance(entries, list):
            raise ValueError(  # noqa: TRY004 (the docstring's ValueError)
                f"{path}: 'defaults' is not a mapping or 'scenarios' not a list"
            )
        settings_names = {f.name for f in dataclasses.fields(E2eSettings)}
        try:
            TypeAdapter(E2eSettings).validate_python(defaults)
        except ValidationError as error:
            raise ValueError(f"{path}: defaults: {error}") from error
        scenarios: list[Scenario] = []
        for index, entry in enumerate(entries):
            name = entry.get("name") if isinstance(entry, dict) else None
            where = f"{path}: scenario {name if name is not None else index}"
            if not isinstance(entry, dict):
                raise ValueError(f"{where}: not a mapping")  # noqa: TRY004
            if "settings" in entry:
                raise ValueError(f"{where}: unknown key 'settings'")
            overrides = {k: v for k, v in entry.items() if k in settings_names}
            fields = {k: v for k, v in entry.items() if k not in settings_names}
            try:
                settings = TypeAdapter(E2eSettings).validate_python(defaults | overrides)
                scenario = TypeAdapter(Scenario).validate_python(fields | {"settings": settings})
            except ValidationError as error:
                raise ValueError(f"{where}: {error}") from error
            if any(s.name == scenario.name for s in scenarios):
                raise ValueError(f"{where}: the name is repeated")
            if not (SCENARIO_DIR / scenario.model / "model_bench.py").is_file():
                raise ValueError(f"{where}: no model {scenario.model}")
            scenarios.append(scenario)
        return cls(scenarios=tuple(scenarios))

    def selected(self, names: Sequence[str]) -> tuple[Scenario, ...]:
        if missing := set(names) - {s.name for s in self.scenarios}:
            raise ValueError(f"no e2e scenario named {sorted(missing)}")
        if not names:
            return self.scenarios
        return tuple(s for s in self.scenarios if s.name in names)
