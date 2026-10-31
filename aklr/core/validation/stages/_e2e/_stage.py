# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import dataclasses
import math
import os
import secrets
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from aklr.core.contract.export import read_calls
from aklr.core.platform.device import physical_device
from aklr.core.platform.env import (
    ENV_DISABLE,
    ENV_EXPORT,
    ENV_EXPORT_DIR,
    ENV_KERNEL_CACHE_DIR,
    ENV_MEASURED_DIR,
)
from aklr.core.validation.stages._e2e._config import SCENARIO_DIR, Scenario
from aklr.core.validation.stages._e2e._report import E2eRecord, E2eReport
from aklr.core.validation.stages._e2e._run import (
    RunNotKilledError,
    RunResult,
    ScenarioRun,
)
from aklr.core.validation.stages._measure import ratio_lower_bound
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from aklr.core.validation.stages.stage import PipelineStage
from aklr.core.validation.stages.validation_scope import ValidationScope

if TYPE_CHECKING:
    from aklr.core.validation.pipeline import PipelineReport

__all__ = ["E2eStage"]


class E2eStage(PipelineStage[E2eReport]):
    name = "e2e"
    report_type = E2eReport

    def __init__(self, scenarios: Sequence[Scenario], devices: Sequence[str]) -> None:
        if not scenarios:
            raise ValueError("e2e needs at least one scenario")
        if not devices:
            raise ValueError("e2e needs at least one device")
        self._scenarios = tuple(scenarios)
        # dropped from the parent's environment; each run sets its own (AKLR_MIN_SPEEDUP
        # stays: the runs serve at the bench's threshold)
        aklr_env_names = (ENV_DISABLE, ENV_EXPORT, ENV_EXPORT_DIR, ENV_MEASURED_DIR)
        env = {k: v for k, v in os.environ.items() if k not in aklr_env_names}
        # ROCR_VISIBLE_DEVICES is left as it is: HIP indexes within it (E11)
        visible = ",".join(map(physical_device, devices))
        env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"] = visible
        # SCENARIO_DIR is <checkout>/aklr/benchmarks/models
        paths = (str(SCENARIO_DIR.parents[2]), os.environ.get("PYTHONPATH"))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, paths))
        if cache_dir := env.get(ENV_KERNEL_CACHE_DIR):
            env[ENV_KERNEL_CACHE_DIR] = str(Path(cache_dir).absolute())
        self._env = env

    def run(self, scope: ValidationScope, earlier: PipelineReport) -> E2eReport:
        """Run every scenario; only a ``RunNotKilledError`` ends the stage."""
        findings: list[Finding] = []
        records: list[E2eRecord] = []
        with tempfile.TemporaryDirectory(prefix="aklr-validation-e2e-") as work_dir:
            # the runs' dispatch reads this run's measurements over the committed schemas,
            # so "on" serves what it will once they land; the checkout stays clean (a
            # copy of aklr on PYTHONPATH would lose to the checkout aiter puts first)
            measured_dir = Path(work_dir, "measured")
            n_rewritten = earlier.write_schemas(scope.root, measured_dir)
            self.progress.write(
                f"serving {n_rewritten} schemas rewritten with this run's measurements,"
                " the others as committed"
            )
            env = self._env | {ENV_MEASURED_DIR: str(measured_dir)}
            for index, scenario in enumerate(self._scenarios):
                where = f"scenario {index + 1}/{len(self._scenarios)} {scenario.name}"
                settings = scenario.settings
                self.progress.write(
                    f"{where}: {settings.n_runs_per_side} rounds of on/off runs"
                    f" + 1 export run, up to {settings.timeout_seconds:g} s per run,"
                    f" needs ratio lower bound >= {settings.min_ratio:.4f}"
                )
                try:
                    scenario_findings, record = self._bench_scenario(
                        scenario, Path(work_dir) / str(index), env
                    )
                except RunNotKilledError as error:
                    findings.append(
                        Finding.unlocated(FindingCode.PLATFORM_FAILURE, f"{scenario.name}: {error}")
                    )
                    self.progress.write(f"{where}: no result")
                    break
                findings += scenario_findings
                records += [record] if record else []
                self.progress.write(
                    f"{where}: {record.console_line(settings.min_ratio) if record else 'no result'}"
                )
        return E2eReport.from_findings(findings, scenario_records=tuple(records))

    def _bench_scenario(
        self, scenario: Scenario, work_dir: Path, env: Mapping[str, str]
    ) -> tuple[list[Finding], E2eRecord | None]:
        rounds = self._compare_rounds(scenario, work_dir, env)
        if isinstance(rounds, Finding):
            return [rounds], None
        findings: list[Finding] = []
        settings = scenario.settings
        ratio = ratio_lower_bound([math.log(on.tokens_per_s / off.tokens_per_s) for on, off in rounds])
        if not ratio >= settings.min_ratio:
            detail = (
                f"{scenario.name}: tokens/s on/off ratio lower bound {ratio:.4f}, "
                f"below {settings.min_ratio:.4f}"
            )
            findings.append(Finding.unlocated(FindingCode.E2E_NOT_FASTER, detail))
        run = ScenarioRun(scenario=scenario, side="export", run_dir=work_dir / "export-run", env={})
        # as the "on" runs
        export_env = {ENV_EXPORT: "1", ENV_EXPORT_DIR: str(run.calls_dir)}
        run = dataclasses.replace(run, env=env | scenario.env | export_env)
        self.progress.write("export run: counting hit and fallback calls")
        if failure := run.execute():
            detail = f"{scenario.name} on, export run: {failure}"
            return [*findings, Finding.unlocated(FindingCode.E2E_CRASH, detail)], None
        try:
            calls = read_calls(run.calls_dir)
        except (ValueError, OSError) as error:
            detail = f"internal: {scenario.name}: call files: {error}"
            finding = Finding.unlocated(FindingCode.PLATFORM_FAILURE, detail)
            return [*findings, finding], None
        n_hit_calls = sum(c.n_calls for c in calls if c.fallback_reason is None)
        n_fallback_calls = sum(c.n_calls for c in calls if c.fallback_reason is not None)
        record = E2eRecord(
            scenario=scenario.name,
            n_runs_per_side=scenario.settings.n_runs_per_side,
            throughput_ratio_lower_bound=ratio,
            latency_ratio_lower_bound=ratio_lower_bound(
                [math.log(off.latency_ms / on.latency_ms) for on, off in rounds]
            ),
            n_hit_calls=n_hit_calls,
            n_fallback_calls=n_fallback_calls,
        )
        return findings, record

    def _compare_rounds(  # structure: allow S4 patched in latency comparison tests
        self, scenario: Scenario, work_dir: Path, env: Mapping[str, str]
    ) -> list[tuple[RunResult, RunResult]] | Finding:
        """Per round: on's and off's throughput and latency."""
        side_envs = {"on": {}, "off": {ENV_DISABLE: "1"}}
        rounds: list[tuple[RunResult, RunResult]] = []
        n = scenario.settings.n_runs_per_side
        for round_index in range(n):
            sides = ["on", "off"]
            secrets.SystemRandom().shuffle(sides)
            self.progress.write(f"round {round_index}/{n}: running {', '.join(sides)}")
            runs = {
                side: ScenarioRun(
                    scenario=scenario,
                    side=side,
                    run_dir=work_dir / f"{round_index}-{side}",
                    env=env | scenario.env | side_envs[side],
                )
                for side in sides
            }
            failures: dict[str, str | None] = {}
            for side in sides:
                failures[side] = runs[side].execute()
                if side == "off" and failures["off"]:
                    break
            if failures["off"]:
                detail = f"{scenario.name} off, round {round_index}: {failures['off']}"
                return Finding.unlocated(FindingCode.PLATFORM_FAILURE, detail)
            if failures["on"]:
                detail = f"{scenario.name} on, round {round_index}: {failures['on']}"
                return Finding.unlocated(FindingCode.E2E_CRASH, detail)
            try:
                on, off = runs["on"].result(), runs["off"].result()
            except ValueError as error:
                detail = f"{scenario.name} round {round_index}: {error}"
                return Finding.unlocated(FindingCode.PLATFORM_FAILURE, detail)
            self.progress.write(
                f"round {round_index}/{n}: on {on.tokens_per_s:.1f} tokens/s,"
                f" latency {on.latency_ms:.3f} ms;"
                f" off {off.tokens_per_s:.1f} tokens/s,"
                f" latency {off.latency_ms:.3f} ms,"
                f" ratio {on.tokens_per_s / off.tokens_per_s:.4f}"
            )
            rounds.append((on, off))
            for run in runs.values():
                shutil.rmtree(run.run_dir)
        return rounds
