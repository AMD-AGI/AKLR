# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The e2e stage: each scenario's cookbook recipe served AKLR off and on, its answers graded."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from aklr.core.contract.export import read_calls
from aklr.core.platform.device import physical_device
from aklr.core.platform.env import (
    ENV_DISABLE,
    ENV_EXPORT,
    ENV_EXPORT_DIR,
    ENV_KERNEL_CACHE_DIR,
    ENV_MEASURED_DIR,
    ENV_MIN_SPEEDUP,
)
from aklr.core.validation.stages._e2e._config import Scenario
from aklr.core.validation.stages._e2e._cookbook import Cookbook
from aklr.core.validation.stages._e2e._deployment import Deployment, RunNotKilledError
from aklr.core.validation.stages._e2e._report import E2eRecord, E2eReport, SideResult
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from aklr.core.validation.stages.stage import PipelineStage

if TYPE_CHECKING:
    from aklr.core.validation.pipeline import PipelineReport
    from aklr.core.validation.stages.validation_scope import ValidationScope

__all__ = ["E2eStage"]

# phase 1, GSM8K on both sides: every test question
GSM8K_N_QUESTIONS: Final = 1319
# enough for a thinking model; others stop far earlier
MAX_TOKENS: Final = 16384
# provisional, about 2.5 sigma of the on - off difference over 1319 shared questions; to be
# set from measured off-against-off noise
GSM8K_MAX_DROP: Final = 0.02
# phase 2, the on side only: one prompt with a short, unique answer
PROMPT: Final = "What is the capital of France? Answer with one word."
EXPECTED_ANSWER: Final = "Paris"
# a side, from its start to its stop
SIDE_TIMEOUT_SECONDS: Final = 5400.0

Side = Literal["off", "on"]


class E2eStage(PipelineStage[E2eReport]):
    name = "e2e"
    report_type = E2eReport

    def __init__(
        self,
        scenarios: Sequence[Scenario],
        devices: Sequence[str],
        *,
        cookbook_repo: str,
        cookbook_source: str,
        hosts: Sequence[str] = ("127.0.0.1",),
        launch: str | None = None,
        work_dir: Path | None = None,
    ) -> None:
        if not scenarios:
            raise ValueError("e2e needs at least one scenario")
        if not devices:
            raise ValueError("e2e needs at least one device")
        if not hosts:
            raise ValueError("e2e needs at least one host")
        self._scenarios = tuple(scenarios)
        self._cookbook_repo = cookbook_repo
        self._cookbook_source = cookbook_source
        self._hosts = tuple(hosts)
        self._launch = launch
        self._work_dir = work_dir
        # every process gets these on its command line, so a remote node gets them too;
        # ROCR_VISIBLE_DEVICES is left as it is: HIP indexes within it
        visible = ",".join(map(physical_device, devices))
        self._env = {"HIP_VISIBLE_DEVICES": visible, "CUDA_VISIBLE_DEVICES": visible}
        if cache_dir := os.environ.get(ENV_KERNEL_CACHE_DIR):
            self._env[ENV_KERNEL_CACHE_DIR] = str(Path(cache_dir).absolute())
        if min_speedup := os.environ.get(ENV_MIN_SPEEDUP):  # on serves at the bench's threshold
            self._env[ENV_MIN_SPEEDUP] = min_speedup

    def run(self, scope: ValidationScope, earlier: PipelineReport) -> E2eReport:
        """Run every scenario; only a ``RunNotKilledError`` ends the stage."""
        findings: list[Finding] = []
        records: list[E2eRecord] = []
        cookbooks: dict[str, Cookbook] = {}
        source = self._cookbook_source
        try:
            installed_sglang = metadata.version("sglang")
        except metadata.PackageNotFoundError:
            installed_sglang = "none"
        with tempfile.TemporaryDirectory(prefix="aklr-validation-e2e-", dir=self._work_dir) as work_dir:
            # on serves this run's measurements over the committed schemas, from the checkout
            measured_dir = Path(work_dir, "measured")
            earlier.write_schemas(scope.root, measured_dir)
            python_path = os.pathsep.join(filter(None, (str(scope.root), os.environ.get("PYTHONPATH"))))
            env = self._env | {"PYTHONPATH": python_path, ENV_MEASURED_DIR: str(measured_dir)}
            for index, scenario in enumerate(self._scenarios):
                where = f"scenario {index + 1}/{len(self._scenarios)} {scenario.name}"
                try:
                    if scenario.cookbook not in cookbooks:
                        cookbooks[scenario.cookbook] = Cookbook.fetch(
                            source, scenario.cookbook, self._cookbook_repo
                        )
                        if not Path(source).is_dir():  # one commit for every scenario of the run
                            source = cookbooks[scenario.cookbook].sha
                    cookbook = cookbooks[scenario.cookbook]
                    # each role's cell cautions, and the sglang of the image the cookbook names
                    notes = [f"installed sglang {installed_sglang}"]
                    image_versions: set[str] = set()
                    for role, match in scenario.roles.items():
                        cell, image = cookbook.cell(match), cookbook.docker_image(match)
                        if version := re.search(r":v(\d+(?:\.\d+)+(?:\.post\d+)?)", image or ""):
                            image_versions.add(version[1])
                        notes += [f"{role}: cookbook image {image}"] if image else []
                        notes += [f"{role}: {cell.notes}"] if cell.notes else []
                except ValueError as error:
                    findings.append(
                        Finding.unlocated(FindingCode.PLATFORM_FAILURE, f"{scenario.name}: {error}")
                    )
                    records.append(E2eRecord(scenario=scenario.name, cookbook_sha=""))
                    self.progress.write(f"{where}: {error}")
                    continue
                self.progress.write(
                    f"{where} ({scenario.display_name}): cookbook {cookbook.sha[:12]}; {'; '.join(notes)}"
                )
                record = E2eRecord(scenario=scenario.name, cookbook_sha=cookbook.sha, notes="; ".join(notes))
                try:
                    run_dir = Path(work_dir, str(index))
                    record = record.model_copy(
                        update={"off": self._run_side(scenario, cookbook, "off", env, run_dir)}
                    )
                    if record.off.failure:
                        if image_versions - {installed_sglang}:
                            note = f"{record.notes}; off failed on a different sglang, a warning only"
                            record = record.model_copy(update={"notes": note})
                            self.progress.write(
                                f"{where}: warning: off failed: {record.off.failure.splitlines()[0]}"
                            )
                        else:
                            detail = f"{scenario.name} off: {record.off.failure}"
                            findings.append(Finding.unlocated(FindingCode.PLATFORM_FAILURE, detail))
                        records.append(record)
                        continue
                    record = record.model_copy(
                        update={"on": self._run_side(scenario, cookbook, "on", env, run_dir)}
                    )
                except RunNotKilledError as error:
                    findings.append(
                        Finding.unlocated(FindingCode.PLATFORM_FAILURE, f"{scenario.name}: {error}")
                    )
                    records.append(record)
                    self.progress.write(f"{where}: no result")
                    break
                on, difference = record.on, record.difference
                if on.failure:
                    findings.append(
                        Finding.unlocated(FindingCode.E2E_CRASH, f"{scenario.name} on: {on.failure}")
                    )
                if difference is not None and difference < -GSM8K_MAX_DROP:
                    detail = (
                        f"{scenario.name}: GSM8K on {on.gsm8k_score:.3f} is {-difference:.3f} below"
                        f" off {record.off.gsm8k_score:.3f}, more than {GSM8K_MAX_DROP}"
                    )
                    findings.append(Finding.unlocated(FindingCode.E2E_ACCURACY_DROP, detail))
                if on.answer_matches is False:
                    detail = f"{scenario.name}: answered {on.prompt_answer!r} to {PROMPT!r}, expected {EXPECTED_ANSWER!r}"
                    findings.append(Finding.unlocated(FindingCode.E2E_WRONG_ANSWER, detail))
                if on.n_hit_calls == 0:
                    record = record.model_copy(update={"notes": f"{record.notes}; AKLR served no call"})
                    self.progress.write(f"{where}: AKLR served no call, every call fell back")
                records.append(record)
                self.progress.write(f"{where}: {record.console_line()}")
        return E2eReport.from_findings(
            findings, prompt=PROMPT, expected_answer=EXPECTED_ANSWER, scenario_records=tuple(records)
        )

    def _run_side(
        self, scenario: Scenario, cookbook: Cookbook, side: Side, env: dict[str, str], run_dir: Path
    ) -> SideResult:
        """Serve ``scenario`` on ``side``, run GSM8K, and on the on side the prompt and the call
        count; a failure is the result's, only a ``RunNotKilledError`` raises."""
        calls_dir = run_dir / f"{side}-calls"
        side_env = {
            ENV_DISABLE: "1" if side == "off" else "0",
            ENV_EXPORT: "0" if side == "off" else "1",
            ENV_EXPORT_DIR: str(calls_dir),
        }
        try:
            deployment = Deployment.plan(
                scenario, cookbook, hosts=self._hosts, launch=self._launch, env=env, side_env=side_env,
                run_dir=run_dir / side,
            )  # fmt: skip
        except ValueError as error:
            return SideResult(failure=str(error))
        deadline = time.monotonic() + SIDE_TIMEOUT_SECONDS
        result: dict[str, object] = {"argvs": deployment.argvs}
        failures: list[str] = []
        self.progress.write(f"{side}: starting {len(deployment.processes)} processes")
        try:
            entry_url = deployment.start(timeout=deadline - time.monotonic())
            self.progress.write(f"{side}: serving at {entry_url}; GSM8K, {GSM8K_N_QUESTIONS} questions")
            argv = [
                sys.executable, "-m", "sglang.test.run_eval", "--eval-name", "gsm8k", "--base-url", entry_url,
                "--num-examples", str(GSM8K_N_QUESTIONS), "--max-tokens", str(MAX_TOKENS), "--temperature", "0",
            ]  # fmt: skip
            try:
                done = subprocess.run(
                    argv, capture_output=True, text=True, timeout=deadline - time.monotonic(), check=False
                )
                scores = re.findall(r"^Score: ([0-9.]+)$", done.stdout, re.M)
                if done.returncode != 0 or not scores:
                    raise RuntimeError(
                        f"run_eval exited with {done.returncode}, no score\n{done.stderr[-2000:]}"
                    )
                result["gsm8k_score"] = float(scores[-1])
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                failures.append(f"GSM8K: {error}")
            if side == "on":
                request = urllib.request.Request(
                    f"{entry_url}/v1/chat/completions",
                    data=json.dumps({
                        "model": scenario.model_path,
                        "messages": [{"role": "user", "content": PROMPT}],
                        "temperature": 0,
                        "max_tokens": MAX_TOKENS,
                    }).encode(),
                    headers={"Content-Type": "application/json"},
                )  # fmt: skip
                try:
                    with urllib.request.urlopen(
                        request, timeout=max(1.0, deadline - time.monotonic())
                    ) as response:
                        answer = json.load(response)["choices"][0]["message"]["content"] or ""
                    result["prompt_answer"] = answer
                    # the whole answer, less surrounding whitespace and trailing punctuation
                    result["answer_matches"] = (
                        re.sub(r"\W+$", "", answer).strip().casefold() == EXPECTED_ANSWER.casefold()
                    )
                except (OSError, ValueError, LookupError, TypeError) as error:
                    failures.append(f"prompt: {type(error).__name__}: {error}")
        except (OSError, RuntimeError) as error:
            failures.append(f"start: {error}")
        finally:
            log_tails = deployment.stop()
        if side == "on":
            try:
                calls = read_calls(calls_dir)
                result["n_hit_calls"] = sum(c.n_calls for c in calls if c.fallback_reason is None)
                result["n_fallback_calls"] = sum(c.n_calls for c in calls if c.fallback_reason is not None)
            except (OSError, ValueError) as error:
                failures.append(f"call files: {error}")
        if failures:
            result["failure"] = "\n".join([*failures, log_tails])
        return SideResult.model_validate(result)
