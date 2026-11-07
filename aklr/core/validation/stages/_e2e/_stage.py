# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The e2e stage: each scenario's cookbook recipe served AKLR off and on, its answers graded."""

from __future__ import annotations

import http.client
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from importlib import metadata
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
    ENV_MIN_SPEEDUP,
)
from aklr.core.validation.stages._e2e._command import export_commands
from aklr.core.validation.stages._e2e._config import E2eConfig, Scenario
from aklr.core.validation.stages._e2e._cookbook import Cookbook
from aklr.core.validation.stages._e2e._deployment import Deployment, RunNotKilledError
from aklr.core.validation.stages._e2e._report import E2eRecord, E2eReport, ModelEvaluationResult
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from aklr.core.validation.stages.stage import PipelineStage

if TYPE_CHECKING:
    from aklr.core.validation.pipeline import PipelineReport
    from aklr.core.validation.stages.validation_scope import ValidationScope

__all__ = ["E2eStage"]


class E2eStage(PipelineStage[E2eReport]):
    name = "e2e"
    report_type = E2eReport

    def __init__(self, devices: Sequence[str]) -> None:
        # phase 1, GSM8K on both sides: every test question; run_eval takes its 5 few-shot examples
        # from them, so 1314 are graded
        self._gsm8k_questions = 1319
        # concurrent requests (run_eval's default is 512: a long-thinking model at 16384 tokens then
        # queues past its client's retries); failed requests tolerated, each scored wrong
        self._gsm8k_concurrency = 64
        self._max_failed_requests = 2
        # enough for a thinking model; others stop far earlier
        self._max_tokens = 16384
        # provisional, about 2.5 sigma of the on - off difference over 1314 shared questions; to be
        # set from measured off-against-off noise
        self._max_accuracy_drop = 0.02
        # below it, off is not the model answering (a broken recipe or grading), and on is not run
        self._min_baseline_accuracy = 0.5
        # a side, from its start to its stop, and of it the start (weights, graph capture)
        self._evaluation_timeout = 5400.0
        self._startup_timeout = 2400.0
        config = E2eConfig.read()
        # phase 2, the on side only: the configured prompt and its short, unique answer
        self._prompt = config.prompt
        self._expected_answer = config.expected_answer
        self._scenarios = config.enabled_scenarios
        if self._scenarios and not devices:
            raise ValueError("e2e needs at least one device")
        self._cookbook_repo = config.cookbook_repo
        self._cookbook_source = config.cookbook_source
        self._hosts = config.hosts
        self._work_dir = config.work_dir
        # every process gets these on its command line, so a remote node gets them too;
        # ROCR_VISIBLE_DEVICES is left as it is: HIP indexes within it
        visible = ",".join(map(physical_device, devices))
        self._env = {"HIP_VISIBLE_DEVICES": visible, "CUDA_VISIBLE_DEVICES": visible}
        if cache_dir := os.environ.get(ENV_KERNEL_CACHE_DIR):
            self._env[ENV_KERNEL_CACHE_DIR] = str(Path(cache_dir).absolute())
        if min_speedup := os.environ.get(ENV_MIN_SPEEDUP):  # on serves at the bench's threshold
            self._env[ENV_MIN_SPEEDUP] = min_speedup
        # where the weights are, and that they are not downloaded: a remote node inherits nothing
        self._env |= {k: v for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE") if (v := os.environ.get(k))}

    def run(self, scope: ValidationScope, earlier: PipelineReport) -> E2eReport:
        """Run every scenario; only a ``RunNotKilledError`` ends the stage."""
        if not self._scenarios:  # every model disabled: no reason to fail the run
            return E2eReport.skipped("no enabled scenario in scenarios.yaml")
        findings: list[Finding] = []
        records: list[E2eRecord] = []
        cookbooks: dict[str, Cookbook] = {}
        # one commit for every scenario of the run
        try:
            source: str | ValueError = Cookbook.pinned(self._cookbook_source, self._cookbook_repo)
        except ValueError as error:
            source = error
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
                record = E2eRecord(
                    scenario=scenario.name,
                    cookbook=scenario.cookbook,
                    cookbook_sha="",
                    matches=scenario.roles,
                    installed_sglang=installed_sglang,
                )
                try:
                    if isinstance(source, ValueError):
                        raise source
                    if scenario.cookbook not in cookbooks:
                        cookbooks[scenario.cookbook] = Cookbook.fetch(
                            source, scenario.cookbook, self._cookbook_repo
                        )
                    cookbook = cookbooks[scenario.cookbook]
                    # each role's cell cautions, and the images the cookbook names
                    notes = [cookbook.cell(match).notes for match in scenario.roles.values()]
                    images = tuple(filter(None, map(cookbook.docker_image, scenario.roles.values())))
                except ValueError as error:
                    findings.append(
                        Finding.unlocated(FindingCode.PLATFORM_FAILURE, f"{scenario.name}: {error}")
                    )
                    records.append(record)
                    self.progress.write(f"{where}: {error}")
                    continue
                record = record.model_copy(
                    update={
                        "cookbook_sha": cookbook.sha,
                        "cookbook_images": images,
                        "notes": "; ".join(filter(None, notes)),
                    }
                )
                self.progress.write(
                    f"{where} ({scenario.display_name}): cookbook {cookbook.sha[:12]},"
                    f" images {', '.join(images) or '-'}, installed sglang {installed_sglang}; {record.notes}"
                )
                try:
                    run_dir = Path(work_dir, str(index))
                    off = self._evaluate_model(scenario, cookbook, env, run_dir, aklr_enabled=False)
                    record = record.model_copy(update={"off": off})
                    if not (fault := record.off_fault(self._min_baseline_accuracy)):
                        on = self._evaluate_model(scenario, cookbook, env, run_dir, aklr_enabled=True)
                        record = record.model_copy(update={"on": on})
                    elif record.is_other_sglang:
                        # the RFC's non-blocking warning: no finding, but on was not compared
                        warning = f"off failed, on not compared; the cookbook names another sglang: {fault}"
                        record = record.model_copy(update={"warning": warning})
                except RunNotKilledError as error:
                    findings.append(
                        Finding.unlocated(FindingCode.PLATFORM_FAILURE, f"{scenario.name}: {error}")
                    )
                    records.append(record)
                    self.progress.write(f"{where}: no result")
                    break
                findings += record.findings(
                    max_drop=self._max_accuracy_drop, min_off_score=self._min_baseline_accuracy
                )
                if record.on is not None and record.on.n_hit_calls == 0:
                    self.progress.write(f"{where}: AKLR served no call, every call fell back")
                records.append(record)
                self.progress.write(f"{where}: {record.console_line()}")
        return E2eReport.from_findings(
            findings,
            prompt=self._prompt,
            expected_answer=self._expected_answer,
            scenario_records=tuple(records),
        )

    def _evaluate_model(
        self,
        scenario: Scenario,
        cookbook: Cookbook,
        env: dict[str, str],
        run_dir: Path,
        *,
        aklr_enabled: bool,
    ) -> ModelEvaluationResult:
        """Start the model, evaluate GSM8K and check the answer and calls when AKLR is enabled.

        Failures are recorded in the result; only ``RunNotKilledError`` raises.
        """
        side = "on" if aklr_enabled else "off"
        calls_dir = run_dir / f"{side}-calls"
        side_env = {
            ENV_DISABLE: "0" if aklr_enabled else "1",
            ENV_EXPORT: "1" if aklr_enabled else "0",
            ENV_EXPORT_DIR: str(calls_dir),
        }
        try:
            commands, entry_url = export_commands(
                scenario,
                cookbook,
                hosts=self._hosts,
                env=env,
                side_env=side_env,
                startup_timeout=self._startup_timeout,
            )
            deployment = Deployment(commands, entry_url, run_dir / side)
        except ValueError as error:
            return ModelEvaluationResult(failure=str(error))
        deadline = time.monotonic() + self._evaluation_timeout
        result: dict[str, object] = {"argvs": deployment.argvs}
        failures: list[str] = []
        is_ready = False
        self.progress.write(f"{side}: starting {len(deployment.processes)} processes")
        try:
            # a slow weight load fails the start, not the grading it would leave no time for
            entry_url = deployment.start(timeout=self._startup_timeout)
            is_ready = True
            self.progress.write(
                f"{side}: serving at {entry_url}; GSM8K, {self._gsm8k_questions} questions less 5 shots"
            )
            argv = [
                sys.executable, "-m", "sglang.test.run_eval", "--eval-name", "gsm8k", "--base-url", entry_url,
                "--num-examples", str(self._gsm8k_questions), "--max-tokens", str(self._max_tokens), "--temperature", "0",
                "--num-threads", str(self._gsm8k_concurrency),
            ]  # fmt: skip
            print(f"Evaluation command: {shlex.join(argv)}", flush=True)
            try:
                done = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=max(1.0, deadline - time.monotonic()),
                    check=False,
                )
                # run_eval scores a request that failed (a 400, retries exhausted) as a wrong
                # answer and exits 0: its sampler's messages count them
                n_failed = done.stdout.count("Bad Request Error") + done.stdout.count(
                    "All retry attempts exhausted"
                )
                # full precision: the metrics dict it prints (numpy 2 shows np.float64(...)), not its
                # rounded "Score:" line
                scores = re.findall(r"'score': (?:np\.float64\()?([0-9.eE+-]+)", done.stdout)
                if done.returncode != 0 or not scores:
                    raise RuntimeError(
                        f"run_eval exited with {done.returncode}, no score\n{done.stderr[-2000:]}"
                    )
                # a few are a busy server's; each is scored wrong, so off's lower its baseline by
                # at most 2/1314 = 0.0015, under a tenth of the allowed accuracy drop; more are a broken one
                result["n_failed_requests"] = n_failed
                if n_failed > self._max_failed_requests:
                    raise RuntimeError(f"{n_failed} GSM8K requests failed\n{done.stdout[-2000:]}")
                result["gsm8k_score"] = float(scores[-1])
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                failures.append(f"GSM8K: {error}")
            if aklr_enabled:
                request = urllib.request.Request(
                    f"{entry_url}/v1/chat/completions",
                    data=json.dumps({
                        "model": scenario.model_path,
                        "messages": [{"role": "user", "content": self._prompt}],
                        "temperature": 0,
                        "max_tokens": self._max_tokens,
                    }).encode(),
                    headers={"Content-Type": "application/json"},
                )  # fmt: skip
                try:
                    with urllib.request.urlopen(
                        request, timeout=max(1.0, deadline - time.monotonic())
                    ) as response:
                        answer = json.load(response)["choices"][0]["message"]["content"] or ""
                    if not isinstance(answer, str):
                        raise TypeError(f"content is not a string: {answer!r}")
                    result["prompt_answer"] = answer
                    # Keys are case- and punctuation-sensitive; ignore only surrounding whitespace.
                    result["answer_matches"] = answer.strip() == self._expected_answer.strip()
                except (OSError, ValueError, LookupError, TypeError, http.client.HTTPException) as error:
                    failures.append(f"prompt: {type(error).__name__}: {error}")
            # a server that died during the phases fails the side, whatever they scored
            if exited := deployment.exited():
                failures.append(f"during the phases: {', '.join(exited)}")
        except (OSError, RuntimeError) as error:
            failures.append(f"start: {error}")
        finally:
            log_tails = deployment.stop()
        # a server that never became ready served nothing: no count, not "AKLR served no call";
        # AKLR writes its call file as it starts, so a ready one with none never started AKLR (a
        # plugin that failed to import, sglang told to load other plugins): on is off then
        if aklr_enabled and is_ready and not (calls_dir.is_dir() and any(calls_dir.iterdir())):
            failures.append("AKLR never started on the on side: no call file")
        elif aklr_enabled and is_ready:
            try:
                calls = read_calls(calls_dir)
                result["n_hit_calls"] = sum(c.n_calls for c in calls if c.fallback_reason is None)
                result["n_fallback_calls"] = sum(c.n_calls for c in calls if c.fallback_reason is not None)
                # no call at all, served or fallen back: AKLR hooked nothing (no route for this sglang,
                # the plugin switched off), so on was off; all fallen back is only flagged (the RFC)
                if not calls:
                    failures.append("AKLR hooked no sglang call on the on side: no call recorded")
            except (OSError, ValueError) as error:
                failures.append(f"call files: {error}")
        if failures:
            result["failure"] = "\n".join([*failures, log_tails])
        return ModelEvaluationResult.model_validate(result)
