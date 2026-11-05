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

# phase 1, GSM8K on both sides: every test question; run_eval takes its 5 few-shot examples
# from them, so 1314 are graded
GSM8K_N_QUESTIONS: Final = 1319
# concurrent requests (run_eval's default is 512: a long-thinking model at 16384 tokens then
# queues past its client's retries); failed requests tolerated, each scored wrong
GSM8K_N_THREADS: Final = 64
GSM8K_MAX_FAILED_REQUESTS: Final = 6
# enough for a thinking model; others stop far earlier
MAX_TOKENS: Final = 16384
# provisional, about 2.5 sigma of the on - off difference over 1314 shared questions; to be
# set from measured off-against-off noise
GSM8K_MAX_DROP: Final = 0.02
# phase 2, the on side only: one prompt with a short, unique answer
PROMPT: Final = "What is the capital of France? Answer with one word."
EXPECTED_ANSWER: Final = "Paris"
# below it, off is not the model answering (a broken recipe or grading), and on is not run
MIN_OFF_GSM8K: Final = 0.5
# a side, from its start to its stop, and of it the start (weights, graph capture)
SIDE_TIMEOUT_SECONDS: Final = 5400.0
START_TIMEOUT_SECONDS: Final = 2400.0

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
        if not devices:
            raise ValueError("e2e needs at least one device")
        if not hosts:
            raise ValueError("e2e needs at least one host")
        self._scenarios = tuple(scenarios)
        self._cookbook_repo = cookbook_repo
        self._cookbook_source = cookbook_source
        self._hosts = tuple(hosts)
        # one {cmd} word: the remote shell reads the command back from it unchanged
        if launch is not None and shlex.split(launch).count("{cmd}") != 1:
            raise ValueError(f"--e2e-launch {launch!r}: needs one {{cmd}} word, e.g. 'ssh {{host}} {{cmd}}'")
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
                    off = self._run_side(scenario, cookbook, "off", env, run_dir)
                    record = record.model_copy(update={"off": off})
                    if not (fault := record.off_fault(MIN_OFF_GSM8K)):
                        on = self._run_side(scenario, cookbook, "on", env, run_dir)
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
                findings += record.findings(max_drop=GSM8K_MAX_DROP, min_off_score=MIN_OFF_GSM8K)
                if record.on is not None and record.on.n_hit_calls == 0:
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
        is_ready = False
        self.progress.write(f"{side}: starting {len(deployment.processes)} processes")
        try:
            # a slow weight load fails the start, not the grading it would leave no time for
            entry_url = deployment.start(timeout=START_TIMEOUT_SECONDS)
            is_ready = True
            self.progress.write(
                f"{side}: serving at {entry_url}; GSM8K, {GSM8K_N_QUESTIONS} questions less 5 shots"
            )
            argv = [
                sys.executable, "-m", "sglang.test.run_eval", "--eval-name", "gsm8k", "--base-url", entry_url,
                "--num-examples", str(GSM8K_N_QUESTIONS), "--max-tokens", str(MAX_TOKENS), "--temperature", "0",
                "--num-threads", str(GSM8K_N_THREADS),
            ]  # fmt: skip
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
                # a few are a busy server's (each scored wrong, at most 0.005 of the score, well
                # under GSM8K_MAX_DROP); more are a broken one
                if n_failed > GSM8K_MAX_FAILED_REQUESTS:
                    raise RuntimeError(f"{n_failed} GSM8K requests failed\n{done.stdout[-2000:]}")
                result["gsm8k_score"] = float(scores[-1])
                result["n_failed_requests"] = n_failed
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
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
                    if not isinstance(answer, str):
                        raise TypeError(f"content is not a string: {answer!r}")
                    result["prompt_answer"] = answer
                    # the whole answer, less surrounding whitespace and trailing punctuation
                    normalized = re.sub(r"\W+$", "", answer).strip().casefold()
                    result["answer_matches"] = normalized == EXPECTED_ANSWER.casefold()
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
        if side == "on" and is_ready and not (calls_dir.is_dir() and any(calls_dir.iterdir())):
            failures.append("AKLR never started on the on side: no call file")
        elif side == "on" and is_ready:
            try:
                calls = read_calls(calls_dir)
                result["n_hit_calls"] = sum(c.n_calls for c in calls if c.fallback_reason is None)
                result["n_fallback_calls"] = sum(c.n_calls for c in calls if c.fallback_reason is not None)
            except (OSError, ValueError) as error:
                failures.append(f"call files: {error}")
        if failures:
            result["failure"] = "\n".join([*failures, log_tails])
        return SideResult.model_validate(result)
