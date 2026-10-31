# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from aklr.core._frozen_model import frozen_dataclass
from aklr.core.schema import load_json_file
from aklr.core.validation.stages._e2e._config import Scenario
from aklr.core.validation.stages._process_group import ProcessGroup

__all__ = ["RunNotKilledError", "RunResult", "RunSide", "ScenarioRun"]

RunSide = Literal["on", "off"]

_RUN_LOG_TAIL_BYTES: Final = 2 << 10


class RunNotKilledError(RuntimeError):
    """A run's process group outlived its ``SIGKILL`` and still holds the devices."""


@frozen_dataclass
class RunResult:
    """A run's result.json: its headline throughput and latency (decode step, or TTFT)."""

    tokens_per_s: float
    latency_ms: float


@dataclass(frozen=True)
class ScenarioRun:
    """``scenario`` run once on ``side``, in ``run_dir``."""

    scenario: Scenario
    side: RunSide
    run_dir: Path
    env: Mapping[str, str]

    def execute(self) -> str | None:
        """Run within the scenario timeout: ``None`` on success, else why and the log tail."""
        self.run_dir.mkdir(parents=True)
        log_path = self.run_dir.with_suffix(".log")
        argv = [sys.executable, *self.scenario.argv(self.run_dir / "result.json")]
        timeout_seconds = self.scenario.settings.timeout_seconds
        with open(log_path, "wb") as log:
            process = subprocess.Popen(
                argv,
                env=dict(self.env),
                cwd=self.run_dir,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
        try:
            return_code = process.wait(timeout=timeout_seconds)
            reason = f"exit {return_code}" if return_code else None
        except subprocess.TimeoutExpired:
            reason = f"timed out after {timeout_seconds:g} s"
        is_gone = ProcessGroup(process.pid).kill()
        if is_gone:  # a leader stuck past its SIGKILL never exits
            process.wait()
        try:
            with open(log_path, "rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - _RUN_LOG_TAIL_BYTES))
                log_tail = log.read().decode(errors="replace")
        except OSError:
            log_tail = ""
        if not is_gone:
            raise RunNotKilledError(f"{self.run_dir.name}: the process group outlived its kill\n{log_tail}")
        return None if reason is None else f"{reason}\n{log_tail}"

    def result(self) -> RunResult:
        from pydantic import ValidationError

        path = self.run_dir / "result.json"
        try:
            result = load_json_file(RunResult, path)
        except (ValidationError, OSError) as error:
            raise ValueError(f"{self.run_dir.name}: {error}") from error
        if not (0 < result.tokens_per_s < math.inf and 0 < result.latency_ms < math.inf):
            raise ValueError(f"{path}: {result} is not positive")
        return result
