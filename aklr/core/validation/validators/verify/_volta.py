# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import functools
import json
import re
import resource
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

if TYPE_CHECKING:
    from aklr.core.validation.validators.verify._extraction import Extraction

__all__ = ["VOLTA_CPU_SECONDS", "Volta", "VoltaCheck", "VoltaRun"]

VOLTA_CPU_SECONDS: Final = 120
VOLTA_MEMORY_BYTES: Final = 384 << 30
ARRAY_ALIGNMENT: Final = 0x10000
_CHECK: Final = re.compile(r"^  (\S.*?) \([^()\n]*\)\s+(PASS|FAIL|UNKNOWN)\n((?: {6}.*\n)*)", re.MULTILINE)
_EVENT: Final = re.compile(r"^\s+e\d+\s+(?:t\d+|async op \d+|generation \d+)\s+(.*)$")


@dataclass(frozen=True)
class VoltaCheck:
    status: str
    witnesses: tuple[tuple[str, tuple[str, ...]], ...] = ()
    omitted: int = 0


@dataclass(frozen=True)
class VoltaRun:
    failure: str = ""
    checks: dict[str, VoltaCheck] = field(default_factory=dict)
    abort: str = ""
    deadlock: str = ""
    arrays: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def incomplete(self) -> str:
        if self.abort:
            return f"aborted: {self.abort}"
        return f"deadlocked: {self.deadlock}" if self.deadlock else ""

    def is_past_an_array(self, address: int) -> bool:
        return any(
            end <= address < (end // ARRAY_ALIGNMENT + 1) * ARRAY_ALIGNMENT for _, end in self.arrays.values()
        )


class Volta:
    _MEMORY_TIERS: ClassVar[tuple[int, ...]] = (4 << 30, 16 << 30, 48 << 30)
    _EXIT_CODES: ClassVar[dict[str, int]] = {"VERIFIED": 0, "REJECTED": 1, "UNKNOWN": 2}
    _DTYPE_BYTES: ClassVar[dict[str, int]] = {
        "bf16": 2,
        "f16": 2,
        "f32": 4,
        "f64": 8,
        "e4m3": 1,
        "e5m2": 1,
        "i8": 1,
        "u8": 1,
        "i16": 2,
        "i32": 4,
        "i64": 8,
    }

    def __init__(self) -> None:
        self._free_memory = VOLTA_MEMORY_BYTES
        self._memory_freed = threading.Condition()

    @functools.cached_property
    def _binary(self) -> tuple[str | None, str]:
        path = shutil.which("volta")
        if path is None:
            return None, "the volta binary is not on PATH"
        try:
            banner = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=60, check=False)
            version = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return None, f"{path}: {e}"
        if not banner.stdout.startswith("The Volta PTX analysis engine"):
            return None, f"{path} is not the Volta prover"
        installed = Path(path).resolve().parents[1] / ".crates.toml"
        commit = re.search(
            r"volta_cli [^\n]*#([0-9a-f]{40})",
            installed.read_text(encoding="utf-8") if installed.is_file() else "",
        )
        return path, f"{version.stdout.strip()} at {commit[1] if commit else 'an unrecorded commit'}"

    @property
    def description(self) -> str:
        return self._binary[1]

    @classmethod
    def laid_out(cls, launch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
        arrays, extents, base = [], {}, ARRAY_ALIGNMENT
        for array in launch.get("arrays", ()):
            n_bytes = cls._DTYPE_BYTES.get(array["dtype"], 1)
            for extent in array["shape"]:
                n_bytes *= extent
            arrays.append(array | {"base": f"0x{base:x}"})
            extents[array["name"]] = (base, base + n_bytes)
            base = ((base + n_bytes) // ARRAY_ALIGNMENT + 1) * ARRAY_ALIGNMENT
        return launch | {"arrays": arrays}, extents

    def run(self, extraction: Extraction, directory: Path) -> VoltaRun:
        volta, description = self._binary
        if volta is None:
            return VoltaRun(failure=f"no-volta: {description}")
        launch, arrays = self.laid_out(extraction.launch or {})
        directory.mkdir(parents=True, exist_ok=True)
        launch_path = directory / "launch.json"
        launch_path.write_text(json.dumps(launch, indent=1), encoding="utf-8")
        argv = [volta, "solve", str(extraction.isa_path), "--config", str(launch_path)]
        argv += ["--no-log-file", "--print-outputs", "0"]
        for memory in self._MEMORY_TIERS:
            with self._memory_freed:
                self._memory_freed.wait_for(lambda m=memory: self._free_memory >= m)
                self._free_memory -= memory

            def limited(memory: int = memory) -> None:
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
                resource.setrlimit(resource.RLIMIT_CPU, (VOLTA_CPU_SECONDS, VOLTA_CPU_SECONDS + 10))

            try:
                done = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=VOLTA_CPU_SECONDS * 20,
                    check=False,
                    preexec_fn=limited,
                )
            except subprocess.TimeoutExpired:
                return VoltaRun(failure=f"timeout: volta waited {VOLTA_CPU_SECONDS * 20} s")
            finally:
                with self._memory_freed:
                    self._free_memory += memory
                    self._memory_freed.notify_all()
            if done.returncode == -signal.SIGXCPU:
                return VoltaRun(failure=f"timeout: volta ran {VOLTA_CPU_SECONDS} s of CPU time")
            log = done.stdout + done.stderr
            (directory / "volta.log").write_text(log, encoding="utf-8")
            conclusion = re.search(r"^GRAPH VERIFICATION: (\w+)", log, re.MULTILINE)
            if conclusion is not None or re.search(r"^error: ", log, re.MULTILINE) or not done.returncode:
                break
        if conclusion is None:
            if not done.returncode:
                raise RuntimeError(f"volta exited 0 with no conclusion: {directory}/volta.log")
            error = re.search(r"^error: (.*)$", log, re.MULTILINE)
            if error is None:
                return VoltaRun(failure=f"crashed: volta exited {done.returncode}")
            return VoltaRun(failure=f"unsupported: {error[1].strip()}")
        if self._EXIT_CODES.get(conclusion[1]) != done.returncode:
            raise RuntimeError(
                f"volta concluded {conclusion[1]} and exited {done.returncode}: {directory}/volta.log"
            )
        checks = {name: self._check(status, listed) for name, status, listed in _CHECK.findall(log)}
        if not checks:
            raise RuntimeError(f"volta listed no check: {directory}/volta.log")
        failed = sorted(name for name, check in checks.items() if check.status == "FAIL")
        deadlock = re.search(r"^  trace: deadlock, (.*)$", log, re.MULTILINE)
        if (conclusion[1] == "REJECTED") != bool(failed or deadlock):
            raise RuntimeError(
                f"volta concluded {conclusion[1]} with the checks {failed} failed: {directory}/volta.log"
            )
        aborted = re.search(r"aborted at (instruction \d+: .*)", log)
        return VoltaRun(
            checks=checks,
            abort=aborted[1].strip() if aborted else "",
            deadlock=f"the trace deadlocked, {deadlock[1].strip()}" if deadlock else "",
            arrays=arrays,
        )

    @staticmethod
    def _check(status: str, listed: str) -> VoltaCheck:
        witnesses: list[tuple[str, list[str]]] = []
        omitted = 0
        for line in listed.splitlines():
            event = _EVENT.match(line)
            more = re.fullmatch(r"\s+\.\.\. \((\d+) more\)", line)
            if event is not None and witnesses:
                witnesses[-1][1].append(" ".join(event[1].split()))
            elif more is not None:
                omitted += int(more[1])
            else:
                witnesses.append((line.strip(), []))
        return VoltaCheck(
            status=status,
            witnesses=tuple((text, tuple(events)) for text, events in witnesses),
            omitted=omitted,
        )
