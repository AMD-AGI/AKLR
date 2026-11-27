# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import ValidationError

import aklr
from aklr.core.contract.spec import FrozenModel, axes_text
from aklr.core.platform.toolchain import unmet_requirement
from aklr.core.validation.validators.verify._flydsl import FlyDSLCompiler
from aklr.core.validation.validators.verify._native import HipCompiler, HsacoCompiler
from aklr.core.validation.validators.verify._recording import RecordingCompiler
from aklr.core.validation.validators.verify._triton import TritonCompiler

if TYPE_CHECKING:
    from aklr.core.contract.op import OpContract

__all__ = [
    "COMPILERS",
    "PR_SET_PDEATHSIG",
    "STARTUP_TIMEOUT_SECONDS",
    "WORKLOAD_TIMEOUT_SECONDS",
    "Compilation",
]

COMPILERS: Final[dict[str, type[FlyDSLCompiler] | type[RecordingCompiler]]] = {
    "flydsl": FlyDSLCompiler,
    "triton": TritonCompiler,
    "hip": HipCompiler,
    "hsaco": HsacoCompiler,
}
MANIFEST: Final = "compilations.jsonl"
PR_SET_PDEATHSIG: Final = 1
STARTUP_TIMEOUT_SECONDS: Final = 300.0
WORKLOAD_TIMEOUT_SECONDS: Final = 300


class Compilation(FrozenModel):
    schema_path: str
    axes: dict[str, int]
    kernel: str | None = None
    isa_path: str | None = None
    launch: dict[str, Any] | None = None
    failure: str = ""

    @classmethod
    def of_schema(
        cls,
        root: Path,
        schema_path: str,
        schema: OpContract,
        directory: Path,
        first: int,
        count: int,
        device: str | None,
    ) -> list[Compilation]:
        workloads = schema.workloads[first : first + count]
        (directory / MANIFEST).unlink(missing_ok=True)
        kernel = schema.kernel
        compiler = COMPILERS.get(kernel.language)
        unmet = unmet_requirement(kernel.requires)
        crash = ""
        if compiler is None:
            crash = f"no compiler for {kernel.language}"
        elif unmet is not None:
            crash = f"requires: {unmet}"
        else:
            directory.mkdir(parents=True, exist_ok=True)
            python_path = os.pathsep.join(
                [str(Path(aklr.__file__).resolve().parents[1]), *filter(None, [os.environ.get("PYTHONPATH")])]
            )
            try:
                environment = compiler.environment(kernel.targets[0].arch, directory) | {
                    "PYTHONPATH": python_path
                }
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                crash = f"compile: {type(error).__name__}: {error}"
            else:
                if device is not None:
                    environment |= {"HIP_VISIBLE_DEVICES": device, "CUDA_VISIBLE_DEVICES": device}
                run = f"from {compiler.__module__} import {compiler.__name__} as compiler"
                run += "; import sys; compiler(*sys.argv[1:]).compile_workloads()"
                argv = [sys.executable, "-c", run, str(root / schema_path), schema_path, str(directory)]
                argv += [str(first), str(count)]
                timeout = STARTUP_TIMEOUT_SECONDS + WORKLOAD_TIMEOUT_SECONDS * len(workloads)
                with (directory / "compiler.log").open("w", encoding="utf-8") as log:
                    compiling = subprocess.Popen(
                        argv,
                        env=os.environ | environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        preexec_fn=lambda: ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL),
                    )
                    try:
                        compiling.wait(timeout)
                    except subprocess.TimeoutExpired:
                        crash = f"compile: the compiler ran {timeout:.0f} s"
                    finally:
                        with contextlib.suppress(ProcessLookupError, PermissionError):
                            os.killpg(compiling.pid, signal.SIGKILL)
                        compiling.wait()
                if not crash and compiling.returncode:
                    logged = (directory / "compiler.log").read_text(encoding="utf-8").strip().splitlines()
                    crash = f"compile: the compiler exited {compiling.returncode}: {(logged or ['no message'])[-1]}"
        manifest = directory / MANIFEST
        lines = manifest.read_text(encoding="utf-8").splitlines() if manifest.is_file() else []
        compilations = []
        for line in filter(None, lines):
            with contextlib.suppress(ValidationError):
                compilations.append(cls.model_validate_json(line))
        compiled = {axes_text(c.axes) for c in compilations}
        return compilations + [
            cls(
                schema_path=schema_path,
                axes=w.axes,
                failure=crash or "compile: the compiler reported nothing",
            )
            for w in workloads
            if axes_text(w.axes) not in compiled
        ]
