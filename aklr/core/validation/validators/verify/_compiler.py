# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import ctypes
import os
import re
import signal
import struct
import subprocess
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from aklr.core.contract.op import OpContract
    from aklr.core.runtime.builder import Runnable

__all__ = ["Compiler"]


class Compiler:
    LANGUAGE: ClassVar[str]

    def __init__(self, schema_file: str, schema_path: str, directory: str, first: str, count: str) -> None:
        self._workloads = slice(int(first), int(first) + int(count))
        self._schema_file = Path(schema_file)
        self._schema_path = schema_path
        self._directory = Path(directory)
        self._where = self._directory
        self._workspaces: set[int] = set()

    @staticmethod
    def environment(arch: str, directory: Path) -> dict[str, str]:
        return {"OMP_NUM_THREADS": "1"}

    def _install(self, schema: OpContract) -> None:
        raise NotImplementedError

    def _built(self, runnable: Runnable) -> None:
        pass

    def _launches(self, tensors: dict[str, Any], mutable: set[str]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _finish(self) -> None:
        pass

    @staticmethod
    def _launch_fields(kernel: str, isa_path: Path, launch: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        fields: dict[str, Any] = {"kernel": kernel}
        try:
            fields |= {"launch": launch(), "isa_path": str(isa_path)}
        except (ValueError, LookupError, AttributeError, TypeError, struct.error, OSError) as e:
            fields |= {"failure": f"launch: {' '.join(str(e).split())[:300]}"}
        except subprocess.CalledProcessError as e:
            fields |= {"failure": f"launch: {e.cmd[0]} exited {e.returncode}: {e.stderr.strip()[-300:]}"}
        return fields

    def compile_workloads(self) -> None:
        import torch

        from aklr.core.contract import semantics
        from aklr.core.contract.registry import ContractRegistry
        from aklr.core.contract.spec import axes_text
        from aklr.core.runtime.builder import BuilderRegistry
        from aklr.core.validation.validators.bench._measure._contract_workload import ContractWorkload
        from aklr.core.validation.validators.verify._compilation import (
            MANIFEST,
            PR_SET_PDEATHSIG,
            WORKLOAD_TIMEOUT_SECONDS,
            Compilation,
        )

        torch.set_num_threads(1)
        schema = ContractRegistry.default().parse_schema(self._schema_file.read_bytes(), self._schema_path)
        self._install(schema)
        with (self._directory / MANIFEST).open("a", encoding="utf-8") as manifest:
            for workload in schema.workloads[self._workloads]:
                child = os.fork()
                if child:
                    status = os.waitpid(child, 0)[1]
                    if status:
                        ended = (
                            f"was killed by signal {os.WTERMSIG(status)}"
                            if os.WIFSIGNALED(status)
                            else f"exited {os.WEXITSTATUS(status)}"
                        )
                        failure = f"compile: the workload's compiler process {ended}"
                        if os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGALRM:
                            failure = f"compile: the workload compiled for {WORKLOAD_TIMEOUT_SECONDS} s"
                        crashed = Compilation(
                            schema_path=self._schema_path, axes=workload.axes, failure=failure
                        )
                        manifest.write("\n" + crashed.model_dump_json() + "\n")
                        manifest.flush()
                    continue
                try:
                    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
                    signal.alarm(WORKLOAD_TIMEOUT_SECONDS)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    if device == "cpu":
                        torch.cuda.current_stream = lambda device=None: None
                        torch.cuda.current_device = lambda: 0
                        torch._C._cuda_getCurrentRawStream = lambda index=0: 0
                    contract = ContractRegistry.default().get(schema.contract_id)
                    self._where = self._directory / re.sub(r"[^\w=,.-]", "_", axes_text(workload.axes))
                    mutable = set(contract.mutable_tensors)
                    tensors: dict[str, Any] = {}
                    failure = None
                    try:
                        drawn = ContractWorkload.of(schema, workload.axes)
                        call = drawn.assemble_call(drawn.draw("trial 0", device), device)
                        for output in contract.allocated_outputs:
                            dtype = semantics.torch_dtypes(drawn.point.dtypes[output])[0]
                            call[output] = torch.empty(
                                contract.storage_shape_of(output, drawn.point), dtype=dtype, device=device
                            )
                        tensors = {name: value for name, value in call.items() if torch.is_tensor(value)}
                        runnable = BuilderRegistry.get_instance().build(schema)
                        self._built(runnable)
                        runnable(**call)
                    except Exception as e:  # noqa: BLE001
                        message = " ".join(str(e).split())[:300]
                        failure = {"failure": f"compile: {type(e).__name__}: {message}"}
                    launched: list[dict[str, Any]] = []
                    for fields in self._launches(tensors, mutable):
                        if fields not in launched:
                            launched.append(fields)
                    if failure is not None:
                        launched.append(failure)
                    nothing = {"failure": f"compile: the call launched no {self.LANGUAGE} kernel"}
                    for fields in launched or [nothing]:
                        compilation = Compilation(schema_path=self._schema_path, axes=workload.axes, **fields)
                        manifest.write(compilation.model_dump_json() + "\n")
                        manifest.flush()
                except BaseException:
                    traceback.print_exc()
                    os._exit(1)
                os._exit(0)
        self._finish()
