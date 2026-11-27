# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from aklr.core.validation.validators.verify._compiler import Compiler
from aklr.core.validation.validators.verify._disassembly import assembly, code_object, rocm_tool
from aklr.core.validation.validators.verify._launch import KernelLaunch, allocations, kernel_arguments

if TYPE_CHECKING:
    from aklr.core.contract.op import OpContract
    from aklr.core.runtime.builder import Runnable

__all__ = ["HipCompiler"]

_RECORDER_SOURCE: Final = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>

typedef struct { unsigned x, y, z; } dim3;
typedef int (*recorder_t)(const void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void **,
                          size_t);
typedef int (*launch_t)(const void *, dim3, dim3, void **, size_t, void *);
static recorder_t recorder;

void aklr_verify_set_recorder(recorder_t r) { recorder = r; }

int hipLaunchKernel(const void *f, dim3 grid, dim3 block, void **args, size_t shared, void *stream) {
    static launch_t launch;
    if (recorder && recorder(f, grid.x, grid.y, grid.z, block.x, block.y, block.z, args, shared)) return 0;
    if (!launch) launch = (launch_t)dlsym(RTLD_NEXT, "hipLaunchKernel");
    return launch(f, grid, block, args, shared, stream);
}
"""
_RECORDER: Final = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, *[ctypes.c_uint] * 6, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
)


class _Symbol(ctypes.Structure):
    _fields_ = [
        ("file", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("name", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
    ]


class HipCompiler(Compiler):
    LANGUAGE = "HIP"

    def __init__(self, schema_file: str, schema_path: str, directory: str, first: str, count: str) -> None:
        super().__init__(schema_file, schema_path, directory, first, count)
        self._recorded: list[
            tuple[str, list[int], list[int], int, list[bytes], list[tuple[int, int]]] | str
        ] = []
        self._library: Path | None = None
        self._kernels: dict[int, str] = {}
        self._isa = ""
        self._isa_path = self._directory
        self._arch = ""
        self._recorder = _RECORDER(self._record)

    @staticmethod
    def environment(arch: str, directory: Path) -> dict[str, str]:
        environment = Compiler.environment(arch, directory)
        directory.mkdir(parents=True, exist_ok=True)
        source, library = directory / "launch_recorder.c", directory / "launch_recorder.so"
        source.write_text(_RECORDER_SOURCE, encoding="utf-8")
        compiler = shutil.which("cc") or shutil.which("gcc")
        if compiler is not None:
            built = subprocess.run(
                [compiler, "-shared", "-fPIC", "-O2", "-o", str(library), str(source), "-ldl"],
                capture_output=True,
                text=True,
                check=False,
            )
            (directory / "launch_recorder.log").write_text(built.stdout + built.stderr, encoding="utf-8")
            if built.returncode == 0:
                preloaded = os.environ.get("LD_PRELOAD")
                environment["LD_PRELOAD"] = " ".join(filter(None, [str(library), preloaded]))
        return environment

    def _install(self, schema: OpContract) -> None:
        from aklr.core.platform.device import DeviceTarget

        target = DeviceTarget(arch=schema.kernel.targets[0].arch, hardware=schema.kernel.targets[0].hardware)
        DeviceTarget.current = classmethod(lambda cls, device=None: target)
        self._arch = target.arch

    def _built(self, runnable: Runnable) -> None:
        set_recorder = getattr(ctypes.CDLL(None), "aklr_verify_set_recorder", None)
        if set_recorder is None:
            raise RuntimeError(f"the launch recorder is not preloaded: {self._directory}/launch_recorder.log")
        self._library = Path(runnable.metadata.misc["library"]).resolve()
        self._where.mkdir(parents=True, exist_ok=True)
        bundle = self._where / "hip_fatbin"
        subprocess.run(
            [
                rocm_tool("llvm-objcopy"),
                f"--dump-section=.hip_fatbin={bundle}",
                str(self._library),
                os.devnull,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        elf = code_object(bundle.read_bytes(), self._arch)
        self._isa = assembly(elf)
        self._isa_path = self._where / f"{self._library.stem}-{hashlib.sha256(elf).hexdigest()[:12]}.s"
        self._isa_path.write_text(self._isa, encoding="utf-8")
        names = set(re.findall(r"^\s+\.name:\s+(\S+)$", self._isa, re.MULTILINE))
        symbols = subprocess.run(
            [rocm_tool("llvm-nm"), "--defined-only", str(self._library)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self._kernels = {
            int(address, 16): name
            for address, name in re.findall(r"^([0-9a-f]+) [A-Za-z] (\S+)$", symbols, re.MULTILINE)
            if name in names
        }
        set_recorder(self._recorder)

    def _record(
        self,
        handle: int,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        block_x: int,
        block_y: int,
        block_z: int,
        args: Any,
        shared: int,
    ) -> int:
        symbol = _Symbol()
        ctypes.CDLL(None).dladdr(ctypes.c_void_p(handle), ctypes.byref(symbol))
        if symbol.file is None or Path(os.fsdecode(symbol.file)).resolve() != self._library:
            return 0
        try:
            kernel = self._kernels.get(handle - symbol.base)
            if kernel is None:
                raise LookupError(f"no kernel symbol at offset {handle - symbol.base:#x} of {self._library}")
            values = [
                ctypes.string_at(args[i], width)
                for i, (_, width, _) in enumerate(
                    k for k in kernel_arguments(self._isa, kernel) if not k[2].startswith("hidden_")
                )
            ]
            launched = ([grid_x, grid_y, grid_z], [block_x, block_y, block_z], shared, values, allocations())
            self._recorded.append((kernel, *launched))
        except (LookupError, ValueError) as e:
            self._recorded.append(f"launch: {e}")
        return 1

    def _launches(self, tensors: dict[str, Any], mutable: set[str]) -> list[dict[str, Any]]:
        return [
            {"failure": recorded}
            if isinstance(recorded, str)
            else self._launch_fields(
                recorded[0],
                self._isa_path,
                lambda recorded=recorded: self._launch(*recorded, tensors, mutable),
            )
            for recorded in self._recorded
        ]

    def _launch(
        self,
        kernel: str,
        grid: list[int],
        block: list[int],
        shared: int,
        values: list[bytes],
        allocated: list[tuple[int, int]],
        tensors: dict[str, Any],
        mutable: set[str],
    ) -> dict[str, Any]:
        launch = KernelLaunch(
            self._isa, kernel, grid, block, shared, tensors, mutable, self._workspaces, allocated
        )
        for (offset, width, kind), value in zip(launch.explicit, values, strict=True):
            launch.data[offset : offset + width] = value
            address = int.from_bytes(value, "little")
            if kind == "global_buffer" and address and not launch.address(offset, address):
                raise ValueError(f"the pointer at kernel argument offset {offset} is in no tensor")
        return launch.fields()
