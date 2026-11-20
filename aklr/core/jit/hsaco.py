# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import ctypes
import functools
import hashlib
import os
from collections.abc import Sequence

import torch

__all__ = ["Kernel", "is_available"]

_BUFFER_POINTER, _BUFFER_SIZE, _END = 1, 2, 3
_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_MODULES: dict[tuple[int, bytes], tuple[ctypes.c_void_p, bytes]] = {}


def is_available() -> bool:
    return torch.version.hip is not None


class Kernel:
    def __init__(self, code_object: bytes, symbol: str) -> None:
        self._code_object = bytes(code_object)
        self._digest = hashlib.sha256(self._code_object).digest()
        self._symbol = symbol
        self._functions: dict[int, ctypes.c_void_p] = {}

    def launch(
        self,
        grid: Sequence[int],
        block: Sequence[int],
        kernarg: bytes | bytearray,
        *,
        shared_memory: int = 0,
        device: torch.device | int | None = None,
    ) -> None:
        index = None if device is None else torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        function = self._functions.get(index)
        if function is None:
            loaded = _MODULES.get((index, self._digest))
            if loaded is None:
                module = ctypes.c_void_p()
                with torch.cuda.device(index):
                    status = _hip().hipModuleLoadData(ctypes.byref(module), self._code_object)
                _check(status, "loading a code object")
                loaded = _MODULES[(index, self._digest)] = (module, self._code_object)
            function = ctypes.c_void_p()
            status = _hip().hipModuleGetFunction(ctypes.byref(function), loaded[0], self._symbol.encode())
            _check(status, f"finding {self._symbol} in its code object")
            self._functions[index] = function
        grid_x, grid_y, grid_z = (*grid, 1, 1)[:3]
        block_x, block_y, block_z = (*block, 1, 1)[:3]
        buffer = ctypes.create_string_buffer(bytes(kernarg), len(kernarg))
        size = ctypes.c_size_t(len(kernarg))
        extra = (ctypes.c_void_p * 5)(
            _BUFFER_POINTER, ctypes.addressof(buffer), _BUFFER_SIZE, ctypes.addressof(size), _END
        )
        stream = _RAW_STREAM(index) if _RAW_STREAM else torch.cuda.current_stream(index).cuda_stream
        status = _hip().hipModuleLaunchKernel(
            function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_memory, stream, None, extra
        )
        _check(status, f"launching {self._symbol}")


@functools.cache
def _hip() -> ctypes.CDLL:
    with open("/proc/self/maps", encoding="utf-8") as maps:
        paths = {line.split(maxsplit=5)[-1].strip() for line in maps if "libamdhip64" in line}
    names = {p: os.path.basename(p) for p in paths}
    path = next((p for p in sorted(paths) if names[p].startswith("libamdhip64") and ".so" in names[p]), None)
    if path is None:
        raise RuntimeError("torch has not loaded the HIP runtime (libamdhip64)")
    hip = ctypes.CDLL(path, mode=os.RTLD_NOLOAD)
    hip.hipGetErrorString.restype = ctypes.c_char_p
    hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        *[ctypes.c_uint] * 7,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return hip


def _check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"{what}: {_hip().hipGetErrorString(status).decode()}")
