# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import functools
import hashlib
import struct
from typing import TYPE_CHECKING, Any

from aklr.core.validation.validators.verify._compiler import Compiler
from aklr.core.validation.validators.verify._disassembly import assembly, code_object
from aklr.core.validation.validators.verify._launch import KernelLaunch, allocations

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from aklr.core.contract.op import OpContract

__all__ = ["HsacoCompiler"]


class HsacoCompiler(Compiler):
    LANGUAGE = "hsaco"

    def __init__(self, schema_file: str, schema_path: str, directory: str, first: str, count: str) -> None:
        super().__init__(schema_file, schema_path, directory, first, count)
        self._recorded: list[tuple[bytes, str, list[int], list[int], bytes, int, list[tuple[int, int]]]] = []

    def _install(self, schema: OpContract) -> None:
        from aklr.core.jit import hsaco

        recorded = self._recorded

        def launch(
            kernel: hsaco.Kernel,
            grid: Sequence[int],
            block: Sequence[int],
            kernarg: bytes | bytearray,
            *,
            shared_memory: int = 0,
            device: Any = None,
        ) -> None:
            recorded.append(
                (
                    kernel._code_object,
                    kernel._symbol,
                    [*grid, 1, 1][:3],
                    [*block, 1, 1][:3],
                    bytes(kernarg),
                    shared_memory,
                    allocations(),
                )
            )

        hsaco.Kernel.launch = launch

    def _launches(self, tensors: dict[str, Any], mutable: set[str]) -> list[dict[str, Any]]:
        launches = []
        for recorded in self._recorded:
            code_object, symbol = recorded[:2]
            self._where.mkdir(parents=True, exist_ok=True)
            isa_path = self._where / f"{symbol}-{hashlib.sha256(code_object).hexdigest()[:12]}.s"
            launches.append(
                self._launch_fields(
                    symbol, isa_path, functools.partial(self._launch, *recorded, isa_path, tensors, mutable)
                )
            )
        return launches

    def _launch(
        self,
        bundle: bytes,
        symbol: str,
        grid: list[int],
        block: list[int],
        kernarg: bytes,
        shared: int,
        allocated: list[tuple[int, int]],
        isa_path: Path,
        tensors: dict[str, Any],
        mutable: set[str],
    ) -> dict[str, Any]:
        if not isa_path.is_file():
            isa_path.write_text(assembly(code_object(bundle)), encoding="utf-8")
        launch = KernelLaunch(
            isa_path.read_text(encoding="utf-8"),
            symbol,
            grid,
            block,
            shared,
            tensors,
            mutable,
            self._workspaces,
            allocated,
        )
        if len(kernarg) > len(launch.data):
            raise ValueError(
                f"{len(kernarg)} bytes of kernel arguments for a {len(launch.data)}-byte segment"
            )
        launch.data[: len(kernarg)] = kernarg
        for offset, width, kind in launch.explicit:
            if kind == "global_buffer":
                address = struct.unpack_from("<Q", launch.data, offset)[0]
                if address and not launch.address(offset, address):
                    raise ValueError(f"the pointer at kernel argument offset {offset} is in no tensor")
            elif kind == "by_value":
                for at in range(-(-offset // 8) * 8, min(offset + width, len(kernarg)) - 7, 8):
                    if address := struct.unpack_from("<Q", launch.data, at)[0]:
                        launch.address(at, address)
        return launch.fields()
