# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any

from aklr.core.validation.validators.verify._extractors._extractor import IsaExtractor
from aklr.core.validation.validators.verify._extractors._launch import KernelLaunch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from aklr.core.contract.op import OpContract
    from aklr.core.runtime.builder import Runnable

__all__ = ["TritonIsaExtractor"]

_SCRATCH = ("global_scratch", "profile_scratch")


def _runtime_values(values: Any, types: Any) -> Iterator[Any]:
    for value, kind in zip(values, types, strict=True):
        if isinstance(kind, tuple):
            yield from _runtime_values(value, kind)
        elif kind != "constexpr":
            yield value


class TritonIsaExtractor(IsaExtractor, name="triton"):
    LANGUAGE = "Triton"

    def __init__(self, schema_file: str, schema_path: str, directory: str, first: str, count: str) -> None:
        super().__init__(schema_file, schema_path, directory, first, count)
        self._recorded: list[tuple[Any, list[int], tuple[Any, ...]]] = []
        self._arch = ""

    def _install(self, schema: OpContract) -> None:
        from triton.compiler.compiler import CompiledKernel

        self._arch = self.target(schema.kernel).arch
        recorded = self._recorded

        def run(kernel: Any) -> Any:
            def record(grid_x: int, grid_y: int, grid_z: int, *launcher: Any) -> None:
                recorded.append((kernel, [grid_x, grid_y, grid_z], launcher[6:]))

            return record

        CompiledKernel.run = property(run)
        CompiledKernel.launch_metadata = lambda kernel, grid, stream, *args: None

    def _built(self, runnable: Runnable) -> None:
        from triton.backends.compiler import GPUTarget
        from triton.runtime import driver

        target = GPUTarget("hip", self._arch, 64)
        driver.active.get_current_target = lambda: target

    def _launches(self, tensors: dict[str, Any], mutable: set[str]) -> list[dict[str, Any]]:
        launches = []
        for kernel, grid, args in self._recorded:
            isa = kernel.asm["amdgcn"]
            self._where.mkdir(parents=True, exist_ok=True)
            isa_path = self._where / f"{kernel.name}-{hashlib.sha256(isa.encode()).hexdigest()[:12]}.s"
            isa_path.write_text(isa, encoding="utf-8")
            launches.append(
                self._launch_fields(
                    kernel.name,
                    isa_path,
                    lambda kernel=kernel, grid=grid, args=args, isa=isa: self._launch(
                        kernel, grid, args, isa, tensors, mutable
                    ),
                )
            )
        return launches

    def _launch(
        self,
        kernel: Any,
        grid: list[int],
        args: tuple[Any, ...],
        isa: str,
        tensors: dict[str, Any],
        mutable: set[str],
    ) -> dict[str, Any]:
        metadata = kernel.metadata
        warp_size = getattr(metadata, "warp_size", None) or metadata.target.warp_size
        launch = KernelLaunch(
            isa,
            kernel.name,
            grid,
            [metadata.num_warps * warp_size, 1, 1],
            metadata.shared,
            tensors,
            mutable,
            self._workspaces,
        )
        values = list(_runtime_values(args, kernel.src.signature.values()))
        if len(launch.explicit) != len(values) + len(_SCRATCH):
            raise ValueError(
                f"{len(launch.explicit)} kernel arguments for {len(values)} values and {len(_SCRATCH)} scratch buffers"
            )
        for (offset, width, kind), value in zip(launch.explicit, values):
            if kind == "global_buffer" and hasattr(value, "data_ptr"):
                launch.tensor(offset, value)
            elif kind == "global_buffer" and value is None:
                continue
            elif kind == "by_value" and isinstance(value, (bool, int, float)):
                launch.value(offset, width, value)
            else:
                raise ValueError(f"a {width}-byte {kind} kernel argument for {type(value).__name__}")
        for (offset, _, _), name in zip(launch.explicit[len(values) :], _SCRATCH, strict=True):
            if n_bytes := getattr(metadata, f"{name}_size", 0) * math.prod(grid):
                launch.buffer(offset, name, n_bytes)
        return launch.fields()
