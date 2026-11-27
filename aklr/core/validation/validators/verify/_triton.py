# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from aklr.core.validation.validators.verify._recording import RecordingCompiler, kernel_metadata


class TritonCompiler(RecordingCompiler):
    @staticmethod
    def environment(arch: str, directory: Path) -> dict[str, str]:
        return RecordingCompiler.environment(arch, directory) | {
            "TRITON_CACHE_DIR": str(directory / "triton"),
            "TRITON_OVERRIDE_ARCH": arch,
        }

    def install(self, schema: Any) -> None:
        import torch
        from triton.backends.compiler import GPUTarget
        from triton.runtime import driver
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction

        if not torch.cuda.is_available():

            class CompileOnlyDriver:
                def get_current_device(self):
                    return 0

                def get_current_stream(self, device):
                    return 0

                def get_benchmarker(self):
                    def unavailable(*args, **kwargs):
                        raise RuntimeError("compile-only verification cannot benchmark")

                    return unavailable

                def get_current_target(inner):
                    return GPUTarget("hip", self.arch, 64)

            driver.set_active(CompileOnlyDriver())
        run = JITFunction.run

        def recording(function, *args, grid, warmup, **kwargs):
            compiled = run(function, *args, grid=grid, warmup=True, **kwargs)
            if compiled is None or warmup:
                return compiled
            signature = inspect.signature(function.fn)
            bound = signature.bind(*args, **{k: v for k, v in kwargs.items() if k in signature.parameters})
            bound.apply_defaults()
            dimensions = grid(bound.arguments) if callable(grid) else grid
            if any(dimension == 0 for dimension in dimensions):
                return compiled
            values = []
            for name, kind in compiled.src.signature.items():
                if kind == "constexpr":
                    continue
                values.append(
                    bound.arguments[name]
                    if isinstance(name, str)
                    else list(bound.arguments.values())[name[0]]
                )
            isa = compiled.asm["amdgcn"]
            explicit = [
                a
                for a in kernel_metadata(isa, compiled.name).get(".args", [])
                if not a[".value_kind"].startswith("hidden_")
            ]
            if len(explicit) == len(values) + 2:
                if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
                    raise ValueError("Triton scratch buffers are not supported")
                values += [0, 0]
            self.record(
                isa,
                compiled.name,
                dimensions,
                (compiled.metadata.num_warps * 64, 1, 1),
                compiled.metadata.shared,
                values,
            )
            return compiled

        JITFunction.run = recording

        def autotuned(function, *args, **kwargs):
            function.nargs = dict(zip(function.arg_names, args))
            try:
                result = None
                for config in function.prune_configs(kwargs):
                    function.best_config = config
                    options = kwargs | config.all_kwargs()
                    if config.pre_hook is not None:
                        config.pre_hook(function.nargs | options)
                    result = function.fn.run(*args, **options)
                return result
            finally:
                function.nargs = None

        Autotuner.run = autotuned
