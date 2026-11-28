# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import hashlib
import re
import shutil
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aklr.core.validation.validators.verify._extractors._extractor import IsaExtractor
from aklr.core.validation.validators.verify._extractors._launch import KernelLaunch

if TYPE_CHECKING:
    from aklr.core.contract.op import OpContract

__all__ = ["FlyDSLIsaExtractor"]

_LAUNCH = re.compile(
    r"gpu\.launch_func[^\n]*?@kernels::@(?P<kernel>\S+) blocks in \((?P<grid>[^)]*)\)"
    r" threads in \((?P<block>[^)]*)\)"
    r"(?:\s+dynamic_shared_memory_size (?P<shared>\S+))?\s+args\((?P<args>.*?)\)\s*(?:\{|loc|$)",
    re.MULTILINE,
)


class FlyDSLIsaExtractor(IsaExtractor, name="flydsl"):
    LANGUAGE = "FlyDSL"

    def __init__(self, schema_file: str, schema_path: str, directory: str, first: str, count: str) -> None:
        super().__init__(schema_file, schema_path, directory, first, count)
        self._calls: list[tuple[list[Any], list[tuple[str, Path]]]] = []
        self._dump = Path(self.environment("", self._directory)["FLYDSL_DUMP_DIR"])

    @staticmethod
    def environment(arch: str, directory: Path) -> dict[str, str]:
        return IsaExtractor.environment(arch, directory) | {
            "COMPILE_ONLY": "1",
            "ARCH": arch,
            "FLYDSL_GPU_ARCH": arch,
            "FLYDSL_DUMP_IR": "1",
            "FLYDSL_DUMP_DIR": str(directory / "flydsl_dump"),
            "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
        }

    def _install(self, schema: OpContract) -> None:
        from flydsl._mlir import ir
        from flydsl.compiler.jit_function import Constexpr, JitFunction

        wrapped = JitFunction._compile_or_execute
        dump = self._dump

        def recording(function: Any, args: Any, kwargs: Any, **options: Any) -> Any:
            if ir.Context.current is not None:
                return wrapped(function, args, kwargs, **options)
            own_args = function._split_self(args)[1]
            bound = function._sig.bind(*own_args, **kwargs)
            bound.apply_defaults()
            runtime = [
                value
                for name, value in bound.arguments.items()
                if not Constexpr.is_constexpr_annotation(function._sig.parameters[name].annotation)
            ]
            function._mem_cache.clear()
            shutil.rmtree(dump, ignore_errors=True)
            try:
                compiled = wrapped(function, args, kwargs, **options)
                if compiled is None and options.get("return_compiled"):
                    return lambda *args, **kwargs: None
                return compiled
            finally:
                modules = []
                for module_directory in sorted(dump.glob("*")):
                    stage_zero = sorted(module_directory.glob("00_*.mlir"))
                    isa_files = sorted(module_directory.glob("*_final_isa.s"))
                    if stage_zero and isa_files:
                        isa = isa_files[-1].read_bytes()
                        name = f"{module_directory.name}-{hashlib.sha256(isa).hexdigest()[:12]}.s"
                        self._where.mkdir(parents=True, exist_ok=True)
                        (self._where / name).write_bytes(isa)
                        modules.append((stage_zero[0].read_text(encoding="utf-8"), self._where / name))
                self._calls.append((runtime, modules))

        JitFunction._compile_or_execute = recording

    def _launches(self, tensors: dict[str, Any], mutable: set[str]) -> list[dict[str, Any]]:
        return [
            self._launch_fields(
                found["kernel"],
                isa_path,
                lambda found=found, isa_path=isa_path, runtime=runtime: self._launch(
                    found, isa_path, runtime, tensors, mutable
                ),
            )
            for runtime, modules in self._calls
            for module, isa_path in modules
            for found in _LAUNCH.finditer(module)
        ]

    def _finish(self) -> None:
        shutil.rmtree(self._dump, ignore_errors=True)

    def _launch(
        self,
        found: re.Match[str],
        isa_path: Path,
        runtime: list[Any],
        tensors: dict[str, Any],
        mutable: set[str],
    ) -> dict[str, Any]:
        grid, block = (
            [
                int(m[1]) if (m := re.fullmatch(r"%c(\d+)(?:_\d+)?", c.strip())) else None
                for c in found[g].split(",")
            ]
            for g in ("grid", "block")
        )
        if None in grid or None in block:
            raise ValueError(f"the grid ({found['grid']}) or block ({found['block']}) is not constant")
        shared = re.fullmatch(r"%c(\d+)(?:_\w+)?", found["shared"] or "%c0")
        if shared is None:
            raise ValueError(f"the dynamic shared memory size ({found['shared']}) is not constant")
        launch = KernelLaunch(
            isa_path.read_text(encoding="utf-8"),
            found["kernel"],
            grid,
            block,
            int(shared[1]),
            tensors,
            mutable,
            self._workspaces,
        )
        explicit = list(launch.explicit)
        for operand in re.split(r",\s*(?=%)", found["args"]):
            name, _, kind_text = (part.strip() for part in operand.partition(" : "))
            if not re.fullmatch(r"%arg\d+", name):
                raise ValueError(f"the kernel argument {name} is not an argument of its launcher")
            value = runtime[int(name[4:])]
            offset, width, kind = explicit.pop(0)
            if kind_text.startswith(("!fly.ptr", "!fly.memref")):
                tensor = getattr(value, "torch_tensor", value)
                if kind != "global_buffer" or not hasattr(tensor, "data_ptr"):
                    raise ValueError(f"{name}: a {kind} kernel argument for {type(value).__name__}")
                launch.tensor(offset, tensor)
                layout = re.search(r"global,\s*(\([^)]*\)|[^:,>]+)\s*:\s*(\([^)]*\)|[^,>]+)", kind_text)
                if layout is not None:
                    shape_text, stride_text = (
                        text.strip().strip("()").split(",") for text in layout.groups()
                    )
                    packed = [tensor.shape[i] for i, e in enumerate(shape_text) if e.strip() == "?"]
                    for i, e in enumerate(stride_text):
                        if e.strip().startswith("?"):
                            packed += [tensor.stride(i), 0] if "i64" in e else [tensor.stride(i)]
                    if packed:
                        layout_offset, layout_width, layout_kind = explicit.pop(0)
                        if (layout_kind, layout_width, layout_offset) != (
                            "by_value",
                            4 * len(packed),
                            offset + 8,
                        ):
                            raise ValueError(
                                f"{name}: its layout {packed} is not the {layout_width} bytes after it"
                            )
                        struct.pack_into(f"<{len(packed)}I", launch.data, layout_offset, *packed)
            elif kind == "by_value" and isinstance(value, (int, float)) and width in (4, 8):
                launch.value(offset, width, value)
            else:
                raise ValueError(f"{name}: a {width}-byte {kind} kernel argument for {type(value).__name__}")
        if explicit:
            raise ValueError(f"{len(explicit)} kernel arguments are not in the launch")
        return launch.fields()
