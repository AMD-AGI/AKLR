# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import signal
import struct
from pathlib import Path
from typing import Any, ClassVar

__all__ = ["FlyDSLCompiler"]

_LAUNCH = re.compile(
    r"gpu\.launch_func[^\n]*?@kernels::@(?P<kernel>\S+) blocks in \((?P<grid>[^)]*)\)"
    r" threads in \((?P<block>[^)]*)\)"
    r"(?:\s+dynamic_shared_memory_size (?P<shared>\S+))?\s+args\((?P<args>.*?)\)\s*(?:\{|loc|$)",
    re.MULTILINE,
)
_KERNARG = re.compile(r"\.offset:\s+(\d+)\n\s+\.size:\s+(\d+)\n\s+\.value_kind:\s+(\w+)")
_CONTENTS_LIMIT = 1 << 16


class FlyDSLCompiler:
    _DTYPES: ClassVar[dict[str, str]] = {
        "bfloat16": "bf16",
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "float8_e4m3fn": "e4m3",
        "float8_e5m2": "e5m2",
        "int8": "i8",
        "uint8": "u8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
    }

    def __init__(self, schema_file: str, schema_path: str, directory: str, first: str, count: str) -> None:
        self._workloads = slice(int(first), int(first) + int(count))
        self._schema_file = Path(schema_file)
        self._schema_path = schema_path
        self._directory = Path(directory)
        self._calls: list[tuple[list[Any], list[tuple[str, Path]]]] = []
        self._workspaces: set[int] = set()

    @staticmethod
    def environment(arch: str, directory: Path) -> dict[str, str]:
        return {
            "COMPILE_ONLY": "1",
            "ARCH": arch,
            "FLYDSL_GPU_ARCH": arch,
            "FLYDSL_DUMP_IR": "1",
            "FLYDSL_DUMP_DIR": str(directory / "flydsl_dump"),
            "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
            "OMP_NUM_THREADS": "1",
        }

    def compile_workloads(self) -> None:
        import torch
        from flydsl._mlir import ir
        from flydsl.compiler.jit_function import Constexpr, JitFunction

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
        wrapped = JitFunction._compile_or_execute
        dump = Path(self.environment("", self._directory)["FLYDSL_DUMP_DIR"])

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
                        where.mkdir(parents=True, exist_ok=True)
                        (where / name).write_bytes(isa)
                        modules.append((stage_zero[0].read_text(encoding="utf-8"), where / name))
                self._calls.append((runtime, modules))

        JitFunction._compile_or_execute = recording
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
                ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
                signal.alarm(WORKLOAD_TIMEOUT_SECONDS)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    torch.cuda.current_stream = lambda device=None: None
                    torch.cuda.current_device = lambda: 0
                    torch._C._cuda_getCurrentRawStream = lambda index=0: 0
                contract = ContractRegistry.default().get(schema.contract_id)
                launched: list[dict[str, Any]] = []
                where = self._directory / re.sub(r"[^\w=,.-]", "_", axes_text(workload.axes))
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
                    BuilderRegistry.get_instance().build(schema)(**call)
                except Exception as e:  # noqa: BLE001
                    message = " ".join(str(e).split())[:300]
                    failure = {"failure": f"compile: {type(e).__name__}: {message}"}
                for runtime, modules in self._calls:
                    for module, isa_path in modules:
                        for found in _LAUNCH.finditer(module):
                            fields: dict[str, Any] = {"kernel": found["kernel"]}
                            try:
                                launch = self._launch(found, isa_path, runtime, tensors, mutable)
                                fields |= {"launch": launch, "isa_path": str(isa_path)}
                            except (ValueError, LookupError, AttributeError, struct.error) as e:
                                fields |= {"failure": f"launch: {' '.join(str(e).split())[:300]}"}
                            if fields not in launched:
                                launched.append(fields)
                if failure is not None:
                    launched.append(failure)
                for fields in launched or [{"failure": "compile: the call launched no FlyDSL kernel"}]:
                    compilation = Compilation(schema_path=self._schema_path, axes=workload.axes, **fields)
                    manifest.write(compilation.model_dump_json() + "\n")
                    manifest.flush()
                os._exit(0)
        shutil.rmtree(dump, ignore_errors=True)

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
        isa = isa_path.read_text(encoding="utf-8")
        entries = [
            entry
            for entry in re.split(r"\n  - ", isa[isa.index("amdhsa.kernels:") :])[1:]
            if re.search(rf"\n    \.name:\s+{re.escape(found['kernel'])}\n", "\n    " + entry)
        ]
        if len(entries) != 1:
            raise ValueError(f"{len(entries)} kernels named {found['kernel']} in the ISA's metadata")
        kernargs = [(int(offset), int(size), kind) for offset, size, kind in _KERNARG.findall(entries[0])]
        size = int(re.search(r"\.kernarg_segment_size:\s+(\d+)", entries[0])[1])
        data = bytearray(-(-size // 64) * 64)
        hidden = dict(zip("xyz", grid, strict=True))
        sizes = dict(zip("xyz", block, strict=True))
        for offset, width, kind in kernargs:
            if kind.startswith("hidden_block_count_"):
                data[offset : offset + width] = hidden[kind[-1]].to_bytes(width, "little")
            elif kind.startswith("hidden_group_size_"):
                data[offset : offset + width] = sizes[kind[-1]].to_bytes(width, "little")
            elif kind == "hidden_grid_dims":
                dims = sum(extent > 1 for extent in grid) or 1
                data[offset : offset + width] = dims.to_bytes(width, "little")
        explicit = [k for k in kernargs if not k[2].startswith("hidden_")]
        arrays: dict[str, dict[str, Any]] = {}
        fresh: set[int] = set()
        pointers: dict[int, str] = {}
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
                known = next((n for n, t in tensors.items() if t.data_ptr() == tensor.data_ptr()), None)
                if known is None and tensor.data_ptr() not in self._workspaces:
                    fresh.add(tensor.data_ptr())
                array = known or f"workspace{offset}"
                dtype = self._DTYPES[str(tensor.dtype).removeprefix("torch.")]
                arrays[array] = {
                    "name": array,
                    "kind": "out" if known in mutable or (known or tensor.data_ptr()) in fresh else "in",
                    "dtype": dtype,
                    "shape": list(tensor.shape)
                    if tensor.is_contiguous()
                    else [
                        1 + sum((n - 1) * step for n, step in zip(tensor.shape, tensor.stride(), strict=True))
                    ],
                }
                if (
                    known is not None
                    and arrays[array]["kind"] == "in"
                    and dtype == "i32"
                    and tensor.numel() <= _CONTENTS_LIMIT
                ):
                    arrays[array]["contents"] = tensor.flatten().tolist()
                pointers[offset] = array
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
                        struct.pack_into(f"<{len(packed)}I", data, layout_offset, *packed)
            elif kind == "by_value" and isinstance(value, float) and width in (4, 8):
                struct.pack_into("<f" if width == 4 else "<d", data, offset, value)
            elif kind == "by_value" and isinstance(value, int) and width in (4, 8):
                data[offset : offset + width] = (value & ((1 << 8 * width) - 1)).to_bytes(width, "little")
            else:
                raise ValueError(f"{name}: a {width}-byte {kind} kernel argument for {type(value).__name__}")
        if explicit:
            raise ValueError(f"{len(explicit)} kernel arguments are not in the launch")
        self._workspaces |= fresh
        params: list[dict[str, Any]] = []
        offset = 0
        while offset < len(data):
            if offset in pointers:
                params.append({"ptr": pointers[offset]})
                offset += 8
            else:
                params.append({"int": struct.unpack_from("<i", data, offset)[0]})
                offset += 4
        return {
            "kernel": found["kernel"],
            "grid": grid,
            "block": block,
            "dynamic_shared_bytes": int(shared[1]),
            "arrays": list(arrays.values()),
            "params": params,
        }
