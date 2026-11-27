# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ctypes
import os
import re
import signal
import struct
from pathlib import Path
from typing import Any

import yaml

from aklr.core.contract.spec import axes_text

_DTYPES = {
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


def kernel_metadata(isa: str, kernel: str) -> dict[str, Any]:
    match = re.search(r"(?ms)^\s*\.amdgpu_metadata\s*\n(.*?)^\s*\.end_amdgpu_metadata", isa)
    if match is None:
        raise ValueError("the ISA has no AMDGPU metadata")
    kernels = yaml.safe_load(match[1])["amdhsa.kernels"]
    matches = [item for item in kernels if item[".name"] == kernel]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} kernels named {kernel} in the ISA metadata")
    return matches[0]


class RecordingCompiler:
    def __init__(self, schema_file: str, schema_path: str, directory: str, first: str, count: str):
        self.schema_file = Path(schema_file)
        self.schema_path = schema_path
        self.directory = Path(directory)
        self.workloads = slice(int(first), int(first) + int(count))
        self.records: list[dict[str, Any]] = []
        self.tensors: dict[str, Any] = {}
        self.mutable: set[str] = set()
        self.workspaces: set[int] = set()

    @staticmethod
    def environment(arch: str, directory: Path) -> dict[str, str]:
        return {"ARCH": arch, "OMP_NUM_THREADS": "1"}

    def install(self, schema: Any) -> None:
        pass

    def prepare(self, runnable: Any) -> None:
        pass

    def compile_workloads(self) -> None:
        import torch

        from aklr.core.contract import semantics
        from aklr.core.contract.registry import ContractRegistry
        from aklr.core.runtime.builder import BuilderRegistry
        from aklr.core.validation.validators.bench._measure._contract_workload import ContractWorkload
        from aklr.core.validation.validators.verify._compilation import (
            MANIFEST,
            PR_SET_PDEATHSIG,
            WORKLOAD_TIMEOUT_SECONDS,
            Compilation,
        )

        torch.set_num_threads(1)
        schema = ContractRegistry.default().parse_schema(self.schema_file.read_bytes(), self.schema_path)
        self.arch = schema.kernel.targets[0].arch
        with (self.directory / MANIFEST).open("a", encoding="utf-8") as manifest:
            for workload in schema.workloads[self.workloads]:
                child = os.fork()
                if child:
                    status = os.waitpid(child, 0)[1]
                    if status:
                        failure = (
                            f"compile: the workload compiler was killed by signal {os.WTERMSIG(status)}"
                            if os.WIFSIGNALED(status)
                            else f"compile: the workload compiler exited {os.WEXITSTATUS(status)}"
                        )
                        manifest.write(
                            "\n"
                            + Compilation(
                                schema_path=self.schema_path, axes=workload.axes, failure=failure
                            ).model_dump_json()
                            + "\n"
                        )
                        manifest.flush()
                    continue
                ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
                signal.alarm(WORKLOAD_TIMEOUT_SECONDS)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.where = self.directory / re.sub(r"[^\w=,.-]", "_", axes_text(workload.axes))
                self.where.mkdir(parents=True, exist_ok=True)
                self.mutable = set(schema.mutable_tensors)
                try:
                    self.install(schema)
                    drawn = ContractWorkload.of(schema, workload.axes)
                    call = drawn.assemble_call(drawn.draw("trial 0", device), device)
                    contract = ContractRegistry.default().get(schema.contract_id)
                    for output in contract.allocated_outputs:
                        dtype = semantics.torch_dtypes(drawn.point.dtypes[output])[0]
                        call[output] = torch.empty(
                            contract.storage_shape_of(output, drawn.point), dtype=dtype, device=device
                        )
                    self.tensors = {name: value for name, value in call.items() if torch.is_tensor(value)}
                    runnable = BuilderRegistry.get_instance().build(schema)
                    self.prepare(runnable)
                    runnable(**call)
                except Exception as e:
                    self.records.append(
                        {"failure": f"compile: {type(e).__name__}: {' '.join(str(e).split())[:300]}"}
                    )
                for fields in self.records or [
                    {"failure": f"compile: the call launched no {schema.kernel.language} kernel"}
                ]:
                    manifest.write(
                        Compilation(
                            schema_path=self.schema_path, axes=workload.axes, **fields
                        ).model_dump_json()
                        + "\n"
                    )
                    manifest.flush()
                os._exit(0)

    def record(
        self,
        isa: str,
        kernel: str,
        grid: Any,
        block: Any,
        shared: int,
        values: list[Any] | None = None,
        kernarg: bytes | None = None,
    ) -> None:
        path = self.where / f"{len(self.records)}.s"
        path.write_text(isa, encoding="utf-8")
        fields: dict[str, Any] = {"kernel": kernel, "isa_path": str(path)}
        try:
            fields["launch"] = self.launch(isa, kernel, grid, block, shared, values, kernarg)
        except (ValueError, KeyError, TypeError, struct.error) as e:
            fields["failure"] = f"launch: {' '.join(str(e).split())[:300]}"
        self.records.append(fields)

    def launch(
        self,
        isa: str,
        kernel: str,
        grid: Any,
        block: Any,
        shared: int,
        values: list[Any] | None,
        kernarg: bytes | None,
    ) -> dict[str, Any]:
        metadata = kernel_metadata(isa, kernel)
        if any(not 1 <= len(dimensions) <= 3 for dimensions in (grid, block)):
            raise ValueError("invalid launch dimensions")
        grid, block = [list(dimensions) + [1] * (3 - len(dimensions)) for dimensions in (grid, block)]
        if len(grid) != 3 or len(block) != 3 or any(not isinstance(v, int) or v < 0 for v in grid + block):
            raise ValueError("invalid launch dimensions")
        size = metadata[".kernarg_segment_size"]
        data = bytearray(-(-size // 64) * 64)
        if kernarg is not None:
            data[: len(kernarg)] = kernarg
        explicit = [arg for arg in metadata.get(".args", []) if not arg[".value_kind"].startswith("hidden_")]
        if values is not None and len(values) != len(explicit):
            raise ValueError(f"{len(values)} arguments for {len(explicit)} ISA parameters")
        arrays: dict[str, Any] = {}
        pointers: dict[int, str] = {}
        for index, arg in enumerate(explicit):
            offset, width, kind = arg[".offset"], arg[".size"], arg[".value_kind"]
            value = values[index] if values is not None else None
            if kind == "global_buffer":
                pointer = (
                    value.data_ptr()
                    if hasattr(value, "data_ptr")
                    else (
                        value
                        if values is not None
                        else int.from_bytes(data[offset : offset + width], "little")
                    )
                )
                if pointer == 0:
                    data[offset : offset + width] = b"\0" * width
                    continue
                tensor = (
                    value
                    if hasattr(value, "data_ptr")
                    else next((t for t in self.tensors.values() if t.data_ptr() == pointer), None)
                )
                if tensor is None:
                    raise ValueError(f"unknown pointer at kernarg offset {offset}")
                known = next((n for n, t in self.tensors.items() if t.data_ptr() == pointer), None)
                if known is None and any(
                    tensor.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
                    for original in self.tensors.values()
                ):
                    raise ValueError("an aliased tensor view cannot be modeled as an independent array")
                name = known or f"workspace{pointer:x}"
                dtype = _DTYPES[str(tensor.dtype).removeprefix("torch.")]
                shape = (
                    list(tensor.shape)
                    if tensor.is_contiguous()
                    else [
                        1
                        + sum(
                            (n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride(), strict=True)
                        )
                    ]
                )
                arrays[name] = {
                    "name": name,
                    "kind": "out"
                    if known in self.mutable or (known is None and pointer not in self.workspaces)
                    else "in",
                    "dtype": dtype,
                    "shape": shape,
                }
                if known is None:
                    self.workspaces.add(pointer)
                if (
                    known is not None
                    and arrays[name]["kind"] == "in"
                    and dtype == "i32"
                    and tensor.numel() <= 1 << 16
                ):
                    arrays[name]["contents"] = tensor.flatten().tolist()
                pointers[offset] = name
            elif kind == "by_value" and values is not None:
                if isinstance(value, float) and width in (4, 8):
                    struct.pack_into("<f" if width == 4 else "<d", data, offset, value)
                elif isinstance(value, (int, bool)):
                    data[offset : offset + width] = (int(value) & ((1 << (8 * width)) - 1)).to_bytes(
                        width, "little"
                    )
                else:
                    raise ValueError(f"unsupported {width}-byte scalar {type(value).__name__}")
            elif kind != "by_value":
                raise ValueError(f"unsupported kernarg kind {kind}")
        for arg in metadata.get(".args", []):
            kind, offset, width = arg[".value_kind"], arg[".offset"], arg[".size"]
            value = None
            if kind.startswith("hidden_block_count_"):
                value = grid["xyz".index(kind[-1])]
            elif kind.startswith("hidden_group_size_"):
                value = block["xyz".index(kind[-1])]
            elif kind == "hidden_grid_dims":
                value = max((i + 1 for i, v in enumerate(grid) if v > 1), default=1)
            elif kind.startswith("hidden_remainder_"):
                value = 0
            elif kind.startswith("hidden_"):
                raise ValueError(f"unsupported kernarg kind {kind}")
            if value is not None:
                data[offset : offset + width] = value.to_bytes(width, "little")
        params = []
        offset = 0
        while offset < len(data):
            if offset in pointers:
                params.append({"ptr": pointers[offset]})
                offset += 8
            else:
                params.append({"int": struct.unpack_from("<i", data, offset)[0]})
                offset += 4
        return {
            "kernel": kernel,
            "grid": grid,
            "block": block,
            "dynamic_shared_bytes": shared,
            "arrays": list(arrays.values()),
            "params": params,
        }
