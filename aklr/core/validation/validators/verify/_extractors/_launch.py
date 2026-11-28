# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import re
import struct
from typing import Any, ClassVar, Final

__all__ = ["KernelLaunch", "allocations", "kernel_arguments"]

_KERNARG: Final = re.compile(
    r"\.offset:\s+(\d+)\n\s+\.size:\s+(\d+)\n(?:\s+\.type_name:[^\n]*\n)?\s+\.value_kind:\s+(\w+)"
)
_CONTENTS_LIMIT: Final = 1 << 16


def _metadata(isa: str, kernel: str) -> str:
    entries = [
        entry
        for entry in re.split(r"\n  - ", isa[isa.index("amdhsa.kernels:") :])[1:]
        if re.search(rf"\n    \.name:\s+{re.escape(kernel)}\n", "\n    " + entry)
    ]
    if len(entries) != 1:
        raise ValueError(f"{len(entries)} kernels named {kernel} in the ISA's metadata")
    return entries[0]


def kernel_arguments(isa: str, kernel: str) -> list[tuple[int, int, str]]:
    return [(int(offset), int(size), kind) for offset, size, kind in _KERNARG.findall(_metadata(isa, kernel))]


def allocations() -> list[tuple[int, int]]:
    import torch

    if not torch.cuda.is_initialized():
        return []
    return [
        (block["address"], block["address"] + (block.get("requested_size") or block["size"]))
        for segment in torch.cuda.memory_snapshot()
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    ]


class KernelLaunch:
    _DTYPES: ClassVar[dict[str, str]] = {
        "bfloat16": "bf16",
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "float8_e4m3fn": "e4m3",
        "float8_e4m3fnuz": "e4m3",
        "float8_e5m2": "e5m2",
        "float8_e5m2fnuz": "e5m2",
        "bool": "u8",
        "int8": "i8",
        "uint8": "u8",
        "int16": "i16",
        "uint16": "i16",
        "int32": "i32",
        "uint32": "i32",
        "int64": "i64",
        "uint64": "i64",
    }

    def __init__(
        self,
        isa: str,
        kernel: str,
        grid: list[int],
        block: list[int],
        dynamic_shared_bytes: int,
        tensors: dict[str, Any],
        mutable: set[str],
        workspaces: set[int],
        allocated: list[tuple[int, int]] | None = None,
    ) -> None:
        metadata = _metadata(isa, kernel)
        kernargs = kernel_arguments(isa, kernel)
        size = int(re.search(r"\.kernarg_segment_size:\s+(\d+)", metadata)[1])
        self.data = bytearray(-(-size // 64) * 64)
        self.explicit = [k for k in kernargs if not k[2].startswith("hidden_")]
        self._kernel, self._grid, self._block = kernel, grid, block
        self._dynamic_shared_bytes = dynamic_shared_bytes
        self._tensors, self._mutable, self._workspaces = tensors, mutable, workspaces
        self._arrays: dict[str, dict[str, Any]] = {}
        self._fresh: set[int] = set()
        self._pointers: dict[int, str] = {}
        self._live: list[Any] | None = None
        self._blocks = allocated
        counts = dict(zip("xyz", grid, strict=True))
        sizes = dict(zip("xyz", block, strict=True))
        for offset, width, kind in kernargs:
            value = None
            if kind.startswith("hidden_block_count_"):
                value = counts[kind[-1]]
            elif kind.startswith("hidden_group_size_"):
                value = sizes[kind[-1]]
            elif kind == "hidden_grid_dims":
                value = sum(extent > 1 for extent in grid) or 1
            elif kind == "hidden_dynamic_lds_size":
                value = dynamic_shared_bytes
            if value is not None:
                self.data[offset : offset + width] = value.to_bytes(width, "little")

    def tensor(self, offset: int, tensor: Any) -> None:
        known = next((n for n, t in self._tensors.items() if t.data_ptr() == tensor.data_ptr()), None)
        if known is None and tensor.data_ptr() not in self._workspaces:
            self._fresh.add(tensor.data_ptr())
        array = known or f"workspace{offset}"
        dtype = self._DTYPES.get(str(tensor.dtype).removeprefix("torch."))
        if dtype is None:
            raise ValueError(f"array `{array}` is {tensor.dtype}, which has no verify dtype")
        self._arrays[array] = {
            "name": array,
            "kind": "out" if known in self._mutable or (known or tensor.data_ptr()) in self._fresh else "in",
            "dtype": dtype,
            "shape": list(tensor.shape)
            if tensor.is_contiguous()
            else [1 + sum((n - 1) * step for n, step in zip(tensor.shape, tensor.stride(), strict=True))],
        }
        if (
            known is not None
            and self._arrays[array]["kind"] == "in"
            and dtype == "i32"
            and tensor.numel() <= _CONTENTS_LIMIT
        ):
            self._arrays[array]["contents"] = tensor.flatten().tolist()
        self._pointers[offset] = array
        struct.pack_into("<Q", self.data, offset, tensor.data_ptr())

    def address(self, offset: int, address: int) -> bool:
        tensor = next((t for t in self._tensors.values() if t.data_ptr() == address), None)
        if tensor is None:
            if self._live is None:
                import gc

                import torch

                self._live = [o for o in gc.get_objects() if isinstance(o, torch.Tensor)]
            tensor = max(
                (t for t in self._live if t.data_ptr() == address and t.numel()),
                key=lambda t: t.numel(),
                default=None,
            )
        if tensor is not None:
            self.tensor(offset, tensor)
            return True
        end = next((end for start, end in self._allocations() if start <= address < end), None)
        if end is None:
            return False
        if address not in self._workspaces:
            self._fresh.add(address)
        name = f"workspace{offset}"
        kind = "out" if address in self._fresh else "in"
        self._arrays[name] = {"name": name, "kind": kind, "dtype": "i32", "shape": [(end - address) // 4]}
        self._pointers[offset] = name
        struct.pack_into("<Q", self.data, offset, address)
        return True

    def buffer(self, offset: int, name: str, n_bytes: int) -> None:
        self._arrays[name] = {"name": name, "kind": "out", "dtype": "i32", "shape": [-(-n_bytes // 4)]}
        self._pointers[offset] = name

    def value(self, offset: int, width: int, value: Any) -> None:
        if isinstance(value, float) and width in (4, 8):
            struct.pack_into("<f" if width == 4 else "<d", self.data, offset, value)
        elif isinstance(value, int) and width in (1, 2, 4, 8):
            self.data[offset : offset + width] = (value & ((1 << 8 * width) - 1)).to_bytes(width, "little")
        else:
            raise ValueError(f"a {width}-byte kernel argument for {type(value).__name__}")

    def fields(self) -> dict[str, Any]:
        self._workspaces |= self._fresh
        params: list[dict[str, Any]] = []
        offset = 0
        while offset < len(self.data):
            if offset in self._pointers:
                params.append({"ptr": self._pointers[offset]})
                offset += 8
            else:
                params.append({"int": struct.unpack_from("<i", self.data, offset)[0]})
                offset += 4
        return {
            "kernel": self._kernel,
            "grid": self._grid,
            "block": self._block,
            "dynamic_shared_bytes": self._dynamic_shared_bytes,
            "arrays": list(self._arrays.values()),
            "params": params,
        }

    def _allocations(self) -> list[tuple[int, int]]:
        if self._blocks is None:
            self._blocks = allocations()
        return self._blocks
