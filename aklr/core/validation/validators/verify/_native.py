# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import yaml

from aklr.core.validation.validators.verify._recording import RecordingCompiler, kernel_metadata


def _llvm(name: str) -> str:
    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm" / "bin"
    path = str(root / name) if (root / name).is_file() else shutil.which(name)
    if path is None:
        raise RuntimeError(f"ROCm tool {name} is unavailable")
    if not Path(path).is_file():
        raise RuntimeError(f"ROCm tool {name} is unavailable")
    return path


def _run(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode:
        raise ValueError(f"{Path(args[0]).name}: {result.stderr.strip()[:240]}")
    return result.stdout


def _unbundle(path: Path, arch: str) -> Path:
    if path.read_bytes().startswith(b"\x7fELF"):
        return path
    bundler = _llvm("clang-offload-bundler")
    targets = _run(bundler, "-list", "-type=o", f"-input={path}").splitlines()
    matching = [
        t
        for t in targets
        if t.startswith(("hip-", "hipv4-")) and t.partition("--")[2].partition(":")[0] == arch
    ]
    if len(matching) != 1:
        raise ValueError(f"{len(matching)} code objects for {arch} in the HIP bundle")
    output = path.with_suffix(".elf")
    _run(bundler, "-unbundle", "-type=o", f"-targets={matching[0]}", f"-input={path}", f"-output={output}")
    return output


def _symbols(data: bytes) -> dict[str, tuple[int, int, int, int]]:
    if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", data, 18)[0] != 224:
        raise ValueError("expected a little-endian ELF64 AMDGPU code object")
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count = struct.unpack_from("<HH", data, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride) for i in range(count)]
    symbols = {}
    for section in sections:
        if section[1] not in (2, 11):
            continue
        strings_section = sections[section[6]]
        strings = data[strings_section[4] : strings_section[4] + strings_section[5]]
        for position in range(section[4], section[4] + section[5], section[9]):
            name, info, _, index, address, size = struct.unpack_from("<IBBHQQ", data, position)
            if not name or not 0 < index < len(sections):
                continue
            owner = sections[index]
            start = owner[4] + address - owner[3]
            symbols[strings[name:].partition(b"\0")[0].decode()] = (address, size, start, info & 15)
    return symbols


def _isa(path: Path, arch: str) -> str:
    path = _unbundle(path, arch)
    data = path.read_bytes()
    symbols = _symbols(data)
    notes = _run(_llvm("llvm-readobj"), "--notes", str(path))
    found = re.findall(r"AMDGPU Metadata: (---\n.*?\n\.\.\.)", notes, re.S)
    if len(found) != 1:
        raise ValueError("expected one AMDGPU metadata note")
    metadata = yaml.safe_load(found[0])
    target = metadata.get("amdhsa.target", "")
    if target.partition("--")[2].partition(":")[0] != arch:
        raise ValueError(f"code object target {target!r} does not match {arch}")
    descriptors = []
    entries = {}
    for kernel in metadata["amdhsa.kernels"]:
        symbol = symbols.get(kernel[".symbol"])
        if symbol is None or symbol[1] != 64:
            raise ValueError(f"no 64-byte descriptor for {kernel['.name']}")
        start = symbol[2]
        entry = symbol[0] + struct.unpack_from("<q", data, start + 16)[0]
        code = symbols.get(kernel[".name"])
        if code is None or code[0] != entry:
            raise ValueError(f"descriptor entry for {kernel['.name']} does not match its kernel symbol")
        entries[entry] = kernel[".name"]
        group, private, kernarg = struct.unpack_from("<III", data, start)
        rsrc3 = struct.unpack_from("<I", data, start + 44)[0]
        rsrc1, rsrc2, properties, preload = struct.unpack_from("<IIHH", data, start + 48)
        if preload & 0x7F:
            raise ValueError("preloaded kernarg registers are unsupported")
        if properties & (1 << 10) or kernel[".wavefront_size"] != 64:
            raise ValueError("native verification requires wave64 kernels")
        if properties & (1 << 11) or kernel.get(".uses_dynamic_stack"):
            raise ValueError("dynamic stack kernels are unsupported")
        fields = {
            "group_segment_fixed_size": group,
            "private_segment_fixed_size": private,
            "kernarg_size": kernarg,
            "next_free_vgpr": kernel[".vgpr_count"],
            "next_free_sgpr": kernel[".sgpr_count"],
            "system_vgpr_workitem_id": (rsrc2 >> 11) & 3,
            "system_sgpr_workgroup_id_x": (rsrc2 >> 7) & 1,
            "system_sgpr_workgroup_id_y": (rsrc2 >> 8) & 1,
            "system_sgpr_workgroup_id_z": (rsrc2 >> 9) & 1,
            "system_sgpr_workgroup_info": (rsrc2 >> 10) & 1,
            "user_sgpr_count": (rsrc2 >> 1) & 31,
            "dx10_clamp": (rsrc1 >> 21) & 1,
            "ieee_mode": (rsrc1 >> 23) & 1,
            "fp16_overflow": (rsrc1 >> 26) & 1,
            "float_round_mode_32": (rsrc1 >> 12) & 3,
            "float_round_mode_16_64": (rsrc1 >> 14) & 3,
            "float_denorm_mode_32": (rsrc1 >> 16) & 3,
            "float_denorm_mode_16_64": (rsrc1 >> 18) & 3,
        }
        for bit, name in enumerate(
            (
                "private_segment_buffer",
                "dispatch_ptr",
                "queue_ptr",
                "kernarg_segment_ptr",
                "dispatch_id",
                "flat_scratch_init",
                "private_segment_size",
            )
        ):
            fields[f"user_sgpr_{name}"] = (properties >> bit) & 1
        if arch in ("gfx90a", "gfx942", "gfx950"):
            fields["accum_offset"] = ((rsrc3 & 63) + 1) * 4
            fields["tg_split"] = (rsrc3 >> 16) & 1
            fields["enable_private_segment"] = rsrc2 & 1
            fields.pop("user_sgpr_private_segment_buffer")
            fields.pop("user_sgpr_flat_scratch_init")
        else:
            fields["system_sgpr_private_segment_wavefront_offset"] = rsrc2 & 1
        descriptors.append(
            f".amdhsa_kernel {kernel['.name']}\n"
            + "".join(f"  .amdhsa_{name} {value}\n" for name, value in fields.items())
            + ".end_amdhsa_kernel\n"
        )
    dump = _run(_llvm("llvm-objdump"), "-d", "--no-show-raw-insn", f"--mcpu={arch}", str(path))
    boundaries = {symbol[0] for symbol in symbols.values() if symbol[3] in (2, 10)} | entries.keys()
    text = [f'.amdgcn_target "{target}"']
    for line in dump.splitlines():
        header = re.fullmatch(r"([0-9a-fA-F]+) <(.+)>:", line)
        if header:
            address = int(header[1], 16)
            if address in boundaries:
                text.extend((".section .text", f"{entries.get(address, header[2])}:"))
            else:
                text.append(f".Lnative_symbol_{address:x}:")
            continue
        instruction = re.match(r"\s+(.*?)\s+// ([0-9A-Fa-f]+):", line)
        if instruction is None:
            continue
        operation, address = instruction[1], int(instruction[2], 16)
        branch = re.fullmatch(r"(s_(?:cbranch_\w+|branch))\s+(\d+)", operation)
        if branch:
            displacement = int(branch[2])
            if displacement & 0x8000:
                displacement -= 0x10000
            operation = f"{branch[1]} .Lnative_{address + 4 + displacement * 4:x}"
        text.extend((f".Lnative_{address:x}:", "  " + operation))
    if not text or any("<unknown>" in line for line in text):
        raise ValueError("code object disassembly contains unsupported instructions")
    return "\n".join(
        text
        + ['.section .rodata,"a",@progbits']
        + descriptors
        + [".amdgpu_metadata", found[0], ".end_amdgpu_metadata", ""]
    )


def _validate_launch(grid, block, shared) -> None:
    if (
        any(
            not 1 <= len(dims) <= 3 or any(type(v) is not int or not 0 < v < 1 << 32 for v in dims)
            for dims in (grid, block)
        )
        or type(shared) is not int
        or not 0 <= shared < 1 << 32
    ):
        raise ValueError("invalid native launch dimensions or shared memory")


class HsacoCompiler(RecordingCompiler):
    def install(self, schema: Any) -> None:
        import torch

        from aklr.core.jit.hsaco import Kernel
        from aklr.core.runtime.builder import BuilderRegistry, HsacoBuilder

        if not torch.cuda.is_available():
            BuilderRegistry.set_instance(BuilderRegistry([HsacoBuilder()]))
        cache = {}

        def launch(kernel, grid, block, kernarg, *, shared_memory=0, device=None):
            _validate_launch(grid, block, shared_memory)
            code = kernel._code_object
            digest = hashlib.sha256(code).hexdigest()
            if digest not in cache:
                path = self.where / f"{digest}.hsaco"
                path.write_bytes(code)
                cache[digest] = _isa(path, self.arch)
            isa = cache[digest]
            metadata = kernel_metadata(isa, kernel._symbol)
            explicit = [a for a in metadata.get(".args", []) if not a[".value_kind"].startswith("hidden_")]
            required = max((a[".offset"] + a[".size"] for a in explicit), default=0)
            if len(kernarg) < required or len(kernarg) > metadata[".kernarg_segment_size"]:
                raise ValueError("kernarg buffer size does not match code object metadata")
            self.record(isa, kernel._symbol, grid, block, shared_memory, kernarg=bytes(kernarg))

        Kernel.launch = launch


_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, *[ctypes.c_uint] * 6, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)
)


class HipCompiler(RecordingCompiler):
    @staticmethod
    def environment(arch: str, directory: Path) -> dict[str, str]:
        directory.mkdir(parents=True, exist_ok=True)
        library = directory / "hip_capture.so"
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("a C++ compiler is required for HIP launch capture")
        _run(
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            str(Path(__file__).with_name("_hip_capture.cpp")),
            "-o",
            str(library),
            "-ldl",
            "-pthread",
        )
        preload = " ".join(filter(None, [str(library), os.environ.get("LD_PRELOAD")]))
        return RecordingCompiler.environment(arch, directory) | {"LD_PRELOAD": preload}

    def install(self, schema: Any) -> None:
        self.isa = None
        self.capture = ctypes.CDLL(str(self.directory / "hip_capture.so"))
        self.capture.aklr_verify_capture.argtypes = [ctypes.c_char_p, _CALLBACK]
        self.callback = _CALLBACK(self._capture)
        self.capture.aklr_verify_capture(
            f"aklr_hip_{schema.kernel.content_hash[:16]}.so".encode(), self.callback
        )

    def prepare(self, runnable: Any) -> None:
        library = Path(runnable.metadata.misc["library"])
        fatbin = self.where / "extension.fatbin"
        _run(_llvm("llvm-objcopy"), f"--dump-section=.hip_fatbin={fatbin}", str(library))
        self.isa = _isa(fatbin, self.arch)
        self.capture.aklr_verify_capture(str(library.resolve()).encode(), self.callback)

    def _capture(self, symbol, gx, gy, gz, bx, by, bz, shared, args) -> None:
        try:
            _validate_launch((gx, gy, gz), (bx, by, bz), shared)
            if self.isa is None:
                raise ValueError("target HIP kernel launched before its binary was prepared")
            if not symbol:
                raise ValueError("unsupported or unregistered target HIP launch")
            kernel = symbol.decode()
            metadata = kernel_metadata(self.isa, kernel)
            kernarg = bytearray(metadata[".kernarg_segment_size"])
            explicit = [a for a in metadata.get(".args", []) if not a[".value_kind"].startswith("hidden_")]
            if explicit and not args:
                raise ValueError("HIP launch has no argument array")
            for index, arg in enumerate(explicit):
                if not args[index]:
                    raise ValueError("HIP launch has a null argument address")
                start, size = arg[".offset"], arg[".size"]
                kernarg[start : start + size] = ctypes.string_at(args[index], size)
            self.record(self.isa, kernel, (gx, gy, gz), (bx, by, bz), shared, kernarg=bytes(kernarg))
        except Exception as error:
            self.records.append(
                {"failure": f"compile: {type(error).__name__}: {' '.join(str(error).split())[:300]}"}
            )
