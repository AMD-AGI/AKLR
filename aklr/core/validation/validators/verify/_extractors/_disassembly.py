# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Final

__all__ = ["assembly", "code_object", "rocm_tool"]

_INSTRUCTION: Final = re.compile(r"^\s+(\S.*?)\s*// ([0-9A-Fa-f]+): [0-9A-Fa-f ]+(?:<[^>]*>)?$")
_BRANCH: Final = re.compile(r"^(s_branch|s_cbranch_\w+|s_call_b64\s+s\[\d+:\d+\],)\s+(\d+)$")
_SYMTAB: Final = 2
_DESCRIPTOR_BYTES: Final = 64
_RSRC2_OFFSET: Final = 52


def rocm_tool(name: str) -> str:
    from torch.utils import cpp_extension

    for root in filter(None, (cpp_extension.ROCM_HOME, os.environ.get("ROCM_PATH"), "/opt/rocm")):
        tool = Path(root, "llvm", "bin", name)
        if tool.is_file():
            return str(tool)
    raise FileNotFoundError(f"ROCm's {name} is not installed")


def _run(*argv: str) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout


def code_object(bundle: bytes, arch: str | None = None) -> bytes:
    if bundle.startswith(b"\x7fELF"):
        return bundle
    with tempfile.TemporaryDirectory() as directory:
        path, unbundled = Path(directory, "bundle"), Path(directory, "kernel.co")
        path.write_bytes(bundle)
        bundler = rocm_tool("clang-offload-bundler")
        listed = _run(bundler, "--list", "--type=o", f"--input={path}").split()
        targets = [
            target
            for target in listed
            if target.startswith("hip")
            and "amdgcn" in target
            and arch in (None, target.rpartition("--")[2].partition(":")[0])
        ]
        if not targets:
            raise ValueError(f"the bundle has no code object for {arch or 'an AMD GPU'}: {listed}")
        _run(
            bundler,
            "--unbundle",
            "--type=o",
            f"--input={path}",
            f"--targets={targets[0]}",
            f"--output={unbundled}",
        )
        return unbundled.read_bytes()


def _descriptors(code_object: bytes) -> dict[str, bytes]:
    section_offset = struct.unpack_from("<Q", code_object, 0x28)[0]
    entry_size, n_sections = struct.unpack_from("<HH", code_object, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", code_object, section_offset + i * entry_size)
        for i in range(n_sections)
    ]
    descriptors = {}
    for _, kind, _, _, offset, size, link, _, _, symbol_size in sections:
        if kind != _SYMTAB:
            continue
        names = sections[link][4]
        for at in range(offset, offset + size, symbol_size):
            name_at, _, _, index, value, extent = struct.unpack_from("<IBBHQQ", code_object, at)
            name = code_object[names + name_at : code_object.index(b"\0", names + name_at)].decode()
            if name.endswith(".kd") and extent == _DESCRIPTOR_BYTES:
                start = sections[index][4] + value - sections[index][3]
                descriptors[name.removesuffix(".kd")] = code_object[start : start + _DESCRIPTOR_BYTES]
    return descriptors


def assembly(elf: bytes) -> str:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "kernel.co")
        path.write_bytes(elf)
        notes = _run(rocm_tool("llvm-readobj"), "--notes", str(path))
        metadata = notes[notes.index("---\n") : notes.index("\n...\n") + 5]
        target = re.search(r"amdhsa\.target:\s+(\S+)", metadata)[1]
        mcpu = f"--mcpu={target.rpartition('--')[2].partition(':')[0]}"
        text = _run(rocm_tool("llvm-objdump"), "-d", mcpu, str(path))
        rodata = _run(rocm_tool("llvm-objdump"), "-d", "-j", ".rodata", mcpu, str(path))
    lines = [f'\t.amdgcn_target "{target}"', "\t.text"]
    for function in re.split(r"\n(?=[0-9a-f]{16} <)", text)[1:]:
        head, *body = function.strip("\n").splitlines()
        name = re.match(r"[0-9a-f]{16} <(.*)>:$", head)[1]
        instructions, targets = [], set()
        for line in body:
            if not line.strip() or line.strip() == "...":
                continue
            found = _INSTRUCTION.match(line)
            if found is None:
                raise ValueError(f"{name}: cannot read the disassembly line {line.strip()!r}")
            address, instruction = int(found[2], 16), found[1]
            branch = _BRANCH.match(instruction)
            if branch is not None:
                offset = int(branch[2])
                target_address = address + 4 + 4 * (offset - (0x10000 if offset >= 0x8000 else 0))
                targets.add(target_address)
                instruction = f"{branch[1]} .L{name}_{target_address:x}"
            instructions.append((address, instruction))
        lines += [f"\t.globl {name}", f"\t.type {name},@function", f"{name}:"]
        for address, instruction in instructions:
            if address in targets:
                lines.append(f".L{name}_{address:x}:")
            lines.append(f"\t{instruction}")
        lines.append(f".Lfunc_end_{name}:")
    lines.append("\t.rodata")
    descriptors = _descriptors(elf)
    for block in re.findall(r"^\.amdhsa_kernel .*?^\.end_amdhsa_kernel", rodata, re.MULTILINE | re.DOTALL):
        if ".amdhsa_user_sgpr_count" not in block:
            rsrc2 = struct.unpack_from("<I", descriptors[block.split()[1]], _RSRC2_OFFSET)[0]
            block = block.replace(
                "\n.end_amdhsa_kernel",
                f"\n\t.amdhsa_user_sgpr_count {(rsrc2 >> 1) & 0x1F}\n.end_amdhsa_kernel",
            )
        lines.append(block)
    lines += ["\t.amdgpu_metadata", metadata.rstrip("\n"), "\t.end_amdgpu_metadata"]
    return "\n".join(lines) + "\n"
