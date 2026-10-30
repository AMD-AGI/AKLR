# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SH0-SH5, the audit of a hip kernel: lexical, on each source's text with comments and
literals blanked. Writes of an input, launches off the current stream and state that
changes an output are caught by bench, not here."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Final

from aklr.core.validation.stages._report._finding import Finding, FindingCode

if TYPE_CHECKING:
    from aklr.core.contract.spec import KernelSpec

__all__ = ["check_hip_kernel"]

_COMMENT_OR_LITERAL: Final = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL
)
# SH1: a library registers only in its build's namespace (TORCH_EXTENSION_NAME), as a
# second library of one namespace aborts the process that loads it
_LIBRARY_MACRO: Final = re.compile(r"\b(TORCH_LIBRARY\w*|PYBIND11_MODULE)\b")
_EXPAND_DEFINITION: Final = re.compile(
    r"#\s*define\s+TORCH_LIBRARY_EXPAND\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)"
    r"\s*TORCH_LIBRARY\s*\(\s*\1\s*,\s*\2\s*\)"
)
_REGISTRATION: Final = re.compile(r"\bTORCH_LIBRARY_EXPAND\s*\(\s*TORCH_EXTENSION_NAME\s*,\s*\w+\s*\)")
# SH2
_INCLUDE: Final = re.compile(r"^[ \t]*#[ \t]*include\b", re.MULTILINE)
_INCLUDED: Final = re.compile(r"#\s*include\s*(?:<([^>\n]+)>|\"([^\"\n]+)\")")
_HEADER_PREFIXES: Final = ("ATen/", "c10/", "hip/")
_HEADERS: Final = frozenset(
    {
        *("torch/library.h", "algorithm", "array", "bit", "cfloat", "climits"),
        *("cmath", "cstddef", "cstdint", "cstring", "functional", "initializer_list"),
        *("iterator", "limits", "numeric", "optional", "string", "tuple"),
        *("type_traits", "utility", "vector"),
    }
)
# SH3: synchronizing, allocating outside torch, mapping memory, changing the device or the
# process, I/O, and CUDA's names (hipify would rewrite them)
_FORBIDDEN: Final = re.compile(
    r"\b(hip(?:DeviceSynchronize|StreamSynchronize|EventSynchronize|SetDevice"
    r"|DeviceReset|DeviceEnablePeerAccess|LaunchHostFunc|StreamAddCallback)"
    r"|hip(?:Malloc|Free|HostMalloc|HostFree|ExtMalloc|Ipc|StreamCreate|Graph|Module)\w*"
    r"|hip(?:Memcpy|Memset)(?!\w*Async)\w*|cuda[A-Z]\w*"
    r"|getenv|secure_getenv|setenv|putenv|system|popen|fopen|freopen|dlopen"
    r"|printf|fprintf|puts|cout|cerr|clog|exit|quick_exit|abort)\b"
)
# SH4: memory kept across calls: a static or __device__ variable that is not const
_STATE: Final = re.compile(
    r"\b(?:__managed__|thread_local)\b"
    r"|\b(?:static|__device__)\b(?![^;{}()]*\b(?:const|constexpr|operator)\b)"
    r"[^;{}()]*[;=]"
)


def check_hip_kernel(schema_path: str, kernel: KernelSpec) -> tuple[Finding, ...]:
    """Every SH0-SH5 rejection of the hip kernel of the schema at ``schema_path``."""
    sources = {source.path: source.content for source in kernel.sources}
    rejections = {
        (path, line, rule, what)
        for path, text in sources.items()
        for rule, line, what in _rejections(path, text, sources, kernel.entry_file)
    }
    return tuple(
        Finding(
            code=FindingCode.SOURCE_REJECTED,
            schema_path=schema_path,
            axes=None,
            detail=f"SH{rule} {path}:{line} {what}",
        )
        for path, line, rule, what in sorted(rejections)
    )


def _rejections(
    path: str, text: str, sources: Mapping[str, str], entry_file: str
) -> Iterator[tuple[int, int, str]]:
    """(rule, line, what) of each rejection in one source."""
    blanked = _COMMENT_OR_LITERAL.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)

    def line(offset: int) -> int:
        return blanked.count("\n", 0, offset) + 1

    if unterminated := re.search(r"/\*|\"", blanked):
        yield 0, line(unterminated.start()), "unterminated comment or literal"
        return
    allowed = [m.span() for m in _EXPAND_DEFINITION.finditer(blanked)]
    registrations = list(_REGISTRATION.finditer(blanked)) if path == entry_file else []
    allowed += [m.span() for m in registrations]
    for m in _LIBRARY_MACRO.finditer(blanked):
        if not any(start <= m.start() < end for start, end in allowed):
            what = f"`{m[0]}` outside TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m)"
            yield 1, line(m.start()), what + " in the entry file"
    if path == entry_file and len(registrations) != 1:
        at = line(registrations[1].start()) if registrations else 1
        what = f"registers {len(registrations)} times, not once, with"
        yield 1, at, what + " TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m)"
    lines = text.splitlines()
    for m in _INCLUDE.finditer(blanked):
        n = line(m.start())
        included = _INCLUDED.match(lines[n - 1].strip())
        header, local = included.groups() if included else (None, None)
        if local is not None and any(
            posixpath.normpath(p) in sources for p in (posixpath.join(posixpath.dirname(path), local), local)
        ):
            continue
        if header is not None and (header.startswith(_HEADER_PREFIXES) or header in _HEADERS):
            continue
        yield 2, n, f"`{lines[n - 1].strip()}`: neither a source nor an allowed header"
    for m in _FORBIDDEN.finditer(blanked):
        yield 3, line(m.start()), f"uses `{m[0]}`"
    for m in _STATE.finditer(blanked):
        what = f"keeps state across calls: `{' '.join(m[0].split())}`"
        yield 4, line(m.start()), what
    if path == entry_file and not re.match(r"[ \t]*(//[ \t]*\S|/\*)", text):
        yield 5, 1, "does not begin with a comment describing the kernel"
