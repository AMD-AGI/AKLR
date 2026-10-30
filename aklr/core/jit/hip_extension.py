# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HIP C++ built into torch extensions by ``torch.utils.cpp_extension``, against the
torch of this process, and loaded with ``torch.ops.load_library``."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import filelock
import torch

__all__ = ["is_available", "load_library"]

_FLAGS: Final = ("-O3",)
# written once the library loads, so a build that died is redone
_BUILT: Final = "built"


def is_available() -> bool:
    """Whether torch is a ROCm build and ROCm's hipcc and ninja are here."""
    if torch.version.hip is None:
        return False
    from torch.utils import cpp_extension

    rocm = cpp_extension.ROCM_HOME
    return rocm is not None and Path(rocm, "bin", "hipcc").is_file() and cpp_extension.is_ninja_available()


@functools.cache
def _toolchain() -> str:
    """What a library depends on besides its sources and arch."""
    from torch.utils import cpp_extension

    rocm_version = Path(cpp_extension.ROCM_HOME, ".info", "version")
    return json.dumps(
        {
            "torch": torch.__version__,
            "rocm": (rocm_version.read_text().strip() if rocm_version.is_file() else None),
            "flags": _FLAGS,
        }
    )


def load_library(name: str, sources: Mapping[str, str], root: Path, arch: str) -> Path:
    """Load the extension ``name`` (its ``TORCH_EXTENSION_NAME``) of ``sources``, the
    text of each .hip translation unit and .h header by relative path, for ``arch``,
    built under ``root`` first on a new toolchain; returns the library's path. A build
    that fails, or a library that does not load, raises ``RuntimeError``."""
    key = hashlib.sha256(f"{_toolchain()} {arch}".encode()).hexdigest()[:16]
    build_dir = root / key
    library = build_dir / f"{name}.so"
    build_dir.mkdir(parents=True, exist_ok=True)
    # beside build_dir: torch's own lock is build_dir/lock
    with filelock.FileLock(root / f"{key}.lock"):
        if (build_dir / _BUILT).is_file():
            torch.ops.load_library(library)
        else:
            _build(name, sources, build_dir, arch)
            (build_dir / _BUILT).touch()
    return library


def _build(name: str, sources: Mapping[str, str], build_dir: Path, arch: str) -> None:
    from torch.utils import cpp_extension

    src = build_dir / "src"
    for path, text in sources.items():
        (src / path).parent.mkdir(parents=True, exist_ok=True)
        (src / path).write_text(text, encoding="utf-8")
    # left by a build that died: we hold the directory
    (build_dir / "lock").unlink(missing_ok=True)
    # an --offload-arch flag would come on top of every arch torch was built for
    previous_arch = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = arch
    try:
        cpp_extension.load(
            name=name,
            sources=[str(src / p) for p in sorted(sources) if p.endswith(".hip")],
            extra_include_paths=[str(src)],
            extra_cuda_cflags=list(_FLAGS),
            build_directory=str(build_dir),
            is_python_module=False,
        )
    except OSError as e:  # dlopen's, such as an undefined symbol: the sources' fault
        raise RuntimeError(f"{name} does not load: {e.__cause__ or e}") from e
    finally:
        if previous_arch is None:
            del os.environ["PYTORCH_ROCM_ARCH"]
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = previous_arch
