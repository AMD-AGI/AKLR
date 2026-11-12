###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from aklr.core.schema._kernel import Kernel, SourceFile


def dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = getattr(torch, dtype_str, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unsupported dtype '{dtype_str}'")


def torch_dtype_to_dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@cache
def _get_integer_dtypes() -> frozenset:
    import torch

    return frozenset(
        (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.bool,
        )
    )


def is_dtype_integer(dtype: torch.dtype) -> bool:
    return dtype in _get_integer_dtypes()


def write_sources_to_path(path: Path, sources: List[SourceFile]) -> List[Path]:
    path.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for src in sources:
        src_path_obj = Path(src.path)

        assert not src_path_obj.is_absolute(), f"Absolute path detected: {src.path}"
        assert ".." not in src_path_obj.parts, f"Path traversal detected: {src.path}"

        src_path = path / src.path

        src_path.parent.mkdir(parents=True, exist_ok=True)

        src_path.write_text(src.content)
        paths.append(src_path)

    return paths


def create_package_name(kernel: Kernel, package_prefix: str = "") -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", kernel.name)
    if not s or s[0].isdigit():
        s = "_" + s

    return package_prefix + s + "_" + kernel.hash()[:6]
