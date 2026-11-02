###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter

T = TypeVar("T")


def save_json_file(object: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(TypeAdapter(type(object)).dump_python(object, mode="json"), indent=2) + "\n")


def load_json_file(model_cls: type[T], path: str | Path) -> T:
    with open(Path(path), "r", encoding="utf-8") as f:
        return TypeAdapter(model_cls).validate_json(f.read())


def save_jsonl_file(objects: list[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        object_strs = [TypeAdapter(type(obj)).dump_json(obj).decode() for obj in objects]
        output_str = "\n".join(object_strs) + "\n"
        f.write(output_str)


def load_jsonl_file(model_cls: type[T], path: str | Path) -> list[T]:
    adapter = TypeAdapter(model_cls)
    out = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(adapter.validate_json(line))
    return out


def append_jsonl_file(objects: list[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    needs_newline_prefix = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            last_char = f.read(1)
            needs_newline_prefix = last_char != b"\n"

    with open(path, "a", encoding="utf-8") as f:
        if needs_newline_prefix:
            f.write("\n")

        object_strs = [TypeAdapter(type(obj)).dump_json(obj).decode() for obj in objects]
        output_str = "\n".join(object_strs) + "\n"
        f.write(output_str)
