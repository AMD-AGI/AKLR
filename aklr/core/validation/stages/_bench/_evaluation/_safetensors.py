# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract import semantics
from aklr.core.validation.stages._bench._evaluation._trials import Step

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["open_candidate_outputs", "outputs_file_name", "read_step_outputs"]


def outputs_file_name(axes_text: str) -> str:
    return f"{axes_text or 'workload'}.safetensors"


def open_candidate_outputs(directory_descriptor: int, file_name: str) -> Any:
    from safetensors import safe_open

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(file_name, flags, dir_fd=directory_descriptor)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("not a regular file")
        return safe_open(f"/proc/self/fd/{descriptor}", framework="pt")
    finally:
        os.close(descriptor)


def read_step_outputs(
    candidate_file: Any,
    step: Step,
    output_shapes: Mapping[str, tuple[int, ...]],
    outputs_json: Mapping[str, Any],
) -> dict[str, Tensor]:
    actual_outputs = {}
    for o, shape in output_shapes.items():
        key, dtype = step.output_key(o), outputs_json[o]["dtype"]["any_of"][0]
        found = tuple(candidate_file.get_slice(key).get_shape())
        if found != shape:
            raise ValueError(f"the output {o} has shape {found}, not {shape} {dtype}")
        actual = actual_outputs[o] = candidate_file.get_tensor(key)
        if actual.dtype not in semantics.torch_dtypes(dtype):
            raise ValueError(f"the output {o} has {actual.dtype}, not {shape} {dtype}")
    return actual_outputs
