###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

"""A kernel function: a dataset kernel as dispatch calls it."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import torch

from aklr.core.runtime.builder.registry import BuilderRegistry
from aklr.core.runtime.builder.runnable import Runnable
from aklr.core.schema import Schema, Tensor
from aklr.core.utils import dtype_str_to_torch_dtype


class KernelFunction:
    """A schema's kernel, built on first use; called on the api's arguments, it returns outputs the api's way."""

    def __init__(self, schema: Schema) -> None:
        assert schema.kernel is not None
        self.schema = schema

    @property
    def name(self) -> str:
        return self.schema.kernel.name

    @cached_property
    def runnable(self) -> Runnable:
        return BuilderRegistry.get_instance().build(self.schema)

    def __call__(self, call: dict[str, Any], axes: dict[str, int], is_inplace: bool = False) -> Any:
        """Run the kernel on call at axes (Const and Var), writing into the output buffers call holds."""
        schema = self.schema
        out_keys = [k for k, spec in schema.outputs.items() if isinstance(spec, Tensor)]
        args = schema.ordered_inputs(call)
        buffers = {k: call[k] for k in out_keys if isinstance(call.get(k), torch.Tensor)}

        if self.runnable.metadata.destination_passing_style:
            produced = []
            for key in out_keys:
                buffer = buffers.get(key)
                if buffer is None:
                    if schema.outputs[key].optional:
                        continue
                    buffer = torch.empty(
                        schema.get_output_shapes(axes)[list(schema.outputs).index(key)],
                        dtype=dtype_str_to_torch_dtype(schema.outputs[key].dtype),
                        device=next((v.device for v in args if isinstance(v, torch.Tensor)), "cpu"),
                    )
                produced.append(buffer)
            self.runnable(*args, *produced)
        else:
            returned = self.runnable(*args)
            produced = list(returned) if isinstance(returned, (tuple, list)) else [returned]
            for i, value in enumerate(produced[: len(out_keys)]):
                buffer = buffers.get(out_keys[i])
                if isinstance(value, torch.Tensor) and buffer is not None:
                    if value.data_ptr() != buffer.data_ptr():
                        buffer.copy_(value)
                    produced[i] = buffer

        if is_inplace or not produced:
            return None
        return produced[0] if len(produced) == 1 else tuple(produced)
