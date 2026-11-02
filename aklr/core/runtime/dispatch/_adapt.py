###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch

from aklr.core.schema import Tensor
from aklr.core.utils import dtype_str_to_torch_dtype


def ordered_input_values(template: Any, namespace: dict[str, Any]) -> list[Any]:
    """The call's input values in the schema's order, None where absent: kernels take them positionally."""
    out: list[Any] = []
    for json_key, desc in template.inputs.items():
        param = getattr(desc, "param", None) or json_key
        val = namespace.get(param)
        tuple_idx = getattr(desc, "tuple_idx", None)
        if tuple_idx is not None and val is not None:
            val = val[tuple_idx] if isinstance(val, (tuple, list)) and len(val) > tuple_idx else None
        out.append(val)
    return out


def adapt_and_call(
    *,
    template: Any,
    fn: Callable,
    namespace: dict[str, Any],
    dps: bool,
    is_inplace: bool,
    dests: dict[str, str],
    axes: dict[str, int],
) -> Any:
    """Run kernel ``fn`` on the call and return its outputs the api's way, written to any caller buffers."""
    out_keys = [k for k, d in template.outputs.items() if isinstance(d, Tensor)]
    args = ordered_input_values(template, namespace)

    def dest_buf(json_key: str):
        param = dests.get(json_key)
        if not param:
            return None
        b = namespace.get(param)
        return b if isinstance(b, torch.Tensor) else None

    if dps:
        out_bufs: dict[str, Any] = {}
        for jk in out_keys:
            buf = dest_buf(jk)
            if buf is None:
                if getattr(template.outputs[jk], "optional", False):
                    continue
                shape = template.get_output_shapes(axes)[list(template.outputs).index(jk)]
                buf = torch.empty(
                    shape,
                    dtype=dtype_str_to_torch_dtype(template.outputs[jk].dtype),
                    device=next((v.device for v in args if isinstance(v, torch.Tensor)), "cpu"),
                )
            out_bufs[jk] = buf
        fn(*args, *out_bufs.values())
        produced = list(out_bufs.values())
    else:
        ret = fn(*args)
        values = list(ret) if isinstance(ret, (tuple, list)) else [ret]
        produced = []
        for i, val in enumerate(values):
            if i < len(out_keys) and isinstance(val, torch.Tensor):
                buf = dest_buf(out_keys[i])
                if buf is not None and val.data_ptr() != buf.data_ptr():
                    buf.copy_(val)
                    val = buf
            produced.append(val)

    if is_inplace:
        return None
    if not produced:
        return None
    if len(produced) == 1:
        return produced[0]
    return tuple(produced)


def _is_inplace_api(original: Callable) -> bool:
    """Whether the api returns None, i.e. writes its results into its arguments."""
    try:
        ann = inspect.signature(original).return_annotation
        return ann is None or ann is type(None) or ann == "None"
    except (TypeError, ValueError):
        return False
