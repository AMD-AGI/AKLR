# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ._distributions import _DISTRIBUTION_ADAPTER
from ._dtypes import dtype_storage_shape


def sorted_key_json(value: Any) -> str:
    """The one text of a json value: sorted keys, no whitespace, UTF-8 kept."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def derive_seed(*parts: str | int) -> int:
    """A 64-bit seed: the first 8 bytes of ``sha256(sorted_key_json(parts))``, big-
    endian."""
    digest = hashlib.sha256(sorted_key_json(list(parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def workload_hash(name: str, workload: Mapping[str, Any]) -> str:
    """First 16 hex digits of sha256 of ``sorted_key_json([name, axes, scalars])`` of a
    ``WorkloadSpec`` dump: the root of every seed of the workload, so the same Point
    and scalars get the same values whichever file serves them."""
    text = sorted_key_json([name, workload["axes"], workload.get("scalars", {})])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def default_scalar(scalar_declaration: Mapping[str, Any]) -> bool | int | float:
    """A ``ScalarSpec`` dump's default: ``default``, else ``values[0]``, else 1.0
    clipped into ``interval``."""
    if scalar_declaration.get("default") is not None:
        return scalar_declaration["default"]
    if scalar_declaration.get("values"):
        return scalar_declaration["values"][0]
    lo, hi = scalar_declaration["interval"]
    return min(max(1.0, lo), hi)


def row_major_inputs(
    schema: Mapping[str, Any],
    workload: Mapping[str, Any],
    distributions: Mapping[str, Mapping[str, Any]],
    *,
    input_set: str,
    device: str,
    rank: int = 0,
) -> dict[str, Any]:
    """Every present input of ``workload`` row-major, drawn on ``device``, in schema order;
    ``rank`` > 0 draws another rank's own inputs, rank 0 those of a one-rank call. The
    values depend on the device's generator: a GPU's differ from the CPU's and between
    GPU models."""
    import torch

    symbols = {
        name: axis["expr"] if axis["kind"] == "derived" else axis["lo"]
        for name, axis in schema["axes"].items()
        if axis["kind"] == "derived" or not axis["is_var"]
    }
    symbols.update(workload["axes"])
    root_hash = workload_hash(schema["name"], workload)
    input_set_seed = derive_seed(root_hash, input_set, *([rank] if rank else []))
    scalars = workload.get("scalars", {})
    inputs = {}
    for name, declaration in schema["inputs"].items():
        if name in scalars:
            is_own = input_set == "workload"
            inputs[name] = scalars[name] if is_own else default_scalar(declaration)
        elif name in distributions:
            dims = [d if isinstance(d, int) else symbols[d] for d in declaration["shape"]]
            dtype = declaration["dtype"]["any_of"][0]
            seed = derive_seed(input_set_seed, "initialize", name)
            generator = torch.Generator(device).manual_seed(seed)
            try:
                with torch.device(device):
                    inputs[name] = _DISTRIBUTION_ADAPTER.validate_python(distributions[name]).sample(
                        dtype_storage_shape(dtype, dims), dtype, generator
                    )
            except ValueError as e:
                raise ValueError(f"{name}: {e}") from None
    return inputs
