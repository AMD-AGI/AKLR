# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Read the fields a task needs out of one kernel-less AKLR schema: what it computes and the workloads it is scored on."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class SchemaError(ValueError):
    """The AKLR checkout does not hold the requested schema, or holds it ambiguously."""


@dataclass(frozen=True)
class Workload:
    """One scored point of a schema: its variable axes."""

    axes: dict[str, int]

    @property
    def case_id(self) -> str:
        """The workload's name in the driver's output: its axes in schema order, one token."""
        return ",".join(f"{name}={value}" for name, value in self.axes.items())


@dataclass(frozen=True)
class Schema:
    """An AKLR schema as a task sees it."""

    path: Path
    op_dir: str
    name: str
    description: str
    op_type: str
    axes: dict[str, dict]
    attrs: dict[str, object]
    inputs: dict[str, dict]
    outputs: dict[str, dict]
    compare: dict[str, object]
    workloads: tuple[Workload, ...]

    @property
    def constant_axes(self) -> dict[str, int]:
        return {name: spec["value"] for name, spec in self.axes.items() if spec.get("type") == "const"}

    @property
    def variable_axes(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self.axes.items() if spec.get("type") == "var")


def find(aklr_root: Path, name: str) -> Path:
    """The kernel-less schema file named ``name`` in an AKLR checkout."""
    pattern = f"aklr/ops/*/impls/*/{name}/*.json"
    matches = [
        path
        for path in sorted(aklr_root.glob(pattern))
        if not json.loads(path.read_text())["kernel"]["sources"]
    ]
    if not matches:
        raise SchemaError(f"no kernel-less schema {name!r} under {aklr_root} ({pattern})")
    if len(matches) > 1:
        raise SchemaError(
            f"{len(matches)} kernel-less schemas named {name!r} under {aklr_root}; expected one"
        )
    return matches[0]


def load(path: Path) -> Schema:
    """Load a schema file kept where AKLR keeps them, aklr/ops/<op>/impls/<language>/<name>/<uuid>.json: the
    op folder is where a delivered kernel goes."""
    path = path.resolve()
    if len(path.parents) < 4 or path.parents[2].name != "impls":
        raise SchemaError(f"{path}: not under aklr/ops/<op>/impls/<language>/<name>/")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        schema = Schema(
            path=path,
            op_dir=path.parents[3].name,
            name=str(data["name"]),
            description=str(data.get("description") or "").strip(),
            op_type=str(data["op_type"]),
            axes=dict(data["axes"]),
            attrs=dict(data.get("attrs") or {}),
            inputs=dict(data["inputs"]),
            outputs=dict(data["outputs"]),
            compare=dict(data.get("compare", {})),
            workloads=tuple(Workload(axes=dict(w["axes"])) for w in data["workloads"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SchemaError(f"{path}: not an AKLR schema: {error}") from None
    if not schema.workloads:
        raise SchemaError(f"{path}: the schema serves no workload")
    return schema
