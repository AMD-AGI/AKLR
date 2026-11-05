###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from pydantic import TypeAdapter

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import RuntimeEnv
from aklr.core.schema._json_utils import load_json_file, save_json_file
from aklr.core.schema._schema import Schema
from aklr.core.schema._workload import Workload

OPS_DIR: Final = "aklr/ops"
# the only fields in which two files of one schema name may differ
_DATA_FIELDS: Final = {"kernel", "workloads"}


@dataclass(frozen=True)
class SchemaLayout(ABC):
    root: Path

    @abstractmethod
    def paths(self) -> list[Path]: ...

    @abstractmethod
    def path(self, schema: Schema) -> Path: ...


@dataclass(frozen=True)
class TraceDumpLayout(SchemaLayout):
    """Where tracing dumps schemas (AKLR_SCHEMA_PATH); dispatch reads its kernels from here too."""

    def paths(self) -> list[Path]:
        return sorted((self.root / "schemas").rglob("*.json"))

    def path(self, schema: Schema) -> Path:
        return self.root / "schemas" / schema.op_type / schema.name / f"{schema.uuid}.json"


@dataclass(frozen=True)
class DefaultLayout(SchemaLayout):
    def paths(self) -> list[Path]:
        return sorted(self.root.glob("*/impls/*/*/*.json"))

    def path(self, schema: Schema) -> Path:
        if schema.kernel is not None:
            raise ValueError(f"{schema.name}: DefaultLayout places kernel-less schemas only")
        files = sorted(self.root.glob(f"*/impls/*/{schema.name}/*.json"))
        if not files:
            raise ValueError(f"{schema.name}: not in {self.root}; set AKLR_SCHEMA_PATH to record it")
        path = next((p for p in files if load_json_file(Schema, p).kernel is None), None)
        if path is not None:
            return path
        op = files[0].relative_to(self.root).parts[0]
        return self.root / op / "impls" / "flydsl" / schema.name / f"{schema.uuid}.json"


@dataclass
class SchemaCollection:
    """A dataset's schemas by name; files of one name differ only in their kernel and workloads."""

    layout: SchemaLayout | None = None
    schemas: dict[str, list[Schema]] = field(default_factory=dict)
    _schema_by_kernel: dict[str, Schema] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        for schemas_list in self.schemas.values():
            for schema in schemas_list:
                if schema.kernel is None:
                    continue
                if schema.kernel.name in self._schema_by_kernel:
                    raise ValueError(f"Duplicate kernel name found: {schema.kernel.name}")
                self._schema_by_kernel[schema.kernel.name] = schema

    @classmethod
    def from_path(cls: type[SchemaCollection], path: str | None = None) -> SchemaCollection:
        """Load path, else AKLR_SCHEMA_PATH, as a TraceDumpLayout; with neither, aklr/ops as a DefaultLayout."""
        root = Path(path) if path else RuntimeEnv().schema_path
        checkout = Path(__file__).resolve().parents[3]
        return cls.from_layout(TraceDumpLayout(root) if root else DefaultLayout(checkout / OPS_DIR))

    @classmethod
    def from_layout(cls: type[SchemaCollection], layout: SchemaLayout) -> SchemaCollection:
        layout.root.mkdir(parents=True, exist_ok=True)
        schemas: dict[str, list[Schema]] = {}

        schema_adapter = TypeAdapter(Schema)
        for p in layout.paths():
            s = load_json_file(Schema, p)
            same_name = schemas.setdefault(s.name, [])
            if same_name and schema_adapter.dump_python(
                same_name[0], exclude=_DATA_FIELDS
            ) != schema_adapter.dump_python(s, exclude=_DATA_FIELDS):
                raise ValueError(f"{p}: schema {s.name} differs from its other files")
            same_name.append(s)

        return cls(layout=layout, schemas=schemas)

    def to_dict(self) -> dict[str, Any]:
        schema_adapter = TypeAdapter(Schema)
        return {
            "schemas": {
                name: [schema_adapter.dump_python(schema, mode="json") for schema in schemas]
                for name, schemas in self.schemas.items()
            },
        }

    def get_schema(self, kernel_name: str) -> Schema | None:
        return self._schema_by_kernel.get(kernel_name)

    def filter_schemas(
        self, name: str, target: DeviceTarget, min_speedup: float, baseline: str
    ) -> list[tuple[Schema, Workload]]:
        """Each (schema, workload) of name that dispatch may serve on target, see Workload.qualifies."""
        return [
            (s, w)
            for s in self.schemas.get(name, [])
            if s.kernel is not None and s.kernel.build.runs_on(target)
            for w in s.workloads
            if w.qualifies(target, min_speedup, baseline)
        ]

    def add_workloads(self, workloads: list[tuple[Schema, Workload]]) -> None:
        """Add each workload to its name's kernel-less schema, made if missing, and save it to the layout."""
        updated: dict[str, Schema] = {}
        for schema, workload in workloads:
            same_name = self.schemas.setdefault(schema.name, [])
            index = next((i for i, s in enumerate(same_name) if s.kernel is None), None)
            if index is None:
                same_name.append(replace(same_name[0] if same_name else schema, kernel=None, workloads=[]))
                index = len(same_name) - 1
            if workload not in same_name[index].workloads:
                same_name[index] = replace(
                    same_name[index], workloads=[*same_name[index].workloads, workload]
                )
            updated[schema.name] = same_name[index]

        if self.layout is not None:
            for schema in updated.values():
                save_json_file(schema, self.layout.path(schema))
