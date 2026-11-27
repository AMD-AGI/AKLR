###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import contextlib
import shutil
import time
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Final

from pydantic import SerializeAsAny, TypeAdapter

from aklr.core.common.frozen_model import frozen_dataclass
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.platform.device import DeviceTarget
from aklr.core.validation.validators import (
    BenchReport,
    Block,
    Heading,
    Paragraph,
    Progress,
    Table,
    ValidationScope,
    Validator,
    ValidatorReport,
    escape,
)
from aklr.core.validation.validators.workload_key import WorkloadKey

__all__ = ["PlatformRecord", "ValidationReport", "validate"]

_REWRITE_TOLERANCE: Final = 0.02
_EXIT_CODES: Final = {"failed": 1, "aborted": 3}


@frozen_dataclass
class PlatformRecord:
    arch: str
    hardware: str
    n_visible_devices: int
    rocm_version: str
    torch_version: str
    triton_version: str
    flydsl_version: str

    @classmethod
    def current(cls) -> PlatformRecord:
        unknown = "unknown"
        try:
            import torch
        except ImportError:
            return cls(**dict.fromkeys((f.name for f in fields(cls)), unknown) | {"n_visible_devices": 0})
        n_visible_devices = torch.cuda.device_count()
        triton_version = flydsl_version = unknown
        with contextlib.suppress(ImportError):
            import triton

            triton_version = triton.__version__
        with contextlib.suppress(ImportError):
            import flydsl

            flydsl_version = flydsl.__version__
        target = DeviceTarget.current("cuda:0") if n_visible_devices else None
        return cls(
            arch=target.arch if target is not None and target.arch else unknown,
            hardware=(target.hardware if target is not None and target.hardware else unknown),
            n_visible_devices=n_visible_devices,
            rocm_version=torch.version.hip or unknown,
            torch_version=torch.__version__,
            triton_version=triton_version,
            flydsl_version=flydsl_version,
        )


@frozen_dataclass
class ValidationReport:
    head_sha: str
    base_sha: str | None
    platform: PlatformRecord
    schema_paths: tuple[str, ...]
    bench_paths: tuple[str, ...]
    validators: tuple[SerializeAsAny[ValidatorReport], ...]

    @property
    def conclusion(self) -> str:
        unpassed = [v for v in self.validators if v.status in _EXIT_CODES]
        return ", ".join(f"{v.status} ({v.name})" for v in unpassed) or "passed"

    @property
    def exit_code(self) -> int:
        statuses = {v.status for v in self.validators}
        return next((code for status, code in _EXIT_CODES.items() if status in statuses), 0)

    def markdown_text(self) -> str:
        p = self.platform
        base = "-" if self.base_sha is None else f"`{self.base_sha[:12]}`"
        provenance = f"head `{self.head_sha[:12]}` · base {base} · " + escape(
            f"{p.hardware} {p.arch} ×{p.n_visible_devices} · rocm {p.rocm_version}"
            f" · torch {p.torch_version} · flydsl {p.flydsl_version}"
        )
        blocks: list[Block] = [
            Heading(2, f"AKLR validation: {self.conclusion}"),
            Paragraph(provenance),
            Table(
                ("validator", "status", "findings", "elapsed seconds"),
                tuple(validator.table_row() for validator in self.validators),
            ),
        ]
        for validator in self.validators:
            blocks += validator.markdown_blocks()
        lines_of_blocks = [block.lines() for block in blocks]
        return "\n\n".join("\n".join(lines) for lines in lines_of_blocks if lines) + "\n"

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(
            TypeAdapter(ValidationReport).dump_json(self, indent=2).decode(),
            encoding="utf-8",
        )
        (directory / "report.md").write_text(self.markdown_text(), encoding="utf-8")

    def write_schemas(self, root: Path, directory: Path) -> int:
        shutil.rmtree(directory, ignore_errors=True)
        measured = self.measured_schemas(root)
        for path, schema in measured.items():
            out = directory / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(schema.schema_text(), encoding="utf-8")
        return len(measured)

    def measured_schemas(self, root: Path) -> dict[str, OpContract]:
        bench = next((v for v in self.validators if isinstance(v, BenchReport)), None)
        target = DeviceTarget.from_names(self.platform.arch, self.platform.hardware)
        if bench is None or not target.is_complete:
            return {}
        registry = ContractRegistry.default()
        rewritten: dict[str, OpContract] = {}
        for path in self.bench_paths:
            measured = bench.evaluations.get(path)
            if measured is None:
                continue
            schema = registry.parse_schema((root / path).read_bytes(), path)
            if schema.kernel.targets != (target,):
                continue
            for w in schema.workloads:
                m = measured[str(WorkloadKey.from_workload(w))]
                if (w.accuracy, w.speedup is None, w.microseconds is None) != (
                    m.accuracy,
                    m.speedup is None,
                    m.microseconds is None,
                ) or (m.speedup is not None and abs(m.speedup / w.speedup - 1) > _REWRITE_TOLERANCE):
                    rewritten[path] = schema.with_measurements(measured)
                    break
        return rewritten


def validate(scope: ValidationScope, validators: Sequence[Validator]) -> ValidationReport:
    if not validators:
        raise ValueError("no validator to run")
    platform = PlatformRecord.current()
    base = "-" if scope.base_sha is None else scope.base_sha[:12]
    Progress().write(
        f"{time.strftime('%Y-%m-%d')}, head {scope.head_sha[:12]}, base {base},"
        f" {platform.hardware} {platform.arch} x{platform.n_visible_devices}"
    )
    reports: list[ValidatorReport] = []
    for validator in validators:
        if not scope.schema_paths:
            report = validator.report_type.skipped("no schema to validate")
        else:
            validator.progress.write("started")
            report = validator.execute(scope, platform)
        validator.progress.finish(report)
        reports.append(report)
    return ValidationReport(
        head_sha=scope.head_sha,
        base_sha=scope.base_sha,
        platform=platform,
        schema_paths=tuple(map(scope.relative_path, scope.schema_paths)),
        bench_paths=tuple(map(scope.relative_path, scope.bench_paths)),
        validators=tuple(reports),
    )
