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
from dataclasses import fields, replace
from pathlib import Path
from typing import Final

from pydantic import Field, SerializeAsAny, TypeAdapter

from aklr.core.platform.device import DeviceTarget
from aklr.core.schema import Schema, load_json_file, save_json_file
from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.validation.stages import (
    STAGE_ORDER,
    AuditStage,
    BenchStage,
    Block,
    Heading,
    Paragraph,
    PipelineStage,
    Progress,
    StageReport,
    Table,
    ValidationScope,
    escape,
)
from aklr.core.validation.stages.workload_key import WorkloadKey

__all__ = ["REPORT_VERSION", "Pipeline", "PipelineReport", "PlatformRecord"]

REPORT_VERSION: Final = 3
_REWRITE_TOLERANCE: Final = 0.02  # a speedup within 2% of the stored one does not rewrite its schema
_EXIT_CODES: Final = {"failed": 1, "aborted": 3}


class Pipeline:
    def __init__(self, stages: Sequence[PipelineStage]) -> None:
        if not stages or not isinstance(stages[0], AuditStage):
            raise ValueError("a pipeline's first stage is an AuditStage")
        orders = [STAGE_ORDER.index(s.name) for s in stages]
        if orders != sorted(set(orders)):
            names = [s.name for s in stages]
            raise ValueError(f"stages {names} are not in the order {STAGE_ORDER}")
        self._stages = tuple(stages)

    def run(self, scope: ValidationScope) -> PipelineReport:
        """Run each stage on scope; once one does not pass, the later ones are skipped."""
        platform = PlatformRecord.current()
        base = "-" if scope.base_sha is None else scope.base_sha[:12]
        Progress().write(
            f"{time.strftime('%Y-%m-%d')}, head {scope.head_sha[:12]}, base {base},"
            f" {platform.hardware} {platform.arch} x{platform.n_visible_devices}"
        )
        skip_reason = None if scope.schema_paths else "no schema to validate"
        reports: list[StageReport] = []

        def pipeline_report() -> PipelineReport:
            return PipelineReport(
                head_sha=scope.head_sha,
                base_sha=scope.base_sha,
                platform=platform,
                schema_paths=tuple(map(scope.relative_path, scope.schema_paths)),
                bench_paths=tuple(map(scope.relative_path, scope.bench_paths)),
                stages=tuple(reports),
            )

        for number, stage in enumerate(self._stages, 1):
            if skip_reason is not None:
                report = stage.report_type.skipped(stage.name, skip_reason)
            else:
                stage.progress.write(f"started (stage {number}/{len(self._stages)})")
                report = stage.execute(scope, pipeline_report())
                if not report.is_passed:
                    skip_reason = f"the previous stage {stage.name} {report.status}"
            reports.append(report)
            stage.progress.finish(report)
        return pipeline_report()


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
class PipelineReport:
    version: int = Field(default=REPORT_VERSION, kw_only=True)
    head_sha: str
    base_sha: str | None
    platform: PlatformRecord
    schema_paths: tuple[str, ...]
    bench_paths: tuple[str, ...]
    stages: tuple[SerializeAsAny[StageReport], ...]

    @property
    def deciding_stage(self) -> StageReport | None:
        for status in ("failed", "aborted"):
            for stage in self.stages:
                if stage.status == status:
                    return stage
        return None

    @property
    def conclusion(self) -> str:
        stage = self.deciding_stage
        return "passed" if stage is None else f"{stage.status} ({stage.stage_name})"

    @property
    def exit_code(self) -> int:
        """0 passed, 1 failed (the candidate's error), 3 aborted (the platform's error)."""
        stage = self.deciding_stage
        return 0 if stage is None else _EXIT_CODES[stage.status]

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
                ("stage", "status", "findings", "elapsed seconds"),
                tuple(stage.table_row() for stage in self.stages),
            ),
        ]
        for stage in self.stages:
            blocks += stage.markdown_blocks()
        lines_of_blocks = [block.lines() for block in blocks]
        return "\n\n".join("\n".join(lines) for lines in lines_of_blocks if lines) + "\n"

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(
            TypeAdapter(PipelineReport).dump_json(self, indent=2).decode(),
            encoding="utf-8",
        )
        (directory / "report.md").write_text(self.markdown_text(), encoding="utf-8")

    def write_schemas(self, root: Path, directory: Path) -> int:
        shutil.rmtree(directory, ignore_errors=True)  # no earlier run's rewrites may remain
        measured = self.measured_schemas(root)
        for path, schema in measured.items():
            save_json_file(schema, directory / path)
        return len(measured)

    def measured_schemas(self, root: Path) -> dict[str, Schema]:
        """Bench schemas whose measurements changed, rewritten, by path, whatever failed elsewhere."""
        bench = next((s for s in self.stages if isinstance(s, BenchStage.report_type)), None)
        target = DeviceTarget.from_names(self.platform.arch, self.platform.hardware)
        if bench is None or not target.is_complete:
            return {}
        rewritten: dict[str, Schema] = {}
        for path in self.bench_paths:
            schema = load_json_file(Schema, root / path)
            if schema.kernel is None or not schema.kernel.build.runs_on(target):
                continue
            measured = bench.evaluations.get(path)
            if measured is None:
                continue
            for w in schema.workloads:
                m = measured[str(WorkloadKey.from_workload(w))]
                stored = w.evaluation(target)
                is_changed = (
                    stored is None
                    or stored.accuracy != m.accuracy
                    or stored.speedup.keys() != m.speedup.keys()
                )
                is_changed = is_changed or any(
                    abs(m.speedup[b].value / stored.speedup[b].value - 1) > _REWRITE_TOLERANCE
                    for b in m.speedup
                )
                if is_changed:
                    rewritten[path] = replace(
                        schema,
                        workloads=[
                            replace(
                                x,
                                evaluations=sorted(
                                    [e for e in x.evaluations if e.hardware != str(target)]
                                    + [measured[str(WorkloadKey.from_workload(x))]],
                                    key=lambda e: e.hardware,
                                ),
                            )
                            for x in schema.workloads
                        ],
                    )
                    break
        return rewritten
