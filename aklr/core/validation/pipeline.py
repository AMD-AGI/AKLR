# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The validation pipeline and the report of one run."""

from __future__ import annotations

import contextlib
import inspect
import json
import shutil
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from pydantic import SerializeAsAny

from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import FrozenModel, axes_text
from aklr.core.platform.device import DeviceTarget
from aklr.core.validation.stages import (
    MEASUREMENT_PROFILE,
    STAGE_ORDER,
    AuditStage,
    BenchReport,
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

__all__ = ["REPORT_VERSION", "Pipeline", "PipelineReport", "PlatformRecord"]

REPORT_VERSION: Final = 2
_REWRITE_TOLERANCE: Final = 0.02
_EXIT_CODES: Final = {"failed": 1, "aborted": 3}


class Pipeline:
    """Runs the stages in ``STAGE_ORDER``; the audit always comes first."""

    def __init__(self, stages: Sequence[PipelineStage]) -> None:
        if not stages or not isinstance(stages[0], AuditStage):
            raise ValueError("a pipeline's first stage is an AuditStage")
        orders = [STAGE_ORDER.index(s.name) for s in stages]
        if orders != sorted(set(orders)):
            names = [s.name for s in stages]
            raise ValueError(f"stages {names} are not in the order {STAGE_ORDER}")
        self._stages = tuple(stages)

    def run(self, scope: ValidationScope) -> PipelineReport:
        """Run each stage on ``scope``; later stages are skipped once one does not pass."""
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
                measurement_profile=dict(MEASUREMENT_PROFILE),
                platform=platform,
                schema_paths=tuple(map(scope.relative_path, scope.schema_paths)),
                bench_paths=tuple(map(scope.relative_path, scope.bench_paths)),
                stages=tuple(reports),
            )

        for number, stage in enumerate(self._stages, 1):
            if skip_reason is not None:
                report = stage.report_type.skipped(skip_reason)
            else:
                stage.progress.write(f"started (stage {number}/{len(self._stages)})")
                report = stage.execute(scope, pipeline_report())
                if not report.is_passed:
                    skip_reason = f"the previous stage {stage.name} {report.status}"
            reports.append(report)
            stage.progress.finish(report)
        return pipeline_report()


class PlatformRecord(FrozenModel):
    """The platform a pipeline ran on: facts only."""

    arch: str
    hardware: str
    n_visible_devices: int
    rocm_version: str
    torch_version: str
    triton_version: str
    flydsl_version: str

    @classmethod
    def current(cls) -> PlatformRecord:
        """This process's platform; ``"unknown"`` for what it cannot read."""
        unknown = "unknown"
        try:
            import torch
        except ImportError:
            return cls(**dict.fromkeys(cls.model_fields, unknown) | {"n_visible_devices": 0})
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


class PipelineReport(FrozenModel):
    """One validation run; written, never read back."""

    version: int = REPORT_VERSION
    head_sha: str
    base_sha: str | None
    measurement_profile: dict[str, Any]
    platform: PlatformRecord
    schema_paths: tuple[str, ...]
    bench_paths: tuple[str, ...]
    stages: tuple[SerializeAsAny[StageReport], ...]

    @property
    def deciding_stage(self) -> StageReport | None:
        """The first failed stage, else the first aborted one."""
        for status in ("failed", "aborted"):
            for stage in self.stages:
                if stage.status == status:
                    return stage
        return None

    @property
    def conclusion(self) -> str:
        """``passed``, or the deciding stage's ``<status> (<stage name>)``."""
        stage = self.deciding_stage
        return "passed" if stage is None else f"{stage.status} ({stage.stage_name})"

    @property
    def exit_code(self) -> int:
        """0 passed, 1 failed (candidate's fault), 3 aborted (platform's fault)."""
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
        """Write ``report.json`` and ``report.md`` under ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(self.model_dump_json(indent=2), encoding="utf-8")
        (directory / "report.md").write_text(self.markdown_text(), encoding="utf-8")

    def write_schemas(self, root: Path, directory: Path) -> None:
        """Write each bench schema with its measurements; one schema's finding does not
        stop the rest (``BenchReport.measurements``)."""
        shutil.rmtree(directory, ignore_errors=True)  # no earlier run's rewrites
        bench = next((s for s in self.stages if isinstance(s, BenchReport)), None)
        target = DeviceTarget.from_names(self.platform.arch, self.platform.hardware)
        if bench is None or not target.is_complete:
            return
        registry = ContractRegistry.default()
        for path in self.bench_paths:
            schema = registry.parse_schema((root / path).read_bytes(), path)
            if schema.kernel.targets != (target,):
                continue
            measured = bench.measurements(path, schema)
            if measured is None:
                continue
            for w in schema.workloads:
                m = measured[axes_text(w.axes)]
                if (w.accuracy, w.speedup is None) != (
                    m.accuracy,
                    m.speedup is None,
                ) or (m.speedup is not None and abs(m.speedup / w.speedup - 1) > _REWRITE_TOLERANCE):
                    out = directory / path
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(
                        schema.with_measurements(measured).schema_text(),
                        encoding="utf-8",
                    )
                    break

    def write_baselines(self, scope: ValidationScope, directory: Path) -> None:
        """Re-export, replacing the last run's, what each benched workload was compared
        with: an incumbent's base schema, else the defined baseline in main's Solution
        shape (``<op_type>/<name>/<name>__baseline.json``); never committed."""
        shutil.rmtree(directory, ignore_errors=True)  # a changed baseline replaces it
        bench = next((s for s in self.stages if isinstance(s, BenchReport)), None)
        if bench is None:
            return
        registry = ContractRegistry.default()
        schemas = {
            path: registry.parse_schema((scope.root / path).read_bytes(), path) for path in self.bench_paths
        }
        # same-named schemas share a definition: (op_type, name, baseline) -> workloads
        groups: dict[tuple[str, str, str], dict[str, dict]] = {}
        for r in bench.workload_records:
            if r.baseline is None:  # the new case is its own baseline: nothing to show
                continue
            schema = schemas[r.schema_path]
            key = axes_text(r.axes)
            workload = next(
                w for w in schema.model_dump(mode="json")["workloads"] if axes_text(w["axes"]) == key
            )
            group = (schema.op_type, schema.name, r.baseline)
            groups.setdefault(group, {})[key] = workload
        for (op_type, name, baseline), workloads in groups.items():
            if baseline.endswith(".json"):  # an incumbent at base, kernel as committed
                data = json.loads(scope.text_at_base(baseline))
                data["workloads"] = [w for w in data["workloads"] if axes_text(w["axes"]) in workloads]
                file_name = PurePosixPath(baseline).name
            else:
                contract = registry.get(next(s for s in schemas.values() if s.name == name).contract_id)
                fn = contract.baseline()
                data = {
                    "name": f"{name}__baseline",
                    "definition": name,
                    "author": "baseline",
                    "spec": {
                        "language": "python",
                        "target": [str(DeviceTarget.from_names(self.platform.arch, self.platform.hardware))],
                        "entry_point": "main.py::run",
                        "destination_passing_style": False,
                    },
                    "sources": [
                        {
                            "path": "main.py",
                            "content": f"{inspect.getsource(fn)}\nrun = {fn.__name__}\n",
                        }
                    ],
                    "description": f"The baseline of {name}: {baseline}.",
                    "workloads": list(workloads.values()),
                }
                file_name = f"{name}__baseline.json"
            out = directory / op_type / name / file_name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
