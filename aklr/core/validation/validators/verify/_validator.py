# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import itertools
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Final

from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import axes_text
from aklr.core.platform.device import physical_device
from aklr.core.validation.validators.report._finding import Finding, FindingCode
from aklr.core.validation.validators.validator import Validator
from aklr.core.validation.validators.verify._compilation import Compilation
from aklr.core.validation.validators.verify._report import PathRecord, VerifyReport
from aklr.core.validation.validators.verify._rules import ISA_RULES
from aklr.core.validation.validators.verify._volta import Volta
from aklr.core.validation.validators.workload_key import WorkloadKey

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from aklr.core.validation.report import PlatformRecord
    from aklr.core.validation.validators.validation_scope import ValidationScope

__all__ = ["VerifyValidator"]

WORKLOADS_PER_COMPILER: Final = 4


class VerifyValidator(Validator, name="verify"):
    def __init__(
        self,
        devices: Sequence[int] = (),
        *,
        n_workers: int | None = None,
        compiler: Callable[..., list[Compilation]] = Compilation.of_schema,
    ) -> None:
        self._devices = tuple(devices)
        self._n_workers = n_workers or min(96, os.cpu_count() or 1)
        self._compiler = compiler

    def run(self, scope: ValidationScope, platform: PlatformRecord) -> VerifyReport:
        directory = Path(tempfile.mkdtemp(prefix="aklr_verify_"))
        registry = ContractRegistry.default()
        schemas = {}
        for path in scope.bench_paths:
            schema_path = scope.relative_path(path)
            schema = registry.parse_schema(path.read_bytes(), schema_path)
            if schema.kernel.sources:
                schemas[schema_path] = schema
        self.progress.write(
            f"compiling and judging {len(schemas)} schemas, {self._n_workers} at a time, in {directory}"
        )
        volta = Volta()

        def schema_directory(schema_path: str) -> Path:
            return directory / "-".join(Path(schema_path).parts[-3:]).removesuffix(".json")

        ordinals = itertools.count()

        def judged(compilation: Compilation) -> PathRecord:
            where = schema_directory(compilation.schema_path) / re.sub(
                r"[^\w=,.-]", "_", axes_text(compilation.axes)
            )
            run = None
            if compilation.isa_path is not None and compilation.launch is not None:
                run = volta.run(compilation, where / f"{compilation.kernel}-{next(ordinals)}")
            return PathRecord(
                schema_path=compilation.schema_path,
                workload=WorkloadKey(axes=compilation.axes),
                kernel=compilation.kernel,
                verdicts=tuple(rule.judge(compilation, run) for rule in ISA_RULES),
            )

        with (
            ThreadPoolExecutor(self._n_workers) as compilers,
            ThreadPoolExecutor(self._n_workers) as judges,
        ):
            judging = compilers.map(
                lambda part: [
                    judges.submit(judged, compilation)
                    for compilation in self._compiler(
                        scope.root,
                        part[1],
                        schemas[part[1]],
                        schema_directory(part[1]) / str(part[2]),
                        part[2],
                        WORKLOADS_PER_COMPILER,
                        physical_device(f"cuda:{self._devices[part[0] % len(self._devices)]}")
                        if self._devices
                        else None,
                    )
                ],
                [
                    (ordinal, schema_path, first)
                    for ordinal, (schema_path, first) in enumerate(
                        (schema_path, first)
                        for schema_path, schema in schemas.items()
                        for first in range(0, len(schema.workloads), WORKLOADS_PER_COMPILER)
                    )
                ],
            )
            try:
                records = tuple(verdict.result() for of_part in judging for verdict in of_part)
            except BaseException:
                compilers.shutdown(wait=False, cancel_futures=True)
                judges.shutdown(wait=False, cancel_futures=True)
                raise
        shutil.rmtree(directory, ignore_errors=True)
        numbers = {rule.name: rule.number for rule in ISA_RULES}
        findings = [
            Finding(
                code=FindingCode.ISA_REJECTED,
                schema_path=record.schema_path,
                workload=record.workload,
                detail=f"VR{numbers[v.rule]} {v.rule} {record.kernel}: {v.detail}",
            )
            for record in records
            for v in record.verdicts
            if v.status == "rejected"
        ]
        return VerifyReport.from_findings(
            findings, path_records=records, rules=numbers, tools={"volta": volta.description}
        )
