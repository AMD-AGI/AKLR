###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The bench's stage report: its schemas' verdicts gathered, and rendered."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from pydantic import Field

from aklr.core.schema import Evaluation
from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.validation.stages.bench._admission._verdict import SchemaVerdict, WorkloadRecord
from aklr.core.validation.stages.bench._measurement._measurement_profile import MeasurementProfile
from aklr.core.validation.stages.report._finding import Finding
from aklr.core.validation.stages.report._markdown import (
    Block,
    BulletList,
    Heading,
    Paragraph,
    Table,
    escape,
    schema_short_name,
)
from aklr.core.validation.stages.stage import StageReport

__all__ = ["BenchReport"]


@frozen_dataclass
class BenchReport(StageReport):
    records: tuple[WorkloadRecord, ...] = ()
    demotions: tuple[Finding, ...] = ()
    profile: MeasurementProfile = Field(default_factory=MeasurementProfile)
    protocol: str = ""  # the profile's digest: results of another protocol are not comparable
    # what the pipeline writes back: by schema path, each workload's Evaluation on this target by its key's text
    evaluations: dict[str, dict[str, Evaluation]] = Field(default_factory=dict)

    @classmethod
    def from_verdicts(
        cls, stage_name: str, verdicts: Sequence[SchemaVerdict], profile: MeasurementProfile
    ) -> BenchReport:
        return cls.from_findings(
            stage_name,
            [f for v in verdicts for f in v.findings],
            records=tuple(r for v in verdicts for r in v.records),
            demotions=tuple(f for v in verdicts for f in v.demotions),
            profile=profile,
            protocol=profile.digest,
            evaluations={v.path: v.evaluations for v in verdicts if v.evaluations is not None},
        )

    def console_lines(self) -> list[str]:
        lines = super().console_lines()
        if self.status != "skipped":
            lines.append(f"workloads: {len(self.records)}, demoted: {len(self.demotions)}")
        return lines

    def record_blocks(self) -> list[Block]:
        p = self.profile
        rows = []
        for r in self.records:
            c = r.comparison
            rows.append(
                (
                    schema_short_name(r.schema_path),
                    escape(str(r.workload)) or "-",
                    escape(r.verdict) + (" (demoted)" if r.is_demoted else ""),
                    escape(r.versus if c is None else f"{r.versus} {c.versus_name}"),
                    "-" if c is None else f"{c.ratio:.3f}",
                    "-" if c is None else f"{c.speedup:.3f}",
                    "-" if c is None else f"{statistics.median(c.candidate_ms):.4f}",
                    "-" if c is None else f"{statistics.median(c.opponent_ms):.4f}",
                )
            )
        profile = (
            f"protocol {self.protocol}: {p.num_trials} trials, each checked, then timed once per participant"
            f" in a {p.graph_budget_ms:g} ms graph; the ratio is their geometric mean;"
            f" Entry {p.entry}, Retain {p.retain}"
        )
        blocks: list[Block] = [
            Paragraph(escape(profile)),
            Table(
                (
                    "schema",
                    "workload",
                    "verdict",
                    "versus",
                    "ratio",
                    "speedup",
                    "candidate ms",
                    "opponent ms",
                ),
                tuple(rows),
            ),
        ]
        if self.demotions:
            blocks += [Heading(4, "Demotions"), BulletList(tuple(f.markdown_item() for f in self.demotions))]
        return blocks
