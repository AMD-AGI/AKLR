###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from aklr.core.validation.stages._audit._stage import AuditStage
from aklr.core.validation.stages._bench._report import BenchReport
from aklr.core.validation.stages._bench._stage import BenchStage
from aklr.core.validation.stages._e2e._stage import E2eStage
from aklr.core.validation.stages._measure import MEASUREMENT_PROFILE
from aklr.core.validation.stages._report._markdown import (
    Block,
    Heading,
    Paragraph,
    Table,
    escape,
)
from aklr.core.validation.stages.stage import PipelineStage, Progress, StageReport
from aklr.core.validation.stages.validation_scope import ValidationScope

STAGE_ORDER = tuple(stage.name for stage in (AuditStage, BenchStage, E2eStage))

__all__ = [
    "MEASUREMENT_PROFILE", "STAGE_ORDER", "AuditStage", "BenchReport", "BenchStage", "Block", "E2eStage", "Heading",
    "Paragraph", "PipelineStage", "Progress", "StageReport", "Table", "ValidationScope", "escape",
]  # fmt: skip
