###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from aklr.core.validation.stages.audit._stage import AuditStage
from aklr.core.validation.stages.bench import BenchStage, ExecutionPolicy
from aklr.core.validation.stages.report._finding import Fault, Finding, FindingCode, Outcome
from aklr.core.validation.stages.report._markdown import (
    Block,
    Heading,
    Paragraph,
    Table,
    escape,
)
from aklr.core.validation.stages.stage import PipelineStage, Progress, StageReport
from aklr.core.validation.stages.validation_scope import ValidationScope

STAGE_ORDER = tuple(stage.name for stage in (AuditStage, BenchStage))

__all__ = [
    "STAGE_ORDER", "AuditStage", "BenchStage", "Block", "ExecutionPolicy", "Fault", "Finding", "FindingCode", "Heading",
    "Outcome", "Paragraph", "PipelineStage", "Progress", "StageReport", "Table", "ValidationScope", "escape",
]  # fmt: skip
