###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExecutorState(str, Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class TaskState(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunnerState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"
    CLOSED = "closed"


class ResourceState(str, Enum):
    AVAILABLE = "available"
    LEASED = "leased"
    UNHEALTHY = "unhealthy"
    QUARANTINED = "quarantined"


class ResourceHealth(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class RunnerHealth(str, Enum):
    HEALTHY = "healthy"
    FAILED = "failed"
    RESOURCE_FAILED = "resource_failed"


@dataclass(frozen=True)
class ResourceCapabilities:
    backend: str


@dataclass(frozen=True)
class ResourceRequest:
    """Which resources a task may run on; an empty field allows any."""

    resource_ids: tuple[str, ...] = ()
    backend: str | None = None


@dataclass(frozen=True)
class ResourceSnapshot:
    resource_id: str
    device: str
    capabilities: ResourceCapabilities
    state: ResourceState


@dataclass(frozen=True, kw_only=True)
class ExecutionTask:
    call: Callable[..., Any]
    timeout_seconds: float
    task_id: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    metadata: Mapping[str, str] = field(default_factory=dict)
    n_start_attempts: int = 0


@dataclass(frozen=True, kw_only=True)
class ExecutionContext:
    """What the runner's child process hands the task's callable as its first argument."""

    device: str
    log_path: Path
    directory: Path
