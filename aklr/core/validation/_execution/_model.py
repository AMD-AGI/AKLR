###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
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
    """Lease ``count`` resources from the allowed IDs/backend; empty filters allow any."""

    resource_ids: tuple[str, ...] = ()
    backend: str | None = None
    count: int = 1

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 1:
            raise ValueError("resource count must be a positive integer")


@dataclass(frozen=True)
class ResourceSnapshot:
    resource_id: str
    device: str
    capabilities: ResourceCapabilities
    state: ResourceState


@dataclass(frozen=True, kw_only=True)
class ExecutionTask:
    call: Callable[..., Any]
    timeout_secs: float
    task_id: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    metadata: Mapping[str, str] = field(default_factory=dict)
    runner: str | None = None
    serial_group: str | None = None
    n_start_attempts: int = 0


@dataclass(frozen=True, kw_only=True)
class ExecutionContext:
    """Leased devices and task files."""

    log_path: Path
    directory: Path
    devices: tuple[str, ...]  # in rank order; this process's device, devices[RANK], is already current

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("execution context requires at least one device")
