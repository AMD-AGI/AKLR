###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from aklr.core.validation.stages._bench._execution._model import (
    ResourceCapabilities,
    ResourceHealth,
    ResourceRequest,
    ResourceSnapshot,
    ResourceState,
)


class ExecutionResource(Protocol):
    """Resource shape consumed by schedulers and runners."""

    resource_id: str
    devices: tuple[str, ...]
    capabilities: ResourceCapabilities
    state: ResourceState


@dataclass
class AcceleratorResource:
    resource_id: str
    devices: tuple[str, ...]
    capabilities: ResourceCapabilities
    state: ResourceState = ResourceState.AVAILABLE

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource_id=self.resource_id,
            devices=self.devices,
            capabilities=self.capabilities,
            state=self.state,
        )

    def matches(self, request: ResourceRequest) -> bool:
        if request.resource_ids and self.resource_id not in request.resource_ids:
            return False
        return request.backend is None or self.capabilities.backend == request.backend


@dataclass(frozen=True)
class ResourceLease:
    lease_id: str
    task_id: str
    resource: ExecutionResource
    acquired_at: float


class ResourceCatalog(ABC):
    """Owns execution resources and grants exclusive leases."""

    @abstractmethod
    def resources(self) -> Sequence[AcceleratorResource]: ...

    @abstractmethod
    def matching(self, request: ResourceRequest) -> Sequence[ResourceSnapshot]: ...

    @abstractmethod
    def available(self, request: ResourceRequest) -> Sequence[ResourceSnapshot]: ...

    @abstractmethod
    def acquire(self, resource_id: str, task_id: str) -> ResourceLease | None: ...

    @abstractmethod
    def release(self, lease: ResourceLease, health: ResourceHealth) -> None: ...


class LocalAcceleratorCatalog(ResourceCatalog):
    """Thread-safe catalog of the devices of this host the caller names, leased
    ``devices_per_resource`` at a time (the devices left over are unused)."""

    def __init__(self, devices: Sequence[str], devices_per_resource: int = 1) -> None:
        if not devices:
            raise ValueError("no device to run on")
        if invalid := [d for d in devices if not re.fullmatch(r"cpu|cuda:\d+", d)]:
            raise ValueError(f"not 'cpu' or 'cuda:<n>': {invalid}")
        if not 1 <= devices_per_resource <= len(devices):
            raise ValueError(f"{devices_per_resource} devices a resource, of {len(devices)}")
        groups = [
            tuple(devices[i : i + devices_per_resource])
            for i in range(0, len(devices) - devices_per_resource + 1, devices_per_resource)
        ]
        self._resources: dict[str, AcceleratorResource] = {
            f"{index}:{'+'.join(group)}": AcceleratorResource(
                resource_id=f"{index}:{'+'.join(group)}",
                devices=group,
                capabilities=ResourceCapabilities(backend=group[0].partition(":")[0]),
            )
            for index, group in enumerate(groups)
        }
        self._lease_id_by_resource: dict[str, str] = {}
        self._lock = threading.Lock()

    def resources(self) -> Sequence[AcceleratorResource]:
        with self._lock:
            return tuple(self._resources.values())

    def matching(self, request: ResourceRequest) -> Sequence[ResourceSnapshot]:
        with self._lock:
            return tuple(
                resource.snapshot()
                for resource in self._resources.values()
                if resource.matches(request)
                and resource.state not in {ResourceState.UNHEALTHY, ResourceState.QUARANTINED}
            )

    def available(self, request: ResourceRequest) -> Sequence[ResourceSnapshot]:
        return tuple(
            snapshot for snapshot in self.matching(request) if snapshot.state == ResourceState.AVAILABLE
        )

    def acquire(self, resource_id: str, task_id: str) -> ResourceLease | None:
        with self._lock:
            resource = self._resources.get(resource_id)
            if resource is None or resource.state != ResourceState.AVAILABLE:
                return None
            resource.state = ResourceState.LEASED
            lease = ResourceLease(
                lease_id=uuid.uuid4().hex,
                task_id=task_id,
                resource=resource,
                acquired_at=time.monotonic(),
            )
            self._lease_id_by_resource[resource_id] = lease.lease_id
            return lease

    def release(self, lease: ResourceLease, health: ResourceHealth) -> None:
        with self._lock:
            resource = self._resources.get(lease.resource.resource_id)
            current_lease_id = self._lease_id_by_resource.get(lease.resource.resource_id)
            if resource is None or current_lease_id != lease.lease_id:
                return
            self._lease_id_by_resource.pop(lease.resource.resource_id, None)
            if health == ResourceHealth.UNHEALTHY:
                resource.state = ResourceState.UNHEALTHY
            elif resource.state == ResourceState.LEASED:
                resource.state = ResourceState.AVAILABLE
