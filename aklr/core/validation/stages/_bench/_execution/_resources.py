###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import re
import threading
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
    resource_id: str
    device: str
    capabilities: ResourceCapabilities
    state: ResourceState


@dataclass
class AcceleratorResource:
    resource_id: str
    device: str
    capabilities: ResourceCapabilities
    state: ResourceState = ResourceState.AVAILABLE

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource_id=self.resource_id,
            device=self.device,
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


class ResourceCatalog(ABC):
    """Owns the resources and leases each to one task at a time."""

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
    def __init__(self, devices: Sequence[str]) -> None:
        if not devices:
            raise ValueError("no device to run on")
        if invalid := [d for d in devices if not re.fullmatch(r"cpu|cuda:\d+", d)]:
            raise ValueError(f"not 'cpu' or 'cuda:<n>': {invalid}")
        self._resources: dict[str, AcceleratorResource] = {
            f"{index}:{device}": AcceleratorResource(
                resource_id=f"{index}:{device}",
                device=device,
                capabilities=ResourceCapabilities(backend=device.partition(":")[0]),
            )
            for index, device in enumerate(devices)
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
