###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import re
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from aklr.core.validation._execution._model import (
    ResourceCapabilities,
    ResourceHealth,
    ResourceRequest,
    ResourceSnapshot,
    ResourceState,
)


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
    resources: tuple[ResourceSnapshot, ...]

    @property
    def resource(self) -> ResourceSnapshot:
        return self.resources[0]

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(resource.device for resource in self.resources)


class ResourceCatalog(ABC):
    """Owns the resources and leases each to one task at a time."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def matching(self, request: ResourceRequest) -> Sequence[ResourceSnapshot]:
        """Snapshot healthy matching resources, including those already leased."""

    @abstractmethod
    def acquire(self, resource_ids: Sequence[str] | str, task_id: str) -> ResourceLease | None:
        """Atomically lease all requested resources, or leave every resource untouched."""

    @abstractmethod
    def release(self, lease: ResourceLease, health: ResourceHealth) -> None: ...


class LocalAcceleratorCatalog(ResourceCatalog):
    def __init__(self, devices: Sequence[str]) -> None:
        if not devices:
            raise ValueError("no device to run on")
        if invalid := [d for d in devices if not re.fullmatch(r"cuda:\d+", d)]:
            raise ValueError(f"not 'cuda:<n>': {invalid}")
        gpu_ids = [int(device.partition(":")[2]) for device in devices]
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("GPU devices must be distinct")
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

    def __len__(self) -> int:
        return len(self._resources)

    def matching(self, request: ResourceRequest) -> Sequence[ResourceSnapshot]:
        with self._lock:
            return tuple(
                resource.snapshot()
                for resource in self._resources.values()
                if resource.matches(request) and resource.state != ResourceState.UNHEALTHY
            )

    def acquire(self, resource_ids: Sequence[str] | str, task_id: str) -> ResourceLease | None:
        resource_ids = (resource_ids,) if isinstance(resource_ids, str) else tuple(resource_ids)
        if not resource_ids or len(set(resource_ids)) != len(resource_ids):
            raise ValueError("a lease requires distinct resource IDs")
        with self._lock:
            resources = [self._resources.get(resource_id) for resource_id in resource_ids]
            if any(resource is None or resource.state != ResourceState.AVAILABLE for resource in resources):
                return None
            lease_id = uuid.uuid4().hex
            for resource in resources:
                resource.state = ResourceState.LEASED
                self._lease_id_by_resource[resource.resource_id] = lease_id
            lease = ResourceLease(
                lease_id=lease_id,
                task_id=task_id,
                resources=tuple(resource.snapshot() for resource in resources),
            )
            return lease

    def release(self, lease: ResourceLease, health: ResourceHealth) -> None:
        with self._lock:
            for snapshot in lease.resources:
                resource = self._resources.get(snapshot.resource_id)
                current_lease_id = self._lease_id_by_resource.get(snapshot.resource_id)
                if resource is None or current_lease_id != lease.lease_id:
                    continue
                self._lease_id_by_resource.pop(snapshot.resource_id, None)
                if health == ResourceHealth.UNHEALTHY:
                    resource.state = ResourceState.UNHEALTHY
                elif resource.state == ResourceState.LEASED:
                    resource.state = ResourceState.AVAILABLE
