# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from pydantic import ConfigDict, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    import torch

__all__ = ["DeviceTarget", "physical_device"]


@dataclass(frozen=True, order=True, config=ConfigDict(extra="forbid"))
class DeviceTarget:
    """Immutable, normalized architecture/model pair."""

    arch: str | None = None
    hardware: str | None = None

    _target_by_device_idx: ClassVar[dict[int, DeviceTarget]] = {}

    def __str__(self) -> str:
        if not self.is_known:
            return "unknown"
        return f"{self.arch or 'unknown'}/{self.hardware or 'unknown'}"

    @classmethod
    def normalize_arch(cls, arch: str | None) -> str | None:
        """The bare arch, such as gfx950, without ROCm feature suffixes."""
        value = (arch or "").strip().lower().split(":", 1)[0]
        return value if re.fullmatch(r"gfx[0-9a-f]+", value) else None

    @classmethod
    def normalize_hardware(cls, hardware: str | None) -> str | None:
        """Normalize names while retaining model, memory and form-factor variants."""
        value = " ".join((hardware or "").upper().split())
        value = re.sub(r"^AMD(?: INSTINCT)?\s+", "", value)
        if value in {"", "UNKNOWN", "UNAVAILABLE", "ANY", "HIP", "GPU"}:
            return None
        # Some drivers report only the ISA in the name field. That is no model ID.
        if cls.normalize_arch(value) is not None:
            return None
        return value

    @field_validator("arch", "hardware")
    @classmethod
    def _normalize_name(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        normalize = cls.normalize_arch if info.field_name == "arch" else cls.normalize_hardware
        normalized = normalize(value)
        if normalized is None:
            raise ValueError(f"{info.field_name} contains an unknown target")
        return normalized

    @classmethod
    def from_names(cls, arch: str | None, hardware: str | None) -> DeviceTarget:
        """Normalize observed device names, retaining unknown values as None."""
        return cls(cls.normalize_arch(arch), cls.normalize_hardware(hardware))

    @classmethod
    def normalize_targets(cls, targets: Sequence[DeviceTarget] | None) -> tuple[DeviceTarget, ...] | None:
        """Validate, deduplicate and sort build constraints; None is unrestricted."""
        if targets is None:
            return None
        if not targets or any(not target.is_known for target in targets):
            raise ValueError("target must contain at least one architecture or hardware model")
        return tuple(
            sorted(
                set(targets),
                key=lambda target: (target.arch or "", target.hardware or ""),
            )
        )

    @classmethod
    def matches_target(cls, target: DeviceTarget, constraints: Sequence[DeviceTarget] | None) -> bool:
        """Match any declared pair, requiring all fields specified by that pair."""
        return constraints is None or any(
            constraint.is_known
            and (constraint.arch is None or target.arch == constraint.arch)
            and (constraint.hardware is None or target.hardware == constraint.hardware)
            for constraint in constraints
        )

    @classmethod
    def current(cls, device: torch.device | str | None = None) -> DeviceTarget:
        """Resolve a device through torch, caching GPU properties by logical ordinal."""
        try:
            import torch

            torch_device = torch.device(device) if device is not None else torch.device("cuda")
            if torch_device.type != "cuda":
                return cls(hardware=torch_device.type.upper())
            if torch_device.index is not None:
                device_idx = torch_device.index
            else:
                if not torch.cuda.is_available():
                    return cls()
                device_idx = torch.cuda.current_device()
            if device_idx in cls._target_by_device_idx:
                return cls._target_by_device_idx[device_idx]
            properties = torch.cuda.get_device_properties(device_idx)
            target = cls.from_names(getattr(properties, "gcnArchName", None), properties.name)
            cls._target_by_device_idx[device_idx] = target
            return target
        except Exception:  # noqa: BLE001 (no torch, accelerator or driver)
            return cls()

    @property
    def is_known(self) -> bool:
        """At least one field is known, so this pair can constrain a build target."""
        return self.arch is not None or self.hardware is not None

    @property
    def is_complete(self) -> bool:
        """A known architecture and model identify a measurement's target."""
        return self.arch is not None and self.hardware is not None


def physical_device(device: str) -> str:
    """The physical GPU id behind ``device`` (``cuda:N``) under the parent's
    HIP/CUDA_VISIBLE_DEVICES."""
    index = int(device.removeprefix("cuda:"))
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",")[index].strip() if visible else str(index)
