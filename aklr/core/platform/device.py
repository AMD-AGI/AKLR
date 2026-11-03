###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

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


# not schema's frozen_dataclass: aklr.core.schema imports this module
@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class DeviceTarget:
    """A normalized GPU arch (gfx950) and hardware model (MI355X); None: unknown, or any in a constraint."""

    arch: str | None = None
    hardware: str | None = None

    _target_by_device_idx: ClassVar[dict[int, DeviceTarget]] = {}

    def __str__(self) -> str:
        if not self.is_known:
            return "unknown"
        return f"{self.arch or 'unknown'}/{self.hardware or 'unknown'}"

    @classmethod
    def normalize_arch(cls, arch: str | None) -> str | None:
        # drop ROCm feature suffixes, e.g. gfx942:sramecc+:xnack-
        value = (arch or "").strip().lower().split(":", 1)[0]
        return value if re.fullmatch(r"gfx[0-9a-f]+", value) else None

    @classmethod
    def normalize_hardware(cls, hardware: str | None) -> str | None:
        value = " ".join((hardware or "").upper().split())
        value = re.sub(r"^AMD(?: INSTINCT)?\s+", "", value)
        if value in {"", "UNKNOWN", "UNAVAILABLE", "ANY", "HIP", "GPU"}:
            return None
        # some drivers report only the ISA as the device name, which is no model ID
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
        """Normalize observed names; an unrecognized one becomes None where the constructor would raise."""
        return cls(cls.normalize_arch(arch), cls.normalize_hardware(hardware))

    @classmethod
    def matches_target(cls, target: DeviceTarget, constraints: Sequence[DeviceTarget] | None) -> bool:
        """Whether target meets any constraint (None: no constraint) in each field that constraint sets."""
        return constraints is None or any(
            constraint.is_known
            and (constraint.arch is None or target.arch == constraint.arch)
            and (constraint.hardware is None or target.hardware == constraint.hardware)
            for constraint in constraints
        )

    @classmethod
    def current(cls, device: torch.device | str | None = None) -> DeviceTarget:
        """The target of device (default: the current GPU), cached per ordinal; unknown without a GPU."""
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
        return self.arch is not None or self.hardware is not None

    @property
    def is_complete(self) -> bool:
        return self.arch is not None and self.hardware is not None


def physical_device(device: str) -> str:
    """The physical GPU id behind cuda:N under the parent's HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES."""
    index = int(device.removeprefix("cuda:"))
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",")[index].strip() if visible else str(index)
