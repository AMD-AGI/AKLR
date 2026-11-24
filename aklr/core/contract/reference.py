# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The base classes of an op's reference: the reference code each op writes, and the check of
a kernel's outputs against it. A reference holds no contract: the contract calls it and passes
in what it needs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from aklr.core.contract.op import OpContract

__all__ = ["DistOpReference", "OpReference"]


class OpReference(ABC):
    @abstractmethod
    def reference(self, **call: Any) -> Mapping[str, Tensor]:
        """The expected outputs and written mutable inputs, by name, of the kernel's call ``call``,
        each in the dtype the contract declares, computed preferably in the op's own precision
        where torch has it, else float32; the op's check may read more entries. Writes no tensor
        of ``call``."""

    def check(
        self, outputs: Mapping[str, Tensor], expected: Mapping[str, Tensor], schema: OpContract
    ) -> dict[str, str]:
        """Each tensor of ``schema`` (the contract at one scenario) with an error bound, against
        ``expected`` under that bound: the ones past it, each with why; none when every one is
        within it. Both sides are in the declared dtypes, compared element by element on their
        device."""
        failures = {}
        for name, spec in schema.tensors().items():
            if spec.error_bounds:
                is_passed, detail = spec.error_bounds[0][1].compare(outputs[name], expected[name])
                if not is_passed:
                    failures[name] = detail
        return failures


class DistOpReference(OpReference):
    """An op that ranks call together. ``check`` is each rank's own."""

    # FIXME(validation-processes): one process holds every rank's call until validation runs
    # the reference a rank a process; then it takes its rank's call, as OpReference.reference
    @abstractmethod
    def reference(self, calls: Sequence[Mapping[str, Any]], rank: int) -> Mapping[str, Tensor]:
        """``OpReference.reference`` of rank ``rank``'s call; ``calls`` holds every rank's call in
        rank order, each on its rank's device."""
