# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The base classes of an op's reference: the reference code each op writes, and the check of
a kernel's outputs against it. A reference holds no contract: the contract calls it and passes
in what it needs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import CompareResult, compare_outputs

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["DistOpReference", "OpReference"]


class OpReference(ABC):
    @abstractmethod
    def reference(self, **call: Any) -> Mapping[str, Tensor]:
        """The expected outputs and written mutable inputs, by name, of the kernel's call ``call``,
        preferably in the op's own precision where torch has it, else float32. Writes no tensor of
        ``call``."""

    def check(
        self, outputs: Mapping[str, Tensor], expected: Mapping[str, Tensor], criteria: Mapping[str, Any]
    ) -> dict[str, CompareResult]:
        """Each of ``outputs`` against ``expected``, by name, under ``criteria``
        (``comparison_criteria``): the compare settings and each output's dtype and error bound."""
        return compare_outputs(criteria, outputs, expected)


class DistOpReference(OpReference):
    """An op that ranks call together. ``check`` is each rank's own."""

    # FIXME(validation-processes): one process holds every rank's call until validation runs
    # the reference a rank a process; then it takes its rank's call, as OpReference.reference
    @abstractmethod
    def reference(self, calls: Sequence[Mapping[str, Any]], rank: int) -> Mapping[str, Tensor]:
        """``OpReference.reference`` of rank ``rank``'s call; ``calls`` holds every rank's call in
        rank order, each on its rank's device."""
