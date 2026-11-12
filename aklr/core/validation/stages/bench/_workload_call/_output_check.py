###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The check of a Candidate's outputs against its Reference's on one input set."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aklr.core.schema import Schema, Tensor
from aklr.core.utils import dtype_str_to_torch_dtype
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised

__all__ = ["OutputCheck"]


class OutputCheck:
    def __init__(self, schema: Schema, axes: Mapping[str, int]) -> None:
        """axes: a workload's; the outputs' declared shapes are at them and the schema's Const axes."""
        shapes = schema.get_output_shapes({**schema.const_axes, **axes})
        self.declared = [
            (tuple(shape), dtype_str_to_torch_dtype(spec.dtype))
            for spec, shape in zip(schema.outputs.values(), shapes, strict=True)
            if isinstance(spec, Tensor) and not spec.optional
        ]
        self.check = schema.standard.check

    def verify(self, expected: list[Any], produced: Any, trial: int) -> None:
        """Raises FindingRaised when what the Candidate produced is not the declared outputs (WRONG_OUTPUTS), or
        the schema's check rejects them against the Reference's expected outputs or raises (OUTPUTS_MISMATCH)."""
        import torch

        actual = [] if produced is None else list(produced) if isinstance(produced, tuple) else [produced]
        got = [(tuple(t.shape), t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__ for t in actual]
        if got[: len(self.declared)] != self.declared:
            raise FindingRaised(
                FindingCode.WRONG_OUTPUTS, f"trial {trial}: outputs {got}, declared {self.declared}"
            )
        try:
            is_correct, detail = bool(self.check(expected, actual)), "the outputs fail the schema's check"
        except Exception as error:  # noqa: BLE001 -- a check that cannot read the outputs rejects them
            is_correct, detail = False, f"the check raised {type(error).__name__}: {error}"
        if not is_correct:
            raise FindingRaised(FindingCode.OUTPUTS_MISMATCH, f"trial {trial}: {detail}")
