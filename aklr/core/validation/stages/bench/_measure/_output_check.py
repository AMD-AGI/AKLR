###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from aklr.core.schema import Schema, Tensor
from aklr.core.utils import dtype_str_to_torch_dtype
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised

__all__ = ["OutputCheck"]


class OutputCheck:
    def __init__(self, schema: Schema, axes: Mapping[str, int]) -> None:
        shapes = schema.get_output_shapes({**schema.const_axes, **axes})
        self.declared = [
            (tuple(shape), dtype_str_to_torch_dtype(spec.dtype))
            for spec, shape in zip(schema.outputs.values(), shapes, strict=True)
            if isinstance(spec, Tensor) and not spec.optional
        ]
        self.check = schema.standard.check
        self.reference_fn = schema.standard.reference
        self.declared_inputs = inspect.signature(self.reference_fn).parameters

    def reference(self, input_sets: Sequence[Mapping[str, Any]], rank: int) -> list[Any]:
        given = {**input_sets[rank], "group_calls": input_sets}
        outputs = self.reference_fn(**{n: v for n, v in given.items() if n in self.declared_inputs})
        return list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]

    def verify(self, reference_outputs: list[Any], produced: Any, trial: int) -> None:
        import torch

        actual = [] if produced is None else list(produced) if isinstance(produced, tuple) else [produced]
        got = [(tuple(t.shape), t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__ for t in actual]
        if got[: len(self.declared)] != self.declared:
            raise FindingRaised(
                FindingCode.WRONG_OUTPUTS, f"trial {trial}: outputs {got}, declared {self.declared}"
            )
        try:
            is_correct, detail = (
                bool(self.check(reference_outputs, actual)),
                "the outputs fail the schema's check",
            )
        except Exception as error:  # noqa: BLE001
            is_correct, detail = False, f"the check raised {type(error).__name__}: {error}"
        if not is_correct:
            raise FindingRaised(FindingCode.OUTPUTS_MISMATCH, f"trial {trial}: {detail}")
