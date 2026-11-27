###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aklr.core.contract import semantics
from aklr.core.contract.op import OpContract
from aklr.core.validation.validators.bench._measure._contract_workload import ContractWorkload
from aklr.core.validation.validators.report._finding import FindingCode, FindingRaised

__all__ = ["OutputCheck"]


class OutputCheck:
    def __init__(self, schema: OpContract, axes: Mapping[str, int]) -> None:
        self.workload = workload = ContractWorkload.of(schema, axes)
        contract, point = workload.contract, workload.point
        self.declared = {
            o: (
                tuple(contract.storage_shape_of(o, point)),
                semantics.torch_dtypes(workload.schema_json["outputs"][o]["dtype"]["any_of"][0]),
            )
            for o in contract.outputs
        }

    def reference(self, input_sets: Sequence[Mapping[str, Any]], rank: int) -> Mapping[str, Any]:
        import torch

        workload = self.workload
        reference = workload.contract.reference
        device = next(v.device for v in input_sets[rank].values() if torch.is_tensor(v))
        if workload.contract.ranks is None:
            return reference.reference(**workload.reference_call(input_sets[rank], device))
        return reference.reference([workload.reference_call(args, device) for args in input_sets], rank)

    def verify(self, reference_outputs: Mapping[str, Any], produced: Any, trial: int) -> None:
        import torch

        got = {
            o: (tuple(t.shape), t.dtype) if torch.is_tensor(t) else type(t).__name__
            for o, t in produced.items()
            if o in self.declared
        }
        if any(
            not isinstance(got[o], tuple) or got[o][0] != shape or got[o][1] not in dtypes
            for o, (shape, dtypes) in self.declared.items()
        ):
            raise FindingRaised(
                FindingCode.WRONG_OUTPUTS, f"trial {trial}: outputs {got}, declared {self.declared}"
            )
        try:
            workload = self.workload
            failed = workload.contract.reference.check(produced, reference_outputs, workload.schema)
            detail = "; ".join(f"{o}: {why}" for o, why in failed.items())
        except Exception as error:  # noqa: BLE001 -- a check that cannot read the outputs rejects them
            failed, detail = {"": ""}, f"the check raised {type(error).__name__}: {error}"
        if failed:
            raise FindingRaised(FindingCode.NUMERIC_MISMATCH, f"trial {trial}: {detail}")
