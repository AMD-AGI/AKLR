###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The check of a Candidate's outputs against its Reference's on one input set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aklr.core.contract import semantics
from aklr.core.contract.op import OpContract
from aklr.core.validation.stages.bench._measure._contract_workload import ContractWorkload
from aklr.core.validation.stages.report._finding import FindingCode, FindingRaised

__all__ = ["OutputCheck"]


class OutputCheck:
    def __init__(self, schema: OpContract, axes: Mapping[str, int]) -> None:
        """axes: a workload's; the mutable tensors' declared shapes and dtypes are at its Point."""
        self.workload = workload = ContractWorkload.of(schema, axes)
        contract, point = workload.contract, workload.point
        self.declared = {
            o: (
                tuple(contract.storage_shape_of(o, point)),
                semantics.torch_dtypes(workload.schema_json["outputs"][o]["dtype"]["any_of"][0]),
            )
            for o in contract.outputs
        }

    def reference(self, input_sets: Sequence[Mapping[str, Any]], rank: int) -> dict[str, dict[str, Any]]:
        """What the Candidate's outputs on input_sets[rank] are compared with, as compare_outputs takes them:
        the Reference's outputs, in float32, or float64 where float32 overflows; the magnitudes a dot_product
        bound scales by; the Baseline's outputs a baseline_relative bound is relative to. The Reference reads
        every rank's input sets where the op's ranks call together."""
        import torch

        workload = self.workload
        contract, point = workload.contract, workload.point
        is_dot_product = workload.has_error_bound("dot_product")
        if is_dot_product:
            magnitudes = [
                dict(args)
                | {
                    n: semantics.absolute_value(v, point.dtypes[n])
                    for n, v in args.items()
                    if torch.is_tensor(v)
                }
                for args in input_sets
            ]
        for decode in (semantics.to_float32, semantics.to_float64):
            compared = {"reference_outputs": contract.reference_outputs(point, input_sets, rank, decode)}
            if is_dot_product:
                compared["magnitude_outputs"] = contract.reference_outputs(point, magnitudes, rank, decode)
            # float32 overflows where the op's value can be finite (a sum that cancels): float64 then
            if all(t.isfinite().all() for outputs in compared.values() for t in outputs.values()):
                break
        if workload.has_error_bound("baseline_relative"):
            baseline = contract.baseline()
            if baseline is None:
                raise ValueError(f"{contract.contract_id}: a baseline_relative bound needs a baseline")
            device = next(v.device for v in input_sets[rank].values() if torch.is_tensor(v))
            call = workload.assemble_call(input_sets[rank], device)
            compared["baseline_outputs"] = contract.call_mutable_tensors(baseline(**call), call)
        return compared

    def verify(self, reference_outputs: dict[str, dict[str, Any]], produced: Any, trial: int) -> None:
        """produced: the Candidate's mutable tensors by name. Raises FindingRaised when its outputs are not the
        declared ones (WRONG_OUTPUTS), or the schema's compare rejects its mutable tensors against the
        Reference's or raises (OUTPUTS_MISMATCH)."""
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
            results = semantics.compare_outputs(self.workload.schema_json, produced, **reference_outputs)
            failed = [f"{o}: {r.detail}" for o, r in results.items() if not r.is_passed]
            detail = "; ".join(failed)
        except Exception as error:  # noqa: BLE001 -- a compare that cannot read the outputs rejects them
            failed, detail = [True], f"the compare raised {type(error).__name__}: {error}"
        if failed:
            raise FindingRaised(FindingCode.OUTPUTS_MISMATCH, f"trial {trial}: {detail}")
