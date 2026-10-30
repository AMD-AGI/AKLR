# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Each committed GDN decode kernel, and SGLang's own decode, on a padded graph batch.

SGLang pads a captured decode batch with several -1 rows; the drawn indices hold one
at most. Both runs must match the reference and leave every padding row's and
unindexed slot's state bit for bit as it was.
"""

import json
from pathlib import Path

import pytest
import torch

from aklr.core.contract import semantics
from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime import seam
from aklr.core.validation.stages._bench._evaluation._trials import TrialRequest
from aklr.plugins.sglang.models.qwen3_5 import reference

IMPLS = Path(__file__).parents[3] / "aklr/plugins/sglang/models/qwen3_5/impls/triton"
SCHEMAS = [
    pytest.param(
        path,
        id=f"{path.parent.name.split('_n_v_heads')[0]}-m{rows}",
    )
    for path in sorted(IMPLS.glob("gdn_decode*/*.json"))
    for rows in [json.loads(path.read_text())["workloads"][0]["axes"]["n_tokens"]]
]


def padded_indices(rows, slots):
    """Distinct live slots in shuffled order; every third row and the last are -1."""
    generator = torch.Generator().manual_seed(rows)
    indices = torch.randperm(slots, generator=generator)[:rows].to(torch.int32)
    indices[2::3] = -1
    indices[-1] = -1
    return indices


@pytest.mark.gpu
@pytest.mark.gfx950
@pytest.mark.parametrize("runner", ["kernel", "sglang"])
@pytest.mark.parametrize("path", SCHEMAS)
def test_padded_graph_batch_matches_reference(path, runner, builder_registry):
    registry = ContractRegistry.default()
    schema = registry.parse_schema(path.read_bytes(), str(path))
    contract = registry.get(schema.contract_id)
    (workload,) = schema.workloads
    request = TrialRequest.of_workload(
        str(path),
        schema,
        contract,
        workload,
        thresholds={},
        incumbent=None,
        baseline=None,
    )
    point = request.point
    rows, slots = point.axes["n_tokens"], point.axes["n_slots"]
    args = semantics.row_major_inputs(
        request.schema_json,
        request.workload_json,
        request.distributions,
        input_set="numeric/0",
    )
    indices = padded_indices(rows, slots)
    args["cache_indices"] = indices
    call = request.assemble_call(args, "cuda")
    if runner == "kernel":
        outputs = {
            o: torch.empty(
                contract.storage_shape_of(o, point),
                dtype=semantics.torch_dtypes(point.dtypes[o])[0],
                device="cuda",
            )
            for o in contract.outputs
        }
        builder_registry.build(schema)(**(call | outputs))
        actual = outputs | {n: call[n] for n in contract.mutable_inputs}
    else:
        actual = contract.call_mutable_tensors(seam.backend_call(contract, call), call)
    torch.cuda.synchronize()
    reference_fn = getattr(reference, contract.reference_name)
    on_device = {k: v.cuda() if torch.is_tensor(v) else v for k, v in args.items()}
    expected = contract.mutable_tensors_by_name(reference_fn(point, on_device, semantics.to_float64))
    results = semantics.compare_outputs(
        request.schema_json,
        {o: t.cpu() for o, t in actual.items()},
        {o: t.cpu() for o, t in expected.items()},
    )
    assert {o: r.detail for o, r in results.items() if not r.is_passed} == {}
    live = set(indices.tolist()) - {-1}
    kept = [s for s in range(slots) if s not in live]
    for name in ("conv_state", "delta_state"):
        assert torch.equal(actual[name][kept].cpu(), args[name][kept]), name
