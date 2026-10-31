# Created by OpenAI
"""State semantics and migrated schema coverage of the GDN plugin."""

import json
from pathlib import Path

from aklr.core.contract.registry import ContractRegistry
from aklr.core.validation.stages._audit._stage import check_kernel
from aklr.plugins.sglang.models.qwen3_5.contract import GDN_PREFILL_GROUP_FP8_QUANT
from tests.op_case import OpCase


def test_contract_compares_outputs_and_both_live_states():
    assert GDN_PREFILL_GROUP_FP8_QUANT.mutable_tensors == (
        "normalized",
        "values",
        "scales",
        "conv_state",
        "delta_state",
    )
    case = OpCase.of("gdn_prefill_group_fp8_quant@1.0")
    assert list(case.reference_outputs(case.inputs())) == list(case.contract.mutable_tensors)


def test_all_implementations_are_self_contained_schemas():
    root = Path(__file__).parents[3] / "aklr/plugins/sglang/models/qwen3_5/impls"
    registry = ContractRegistry.default()
    sources = set()
    workloads = 0
    for path in root.rglob("gdn_prefill_*/*.json"):
        schema = registry.parse_schema(path.read_bytes(), str(path))
        assert schema.kernel.language == "triton"
        assert schema.kernel.author == "OpenAI"
        assert schema.kernel.requires == {"triton": ">=3.8"}
        assert not schema.kernel.state_bytes and not schema.kernel.peer_bytes
        assert all(w.accuracy is None and w.speedup is None for w in schema.workloads)
        findings = list(
            check_kernel(
                str(path),
                schema.kernel,
                call_keys=schema.kernel_call_keys,
                tensor_names=tuple(schema.tensors()),
                mutable_names=schema.mutable_tensors,
            )
        )
        assert findings == []
        sources.add(schema.kernel.content_hash)
        workloads += len(schema.workloads)
        assert json.loads(path.read_text())["contract_id"] == "gdn_prefill_group_fp8_quant@1.0"
    assert len(sources) == 9 and workloads == 21
