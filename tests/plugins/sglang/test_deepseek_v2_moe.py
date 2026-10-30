# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""deepseek_v2_moe: its reference's routing and its committed kernels."""

from pathlib import Path

import pytest
import torch

from aklr.core.contract import semantics
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import PointSpec
from aklr.core.validation.stages._audit._stage import check_kernel
from aklr.plugins.sglang.models.deepseek_v2.reference import (
    deepseek_v2_moe_reference,
)

_IMPLS = Path(__file__).parents[3] / "aklr/plugins/sglang/models/deepseek_v2/impls"
# the GLM-5.2-MXFP4 decode Points the kernels serve: an intermediate size per rank of 512
# at TP4, 256 at TP8
_POINTS = {512: [1, 2, 4, 8, 16, 256], 256: [32, 128, 256]}


def test_kernels_are_openai_gluon_schemas_of_sglangs_layout():
    """Each deepseek_v2_moe kernel is a Triton (Gluon) one of OpenAI's, every source marked
    as theirs and passing the audit, on sglang's shuffled experts of GLM-5 at TP4 or TP8;
    together they serve the decode Points once each."""
    registry, points = ContractRegistry.default(), {}
    for path in sorted(_IMPLS.rglob("*.json")):
        schema = registry.parse_schema(path.read_bytes(), str(path))
        if schema.contract_id != "deepseek_v2_moe@1.0":
            continue
        kernel = schema.kernel
        assert (kernel.language, kernel.author) == ("triton", "openai")
        assert kernel.requires == {"triton": ">=3.8"}
        assert not kernel.state_bytes and not kernel.peer_bytes
        assert all(s.content.startswith("# Created by OpenAI\n") for s in kernel.sources)
        assert schema.layout_variants.names() == ("shuffled_16x16",)
        axes = {n: schema.axes[n].lo for n in ("model_dim", "n_routed", "inter_dim")}
        assert axes["model_dim"] == 6144 and axes["n_routed"] == 256
        findings = check_kernel(
            str(path),
            kernel,
            call_keys=schema.kernel_call_keys,
            tensor_names=tuple(schema.tensors()),
            mutable_names=schema.mutable_tensors,
        )
        assert findings == ()
        for w in schema.workloads:
            assert w.scalars == {"routed_scaling_factor": 2.5}
            assert w.accuracy is None and w.speedup is None
            points.setdefault(axes["inter_dim"], []).append(w.axes["n_tokens"])
    assert {k: sorted(v) for k, v in points.items()} == _POINTS


def _moe_case(second_logit_term: float) -> tuple[PointSpec, dict]:
    """One token through 16 routed experts, top-8: experts 0-6 win on large logits, and
    experts 8 and 9 compete for the eighth slot, 8's logit 1 + second_logit_term, 9's 1 at
    a correction bias 1e-5 larger; expert 8 alone has nonzero weights."""
    point = PointSpec(
        contract_id="deepseek_v2_moe@1.0",
        axes={"n_tokens": 1, "model_dim": 256, "n_routed": 16, "inter_dim": 256},
        dtypes={
            "hidden_states": "bfloat16",
            "router_weight": "bfloat16",
            "correction_bias": "float32",
            "w1": "float4_e2m1",
            "w1_scale": "float8_e8m0",
            "w2": "float4_e2m1",
            "w2_scale": "float8_e8m0",
            "out": "bfloat16",
        },
        attrs={
            "topk": 8,
            "num_expert_group": 1,
            "topk_group": 1,
            "renormalize": True,
            "scoring_func": "sigmoid",
        },
        layout_variant_name="row_major",
    )
    hidden = torch.zeros(1, 256, dtype=torch.bfloat16)
    hidden[0, :2] = 1.0
    router = torch.zeros(16, 256, dtype=torch.bfloat16)
    router[:7, 0], router[10:, 0] = 4.0, -4.0
    router[8, 0], router[8, 1], router[9, 0] = 1.0, second_logit_term, 1.0
    bias = torch.zeros(16)
    bias[9] = 1e-5
    w1 = torch.zeros(17, 512, 128, dtype=torch.uint8)
    w2 = torch.zeros(17, 256, 128, dtype=torch.uint8)
    w1[8] = w2[8] = 0x22  # every fp4 value 1.0
    args = {
        "hidden_states": hidden,
        "router_weight": router,
        "correction_bias": bias,
        "w1": w1,
        "w1_scale": torch.full((17, 512, 8), 127, dtype=torch.uint8),  # e8m0 1.0
        "w2": w2,
        "w2_scale": torch.full((17, 256, 8), 127, dtype=torch.uint8),
        "routed_scaling_factor": 2.5,
    }
    return point, args


@pytest.mark.parametrize(
    ("second_logit_term", "routed_to_8"),
    [(2.0**-10, False), (2.0**-6, True)],
    ids=["below-bf16", "in-bf16"],
)
def test_reference_routes_on_the_logits_in_bf16(second_logit_term, routed_to_8):
    """Experts 8 and 9 tie once the logits round to bf16, where 1 + 2**-10 is 1, and the
    bias picks expert 9: exact logits would pick expert 8. A difference bf16 holds, 2**-6,
    picks expert 8."""
    point, args = _moe_case(second_logit_term)
    out = deepseek_v2_moe_reference(point, args, semantics.to_float64)
    assert out.dtype == torch.float64 and out.shape == (1, 256)
    assert bool(out.abs().sum() > 0) is routed_to_8
