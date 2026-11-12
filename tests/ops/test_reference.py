# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.ops.topk.schema import BIASED_GROUPED_TOPK
from tests.op_case import OpCase


class TestReference:
    def test_outputs_have_the_declared_shapes(self, op_case: OpCase) -> None:
        schema = op_case.schema()
        outputs = op_case.reference(op_case.calls())
        assert [tuple(t.shape) for t in outputs] == [
            tuple(s) for s in schema.get_output_shapes({**schema.const_axes, **op_case.axes})
        ]
        assert all(t.isfinite().all() for t in outputs)

    def test_check_passes_the_reference_in_the_output_dtypes(self, op_case: OpCase) -> None:
        reference = op_case.reference(op_case.calls())
        rounded = [t.to(d) for t, d in zip(reference, op_case.output_dtypes, strict=True)]
        assert op_case.template.standard.check(reference, rounded)

    def test_check_fails_an_output_off_the_reference(self, op_case: OpCase) -> None:
        reference = op_case.reference(op_case.calls())
        off = [
            (-t - 1 if t.is_floating_point() else t + 1).to(d)
            for t, d in zip(reference, op_case.output_dtypes, strict=True)
        ]
        assert not op_case.template.standard.check(reference, off)

    @pytest.mark.gpu
    @pytest.mark.gfx950
    def test_backend_matches_the_reference(self, op_case: OpCase) -> None:
        if op_case.ranks > 1:
            pytest.skip(
                "a group op's backend runs on every rank of its group: the bench's GPU tests cover it"
            )
        schema = op_case.schema()
        call = op_case.call(device="cuda")
        shapes = schema.get_output_shapes({**schema.const_axes, **op_case.axes})
        for name, shape, dtype in zip(schema.outputs, shapes, op_case.output_dtypes, strict=True):
            call.setdefault(name, torch.empty(shape, dtype=dtype, device="cuda"))
        baseline = op_case.template.standard.baseline
        run = None if baseline is None else baseline(**call)
        if run is None:
            pytest.skip("no backend reads this storage")
        result = run()
        returned = list(result) if isinstance(result, (tuple, list)) else [result]
        actual = (
            returned
            if all(isinstance(t, torch.Tensor) for t in returned)
            else [call[name] for name in schema.outputs]
        )
        check = op_case.template.standard.check
        assert check([t.cpu() for t in op_case.reference([call])], [t.cpu() for t in actual])


@pytest.mark.parametrize(
    ("case_id", "seed", "axes"),
    [
        pytest.param("gemm_bf16_nt", 1410285289, {"M": 4, "N": 32, "K": 64}, id="gemm_bf16_nt_n32_k64"),
        pytest.param(
            "fused_moe_shuffled_16x16",
            4215199382,
            {"n_tokens": 4, "model_dim": 512, "n_experts": 8, "topk": 2, "inter_dim": 256},
            id="fused_moe_shuffled_16x16_model_dim512_n_experts8_topk2_inter_dim256",
        ),
    ],
)
def test_an_op_test_draws_the_bench_draw_of_its_workload(case_id: str, seed: int, axes: dict) -> None:
    """The bench draws trial 0 of a workload from the seed of its schema's name and Var axes."""
    case = OpCase.from_id(case_id)
    expected = case.template.standard.init(**axes, device="cpu", seed=seed)
    drawn = case.call()
    tensors = [n for n, v in expected.items() if isinstance(v, torch.Tensor)]
    # bytes, not values: fp4 weights have no equality
    assert tensors and all(
        torch.equal(drawn[n].view(torch.uint8), expected[n].view(torch.uint8)) for n in tensors
    )


def test_a_group_reference_is_the_whole_groups_experts_in_one_fused_moe() -> None:
    """mega_moe on rank 1 of 2 equals fused_moe over both ranks' experts, rank 0's first, on rank 1's tokens."""
    case = OpCase.from_id("mega_moe_gate_up_interleaved")
    calls = case.calls()
    merged = {n: torch.cat([c[n] for c in calls]) for n in ("w1", "w2", "w1_scale", "w2_scale")}
    fused_moe_reference = OpCase.from_id("fused_moe_gate_up_interleaved").template.standard.reference
    expected = fused_moe_reference(
        calls[1]["hidden_states"],
        merged["w1"],
        merged["w2"],
        calls[1]["topk_weights"],
        calls[1]["topk_ids"],
        activation="silu",
        w1_scale=merged["w1_scale"],
        w2_scale=merged["w2_scale"],
        swiglu_limit=calls[1]["swiglu_limit"],
        act_quant="mxfp8",
    )
    (actual,) = case.reference(calls, rank=1)
    # summed per owner rank, not in one pass: equal up to the order of the double additions
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12 * expected.abs().max().item())
    assert calls[1]["topk_ids"].max() >= case.axes["n_experts"]  # some token routes to rank 1's experts


def test_biased_grouped_topk_ranks_tied_experts_lower_first() -> None:
    """Of experts tied on the biased score, the lower one is chosen first."""
    reference = BIASED_GROUPED_TOPK.standard.reference
    logits = torch.zeros(1, 64, dtype=torch.float32)
    logits[0, 40] = 1.0  # one expert above 63 tied ones
    _, ids = reference(logits, torch.zeros(64), topk=4, num_expert_group=1, topk_group=1, renormalize=False)
    assert ids.tolist() == [[40.0, 0.0, 1.0, 2.0]]


def test_unified_attention_of_one_token_a_request_is_paged_decode() -> None:
    """With one new token a request, unified_attention reads what paged_attention_decode reads."""
    case = OpCase.from_id("paged_attention_decode_kv_split_pages")
    call = case.call()
    unified = OpCase.from_id("unified_attention").template.standard.reference
    (expected,) = case.reference([call])
    actual = unified(
        call["query"],
        call["key_cache"],
        call["value_cache"],
        torch.arange(call["query"].shape[0] + 1, dtype=torch.int32),
        call["context_lens"],
        call["block_tables"],
        call["scale"],
        max_seqlen_q=1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("case_id", "outputs"),
    [
        pytest.param(
            "mhc_fused_post_pre",
            [(torch.float32, 2e-3), (torch.float32, 2e-3), (torch.bfloat16, 1e-2), (torch.bfloat16, 1e-2)],
            id="mhc_test_2e-3",
        ),
        pytest.param(
            "flash_mla_with_kvcache_sparse",
            [(torch.bfloat16, 1e-2), (torch.float32, 1e-3)],
            id="mla_sparse_lse_1e-3",
        ),
        pytest.param(
            "flash_mla_with_kvcache_dense",
            [(torch.bfloat16, 1e-2), (torch.float32, 1e-3)],
            id="mla_dense_lse_1e-3",
        ),
        pytest.param(
            "flash_mla_with_kvcache_ragged",
            [(torch.bfloat16, 1e-2), (torch.float32, 1e-3)],
            id="mla_ragged_lse_1e-3",
        ),
        pytest.param("paged_attention_decode", [(torch.bfloat16, 1e-2)], id="paged_decode_bf16"),
        pytest.param("paged_attention_decode_kv_split_pages", [(torch.float16, 1e-3)], id="paged_decode_f16"),
        pytest.param("unified_attention", [(torch.bfloat16, 1e-2)], id="unified_bf16"),
        pytest.param("unified_attention_kv_split_pages", [(torch.float16, 1e-3)], id="unified_f16"),
    ],
)
def test_check_holds_each_element_to_flashinfers_test_tolerance(case_id: str, outputs: list) -> None:
    """rtol = atol as flashinfer's tests set them: attention 1e-2 in bfloat16 and 1e-3 in float16, the
    mHC mixes 2e-3 (test_mhc_pre_big_fuse), the MLA lse 1e-3 (test_deepseek_mla)."""
    case = OpCase.from_id(case_id)
    reference = case.reference(case.calls())
    check = case.template.standard.check
    for i, (dtype, tolerance) in enumerate(outputs):
        for factor, passes in ((0.5, True), (3, False)):
            actual = [t.to(d).contiguous() for t, (d, _) in zip(reference, outputs, strict=True)]
            first = reference[i].reshape(-1)[0].item()
            actual[i].view(-1)[0] = first + factor * tolerance * (1 + abs(first))
            assert check(reference, actual) == passes, f"output {i} ({dtype}) off by {factor}x its tolerance"


@pytest.mark.parametrize(
    ("case_id", "passing", "failing"),
    [
        # SQNR >= 13 dB: flashinfer's MXFP4 MoE tests allow more (92% of elements within rtol 0.85)
        pytest.param("fused_moe_gate_up_interleaved", 0.10, 0.30, id="fused_moe_13dB"),
        # flashinfer's DeepGEMM mega MoE test: relative L2 error < 0.02
        pytest.param("mega_moe_gate_up_interleaved", 0.015, 0.03, id="mega_moe_rel_l2_0.02"),
    ],
)
def test_moe_check_bounds_the_relative_l2_error(case_id: str, passing: float, failing: float) -> None:
    case = OpCase.from_id(case_id)
    (reference,) = case.reference(case.calls())
    check = case.template.standard.check
    assert check([reference], [(reference * (1 + passing)).to(torch.bfloat16)])
    assert not check([reference], [(reference * (1 + failing)).to(torch.bfloat16)])


def test_swiglu_is_gpt_oss_clamped_at_its_limit() -> None:
    """aiter's swiglu, gpt-oss's: gate at most the limit and up within +/- it, then gate * sigmoid(1.702 * gate)
    * (up + 1), rounded to bf16. Every weight is 1.0 at element 0, 0 elsewhere: gate = up = the token's first
    element, which the down projection's element 0 returns."""
    from aklr.ops.fused_moe.schema import FUSED_MOE

    one = 2  # e2m1's 1.0, in a byte's low nibble: a row's element 0
    w1 = torch.zeros(1, 64, 16, dtype=torch.uint8)
    w1[..., 0] = one
    w2 = torch.zeros(1, 32, 16, dtype=torch.uint8)
    w2[0, 0, 0] = one
    hidden_states = torch.zeros(2, 32, dtype=torch.bfloat16)
    hidden_states[:, 0] = torch.tensor([1.0, 8.0])
    out = FUSED_MOE.standard.reference(
        hidden_states=hidden_states,
        w1=w1.view(torch.float4_e2m1fn_x2),
        w2=w2.view(torch.float4_e2m1fn_x2),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        activation="swiglu",
        w1_scale=torch.full((1, 64, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        w2_scale=torch.full((1, 32, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        swiglu_limit=7.0,
    )
    # 1 * sigmoid(1.702) * 2 = 1.69159 and 7 * sigmoid(11.914) * 8 = 55.99963, each to bf16
    assert out[:, 0].tolist() == [1.6953125, 56.0]
    assert not out[:, 1:].any()


def test_mega_moe_check_holds_each_element_to_atol_0_15_rtol_0_05() -> None:
    """flashinfer's DeepGEMM mega MoE test also asserts every element close at atol 0.15, rtol 0.05."""
    case = OpCase.from_id("mega_moe_gate_up_interleaved")
    (reference,) = case.reference(case.calls())
    actual = reference.to(torch.bfloat16).contiguous()
    first = reference.reshape(-1)[0].item()
    actual.view(-1)[0] = first + 3 * (0.15 + 0.05 * abs(first))
    assert not case.template.standard.check([reference], [actual])
