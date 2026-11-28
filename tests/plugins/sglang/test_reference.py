# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.plugins.sglang.layers.dsa_indexer.reference import key_columns
from aklr.plugins.sglang.monokernels.glm53.reference import INDEX_DIM, INDEX_PAGE, ROPE_DIM
from tests.op_case import OpCase
from tests.plugins.sglang import cases


@pytest.mark.gpu
@pytest.mark.parametrize("op_case", OpCase.params(cases.BACKEND_POINTS))
def test_backend_matches_reference(op_case: OpCase) -> None:
    """``check_backend_matches_reference`` in sglang-test: sglang's call on the aiter its image ships."""
    op_case.check_backend_matches_reference()


def test_router_ids_of_equal_logits_may_swap() -> None:
    """qwen2_moe_router: experts 0 and 1 share a router row, so their bf16 logits are equal and
    either may come first; swapping a row's first and last chosen experts still fails."""
    case = OpCase.of("qwen2_moe_router@1.0")
    args = case.inputs()
    args["router_weight"][1] = args["router_weight"][0]
    reference = case.reference_outputs(args)
    ids = reference["topk_ids"]
    rows = ((ids == 0).any(-1) & (ids == 1).any(-1)).nonzero()[:, 0]
    assert len(rows) > 0
    swapped = ids.clone()
    for r in rows.tolist():
        swapped[r][ids[r] == 0], swapped[r][ids[r] == 1] = 1, 0
    assert "topk_ids" not in case.compare({**reference, "topk_ids": swapped}, args)

    scores = reference["topk_id_scores"].float()
    assert scores[0, ids[0, 0]] - scores[0, ids[0, -1]] > 0.25  # more than two bf16 values below 16
    far = ids.clone()
    far[0, [0, -1]] = ids[0, [-1, 0]]
    assert "topk_ids" in case.compare({**reference, "topk_ids": far}, args)


@pytest.mark.parametrize("contract_id", ["dsa_indexer_key_prepare@1.0", "dsa_indexer_prepare@1.0"])
def test_index_k_cache_key_at_twice_the_scale_stands_for_the_reference(contract_id: str) -> None:
    """A key stored at twice the reference's scale, its fp8 values half, passes; at four times the
    scale, or at twice it with the values left as they are, it still fails."""
    import torch

    from aklr.plugins.sglang.layers.dsa_indexer.reference import HEAD_DIM, PAGE_SIZE

    case = OpCase.of(contract_id)
    args = case.inputs()
    reference = case.reference_outputs(args)
    page, values, scales = (c[:1] for c in key_columns(reference["written_slots"]))

    def stored_at(factor: float, value_factor: float) -> dict:
        pages = reference["index_k_cache"].clone().view(torch.uint8).reshape(-1, PAGE_SIZE * (HEAD_DIM + 4))
        scale = pages[page, scales].contiguous().view(torch.float32) * factor
        pages[page, scales] = scale.view(torch.uint8)
        codes = pages[page, values].view(torch.float8_e4m3fn).float() * value_factor
        pages[page, values] = codes.to(torch.float8_e4m3fn).view(torch.uint8)
        return {
            **reference,
            "index_k_cache": pages.view(torch.float8_e4m3fn).reshape(args["index_k_cache"].shape),
        }

    def passes(outputs: dict) -> bool:
        return "index_k_cache" not in case.compare(outputs, args)

    assert passes(stored_at(2.0, 0.5))
    assert not passes(stored_at(4.0, 0.25))
    assert not passes(stored_at(2.0, 1.0))


def _glm53_rank_call(call: dict, point, rank: int, tp: int) -> dict:
    """``rank``'s call in a group of ``tp``: its share of ``call``'s heads and MLP intermediates (each
    expert's, or the dense MLP's), the rest whole."""
    ax = point.axes
    heads = ax["n_heads"] // tp
    q_rows, o_cols = heads * (ax["qk_nope_dim"] + ROPE_DIM), heads * ax["v_head_dim"]

    def share(tensor, size, dim, blocks=1):
        size //= blocks
        return tensor.narrow(dim, rank * size, size).contiguous()

    def gate_up(tensor, size, dim, blocks=1):
        gate, up = tensor.chunk(2, dim=dim)
        return torch.cat([share(gate, size, dim, blocks), share(up, size, dim, blocks)], dim=dim)

    shares = {
        "q_b_weight": share(call["q_b_weight"], q_rows, 0),
        "q_b_scale": share(call["q_b_scale"], q_rows, 0, 128),
        "w_kc": share(call["w_kc"], heads, 0),
        "w_vc": share(call["w_vc"], heads, 0),
        "o_weight": share(call["o_weight"], o_cols, 1),
        "o_scale": share(call["o_scale"], o_cols, 1, 128),
    }
    if "w1" in call:
        inter = ax["inter_dim"] // tp
        shares |= {
            "w1": gate_up(call["w1"], inter, 1),
            "w1_scale": gate_up(call["w1_scale"], inter, 1, 128),
            "w2": share(call["w2"], inter, 2),
            "w2_scale": share(call["w2_scale"], inter, 2, 128),
        }
    else:
        inter = ax["dense_inter"] // tp
        shares |= {
            "gate_up_weight": gate_up(call["gate_up_weight"], inter, 0),
            "gate_up_scale": gate_up(call["gate_up_scale"], inter, 0, 128),
            "down_weight": share(call["down_weight"], inter, 1),
            "down_scale": share(call["down_scale"], inter, 1, 128),
        }
    return {**call, **shares, "tp_size": tp}


@pytest.mark.parametrize(
    "contract_id", ["glm53_moe_layer@1.0", "glm53_moe_index_layer@1.0", "glm53_dense_index_layer@1.0"]
)
def test_glm53_reference_over_two_ranks_is_one_ranks(contract_id: str) -> None:
    """Heads and MLP intermediates split over two ranks in whole 128-blocks, the rest on both: each
    rank's out is the one rank's, but for the sums' order (a bf16 step), and so are its pools and its
    top-k."""
    case = OpCase.of(contract_id)
    args = case.inputs()
    whole = case.reference_outputs(args)
    call = case.request.reference_call(args, "cpu")
    group = [_glm53_rank_call(call, case.point, rank, 2) for rank in range(2)]
    out = "out" if "out" in whole else "layer_out"
    for rank in range(2):
        split = case.contract.reference.reference(group, rank)
        torch.testing.assert_close(split[out].float(), whole[out].float(), rtol=2**-7, atol=1e-5)
        assert all(torch.equal(split[n], t) for n, t in whole.items() if n != out)


def _vocab_split(case: OpCase, args: dict, name: str) -> list[dict]:
    """Two ranks' calls of ``args``, ``name``'s rows (the vocabulary's) split between them, the rest on both."""
    call = case.request.reference_call(args, "cpu")
    rows = case.point.axes["vocab_rows"] // 2
    return [{**call, name: call[name][r * rows : (r + 1) * rows], "tp_size": 2} for r in range(2)]


def test_glm53_head_over_two_ranks_is_one_ranks() -> None:
    """The vocabulary's rows split over two ranks, hidden_states on both: each rank's logits are the one rank's."""
    case = OpCase.of("glm53_head@1.0")
    args = case.inputs()
    whole = case.reference_outputs(args)["logits"]
    group = _vocab_split(case, args, "lm_head_weight")
    for rank in range(2):
        assert torch.equal(case.contract.reference.reference(group, rank)["logits"], whole)


def test_glm53_nextn_entry_over_two_ranks_is_one_ranks() -> None:
    """The embedding's rows split over two ranks, the rest on both: each rank's out is the one rank's."""
    case = OpCase.of("glm53_nextn_entry@1.0")
    args = case.inputs()
    whole = case.reference_outputs(args)["out"]
    group = _vocab_split(case, args, "embed_weight")
    for rank in range(2):
        assert torch.equal(case.contract.reference.reference(group, rank)["out"], whole)


def test_glm53_embed_over_two_ranks_is_one_ranks() -> None:
    """The embedding's rows split over two ranks, the token ids on both: each rank's embeddings are the one rank's,
    each row a row of the table."""
    case = OpCase.of("glm53_embed@1.0")
    args = case.inputs()
    whole = case.reference_outputs(args)["embeddings"]
    assert torch.equal(whole.float(), args["embed_weight"][args["input_ids"]].float())
    group = _vocab_split(case, args, "embed_weight")
    for rank in range(2):
        assert torch.equal(case.contract.reference.reference(group, rank)["embeddings"], whole)


@pytest.mark.parametrize("contract_id", ["glm53_moe_index_layer@1.0", "glm53_dense_index_layer@1.0"])
def test_glm53_a_kv_pool_smaller_than_the_page_table_is_outside_the_domain(contract_id: str) -> None:
    """A page table names pages of the KV pool: max_pages of them must fit in it."""
    point = OpCase.of(contract_id).point
    with pytest.raises(ValueError, match="outside the domain"):
        OpCase(point.model_copy(update={"axes": {**point.axes, "n_slots": INDEX_PAGE}}))


def test_glm53_page_table_names_only_pages_of_the_kv_pool() -> None:
    """The index-K cache may hold a page more than the KV pool, as SGLang's does; the page table never names it."""
    point = OpCase.of("glm53_moe_index_layer@1.0").point
    case = OpCase(point.model_copy(update={"axes": {**point.axes, "n_pages": point.axes["n_pages"] + 1}}))
    assert int(case.inputs()["page_table"].max()) < point.axes["n_slots"] // INDEX_PAGE


def test_glm53_topk_of_equal_scores_may_swap() -> None:
    """glm53_moe_index_layer: row 1 scores every key, row 0 fewer than it takes. A copy of row 1's
    lowest taken key at a slot it leaves scores alike, and either may be taken; row 1 may come in any
    order. Row 0 reordered, a slot where it has none, or a slot of row 1 far below its taken ones fails."""
    case = OpCase.of("glm53_moe_index_layer@1.0")
    k, n_keys = case.point.axes["index_topk"], case.point.axes["max_pages"] * INDEX_PAGE
    base = case.inputs()
    base["seq_lens"][:2] = torch.tensor([k - 3, n_keys])
    first = case.reference_outputs(base)
    top, written = first["topk_indices"][1].tolist(), base["cache_locations"].tolist()
    left = next(s for s in range(n_keys) if s not in top and s not in written)

    def key_bytes(cache: torch.Tensor, slot: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(the cache's pages as bytes, slot's page, its key's byte columns)."""
        page, values, scales = key_columns(torch.tensor([slot]))
        return (
            cache.view(torch.uint8).view(-1, INDEX_PAGE * (INDEX_DIM + 4)),
            page[0, 0],
            torch.cat([values[0], scales[0]]),
        )

    for taken in top:
        args = {n: t.clone() if torch.is_tensor(t) else t for n, t in base.items()}
        to_pages, to_page, to = key_bytes(args["index_k_cache"], left)
        from_pages, from_page, from_ = key_bytes(first["index_k_cache"], taken)
        to_pages[to_page, to] = from_pages[from_page, from_]
        reference = case.reference_outputs(args)
        ids = reference["topk_indices"]
        if (ids[1] == taken).any() != (ids[1] == left).any():
            break
    else:
        raise AssertionError("no taken key of row 1 ties a copy at its last place")

    def passes(row: int, row_ids: torch.Tensor) -> bool:
        actual = ids.clone()
        actual[row] = row_ids
        return "topk_indices" not in case.compare({**reference, "topk_indices": actual}, args)

    swapped = torch.where(ids[1] == taken, left, torch.where(ids[1] == left, taken, ids[1]))
    assert passes(1, swapped)
    assert passes(1, ids[1].flip(0))
    assert not passes(0, torch.cat([ids[0, : k - 3].flip(0), ids[0, k - 3 :]]))
    other = next(s for s in range(n_keys) if s not in ids[0].tolist())
    assert not passes(
        0, torch.cat([ids[0, : k - 3], torch.tensor([other], dtype=ids.dtype), ids[0, k - 2 :]])
    )
    scores = reference["topk_id_scores"][1]
    lowest, far = int(ids[1][scores[ids[1].long()].argmin()]), int(scores.argmin())
    assert not passes(1, torch.where(ids[1] == lowest, far, ids[1]))
