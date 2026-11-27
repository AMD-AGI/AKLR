from pathlib import Path

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from tests.op_case import OpCase


@pytest.fixture
def paged_schema():
    path = Path(
        "aklr/plugins/aiter/provider/topk/impls/flydsl/topk_transform_paged_page_size64_has_page_tables_topk512/b400fb652a4ededa.json"
    )
    return ContractRegistry.default().parse_schema(path.read_bytes(), str(path))


@pytest.mark.gfx950
@pytest.mark.parametrize("length", [0, 512, 513, 16384, 16385, 262208])
@pytest.mark.parametrize("kind", ["negative", "tied"])
def test_paged_topk_handles_boundaries_and_finite_scores(builder_registry, paged_schema, length, kind):
    case = OpCase(
        paged_schema.point({"n_tokens": 1, "n_candidates": 262208, "n_plan_rows": 2, "n_pages": 4097})
    )
    inputs = case.inputs()
    if kind == "negative":
        inputs["scores"].copy_(-torch.arange(262208, 0, -1, dtype=torch.float32)[None])
    else:
        inputs["scores"].copy_((torch.arange(262208, dtype=torch.float32) % 3 - 1)[None])
    inputs["seq_lens"].fill_(length)
    call = case.call(inputs, "cuda:0")
    call["out_page_indices"] = torch.full((1, 512), -99, dtype=torch.int32, device="cuda:0")
    builder_registry.build(paged_schema)(**call)
    torch.cuda.synchronize(0)
    output = call["out_page_indices"].cpu()[0].long()
    count = min(length, 512)
    assert (output[count:] == -1).all()
    slots = output[:count]
    assert (slots >= 0).all()
    pages = inputs["page_tables"][0].long()
    inverse = torch.empty_like(pages)
    inverse[pages] = torch.arange(pages.numel())
    positions = inverse[slots // 64] * 64 + slots % 64
    assert ((positions >= 0) & (positions < length)).all()
    assert positions.unique().numel() == count
    selected = inputs["scores"][0, positions].sort().values
    expected = torch.topk(inputs["scores"][0, :length], count).values.sort().values
    torch.testing.assert_close(selected, expected, atol=0, rtol=0)
