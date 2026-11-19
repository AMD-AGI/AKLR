# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DeepSeek-V4.1 references on a rank of a TP group."""

import torch

from tests.op_case import OpCase


def test_the_other_ranks_wo_b_outputs_join_this_ranks_before_the_ffn_boundary():
    case = next(
        case
        for case in OpCase.all()
        if case.contract_id == "deepseekv41_mono_k2_attention@1.0" and case.point.attrs["has_peer_x"]
    )
    args = case.inputs()
    # every stream is then the attention's output alone
    args |= {"post_mix": torch.ones_like(args["post_mix"]), "comb_mix": torch.zeros_like(args["comb_mix"])}
    peers = args["peer_x"]
    alone = case.reference_outputs({**args, "peer_x": torch.zeros_like(peers)})["residual_out"][:, 0]
    summed = case.reference_outputs(args)["residual_out"][:, 0]
    assert peers.abs().sum() > 0
    assert torch.equal(summed, (alone + peers).bfloat16().float())
