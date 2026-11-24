# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The targets the aiter families declare, against the ops aiter routes at the pinned commit."""

import pytest

from aklr.plugins.aiter import plugin
from tests.plugins.aiter.routed_ops import ROUTED_OPS


def test_the_declared_targets_are_aiter_ops_at_their_versions() -> None:
    assert plugin.targets() == {
        "biased_grouped_topk": (1,),
        "mhc_pre": (1,),
        "topk_gating": (1,),
        "gemm_a16w16": (1,),
        "gemm_a16w16_activation": (1,),
        "gemm_a8w8_blockscale": (1,),
        "gemm_a8w8_blockscale_bpreshuffle": (1,),
        "gemm_a8w8_blockscale_out": (1,),
        "gemm_a8w8_bpreshuffle": (1,),
        "rmsnorm": (2,),
        "rmsnorm_with_add": (1,),
    }
    assert plugin.targets() == {op: (ROUTED_OPS[op][0],) for op in plugin.targets()}


@pytest.mark.usefixtures("dispatch_starts")
def test_each_contract_takes_the_params_of_its_op() -> None:
    """What ``Provider`` checks when aiter first asks for an op: a mismatch there only logs an error and
    falls back to aiter."""
    for target, (version,) in plugin.targets().items():
        for dispatcher in plugin.dispatchers(target, version):
            dispatcher.contract.check_api_signature(ROUTED_OPS[target][1])
