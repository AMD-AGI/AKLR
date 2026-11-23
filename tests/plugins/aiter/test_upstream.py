# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the installed entry point and the public op on the pinned AITER CI stack."""

import importlib
import subprocess
import sys

import pytest


def test_aiter_public_names_resolve_to_the_installed_host_functions():
    pytest.importorskip("aiter.plugin")
    pytest.importorskip("aiter.ops.rmsnorm")

    from aklr.plugins.aiter import api

    expected = {
        "aiter.fused_moe": ("fused_moe",),
        "aiter.mla": ("mla_decode_fwd",),
        "aiter.tuned_gemm": ("gemm_a16w16", "gemm_a16w16_out"),
        "aiter.ops.gemm_op_a8w8": (
            "gemm_a8w8_blockscale",
            "gemm_a8w8_blockscale_bpreshuffle",
            "gemm_a8w8_bpreshuffle",
        ),
        "aiter.ops.mha": ("flash_attn_varlen_func",),
        "aiter.ops.mhc": ("mhc_post", "mhc_pre", "mhc_fused_post_pre"),
        "aiter.ops.mla_sparse_prefill": ("mla_sparse_prefill_v4",),
        "aiter.ops.quant": ("per_group_quant_hip", "per_1x32_mx_quant_hip"),
        "aiter.ops.rmsnorm": ("rms_norm", "rmsnorm2d_fwd", "rmsnorm2d_fwd_with_add"),
        "aiter.ops.topk": ("biased_grouped_topk", "topk_gating"),
        "aiter.ops.triton.attention.pa_mqa_logits": ("deepgemm_fp8_paged_mqa_logits",),
        "aiter.ops.triton.gated_delta_net.causal_conv1d_prefill": ("causal_conv1d_split_qkv_triton_tile_fn",),
        "aiter.ops.triton.gluon.mla_gluon": ("mla_gluon",),
    }
    assert set(api.__all__) == {name for names in expected.values() for name in names}
    for module, names in expected.items():
        upstream = importlib.import_module(module)
        for name in names:
            assert getattr(api, name) is getattr(upstream, name)


@pytest.mark.gfx950
def test_real_aiter_discovers_validates_and_runs_the_plugin(monkeypatch, committed_kernel_cache):
    from aklr.core.platform.env import ENV_KERNEL_CACHE_DIR

    pytest.importorskip("aiter.plugin", reason="this AITER release predates the provider interface")
    monkeypatch.setenv(ENV_KERNEL_CACHE_DIR, str(committed_kernel_cache))
    # Isolate AITER configuration and GPU dispatch from tests that install other providers.
    subprocess.run([sys.executable, "-m", "tests.plugins.aiter.test_upstream"], check=True)


if __name__ == "__main__":
    import torch
    from aiter import plugin

    from aklr.plugins.aiter.api import rmsnorm2d_fwd
    from aklr.plugins.aiter.provider import declining, enable, provider

    # AITER registers its lazy routes when their owning modules are imported.
    for module in (
        "aiter.fused_moe",
        "aiter.mla",
        "aiter.tuned_gemm",
        "aiter.ops.gemm_op_a8w8",
        "aiter.ops.mha",
        "aiter.ops.mhc",
        "aiter.ops.mla_sparse_prefill",
        "aiter.ops.quant",
        "aiter.ops.topk",
        "aiter.ops.triton.attention.pa_mqa_logits",
        "aiter.ops.triton.gated_delta_net.causal_conv1d_prefill",
        "aiter.ops.triton.gemm.basic.gemm_a16w16",
        "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale",
        "aiter.ops.triton.gluon.mla_gluon",
    ):
        importlib.import_module(module)
    report = plugin.validate("aklr")
    assert report and set(report.ops) == set(provider.op_versions), report
    assert enable("AKLR integration test")
    for width, expected_route in ((6144, "provider"), (16, "native")):
        x = torch.randn(512, width, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(width, device=x.device, dtype=x.dtype)
        with declining():
            reference = rmsnorm2d_fwd(x, weight, 1e-5)
        explanation = plugin.explain("rmsnorm", x, weight, epsilon=1e-5, use_model_sensitive_rmsnorm=False)
        assert explanation.decision.route == expected_route
        result = rmsnorm2d_fwd(x, weight, epsilon=1e-5)
        assert plugin.last_decision().route == expected_route
        torch.testing.assert_close(result, reference, atol=0.03125, rtol=0.015625)
