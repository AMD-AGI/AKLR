# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aklr.plugins.vllm.models.muse_spark.qkvo import supports_qkvo


@pytest.mark.parametrize(
    ("m", "n", "k", "input_stride", "weight_stride", "with_bias", "dtype", "expected"),
    [
        (8, 2304, 8192, 8192, 8192, False, torch.bfloat16, True),
        (256, 8192, 1024, 1024, 1024, False, torch.bfloat16, True),
        (128, 2304, 8192, 8192, 8192, False, torch.bfloat16, True),
        (384, 2304, 8192, 8192, 8192, False, torch.bfloat16, False),
        (512, 8192, 1024, 1024, 1024, False, torch.bfloat16, False),
        (64, 8192, 1024, 1024, 1024, False, torch.bfloat16, False),
        (8, 2304, 8192, 8192, 8192, True, torch.bfloat16, False),
        (8, 2304, 8192, 8192, 8200, False, torch.bfloat16, False),
        (256, 8192, 1024, 1200, 1024, False, torch.bfloat16, False),
        (256, 4096, 8192, 8192, 8192, False, torch.bfloat16, False),
        (8, 2304, 8192, 8192, 8192, False, torch.float16, False),
    ],
)
def test_only_verified_dense_native_points_are_selected(
    m, n, k, input_stride, weight_stride, with_bias, dtype, expected
):
    x = torch.empty_strided((m, k), (input_stride, 1), device="meta", dtype=dtype)
    weight = torch.empty_strided((n, k), (weight_stride, 1), device="meta", dtype=dtype)
    assert supports_qkvo(x, weight, object() if with_bias else None) is expected


@pytest.mark.parametrize("selected_rows", [(8, 256), ()])
def test_qualified_rows_require_aklr_selection(selected_rows):
    from types import SimpleNamespace

    from aklr.plugins.vllm.models.muse_spark.qkvo import qualified_rows

    def explain(op, x, weight):
        assert op == "gemm_a16w16" and x.shape[1] == weight.shape[1]
        m = x.shape[0]
        if m in selected_rows:
            decision = SimpleNamespace(route="provider", provider="aklr")
        elif m == 128:
            decision = SimpleNamespace(route="provider", provider="other")
        elif m == 512:
            decision = None
        else:
            decision = SimpleNamespace(route="native", provider=None)
        return SimpleNamespace(decision=decision)

    weight = torch.empty((2304, 8192), device="meta", dtype=torch.bfloat16)
    assert qualified_rows(SimpleNamespace(explain=explain), weight) == frozenset(selected_rows)


@pytest.mark.gfx950
def test_muse_layers_use_only_provider_qualified_rows(monkeypatch):
    from types import SimpleNamespace

    import aiter
    from aiter import tuned_gemm
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.muse_spark.amd.model import MuseSparkAttention

    from aklr.plugins.vllm.models.muse_spark.qkvo import configure_qkvo

    # The AITER interface is external. Exercise the real vLLM layer methods and GPU tensors.
    calls = []

    def gemm(x, w):
        calls.append(x.shape[0])
        return torch.nn.functional.linear(x, w)

    def explain(op, x, w):
        assert op == "gemm_a16w16"
        choice = w.shape[0] == 2304 and x.shape[0] == 8
        return SimpleNamespace(
            decision=SimpleNamespace(
                route="provider" if choice else "native", provider="aklr" if choice else None
            )
        )

    plugin = SimpleNamespace(status=lambda: {"routed_ops": {"gemm_a16w16": {}}}, explain=explain)
    monkeypatch.setattr(aiter, "plugin", plugin, raising=False)
    monkeypatch.setattr(tuned_gemm, "gemm_a16w16", gemm)
    model = MuseSparkAttention.__new__(MuseSparkAttention)
    torch.nn.Module.__init__(model)
    model.qkv_proj = torch.nn.Linear(8192, 2304, bias=False, device="cuda", dtype=torch.bfloat16)
    model.o_proj = torch.nn.Linear(1024, 8192, bias=False, device="cuda", dtype=torch.bfloat16)
    original = UnquantizedLinearMethod()
    model.qkv_proj.quant_method = model.o_proj.quant_method = original
    parallel = SimpleNamespace(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        nnodes=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        expert_placement_strategy="linear",
        enable_eplb=False,
        use_sequence_parallel_moe=False,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="muse_spark_text")),
    )
    worker = SimpleNamespace(
        rank=0, vllm_config=config, model_runner=SimpleNamespace(get_model=lambda: model)
    )
    configure_qkvo(worker)
    assert model.o_proj.quant_method is original
    assert model.qkv_proj.quant_method is not original
    for m in (8, 128, 384):
        x = torch.ones((m, 8192), dtype=torch.bfloat16, device="cuda")
        out = model.qkv_proj.quant_method.apply(model.qkv_proj, x)
        expected = (
            torch.nn.functional.linear(x, model.qkv_proj.weight)
            if m == 8
            else original.apply(model.qkv_proj, x)
        )
        assert torch.equal(out, expected)
    assert calls == [8]
