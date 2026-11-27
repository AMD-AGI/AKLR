import pytest

pytest.importorskip("flydsl.expr")


def test_small_moe_artifact_loads_on_the_provider_api(builder_registry, small_moe_schema):
    schema = small_moe_schema
    builder_registry.build(schema)


@pytest.mark.parametrize("rows", [1, 2])
def test_small_tokens_launch_without_aiter_sort_configuration(
    builder_registry, monkeypatch, rows, small_moe_schema
):
    import sys
    from types import ModuleType, SimpleNamespace

    import torch

    modules = {}
    for name in ("aiter", "aiter.fused_moe", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.moe_kernels"):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    aiter = modules["aiter"]
    aiter.fused_moe = modules["aiter.fused_moe"]
    aiter.ActivationType = SimpleNamespace(Silu=1)
    aiter.QuantType = SimpleNamespace(per_1x32=1)
    aiter.dtypes = SimpleNamespace(fp4x2=torch.uint8, bf16=torch.bfloat16)

    def unrelated_config(*args):
        raise AssertionError("small-token execution asked AITER for an unrelated sort configuration")

    aiter.fused_moe.get_inter_dim = unrelated_config
    launched = []
    modules["aiter.ops.flydsl.moe_kernels"]._run_compiled = lambda kernel, call: launched.append(call)
    streams = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: streams.append(device) or object())
    schema = small_moe_schema
    runnable = builder_registry.build(schema)
    hidden = torch.empty(rows, 6144, dtype=torch.bfloat16)
    w1 = torch.empty(257, 1024, 3072, dtype=torch.uint8)
    w2 = torch.empty(257, 6144, 256, dtype=torch.uint8)
    w1.is_shuffled = w2.is_shuffled = True
    runnable(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_ids=torch.zeros(rows, 9, dtype=torch.int32),
        topk_weights=torch.ones(rows, 9),
        w1_scale=torch.empty(257, 1024, 192, dtype=torch.uint8),
        w2_scale=torch.empty(257, 6144, 16, dtype=torch.uint8),
        activation="silu",
        quant_type="per_1x32",
        doweight_stage1=False,
        gate_mode="separated",
        act_quant=4,
        expert_mask=None,
        beta=None,
        linear_beta=None,
        out_dtype=torch.bfloat16,
        out=torch.empty_like(hidden),
    )
    assert len(launched) == 2
    assert streams == [torch.device("cpu")]
