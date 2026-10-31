# Created by OpenAI
from types import SimpleNamespace

import pytest
import torch

import aklr.plugins.sglang.models.qwen3_5.api as adapter
from aklr.core.runtime.dispatch import (
    DispatcherRegistry,
    FallbackReason,
    KernelDispatcher,
)
from aklr.plugins.sglang.hook import _around


class Fp8Method:
    pass


def consumer():
    pass


class Projection:
    def __init__(self, calls):
        self.calls = calls
        self.quant_method = Fp8Method()
        self.quant_method.block_quant = True
        self.quant_method.use_mxfp8 = False
        self.quant_method.quant_config = SimpleNamespace(weight_block_size=[128, 128])
        self.quant_method.w8a8_block_fp8_linear = consumer
        self.weight = torch.empty(1).expand(4096, 2048)
        self.received = None

    def __call__(self, pair):
        self.calls.append("output projection")
        self.received = pair
        return "output", None


@pytest.fixture
def case(monkeypatch):
    calls = []
    monkeypatch.setattr(KernelDispatcher, "has_workload", lambda self, point: True)
    rows = 1025
    conv = torch.zeros(3, 3072, 3, dtype=torch.bfloat16)
    state = torch.zeros(3, 16, 128, 128, dtype=torch.float32)
    metadata = SimpleNamespace(
        has_mamba_track_mask=False,
        mamba_cache_indices=torch.tensor([1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
    )
    pool = SimpleNamespace(mamba2_layer_cache=lambda _: SimpleNamespace(conv=[conv], temporal=state))
    backend = SimpleNamespace(forward_metadata=metadata, req_to_token_pool=pool)
    batch = SimpleNamespace(
        forward_mode="extend",
        batch_size=1,
        extend_seq_lens_cpu=[rows],
        extend_prefix_lens_cpu=[0],
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        attn_backend=SimpleNamespace(linear_attn_backend=backend),
    )
    qkvz = torch.zeros(rows, 5120, dtype=torch.bfloat16)
    ba = torch.zeros(rows, 32, dtype=torch.bfloat16)

    def input_proj(_):
        calls.append("input projection")
        return qkvz, ba

    module = SimpleNamespace(
        attn_tp_size=4,
        attn=SimpleNamespace(
            num_k_heads=4,
            num_v_heads=16,
            head_k_dim=128,
            head_v_dim=128,
            bias=None,
            activation="silu",
            layer_id=0,
            conv_weights=torch.zeros(3072, 4, dtype=torch.bfloat16),
        ),
        norm=SimpleNamespace(
            eps=1e-6,
            norm_before_gate=True,
            activation="swish",
            group_size=None,
            weight=torch.ones(128, dtype=torch.bfloat16),
        ),
        # Expanded placeholders carry metadata; no CPU projection is benchmarked.
        in_proj_qkvz=SimpleNamespace(weight=torch.empty(1).expand(5120, 4096)),
        in_proj_ba=SimpleNamespace(weight=torch.empty(1).expand(32, 4096)),
        A_log=torch.zeros(16, dtype=torch.float32),
        dt_bias=torch.zeros(16, dtype=torch.bfloat16),
        _forward_input_proj=input_proj,
        out_proj=Projection(calls),
    )
    hidden = torch.zeros(rows, 4096, dtype=torch.bfloat16)

    def stock(*_):
        calls.append("stock")
        return "stock result"

    def kernel(**call):
        calls.append("kernel")
        assert call["conv_state"] is conv and call["delta_state"] is state
        assert call["cache_indices"] is metadata.mamba_cache_indices
        assert call["cu_seqlens"] is metadata.query_start_loc
        assert call["max_seq_len"] == rows
        conv[1].fill_(7)
        state[1].fill_(9)
        return (
            torch.empty(rows, 16, 128, dtype=torch.bfloat16),
            torch.empty(rows, 2048, dtype=torch.float8_e4m3fn),
            torch.empty(rows, 16, dtype=torch.float32),
        )

    monkeypatch.setattr(adapter, "_hardware_supported", lambda _: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        adapter,
        "_dependencies",
        lambda: SimpleNamespace(fp8_method=Fp8Method, consumer=consumer, extend="extend", decode="decode"),
    )
    monkeypatch.setattr(adapter, "gdn_prefill_group_fp8_quant", kernel)
    return SimpleNamespace(
        module=module,
        batch=batch,
        hidden=hidden,
        stock=stock,
        calls=calls,
        conv=conv,
        state=state,
        qkvz=qkvz,
        ba=ba,
        metadata=metadata,
        kernel=kernel,
    )


def run(case):
    return _around(adapter._gdn_forward)(case.stock, case.module, case.hidden, case.batch)


def test_route_preserves_pool_identity_and_supplies_fp8_pair(case):
    assert run(case) == "output"
    assert case.calls == ["input projection", "kernel", "output projection"]
    assert case.conv[1].eq(7).all() and case.state[1].eq(9).all()
    assert case.conv[[0, 2]].eq(0).all() and case.state[[0, 2]].eq(0).all()
    values, scales = case.module.out_proj.received
    assert values.dtype == torch.float8_e4m3fn and scales.shape == (1025, 16)
    assert case.qkvz.eq(0).all() and case.ba.eq(0).all()


@pytest.mark.parametrize(
    "invalid",
    [
        "decode",
        "verify",
        "graph",
        "short",
        "length_sum",
        "prefix",
        "tp",
        "norm",
        "heads",
        "track",
        "consumer",
        "scale_block",
        "cache_api",
        "cache_dtype",
        "cache_stride",
        "indices_dtype",
        "starts_shape",
        "pool_size",
        "weights",
        "marlin",
        "lora",
        "output_weight",
    ],
)
def test_fallback_precedes_projection_and_state_changes(case, monkeypatch, invalid):
    if invalid in ("decode", "verify"):
        case.batch.forward_mode = invalid
    elif invalid == "graph":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif invalid == "short":
        case.hidden = case.hidden[:511]
        case.batch.extend_seq_lens_cpu = [511]
    elif invalid == "length_sum":
        case.batch.extend_seq_lens_cpu = [513]
    elif invalid == "prefix":
        case.batch.extend_prefix_lens_cpu = [-1]
    elif invalid == "tp":
        case.module.attn_tp_size = 8
    elif invalid == "norm":
        case.module.norm.eps = 1e-05
    elif invalid == "heads":
        case.module.attn.num_v_heads = 8
    elif invalid == "track":
        case.metadata.has_mamba_track_mask = True
    elif invalid == "consumer":
        case.module.out_proj.quant_method.w8a8_block_fp8_linear = object()
    elif invalid == "scale_block":
        case.module.out_proj.quant_method.quant_config.weight_block_size = [64, 128]
    elif invalid == "cache_api":
        case.batch.attn_backend.linear_attn_backend.req_to_token_pool = object()
    elif invalid == "cache_dtype":
        case.state = case.state.to(torch.bfloat16)
        case.batch.attn_backend.linear_attn_backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _: SimpleNamespace(conv=[case.conv], temporal=case.state)
        )
    elif invalid == "cache_stride":
        case.conv = case.conv.transpose(1, 2).contiguous().transpose(1, 2)
        case.batch.attn_backend.linear_attn_backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _: SimpleNamespace(conv=[case.conv], temporal=case.state)
        )
    elif invalid == "indices_dtype":
        case.metadata.mamba_cache_indices = case.metadata.mamba_cache_indices.long()
    elif invalid == "starts_shape":
        case.metadata.query_start_loc = case.metadata.query_start_loc[:1]
    elif invalid == "pool_size":
        case.state = case.state[:2]
        case.batch.attn_backend.linear_attn_backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _: SimpleNamespace(conv=[case.conv], temporal=case.state)
        )
    elif invalid == "weights":
        case.module.in_proj_ba.weight = torch.empty(31, 4096)
    elif invalid == "marlin":
        case.module.out_proj.quant_method.use_marlin = True
    elif invalid == "lora":
        case.module.out_proj.set_lora = lambda: None
    elif invalid == "output_weight":
        case.module.out_proj.weight = torch.empty(1).expand(4097, 2048)
    assert run(case) == "stock result"
    assert case.calls == ["stock"]
    assert case.conv.eq(0).all() and case.state.eq(0).all()


def test_dispatch_does_not_read_device_scalars(case, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("device scalar read in adapter"))
    assert run(case) == "output"


def test_unsupported_hardware_does_not_import_amd_dependencies(case, monkeypatch):
    monkeypatch.setattr(adapter, "_hardware_supported", lambda _: False)
    monkeypatch.setattr(
        adapter,
        "_dependencies",
        lambda: pytest.fail("AMD dependency on unsupported hardware"),
    )
    assert run(case) == "stock result"
    assert case.calls == ["stock"]
    assert case.conv.eq(0).all() and case.state.eq(0).all()


def test_unmeasured_call_falls_back_before_state_update(case, monkeypatch):
    monkeypatch.setattr(adapter, "gdn_prefill_group_fp8_quant", lambda **_: FallbackReason.NOT_MEASURED)
    assert run(case) == "stock result"
    assert case.calls == ["input projection", "stock"]
    assert case.conv.eq(0).all() and case.state.eq(0).all()


def test_selected_kernel_error_never_retries_native(case, monkeypatch):
    def fail(**call):
        case.calls.append("kernel")
        call["conv_state"][1].fill_(7)
        raise RuntimeError("selected kernel failed")

    monkeypatch.setattr(adapter, "gdn_prefill_group_fp8_quant", fail)
    with pytest.raises(RuntimeError, match="selected kernel failed"):
        run(case)
    assert case.calls == ["input projection", "kernel"]
    assert case.conv[1].eq(7).all()


def test_bad_projection_layout_falls_back_without_state_update(case):
    case.module._forward_input_proj = lambda _: (case.qkvz[:, :-1], case.ba)
    assert run(case) == "stock result"
    assert case.calls == ["stock"]
    assert case.conv.eq(0).all() and case.state.eq(0).all()


def test_amd_fused_input_quant_tuple_reaches_native_projection(case):
    captured = []
    project = case.module._forward_input_proj

    def input_proj(value):
        captured.append(value)
        return project(value)

    case.module._forward_input_proj = input_proj
    quantized = (case.hidden, torch.empty(1), torch.empty(1))
    case.hidden = quantized
    assert run(case) == "output"
    assert captured[0] is quantized


@pytest.mark.parametrize("kind", ["checkpoints", "mis"])
def test_additional_state_side_effects_use_native(case, kind):
    if kind == "checkpoints":
        case.metadata.num_state_checkpoints = 1
    else:
        case.batch.attn_backend.linear_attn_backend.mis_metadata = object()
    assert run(case) == "stock result"
    assert case.calls == ["stock"]


def test_unmeasured_preflight_does_not_repeat_input_projections(case, monkeypatch):
    monkeypatch.setattr(KernelDispatcher, "has_workload", lambda self, point: False)
    assert run(case) == "stock result"
    assert case.calls == ["stock"]
    assert case.conv.eq(0).all() and case.state.eq(0).all()


def test_export_still_reaches_api_on_preflight_miss(case, monkeypatch):
    monkeypatch.setattr(KernelDispatcher, "has_workload", lambda self, point: False)
    monkeypatch.setattr(DispatcherRegistry.default(), "recorder", object())
    assert run(case) == "output"
    assert case.calls == ["input projection", "kernel", "output projection"]
