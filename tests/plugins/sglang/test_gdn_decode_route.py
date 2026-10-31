# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the GDN route tells the decode ops, and when decode falls back to sglang."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import aklr.plugins.sglang.models.qwen3_5.api as adapter
from aklr.core.contract.fastkey import FastKeyCodec
from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.dispatch import (
    DispatcherRegistry,
    FallbackReason,
    KernelDispatcher,
)
from aklr.plugins.sglang.hook import _around
from aklr.plugins.sglang.models.qwen3_5.contract import (
    GDN_DECODE,
    GDN_DECODE_GROUP_FP8_QUANT,
)


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

    def __call__(self, value):
        self.calls.append("output projection")
        self.received = value
        return "output", None


ROWS = 8


@pytest.fixture
def case(monkeypatch):
    calls, served = [], {GDN_DECODE_GROUP_FP8_QUANT.contract_id}
    monkeypatch.setattr(
        KernelDispatcher,
        "has_workload",
        lambda self, point: point.contract_id in served and point.axes["n_tokens"] == ROWS,
    )
    conv = torch.zeros(10, 3072, 3, dtype=torch.bfloat16)
    state = torch.zeros(10, 16, 128, 128, dtype=torch.float32)
    # SGLang pads a graph batch with -1 rows
    indices = torch.tensor([3, 1, 4, 0, 9, 2, -1, -1], dtype=torch.int32)
    metadata = SimpleNamespace(
        mamba_cache_indices=indices,
        query_start_loc=torch.arange(ROWS + 1, dtype=torch.int32),
        replayssm_write_pos=None,
        replayssm_force_flush=None,
        # prefill's tracking flags; decode tracks through the backend instead
        has_mamba_track_mask=True,
    )
    layer_cache = SimpleNamespace(
        conv=[conv],
        temporal=state,
        replayssm_d=None,
        replayssm_k=None,
        replayssm_g=None,
    )
    pool = SimpleNamespace(mamba2_layer_cache=lambda _: layer_cache)

    def track(batch, conv_states, ssm_states, cache_indices, layer_id):
        calls.append("track")
        assert conv_states is conv and ssm_states is state
        assert cache_indices is indices and layer_id == 7

    backend = SimpleNamespace(
        forward_metadata=metadata,
        req_to_token_pool=pool,
        _track_mamba_state_decode=track,
        # stale prefill metadata a graph replay does not reset
        mis_metadata=object(),
    )
    batch = SimpleNamespace(
        forward_mode="decode",
        batch_size=ROWS,
        attn_backend=SimpleNamespace(linear_attn_backend=backend),
    )
    qkvz = torch.zeros(ROWS, 5120, dtype=torch.bfloat16)
    ba = torch.zeros(ROWS, 32, dtype=torch.bfloat16)

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
            layer_id=7,
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
    hidden = torch.zeros(ROWS, 4096, dtype=torch.bfloat16)

    def stock(*_):
        calls.append("stock")
        return "stock result"

    def check_call(call):
        assert call["conv_state"] is conv and call["delta_state"] is state
        assert call["cache_indices"] is indices
        assert call["projected_qkvz"] is qkvz and call["projected_ba"] is ba
        assert call["conv_weight"] is module.attn.conv_weights
        assert call["a_log"] is module.A_log and call["dt_bias"] is module.dt_bias
        assert call["norm_weight"] is module.norm.weight
        assert call["scale"] == 128**-0.5 and call["eps"] == 1e-6
        assert set(call) == {p.name for p in GDN_DECODE.api_params} - {"normalized"}

    def fp8_kernel(**call):
        calls.append("fp8 kernel")
        check_call(call)
        conv[3].fill_(7)
        state[3].fill_(9)
        return (
            torch.empty(ROWS, 2048, dtype=torch.float8_e4m3fn),
            torch.empty(ROWS, 16, dtype=torch.float32),
        )

    def bf16_kernel(**call):
        calls.append("bf16 kernel")
        check_call(call)
        conv[3].fill_(7)
        state[3].fill_(9)
        return torch.zeros(ROWS, 16, 128, dtype=torch.bfloat16)

    monkeypatch.setattr(adapter, "_hardware_supported", lambda _: True)
    monkeypatch.setattr(
        adapter,
        "_dependencies",
        lambda: SimpleNamespace(fp8_method=Fp8Method, consumer=consumer, extend="extend", decode="decode"),
    )
    monkeypatch.setattr(adapter, "gdn_decode_group_fp8_quant", fp8_kernel)
    monkeypatch.setattr(adapter, "gdn_decode", bf16_kernel)
    monkeypatch.setattr(
        adapter,
        "gdn_prefill_group_fp8_quant",
        lambda **_: pytest.fail("decode reached the prefill op"),
    )
    return SimpleNamespace(
        module=module,
        batch=batch,
        hidden=hidden,
        stock=stock,
        calls=calls,
        served=served,
        conv=conv,
        state=state,
        indices=indices,
        metadata=metadata,
        layer_cache=layer_cache,
        backend=backend,
        qkvz=qkvz,
    )


def run(case):
    return _around(adapter._gdn_forward)(case.stock, case.module, case.hidden, case.batch)


def untouched(case):
    return case.conv.eq(0).all() and case.state.eq(0).all()


def test_fp8_op_hands_out_proj_its_quantized_pair(case):
    assert run(case) == "output"
    assert case.calls == [
        "input projection",
        "fp8 kernel",
        "track",
        "output projection",
    ]
    values, scales = case.module.out_proj.received
    assert values.dtype == torch.float8_e4m3fn and values.shape == (ROWS, 2048)
    assert scales.dtype == torch.float32 and scales.shape == (ROWS, 16)
    assert case.conv[3].eq(7).all() and case.state[3].eq(9).all()
    assert case.indices.tolist()[-2:] == [-1, -1]


def test_bf16_op_serves_where_the_fp8_op_has_no_kernel(case):
    case.served.clear()
    case.served.add(GDN_DECODE.contract_id)
    assert run(case) == "output"
    assert case.calls == [
        "input projection",
        "bf16 kernel",
        "track",
        "output projection",
    ]
    received = case.module.out_proj.received
    assert received.dtype == torch.bfloat16 and received.shape == (ROWS, 2048)


def test_fp8_fallback_tries_the_bf16_op_before_any_state_change(case, monkeypatch):
    case.served.add(GDN_DECODE.contract_id)
    monkeypatch.setattr(
        adapter,
        "gdn_decode_group_fp8_quant",
        lambda **_: FallbackReason.NOT_WARMED,
    )
    assert run(case) == "output"
    assert case.calls == [
        "input projection",
        "bf16 kernel",
        "track",
        "output projection",
    ]


def test_both_ops_falling_back_run_sglang_without_tracking(case, monkeypatch):
    monkeypatch.setattr(
        adapter,
        "gdn_decode_group_fp8_quant",
        lambda **_: FallbackReason.NOT_MEASURED,
    )
    assert run(case) == "stock result"
    assert case.calls == ["input projection", "stock"]
    assert untouched(case)


def test_unserved_rows_skip_the_projections(case):
    case.batch.batch_size = 4
    case.hidden = case.hidden[:4]
    case.metadata.mamba_cache_indices = case.indices[:4]
    assert run(case) == "stock result"
    assert case.calls == ["stock"]
    assert untouched(case)


def test_graph_capture_still_reaches_the_op(case, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert run(case) == "output"
    assert "fp8 kernel" in case.calls


def test_dispatch_does_not_read_device_scalars(case, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("device scalar read in adapter"))
    monkeypatch.setattr(torch.Tensor, "tolist", lambda *_: pytest.fail("device list read in adapter"))
    assert run(case) == "output"


def _replace_pool(case, **fields):
    cache = SimpleNamespace(**{**vars(case.layer_cache), **fields})
    case.backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda _: cache)
    return cache


@pytest.mark.parametrize(
    "invalid",
    [
        "verify",
        "idle",
        "replayssm_cursor",
        "replayssm_flush",
        "replayssm_ring",
        "no_tracker",
        "no_metadata",
        "indices_dtype",
        "indices_rows",
        "conv_layout",
        "state_dtype",
        "pool_rows",
        "two_conv_states",
        "tp",
        "heads",
        "conv_bias",
        "norm",
        "consumer",
        "lora",
        "weights",
    ],
)
def test_fallback_precedes_projection_and_state_changes(case, invalid):
    if invalid in ("verify", "idle"):
        case.batch.forward_mode = invalid
    elif invalid == "replayssm_cursor":
        case.metadata.replayssm_write_pos = torch.zeros(ROWS, dtype=torch.int32)
    elif invalid == "replayssm_flush":
        case.metadata.replayssm_force_flush = torch.zeros(ROWS, dtype=torch.int32)
    elif invalid == "replayssm_ring":
        _replace_pool(case, replayssm_d=torch.zeros(1))
    elif invalid == "no_tracker":
        del case.backend._track_mamba_state_decode
    elif invalid == "no_metadata":
        case.backend.forward_metadata = None
    elif invalid == "indices_dtype":
        case.metadata.mamba_cache_indices = case.indices.long()
    elif invalid == "indices_rows":
        case.metadata.mamba_cache_indices = case.indices[:-1]
    elif invalid == "conv_layout":
        _replace_pool(case, conv=[case.conv.transpose(1, 2).contiguous().mT])
    elif invalid == "state_dtype":
        _replace_pool(case, temporal=case.state.bfloat16())
    elif invalid == "pool_rows":
        _replace_pool(case, temporal=case.state[:9])
    elif invalid == "two_conv_states":
        _replace_pool(case, conv=[case.conv, case.conv])
    elif invalid == "tp":
        case.module.attn_tp_size = 8
    elif invalid == "heads":
        case.module.attn.num_v_heads = 8
    elif invalid == "conv_bias":
        case.module.attn.bias = torch.zeros(3072, dtype=torch.bfloat16)
    elif invalid == "norm":
        case.module.norm.eps = 1e-5
    elif invalid == "consumer":
        case.module.out_proj.quant_method.w8a8_block_fp8_linear = object()
    elif invalid == "lora":
        case.module.out_proj.set_lora = lambda: None
    elif invalid == "weights":
        case.module.in_proj_ba.weight = torch.empty(31, 4096)
    assert run(case) == "stock result"
    assert case.calls == ["stock"]
    assert untouched(case)


def test_extend_keeps_the_prefill_path(case, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    case.batch.forward_mode = "extend"
    assert run(case) == "stock result"  # the decode metadata is no prefill batch
    assert case.calls == ["stock"]


def test_unsupported_hardware_does_not_import_amd_dependencies(case, monkeypatch):
    monkeypatch.setattr(adapter, "_hardware_supported", lambda _: False)
    monkeypatch.setattr(
        adapter,
        "_dependencies",
        lambda: pytest.fail("AMD dependency on unsupported hardware"),
    )
    assert run(case) == "stock result"
    assert case.calls == ["stock"]


def test_bad_projection_layout_falls_back_without_state_update(case):
    case.module._forward_input_proj = lambda _: (case.qkvz[:, :-1], case.qkvz[:, :32])
    assert run(case) == "stock result"
    assert case.calls == ["stock"]
    assert untouched(case)


def test_selected_kernel_error_never_retries_native(case, monkeypatch):
    def fail(**call):
        case.calls.append("fp8 kernel")
        call["conv_state"][3].fill_(7)
        raise RuntimeError("selected kernel failed")

    monkeypatch.setattr(adapter, "gdn_decode_group_fp8_quant", fail)
    with pytest.raises(RuntimeError, match="selected kernel failed"):
        run(case)
    assert case.calls == ["input projection", "fp8 kernel"]


def test_export_reaches_both_ops_without_a_kernel(case, monkeypatch):
    case.served.clear()
    monkeypatch.setattr(DispatcherRegistry.default(), "recorder", object())
    reached = []

    def record(name):
        def op(**_):
            reached.append(name)
            return FallbackReason.NO_WORKLOAD

        return op

    monkeypatch.setattr(adapter, "gdn_decode_group_fp8_quant", record("fp8"))
    monkeypatch.setattr(adapter, "gdn_decode", record("bf16"))
    assert run(case) == "stock result"
    assert reached == ["fp8", "bf16"]
    assert case.calls == ["input projection", "stock"]


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


@pytest.mark.parametrize("contract", [GDN_DECODE_GROUP_FP8_QUANT, GDN_DECODE])
def test_route_points_key_the_committed_schemas_at_any_pool_size(contract):
    """The preflight's Point names each committed workload, whatever SGLang's pool."""
    registry = ContractRegistry.default()
    codec = FastKeyCodec.from_contract(contract)
    name = contract.contract_id.split("@")[0]
    root = Path(adapter.__file__).parent / "impls/triton"
    runnables, rows = {}, []
    for path in sorted(root.glob(f"{name}_n_v_heads*/*.json")):
        schema = registry.parse_schema(path.read_bytes(), str(path))
        (workload,) = schema.workloads
        point = schema.point_with_contract_id(workload.axes, contract.contract_id)
        runnables[codec.match_key(codec.to_fast_key(point))] = object()
        rows.append(workload.axes["n_tokens"])
    dispatcher = KernelDispatcher(contract)
    dispatcher.table = SimpleNamespace(match_key=codec.match_key, runnables=runnables)
    for n in rows:
        assert dispatcher.has_workload(adapter._decode_point(contract, n, 1024))
        # past the pool bound that the kernels' 32-bit offsets assume
        assert not dispatcher.has_workload(adapter._decode_point(contract, n, 2048))
    assert not dispatcher.has_workload(adapter._decode_point(contract, 3, 1024))
