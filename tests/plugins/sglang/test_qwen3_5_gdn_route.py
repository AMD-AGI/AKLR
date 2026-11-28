# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Qwen3.5 GDN route's choice of op, on stand-ins of sglang 0.5.22: a block-FP8 layer (the
FP8 checkpoint) decodes on the FP8 op first and prefills on the FP8 op, an unquantized one (the
MXFP4 checkpoint) decodes and prefills on the BF16 ops alone, on contiguous projections. A draft
chain's verify takes the op whose output out_proj takes: per-head FP8 on block FP8, per-row FP8
or BF16 rows on Quark's per-token FP8 (the MXFP4 V2 checkpoint), on its sliced projections."""

import enum
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.qwen3_5 import api

ROWS = 4


class ForwardMode(enum.Enum):
    DECODE = 1
    EXTEND = 2
    TARGET_VERIFY = 3


class QuarkW8A8Fp8:
    per_token, weight_qscheme = True, "per_channel"


class TritonGDNKernel:
    pass


class Fp8LinearMethod:
    block_quant, use_mxfp8, block_fp8_as_mxfp8 = True, False, False

    def __init__(self):
        self.w8a8_block_fp8_linear = aiter_w8a8_block_fp8_linear


class UnquantizedLinearMethod:
    pass


def aiter_w8a8_block_fp8_linear(*args, **kwargs):
    raise AssertionError("out_proj's GEMM is a stand-in")


class Op:
    """An op's api: takes every call, records what it was handed."""

    def __init__(self, result, ask=None):
        self.result, self.ask, self.asked, self.calls = result, ask, 0, []

    def fallback_reason(self, **kwargs):
        self.asked += 1
        return self.ask

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class Linear:
    def __init__(self, method):
        self.quant_method, self.weight, self.inputs = method, torch.empty(4096, 2048), []
        if isinstance(method, QuarkW8A8Fp8):
            self.scheme = method

    def __call__(self, x):
        self.inputs.append(x)
        return "out_proj output", None


class Layer:
    """A GDN layer at TP4 whose projections are one packed GEMM's column slices, as sglang's
    BF16 in_proj hands them for decode batches of up to 64 rows."""

    def __init__(self, methods):
        self.in_proj_qkvz, self.in_proj_ba, self.out_proj = (Linear(m) for m in methods)
        self.attn = SimpleNamespace(
            activation="silu", layer_id=3, head_k_dim=128, conv_weights=torch.zeros(3072, 4)
        )
        self.norm = SimpleNamespace(activation="swish", eps=1e-6, weight=torch.zeros(128))
        self.A_log, self.dt_bias = torch.zeros(16), torch.zeros(16)
        packed = torch.randn(ROWS, 5120 + 32).to(torch.bfloat16)
        self.projections = packed[:, :5120], packed[:, 5120:]
        # sglang packs the input projections of unquantized and Quark per-token FP8 layers
        self._fused_in_proj_weight = None if isinstance(methods[0], Fp8LinearMethod) else packed

    def _forward_input_proj(self, hidden_states):
        return self.projections


@pytest.fixture
def sglang(monkeypatch):
    """sglang 0.5.22's modules the route imports, as stand-ins; returns its attention backend."""
    window = torch.zeros(1, 3, 3072, 6)  # one GDN layer, global layer 3
    cache = SimpleNamespace(
        conv=[torch.zeros(9, 3072, 3)],
        temporal=torch.zeros(9, 16, 128, 128),
        replayssm_d=None,
        replayssm_k=None,
        replayssm_g=None,
        replayssm_rawv=None,
        intermediate_ssm=torch.zeros(3, 4, 16, 128, 128),
        # sglang's overlapping per-step view of the deduplicated window
        intermediate_conv_window=[window[0].as_strided((3, 4, 3072, 3), (3072 * 6, 1, 6, 1))],
    )
    metadata = SimpleNamespace(
        mamba_cache_indices=torch.zeros(ROWS, dtype=torch.int32),
        replayssm_write_pos=None,
        replayssm_force_flush=None,
        has_mamba_track_mask=False,
        query_start_loc=torch.tensor([0, ROWS // 2, ROWS], dtype=torch.int32),
        retrieve_parent_token=None,
    )
    backend = SimpleNamespace(
        forward_metadata=metadata,
        req_to_token_pool=SimpleNamespace(
            mamba2_layer_cache=lambda layer_id: cache,
            # the layer's index among the GDN layers, which the pools are indexed by
            mamba2_layer_index=lambda layer_id: layer_id - 3,
            mamba_pool=SimpleNamespace(_intermediate_conv_window_phys=[window]),
        ),
        mis_metadata=None,
        tracked=[],
        kernel_dispatcher=SimpleNamespace(verify_kernel=TritonGDNKernel()),
        verify_intermediate_state_indices=torch.arange(3, dtype=torch.int32),
        smallm=True,
    )
    backend._track_mamba_state_decode = lambda *args: backend.tracked.append(args)
    modules = {
        "sglang.srt.model_executor.forward_batch_info": {"ForwardMode": ForwardMode},
        "sglang.srt.model_executor.forward_context": {
            "get_attn_backend": lambda: SimpleNamespace(linear_attn_backend=backend)
        },
        "sglang.srt.runtime_context": {"get_lora": lambda: SimpleNamespace(enable_lora=False)},
        "sglang.srt.layers.quantization.fp8": {"Fp8LinearMethod": Fp8LinearMethod},
        "sglang.srt.layers.quantization.fp8_utils": {
            "aiter_w8a8_block_fp8_linear": aiter_w8a8_block_fp8_linear,
            "use_aiter_triton_gemm_w8a8_tuned_gfx950": lambda n, k: False,
        },
        "sglang.srt.layers.quantization.unquant": {"UnquantizedLinearMethod": UnquantizedLinearMethod},
        "sglang.srt.layers.attention.linear.utils": {"pp_spec_stable_rows_enabled": lambda: False},
        # whether sglang hands this out_proj a per-row FP8 pair (its small-M FP8 GEMM)
        "sglang.srt.models.qwen3_5": {
            "_fp8_tuple_input": lambda linear, rows: (
                isinstance(linear.quant_method, QuarkW8A8Fp8) and backend.smallm
            )
        },
    }
    for name, attrs in modules.items():
        stand_in = ModuleType(name)
        stand_in.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stand_in)
    return backend


@pytest.fixture
def ops(monkeypatch):
    """The two decode ops, each taking every call."""
    fp8 = Op((torch.zeros(ROWS, 2048), torch.zeros(ROWS, 16)))
    bf16 = Op(torch.zeros(ROWS, 16, 128, dtype=torch.bfloat16))
    monkeypatch.setattr(api, "gdn_decode_group_fp8_quant", fp8)
    monkeypatch.setattr(api, "gdn_decode", bf16)
    return fp8, bf16


@pytest.fixture
def prefill_ops(monkeypatch):
    """The two prefill ops, each taking every call; no graph capture (as on a CPU-only torch)."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fp8 = Op((torch.zeros(ROWS, 16, 128), torch.zeros(ROWS, 2048), torch.zeros(ROWS, 16)))
    bf16 = Op(torch.zeros(ROWS, 16, 128, dtype=torch.bfloat16))
    monkeypatch.setattr(api, "gdn_prefill_group_fp8_quant", fp8)
    monkeypatch.setattr(api, "gdn_prefill", bf16)
    return fp8, bf16


@pytest.fixture
def verify_ops(monkeypatch):
    """The three verify ops, each taking every call."""
    group = Op((torch.zeros(ROWS, 2048), torch.zeros(ROWS, 16)))
    token = Op((torch.zeros(ROWS, 2048), torch.zeros(ROWS, 1)))
    bf16 = Op(torch.zeros(ROWS, 16, 128, dtype=torch.bfloat16))
    monkeypatch.setattr(api, "gdn_verify_group_fp8_quant", group)
    monkeypatch.setattr(api, "gdn_verify_token_fp8_quant", token)
    monkeypatch.setattr(api, "gdn_verify", bf16)
    return group, token, bf16


def forward(layer, mode=ForwardMode.DECODE, draft_tokens=4):
    # a prefill of two fresh sequences of ROWS // 2 tokens; a verify of one request's draft chain
    batch = SimpleNamespace(
        forward_mode=mode,
        extend_seq_lens_cpu=[ROWS // 2] * 2,
        batch_size=2,
        extend_prefix_lens=torch.zeros(2, dtype=torch.int32),
        spec_info=SimpleNamespace(draft_token_num=draft_tokens, ragged_verify_layout=None),
    )
    return api._gdn_forward(layer, torch.zeros(ROWS, 4096), batch)


@pytest.mark.parametrize("fp8_declines", [False, True], ids=["fp8_op", "bf16_op"])
def test_a_block_fp8_layer_decodes_on_the_fp8_op_first(sglang, ops, fp8_declines):
    """The FP8 checkpoint's layer: the FP8 op, else the BF16 op, both handed sglang's
    projections as they are."""
    fp8, bf16 = ops
    fp8.ask = FallbackReason.NO_WORKLOAD if fp8_declines else None
    layer = Layer((Fp8LinearMethod(), UnquantizedLinearMethod(), Fp8LinearMethod()))
    assert forward(layer) == "out_proj output"
    served = bf16 if fp8_declines else fp8
    (call,) = served.calls
    assert call["projected_qkvz"] is layer.projections[0] and call["projected_ba"] is layer.projections[1]
    assert (fp8.asked, bf16.asked) == ((1, 1) if fp8_declines else (1, 0))
    (projected,) = layer.out_proj.inputs
    if fp8_declines:
        assert projected.shape == (ROWS, 2048)
    else:
        assert projected is fp8.result
    assert len(sglang.tracked) == 1


def test_an_unquantized_layer_decodes_on_the_bf16_op_alone(sglang, ops):
    """The MXFP4 checkpoint's layer: the BF16 op only, handed contiguous copies of the packed
    projections; out_proj takes its BF16 output."""
    fp8, bf16 = ops
    layer = Layer((UnquantizedLinearMethod(),) * 3)
    assert forward(layer) == "out_proj output"
    assert fp8.asked == 0 and not fp8.calls and bf16.asked == 1
    (call,) = bf16.calls
    for name, projection in zip(("projected_qkvz", "projected_ba"), layer.projections, strict=True):
        assert call[name].is_contiguous() and not projection.is_contiguous()
        assert torch.equal(call[name], projection)
    (projected,) = layer.out_proj.inputs
    assert projected.shape == (ROWS, 2048) and projected.dtype == torch.bfloat16
    assert len(sglang.tracked) == 1


def test_a_block_fp8_layer_prefills_on_the_fp8_op(sglang, prefill_ops):
    """The FP8 checkpoint's layer: the FP8 prefill op at the batch's longest sequence, out_proj
    taking its values and scales."""
    fp8, bf16 = prefill_ops
    layer = Layer((Fp8LinearMethod(), UnquantizedLinearMethod(), Fp8LinearMethod()))
    assert forward(layer, ForwardMode.EXTEND) == "out_proj output"
    assert bf16.asked == 0 and not bf16.calls
    (call,) = fp8.calls
    assert call["max_seq_len"] == ROWS // 2 and call["projected_qkvz"] is layer.projections[0]
    (projected,) = layer.out_proj.inputs
    assert projected == fp8.result[1:]


def test_an_unquantized_layer_prefills_on_the_bf16_op(sglang, prefill_ops):
    """The MXFP4 checkpoint's layer: the BF16 prefill op alone, handed contiguous copies of the
    projections; out_proj takes its BF16 rows."""
    fp8, bf16 = prefill_ops
    layer = Layer((UnquantizedLinearMethod(),) * 3)
    assert forward(layer, ForwardMode.EXTEND) == "out_proj output"
    assert fp8.asked == 0 and not fp8.calls and bf16.asked == 1
    (call,) = bf16.calls
    assert call["max_seq_len"] == ROWS // 2
    for name, projection in zip(("projected_qkvz", "projected_ba"), layer.projections, strict=True):
        assert call[name].is_contiguous() and torch.equal(call[name], projection)
    (projected,) = layer.out_proj.inputs
    assert projected.shape == (ROWS, 2048) and projected.dtype == torch.bfloat16


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.EXTEND], ids=["decode", "prefill"])
@pytest.mark.parametrize(
    "methods",
    [
        (UnquantizedLinearMethod(), UnquantizedLinearMethod(), object()),
        (object(), UnquantizedLinearMethod(), UnquantizedLinearMethod()),
    ],
    ids=["other_out_proj", "other_in_proj"],
)
def test_other_layers_stay_on_sglang(sglang, ops, prefill_ops, methods, mode):
    """Projections quantized other than block FP8 on out_proj: sglang's own path, before any
    op or projection runs."""
    assert forward(Layer(methods), mode) is FallbackReason.OUT_OF_CONTRACT
    assert all(op.asked == 0 and not op.calls for op in (*ops, *prefill_ops))


@pytest.mark.parametrize(
    ("methods", "smallm", "served", "storage"),
    [
        ((Fp8LinearMethod(), UnquantizedLinearMethod(), Fp8LinearMethod()), True, 0, "row_major"),
        ((QuarkW8A8Fp8(),) * 3, True, 1, "strided_rows"),
        ((QuarkW8A8Fp8(),) * 3, False, 2, "strided_rows"),
        ((UnquantizedLinearMethod(),) * 3, True, 2, "strided_rows"),
    ],
    ids=["block_fp8", "quark_per_token_small_m", "quark_per_token", "unquantized"],
)
def test_a_draft_chain_verifies_on_the_op_out_proj_takes(
    sglang, verify_ops, methods, smallm, served, storage
):
    """The verify op whose output out_proj takes, handed sglang's projections as they are, the
    request's slot and buffer row, the layer's intermediate buffers (the conv window as its
    deduplicated buffer) and how the projections are stored."""
    sglang.smallm = smallm
    layer = Layer(methods)
    assert forward(layer, ForwardMode.TARGET_VERIFY) == "out_proj output"
    op = verify_ops[served]
    assert [o.asked for o in verify_ops] == [int(o is op) for o in verify_ops]
    (call,) = op.calls
    assert call["projected_qkvz"] is layer.projections[0] and call["projected_ba"] is layer.projections[1]
    assert call["projection_storage"] == storage and call["draft_tokens"] == 4
    cache = sglang.req_to_token_pool.mamba2_layer_cache(0)
    assert call["inter_ssm"] is cache.intermediate_ssm
    (window,) = sglang.req_to_token_pool.mamba_pool._intermediate_conv_window_phys
    assert call["inter_conv"].data_ptr() == window.data_ptr() and call["inter_conv"].shape == (3, 3072, 6)
    assert call["cache_indices"].shape == call["inter_indices"].shape == (1,)
    (projected,) = layer.out_proj.inputs
    if served == 2:
        assert projected.shape == (ROWS, 2048) and projected.dtype == torch.bfloat16
    else:
        assert projected is op.result
    assert not sglang.tracked


def test_a_draft_chain_hands_bf16_rows_to_a_gemm_that_quantizes_them(sglang, verify_ops, monkeypatch):
    """sglang 0.5.23's block-FP8 GEMM on gfx950 quantizes a small M's BF16 rows itself (its
    fused small-M GEMM): out_proj takes the BF16 op's rows there, the FP8 op's above."""
    gemm = ModuleType("sglang.kernels.ops.gemm.smallm_fp8_blockscale_gfx950")
    gemm.SHAPES = {(4096, 2048): ((ROWS, 1, 4, 8),)}
    monkeypatch.setitem(sys.modules, gemm.__name__, gemm)
    monkeypatch.setattr(
        sys.modules["sglang.srt.layers.quantization.fp8_utils"],
        "_use_aiter_bpreshuffle_gfx95",
        True,
        raising=False,
    )
    layer = Layer((Fp8LinearMethod(), UnquantizedLinearMethod(), Fp8LinearMethod()))
    layer.out_proj.weight.is_shuffled = True
    layer.out_proj.bias = None
    assert forward(layer, ForwardMode.TARGET_VERIFY) == "out_proj output"
    group, _, bf16 = verify_ops
    assert (group.asked, bf16.asked) == (0, 1)
    (projected,) = layer.out_proj.inputs
    assert projected.shape == (ROWS, 2048) and projected.dtype == torch.bfloat16
    gemm.SHAPES = {(4096, 2048): ((ROWS - 1, 1, 4, 8),)}
    forward(layer, ForwardMode.TARGET_VERIFY)
    assert (group.asked, bf16.asked) == (1, 1)


@pytest.mark.parametrize(
    "change",
    ["draft_tokens", "tree", "kernel", "replayssm", "dense_window", "tree_window", "ptpc", "out_proj"],
)
def test_other_verifies_stay_on_sglang(sglang, verify_ops, change):
    """Other draft shapes or verify kernels, ReplaySSM, a dense conv window (top-k drafts too), a
    block-FP8 out_proj with a per-token copy, or another out_proj:
    sglang's own path, before any op or projection runs."""
    methods = (Fp8LinearMethod(), UnquantizedLinearMethod(), Fp8LinearMethod())
    cache = sglang.req_to_token_pool.mamba2_layer_cache(0)
    if change == "tree":
        sglang.forward_metadata.retrieve_parent_token = torch.zeros(1, 4, dtype=torch.int64)
    elif change == "kernel":
        sglang.kernel_dispatcher.verify_kernel = object()
    elif change == "replayssm":
        cache.replayssm_rawv = torch.zeros(1)
    elif change == "dense_window":
        cache.intermediate_conv_window = [torch.zeros(3, 4, 3072, 3)]
    elif change == "tree_window":
        # top-k drafts keep a dense window of every draft token: a 5-D buffer per GDN layer
        sglang.req_to_token_pool.mamba_pool._intermediate_conv_window_phys = [torch.zeros(1, 3, 4, 3072, 3)]
    elif change == "out_proj":
        methods = (UnquantizedLinearMethod(), UnquantizedLinearMethod(), object())
    layer = Layer(methods)
    if change == "ptpc":
        # an MXFP8 weight served as block FP8 also keeps a per-token copy for small M
        layer.out_proj.weight._ptpc_weight = torch.zeros(1)
    result = forward(layer, ForwardMode.TARGET_VERIFY, draft_tokens=2 if change == "draft_tokens" else 4)
    assert result is FallbackReason.OUT_OF_CONTRACT
    assert all(op.asked == 0 and not op.calls for op in verify_ops)
