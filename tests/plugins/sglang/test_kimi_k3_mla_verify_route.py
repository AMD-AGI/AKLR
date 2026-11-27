# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's DCP target verify on sglang 0.5.23's aiter backend: which calls of forward_absorb_rocm_core take
the route's verify branch, the query it packs for the fp8 kernel and what it hands sglang's steps, the
backend's _forward_verify_asm_cprr for that query, and the query op's reference and baseline, on stand-ins of
sglang 0.5.23."""

import enum
import functools
import inspect
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.kimi_k3 import api
from aklr.plugins.sglang.models.kimi_k3.baseline import KimiK3MlaVerifyQBaseline
from aklr.plugins.sglang.models.kimi_k3.contract import KIMI_K3_MLA_VERIFY_Q
from aklr.plugins.sglang.models.kimi_k3.reference import KimiK3MlaVerifyQReference

# two heads a rank under DCP 3: six gathered heads a token, tiled to a kernel of eight
LOCAL_HEADS, DCP = 2, 3
HEADS, KERNEL_HEADS = LOCAL_HEADS * DCP, 8
TOKENS = 4
# Kimi-K3's absorbed latent and rotated values of a query head, and the head of the query projection the
# rotated ones are sliced from
RANK, ROPE, PROJECTED = 512, 64, 192
FP8 = torch.float8_e4m3fn

P, EMPTY = inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty
# sglang 0.5.23's forward_absorb_rocm_core
# (srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py)
SGLANG_ABSORB_CORE = [
    *[(n, P, EMPTY) for n in ("self", "q_pe", "k_pe", "q_nope_out", "k_nope", "forward_batch")],
    *[(n, P, EMPTY) for n in ("zero_allocator", "positions", "topk_indices", "llama_4_scaling")],
    ("q_nope_unabsorbed", P, None),
]
# and AiterAttnBackend._forward_verify_asm_cprr (srt/layers/attention/aiter_backend.py)
SGLANG_VERIFY_ASM_CPRR = [(n, P, EMPTY) for n in ("self", "q", "layer", "k_descale", "n_rows")]
# its FORWARD_ABSORB_CORE_ATTENTION_BACKENDS, whose calls take the core's first branch
# (srt/models/deepseek_common/utils.py)
ABSORB_CORE_BACKENDS = "fa3 fa4 dsa nsa flashinfer trtllm_mla cutedsl_mla ascend intel_xpu".split()

FALLS_BACK = "aklr kimi_k3 route falls back: "


def parameters(fn):
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


def test_the_routes_take_sglangs_parameters():
    assert parameters(api._forward_absorb_rocm_core) == SGLANG_ABSORB_CORE
    assert parameters(api._forward_verify_asm_cprr) == SGLANG_VERIFY_ASM_CPRR


class ForwardMode(enum.Enum):
    DECODE = 1
    TARGET_VERIFY = 2

    def is_decode(self):
        return self is ForwardMode.DECODE

    def is_target_verify(self):
        return self is ForwardMode.TARGET_VERIFY


class Op:
    """The query op's api as the route sees it: ``run(call)`` is the call's result, the buffer the route
    hands it (a kernel took the call) or the op's FallbackReason; the calls are recorded."""

    def __init__(self):
        self.run = lambda call: call["q"]
        self.calls, self.results = [], []

    def __call__(self, **call):
        self.calls.append(call)
        self.results.append(self.run(call))
        return self.results[-1]


@functools.cache
def _fast_key():
    codec = FastKeyCodec.from_contract(KIMI_K3_MLA_VERIFY_Q)
    return codec, FastKeyFnGenerator(codec, torch.device("cpu")).generate()


def point_of(call):
    """The contract's Point of a call the route hands the op, the api's defaults filled in; None outside the
    contract."""
    codec, fast_key = _fast_key()
    key = fast_key({p.name: call.get(p.name, p.default) for p in KIMI_K3_MLA_VERIFY_Q.api_params})
    return None if key is None else codec.to_point(key)


def randn(gen, *shape):
    return torch.randn(*shape, generator=gen).to(torch.bfloat16)


def bits(t):
    return t.contiguous().view(torch.uint8)


def messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == api.__name__]


def changed(**fields):
    """A change of a call: each ``holder__field``, or field of the call itself, set to its value."""

    def change(call):
        for path, value in fields.items():
            holder, separator, field = path.partition("__")
            setattr(getattr(call, holder) if separator else call, field if separator else holder, value)

    return change


@pytest.fixture(autouse=True)
def _fresh_routes(monkeypatch):
    """A CPU torch: no graph capture. The routes log each reason once and keep each scale tensor per process:
    every test starts with none. The core's DCP branch asks for the sglang release of the backend route."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(api, "_SEEN", set())
    monkeypatch.setattr(api, "_SCALES", {})
    # the installed sglang is one whose backend takes the packed query (0.5.23)
    monkeypatch.setattr(api, "_packed_query_route_installed", lambda: True)


@pytest.fixture
def sglang(monkeypatch):
    """sglang 0.5.23's modules the routes import, as stand-ins of a rank under DCP on the aiter backend whose
    metadata takes the asm cp round-robin verify, the all-to-all exchanging the partials. Returns what a test
    turns, the calls of sglang's steps in order (``steps``) and of aiter's kernel (``kernel``)."""
    state = SimpleNamespace(steps=[], kernel=[], autotune=False)
    # the metadata's tables, each its name: the kernel's call tells them apart
    tables = (
        "qo_indptr cprr_kv_indptr cprr_kv_indices kv_last_page_len work_metadata work_indptr "
        "work_info_set reduce_indptr reduce_final_map reduce_partial_map global_kv_indptr"
    ).split()
    state.metadata = SimpleNamespace(
        use_asm_cprr_verify=True, cprr_kernel_heads=KERNEL_HEADS, max_q_len=2, **dict(zip(tables, tables))
    )
    state.backend = SimpleNamespace(forward_metadata=state.metadata)
    state.parallel = SimpleNamespace(dcp_comm_backend="a2a", dcp_group="dcp group", attn_dcp_rank=1)
    # the merged partials: this rank's heads of a token, each element its head's index plus a sixteenth of its
    # token's
    merged = torch.arange(LOCAL_HEADS).view(1, -1, 1) + torch.arange(TOKENS).view(-1, 1, 1) / 16
    merged = merged.expand(TOKENS, LOCAL_HEADS, RANK).to(torch.bfloat16)

    def step(name, result=None):
        """A step of sglang's that records its call and returns ``result``."""

        def run(*args, **kwargs):
            state.steps.append((name, args, kwargs))
            return result

        return run

    state.step = step

    def dcp_a2a_lse_reduce(
        cp_attn_out, cp_attn_lse, cp_group, is_lse_base_on_e=True, cuda_graph_buffers=None, comm_backend="a2a"
    ):
        # srt/layers/dcp/comm.py: [tokens, heads, rank] partials to this rank's [tokens, local heads, rank]
        options = {"is_lse_base_on_e": is_lse_base_on_e, "comm_backend": comm_backend}
        state.steps.append(("dcp_a2a_lse_reduce", (cp_attn_out, cp_attn_lse, cp_group), options))
        return merged.contiguous()

    def cp_lse_ag_out_rs_mla(cp_attn_out, cp_attn_lse, cp_group, ctx=None, is_lse_base_on_e=False):
        # srt/layers/dcp/comm.py: the reduce-scatter's output is head-major, [local heads, tokens, rank]
        options = {"is_lse_base_on_e": is_lse_base_on_e}
        state.steps.append(("cp_lse_ag_out_rs_mla", (cp_attn_out, cp_attn_lse, cp_group), options))
        return merged.transpose(0, 1).contiguous()

    def mla_decode_fwd(*args, **kwargs):
        # aiter's kernel: each element of the output its head's index, the lse flat
        state.kernel.append((args, kwargs))
        out = args[2]
        out.copy_(torch.arange(out.shape[1]).view(1, -1, 1).expand_as(out))
        return None, torch.arange(out.shape[0] * out.shape[1], dtype=torch.float32)

    state.aiter_backend = SimpleNamespace(fp8_dtype=FP8, mla_decode_fwd=mla_decode_fwd)
    state.forward_mla_rocm = SimpleNamespace(
        _use_aiter=True,
        _SGLANG_EXPERIMENTAL_LORA_OPTI=False,
        is_kv_b_lora_active=lambda attn: False,
        rocm_absorb_v_bmm=step("rocm_absorb_v_bmm", "absorbed"),
    )
    forward_mla = SimpleNamespace(
        is_mla_dcp_lse_base_on_e=lambda backend: backend in {"flashmla", "cutedsl_mla", "aiter"}
    )
    kv_pool = SimpleNamespace(set_mla_kv_buffer=step("set_mla_kv_buffer"))
    methods = "sglang.srt.models.deepseek_common.attention_forward_methods"
    modules = {
        "sglang.srt.layers.attention": {"aiter_backend": state.aiter_backend},
        "sglang.srt.layers.dcp": {
            "cp_lse_ag_out_rs_mla": cp_lse_ag_out_rs_mla,
            "dcp_a2a_lse_reduce": dcp_a2a_lse_reduce,
        },
        "sglang.srt.layers.logits_processor": {"get_in_autotune_dummy_run": lambda: state.autotune},
        "sglang.srt.mem_cache.memory_pool": {
            "KVWriteLoc": SimpleNamespace(for_batch=lambda forward_batch: ("write loc", forward_batch))
        },
        "sglang.srt.model_executor.forward_context": {"get_token_to_kv_pool": lambda: kv_pool},
        methods: {"forward_mla": forward_mla, "forward_mla_rocm": state.forward_mla_rocm},
        f"{methods}.forward_mha": {"resolve_attn_backend": lambda forward_batch: state.backend},
        "sglang.srt.models.deepseek_common.utils": {
            "FORWARD_ABSORB_CORE_ATTENTION_BACKENDS": ABSORB_CORE_BACKENDS
        },
        "sglang.srt.runtime_context": {"get_parallel": lambda: state.parallel},
    }
    for name, attrs in modules.items():
        stand_in = ModuleType(name)
        stand_in.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stand_in)
    return state


@pytest.fixture
def op(monkeypatch):
    """The query op, a kernel taking every call."""
    op = Op()
    monkeypatch.setattr(api, "kimi_k3_mla_verify_q", op)
    return op


@pytest.fixture
def plain_branch(monkeypatch):
    """The checks of the core's plain branch, which follow the verify branch's: recorded when reached, they
    send the call back to sglang."""
    reached = []
    monkeypatch.setattr(api, "_absorb_core_falls_back", lambda *args: reached.append(True) or True)
    return reached


@pytest.fixture
def core(sglang):
    """A target verify's call of forward_absorb_rocm_core on a DCP rank: the absorbed query's latent values
    [tokens, heads, 512] dense, its rotated ones the last columns of the query projection's heads, the new
    keys, and the attention whose DCP decode returns the backend's partials, the gathered heads of the
    kernel's output and lse. ``run`` calls the route on them as they are then."""
    gen = torch.Generator().manual_seed(5)
    partials = randn(gen, TOKENS, KERNEL_HEADS, RANK)[:, :HEADS, :]
    lse = torch.randn(TOKENS, KERNEL_HEADS, generator=gen)[:, :HEADS]
    attn = SimpleNamespace(
        dcp_size=DCP,
        num_local_heads=LOCAL_HEADS,
        kv_lora_rank=RANK,
        current_attention_backend="aiter",
        next_skip_topk=None,
        use_deep_gemm_bmm=False,
        attn_mqa="attn_mqa",
        attn_mqa_for_dcp_decode=sglang.step("attn_mqa_for_dcp_decode", (partials, lse)),
        o_proj=sglang.step("o_proj", ("o_proj output", None)),
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    # forward_mla.py's, for a model without DSA
    attn.is_dcp_mla_decode_phase = lambda forward_batch: (
        attn.dcp_size > 1
        and (forward_batch.forward_mode.is_decode() or forward_batch.forward_mode.is_target_verify())
    )
    projected = randn(gen, TOKENS, HEADS * PROJECTED).view(TOKENS, HEADS, PROJECTED)
    call = SimpleNamespace(
        sglang=sglang,
        attn=attn,
        batch=batch,
        metadata=sglang.metadata,
        rocm=sglang.forward_mla_rocm,
        aiter=sglang.aiter_backend,
        partials=partials,
        lse=lse,
        q_nope_out=randn(gen, TOKENS, HEADS, RANK),
        q_pe=projected[..., PROJECTED - ROPE :],
        k_nope=randn(gen, TOKENS, 1, RANK),
        k_pe=randn(gen, TOKENS, 1, ROPE),
        topk_indices=None,
        llama_4_scaling=None,
        q_nope_unabsorbed=None,
    )
    call.run = lambda: api._forward_absorb_rocm_core(
        call.attn,
        call.q_pe,
        call.k_pe,
        call.q_nope_out,
        call.k_nope,
        call.batch,
        "zero_allocator",
        "positions",
        call.topk_indices,
        call.llama_4_scaling,
        call.q_nope_unabsorbed,
    )
    return call


# ---- the core's verify branch


@pytest.mark.parametrize("comm_backend", ["a2a", "fi_a2a"])
def test_a_dcp_target_verify_packs_the_query_by_the_op_then_runs_sglangs_steps(
    monkeypatch, core, op, plain_branch, comm_backend
):
    """The op packs the absorbed query's rows, read where sglang holds them, into the fp8 query at the
    kernel's head count, which the attention takes as it is with no verify window; sglang's steps follow in
    its order, the partials going to the all-to-all's reduce as the backend returns them, uncopied."""
    core.sglang.parallel.dcp_comm_backend = comm_backend
    with monkeypatch.context() as patch:
        patch.setattr(torch, "cat", lambda *a, **k: pytest.fail("a bf16 query or a verify window was built"))
        assert core.run() == "o_proj output"
    assert plain_branch == []

    (call,) = op.calls
    q_nope, q_pe, q = call["q_nope"], call["q_pe"], call["q"]
    assert q_nope.data_ptr() == core.q_nope_out.data_ptr() and q_nope.stride() == (RANK, RANK, 1)
    assert torch.equal(q_nope, core.q_nope_out.reshape(TOKENS * HEADS, 1, RANK))
    assert q_pe.data_ptr() == core.q_pe.data_ptr() and q_pe.stride() == (PROJECTED, ROPE, 1)
    assert torch.equal(q_pe, core.q_pe.reshape(TOKENS * HEADS, 1, ROPE))
    assert q.shape == (TOKENS, KERNEL_HEADS, RANK + ROPE) and q.dtype == FP8
    assert call["storage"] == "strided_rows" and call.keys() == {"q_nope", "q_pe", "q", "storage"}
    point = point_of(call)
    assert point.layout_variant_name == "strided_rows"
    assert point.axes == {"n_tokens": TOKENS, "n_kernel_heads": KERNEL_HEADS, "n_rows": TOKENS * HEADS}

    write, attention, reduce, absorb, project = core.sglang.steps
    assert write == ("set_mla_kv_buffer", ("attn_mqa", ("write loc", core.batch), core.k_nope, core.k_pe), {})
    assert attention == ("attn_mqa_for_dcp_decode", (q, None, None, core.batch), {"save_kv_cache": False})
    assert attention[1][0] is op.results[0]
    name, (partials, lse, group), options = reduce
    assert name == "dcp_a2a_lse_reduce" and group == "dcp group"
    assert options == {"is_lse_base_on_e": True, "comm_backend": comm_backend}
    assert partials.shape == (TOKENS, HEADS, RANK) and partials.data_ptr() == core.partials.data_ptr()
    assert lse is core.lse and not partials.is_contiguous() and not lse.is_contiguous()
    name, (attn, merged), options = absorb
    assert name == "rocm_absorb_v_bmm" and attn is core.attn and options == {}
    assert merged.shape == (TOKENS, LOCAL_HEADS, RANK)
    assert project == ("o_proj", ("absorbed",), {})


def test_a_dcp_target_verify_on_all_gather_takes_sglangs_reduce_scatter(core, op, plain_branch):
    """--dcp-comm-backend ag_rs: sglang's cp_lse_ag_out_rs_mla on the same partials, its head-major output
    turned token-major for the value absorption."""
    core.sglang.parallel.dcp_comm_backend = "ag_rs"
    assert core.run() == "o_proj output"
    _, _, (name, (partials, lse, group), options), (_, (_, merged), _), _ = core.sglang.steps
    assert name == "cp_lse_ag_out_rs_mla" and group == "dcp group" and options == {"is_lse_base_on_e": True}
    assert partials.data_ptr() == core.partials.data_ptr() and lse is core.lse
    heads = torch.arange(LOCAL_HEADS).view(1, -1, 1) + torch.arange(TOKENS).view(-1, 1, 1) / 16
    assert torch.equal(merged.float(), heads.expand(TOKENS, LOCAL_HEADS, RANK))


def test_a_kernel_of_the_gathered_head_count_takes_the_query_untiled(core, op, plain_branch):
    """No kernel head count in the metadata: the kernel runs the gathered heads."""
    core.metadata.cprr_kernel_heads = 0
    assert core.run() == "o_proj output"
    (call,) = op.calls
    assert call["q"].shape == (TOKENS, HEADS, RANK + ROPE)
    assert point_of(call).axes == {"n_tokens": TOKENS, "n_kernel_heads": HEADS, "n_rows": TOKENS * HEADS}


LEFT_TO_SGLANG = {
    "absorb_core_backend": (changed(attn__current_attention_backend="dsa"), "attention backend"),
    "not_aiter": (changed(rocm___use_aiter=False), "not aiter"),
    "not_the_dcp_decode_phase": (
        changed(attn__is_dcp_mla_decode_phase=lambda forward_batch: False),
        "not the DCP decode phase",
    ),
    "indexer_topk": (changed(topk_indices=torch.zeros(1)), "indexer top-k"),
    "topk_for_the_next_layer": (changed(attn__next_skip_topk=False), "indexer top-k"),
    "llama_4_scaling": (changed(llama_4_scaling=torch.ones(1)), "llama 4 scaling"),
    "unabsorbed_query": (changed(q_nope_unabsorbed=torch.zeros(1)), "unabsorbed query"),
    "deep_gemm": (changed(attn__use_deep_gemm_bmm=True), "DeepGEMM absorption"),
    "lora_side_stream": (changed(rocm___SGLANG_EXPERIMENTAL_LORA_OPTI=True), "LoRA on kv_b"),
    "kv_b_lora": (changed(rocm__is_kv_b_lora_active=lambda attn: True), "LoRA on kv_b"),
    "autotune_dummy_run": (changed(sglang__autotune=True), "autotune's dummy run"),
    "gluon_verify": (changed(metadata__use_asm_cprr_verify=False), "no asm cp round-robin verify"),
    "backend_without_metadata": (
        changed(sglang__backend=SimpleNamespace()),
        "no asm cp round-robin verify",
    ),
    "other_fp8_type": (changed(aiter__fp8_dtype=torch.float8_e5m2), "fp8 type"),
    "two_reasons": (
        changed(llama_4_scaling=torch.ones(1), attn__use_deep_gemm_bmm=True),
        "llama 4 scaling, DeepGEMM absorption",
    ),
}


@pytest.mark.parametrize(("change", "reason"), LEFT_TO_SGLANG.values(), ids=list(LEFT_TO_SGLANG))
def test_a_dcp_target_verify_of_another_branch_stays_with_sglang_and_logs_why_once(
    core, op, plain_branch, caplog, change, reason
):
    change(core)
    with caplog.at_level(logging.WARNING, logger=api.__name__):
        results = [core.run() for _ in range(2)]
    assert results == [FallbackReason.OUT_OF_CONTRACT] * 2 and plain_branch == [True, True]
    assert not op.calls and core.sglang.steps == []
    assert messages(caplog) == [f"{FALLS_BACK}DCP target verify left to sglang: {reason}"]


@pytest.mark.parametrize(
    "change",
    [changed(batch__forward_mode=ForwardMode.DECODE), changed(attn__dcp_size=1)],
    ids=["dcp_decode", "verify_without_dcp"],
)
def test_other_calls_go_on_to_the_plain_branchs_checks_unlogged(core, op, plain_branch, caplog, change):
    change(core)
    with caplog.at_level(logging.WARNING, logger=api.__name__):
        assert core.run() is FallbackReason.OUT_OF_CONTRACT
    assert plain_branch == [True] and not op.calls and core.sglang.steps == []
    assert messages(caplog) == []


def test_on_another_sglang_release_the_target_verify_goes_on_to_the_plain_branchs_checks(
    monkeypatch, core, op, plain_branch, caplog
):
    """The backend route that takes the packed query is of one release: elsewhere the core packs none."""
    monkeypatch.setattr(api, "_packed_query_route_installed", lambda: False)
    with caplog.at_level(logging.WARNING, logger=api.__name__):
        assert core.run() is FallbackReason.OUT_OF_CONTRACT
    assert plain_branch == [True] and not op.calls and core.sglang.steps == []
    assert messages(caplog) == []


@pytest.mark.parametrize(
    ("installed", "packs"), [("0.5.23", True), ("0.5.22", False), ("0.5.24.dev809", False)]
)
def test_the_packed_query_is_of_the_backend_routes_releases(monkeypatch, installed, packs):
    monkeypatch.undo()  # the fixture's stand-in of the check itself
    monkeypatch.setattr(api, "sglang_version", lambda: installed)
    api._packed_query_route_installed.cache_clear()
    try:
        assert api._packed_query_route_installed() is packs
    finally:
        api._packed_query_route_installed.cache_clear()


def test_a_query_whose_heads_are_unevenly_apart_stays_with_sglang(core, op, plain_branch, caplog):
    """Latent values written head-major, a token's heads a token's width apart: no rows of one stride hold
    them. The call goes back before sglang's steps start, so that sglang writes the keys once."""
    core.q_nope_out = randn(torch.Generator().manual_seed(6), HEADS, TOKENS, RANK).transpose(0, 1)
    with caplog.at_level(logging.WARNING, logger=api.__name__):
        results = [core.run() for _ in range(2)]
    assert results == [FallbackReason.OUT_OF_CONTRACT] * 2
    assert not op.calls and core.sglang.steps == []
    assert messages(caplog) == [
        f"{FALLS_BACK}DCP target verify's query: heads unevenly apart at "
        f"q_nope ({TOKENS}, {HEADS}, {RANK}) torch.bfloat16 strides ({RANK}, {TOKENS * RANK}, 1), "
        f"q_pe ({TOKENS}, {HEADS}, {ROPE}) torch.bfloat16 strides ({HEADS * PROJECTED}, {PROJECTED}, 1), "
        f"{KERNEL_HEADS} kernel heads"
    ]


def test_a_query_no_kernel_takes_stays_with_sglang(core, op, plain_branch, caplog):
    """The op's reason goes back before sglang's steps start, logged once with the operands' shapes and
    strides."""
    op.run = lambda call: FallbackReason.NO_WORKLOAD
    with caplog.at_level(logging.WARNING, logger=api.__name__):
        results = [core.run() for _ in range(2)]
    assert results == [FallbackReason.NO_WORKLOAD] * 2 and len(op.calls) == 2 and core.sglang.steps == []
    assert messages(caplog) == [
        f"{FALLS_BACK}DCP target verify's query: no_workload at "
        f"q_nope ({TOKENS}, {HEADS}, {RANK}) torch.bfloat16 strides ({HEADS * RANK}, {RANK}, 1), "
        f"q_pe ({TOKENS}, {HEADS}, {ROPE}) torch.bfloat16 strides ({HEADS * PROJECTED}, {PROJECTED}, 1), "
        f"{KERNEL_HEADS} kernel heads"
    ]


# ---- a tensor's heads as rows


def rotated(gen):
    """A query's rotated values: the last columns of the query projection's heads."""
    return randn(gen, TOKENS, HEADS * PROJECTED).view(TOKENS, HEADS, PROJECTED)[..., PROJECTED - ROPE :]


HEADS_EVENLY_APART = {
    "dense": lambda gen: randn(gen, TOKENS, HEADS, RANK),
    "column_slice": rotated,
    "one_token": lambda gen: rotated(gen)[:1],
}
HEADS_UNEVENLY_APART = {
    "head_major": lambda gen: randn(gen, HEADS, TOKENS, ROPE).transpose(0, 1),
    "some_heads_of_a_token": lambda gen: randn(gen, TOKENS, HEADS + 2, ROPE)[:, :HEADS],
    "every_other_value": lambda gen: randn(gen, TOKENS, HEADS, 2 * ROPE)[..., ::2],
    # windows of one row, half a head apart
    "overlapping_heads": lambda gen: (
        randn(gen, (TOKENS * HEADS + 1) * ROPE // 2).unfold(0, ROPE, ROPE // 2).view(TOKENS, HEADS, ROPE)
    ),
}


@pytest.mark.parametrize("make", HEADS_EVENLY_APART.values(), ids=list(HEADS_EVENLY_APART))
def test_heads_evenly_apart_are_rows_of_one_stride(make):
    """Each head a row where it lies, uncopied, as the op's strided_rows layout takes rows: dense in their
    values, a row or more apart."""
    t = make(torch.Generator().manual_seed(7))
    tokens, heads, dim = t.shape
    rows = api._head_rows(t)
    assert rows.shape == (tokens * heads, 1, dim) and rows.data_ptr() == t.data_ptr()
    assert torch.equal(rows, t.reshape(tokens * heads, 1, dim))
    assert rows.stride()[1:] == (dim, 1) and rows.stride(0) >= dim


@pytest.mark.parametrize("make", HEADS_UNEVENLY_APART.values(), ids=list(HEADS_UNEVENLY_APART))
def test_heads_unevenly_apart_are_no_rows(make):
    assert api._head_rows(make(torch.Generator().manual_seed(7))) is None


# ---- the backend's verify on the packed query

ROWS = TOKENS


@pytest.fixture
def backend(sglang):
    """The aiter backend of a DCP rank as _forward_verify_asm_cprr reads it, with its attention layer of the
    gathered heads; the key pool holds 32 latents."""
    keys = torch.zeros(32, 1, RANK + ROPE, dtype=FP8)
    backend = SimpleNamespace(
        forward_metadata=sglang.metadata,
        input_dtype=torch.bfloat16,
        token_to_kv_pool=SimpleNamespace(get_key_buffer={5: keys}.__getitem__),
        cprr_intra_batch_mode=True,
        dcp_world_size=DCP,
        _resolve_fp8_kv_scale_float=lambda layer, k_descale: 1.0 if k_descale is None else float(k_descale),
        _kernel_num_kv_splits=lambda: 7,
    )
    layer = SimpleNamespace(
        layer_id=5, tp_q_head_num=HEADS, qk_head_dim=RANK + ROPE, v_head_dim=RANK, scaling=0.0625
    )
    return SimpleNamespace(backend=backend, layer=layer, keys=keys)


def packed(heads=KERNEL_HEADS, dtype=FP8, rows=ROWS):
    return torch.zeros(rows, heads, RANK + ROPE, dtype=dtype)


def test_the_backend_hands_a_packed_query_to_aiters_kernel_as_it_is(sglang, backend):
    """A float8_e4m3fn query at the kernel's head count: aiter's mla_decode_fwd takes it uncast and untiled,
    with the arguments sglang 0.5.23's _forward_verify_asm_cprr passes, the two scales as tensors; the
    gathered heads of the kernel's output and lse go back, views of them."""
    q = packed()
    out, lse = api._forward_verify_asm_cprr(backend.backend, q, backend.layer, 0.5, ROWS)
    ((args, kwargs),) = sglang.kernel
    query, keys, kernel_out, *tables = args
    assert query is q
    assert keys.shape == (32, 1, 1, RANK + ROPE) and keys.data_ptr() == backend.keys.data_ptr()
    assert kernel_out.shape == (ROWS, KERNEL_HEADS, RANK) and kernel_out.dtype == torch.bfloat16
    assert tables == ["qo_indptr", "cprr_kv_indptr", "cprr_kv_indices", "kv_last_page_len", 2]
    q_scale, kv_scale = kwargs.pop("q_scale"), kwargs.pop("kv_scale")
    assert (q_scale.shape, q_scale.dtype, q_scale.item()) == ((), torch.float32, 1.0)
    assert (kv_scale.shape, kv_scale.dtype, kv_scale.item()) == ((), torch.float32, 0.5)
    assert kwargs == {
        "sm_scale": 0.0625,
        "work_meta_data": "work_metadata",
        "work_indptr": "work_indptr",
        "work_info_set": "work_info_set",
        "reduce_indptr": "reduce_indptr",
        "reduce_final_map": "reduce_final_map",
        "reduce_partial_map": "reduce_partial_map",
        "intra_batch_mode": True,
        "num_kv_splits": 7,
        "return_lse": True,
        "g_kv_indptr": "global_kv_indptr",
        "cp_world_size": DCP,
        "cp_rank": 1,
        "causal": True,
    }
    assert out.shape == (ROWS, HEADS, RANK) and out.data_ptr() == kernel_out.data_ptr()
    assert torch.equal(out.float(), torch.arange(HEADS).view(1, -1, 1).expand(ROWS, HEADS, RANK).float())
    kernel_lse = torch.arange(ROWS * KERNEL_HEADS, dtype=torch.float32).view(ROWS, KERNEL_HEADS)
    assert torch.equal(lse, kernel_lse[:, :HEADS])


def test_the_backend_returns_a_native_kernels_output_whole(sglang, backend):
    """No kernel head count in the metadata: the kernel runs the gathered heads, and its output and lse go
    back unsliced."""
    sglang.metadata.cprr_kernel_heads = 0
    q = packed(HEADS)
    out, lse = api._forward_verify_asm_cprr(backend.backend, q, backend.layer, None, ROWS)
    ((args, kwargs),) = sglang.kernel
    assert args[0] is q and out is args[2] and out.shape == (ROWS, HEADS, RANK)
    assert lse.shape == (ROWS, HEADS) and kwargs["kv_scale"].item() == 1.0


OTHER_QUERIES = {
    # sglang's own: forward_absorb_rocm_core's bf16 cat at the gathered heads
    "bf16": lambda: packed(HEADS, torch.bfloat16),
    "bf16_at_the_kernels_heads": lambda: packed(KERNEL_HEADS, torch.bfloat16),
    "fp8_at_the_gathered_heads": lambda: packed(HEADS),
    "fp8_e5m2": lambda: packed(dtype=torch.float8_e5m2),
    "flat": lambda: packed().flatten(1),
    "other_rows": lambda: packed(rows=ROWS + 1),
}


@pytest.mark.parametrize("make", OTHER_QUERIES.values(), ids=list(OTHER_QUERIES))
def test_any_other_query_is_the_backends(sglang, backend, make):
    result = api._forward_verify_asm_cprr(backend.backend, make(), backend.layer, 0.5, ROWS)
    assert result is FallbackReason.OUT_OF_CONTRACT and sglang.kernel == []


def scales(sglang):
    return [(kwargs["q_scale"], kwargs["kv_scale"]) for _, kwargs in sglang.kernel]


def test_the_kernels_scales_are_made_once_outside_a_graph_capture(sglang, backend):
    for _ in range(2):
        api._forward_verify_asm_cprr(backend.backend, packed(), backend.layer, 0.5, ROWS)
    (q_scale, kv_scale), again = scales(sglang)
    assert again[0] is q_scale and again[1] is kv_scale and q_scale is not kv_scale


def test_scales_made_inside_a_graph_capture_are_not_kept(monkeypatch, sglang, backend):
    """A tensor filled inside a capture is the graph's: each captured call fills its own, and the first call
    outside a capture makes the ones that are kept, which a later capture reads."""

    def verify(capturing):
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
        api._forward_verify_asm_cprr(backend.backend, packed(), backend.layer, 0.5, ROWS)

    verify(True), verify(True), verify(False), verify(True)
    captured, captured_again, kept, captured_later = scales(sglang)
    for mine, others in ((captured, captured_again), (captured, kept), (captured_again, kept)):
        assert mine[0] is not others[0] and mine[1] is not others[1]
    assert captured_later[0] is kept[0] and captured_later[1] is kept[1]
    assert [(q.item(), kv.item()) for q, kv in scales(sglang)] == [(1.0, 0.5)] * 4


# ---- the query op's reference and baseline


def sglangs_query(q_nope_out, q_pe, kernel_heads):
    """The query as sglang 0.5.23 hands it to aiter's fp8 kernel: forward_absorb_rocm_core's cat, then
    _forward_verify_asm_cprr's cast and, at a larger kernel head count, AiterAttnBackend.pad_heads."""
    q = torch.cat([q_nope_out, q_pe], dim=-1)
    rows, heads, width = q.shape
    q = q.view(rows, -1).to(FP8).view(rows, heads, width)
    if kernel_heads != heads:
        reps = -(-kernel_heads // heads)
        q = q.repeat(1, reps, 1)[:, :kernel_heads, :].contiguous()
    return q


KERNEL_HEAD_COUNTS = pytest.mark.parametrize(
    "kernel_heads", [HEADS, KERNEL_HEADS, 2 * HEADS + 1], ids=["native", "padded", "tiled_twice"]
)


@KERNEL_HEAD_COUNTS
def test_the_reference_is_torchs_cast_of_the_cat_with_the_heads_tiled(kernel_heads):
    """On the operands' dense rows, as validation hands them over: sglang's query, bit for bit."""
    gen = torch.Generator().manual_seed(8)
    q_nope_out, q_pe = randn(gen, TOKENS, HEADS, RANK), rotated(gen)
    out = KimiK3MlaVerifyQReference().reference(
        q_nope=q_nope_out.reshape(TOKENS * HEADS, 1, RANK),
        q_pe=q_pe.reshape(TOKENS * HEADS, 1, ROPE),
        q=packed(kernel_heads),
        storage="row_major",
    )["q"]
    assert out.dtype == FP8 and out.shape == (TOKENS, kernel_heads, RANK + ROPE)
    assert torch.equal(bits(out), bits(sglangs_query(q_nope_out, q_pe, kernel_heads)))


@pytest.mark.parametrize(
    ("value", "rounded"),
    [
        (1.09375, 1.125),
        (1.0625, 1.0),
        (1.1875, 1.25),
        (1.3125, 1.25),
        (-1.0625, -1.0),
        (448.0, 448.0),
        (2.0**-10, 0.0),
        (3 * 2.0**-10, 2.0**-8),
    ],
)
def test_the_reference_rounds_to_the_nearest_fp8_value_ties_to_even(value, rounded):
    """float8_e4m3fn steps by 1/8 from 1 to 2 and by 2**-9 below 2**-6: a value midway between two of its
    values takes the one whose last bit is 0."""
    q_nope = torch.full((1, 1, RANK), value, dtype=torch.bfloat16)
    q_pe = torch.zeros(1, 1, ROPE, dtype=torch.bfloat16)
    out = KimiK3MlaVerifyQReference().reference(q_nope=q_nope, q_pe=q_pe, q=packed(1, rows=1))["q"].float()
    assert bool((out[0, 0, :RANK] == rounded).all()) and not out[0, 0, RANK:].any()


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (464.0, 0x7E),
        (-464.0, 0xFE),
        (466.0, 0x7F),
        (-466.0, 0xFF),
        (float("inf"), 0x7F),
        (float("-inf"), 0xFF),
        (float("nan"), 0x7F),
    ],
)
def test_the_reference_gives_values_past_fp8s_range_its_nan_code(value, code):
    """float8_e4m3fn ends at 448 (0x7E) and has no infinity: up to midway to the next step, 464, a value
    rounds to 448; past it, as an infinity or a NaN, it takes the NaN code under its own sign."""
    q_nope = torch.full((1, 1, RANK), value, dtype=torch.bfloat16)
    q_pe = torch.zeros(1, 1, ROPE, dtype=torch.bfloat16)
    out = KimiK3MlaVerifyQReference().reference(q_nope=q_nope, q_pe=q_pe, q=packed(1, rows=1))["q"]
    assert bool((bits(out)[0, 0, :RANK] == code).all()) and not bits(out)[0, 0, RANK:].any()


@KERNEL_HEAD_COUNTS
def test_the_baseline_packs_the_query_as_sglang_does(core, op, plain_branch, kernel_heads):
    """On the call the route hands the op, rows apart: forward_absorb_rocm_core's cat of sglang's own
    operands, the backend's cast and its pad_heads, bit for bit."""
    core.metadata.cprr_kernel_heads = 0 if kernel_heads == HEADS else kernel_heads
    core.run()
    (call,) = op.calls
    out = KimiK3MlaVerifyQBaseline().baseline(**call)["q"]
    assert out.dtype == FP8 and out.is_contiguous()
    assert torch.equal(bits(out), bits(sglangs_query(core.q_nope_out, core.q_pe, kernel_heads)))
