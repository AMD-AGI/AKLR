import functools
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Event, Lock
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry

pytest.importorskip("flydsl.expr")


@pytest.mark.parametrize("initial", [None])
def test_concurrent_kernel_calls_preserve_stage2_environment(builder_registry, monkeypatch, initial):
    key = "AITER_FLYDSL_STAGE2_FP8"
    monkeypatch.delenv(key, raising=False)
    aiter = ModuleType("aiter")
    aiter.__path__ = []
    fm = ModuleType("aiter.fused_moe")
    fm.__path__ = []
    ops = ModuleType("aiter.ops")
    ops.__path__ = []
    flydsl = ModuleType("aiter.ops.flydsl")
    flydsl.__path__ = []
    kernels = ModuleType("aiter.ops.flydsl.moe_kernels")
    for module in (aiter, fm, ops, flydsl, kernels):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    aiter.fused_moe = fm
    aiter.ActivationType = Enum("ActivationType", {"Swiglu": 1})
    aiter.QuantType = Enum("QuantType", {"per_1x32": 1})
    aiter.dtypes = SimpleNamespace(fp8=torch.float8_e4m3fn, fp4x2=torch.uint8)
    fm.GateMode = SimpleNamespace(INTERLEAVE=1)
    fm.get_padded_M = lambda rows: rows
    fm.get_inter_dim = lambda w1, w2: (385, 7168, 384)
    fm.stage2_uses_route_reduce = lambda stage2: True
    kernels.get_flydsl_kernel_params = lambda name: {"mode": "reduce"}
    fm._flydsl_stage1_wrapper = lambda *args, **kwargs: None
    first_entered, second_entered, release_second = Event(), Event(), Event()
    lock = Lock()
    entered = []

    def stage2(*args, **kwargs):
        with lock:
            order = len(entered)
            entered.append(order)
        if order == 0:
            first_entered.set()
            assert second_entered.wait(10)
        else:
            second_entered.set()
            assert release_second.wait(10)

    fm._flydsl_stage2_wrapper = stage2

    @dataclass
    class Metadata:
        block_m: int = 128
        stage1: object = None
        stage2: object = None
        run_1stage: bool = False
        output_aux: bool = False
        flat: bool = False

    def config(*args, **kwargs):
        return Metadata(stage1=functools.partial(fm._flydsl_stage1_wrapper), stage2=functools.partial(stage2))

    fm.get_2stage_cfgs = config
    aiter.moe_sorting_opus_get_workspace_size = lambda *args: 0
    aiter.moe_sorting_opus_fwd = lambda *args: None

    def pipeline(*args, **kwargs):
        kwargs["_metadata_transform"](config()).stage2()

    fm.fused_moe_2stages = pipeline
    path = Path(
        "aklr/plugins/aiter/provider/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts385_topk7_inter_dim384/0b5720c82bbf38f3.json"
    )
    schema = ContractRegistry.default().parse_schema(path.read_bytes(), str(path))
    runnable = builder_registry.build(schema)
    hidden = torch.empty(4096, 7168, dtype=torch.bfloat16)
    call = dict(
        hidden_states=hidden,
        w1=torch.empty(385, 768, 3584, dtype=torch.uint8),
        w2=torch.empty(385, 7168, 192, dtype=torch.uint8),
        topk_ids=torch.zeros(4096, 7, dtype=torch.int32),
        topk_weights=torch.ones(4096, 7),
        w1_scale=torch.empty(385, 768, 224, dtype=torch.uint8),
        w2_scale=torch.empty(385, 7168, 12, dtype=torch.uint8),
        activation="swiglu",
        quant_type="per_1x32",
        doweight_stage1=False,
        gate_mode="interleave",
        act_quant=8,
        expert_mask=None,
        beta=None,
        linear_beta=None,
        swiglu_limit=10.0,
        out_dtype=torch.bfloat16,
        out=torch.empty_like(hidden),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runnable, **call)
        assert first_entered.wait(10)
        second = pool.submit(runnable, **(call | {"out": torch.empty_like(hidden)}))
        first.result(timeout=20)
        release_second.set()
        second.result(timeout=20)
    actual = os.environ.get(key)
    assert actual is initial
