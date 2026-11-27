# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the mHC boundary + norm routes leave to sglang, and how a deferred boundary is
materialized when something other than the norm reads it, on stand-ins of sglang 0.5.23."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.deepseek_v4_hc_pre_norm import api

HIDDEN, HC, MIX, ROWS = 256, 4, 24, 6


class _Coefficients:
    """sglang's HcCoefficients: partials until a reader materializes them."""

    def __init__(self, part_mix, part_sq, hc_scale, hc_base, **kw):
        self.part_mix, self.part_sq = part_mix, part_sq
        self.hc_scale, self.hc_base = hc_scale, hc_base
        for name, value in kw.items():
            setattr(self, name, value)
        self.num_rows = part_sq.shape[1]
        self._pre = torch.empty(self.num_rows, HC)
        self._post = torch.empty(self.num_rows, HC)
        self._comb = torch.empty(self.num_rows, HC, HC)
        self.materialized = self.num_rows == 0

    def materialize(self):
        self.materialized = True


@pytest.fixture
def sglang(monkeypatch):
    """The stand-in module, with the launches each route may leave to sglang recorded."""
    ran = []
    module = ModuleType("sglang.kernels.ops.layernorm.mhc_boundary_hip")
    module.HcCoefficients = _Coefficients
    module._HC_BOUNDARY_BLOCK_K = 64
    module._hc_boundary_use_prefill = lambda m, post, combine, x: False

    def partials(x, residual, post_in, comb_in, pre_prev, hc_fn, residual_out, y, *, hc_mult, **_):
        ran.append("partials")
        y.fill_(1.0)
        rows, slices = residual.shape[0], residual.shape[2] // 64
        return torch.zeros(slices, rows, MIX), torch.zeros(slices, rows)

    module._hc_boundary_partials = partials
    module.hc_mix_reduce_sinkhorn_vec = lambda *a, **k: ran.append("sinkhorn")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    quant = ModuleType("sglang.kernels.ops.quantization.mxfp8_amd_gfx95")
    quant.Mxfp8Activation = lambda q, scale: SimpleNamespace(q=q, scale=scale)
    monkeypatch.setitem(sys.modules, quant.__name__, quant)
    monkeypatch.setattr(api, "_DEFERRED", None)
    return ran


def _boundary(sglang_mod_ran, x=None, pre_prev=None, hidden=HIDDEN):
    residual = torch.zeros(ROWS, HC, hidden, dtype=torch.bfloat16)
    return api._hc_boundary_fused_deferred(
        x,
        residual,
        None,
        None,
        torch.zeros(ROWS, HC) if pre_prev is None else pre_prev,
        torch.zeros(MIX, HC * hidden),
        torch.ones(3),
        torch.zeros(MIX),
        HC,
        20,
        1e-20,
        1e-6,
    )


def test_a_boundary_with_a_pending_post_stays_on_sglang(sglang):
    """The post variant keeps sglang's residual_out, which this op does not write."""
    out = api._hc_boundary_fused_deferred(
        torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16),
        torch.zeros(ROWS, HC, HIDDEN, dtype=torch.bfloat16),
        torch.zeros(ROWS, HC),
        torch.zeros(ROWS, HC, HC),
        torch.zeros(ROWS, HC),
        torch.zeros(MIX, HC * HIDDEN),
        torch.ones(3),
        torch.zeros(MIX),
        HC,
        20,
        1e-20,
        1e-6,
    )
    assert out is FallbackReason.OUT_OF_CONTRACT
    assert sglang == []


def test_a_deferred_boundary_launches_nothing_until_it_is_read(sglang):
    """The route hands back the collapse's buffer and coefficients without a launch."""
    residual_out, y, coefficients = _boundary(sglang)
    assert residual_out is None
    assert y.shape == (ROWS, HIDDEN)
    assert sglang == []
    assert not coefficients.materialized


def test_a_reader_other_than_the_norm_materializes_the_boundary_as_sglang_would(sglang):
    """pre / post / comb, or a norm sglang takes down another path, run both sglang launches."""
    _, _, coefficients = _boundary(sglang)
    coefficients.materialize()
    assert sglang == ["partials", "sinkhorn"]
    assert coefficients.materialized
    coefficients.materialize()
    assert sglang == ["partials", "sinkhorn"]


def test_a_norm_that_is_not_the_boundarys_leaves_the_call_to_sglang(sglang):
    """Another tensor's norm materializes the pending boundary, then falls back."""
    _, _, coefficients = _boundary(sglang)
    out = api._rmsnorm_with_sinkhorn(
        torch.zeros(ROWS, HIDDEN), torch.ones(HIDDEN), 1e-20, coefficients, emit_fp8=True
    )
    assert out is FallbackReason.OUT_OF_CONTRACT
    assert sglang == ["partials", "sinkhorn"]


def test_a_norm_whose_fake_quant_is_bf16_leaves_the_call_to_sglang(sglang):
    """The op writes the fp8 codes the next projection takes; sglang's bf16 fake quant is its own."""
    _, y, coefficients = _boundary(sglang)
    out = api._rmsnorm_with_sinkhorn(y, torch.ones(HIDDEN), 1e-20, coefficients, emit_fp8=False)
    assert out is FallbackReason.OUT_OF_CONTRACT
    assert sglang == ["partials", "sinkhorn"]


def test_a_norm_with_no_fake_quant_asks_the_plain_op(sglang, monkeypatch):
    """The FFN norm without the MXFP8 operand: the same pair, the op without the codes."""
    asked = []
    monkeypatch.setattr(
        api,
        "deepseek_v4_hc_pre_norm",
        lambda *a, **k: asked.append(a) or (torch.zeros(ROWS, HIDDEN),) * 5,
    )
    _, y, coefficients = _boundary(sglang)
    quant, out_norm = api._rmsnorm_with_sinkhorn(
        y, torch.ones(HIDDEN), 1e-20, coefficients, fake_quant=False
    )
    assert quant is None
    assert out_norm.shape == (ROWS, HIDDEN)
    assert len(asked) == 1
    assert sglang == []


def test_the_boundarys_own_norm_asks_the_op_once(sglang, monkeypatch):
    """The pair becomes one op call, and nothing is left for a later materialize."""
    asked = []

    def op(*args, **kwargs):
        asked.append((args, kwargs))
        rows = args[0].shape[0]
        return (
            torch.zeros(rows, HIDDEN),
            torch.zeros(rows, HIDDEN),
            torch.zeros(rows, HC),
            torch.zeros(rows, HC),
            torch.zeros(rows, HC, HC),
            torch.zeros(rows, HIDDEN),
            torch.zeros(rows, HIDDEN // 32),
        )

    monkeypatch.setattr(api, "deepseek_v4_hc_pre_norm_mxfp8", op)
    _, y, coefficients = _boundary(sglang)
    quant, out_norm = api._rmsnorm_with_sinkhorn(
        y, torch.ones(HIDDEN), 1e-20, coefficients, emit_fp8=True
    )
    assert len(asked) == 1
    assert out_norm.shape == (ROWS, HIDDEN)
    assert quant.q.shape == (ROWS, HIDDEN)
    assert sglang == []
    coefficients.materialize()
    assert sglang == []


def test_an_op_that_falls_back_leaves_the_pair_to_sglang(sglang, monkeypatch):
    """A shape with no kernel: the boundary is materialized and sglang's norm runs."""
    monkeypatch.setattr(
        api, "deepseek_v4_hc_pre_norm_mxfp8", lambda *a, **k: FallbackReason.NOT_MEASURED
    )
    _, y, coefficients = _boundary(sglang)
    out = api._rmsnorm_with_sinkhorn(y, torch.ones(HIDDEN), 1e-20, coefficients, emit_fp8=True)
    assert out is FallbackReason.NOT_MEASURED
    assert sglang == ["partials", "sinkhorn"]
