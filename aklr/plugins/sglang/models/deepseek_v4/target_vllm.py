# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The target's layers in vLLM's mono kernels (``kernels/vllm_mono``): the same K1 / K2 split
and KV records as the SGLang-AMD-Mono launches, reading SGLang's caches and tables. SGLang's
top-k entries are slots already, so every row reads one identity block table."""

from __future__ import annotations

import os
from typing import Any

import torch

from aklr.plugins.sglang import stats

# "sgl": the target's layers in SGLang-AMD-Mono's launches too
ENV_TARGET_MONO = "AKLR_DSV41_TARGET_MONO"
INFO = "deepseek_v4.target:mono"


def wanted() -> bool:
    return os.environ.get(ENV_TARGET_MONO, "vllm") != "sgl"


def layer_weights(L) -> Any:
    """Layer ``L``'s tensors (its SGLang-AMD-Mono weights) as vLLM's launches read
    them: the same tensors, other names."""
    from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.runner import (
        AttnWeights,
        MonoLayerWeights,
    )

    k1, k2, f = L.k1, L.k2, L.ffn
    attn = AttnWeights(
        layer_id=L.i,
        wqkv=k1.wqkv,
        wqkv_scale=k1.wqkv_s,
        q_norm=k1.q_norm,
        kv_norm=k1.kv_norm,
        wq_b=k1.wqb,
        wq_b_scale=k1.wqb_s,
        wo_a=k2.woa,
        wo_a_scale=k2.woa_s,
        wo_b=k2.wob,
        wo_b_scale=k2.wob_s,
        attn_sink=k2.sink,
        cos_sin=k1.cos_sin,
        ratio=L.ratio,
    )
    return MonoLayerWeights(
        attn=attn,
        hc_attn_fn=k1.hc_fn,
        hc_attn_scale=k1.hc_scale,
        hc_attn_base=k1.hc_base,
        attn_norm=k1.norm,
        **{n: getattr(f, n) for n in f.__struct_fields__},
    )


def _paged(cache: torch.Tensor, block: int) -> torch.Tensor:
    """A [pages, page bytes] cache as vLLM's [pages, block, 584] (the page stride kept)."""
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.runner import (
        RECORD,
        check_cache,
    )

    check_cache("vLLM mono cache", cache, block)
    return cache.as_strided((cache.shape[0], block, RECORD), (cache.stride(0), RECORD, 1))


def checked_weights(tp: int, layers: dict) -> dict:
    """Each layer's ``layer_weights``, as vLLM's launches check them."""
    from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.attention.plan import (
        Dims,
    )

    d, out = Dims(tp), {}
    for i, L in layers.items():
        w = layer_weights(L)
        w.attn.check(d)
        w.check(tp)
        out[i] = w
    return out


class VllmTarget:
    """One rank's vLLM mono runner and each target layer's weights in its names
    (``group``: the gloo group of the peer handshake; collective)."""

    def __init__(self, tp: int, rank: int, group, device, weights: dict):
        from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.runner import (
            DSV41MonoLayer,
        )

        self.weights = weights
        self.runner = DSV41MonoLayer(tp, rank, group, device)
        self.device = self.runner.device
        self.t2r = torch.zeros(self.runner.q.shape[0], dtype=torch.int32, device=self.device)
        self._table = torch.zeros(0, dtype=torch.int32, device=self.device)
        self._views: dict = {}

    def precompile(self, tokens: int, kinds) -> None:
        """This width's scratch and launches for each (ratio, fold, index) in ``kinds``."""
        from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.runner import (
            compile_only,
        )

        self.runner.scratch(tokens)
        for ratio, fold, index in sorted(kinds):
            for launch in self.runner.kernels(tokens, ratio, fold, index):
                compile_only(launch)

    def _caches(self, L):
        """Layer L's caches as vLLM's views, and the identity block table (grown to
        the largest compressed cache, at a layer's first launch: eager)."""
        v = self._views.get(L.i)
        if v is None:
            swa = _paged(L.swa, L.swa_block)
            comp = None if not L.ratio else _paged(L.comp, L.comp_block)
            if comp is not None and comp.shape[0] > self._table.numel():
                self._table = torch.arange(comp.shape[0], dtype=torch.int32, device=self.device).view(1, -1)
            v = self._views[L.i] = (swa, comp)
        return v

    def forward(self, L, x, residual, post, comb, pre, pos, slot, keys, select=None):
        """Layer L: the pending post (x None: none owed) folded in, K1, ``select(normed,
        qr)`` on an index source, K2 -> (x, residual, post, comb, pre)."""
        r, M = self.runner, residual.shape[0]
        swa, comp = self._caches(L)
        fold = x is not None
        if not fold:  # unread: the seam takes the residual as it is
            x, post, comb = r.normed[:M], pre, pre
        swa_idx, swa_lens, topk, _ = keys
        kw = {}
        if L.ratio:
            kw = {
                "topk_indices": topk,
                "comp_cache": comp,
                "comp_block_table": self._table,
            }
        return r.forward(
            self.weights[L.i],
            x,
            residual,
            post,
            comb,
            pre,
            pos,
            slot,
            swa,
            swa_idx,
            swa_lens,
            self.t2r,
            fold=fold,
            select=select,
            **kw,
        )

    @property
    def res_mid(self) -> torch.Tensor:
        return self.runner.res_mid


def build(tp: int, rank: int, group, device, layers: dict, widths, kinds):
    """The vLLM target, its launches compiled for ``widths``, or None (SGLang-AMD-Mono's
    then; the reason in the step stats' info). Collective: every rank decides alike."""
    import torch.distributed as dist

    def agree(ok: bool) -> bool:
        flag = torch.tensor([int(ok)], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        return bool(flag.item())

    target, weights, why = None, None, None
    if not wanted():
        why = f"{ENV_TARGET_MONO}=sgl"
    else:
        try:
            weights = checked_weights(tp, layers)
        except Exception as e:  # noqa: BLE001 (any refusal keeps SGLang-AMD-Mono's)
            why = f"vLLM's launches refuse the weights: {type(e).__name__}: {e}"
    if agree(weights is not None):  # the handshake is collective: all ranks or none
        try:
            target = VllmTarget(tp, rank, group, device, weights)
        except RuntimeError as e:  # the handshake failed, on every rank together
            why = f"vLLM's peer buffer: {e}"
    elif weights is not None:
        why = "vLLM's launches refuse another rank's weights"
    if target is not None:
        start = len(widths) * rank // tp
        try:
            for m in list(widths[start:]) + list(widths[:start]):
                target.precompile(m, kinds)
        except Exception as e:  # noqa: BLE001
            why = f"vLLM's launches did not build: {type(e).__name__}: {e}"
        if not agree(why is None):
            why = why or "vLLM's launches did not build on another rank"
            target.runner.peer.close()
            target = None
    stats.set_info(INFO, "vllm_mono" if target is not None else f"sgl_amd_mono ({why})")
    return target
