# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A DSpark verify's committed KV into the draft stages' SWA caches in one launch, as
Infera-Flash's ``kv_norm_rope_qdq_multi``: ``hipops.swa_store`` reads each stage's slice off
the stacked wkv GEMM's output. ``AKLR_DSV41_INJECT_STORE=0`` keeps sglang's stores."""

from __future__ import annotations

import os

import torch

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import serving
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

ENV = "AKLR_DSV41_INJECT_STORE"
DSPARK = "sglang.srt.models.deepseek_v4_dspark"
DIM = 512
MAX_STAGES = 4  # swa_store's

SGLANG_READS = {
    "srt/models/deepseek_v4_dspark.py": (
        "DeepseekV4ForCausalLMDSpark",
        "write_target_hidden_kv(main_hidden)",
        "write_target_hidden_kv(swa_loc)",
        "write_target_hidden_kv(positions)",
        "write_target_hidden_kv(pool)",
        "project_target_hidden",
        "CommitKvProj",
        "is_unified_kv_triton",
        "stages",
        "freqs_cis",
    ),
    "srt/models/deepseek_v4.py": ("self_attn", "wkv", "kv_norm", "layer_id", "eps"),
    "kernels/ops/speculative/dspark/dspark_draft_model.py": (
        "CommitKvProj",
        "execute(main_x)",
        "execute(wkv_linears)",
        "execute(allow_strided_output)",
    ),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "uniform_fp8",
        "kv_layout",
        "swa_page_size",
        "get_swa_raw_buffer",
    ),
    "kernels/ops/attention/dsv4/kv_layout.py": ("KVLayout", "V4", "V41", '"v41"'),
}


def _enabled() -> bool:
    value = os.environ.get(ENV) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV}={value!r}: expected '1' or '0'")
    return value == "1"


class _Stages:
    """The draft's stages as swa_store reads them: their norms, rope tables and layers."""

    def __init__(self, model):
        attns = [stage.self_attn for stage in model.stages]
        self.wkv = [attn.wkv for attn in attns]
        self.weights = [attn.kv_norm.weight.data for attn in attns]
        self.freqs = [torch.view_as_real(attn.freqs_cis).flatten(-2).contiguous() for attn in attns]
        self.layers = [attn.layer_id for attn in attns]
        self.eps = attns[0].eps
        self.why = None
        if not 0 < len(attns) <= MAX_STAGES:
            self.why = f"{len(attns)} stages"
        elif any(attn.eps != self.eps for attn in attns):
            self.why = "the stages' eps differ"
        elif any(w.dtype != torch.bfloat16 or w.numel() != DIM for w in self.weights):
            self.why = "kv_norm weights not [512] bf16"
        elif any(f.dim() != 2 or f.shape[1] != 64 for f in self.freqs):
            self.why = "rope tables not [positions, 64]"


def _why_not(model, pool) -> str | None:
    if torch.version.hip is None:
        return "not HIP"
    if not serving():
        return "not a DeepSeek-V4.1 server AKLR serves"
    if not _enabled():
        return f"{ENV}=0"
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.models import deepseek_v4_dspark

    # the gate's module differs between sglang builds; the model's import is the one it reads
    if deepseek_v4_dspark.is_unified_kv_triton():
        return "unified KV"
    if pool.uniform_fp8 or pool.kv_layout not in (KVLayout.V4, KVLayout.V41):
        return f"a {pool.kv_layout.value} pool"
    stages = getattr(model, "_aklr_inject", None)
    if stages is None:
        stages = model._aklr_inject = _Stages(model)
    return stages.why


@sglang_route(
    f"{DSPARK}.DeepseekV4ForCausalLMDSpark.write_target_hidden_kv",
    versions=VERSIONS,
    around=True,
)
def _write_target_hidden_kv(orig, self, *, main_hidden, swa_loc, positions, pool, **kwargs):
    committed = swa_loc if isinstance(swa_loc, _Committed) else None
    why = None if kwargs else _why_not(self, pool)
    if committed is not None and (why is not None or kwargs):
        swa_loc = committed.slots()
    if kwargs or why is not None:
        stats.set_info("inject_store", f"sglang's per-stage stores ({why})")
        return orig(
            self,
            main_hidden=main_hidden,
            swa_loc=swa_loc,
            positions=positions,
            pool=pool,
            **kwargs,
        )
    from sglang.kernels.ops.speculative.dspark.dspark_draft_model import CommitKvProj

    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    stages = self._aklr_inject
    main_x = self.project_target_hidden(main_hidden)
    kvs = CommitKvProj.execute(main_x=main_x, wkv_linears=stages.wkv, allow_strided_output=True)
    caches = [pool.get_swa_raw_buffer(layer) for layer in stages.layers]
    v41 = pool.kv_layout.value == "v41"
    # the committed rows' slots in the store itself (_inject_mla), else the given ones
    slots = {}
    if committed is not None:
        loc = committed.commit
        slots = {
            "cache_loc": committed.cache_loc,
            "mapping": committed.mapping,
            "commit": committed.commit,
            "width": committed.width,
        }
    else:
        loc = swa_loc.to(torch.int32)
    pos = positions.contiguous()
    step = kvs[0].element_size()
    stage_stride = (kvs[1].data_ptr() - kvs[0].data_ptr()) // step if len(kvs) > 1 else 0
    if all(
        kv.stride() == kvs[0].stride() and kv.data_ptr() == kvs[0].data_ptr() + s * stage_stride * step
        for s, kv in enumerate(kvs)
    ):
        hipops.ops().swa_store(
            kvs[0],
            stage_stride,
            stages.weights,
            stages.freqs,
            pos,
            loc,
            caches,
            stages.eps,
            pool.swa_page_size,
            v41,
            **slots,
        )
        stats.set_info(
            "inject_store",
            "one launch, every stage" + (", the committed slots in it" if slots else ""),
        )
        return
    if committed is not None:
        loc = committed.slots()
    for s, kv in enumerate(kvs):
        hipops.ops().swa_store(
            kv,
            0,
            stages.weights[s : s + 1],
            stages.freqs[s : s + 1],
            pos,
            loc,
            caches[s : s + 1],
            stages.eps,
            pool.swa_page_size,
            v41,
        )
    stats.set_info("inject_store", "a launch a stage")


INJECTOR = "sglang.srt.speculative.dspark_components.dspark_kv_inject.TargetHiddenKvInjector"
# "0": the committed rows' SWA slots by sglang's own launch before the store
ENV_COMMITTED = "AKLR_DSV41_INJECT_COMMITTED"


class _Committed:
    """A verify's committed rows for the store to translate (``committed_swa_locations``'s
    inputs); ``slots()`` makes sglang's int32 slots where the store cannot."""

    def __init__(self, cache_loc, mapping, commit, width: int):
        self.cache_loc, self.mapping, self.commit, self.width = (
            cache_loc,
            mapping,
            commit,
            width,
        )

    def slots(self):
        from sglang.kernels.ops.speculative.dspark.commit_swa import (
            committed_swa_locations,
        )

        return committed_swa_locations(self.cache_loc, self.mapping, self.commit, self.width)


@sglang_route(f"{INJECTOR}._inject_mla", versions=VERSIONS, around=True)
def _inject_mla(orig, self, *args, **kwargs):
    """sglang's injection of a verify's committed rows, their SWA slots made in the store's
    launch (``hipops.swa_store``) instead of by ``committed_swa_locations``."""
    pool = kwargs.get("pool")
    cache_loc, cache_loc_2d = kwargs.get("cache_loc"), kwargs.get("cache_loc_2d")
    commit_lens = kwargs.get("commit_lens")
    mapping = getattr(pool, "full_to_swa_index_mapping", None)
    if (
        args
        or not serving()
        or os.environ.get(ENV_COMMITTED, "1") == "0"
        or not _enabled()
        or getattr(pool, "swa_req_ring_size", None) is not None
        # a ring slot: the committed slots here are not the ring's
        or kwargs.get("state_slot") is not None
        or cache_loc is None
        or cache_loc_2d is None
        or commit_lens is None
        or mapping is None
        or not cache_loc.is_cuda
        or not cache_loc.is_contiguous()
        or cache_loc.dtype != torch.int64
        or not commit_lens.is_contiguous()
        or commit_lens.dtype != torch.int32
        or mapping.dtype != torch.int64
        or not mapping.is_contiguous()
        or cache_loc.numel() != cache_loc_2d.numel()
    ):
        return orig(self, *args, **kwargs)
    with torch.inference_mode():
        self.draft_model.write_target_hidden_kv(
            main_hidden=kwargs["target_hidden"],
            swa_loc=_Committed(cache_loc, mapping, commit_lens, cache_loc_2d.shape[1]),
            positions=kwargs["positions"],
            pool=pool,
        )
