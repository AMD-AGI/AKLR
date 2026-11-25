# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
import struct

from aklr.core.contract.baseline import DistOpBaseline

__all__ = ["MuseSparkLayerBaseline"]

_LAYER = 0  # the mailbox tags' layer slot: one slot serves every call, the epoch tells them apart
_DEADLINE_S = 20.0  # every in-kernel wait's deadline


@functools.cache
def _aiter_kernel_signature() -> None:
    """sgl-amd-mono names its kernels with AITER's kernel_signature, newer than the AITER the
    bench pins: install AITER's two lines where it is missing, before sgl-amd-mono imports it."""
    from aiter.ops.flydsl.kernels import kernels_common

    if hasattr(kernels_common, "kernel_signature"):
        return

    def kernel_signature(**params: object) -> str:
        parts = [
            f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
        ]
        return kernels_common.format_kernel_name("_".join(parts))

    kernels_common.kernel_signature = kernel_signature


@functools.cache
def _group_state(device):
    """This rank's epoch, scratch and peer buffer, as sgl-amd-mono's MuseMonoRunner
    allocates them, the peer buffer in mori's symmetric heap; built by every rank of the
    group at its first call."""
    import mori.shmem as ms
    import torch
    from mori.shmem import mori_shmem_create_tensor
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.plan import BLOCKS
    from sgl_amd_mono.muse_spark.gfx950.config import HIDDEN, MAX_TOKENS, MOE_HIDDEN, Q_K, TOP_K
    from sgl_amd_mono.muse_spark.gfx950.layout import k1_scratch_layout, kattn_scratch_layout, scratch_bytes
    from sgl_amd_mono.muse_spark.gfx950.stages.ffn import k3_scratch_layout
    from sgl_amd_mono.muse_spark.gfx950.stages.post_attn import k2_scratch_layout, sym_bytes

    rank, npes = ms.shmem_mype(), ms.shmem_npes()
    epoch = torch.zeros(guard.GUARD_WORDS + BLOCKS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(_DEADLINE_S)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def zeros(n_bytes):
        return torch.zeros(n_bytes, dtype=torch.uint8, device=device)

    peers = mori_shmem_create_tensor((sym_bytes(npes) + guard.FLAG_BYTES,), torch.uint8)
    peers.zero_()
    table = torch.tensor(
        [ms.shmem_ptr_p2p(peers.data_ptr(), rank, pe) for pe in range(npes)], dtype=torch.int64, device=device
    )
    torch.cuda.synchronize(device)
    ms.shmem_barrier_all()
    bf16 = torch.bfloat16
    return {
        "rank": rank,
        "npes": npes,
        "epoch": epoch,
        "k1": zeros(scratch_bytes(k1_scratch_layout())),
        "attn": zeros(max(scratch_bytes(kattn_scratch_layout(MAX_TOKENS, q)) for q in (1, 8))),
        "k2": zeros(scratch_bytes(k2_scratch_layout())),
        "k3": zeros(scratch_bytes(k3_scratch_layout())),
        "peers": peers,
        "table": table,
        # the launch's intermediate rows: attention output, picks and their scores, the
        # experts' input
        "o": torch.empty(MAX_TOKENS, Q_K, dtype=bf16, device=device),
        "ids": torch.empty(MAX_TOKENS, TOP_K, dtype=torch.int32, device=device),
        "scores": torch.empty(MAX_TOKENS, TOP_K, dtype=torch.float32, device=device),
        "xin": torch.empty(MAX_TOKENS, MOE_HIDDEN, dtype=bf16, device=device),
        "hidden": HIDDEN,
        "pooled": MOE_HIDDEN,
    }


@functools.cache
def _launch(rows: int, npes: int):
    """sgl-amd-mono's whole-layer launch as its runner builds it by default: TP experts,
    the pre-quantized expert input, the dynamic gate-up queue, and the pooler sharded by
    rows up to 16 rows."""
    from sgl_amd_mono.muse_spark.gfx950.layer import LayerBuild, build_layer

    key = LayerBuild(
        tokens=rows,
        q_len=8,
        npes=npes,
        next_k1=True,
        tpx=True,
        xq_pre=True,
        ug_dyn=True,
        pshard=npes == 8 and rows <= 16,
    )
    return build_layer(key), key.pshard


def muse_spark_layer_baseline(
    q,
    o_gate,
    residual,
    k_cache,
    v_cache,
    page_table,
    context_lens,
    o_proj,
    alpha,
    beta,
    pre_norm_weight,
    router_hi,
    router_lo,
    router_bias,
    pooler,
    pooled_norm_weight,
    w13,
    w13_scale,
    w2,
    w2_scale,
    expert_norm_weight,
    unpooler,
    post_ffn_weight,
    alpha2,
    beta2,
    next_norm_weight,
    next_qkvg,
    rotary,
    positions,
    slots,
    swa_map,
    next_k_cache,
    next_v_cache,
    scale=0.125,
    sliding_window=0,
    next_rope=True,
    next_sliding=False,
    tp_size=8,
    next_input=None,
    next_k=None,
    next_v=None,
):
    """sgl-amd-mono's Muse Spark whole-layer launch (the commit SGLang pins), on every rank
    of the group: its attention, K2, K3 and the next layer's K1 in one launch."""
    import torch

    _aiter_kernel_signature()
    from sgl_amd_mono.muse_spark.gfx950 import config

    # the launch is built for Muse Spark's shapes at TP8 alone
    muse = (residual.shape[1], q.shape[1], k_cache.shape[1], w13.shape[0], pooler.shape[0], w2.shape[2] * 2)
    if (tp_size, muse) != (
        8,
        (
            config.HIDDEN,
            config.LOCAL_Q_HEADS,
            config.LOCAL_KV_HEADS,
            config.EXPERTS,
            config.MOE_HIDDEN,
            config.INTER,
        ),
    ):
        raise NotImplementedError(
            f"sgl-amd-mono's launch serves Muse Spark at TP8, not tp_size {tp_size}, shapes {muse}"
        )
    s = _group_state(q.device)
    if s["npes"] != tp_size:
        raise NotImplementedError(f"mori's group has {s['npes']} ranks, the call's tp_size is {tp_size}")
    rows = q.shape[0]
    launch, pshard = _launch(rows, tp_size)
    if next_input is None:
        next_input = torch.empty_like(residual, dtype=torch.bfloat16)
    if next_k is None:
        next_k = torch.empty(rows, *k_cache.shape[1:], dtype=torch.bfloat16, device=q.device)
    if next_v is None:
        next_v = torch.empty_like(next_k)
    rank = s["rank"]
    slice_rows = s["pooled"] // tp_size
    pool = pooler[rank * slice_rows : (rank + 1) * slice_rows] if pshard else pooler
    scale_bits = struct.unpack("i", struct.pack("f", float(scale)))[0]
    p = torch.Tensor.data_ptr
    launch(
        # attention
        p(q), p(s["o"]), p(k_cache), p(v_cache), p(page_table), page_table.shape[1], page_table.numel() * 4,
        p(context_lens), 0, 8, int(sliding_window), 0, scale_bits, p(s["attn"]),
        # K2: o-norm-gate, o_proj and its all-reduce, the attention boundary, router, pooler
        p(o_gate), p(residual), p(o_proj), p(alpha), p(beta), p(pre_norm_weight), p(router_hi),
        p(router_lo), p(router_bias), p(pool), p(pooled_norm_weight), p(s["ids"]), p(s["scores"]),
        p(s["xin"]), p(s["k2"]),
        # K3: the experts, finalize, unpool, the FFN boundary
        0, p(w13), p(w13_scale), p(w2), p(w2_scale), p(expert_norm_weight), p(post_ffn_weight),
        p(alpha2), p(beta2), p(next_norm_weight), p(unpooler), p(next_input), p(s["k3"]),
        # the next layer's K1
        p(next_qkvg), p(rotary), p(positions), p(slots), p(swa_map), p(next_k_cache), p(next_v_cache),
        p(next_k), p(next_v), p(s["k1"]), int(bool(next_rope)), int(bool(next_sliding)), 1, 1,
        # the group
        p(s["peers"]), p(s["table"]), rank, p(s["epoch"]), _LAYER, 0,
        stream=torch.cuda.current_stream(q.device),
    )  # fmt: skip
    s["epoch"][0].add_(1)
    return next_input, next_k, next_v


class MuseSparkLayerBaseline(DistOpBaseline):
    def setup(self):
        """mori's symmetric heap over the group: the kernels' and the baseline's peer buffers
        (the partial rows and flags each rank pushes to the others) live there."""
        import mori
        import torch
        import torch.distributed as dist

        torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
        mori.shmem.shmem_torch_process_group_init("mori")

    def baseline(self, **call):
        """The layer's next_input, next_k and next_v, and the mutable inputs q, o_gate, residual,
        next_k_cache and next_v_cache as the launch leaves them."""
        next_input, next_k, next_v = muse_spark_layer_baseline(**call)
        return {
            "next_input": next_input,
            "next_k": next_k,
            "next_v": next_v,
            "q": call["q"],
            "o_gate": call["o_gate"],
            "residual": call["residual"],
            "next_k_cache": call["next_k_cache"],
            "next_v_cache": call["next_v_cache"],
        }
