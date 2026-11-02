# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/moe.py
"""The mono layer's MoE stages: one V4.1 layer's TP-sharded FFN from the FFN seam's
normed rows to the reduced output, its all-reduce in-kernel. Stages ``TL_STAGES`` (xq
.. reduce), handing off through the tagged mailbox."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aklr.plugins.sglang.models.deepseek_v4.kernels.common.aiter_flydsl import (
    buffer_ops as bo,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.common.mx import (
    FP4,
    FP8,
    FP8_MAX,
    UNIT_SCALE,
    clamp_fp8,
    code_ceil,
    fp4_tile_load,
    mfma_scaled,
    rcp_pow2,
    scale_index,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.common.ops import (
    CM_DEV,
    CM_NT,
    _hw_f32,
    ballot,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    div_rn,
    fp8_pack4,
    fresh,
    hw_exp2,
    hw_rcp,
    lane_gather,
    lanes_below,
    popcount,
    row_sum,
    rsrc,
    stamp,
    sum_partials,
    traced,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
    pair_layout,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.common.sync import (
    SPIN_SLEEP,
    preg,
    publish,
    sreg,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.stages import (
    moe_plan as mplan,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.stages.dims import UG_PART

# helpers the kernel calls are imported by name: FlyDSL keys a build by the
# source of the same-directory functions it names, never through a module
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.stages.gemv import (
    ROWS,
    TILE,
    gemv_fp8_loads,
    gemv_fp8_mfmas,
    ld_f32,
    lds_row,
    lds_tail,
    plus,
    poll_copy,
    tile_rows,
    token_tiles,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.stages.moe_shape import (
    MASK_WORDS,
    MoeBuild,
    route_shape,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.stages.router import (
    PARTS as ROUTER_PARTS,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.stages.router import (
    router_tile_mfma,
    router_weight_loads,
    router_x_loads,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.vllm_mono.stages.seam import (
    HIDDEN,
    attn_peer_bytes,
)

# a pick tile is a token tile's MFMA N: ``gather_x8r`` fills one from the other
assert mplan.TILE == TILE
SCALE = 1.5
LIMIT = 10.0  # swiglu_limit
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
DOWN_TASKS = HIDDEN // ROWS  # 320
# (task, slot) scale blocks a down wave holds a round (``stage_down``): 4 spills
DOWN_R_MAX = 2
# a down slot's K steps on this many independent MFMA accumulators
DOWN_CHAINS = 2
# a reduce task's columns: wider tasks poll more pairs a thread, each pass a round
# trip in series
RED_COLS = 32
RED_TASKS = HIDDEN // RED_COLS  # 160
# CTA b polls route copy b % ROUTE_COPIES: every CTA on one small region
# serializes on its lines
ROUTE_COPIES = 8
X8_WORDS = HIDDEN // 4
TAGS = 256  # the ug queue's / down count's slots, a word each: ``c["tag"]`` picks one

XQ0 = 0
DOWN0 = 0
RED0 = 0


# ``timeline`` stamps: kernel start, then the end of each stage's task loop
TL_STAGES = ("xq", "router", "route", "shared", "ug", "down", "reduce")
TL_POINTS = 1 + len(TL_STAGES)


# the MoE stages' kernel arguments, in launch order
ARGS = (
    "x",
    "gate_w",
    "bias",
    "w13",
    "w13_s",
    "w2",
    "w2_s",
    "sgu",
    "sgu_s",
    "sw2",
    "sw2_s",
    "out",
)


def scratch_layout(key: MoeBuild) -> dict[str, tuple[int, int]]:
    tokens, experts, topk = key.tokens, key.experts, key.topk
    rs = route_shape(key)
    d = rs.dims
    pairs = {
        # the routed experts' MXFP8 x (aiter's fused_mx_quant_moe_sort) and the
        # shared expert's (ATOM quantize_fp8): their amax floors differ
        "x8r": tokens * X8_WORDS // 2,
        "x8rs": tokens * HIDDEN // 32 // 2,
        "x8": tokens * X8_WORDS // 2,
        "x8s": tokens * HIDDEN // 32 // 2,
        # a token's four MXFP8 regions written (plain words): ``stage_xq``
        "x8rdy": tokens,
        "logit": tokens * experts * ROUTER_PARTS,
        "route": ROUTE_COPIES * tokens * 2 * topk,
        "smid": tokens * d.sh_inter,
        # SMID quantize_fp8'd (``smid_quant``): the down stage's MXFP8 rows
        "smq": tokens * d.sh_inter // 4,
        "smc": tokens * d.sh_inter // 32,
        # a pick a row, in the slot order (``RouteShape.order0``)
        "mid": rs.picks * d.mid_words,
        "mids": rs.picks * d.inter // 32,
        # an ug task's done flag (``stage_ug``)
        "ugf": (rs.max_tiles if rs.multi else rs.max_u) * rs.ug_parts,
        # the ug task queue: a word a mailbox tag (a layer), counting up from the
        # step's clear (``take_ug_task``); plain words, 2 a pair
        "ugq": TAGS // 2,
        # the down stage's arrival count, a word a tag (``take_down_rank``)
        "dq": TAGS // 2,
    }
    return pair_layout(pairs.items())


def peer_bytes(tokens: int, tp: int) -> int:
    """FFN: ``[source rank][token][hidden / 2]`` bf16 pairs."""
    return tp * tokens * HIDDEN // 2 * 8


def ffn_offset(tokens: int, tp: int) -> int:
    """FFN's byte offset in the peer buffer: past the ATTN region."""
    return attn_peer_bytes(tokens, tp)


def region_pair(src, t, s, col):
    return (src * s + t) * (HIDDEN // 2) + col // 2


# ---------------------------------------------------------------- xq
@traced
def stage_xq(c, t):
    """Token t, a group of 32 a thread."""
    tid = c["tid"]
    a = c["args"]
    # thread g < 160 owns the 32 values of group g; the rest mirror thread 159
    g = fx.min(tid, HIDDEN // 32 - 1)
    mine = tid < HIDDEN // 32
    # 32 bf16 as four 16-byte loads (a load an element made the stage issue bound)
    words = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["x"]),
                (t * HIDDEN + g * 32) // 2 + 4 * q,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=c["x_cm"],
            )
        )
        for q in range(4)
    ]
    v = [(bf_lo if i % 2 == 0 else bf_hi)(words[i // 8][i % 8 // 2]) for i in range(32)]
    amax0 = abs(v[0])
    for y in v[1:]:
        amax0 = fx.max(amax0, abs(y))
    # the routed experts' (aiter fused_mx_quant_moe_sort): amax floored 1e-10;
    # the shared expert's: vLLM's mxfp8_e4m3_quantize (its MXFP8 linear)
    if mine:
        put_mxfp8_group(c, "x8r", "x8rs", t, g, v, fx.max(amax0, fx.Float32(1e-10)))
        put_mxfp8_group_vllm(c, "x8", "x8s", t, g, v, amax0)


def put_mxfp8_group(c, words_mb, codes_mb, t, g, v, amax):
    """Token t's group g of 32 values ``v`` as MXFP8 (the ceil code of
    ``amax`` / 448, x / scale clamped, RNE) -> scratch ``words_mb`` /
    ``codes_mb`` (plain words: the token's X8RDY announces them)."""
    code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
    inv = rcp_pow2(code)
    words = []
    for w4 in range_constexpr(4):
        ws = [fp8_pack4(*[clamp_fp8(v[8 * w4 + 4 * h + i] * inv) for i in range(4)]) for h in range(2)]
        words.append(ws)
    for h in range_constexpr(2):
        st_dev(
            fx.Vector.from_elements(words[2 * h] + words[2 * h + 1], fx.Int32),
            c[words_mb],
            t * X8_WORDS + g * 8 + 4 * h,
        )
    st_dev(fx.Int32(code), c[codes_mb], t * (HIDDEN // 32) + g)


FLT_MIN = 1.1754943508222875e-38


def vllm_mx_code(amax):
    """VLLM's ``mxfp8_e4m3_quantize`` scale code of a 32-group: the smallest
    power of two >= RN(max(amax, FLT_MIN) / 448), clamped to [0, 254]."""
    a = fx.max(amax, fx.Float32(FLT_MIN))
    q = div_rn(a, fx.Float32(FP8_MAX), fx.Float32(1.0 / FP8_MAX))
    bits = q.bitcast(fx.Int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    return fx.min(fx.max(e, fx.Int32(0)), fx.Int32(254))


def put_mxfp8_group_vllm(c, words_mb, codes_mb, t, g, v, amax):
    """``put_mxfp8_group`` with vLLM's rule (``vllm_mx_code``; x * 2^(127 - code),
    which code 0 keeps finite)."""
    code = vllm_mx_code(amax)
    inv = ((254 - code) << 23).bitcast(fx.Float32)
    words = []
    for w4 in range_constexpr(4):
        ws = [fp8_pack4(*[clamp_fp8(v[8 * w4 + 4 * h + i] * inv) for i in range(4)]) for h in range(2)]
        words.append(ws)
    for h in range_constexpr(2):
        st_dev(
            fx.Vector.from_elements(words[2 * h] + words[2 * h + 1], fx.Int32),
            c[words_mb],
            t * X8_WORDS + g * 8 + 4 * h,
        )
    st_dev(fx.Int32(code), c[codes_mb], t * (HIDDEN // 32) + g)


# ---------------------------------------------------------------- router
@traced
def stage_router(c, task):
    """Router task (token group, row tile, K part): the ``router`` tile over the group's
    tokens, a token tile at a time -> the part's f32 sum (``route`` sums parts in order).
    Every tile's x loads issue first, else they would be read serially."""
    s, tid, red, lane, wave = c["S"], c["tid"], c["red"], c["lane"], c["wave"]
    a = c["args"]
    experts = c["rs"].experts
    groups, per = c["rs"].router_groups
    per_group = experts // ROWS * ROUTER_PARTS
    tile, part = task % per_group // ROUTER_PARTS, task % ROUTER_PARTS
    g0 = 0 if groups == 1 else task // per_group * per
    # the group's token tiles: (first token, tokens), tokens traced past one group
    tiles = [
        (
            plus(g0, j0),
            (min(TILE, s - j0) if groups == 1 else fx.min(fx.Int32(min(TILE, per - j0)), s - plus(g0, j0))),
        )
        for j0 in range(0, per, TILE)
    ]
    wvs = router_weight_loads(a["gate_w"], tile, part, lane, wave, CM_NT)
    xs = [router_x_loads(a["x"], part, n, lane, wave, c["x_cm"], t0) for t0, n in tiles]
    for (t0, n), xvs in zip(tiles, xs):
        router_tile_mfma(wvs, xvs, lane, wave, red)
        gpu.barrier()
        if tid < ROWS * n:
            r = tid % ROWS
            tl = tid // ROWS
            v = row_sum(red, r, tl)
            c["put"](
                c["logit"],
                (plus(t0, tl) * experts + tile * ROWS + r) * ROUTER_PARTS + part,
                v,
            )
        gpu.barrier()


# ---------------------------------------------------------------- route
def _score(x):
    """sqrt(log2(1 + exp2(x log2e)) ln2), x itself past 20 (topk_gating)."""
    sp = fx.Float32(
        (x > fx.Float32(20.0)).select(
            x,
            _hw_f32("llvm.amdgcn.log.f32", fx.Float32(1.0) + hw_exp2(x * fx.Float32(LOG2E)))
            * fx.Float32(LN2),
        )
    )
    return fx.Float32(fmath.sqrt(sp))


@traced
def stage_route(c, t):
    """VLLM's topk_hash_softplus_sqrt for token t on wave 0 (f32 logits)."""
    lane, wave = c["lane"], c["wave"]
    experts, topk = c["rs"].experts, c["rs"].topk
    a = c["args"]
    if wave == 0:
        ept = experts // 64
        # each logit's router part sums, added in part order (vLLM's gate: f32 out)
        assert ROUTER_PARTS == 4
        # infera-flash: the bias in flight before the poll, off the route's critical path
        bias = [ld_f32(a["bias"], lane + 64 * i) for i in range(ept)]
        got = c["poll"](
            [
                (c["logit"], (t * experts + lane + 64 * i) * ROUTER_PARTS + h, 2)
                for i in range(ept)
                for h in (0, 2)
            ]
        )
        logits = []
        for i in range_constexpr(ept):
            parts = got[2 * i] + got[2 * i + 1]
            tot = fx.Float32(0.0)
            for p in range_constexpr(ROUTER_PARTS):
                tot = tot + parts[p].bitcast(fx.Float32)
            logits.append(tot)
        vals, orig, idxs = [], [], []
        for i in range_constexpr(ept):
            e = lane + 64 * i
            sc = _score(logits[i])
            # aiter's NaN guards: a NaN score weighs 0 and never wins
            orig.append(fx.Float32(fx.isnan(sc).select(fx.Float32(0.0), sc)))
            vals.append(fx.max(sc + bias[i], fx.Float32(float("-inf"))))
            idxs.append(fx.Int32(e))
        for p, q in c["rs"].sort_net:
            sw = vals[p] < vals[q]
            vals[p], vals[q] = sw.select(vals[q], vals[p]), sw.select(vals[p], vals[q])
            orig[p], orig[q] = sw.select(orig[q], orig[p]), sw.select(orig[p], orig[q])
            idxs[p], idxs[q] = sw.select(idxs[q], idxs[p]), sw.select(idxs[p], idxs[q])
        cursor = fx.Int32(0)
        total = fx.Float32(0.0)
        my_w, my_id = fx.Float32(0.0), fx.Int32(0)
        for k in range_constexpr(topk):
            cv, ci, co = fx.Float32(float("-inf")), fx.Int32(0), fx.Float32(0.0)
            for i in range_constexpr(ept):
                at = cursor == i
                cv, ci, co = (
                    at.select(vals[i], cv),
                    at.select(idxs[i], ci),
                    at.select(orig[i], co),
                )
            best = butterfly(cv, (32, 16, 8, 4, 2, 1), fx.max)
            ball = ballot(cv == best)
            win = fx.Int32((ball == fx.Int64(0)).select(fx.Int32(0), fx.Int32(fmath.cttz(ball))))
            win_id = fx.Int32(rocdl.readlane(T.i32, ci.ir_value(), win.ir_value()))
            won = (cursor < ept) & (ci == win_id)
            mine_o = fx.Float32(won.select(co, fx.Float32(0.0)))
            w = fx.Int32(
                rocdl.readlane(T.i32, mine_o.bitcast(fx.Int32).ir_value(), (win_id & 63).ir_value())
            ).bitcast(fx.Float32)
            cursor = cursor + won.select(1, 0)
            total = total + w
            my_w = (lane == k).select(w, my_w)
            my_id = (lane == k).select(win_id, my_id)
        scale = fx.Float32(SCALE) / fx.max(total, fx.Float32(1e-20))
        if lane < topk:
            for r in range_constexpr(ROUTE_COPIES):
                at = (r * c["S"] + t) * 2 * topk + lane
                c["put"](c["route"], at, my_id)
                c["put"](c["route"], at + topk, my_w * scale)
    gpu.barrier()


@traced
def load_route(c, tab):
    """Every token's (ids, weights) into LDS ``tab``, then U and its slots (``RouteShape``).
    A pick a thread ORs its token's bit into its expert's bitmap; counts and ranks are
    popcounts, so every CTA derives the same order whatever the OR order."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rs = c["rs"]
    topk, max_u, experts = rs.topk, rs.max_u, rs.experts
    n = s * topk
    assert s <= 32 * MASK_WORDS
    # a thread's route words in one poll batch (a pass each would be round trips in
    # series); a word past 2n re-reads the last and is not stored
    idxs = [plus(pas, tid) for pas in range(0, 2 * n, THREADS)]
    copy_at = c["bid"] % ROUTE_COPIES * (2 * n)
    got = c["poll"]([(c["route"], copy_at + fx.min(idx, 2 * n - 1), 1) for idx in idxs])
    for i in range_constexpr(len(idxs)):
        if idxs[i] < 2 * n:
            fx.ptr_store(got[i][0], tab + idxs[i])
    for pas in range_constexpr(0, MASK_WORDS * experts, THREADS):
        idx = plus(pas, tid)
        if idx < MASK_WORDS * experts:
            fx.ptr_store(fx.Int32(0), tab + (rs.mask0 + idx))
    gpu.barrier()
    if tid < n:
        t = tid // topk
        eid = fx.ptr_load(tab + (t * 2 * topk + tid % topk))
        fx.llvm.atomic_or(
            tab + (rs.mask0 + eid * MASK_WORDS + t // 32),
            fx.Int32(1) << (t % 32),
            syncscope="workgroup",
        )
    gpu.barrier()
    base = 2 * n
    # a chunk of 64 experts a wave: each chunk's counts and prefixes, its totals
    # through LDS ``red`` (free here), then offset by the chunks below it -- the
    # serial walk's values
    chunks = experts // 64
    assert chunks <= WAVES
    sums = c["red"]
    e = lane + 64 * fx.min(wave, chunks - 1)
    count = fx.Int32(0)
    for w in range_constexpr(MASK_WORDS):
        word = fx.ptr_load(tab + (rs.mask0 + e * MASK_WORDS + w))
        count = count + popcount(word)
    ball = ballot(count != 0)
    # this chunk's inclusive prefixes, lanes ascending: picks, and pick tiles
    incl = lane_prefix(count, lane)
    tiles = mplan.pick_tiles(count)
    tincl = lane_prefix(tiles, lane) if rs.multi else tiles
    totals = [popcount(ball), lane_gather(incl, 63), lane_gather(tincl, 63)]
    if (wave < chunks) & (lane == 0):
        for k in range_constexpr(3):
            fx.ptr_store(totals[k].bitcast(fx.Float32), sums + (wave * 3 + k))
    gpu.barrier()
    below_c = [fx.Int32(0), fx.Int32(0), fx.Int32(0)]  # n_u, first, tile0
    for j in range_constexpr(chunks - 1):
        for k in range_constexpr(3):
            v = fx.ptr_load(sums + (j * 3 + k)).bitcast(fx.Int32)
            below_c[k] = below_c[k] + (j < wave).select(v, fx.Int32(0))
    n_u, first, tile0 = below_c
    slot = n_u + lanes_below(ball, lane)
    # e afresh: hoisted, its address held a VGPR across the prefix sums (spill)
    e_at = fresh(e)
    if wave < chunks:
        fx.ptr_store(slot, tab + (rs.expert_slot0 + e_at))
        if count != 0:
            fx.ptr_store(fx.Int32(e), tab + (base + 1 + slot))
            fx.ptr_store(first + incl - count, tab + (rs.first0 + slot))
        if const_expr(rs.multi):
            # the slot's pick tiles: their first index, and each one's slot
            tf = tile0 + tincl - tiles
            if count != 0:
                fx.ptr_store(tf, tab + (rs.tile_first0 + slot))
            for j in range_constexpr(rs.slot_tiles):
                if j < tiles:
                    fx.ptr_store(slot, tab + (rs.tile_slot0 + tf + j))
    if (wave == chunks - 1) & (lane == 0):
        n_all = n_u + totals[0]
        fx.ptr_store(n_all, tab + base)
        fx.ptr_store(fx.Int32(n), tab + (rs.first0 + n_all))
        if const_expr(rs.multi):
            fx.ptr_store(tile0 + totals[2], tab + (rs.tile_first0 + n_all))
    gpu.barrier()
    if tid < n:
        t = tid // topk
        eid = fx.ptr_load(tab + (t * 2 * topk + tid % topk))
        slot = fx.ptr_load(tab + (rs.expert_slot0 + eid))
        fx.ptr_store(slot, tab + (base + 1 + max_u + tid))
        # the tokens below t that picked this expert: the picks before tid in it
        rank = fx.Int32(0)
        for w in range_constexpr(MASK_WORDS):
            word = fx.ptr_load(tab + (rs.mask0 + eid * MASK_WORDS + w))
            lo = t - 32 * w
            keep = (lo >= 32).select(
                fx.Int32(-1),
                (lo > 0).select((fx.Int32(1) << fx.max(lo, 0)) - 1, fx.Int32(0)),
            )
            rank = rank + popcount(word & keep)
        pos = fx.ptr_load(tab + (rs.first0 + slot)) + rank
        fx.ptr_store(tid, tab + (rs.order0 + pos))
    gpu.barrier()


def lane_prefix(v, lane):
    """The wave's inclusive prefix sum of v, lanes ascending."""
    incl = v
    for d in (1, 2, 4, 8, 16, 32):
        moved = lane_gather(incl, fx.max(lane - d, 0))
        incl = incl + (lane >= d).select(moved, fx.Int32(0))
    return incl


# ---------------------------------------------------------------- shared ug
def _silu_mul(g, u):
    """VLLM's silu_and_mul_with_clamp (the shared expert's): gate min limit,
    up clamped; silu = g / (1 + expf(-g)) rounded to bf16, then times up."""
    g = bf16_round(fx.min(g, fx.Float32(LIMIT)))
    u = fx.max(fx.min(u, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
    act = bf16_round(g / (fx.Float32(1.0) + fx.Float32(fmath.exp(-g))))
    return act * u


@traced
def stage_shared(c, task, gops, t0, n):
    """Gate rows 16 task .. (weights ``gops``) and the up rows against tokens t0 .. t0 + n
    (x in LDS; ``n`` traced: a tile picked at run time) -> silu x up -> SMID; an odd task
    then quantizes its group of 32 (``smid_quant``)."""
    tid, red, d = c["tid"], c["red"], c["d"]
    a = c["args"]
    rows = None if isinstance(n, int) else n
    gemv_fp8_mfmas(c, HIDDEN, c["xl"], c["xsl"], red, gops, rows=rows)
    gpu.barrier()
    gate = fx.Float32(0.0)
    if tid < ROWS * n:
        gate = row_sum(red, tid % ROWS, tid // ROWS)
    gpu.barrier()
    uops = gemv_fp8_loads(c, a["sgu"], a["sgu_s"], HIDDEN, d.shared_tasks + task, row_major=True)
    gemv_fp8_mfmas(c, HIDDEN, c["xl"], c["xsl"], red, uops, rows=rows)
    gpu.barrier()
    if tid < ROWS * n:
        r = tid % ROWS
        t = plus(t0, tid // ROWS)
        up = row_sum(red, r, tid // ROWS)
        v = fx.Float32(_silu_mul(bf16_round(gate), bf16_round(up)).to(fx.BFloat16))
        c["put"](c["smid"], t * d.sh_inter + task * ROWS + r, v)
        fx.ptr_store(v, c["pair"] + tid)
    gpu.barrier()
    if task % 2 == 1:
        smid_quant(c, task, t0, n)
    gpu.barrier()


@traced
def smid_quant(c, task, t0, n):
    """Odd shared task ``task``: its 16 columns (LDS ``pair``) and the even task's (SMID)
    quantized as one group of 32 -> SMQ / SMC, a thread a token. Once here, not in every
    down CTA that reads them."""
    tid, sh = c["tid"], c["d"].sh_inter
    assert c["d"].shared_tasks % 2 == 0  # tasks pair into groups
    if tid < n:
        t = plus(t0, tid)
        got = c["poll"]([(c["smid"], t * sh + (task - 1) * ROWS + r, 1) for r in range(ROWS)])
        vals = [got[r][0].bitcast(fx.Float32) for r in range(ROWS)] + [
            fx.ptr_load(c["pair"] + (tid * ROWS + r)) for r in range(ROWS)
        ]
        amax = abs(vals[0])
        for y in vals[1:]:
            amax = fx.max(amax, abs(y))
        # vLLM's down_proj input quant (mxfp8_e4m3_quantize)
        code = vllm_mx_code(amax)
        inv = ((254 - code) << 23).bitcast(fx.Float32)
        g = task // 2
        for w in range_constexpr(8):
            word = fp8_pack4(*[clamp_fp8(vals[4 * w + i] * inv) for i in range(4)])
            c["put_words"](c["smq"], t * (sh // 4) + g * 8 + w, [word])
        c["put_words"](c["smc"], t * (sh // 32) + g, [code])


# ---------------------------------------------------------------- routed ug
@traced
def gemv_fp4_pair(c, w, ws, e, n_rows, k, rg0, ks, xl, xsl, scale_cols, col):
    """Tiles rg0, rg0 + 1 (one 32-row scale block) of expert e's preshuffled MXFP4 weight
    [n_rows, k] times 16 MXFP8 LDS rows over K slice ``ks`` -> both tiles' partial C (8
    values, LDS row ``col``). Each B operand read from LDS feeds both tiles' MFMAs."""
    lane, wave = c["lane"], c["wave"]
    sl = c["wsl"] + wave * wsl_words(c["d"])
    slices = c["rs"].ug_k_slices
    rounds = ug_rounds(k, ks, slices)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    # rounds whose loads are in flight past the one computing: a quarter's
    # 2-step rounds two
    ahead = slices // 2
    ops = [
        fp4_pair_loads(c, w, ws, e, n_rows, k, rg0, scale_cols, *rounds[q])
        for q in range(min(ahead, len(rounds)))
    ]
    for r in range_constexpr(len(rounds)):
        sc0, st0, n, sc_steps = rounds[r]
        sv, avs = ops[r]
        fx.ptr_store(sv, sl + lane * (sc_steps // 2))
        sas = [
            [
                lds_u8(
                    sl,
                    scale_index(t * 16 + lane % 16, (st0 + i) * 4 + lane // 16, scale_cols) - 128 * sc0,
                )
                for i in range(n)
            ]
            for t in range(2)
        ]
        accs = fp4_pair_mfma(c, k, xl, xsl, st0, n, avs, sas, accs, col)
        if const_expr(r + ahead < len(rounds)):
            ops.append(fp4_pair_loads(c, w, ws, e, n_rows, k, rg0, scale_cols, *rounds[r + ahead]))
    return fx.Vector.from_elements([accs[t][q] for t in range(2) for q in range(4)], fx.Float32)


def ug_rounds(k, ks, slices):
    """K slice ``ks``'s rounds (scale step, step, steps, scale steps); ``ks`` traced. Halves:
    8-step scale blocks 2 r + ks, an odd block split 4 + 4. Quarters: steps 8 i + 2 ks, + 1
    of block i, a round loading only their scales (2 steps' share a dword)."""
    blocks = k // 128 // 8
    if slices == 4:
        return [(fx.Int32(8 * i) + ks * 2,) * 2 + (2, 2) for i in range(blocks)]
    assert slices == 2
    rounds = [(fx.Int32(16 * r) + ks * 8, fx.Int32(16 * r) + ks * 8, 8, 8) for r in range(blocks // 2)]
    if blocks % 2:
        last = (blocks - 1) * 8
        rounds.append((fx.Int32(last), fx.Int32(last) + ks * 4, 4, 8))
    return rounds


def fp4_pair_loads(c, w, ws, e, n_rows, k, rg0, scale_cols, sc0, st0, n, sc_steps):
    """``gemv_fp4_pair``'s loads of a round (loads only): the scales of steps sc0 .. sc0 +
    sc_steps (128 bytes a step, sc_steps / 2 dwords a lane) and both tiles' weights of steps
    st0 .. st0 + n."""
    lane = c["lane"]
    block = (e * n_rows + rg0 * 16) // 32 * (scale_cols * 8)
    words = sc_steps // 2
    sv = bo.buffer_load(
        rsrc(ws),
        block + 32 * sc0 + lane * words,
        vec_width=words,
        dtype=T.i32,
        cache_modifier=CM_NT,
    )
    sv = fx.Vector(sv) if words > 1 else fx.Int32(sv)
    return sv, [fp4_tile_weights(c, w, e, n_rows, k, rg0 + t, st0, n) for t in range(2)]


def scale_block_loads(c, ws, first_row, scale_cols):
    """The loads of the 32-row e8m0 block holding ``first_row``: contiguous in aiter's
    shuffle_scale order, so a few 16-byte loads a lane (not a byte load a K step, issue
    bound). A lane past the block re-reads its last 16 bytes."""
    lane = c["lane"]
    words = scale_cols * 8
    block = first_row // 32 * words
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(ws),
                block + fx.min(256 * i + lane * 4, words - 4),
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for i in range((words + 255) // 256)
    ]


def down_scale_words(d) -> int:
    """A down tile's 32-row scale block in LDS: ``scale_block_store`` writes
    whole 256-word rounds."""
    return down_lds_words(d.down_scale_cols * 8)


def wsl_words(d) -> int:
    """A wave's LDS scale words: an ug batch's 256, or a down round's blocks --
    a (task, slot) each, ``DOWN_R_MAX`` at most (``stage_down``)."""
    return max(256, DOWN_R_MAX * down_scale_words(d))


def scale_block_store(c, svs, sl):
    """``scale_block_loads``' words -> this wave's LDS block ``sl``
    (``down_scale_words``; a lane past the block writes past it, inside it)."""
    lane = c["lane"]
    for i in range_constexpr(len(svs)):
        fx.ptr_store(svs[i], sl + (256 * i + lane * 4))


def lds_scales(sl, local_row, lane, st0, blk, scale_cols):
    """K steps st0 .. st0 + blk's scale bytes of this lane's row (``local_row``) from a
    32-row scale block in LDS. An even step and the next share a word (``scale_index``'s d):
    a read a pair, not a step."""
    out, word = [], None
    for i in range(blk):
        idx = scale_index(local_row, (st0 + i) * 4 + lane // 16, scale_cols)
        if (st0 + i) % 2 == 0 or word is None:
            word = fx.ptr_load(sl + idx // 4)
        out.append((word >> (idx % 4 * 8)) & 0xFF)
    return out


def lds_u8(ptr, i):
    return (fx.ptr_load(ptr + i // 4) >> (i % 4 * 8)) & 0xFF


def fp4_tile_weights(c, w, e, n_rows, k, rg, st0, blk, k_real=None):
    """The weight operands of expert e's 16-row tile rg, K steps st0 .. st0 + blk (loads
    only). ``k_real``: a lane whose 32 columns lie past it (zero weights) reads its row
    group's first lane's address instead (no extra traffic) and holds zero."""
    lane = c["lane"]
    base = e * n_rows * (k // 2) // 4  # dwords: expert e
    out = []
    for i in range_constexpr(blk):
        past = k_real is not None and ((st0 + i) * 128 + 96) >= k_real
        if const_expr(past):
            real = (st0 + i) * 128 + lane // 16 * 32 < k_real
            at = real.select(lane * 4, lane % 16 * 4)
        else:
            at = lane * 4
        v = fp4_tile_load(rsrc(w), base, rg, k, st0 + i, at)
        if const_expr(past):
            v = fx.Vector.from_elements([real.select(v[q], fx.Int32(0)) for q in range(4)], fx.Int32)
        out.append(v)
    return out


def fp4_pair_mfma(c, k, xl, xsl, st0, n, avs, sas, accs, col):
    """``accs`` (two tiles') chained through ``n`` K steps from st0 against MXFP8
    rows in LDS (column lane % 16: row ``col``): a step's B operand loaded once
    for both tiles' MFMAs."""
    lane = c["lane"]
    x_row = lds_row(k // 4)
    s_row = lds_row(k // 32)
    j = lane // 16
    accs = list(accs)
    for i in range_constexpr(n):
        bv = fp8_operand(
            [
                fx.ptr_load(
                    xl + (col * x_row + (st0 + i) * 32 + 16 * h + j * 4),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                for h in range(2)
            ]
        )
        sb = fx.ptr_load(xsl + (col * s_row + (st0 + i) * 4 + j))
        for t in range_constexpr(2):
            accs[t] = mfma_scaled(avs[t][i], bv, accs[t], sas[t][i], sb, FP4, FP8)
    return accs


def fp8_operand(halves):
    """A lane's MXFP8 MFMA operand of a K step: its two 16-byte halves (K
    16 j .. and 64 + 16 j .. of the step, j = lane / 16)."""
    return fx.Vector.from_elements(
        [fx.Vector(h).bitcast(fx.Int32)[d] for h in halves for d in range(4)],
        fx.Int32,
    )


def down_lds_words(n):
    """A down wave's LDS block of n words: ``scale_block_store`` writes whole
    256-word rounds."""
    return -(-n // 256) * 256


def st_dev(v, region, i):
    """Word i of scratch ``region`` (plain words, not tagged pairs) at device
    scope: past the per-XCD L2, so another CTA reads it after the flag."""
    bo.buffer_store(v, rsrc(region.value), i, cache_modifier=CM_DEV)


@traced
def take_ug_task(c):
    """This CTA's next ug task past the first round, from the launch's queue (``ugq``, a
    word a tag): CTAs busy with shared / router / route tasks come late and take fewer. A
    task's result is the same whichever CTA runs it."""
    if c["tid"] == 0:
        at = c["ugq"].value + fx.Int64(c["tag"] * 4)
        ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), at)
        task = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        fx.ptr_store(fx.Int32(task), c["tab"] + c["rs"].queue0)
    gpu.barrier()
    return fx.ptr_load(c["tab"] + c["rs"].queue0)


@traced
def stage_ug(c, task):
    """One token tile: slot task // ug_parts of U, ``ug_groups`` FP4 groups of its
    intermediate (``ug_wave_pair``), then silu x up in bf16 and the MXFP8 requant -> MID.
    Past one token tile: ``stage_ug_tile``."""
    s, tid, lane, tab = c["S"], c["tid"], c["lane"], c["tab"]
    rs = c["rs"]
    ug_parts = rs.ug_parts
    u = task // ug_parts
    if u < c["rs"].n_union(tab, s):
        e = c["rs"].union_expert(tab, s, u)
        # the slot's picks, a column each (a column past them repeats the last);
        # every token's row is in LDS: a column reads its pick's token's
        first = c["rs"].slot_first(tab, u)
        count = c["rs"].slot_first(tab, u + 1) - first
        pick = c["rs"].pick_at(tab, first + fx.min(lane % 16, count - 1))
        ug_pass(c, e, task % ug_parts * rs.ug_groups, first, count, pick // rs.topk)
        publish(c["put"], c["ugf"], task, 1, tid == 0)
    gpu.barrier()


def ug_wave_pair(c, part0):
    """This wave's share of an ug task from group ``part0``: (group p, w13's gate tile 2 b
    with up tile 2 b + 1 interleaved by aiter's GUGU, K slice ks, whether the group is in
    the real width); b = 2 (part0 + p) + wave % 2."""
    wave, slices = c["wave"], c["rs"].ug_k_slices
    p = wave // (2 * slices)
    b = (part0 + p) * 2 + wave % 2
    return p, 2 * b, wave // 2 % slices, part0 + p < c["d"].inter_real // UG_PART


def ug_red_slot(rs, p, bb, ks, t):
    """RED's 64-lane slot of an ug task's partial C: group p, half bb, K slice
    ks, gate (t 0) or up (t 1)."""
    return ((p * 2 + bb) * rs.ug_k_slices + ks) * 2 + t


def pairwise_sum(vs):
    """Vs summed as a balanced tree, neighbours first: (v0 + v1) + (v2 + v3)."""
    while len(vs) > 1:
        vs = [vs[i] + vs[i + 1] for i in range(0, len(vs), 2)]
    return vs[0]


def ug_tile_task(c, task):
    """Past one token tile, ug task ``task``: its slot's expert, its first
    group, and its pick tile's first pick and the picks from it."""
    tab, rs, s = c["tab"], c["rs"], c["S"]
    ug_parts = rs.ug_parts
    tile = task // ug_parts
    u = rs.tile_slot(tab, tile)
    first = mplan.tile_pick0(rs.slot_first(tab, u), tile, rs.tile_first(tab, u))
    count = rs.slot_first(tab, u + 1) - first
    return rs.union_expert(tab, s, u), task % ug_parts * rs.ug_groups, first, count


@traced
def stage_ug_tile(c, task, total):
    """Past one token tile: ug task ``task`` of ``total`` (a pick tile of its slot), then
    the next from the queue (``take_ug_task``). The next task's weights are not prefetched:
    held across the epilogue and gather they spill the merged K2."""
    tid, lane = c["tid"], c["lane"]
    e, part0, first, count = ug_tile_task(c, task)
    gather_x8r(c, first, count)
    ug_pass(c, e, part0, first, count, fx.min(lane % 16, count - 1))
    nxt = (total > BLOCKS).select(take_ug_task(c) + BLOCKS, total)
    # the epilogue's MID stores done before the flag
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](c["ugf"], task, 1)
    return nxt


@traced
def ug_pass(c, e, part0, first, count, col):
    """An ug task's GEMV and epilogue for the slot's picks first .. first +
    min(count, 16) (column lane % 16's x in LDS row ``col``)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a, d, rs = c["args"], c["d"], c["rs"]
    p, rg0, ks, real = ug_wave_pair(c, part0)
    # a group past the real width (the padding to inter) has zero weights:
    # its GEMV is zero unread, and the epilogue still writes its zero MID
    acc = fx.Vector.filled(8, 0.0, fx.Float32)
    if real:
        acc = gemv_fp4_pair(
            c,
            a["w13"],
            a["w13_s"],
            e,
            2 * d.inter,
            HIDDEN,
            rg0,
            ks,
            c["xl"],
            c["xsl"],
            HIDDEN // 32,
            col,
        )
    for t in range_constexpr(2):
        slot = ug_red_slot(rs, p, wave % 2, ks, t)
        fx.ptr_store(
            fx.Vector.from_elements([acc[4 * t + q] for q in range(4)], fx.Float32),
            red + (slot * 64 + lane) * 4,
        )
    gpu.barrier()
    # thread (p, j, i): group p, the slot's pick j, intermediate column
    # 32 part + i, a pass of THREADS at a time; a group's 32 threads stay
    # lane-aligned
    cols = rs.tile_picks
    for pas in range_constexpr(0, rs.ug_groups * cols * UG_PART, THREADS):
        ug_epilogue(c, part0, pas + tid, first, count)


@traced
def load_x8_rows(c, words_mb, codes_mb, xl, xsl, token_of, live):
    """MXFP8 rows ``token_of(j)``, j < ``live`` (<= TILE), of ``words_mb`` / ``codes_mb``
    into LDS rows j of ``xl`` / ``xsl``: wait on each token's X8RDY, then plain 16-byte
    loads (half the bytes of tagged pairs)."""
    tid = c["tid"]
    if tid < TILE:
        c["poll"]([(c["x8rdy"], token_of(fx.min(tid, live - 1)), 1)])
    gpu.barrier()
    for mb, width, dst in ((words_mb, X8_WORDS, xl), (codes_mb, HIDDEN // 32, xsl)):
        quads = width // 4
        for i in range_constexpr(-(-TILE * quads // THREADS)):
            q = tid + THREADS * i
            row = q // quads
            if (q < TILE * quads) & (row < live):
                v = fx.Vector(
                    bo.buffer_load(
                        rsrc(c[mb].value),
                        token_of(row) * width + q % quads * 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                fx.ptr_store(v, dst + (row * lds_row(width) + q % quads * 4))
    gpu.barrier()


def gather_x8r(c, first, count):
    """The X8R rows and codes of the slot's picks first .. first + min(count, 16)
    into LDS rows 0 .. (only those rows: an ug pass reads a pick's row, the
    columns past them repeat the last one's)."""
    tab, topk = c["tab"], c["rs"].topk

    def token(j):
        return c["rs"].pick_at(tab, first + fx.min(j, count - 1)) // topk

    load_x8_rows(c, "x8r", "x8rs", c["x8rl"], c["x8rsl"], token, fx.min(count, TILE))


@traced
def ug_epilogue(c, part0, idx, first, count):
    """Thread ``idx``'s value of an ug task: silu x up of its (group, pick j,
    column), the group's MXFP8 scale, its byte of MID (the pick's row, ``first``
    + j in the pick order)."""
    lane, red, d, rs = c["lane"], c["red"], c["d"], c["rs"]
    cols = rs.tile_picks
    if idx < rs.ug_groups * cols * UG_PART:
        p = idx // (cols * UG_PART)
        part = part0 + p
        t = idx % (cols * UG_PART) // UG_PART
        i = idx % UG_PART
        tile = i // 16
        r = i % 16
        at = (16 * (r // 4) + t) * 4 + r % 4
        g, up = [
            pairwise_sum(
                [
                    fx.ptr_load(red + ug_red_slot(rs, p, tile, ks, t) * 64 * 4 + at)
                    for ks in range(rs.ug_k_slices)
                ]
            )
            for t in range(2)
        ]
        gc = fx.min(g, fx.Float32(LIMIT))
        uc = fx.max(fx.min(up, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
        sig = hw_rcp(fx.Float32(1.0) + hw_exp2(gc * fx.Float32(-LOG2E)))
        # the stage-1 kernel stores the intermediate as bf16; the stage-2 input
        # quant (fused_mx_quant_moe_sort) makes it MXFP8: amax floored 1e-10,
        # the ceil code, x / scale clamped, RNE
        v = bf16_round(gc * sig * uc)
        amax = fx.max(butterfly(abs(v), (1, 2, 4, 8, 16), fx.max), fx.Float32(1e-10))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        q8 = clamp_fp8(v * rcp_pow2(code))
        nb = [
            lane_gather(q8.bitcast(fx.Int32), fx.min(lane + q, 63)).bitcast(fx.Float32) for q in range(1, 4)
        ]
        # 4 threads' bytes -> one word
        word = fp8_pack4(q8, *nb)
        row = first + t
        if (i % 4 == 0) & (t < count):
            st_dev(word, c["mid"], row * d.mid_words + part * 8 + i // 4)
        if (i == 0) & (t < count):
            st_dev(code, c["mids"], mid_code_index(d, row, part))


# ---------------------------------------------------------------- down
@traced
def stage_down(c, tasks):
    """Rows 16 task .. of the hidden output for each of ``tasks`` (one or two, run
    together): routed experts of U (a wave each, round robin), the shared expert, their
    sum -> pushed to every rank."""
    s, wave, tab, d = c["S"], c["wave"], c["tab"], c["d"]
    n = len(tasks)
    nu = c["rs"].n_union(tab, s)
    sl = c["wsl"] + wave * wsl_words(d)
    sw = down_scale_words(d)
    # slots of U a round: the loads of a round stay the same
    slots = DOWN_R_MAX // n
    per_round = WAVES * slots
    # the shared expert's down projection first: its input is ready long
    # before the routed results, which a CTA mostly waits on here
    down_shared(c, tasks)
    for rnd in range(0, (nu + per_round - 1) // per_round, 1):
        # the round's slots of U (wave, wave + WAVES, ...): every weight load in flight
        # before the first wait, then MID rows straight into registers (``mid_operands``);
        # a slot past U reloads the last slot (L2 hits) and contributes nothing
        us = [wave + WAVES * (slots * rnd + i) for i in range(slots)]
        ucs = [fx.min(u, nu - 1) for u in us]
        firsts = [c["rs"].slot_first(tab, u) for u in ucs]
        # only this round's slots' ug flags: the round starts while later slots' ug run
        # infera-flash: the flags' first loads issue before the weights' (``poll_under``)
        loads = poll_under(c, down_ug_flags(c, ucs), down_weights, c, tasks, rnd, slots, nu)
        svs = loads[0]
        avs = loads[1]
        # infera-flash: timeline point 27 = a round's ug flags seen, 28 = the rounds done
        down_stamp(c, 27)
        counts = [c["rs"].slot_first(tab, u + 1) - f for u, f in zip(ucs, firsts)]
        mids = [mid_operands(c, f, k) for f, k in zip(firsts, counts)]
        for q in range_constexpr(len(svs)):
            scale_block_store(c, svs[q], sl + q * sw)
        for i in range_constexpr(slots):
            for j in range_constexpr(n):
                down_mfma(
                    c,
                    j,
                    tasks[j],
                    sl + (j * slots + i) * sw,
                    mids[i],
                    firsts[i],
                    counts[i],
                    avs[j * slots + i],
                    us[i] < nu,
                )
            # a slot of more than 16 picks: its next pick tile's rows
            if const_expr(s > TILE):
                for j0 in range_constexpr(TILE, s, TILE):
                    if j0 < counts[i]:
                        more = mid_operands(c, firsts[i] + j0, counts[i] - j0)
                        for j in range_constexpr(n):
                            down_mfma(
                                c,
                                j,
                                tasks[j],
                                sl + (j * slots + i) * sw,
                                more,
                                firsts[i] + j0,
                                counts[i] - j0,
                                avs[j * slots + i],
                                us[i] < nu,
                            )
    gpu.barrier()
    down_stamp(c, 28)
    for t0, nn in token_tiles(s):
        for j in range_constexpr(n):
            down_combine(c, tasks[j], j, t0, nn)


@traced
def poll_under(c, flags, issue, *args):
    """infera-flash: ug flags polled with ``issue(*args)``'s loads between their first load and check.
    Loads complete in issue order, so flags polled after the weights would wait on every one; the
    first check stays outside the spin loop, which would merge it with the reloads."""
    tag = c["ptag"]

    def tags():
        return [
            fx.Int32(
                bo.buffer_load(
                    rsrc(c["ugf"].value),
                    fx.Int32(f) * 2 + 1,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            for f in flags
        ]

    def stale(ts):
        bad = ts[0] != tag
        for t in ts[1:]:
            bad = bad | (t != tag)
        return bad

    first = tags()
    out = issue(*args)
    bad = stale(first)
    while bad:
        rocdl.s_sleep(SPIN_SLEEP)
        bad = stale(tags())
    return out


def down_stamp(c, point):
    """infera-flash: a K2 timeline build's global stamp (none otherwise)."""
    if c.get("tl") is not None:
        from ..attention.device import stamp as global_stamp

        global_stamp(c, point)


@traced
def take_down_rank(c):
    """This CTA's arrival rank at the down stage (DQ, a word a tag): which
    CTAs take the extra tiles. A tile's result is the same whichever CTA
    computes it."""
    if c["tid"] == 0:
        at = c["dq"].value + fx.Int64(c["tag"] * 4)
        ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), at)
        rank = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        fx.ptr_store(fx.Int32(rank), c["tab"] + c["rs"].queue0)
    gpu.barrier()
    return fx.ptr_load(c["tab"] + c["rs"].queue0)


def down_weights(c, tasks, rnd, slots, nu):
    """Round ``rnd``'s weights of ``stage_down``: each (task, slot)'s scale block
    and weight operands, this wave's slots of U (a slot past U: the last's).
    Loads only."""
    s, wave, tab, a, d = c["S"], c["wave"], c["tab"], c["args"], c["d"]
    us = [wave + WAVES * (slots * rnd + i) for i in range(slots)]
    es = [c["rs"].union_expert(tab, s, fx.min(u, nu - 1)) for u in us]
    svs = [
        scale_block_loads(c, a["w2_s"], e * HIDDEN + task * 16, d.down_scale_cols)
        for task in tasks
        for e in es
    ]
    avs = [
        fp4_tile_weights(c, a["w2"], e, HIDDEN, d.inter, task, 0, d.inter // 128, d.inter_real)
        for task in tasks
        for e in es
    ]
    return svs, avs


def mid_operands(c, first, count):
    """The MFMA B operands of picks first .. first + min(count, 16) (column lane % 16): each
    K step's two 16-byte halves of the lane's MID row and its code, loaded at device scope
    straight into registers (no LDS block a slot, so a round holds more slots)."""
    lane, d = c["lane"], c["d"]
    j = lane // 16
    row = first + fx.min(lane % 16, count - 1)
    steps = d.inter // 128
    # the step's constant after an opaque lane word folds into the immediate
    # offset; summed with j * 4 it would be a hoisted VGPR a step (spill)
    lane_word = fresh(row * d.mid_words + j * 4)
    bvs = []
    for st in range_constexpr(steps):
        halves = []
        for h in range_constexpr(2):
            v = fx.Vector(
                bo.buffer_load(
                    rsrc(c["mid"].value),
                    lane_word + (st * 32 + 16 * h),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            if const_expr(st * 128 + 64 * h + 48 >= d.inter_real):
                # columns past the rank's real intermediate (the loader pads
                # it with zero weights) are never written: stale bytes, which
                # a NaN pattern would carry through the zero weights
                past = st * 128 + 64 * h + 16 * j >= d.inter_real
                v = fx.Vector.from_elements([past.select(fx.Int32(0), v[q]) for q in range(4)], fx.Int32)
            halves.append(v)
        bvs.append(fp8_operand(halves))
    # this lane's codes, one a K step: contiguous in MIDS (``mid_code_index``),
    # 4 a load, since the stage is load-issue bound
    base = mid_code_index(d, row, j)
    sbs = []
    for w0 in range_constexpr(0, steps, 4):
        n = min(4, steps - w0)
        v = bo.buffer_load(
            rsrc(c["mids"].value),
            base + w0,
            vec_width=n,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
        if const_expr(n == 1):
            sbs.append(fx.Int32(v))
        else:
            v = fx.Vector(v)
            sbs += [fx.Int32(v[k]) for k in range(n)]
    for st in range_constexpr(steps):
        if const_expr((st + 1) * 128 > d.inter_real):
            # a group past the real intermediate has no code written either
            past = (4 * st + j) * 32 >= d.inter_real
            sbs[st] = past.select(fx.Int32(UNIT_SCALE), sbs[st])
    return bvs, sbs


def mid_code_index(d, row, group):
    """MIDS word of pick row ``row``'s MXFP8 group ``group`` (32 columns): a K
    step's 4 groups apart, so the steps of one group % 4 lie together
    (``mid_operands``' lane j reads groups j, j + 4, ..)."""
    steps = d.inter // 128
    return row * (d.inter // 32) + group % 4 * steps + group // 4


def down_ug_flags(c, ucs):
    """This lane's ug flags of a down round's slots ``ucs``: flag f a (slot, pick tile past
    one token tile, part), a tile past the slot's repeating its last; lane l takes flags l,
    l + 64, .."""
    rs = c["rs"]
    ug_parts = rs.ug_parts
    per_slot = rs.slot_tiles * ug_parts if rs.multi else ug_parts
    n = len(ucs) * per_slot
    return [_down_ug_flag(c, ucs, fx.min(c["lane"] + 64 * j, n - 1), per_slot) for j in range(-(-n // 64))]


def _down_ug_flag(c, ucs, f, per_slot):
    rs, tab = c["rs"], c["tab"]
    ug_parts = rs.ug_parts
    mine = ucs[0]
    for i in range_constexpr(1, len(ucs)):
        mine = (f // per_slot == i).select(ucs[i], mine)
    if const_expr(not rs.multi):
        return mplan.ug_flag(mine, f % ug_parts, ug_parts)
    tile0 = rs.tile_first(tab, mine)
    tiles = rs.tile_first(tab, mine + 1) - tile0
    return mplan.ug_flag(
        mplan.polled_tile(tile0, tiles, f % per_slot // ug_parts),
        f % ug_parts,
        ug_parts,
    )


@traced
def down_shared(c, tasks):
    """Each of ``tasks``' rows of the shared expert's down projection (its MXFP8
    intermediate, ``load_smid_tile``) for every token, bf16 -> LDS
    ``sdl[task][token][16 rows]``; every task's weights in flight first."""
    s, tid, red, d = c["S"], c["tid"], c["red"], c["d"]
    a = c["args"]
    shops = [gemv_fp8_loads(c, a["sw2"], a["sw2_s"], d.sh_inter, task, row_major=True) for task in tasks]
    for t0, nn in token_tiles(s):
        load_smid_tile(c, t0, nn, d.sh_inter // 32)
        for j in range_constexpr(len(tasks)):
            gemv_fp8_mfmas(c, d.sh_inter, c["shared_xl"], c["shared_xsl"], red, shops[j], rows=nn)
            gpu.barrier()
            if tid < ROWS * nn:
                v = row_sum(red, tid % ROWS, tid // ROWS)
                at = (j * s + plus(t0, tid // ROWS)) * ROWS + tid % ROWS
                fx.ptr_store(bf16_round(v).bitcast(fx.Int32), c["sdl"] + at)
            gpu.barrier()


@traced
def down_combine(c, task, j, t0, n):
    """Down task ``task`` (the CTA's j-th) for tokens t0 .. t0 + n: the routed
    contributions summed in bf16 in each token's top-k order, plus the shared expert's
    (``down_shared``), the sum pushed to every rank."""
    s, tid, d, tab = c["S"], c["tid"], c["d"], c["tab"]
    topk = c["rs"].topk
    if tid < ROWS * n:
        r = tid % ROWS
        tl = tid // ROWS
        t = plus(t0, tl)
        shared = fx.ptr_load(c["sdl"] + ((j * s + t) * ROWS + r)).bitcast(fx.Float32)
        # aiter's stage 2 adds each bf16(v * w) into a bf16 buffer: a rounding
        # at every add (its order is the atomics'; here top-k order)
        routed = fx.Float32(0.0)
        for k in range_constexpr(topk):
            w = c["rs"].route_w(tab, s, t, k)
            v = contrib_load(c, j, (t * topk + k) * 16 + r)
            routed = bf16_round(routed + bf16_round(v * w))
        fx.ptr_store(bf16_round(routed + shared), c["pair"] + tid)
    gpu.barrier()
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        v0 = fx.ptr_load(c["pair"] + (tl * ROWS + 2 * rp))
        v1 = fx.ptr_load(c["pair"] + (tl * ROWS + 2 * rp + 1))
        col = task * ROWS + 2 * rp
        for p in range_constexpr(d.tp):
            dst = preg(c["peer_addr"](p), c["ffn_off"], "ffn")
            c["put_bf"](dst, 2 * region_pair(c["rank"], t, s, col), [v0, v1])
    gpu.barrier()


@traced
def down_mfma(c, j, task, sl, mid, first, count, av, live):
    """Down task ``task`` (the CTA's j-th) on a slot of U: the routed MFMA of its
    picks first .. first + min(count, 16) (their B operands ``mid``:
    ``mid_operands``; the scale block in ``sl``), each pick's contribution."""
    lane, tab, d = c["lane"], c["tab"], c["d"]
    sas = lds_scales(sl, task % 2 * 16 + lane % 16, lane, 0, d.inter // 128, d.down_scale_cols)
    jj = lane % 16
    col = fx.min(jj, count - 1)
    bvs, sbs = mid
    # the K steps on DOWN_CHAINS accumulators in turn, summed in order: one
    # chain put every FP8-rate MFMA's latency in series
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(DOWN_CHAINS)]
    for i in range_constexpr(len(av)):
        h = i % DOWN_CHAINS
        accs[h] = mfma_scaled(av[i], bvs[i], accs[h], sas[i], sbs[i], FP4, FP8)
    acc = accs[0]
    for h in range_constexpr(1, DOWN_CHAINS):
        acc = acc + accs[h]
    pick = c["rs"].pick_at(tab, first + col)
    for q in range_constexpr(4):
        r = 4 * (lane // 16) + q
        if live & (jj < count):
            contrib_store(c, j, pick * 16 + r, acc[q])


def contrib_store(c, j, i, v):
    """Task j's routed contribution word i (``[pick (t, k)][16 rows]`` f32 bits)."""
    fx.ptr_store(v.bitcast(fx.Int32), c["contrib"] + (j * c["rs"].picks * 16 + i))


def contrib_load(c, j, i):
    return fx.ptr_load(c["contrib"] + (j * c["rs"].picks * 16 + i)).bitcast(fx.Float32)


# ---------------------------------------------------------------- loaders


@traced
def load_smid_tile(c, t0, n, groups):
    """The shared expert's MXFP8 intermediate of tokens t0 .. t0 + n (SMQ /
    SMC, ``smid_quant``) into LDS ``shared_xl`` / ``shared_xsl`` rows 0 .., by every
    thread."""
    tid, words = c["tid"], c["d"].sh_inter // 4
    poll_copy(
        c,
        tid,
        THREADS,
        [
            (c["smq"], lambda u: plus(t0 * words, u), n * words, c["shared_xl"], words),
            (
                c["smc"],
                lambda u: plus(t0 * groups, u),
                n * groups,
                c["shared_xsl"],
                groups,
            ),
        ],
    )
    gpu.barrier()


# ---------------------------------------------------------------- reduce
@traced
def stage_reduce(c, task):
    """Columns 32 task ..: the TP ranks' partials in rank order (``sum_partials``), in
    fp32 -> bf16 -> ``out``."""
    s, tid = c["S"], c["tid"]
    for pas in range_constexpr(0, s * RED_COLS // 2, THREADS):
        reduce_pair(c, task, pas + tid)


@traced
def reduce_pair(c, task, idx):
    """Thread ``idx``'s column pair of token idx / 16."""
    s, tp = c["S"], c["d"].tp
    a = c["args"]
    if idx < s * RED_COLS // 2:
        t = idx // (RED_COLS // 2)
        col = task * RED_COLS + 2 * (idx % (RED_COLS // 2))
        own = preg(c["sym"], c["ffn_off"], "ffn")
        acc0, acc1 = sum_partials(c["poll"], own, lambda src: region_pair(src, t, s, col), tp)
        bo.buffer_store(acc0.to(fx.BFloat16), rsrc(a["out"]), t * HIDDEN + col)
        bo.buffer_store(acc1.to(fx.BFloat16), rsrc(a["out"]), t * HIDDEN + col + 1)


def moe_smem(s, rs, timeline):
    """The MoE stages' LDS (``timeline``: with its own stamp record)."""

    @fx.struct
    class Smem:
        # an ug task's partial C: 2 tiles a wave (``ug_red_slot``)
        red: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]
        tab: fx.Array[fx.Int32, rs.tab_words, 16]
        pair: fx.Array[fx.Float32, ROWS * s, 16]
        # one stage's buffers at a time (``stage_lds``)
        stage: fx.Array[fx.Int32, sum(stage_lds(s, rs)["words"].values()), 16]
        tls: fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16]

    return Smem


def stage_lds(s, rs):
    """Word offsets into ``Smem.stage`` of the buffers only one stage reads (shared: its x
    tile; ug: the waves' scale blocks and routed x tile; down: scale blocks, the shared
    intermediate tile, contributions, shared rows). A barrier ends every stage."""
    d = rs.dims
    # the x tile the shared / ug stages hold (every token's up to one tile)
    rows = tile_rows(s)
    x8 = rows * lds_row(X8_WORDS)
    groups = rows * lds_row(HIDDEN // 32)
    wsl = WAVES * wsl_words(d)
    shared_xl = rows * lds_row(d.sh_inter // 4) + lds_tail(d.sh_inter)
    shared_xsl = rows * lds_row(d.sh_inter // 32)
    contrib = 2 * rs.picks * 16  # two down tasks' at most
    offsets = {
        "x8l": 0,
        "x8sl": x8,
        "wsl": 0,
        "x8rl": wsl,
        "x8rsl": wsl + x8,
        "shared_xl": wsl,
        "shared_xsl": wsl + shared_xl,
        "contrib": wsl + shared_xl + shared_xsl,
        "sdl": wsl + shared_xl + shared_xsl + contrib,
    }
    words = {
        "shared": x8 + groups,
        "ug": wsl + x8 + groups,
        "down": wsl + shared_xl + shared_xsl + contrib + 2 * s * ROWS,
    }
    return {"offsets": offsets, "words": {"all": max(words.values())}}


def moe_context(s, rs, lds, mb, bases, layout, scratch, args, rank, sym):
    """The MoE stages' context: LDS, mailbox, scratch regions, arguments. Its lane
    math starts from a ``fresh`` thread id: in K2 it is not the attention and
    seam stages', held from theirs on."""
    tid = fresh(fx.thread_idx.x)
    c = {
        "S": s,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "tab": lds.tab.ptr,
        "pair": lds.pair.ptr,
        "put": mb.put,
        "put_bf": mb.put_bf,
        "put_words": mb.put_words,
        "poll": mb.poll,
        "tag": mb.tag,
        "ptag": mb.tag,  # infera-flash: the pairs' tag (``poll_under``; layer.py sets ``tag`` to the epoch's slot)
        "peer_addr": lambda p: bases[p],
        "rank": rank,
        "sym": sym,
        "ffn_off": ffn_offset(s, rs.tp),
        "rs": rs,
        "d": rs.dims,
        "args": args,
        # x (``ffn_normed``) before this launch: plain loads, no readiness wait
        "x_cm": 0,
        "x_ready": False,
    }
    for name, off in stage_lds(s, rs)["offsets"].items():
        c[name] = lds.stage.ptr + off
    for region, (off, _) in layout.items():
        c[region] = sreg(scratch, off, region)
    return c


def _moe_stamp(c, timeline, tls, tid, k):
    # infera-flash: with a global timeline in the context (K2 timeline builds), its point 16 + k
    if c.get("tl") is not None:
        from ..attention.device import stamp as global_stamp

        global_stamp(c, 16 + k)
    else:
        stamp(timeline, tls, tid, k)


@traced
def run_moe(c, key, bid, tls, tl0):
    """The MoE stages, in order; ``timeline`` stamps from point ``tl0`` + 1."""
    s, timeline, tid = key.tokens, key.timeline, c["tid"]
    router_tasks = c["rs"].router_tasks
    route0, shared0 = c["rs"].route0, c["rs"].shared0
    shared_tasks = c["d"].shared_tasks
    ug_tasks = c["rs"].max_u * c["rs"].ug_parts
    for t in range(first_task(bid, XQ0), s, BLOCKS):
        await_x(c, [t])
        stage_xq(c, t)
        publish(c["put"], c["x8rdy"], t, 1, tid == 0)
    _moe_stamp(c, timeline, tls, tid, tl0 + 1)
    multi = len(token_tiles(s)) > 1
    for task in range(first_task(bid, c["rs"].router0), router_tasks, BLOCKS):
        await_x(c, list(range(s)))
        stage_router(c, task)
    _moe_stamp(c, timeline, tls, tid, tl0 + 2)
    for t in range(first_task(bid, route0), s, BLOCKS):
        stage_route(c, t)
    _moe_stamp(c, timeline, tls, tid, tl0 + 3)
    c["xl"], c["xsl"] = c["x8l"], c["x8sl"]
    # a CTA's one shared unit (task, token tile) at most: straight-line code, whose
    # tile loads LLVM does not hoist into one live set as out of a task loop (spills),
    # and no tiles in series on the path down waits on
    assert c["rs"].shared_units <= BLOCKS
    unit = first_task(bid, shared0)
    if unit < c["rs"].shared_units:
        # the gate weights in flight before x's wait
        a = c["args"]
        gops = gemv_fp8_loads(c, a["sgu"], a["sgu_s"], HIDDEN, unit % shared_tasks, row_major=True)
        if const_expr(not multi):
            load_x8_rows(c, "x8", "x8s", c["x8l"], c["x8sl"], lambda j: j, s)
            stage_shared(c, unit, gops, 0, s)
        else:
            t0 = unit // shared_tasks * TILE
            n = fx.min(fx.Int32(TILE), s - t0)
            load_x8_rows(c, "x8", "x8s", c["x8l"], c["x8sl"], lambda j: t0 + j, n)
            stage_shared(c, unit % shared_tasks, gops, t0, n)
    _moe_stamp(c, timeline, tls, tid, tl0 + 4)
    c["xl"], c["xsl"] = c["x8rl"], c["x8rsl"]
    if const_expr(multi):
        # past one token tile a task gathers its pick tile's rows itself; the first
        # round placed (the shared units' CTAs last), the rest from the queue, so a
        # CTA a shared or router task delayed takes fewer
        load_route(c, c["tab"])
        n_tiles = c["rs"].tile_first(c["tab"], c["rs"].n_union(c["tab"], s))
        ug_total = mplan.ug_tasks(n_tiles, c["rs"].ug_parts)
        task = first_task(bid, c["rs"].ug0)
        while task < ug_total:
            task = stage_ug_tile(c, task, ug_total)
    else:
        # one token tile: every task placed (its x in LDS once a CTA)
        loaded = fx.Int32(0)
        for task in range(first_task(bid, c["rs"].ug0), ug_tasks, BLOCKS):
            if loaded == 0:
                # X8R (ready long before the routing) first: after the routing
                # lands only its own poll remains
                load_x8_rows(c, "x8r", "x8rs", c["x8rl"], c["x8rsl"], lambda j: j, s)
                load_route(c, c["tab"])
            loaded = fx.Int32(1)
            stage_ug(c, task)
    _moe_stamp(c, timeline, tls, tid, tl0 + 5)
    assert BLOCKS < DOWN_TASKS <= 2 * BLOCKS
    # the route table is in LDS already past one token tile (every CTA loads it
    # for the ug queue) or when every CTA ran an ug iteration (more ug task
    # slots than CTAs), whose first loads it; else load it
    if const_expr(not multi and ug_tasks <= BLOCKS):
        load_route(c, c["tab"])
    # every CTA its tile; the first DOWN_TASKS - BLOCKS to get here an extra
    # one too: the CTAs whose ug ended first (by bid, those ending last held
    # two tiles and ended the stage)
    down_task = first_task(bid, DOWN0)
    rank = take_down_rank(c)
    if rank < DOWN_TASKS - BLOCKS:
        stage_down(c, [down_task, BLOCKS + rank])
    else:
        stage_down(c, [down_task])
    _moe_stamp(c, timeline, tls, tid, tl0 + 6)
    for task in range(first_task(bid, RED0), RED_TASKS, BLOCKS):
        stage_reduce(c, task)
    _moe_stamp(c, timeline, tls, tid, tl0 + 7)


def await_x(c, tokens):
    """In a launch that also writes x (K2: the FFN seam's norm): tokens' rows
    ready."""
    if const_expr(c["x_ready"]):
        c["poll"]([(c["xrdy"], t, 1) for t in tokens])


def moe_args(*values) -> dict:
    """The MoE stages' arguments by name, in ``ARGS`` order."""
    assert len(values) == len(ARGS)
    return dict(zip(ARGS, values))
