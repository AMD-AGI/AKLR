---
title: "MXFP4 fused MoE, hidden 6144, 257 experts, top-9, inter 256, 16×16-shuffled weights, at 8, 16, 32 and 64 tokens — the code of the kernel that folds the routing sort into stage 1: the sort rebuilt in LDS by every stage-1 workgroup, MXFP4 quantization of the gathered rows inside stage 1, an atomic stage 2"
kind: case
ops: [moe, fused_moe]
dtypes: [mxfp4]
regimes: ["tokens 8-64"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's fused MoE, per token count: one-sided Student-t lower bound of aiter's time over the kernel's, L2 flushed before each call, one run", value: "1.113, 1.079, 1.112, 1.094 at 8, 16, 32, 64 tokens", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 257 experts, top-9, inter 256; 8, 16, 32, 64 tokens", date: "2026-11"}
  - {metric: "lower-bound speedup over aiter's fused MoE, per token count, L2 flushed per call, a separate run", value: "1.122, 1.079, 1.099, 1.081 at 8, 16, 32, 64 tokens", gpu: gfx950, workload: "the same layer and token counts", date: "2026-10"}
  - {metric: "time of a call with the routing sort as a graph node of its own against the sort folded into stage 1, per-kernel CUDA-graph timer, one run per arm", value: "39.09 → 34.38, 53.20 → 49.68, 84.28 → 79.19, 102.04 → 95.67 µs at 8, 16, 32, 64 tokens", gpu: gfx950, workload: "the same layer and token counts", date: "2026-10"}
  - {metric: "what aiter's MXFP4 sort node costs in front of the folded kernel, differences paired per round", value: "3.9–6.0 µs on the per-kernel CUDA-graph timer, 2.8–5.8 µs back to back without a flush, at 8–64 tokens", gpu: MI355X, workload: "the same layer and token counts", date: "2026-11"}
  - {metric: "SQNR of the bf16 output between two runs on the same inputs", value: "48.4–51.9 dB over 8, 16, 32, 64 tokens", gpu: gfx950, workload: "the same layer and token counts, three input sets per token count", date: "2026-10"}
---

# MXFP4 fused MoE at 8–64 tokens with the routing sort folded into stage 1: the kernel's code (hidden 6144, 257 experts, top-9, inter 256)

This page shows the code of one kernel of the family that [fused_moe_shuffled_16x16](fused_moe_shuffled_16x16.md)
describes: the FlyDSL fused MoE with MXFP4 weights in aiter's 16×16 preshuffle and MXFP4 activations, hidden 6144, 257
experts, top-9, inter dim 256, built for 8, 16, 32 and 64 tokens. It runs two launches, one per GEMM: every stage-1
workgroup rebuilds in LDS the part of the routing sort it needs and the stage-1 grid zeroes the output, so no sort
launch runs. The family page has the layer's other kernels and the results per token count; it names this kernel the
257/256 fold kernel and the layer's kernel for every token count the 257/256 all-m kernel, and its evidence lines,
copied under § Evidence, use those names, with "m" for the tokens in the call, and report the family's other kernels
beside this one. [moe/few_tokens](../moe/few_tokens.md) explains the technique; § Code has the excerpts that it and
its FlyDSL side, `../flydsl/moe.md`, describe.

## Applies when

- **The op** is a fused MoE layer of this shape: `w1` (257, 512, 6144) and `w2` (257, 6144, 256) as fp4 e2m1 in
  aiter's 16×16 preshuffle, one E8M0 scale per 1×32, shuffled and padded to (256, 8); bf16 `hidden_states`, fp32
  `topk_weights`, int32 `topk_ids` with 9 distinct expert ids per token, no expert mask; SiLU on separate gate and up
  rows, activations quantized to MXFP4, the routing weight applied in stage 2; bf16 `out`. A routed item is one
  (token, expert) pair of `topk_ids`, a route on the family page.
- **The token counts** are 8, 16, 32 and 64, where a call lasts tens of µs and the routing sort would be a graph node
  of its own: 34.4–95.7 µs folded *(field, E1)*.
  The token count is a compile-time constant of stage 1, one compile per count, which suits a caller with a fixed
  list of token counts.
- **The sort fits one 256-thread workgroup**: 72–576 routed items in 1–3 passes of the threads, and one thread per
  expert in the scan with expert 256 counted apart, so up to 256 + 1 experts as written.
- **Stage 1 runs tile-M 16 with inline activation quantization**, and asserts both for the fold.
- **The baseline** is aiter's `fused_moe`, judged per token count; a token count counts as won from 1.03 over it.
  aiter's table picks its inline-quantizing `f16in` stage 1 for 4–128 tokens of this shape (S1).
- **Accuracy is bounded**, by one bound on the whole output: SQNR ≥ 13 dB against a reference computation of the
  layer in bf16 that quantizes the input rows and the intermediate to MXFP4 too, with no per-element bound. Two runs
  need not give the same bytes, so stage 2 may add with atomics.
- **The model.** The kernel was written for GLM-5.2's MXFP4 MoE (256 routed experts plus the fused shared one,
  top-9); inter 256 is that layer's intermediate 2048 split 8 ways.

## Counter-signals

- **Output bits differ from run to run.** Stage 2 adds a token's 9 weighted rows into `out` with bf16 atomics in
  arrival order; two runs on the same inputs read 48.4–51.9 dB SQNR against each other
  *(field, E2)*.
  Where two runs must give the same bytes, use a fixed-order stage 2 (§ Code, entry 7).
- **The kernel is not built for other token counts.** At 1 and 2 tokens the layer's all-m kernel runs two launches
  with no sort and reads 2.489 and 2.016 over aiter; at 4 and 128 it reads 0.137 and 0.314, so no kernel of the layer
  reaches 1.03 there *(field, E3)*.
  From 128 tokens the replicated sort grows with the 9 routed items per token and runs in every workgroup of a grid
  that grows with the token count (estimated).
- **The thinnest margin is at 16 tokens**: 1.079, 1.079 and 1.087 over aiter in the three readings of § Result, each
  of one run; later builds of the kernel read differently
  *(field, E3, E4, E5)*.
- **The head is repeated work.** Every stage-1 workgroup, those past the device bound included, loads the items,
  builds the histogram and runs step d's scan, 144–554 workgroups at 8–64 tokens; steps e–g run only in the
  workgroups below the device bound. The head's cost is not measured. HBM is idle during the head until step e names
  the expert; the early issue of the first weight stages (§ Mechanism) is the mitigation, and its effect is not
  measured either.
- **Read, not run.** That the implementation this kernel was ported from runs three graph nodes at these token counts
  is read from its source, not from a trace (S2).
  Stage 1's split-K exchange, n-fusion, item-direct head and k-rotation and stage 2's non-atomic epilogues are code
  that none of the four token counts reaches; no measurement covers them.
- **Where the fold loses on other kernels** (a sort node that is already one small kernel, a wide stage 1):
  `../moe/few_tokens.md`, Counter-signals.

## Mechanism

- **Two launches.** The implementation this kernel was ported from does not fold at these token counts: it calls
  aiter's fused MoE and runs three graph nodes, aiter's MXFP4 routing sort (per expert, padded to 16-row blocks; it
  also zeroes the output buffer), stage 1 and stage 2 (S2).
  The kernel drops the sort launch: stage 1 starts with a head that rebuilds the sort, and its grid zeroes `out`.
  What a graph node costs at a few tokens: `../moe/few_tokens.md` §1.
- **Stage 1**, compiled per token count: the head, then the gate/up GEMM on the block's 16 gathered hidden rows,
  which it quantizes to MXFP4 itself (§ Code, entry 6), SiLU(gate) · up, and the MXFP4 intermediate written in
  sorted-row order. One workgroup of 256 threads takes one m-block, 16 sorted rows of one expert, and one of the 2
  n-blocks of 128 gate + 128 up columns; the grid is an upper bound of 144, 288, 518 and 554 workgroups at 8, 16, 32
  and 64 tokens (§ Code, entry 5). Its K loop, rings and tile order:
  [fused_moe_shuffled_16x16](fused_moe_shuffled_16x16.md) § Mechanism, 4–128 tokens.
- **The head**, in the lettered steps of `../moe/few_tokens.md` (How to write it), which the entries of § Code refer
  to:
  - **a.** Zero `out` as int32 words, striped over the whole grid, surplus workgroups included.
  - **b.** Load the workgroup's `topk_ids` items and their routing weights into registers; a padding item reads −1.
  - **c.** Zero a 272-word LDS histogram while those loads are in flight, then one LDS atomic add per item.
  - **d.** Per expert, ⌈count / 16⌉ blocks; an inclusive DPP scan per wave, the four wave totals through LDS, an
    exclusive block offset per expert; expert 256 (the 257th) is placed last.
  - **e.** The thread whose expert owns the workgroup's m-block writes the expert id and the block's index within the
    expert to LDS.
  - **f.** Rank that expert's items in item order (a DPP scan of `item == expert`) and keep ranks [lo, lo + 16) as
    the block's rows: token ids, `(slot << 24) | token` ids and routing weights.
  - **g.** The workgroup of n-block 0 publishes the block's sorted ids, weights and expert id for stage 2; that of
    block 0 also the valid count.

  Steps a–d run before the bound test, in every workgroup. The device then computes the bound, step d's total of
  blocks × 2 n-blocks; a workgroup at or past it exits, and at 32 tokens the tile id is remapped per XCD only after
  that test (`../moe/grouped_sorted_tokens.md` §2). The first two weight and weight-scale stages of the GEMM are
  issued as soon as step e has the expert id, before the LDS-only steps f and g, so the head runs under the weight
  stream.
- **Stage 2**, compiled once for the four token counts: a port of aiter's `mxfp4_gemm2.py`. Per m-block, 24
  workgroups of 256 output columns take K = 256 in one tile, multiply each row by its routing weight and add it into
  `out` with bf16x2 atomics (§ Code, entry 7), on upper-bound grids of 1728–6648 workgroups whose surplus exits on the
  device (`../moe/grouped_sorted_tokens.md` §2, §6). Its tile remap, loads and kernel-argument preload: the same
  section of the family page.
- **Compiled resources.** One stage-1 compile per token count and one stage-2 compile, kept by `functools.cache`
  compile functions; the workspace is allocated per call. The sort scratch takes (352 + 256 · ⌈9M / 256⌉) words of LDS
  at M tokens, 2.4–4.4 KiB, beside the GEMM's stages.

## Code

### 1. The item loads, the zeroing of `out` and the expert histogram (steps a–c)

Steps a to c of the head, at the top of the stage-1 kernel. `tx_i32` is the thread id, `bx_i32` the workgroup id,
`_MT` the number of routed items and `_NREP` the passes of the workgroup's threads over them; `arg_moebuf` is the
address of `out`, `_NDW` its size in int32 words, `_GRID` the number of workgroups launched and `_NREP_Z` the passes
of the grid's threads over those words. The loads of step b are issued first, the zero stores of step a behind them,
and the histogram `cnt_tiles` is zeroed while the loads are in flight. A thread past the last item loads index 0, and
its id becomes −1 at its first use. `_TKWN` is a compile-time switch, on in this kernel: the routing weights are
loaded beside the ids, for step f.

```python
for _r in range_constexpr(_NREP):  # (b) issue the loads of this thread's items; a padding item reads index 0
    _i = tx_i32 + fx.Int32(_r * 256)
    _m = _ashr(_i - fx.Int32(_MT), 31)      # -1 iff _i < MT
    _tk_regs.append((_global_i32_at(arg_topk, _i & _m), _m))
    if const_expr(_TKWN):
        _tkw_regs.append(_global_i32_at(arg_topkw, _i & _m))
# (a) moe_buf zero-init, issued behind the topk loads
mb_tiles = _global_scalar_tiles(arg_moebuf, fx.Int32, _NDW)
_gtid = bx_i32 * fx.Int32(256) + tx_i32
for _rz in range_constexpr(_NREP_Z):
    _zi = _gtid + fx.Int32(_rz * _GRID * 256)
    if _zi < fx.Int32(_NDW):
        _scalar_store(mb_tiles, _zi, fx.Int32(0), fx.Int32)
# (c0) zero the histogram while the topk loads fly
_e_a = tx_i32
for _r in range_constexpr(2):
    _ci = tx_i32 + fx.Int32(_r * 256)
    if _ci < fx.Int32(272):
        _scalar_store(cnt_tiles, _ci, fx.Int32(0), fx.Int32)
_tk_regs = [  # the first use of the loaded ids: a padding item becomes -1
    (_tv & _m) | (fx.Int32(-1) ^ _m) for (_tv, _m) in _tk_regs
]
gpu.barrier()
for _r in range_constexpr(_NREP):  # (c) one LDS atomic add per item
    _i = tx_i32 + fx.Int32(_r * 256)
    if _i < fx.Int32(_MT):
        _lds_atomic_add(cnt_tiles, _tk_regs[_r], 1)
gpu.barrier()
```

Explained in [moe/few_tokens](../moe/few_tokens.md) § How to write it.

### 2. An inclusive add-scan over the 64 lanes of a wave (the scan of steps d and f)

The scan that steps d and f run: an inclusive add-scan over the lanes of a wave, in six DPP steps.
`dpp_utils.update_dpp_i32` is the kernel's wrapper of the `llvm.amdgcn.update.dpp.i32` intrinsic (old value, source,
control, row mask, bank mask, `bound_ctrl`): the controls 0x111, 0x112, 0x114 and 0x118 are `row_shr` 1, 2, 4 and 8,
0x142 under row mask 0xA is `row_bcast15` and 0x143 under row mask 0xC `row_bcast31`, and `bound_ctrl` is set so that
a shifted-in lane adds 0. The scan needs a full EXEC mask (`../moe/grouped_group_mapping.md` §5). It ends at the
wave: the workgroup's four wave totals go through LDS, in entries 3 and 4.

```python
def _dpp_scan_wave(x):
    """64-lane inclusive add-scan, mirroring the HIP dpp_inclusive_scan_wave."""
    for ctrl in (0x111, 0x112, 0x114, 0x118):
        t = dpp_utils.update_dpp_i32(0, _raw(x), ctrl, 0xF, 0xF, True)
        x = x + fx.Int32(t)
    t = dpp_utils.update_dpp_i32(0, _raw(x), 0x142, 0xA, 0xF, True)
    x = x + fx.Int32(t)
    t = dpp_utils.update_dpp_i32(0, _raw(x), 0x143, 0xC, 0xF, True)
    x = x + fx.Int32(t)
    return x
```

Explained in [moe/few_tokens](../moe/few_tokens.md) § How to write it.

### 3. Blocks per expert and their offsets (step d)

Step d. Thread t holds expert t's count (`_e_a` is the thread id), read from the histogram of step c; `BM` is the
rows of a block, `wave` and `lane` the thread's wave and lane, and `wt_tiles` the LDS words of the wave totals. Each
thread turns its count into ⌈count / 16⌉ blocks and scans them inside its wave (entry 2); lane 63 of each wave stores
the wave's total, and each thread adds the totals of the waves before its own, which gives its expert's exclusive
block offset. Expert 256, the 257th, has no thread: its blocks are counted apart and placed last. The last value
returned is the call's total of blocks, from which the device computes the bound (entry 5).

```python
def _scan_d():
    _cnt_a = _global_i32_load(cnt_tiles, _e_a)
    _nb256 = (  # expert 256 (the 257th) has no thread: counted apart
        _global_i32_load(cnt_tiles, fx.Int32(256)) + fx.Int32(BM - 1)
    ) >> fx.Int32(4)
    _nb_a = (_cnt_a + fx.Int32(BM - 1)) >> fx.Int32(4)  # ceil(count / 16) blocks

    # ---- (d) block-wide inclusive scan of the padded block counts --
    _inc = _dpp_scan_wave(_nb_a)
    gpu.barrier()
    if lane == fx.Int32(63):  # each wave's total, through LDS
        _scalar_store(wt_tiles, wave, _inc, fx.Int32)
    gpu.barrier()
    _w0 = _global_i32_load(wt_tiles, fx.Int32(0))
    _w1 = _global_i32_load(wt_tiles, fx.Int32(1))
    _w2 = _global_i32_load(wt_tiles, fx.Int32(2))
    _w3 = _global_i32_load(wt_tiles, fx.Int32(3))
    _pref = (  # the totals of the waves before this one
        (_w0 & _ashr(fx.Int32(0) - wave, 31))
        + (_w1 & _ashr(fx.Int32(1) - wave, 31))
        + (_w2 & _ashr(fx.Int32(2) - wave, 31))
    )
    _t256 = _w0 + _w1 + _w2 + _w3  # expert 256's blocks start here: it is placed last
    _excl = _inc + _pref - _nb_a  # exclusive block offset of this thread's expert
    gpu.barrier()
    return _nb_a, _excl, _t256, _t256 + _nb256
```

Explained in [moe/few_tokens](../moe/few_tokens.md) § How to write it.

### 4. The rank of an item among its expert's items, and the block's 16 rows (step f)

Step f, in a workgroup that passed the bound test. `fold_e` is the block's expert from step e, `_lo` the rank of the
block's first row among the expert's items and `_is_eq` a branch-free equality test for non-negative operands; the
wave totals `_u0` to `_u3` and the prefix `_p2` go through LDS as in step d (entry 3), and `_base` carries the count
of the passes before. For an item of the block's expert, `_d` is its rank among that expert's items minus `_lo`:
with `_d` in [0, 16) the item is row `_d` of the block and stores to LDS its token id (`rw_tiles`, the rows stage 1
gathers), its `(slot << 24) | token` id (`st_tiles`) and its routing weight (`wg_tiles`). `_HD2N` and `_TKWN` are
compile-time switches, both on in this kernel, so the ids and weights come from the registers of step b (entry 1)
and the `else` branches are not compiled. A padding item's −1 (entry 1) also tests equal, so `_x` is 1 in the
padding lanes too. They come after every item and change no item's rank; where one falls in the block's 16 rows it
stores a token id at or above the token count, a row that stage 2 skips (entry 7). This is read from the kernel's
source, not run.

```python
# (f) rank the items of this expert and keep rows [lo, lo+16)
_base = fx.Int32(0)
for _r in range_constexpr(_NREP):
    _i = tx_i32 + fx.Int32(_r * 256)
    if const_expr(_HD2N):
        _v = _tk_regs[_r]
    else:
        _v = _global_i32_load(tk_tiles, _i)
    _x = _is_eq(_v, fold_e)
    _inc2 = _dpp_scan_wave(_x)
    # ... the four wave totals _u0 to _u3 through LDS and the prefix _p2 of the earlier waves, as in step d
    _d = _base + _p2 + _inc2 - _x - _lo
    if _x == fx.Int32(1):
        if fx.Uint32(_d) < fx.Uint32(BM):
            _tok = _i // fx.Int32(_TOPK)
            _kk = _i % fx.Int32(_TOPK)
            _scalar_store(rw_tiles, _d, _tok, fx.Int32)
            _scalar_store(
                st_tiles, _d, _tok | (_kk << fx.Int32(24)), fx.Int32
            )
            if const_expr(_TKWN):
                _scalar_store(wg_tiles, _d, _tkw_regs[_r], fx.Int32)
            else:
                _scalar_store(
                    wg_tiles, _d, _global_i32_at(arg_topkw, _i), fx.Int32
                )
    _base = _base + _u0 + _u1 + _u2 + _u3
gpu.barrier()
```

Explained in [moe/few_tokens](../moe/few_tokens.md) § How to write it.

### 5. The upper bound of the stage-1 grid

The grid of stage 1, which the host computes from the token count alone: an upper bound on the m-blocks (`NE`
experts, `BM` rows a block) times the n-blocks. Here `num_n_blocks` is 2 and `NF`, the n-blocks a workgroup takes
under n-fusion, is 1, which gives 144, 288, 518 and 554 workgroups at 8, 16, 32 and 64 tokens. The host reads no
count back: the device computes its bound from step d's total of blocks (entry 3), and a workgroup at or past it
exits (`../moe/grouped_sorted_tokens.md` §2).

```python
# every routed item, plus up to BM - 1 padding rows for each expert the items can hit
active = min(n_tokens * TOPK, NE)
max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
tiles = max_m_blocks * (num_n_blocks // NF)
```

Explained in [moe/few_tokens](../moe/few_tokens.md) § How to write it.

### 6. MXFP4 quantization of the gathered rows inside stage 1

Not a step of the head: after it, stage 1 quantizes its 16 gathered hidden rows to MXFP4 itself instead of reading a
pre-quantized copy. The excerpt takes a batch of `n` loads; `h_dw[i]` holds the four bf16x2 words a lane loaded, 8
bf16 values. Each lane takes the largest magnitude of its 8 values (`_pkmax_u16` is an unsigned max of the 16-bit
halves of two words), a DPP quad max leaves the four lanes of a quad, which hold one group of 32 values, with one
amax, `_inline_e8m0` turns it into the E8M0 scale, rounded up, and `cvt_scalef32_pk_fp4_bf16` packs the lane's 8
values into one dword of fp4. That dword goes to the swizzled LDS A tile.

```python
la = [None] * n
for i in range_constexpr(n):
    hm = [h_dw[i][j] & fx.Int32(0x7FFF7FFF) for j in range_constexpr(4)]  # |x| of the lane's 8 bf16 values
    m01 = _pkmax_u16(hm[0], hm[1])
    m23 = _pkmax_u16(hm[2], hm[3])
    m0123 = _pkmax_u16(m01, m23)
    lo = m0123 & fx.Int32(0xFFFF)
    hi = m0123.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)
    la[i] = _umax_i32(lo, hi)
a = [fx.Int32(_raw(la[i])) for i in range_constexpr(n)]
s1 = [  # DPP quad max: quad_perm [1, 0, 3, 2], then [2, 3, 0, 1]; the four lanes of a quad end with one amax
    fx.Int32(
        dpp_utils.update_dpp_i32(_raw(a[i]), _raw(a[i]), 0xB1, 0xF, 0xF, True)
    )
    for i in range_constexpr(n)
]
a = [_umax_i32(a[i], s1[i]) for i in range_constexpr(n)]
s2 = [
    fx.Int32(
        dpp_utils.update_dpp_i32(_raw(a[i]), _raw(a[i]), 0x4E, 0xF, 0xF, True)
    )
    for i in range_constexpr(n)
]
a = [_umax_i32(a[i], s2[i]) for i in range_constexpr(n)]
e8 = [_inline_e8m0(a[i]) for i in range_constexpr(n)]  # the E8M0 scale, rounded up
for i in range_constexpr(n):
    qs_raw = _raw(fx.Float32(_raw(e8[i] << fx.Int32(23)).bitcast(T.f32)))
    pk = _raw(fx.Int32(0))
    for j in range_constexpr(4):  # 8 bf16 values into one dword of fp4
        src_bf16x2 = _raw(
            fx.Vector.from_elements([h_dw[i][j]], fx.Int32).bitcast(fx.BFloat16)
        )
        pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, src_bf16x2, qs_raw, j)
```

Explained in [moe/few_tokens](../moe/few_tokens.md) § How to write it.

### 7. Stage 2 adds each row's bf16 pairs into `out` with atomics

The end of stage 2's epilogue, the accumulation into `out`. Each row of the m-block first has its fp32 results
multiplied by its route's routing weight and packed as pairs of bf16 (the excerpt's first statement; `_pk_all[mr][s]`
holds them, 4 pairs for each of the thread's `M_REPS` rows). `packed[mr]` is the row's sorted id `(slot << 24) | token`, which stage 1 published in
step g, and `i32_M` the token count: a padding row's token id is not below the token count, so it adds nothing. Every
other row adds its pairs into its token's row of `out`, one `atomicrmw fadd` per pair; `N_OUT` is the 6144 columns of
`out`, `BN` the 256 output columns the workgroup takes and `col_start` the thread's first column among them.

```python
# per row and pair of columns: the fp32 result × the route's routing weight, packed as 2 × bf16
pk = Vec.from_elements(
    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
).to(fx.BFloat16)
# then every row of the block adds its pairs into its token's row of `out`
for mr in range_constexpr(M_REPS):
    token_id = packed[mr] & fx.Int32(0x00FFFFFF)
    if token_id < i32_M:
        row_base_addr = (
            token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
        )
        for s in range_constexpr(4):
            off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
            out_ptr = _gep1(out_base, off)
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.fadd,
                out_ptr,
                _raw(_pk_all[mr][s]),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )
```

Not this: these adds where two runs must give the same bytes. A token's 9 routes sit in different expert blocks, so
their adds land in arrival order and two runs of the kernel read 48.4–51.9 dB SQNR against each other
*(field, E2)*;
a fixed-order stage 2 keeps the bytes, for example one wave per route and an LDS sum in route order, as the layer's
all-m kernel runs at 1–2 tokens ([fused_moe_shuffled_16x16](fused_moe_shuffled_16x16.md) § Mechanism, 1–2 tokens).

Not this: the same loop in a plain Python helper, where the `if` runs at trace time (a `Numeric` value raises, a raw
value such as an `llvm.load` result reads as true), so the guard is dropped silently and the atomics fire on the
padding rows; the kernel's epilogue is a `@flyc.jit` function, which rewrites `if token_id < i32_M` into a device
branch. By its source comment, FlyDSL's `a16wmix` stage 2 ran about 13× slower without the guard at 896 experts
*(field, E6)* (`../flydsl/moe.md` § grouped_sorted_tokens §2).

Explained in [cases/fused_moe_shuffled_16x16](fused_moe_shuffled_16x16.md) § Mechanism and [moe/few_tokens](../moe/few_tokens.md) § When it applies.

### 8. Steps c and f in short: the LDS atomic add and the rank from a scan

Steps c and f again, cut down to their FlyDSL calls; the lines are among those of entries 1 and 4. Step c is one LDS
atomic add per routed item, on the histogram word of the item's expert: `_lds_atomic_add(tiles, idx, val)` puts `val`
in a one-element register tensor and copies it to word `idx` of the LDS view `tiles` with a copy atom made from
`fx.UniversalAtomicAdd(fx.Int32)`. Step f is the rank: `_is_eq` gives 1 where the item routes to the block's expert
(and for a padding item, entry 4), `_dpp_scan_wave` (entry 2) scans those flags inside the wave, and `_d` adds the
earlier waves and passes and takes off the flag itself and `_lo`.

```python
# c: one LDS atomic add per routed item
for _r in range_constexpr(_NREP):
    _i = tx_i32 + fx.Int32(_r * 256)
    if _i < fx.Int32(_MT):
        _lds_atomic_add(cnt_tiles, _tk_regs[_r], 1)
# f: rank among this expert's items, in item order
_x = _is_eq(_v, fold_e)          # 1 where this item routes to the block's expert
_inc2 = _dpp_scan_wave(_x)       # inclusive scan inside the wave, then wave totals through LDS
_d = _base + _p2 + _inc2 - _x - _lo
```

Explained in [flydsl/moe](../flydsl/moe.md) § few_tokens How to write it.

## Result

Against aiter's `fused_moe`, per token count, three readings of one run each:

| Reading | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| one-sided Student-t lower bound of aiter's time over the kernel's after 16 or 32 rounds of CUDA-graph replays, L2 flushed before each call, kernel times through Kineto, on an MI355X *(field, E3)* | 1.113 | 1.079 | 1.112 | 1.094 |
| lower-bound speedup, L2 flushed per call, on FlyDSL 0.3.2, one run of 2026-10 *(field, E4)* | 1.122 | 1.079 | 1.099 | 1.081 |
| a separate run of 2026-10, its ruler not stated in the record *(field, E5)* | 1.129 | 1.087 | 1.117 | 1.090 |

The first row is the family page's reading: all four token counts at or above 1.03 and passing the accuracy check,
where the layer's all-m kernel reads 0.925, 0.180, 0.820 and 0.813 *(field, E3)*.
What the rulers' names mean: `../gemm/measurement.md` §1.

What the fold gained: the sort as a graph node of its own (three nodes) against the fold (two), on the per-kernel
CUDA-graph timer.

Measured with the same inputs, one median per arm *(field, E1)*:

| tokens | 3 nodes (µs) | folded (µs) | vs aiter, 3 nodes | vs aiter, folded |
|---|---|---|---|---|
| 8 | 39.09 | 34.38 | 1.007 | 1.146 |
| 16 | 53.20 | 49.68 | 1.015 | 1.088 |
| 32 | 84.28 | 79.19 | 1.058 | 1.132 |
| 64 | 102.04 | 95.67 | 1.043 | 1.099 |

Removing the sort node saved 3.5–6.4 µs per call, 6–12% of the call, in that one run per arm
*(field, E1)*.
Remeasured, the node itself cost 3.9–6.0 µs at 8–64 tokens on the timer, through Kineto and through a rocprofiler-sdk
tracer, and 2.8–5.8 µs back to back without a flush, which bounds the saving *(field, E7)*.

The output is judged by SQNR, since stage 2's atomics make it differ from run to run: `../moe/few_tokens.md` §3 calls
a port numerically equivalent to its source when their cross-SQNR lies within the port's own run-to-run SQNR. On 3
input sets per token count *(field, E8)*:
the kernel against aiter 45.9–49.9 dB, the implementation it was ported from against aiter 45.8–51.9 dB, the kernel
against that implementation 48.5–51.5 dB, the kernel's second run against its first 48.4–51.9 dB. The accuracy bar
is 13 dB.

Two calls at once on two streams, each with its own inputs and outputs, on unrestricted streams and on two streams
limited to the same first 64 CUs, each compared with an eager call alone, gave 0 wrong outputs in 20 tries at every
token count *(field, E9)*.

To confirm the fold in a run of your own: a kernel trace of one call shows two kernels, the GEMM with the fold and the
GEMM with the atomics, and no `mxfp4_moe_sort` or `moe_sorting` kernel (`../moe/few_tokens.md`, How to confirm it
took effect).

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, per-kernel CUDA-graph timer (Kineto); one run per arm, aiter re-timed in each arm] the source with the fold off at these counts (three nodes) against the fold on: 8 tokens 39.086 vs 34.376 µs, 16 53.201 vs 49.683, 32 84.284 vs 79.193, 64 102.04 vs 95.673; aiter / candidate 1.007 vs 1.146, 1.015 vs 1.088, 1.058 vs 1.132, 1.043 vs 1.099 (257 experts, inter 256, top-9, the bench's input sets)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=3, three of the ruler's input sets per token count, SQNR of the bf16 output between two runs] 48.35–51.87 dB at m 8, 16, 32, 64 (the 257/256 fold kernel's sources, byte-identical)
- E3 [2026-11, gfx950, FlyDSL ?, N=1, one-sided Student-t lower bound of aiter's time over the kernel's after 16 or 32 rounds of CUDA-graph replays with L2 flushed before each call, kernel times from a rocprofiler-sdk tracer (the 257/256 fold kernel: through Kineto), one MI355X] speedup over aiter in the latest run, per token count m 1–4096: 257/256 all-m 2.489, 2.016, 0.137, 0.925, 0.180, 0.820, 0.813, 0.314, 0.855, 0.372, 0.431, 0.793, 0.741; 33-expert 1.272, 1.109, 1.052, 1.021, 0.973, 1.022, 1.053, 1.186, 1.237, 1.285, 1.670, 1.090, 1.072; 65-expert all-m 1.316, 1.085, 1.031, 0.984, 0.965, 0.998, 1.012, 0.898, 0.958, 1.020, 1.238, 0.868, 0.811; 257/256 fold at m 8, 16, 32, 64: 1.113, 1.079, 1.112, 1.094, from an earlier run and not re-measured in the latest
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, lower-bound speedup over aiter, one run, L2 flushed per call] 1.1220, 1.0786, 1.0987, 1.0805 at 8, 16, 32, 64 tokens (the fold kernel's sources)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, a separate run of 2026-10, ruler not stated in the record; later builds of the kernel read differently] speedup over aiter 1.129, 1.087, 1.117, 1.090 at 8, 16, 32, 64 tokens (257 experts, top-9, model dim 6144, inter 256, the folded kernel)
- E6 [2026-08, gfx950 (the source does not name the GPU; its `NUM_CU = 256` is commented as gfx950's), FlyDSL ?, N=?, ruler not stated] FlyDSL's own source comment: without `@flyc.jit` on the atomic bf16 epilogue helper, its `token_id < M` guard runs at trace time and is dropped, so the atomic scatter fires on padded and out-of-range rows: stage 2 ~13× slower, 39 → ~490 µs at 896 experts (FlyDSL `moe_2stage_a16wmix` stage 2, shape not stated; ROCm/FlyDSL#948, commit 47ed57a6, read at dd8373349:kernels/moe/moe_2stage_a16wmix/gemm2.py:39-41)
- E7 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=8 / 6 / 5 rounds; back-to-back replay of a 20-call graph without a flush, and the per-kernel CUDA-graph timer through a rocprofiler-sdk tracer and through Kineto; differences paired per round] the folded kernel with aiter's MXFP4 sort node (`mxfp4_moe_sort`) in front minus the folded kernel alone, at 8–64 tokens: back to back 2.80–5.82 µs, the timer through the rocprofiler-sdk tracer 3.96–6.03, through Kineto 3.85–5.80 (257 experts, top-9, model dim 6144, inter 256)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=3, three of the bench's input sets per token count, SQNR of the bf16 outputs] port vs aiter 45.88–49.89 dB, source vs aiter 45.85–51.88, port vs source 48.49–51.53, port run 2 vs run 1 48.35–51.87 at 8, 16, 32, 64 tokens
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and token count, check (c) of E10 on the unmodified kernels] wrong output, tries out of 20 on unrestricted streams / on the shared 64 CUs: 33-expert m 1 16/20, m 2 19/20, m 4 18/20, m 8 19/20, m 16 19/20, m 32–4096 20/20; 65-expert all-m m 1 0/20, m 2 18/20, m 4–32 19/20, m 64–4096 20/20; 257/256 all-m m 1 0/0, m 2 0/3, m 4 20/20, m 8 0/20, m 16 20/20, m 32 19/20, m 64–128 20/20, m 256 19/20, m 512–2048 20/20, m 4096 20 of 20 unrestricted, then a GPU memory access fault in one of its grouped GEMM kernels; 257/256 fold 0 at every token count
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and token count (24 graph replays), stress checks on MI355X] 0 differing tries for the earlier builds of the five newer kernels at every token count they are built for under (a) a stream limited to 16 CUs, one limited to 64 CUs, a second call with the same inputs on another stream, replays of one captured graph; (b) one CUDA graph replayed 24 times with three input sets copied into its buffers, each replay compared with an eager call on the same inputs; (c) two calls at once on two streams, each with its own input set and outputs, on unrestricted streams and on two streams limited to the same first 64 CUs, each compared with an eager call alone
- S1 aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv:52 tokens 4–128 at model_dim 6144, inter_dim 256, 257 experts: `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt` (rows 52–57); tokens 1 and 2: `flydsl_moe1_afp4_wfp4_bf16_…_fp4` (rows 50–51)
- S2 the AgentX deployment's fused-MoE seam (not public) folds only at M in {6, 12, …, 42} and otherwise calls aiter's `_fused_moe_impl`, whose sort is `aiter.mxfp4_moe_sort` (`aiter/fused_moe.py:161` of the baseline); the baseline's `aiter/ops/flydsl/moe_sorting.py:4-8` is a drop-in for `moe_sorting_fwd`, with the same signature
