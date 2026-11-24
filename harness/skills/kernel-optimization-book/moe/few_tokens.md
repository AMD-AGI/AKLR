---
title: "MXFP4 fused MoE at 8–64 tokens per call: count the graph nodes, fold the routing sort into stage 1"
kind: technique
lever: moe_few_tokens
ops: [moe, fused_moe]
dtypes: [mxfp4]
regimes: ["tokens 8-64", small-m, launch-bound]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
sources:
  - aiter/fused_moe.py
  - aiter/ops/flydsl/moe_sorting.py
  - aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv
---

# MXFP4 fused MoE at 8–64 tokens: one graph node less

The kernel this page reads from is a FlyDSL fused MoE with MXFP4 weights in aiter's 16×16 preshuffle and MXFP4
activations (E8M0 scale per 32), 257 experts, top-9, model dim 6144, inter dim 256, SiLU with a separate gate, built
for 8, 16, 32 and 64 tokens: two GEMM kernels, one launch each. The code excerpts below are from it.
"Tokens" (M in formulas) is the number of tokens in the call. What it measured, and what the
kernel of the same shape does at 1–2 tokens: [fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md).
aiter's MoE kernels, their names and knobs: `../flydsl/kernels/flydsl_kernel_library.md` (MoE).

## What it does

A MoE call of 8–64 tokens (as a decode step makes) lasts a few tens of µs, and every launch in it costs a fixed
amount under graph replay. The
kernel drops the routing-sort launch: every stage-1 workgroup rebuilds, in LDS, the part of the sort it needs, and
the zeroing of the output moves into the same launch. Two graph nodes instead of three.

### 1. Count the graph nodes

- The unfolded source runs three nodes at these sizes: aiter's MXFP4 routing sort (per expert, padded to 16-row
  blocks; it also zeroes the output buffer), stage 1 (gate/up GEMM, activation, quantization of the intermediate)
  and stage 2 (down GEMM, weighted accumulation into the output) (S2; read from the source, not
  from a trace).
- A call took 39.1–102.0 µs with three nodes and 34.4–95.7 µs folded *(field, E1)*.
- On the per-kernel CUDA-graph timer (`../gemm/measurement.md` §1; here through Kineto, queue profiling forced) a
  node that did almost no work
  still read about 4 µs: a fully empty kernel timed in place of a dense GEMM read 4.21 µs at m = 8 *(field, E2)*, and
  an add + RMSNorm kernel (a HIP kernel, not a MoE) read 3.93–4.04 µs at 1–256 rows of 6144 *(field, E3)*. The
  floor holds through a rocprofiler-sdk tracer too; more on it: `../gemm/small_m.md` §1.
- Removing the sort node saved 3.5–6.4 µs per call, 6–12% of the call, in one run per arm on that timer
  through Kineto *(field, E1)*. Remeasured, the node itself cost 3.9–6.0 µs at 8–64 tokens on the timer, through
  Kineto and through a rocprofiler-sdk tracer, and 2.8–5.8 µs
  back to back without a flush, which bounds the saving *(field, E12)*.

### 2. Fold the routing sort into stage 1

Two nodes instead of three:

- Every stage-1 workgroup (one per 16-row expert block × 256-column gate/up block) rebuilds the whole padded sort of
  the M × 9 routed items from `topk_ids` in LDS: a histogram of the experts, a wave scan of the per-expert block
  counts, the rank of each item of its own block's expert. The work is repeated in every workgroup and needs no grid
  barrier and no communication between workgroups.
- The workgroup with n-block 0 of each expert block publishes that block's sorted token ids, routing weights and
  expert id for stage 2 (block 0 also the valid count), and the whole grid zeroes the output in stripes, since
  stage 2 adds into it with atomics.
- Stage 1 quantizes its 16 gathered hidden rows to MXFP4 inline instead of reading a pre-quantized copy: amax over
  8 bf16 values, a DPP quad max, E8M0 round-up, `cvt_scalef32_pk_fp4_bf16` into the swizzled LDS A tile.

Measured with the same inputs, one median per arm *(field, E1)*:

| tokens | 3 nodes (µs) | folded (µs) | vs aiter, 3 nodes | vs aiter, folded |
|---|---|---|---|---|
| 8 | 39.09 | 34.38 | 1.007 | 1.146 |
| 16 | 53.20 | 49.68 | 1.015 | 1.088 |
| 32 | 84.28 | 79.19 | 1.058 | 1.132 |
| 64 | 102.04 | 95.67 | 1.043 | 1.099 |

Lower-bound speedups of the folded kernel over aiter (`../gemm/measurement.md` §1), L2 flushed per call, on
FlyDSL 0.3.2: 1.122, 1.079, 1.099, 1.081
*(field, E4)*. A separate run of 2026-10, its ruler not stated in the record, also on FlyDSL 0.3.2 (its install replaced the image's 0.3.4.1 with that pin,
`../pitfalls/measured_a_different_build.md`): 1.129, 1.087, 1.117, 1.090 *(field, E5)*; later builds of the kernel read differently.

## How to write it

The head runs at the top of the stage-1 kernel, before and just after the bound check:

| Step | What |
|---|---|
| a | zero `out` as int32 words striped over the whole grid, surplus workgroups included |
| b | load the workgroup's `topk_ids` items and their routing weights into registers; padding items read −1 |
| c | zero a 272-word LDS histogram while those loads fly, then one LDS atomic add per item |
| d | per expert `ceil(count / 16)` blocks; an inclusive DPP scan per wave, the four wave totals through LDS, an exclusive block offset per expert; expert 256 (the 257th) is placed last |
| e | the thread whose expert owns this m-block writes the expert id and the block's index within the expert to LDS |
| f | rank this expert's items in item order (a DPP scan of `item == expert`), keep ranks [lo, lo + 16) as the block's rows: token ids, `(slot << 24) \| token` ids and routing weights |
| g | n-block 0 publishes the block's sorted ids, weights and expert id; block 0 also `num_valid_ids` |

Steps a–c. `tx_i32` is the thread id, `bx_i32` the workgroup id, `_MT` the number of routed items, `_NREP` the
passes of the workgroup's threads over them:

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

The scan of steps d and f, an inclusive add-scan over the lanes of a wave:

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

Step d. Thread t holds expert t's count (`_e_a` is the thread id); `BM` is the rows of a block, `wave` and `lane`
the thread's wave and lane:

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

Step f. `fold_e` is the block's expert from step e, `_lo` the rank of the block's first row among the expert's
items, `_is_eq` a branch-free equality test; the wave totals go through LDS as in step d:

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

- The device then computes `bound = total_m_blocks × 2` n-blocks; a workgroup at or past it exits, and at 32
  tokens the tile id is remapped per XCD only after that test
  (`grouped_sorted_tokens.md` §2).
- The first two weight and weight-scale prologue stages are issued as soon as step e has the expert id, before the
  LDS-only steps f and g, so the head runs under the weight stream. Its effect is not
  measured here.
- The token count is a compile-time constant: the sort loops unroll to ⌈9M / 256⌉ passes of 256
  threads (1–3 here), and the grid is the upper bound on 16-row blocks times 2 n-blocks:
  144, 288, 518 and 554 workgroups at 8, 16, 32 and 64 tokens.
- Stage 1 asserts tile-M 16 with inline quantization for the fold.

The grid's upper bound (`NE` experts, `BM` rows a block):

```python
# every routed item, plus up to BM - 1 padding rows for each expert the items can hit
active = min(n_tokens * TOPK, NE)
max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
tiles = max_m_blocks * (num_n_blocks // NF)
```

The inline quantization, for a batch of `n` loads; `h_dw[i]` holds the four bf16x2 words a lane loaded:

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

In FlyDSL: `../flydsl/moe.md` § few_tokens How to write it.

## When it applies

- Token counts where a call lasts tens of µs and the sort is a node of its own: 8–64 tokens here, 34.4–95.7
  µs folded *(field, E1)*.
- The sort fits one 256-thread workgroup: 9M items (72–576 here); one thread per expert in the scan, with expert
  256 counted apart, so up to 256 + 1 experts as written.
- Tile-M 16 and inline activation quantization in stage 1. One compile per token count, which suits a caller with a
  fixed list of token counts; a server calling arbitrary token counts would compile per count.
- An accuracy requirement that is bounded, not bitwise: stage 2 adds the weighted rows into `out` with bf16x2
  atomics, and the output is judged by an SQNR bound (§3).

Stage 2's accumulation into `out`:

```python
for mr in range_constexpr(M_REPS):
    token_id = packed[mr] & fx.Int32(0x00FFFFFF)  # the sorted id is (slot << 24) | token
    if token_id < i32_M:
        row_base_addr = (
            token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
        )
        for s in range_constexpr(4):
            off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
            out_ptr = _gep1(out_base, off)
            llvm.AtomicRMWOp(  # one atomic fadd of a bf16x2 pair, already weighted and truncated
                llvm.AtomicBinOp.fadd,
                out_ptr,
                _raw(_pk_all[mr][s]),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )
```

## Counter-signals

- **1 and 2 tokens.** The kernel is not built there. The kernel of the same shape for those counts
  runs two launches with no sort at all, one set of workgroups per route, and measured
  2.472 and 1.990 over aiter in one run of 2026-10, its ruler not stated in the record *(field, E6)*.
- **4 tokens, and 128 or more.** Not built folded, so not measured. The source had no folded build at 4 tokens, and
  its unfolded path calls aiter's C++ MXFP4 sort, which a kernel that stands without aiter cannot call (S2). At 128 or more the
  replicated sort grows with 9M items, and it runs in every workgroup of a grid that grows with M (estimated).
- **A sort node that is already one small kernel.** *(field, E8)* On an fp8 × MXFP4 MoE (896 experts, top-16)
  whose sort, token quantization, scale scatter and output zeroing ran as one 3.9–5.4 µs kernel, folding lost
  0.2–2.3% at 8–32 tokens while stage 1 quantized every row of its block. It paid once stage 1 loaded and quantized
  just its block's live rows: in the kernel built for 8–32 tokens, 6.6–10.2% less time at 8–16 tokens and 4.9–5.0% at
  24–32 than with its 4.6–5.0 µs prelude kernel, 15–27% of that at 8–16 tokens from stage 2 computing 16 rows per
  block once folded *(field, E11)*.
- **A wide stage 1.** On an a8w4 MoE with 112 of 896 experts local and inter 3072 (48–96 n-blocks per row) the fold
  lost: 80.7 against 45.7 µs at 8 tokens; on the exact m-block grid, 43.4 against 42.6 for sort + stage 1 *(field, E10)*.
- **A separate counting sort.** A counting-sort kernel in place of the MXFP4 sort node is untested; aiter's FlyDSL one
  (`aiter/ops/flydsl/moe_sorting.py`) is a drop-in for `moe_sorting_fwd`, not for `aiter.mxfp4_moe_sort` (S2).
- **What aiter runs there.** The baseline aiter's table picks its inline-quantizing `f16in` stage 1 for 4–128
  tokens of this shape and a pre-quantized-activation stage 1 at 1 and 2 (S6).
- **Output that must be bitwise reproducible.** The atomic stage 2 adds a token's 9 rows in arrival order. Use a fixed-order stage 2 there,
  for example one wave per route and an LDS sum in route order, as the kernel of the same shape does at 1–2 tokens
  (`grouped_sorted_tokens.md` §6, `../gemm/correctness.md` §8).

## How to confirm it took effect

- A kernel trace of one call shows two kernels, the GEMM with the fold and the GEMM with the atomics, and no
  `mxfp4_moe_sort` or `moe_sorting` kernel.
- The call ran the folded kernel and not a fallback path
  (`../pitfalls/measured_a_different_build.md` § How to detect).

### 3. Equivalence under atomics

Stage 2's atomic accumulation makes the output differ run to run (the accuracy bar is a whole-output SQNR bound of
13 dB, not bit equality). Call a port numerically equivalent to its source when their cross-SQNR lies within the port's own run-to-run
SQNR. Measured on 3 input sets per token count *(field, E7)*: folded port vs aiter 45.9–49.9 dB, the source vs aiter
45.8–51.9 dB, port vs source 48.5–51.5 dB, port run 2 vs run 1 48.4–51.9 dB.

## Cost and risk

- **Repeated work.** Every stage-1 workgroup, those past the device bound included, loads the items, builds the
  histogram and runs the scans: 144–554 workgroups at 8–64 tokens. The head's own cost is not measured here; on the
  896-expert fold above, running the sort chain twice added 0.5–1.5%, and a barrier-free per-wave sort did not gain *(field, E9)*.
- **LDS.** (352 + 256 · ⌈9M / 256⌉) words of sort scratch, 2.4–4.4 KiB, beside the GEMM's stages.
- **Idle HBM during the head**, until step e names the expert; the early weight issue above is the mitigation.
- **Output bits differ run to run** (§3); only where bounded accuracy is accepted.
- **Compiles.** One stage-1 compile per token count; stage 2 is compiled once for all four.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, per-kernel CUDA-graph timer (Kineto); one run per arm, aiter re-timed in each arm] the source with the fold off at these counts (three nodes) against the fold on: 8 tokens 39.086 vs 34.376 µs, 16 53.201 vs 49.683, 32 84.284 vs 79.193, 64 102.04 vs 95.673; aiter / candidate 1.007 vs 1.146, 1.015 vs 1.088, 1.058 vs 1.132, 1.043 vs 1.099 (257 experts, inter 256, top-9, the bench's input sets)
- E2 [2026-10, gfx950, FlyDSL ?, N=?, per-kernel CUDA-graph timer (Kineto) in an in-process harness] a fully empty kernel, timed in place of the GEMM, read 4.21 µs at m = 8 (a dense bf16 GEMM, N 2112 × K 7168)
- E3 [2026-10, gfx950, HIP, N=5, per-kernel CUDA-graph timer (Kineto), median of 5 rounds] an add + RMSNorm HIP kernel read 3.93–4.04 µs at m 1, 2, 4, 8, 12, 16, 32, 64, 128, 256 with hidden 6144; aiter's 4.04–4.46 µs (add + RMSNorm)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, lower-bound speedup over aiter, one run, L2 flushed per call] 1.1220, 1.0786, 1.0987, 1.0805 at 8, 16, 32, 64 tokens (the same kernel sources)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, a separate run of 2026-10, ruler not stated in the record; later builds of the kernel read differently] speedup over aiter 1.129, 1.087, 1.117, 1.090 at 8, 16, 32, 64 tokens (257 experts, top-9, model dim 6144, inter 256, the folded kernel)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one run of 2026-10 on the kernel as it was then, ruler not stated in the record; later builds of the kernel read differently] speedup over aiter 2.472 at 1 token and 1.990 at 2 (257 experts, top-9, model dim 6144, inter 256, the kernel for 1 and 2 tokens)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=3, three of the bench's input sets per token count, SQNR of the bf16 outputs] port vs aiter 45.88–49.89 dB, source vs aiter 45.85–51.88, port vs source 48.49–51.53, port run 2 vs run 1 48.35–51.87 at 8, 16, 32, 64 tokens
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, the run's bench harness (warmup 3, 20 iterations), arms back to back in one session; prelude time from Kineto] separate prelude 3.9 / 4.7 / 4.7 / 5.1 / 5.4 µs at 8 / 12 / 16 / 24 / 32 tokens; folded with every row of the block quantized: 54.37 / 72.75 / 87.59 / 116.84 / 141.47 against 53.40 / 72.03 / 87.39 / 115.39 / 138.34 µs; folded with only rows 0–15 loaded and quantized into LDS (plus a swizzle at 8 tokens and one weight tile in flight at 12): 51.29 / 71.14 / 87.70 against 53.46 / 72.72 / 88.07 µs at 8 / 12 / 16 tokens; all 32 rows quantized into LDS read 58.62 / 84.02 / 93.88 against 51.82 / 72.08 / 89.46 µs (Kineto); the kept fold +1.07% of the score in a paired ablation (fp8-activation × MXFP4 SiTUv2 fused MoE, 896 experts, top-16, inter 384, model dim 3584)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=1–2, the run's bench harness (warmup 3, 20 iterations), one session] the sort chain (mask clear to placement barrier) run twice per stage-1 workgroup: +0.64 / +0.94 / +0.39 / +1.60 / +1.09 µs over 49.86 / 67.78 / 83.09 / 109.90 / 135.88 µs at 8 / 12 / 16 / 24 / 32 tokens; a per-wave sort with ballot, DPP prefix and no barrier on the load path: 50.15 / 67.67 / 83.38 / 110.92 / 136.38 µs, the same in a second bench (same kernel as E8)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=1, per-kernel CUDA-graph timer (Kineto), median of the readings; per-kernel times from Kineto after a flush] sort, token quant and `out` zeroing folded into stage 1, grid y = min(upper-bound m-blocks, 112): 80.66 / 83.84 / 110.13 / 128.70 / 175.01 µs against 45.66 / 62.58 / 97.07 / 119.84 / 169.06 at 8 / 12 / 16 / 24 / 32 tokens; stage 1 at 8 tokens 78.9 µs folded, 43.4 with an exact host-computed grid, against 38.0 + 4.6 µs sort (a8w4 SiTUv2 MoE, 112 of 896 experts local, top-16, model dim 3584, inter 3072)
- E11 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=9, CUDA-graph replay of 20 back-to-back calls without a flush, arms alternated, ratios paired per round; per-kernel times from Kineto] the fold (stage 1 rebuilds the sort and quantizes only the live rows; stage 2 at 16 rows per block) against the same kernel's 4.64–5.00 µs prelude kernel with the unfolded stage 1: 10.2% / 8.3% / 6.6% / 5.0% / 4.9% less at 8 / 12 / 16 / 24 / 32 tokens; stage 2 1.28 / 1.48 / 0.98 µs of the 5.10 / 5.42 / 6.54 µs difference at 8 / 12 / 16 tokens (fp8-activation × MXFP4 SiTUv2 fused MoE, gate and up rows interleaved, 896 experts, top-16, model dim 3584, inter 384, the kernel built for 8–32 tokens)
- E12 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=8 / 6 / 5 rounds; back-to-back replay of a 20-call graph without a flush, and the per-kernel CUDA-graph timer through a rocprofiler-sdk tracer and through Kineto; differences paired per round] the folded kernel with aiter's MXFP4 sort node (`mxfp4_moe_sort`) in front minus the folded kernel alone, at 8–64 tokens: back to back 2.80–5.82 µs, the timer through the rocprofiler-sdk tracer 3.96–6.03, through Kineto 3.85–5.80 (257 experts, top-9, model dim 6144, inter 256)
- S2 the AgentX deployment's fused-MoE seam (not public) folds only at M in {6, 12, …, 42} and otherwise calls aiter's `_fused_moe_impl`, whose sort is `aiter.mxfp4_moe_sort` (`aiter/fused_moe.py:161` of the baseline); the baseline's `aiter/ops/flydsl/moe_sorting.py:4-8` is a drop-in for `moe_sorting_fwd`, with the same signature
- S6 aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv:52 tokens 4–128 at model_dim 6144, inter_dim 256, 257 experts: `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt` (rows 52–57); tokens 1 and 2: `flydsl_moe1_afp4_wfp4_bf16_…_fp4` (rows 50–51)

## Related

`grouped_sorted_tokens.md` (how MoE expert GEMMs map work; cache policy; stage 2) ·
`../flydsl/kernels/flydsl_kernel_library.md` ·
[fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md) · `INDEX.md`
