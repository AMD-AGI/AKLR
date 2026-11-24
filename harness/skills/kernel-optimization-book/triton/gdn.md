---
title: "GatedDeltaNet in Gluon on gfx950: one token per sequence step on the VALU, chunks of many tokens on the matrix cores (Qwen3.5, Qwen3-Next)"
kind: topic
ops: [gdn]
dtypes: [bf16, fp8, fp32]
regimes: ["one token per sequence, 1-256 rows", "chunked, 512-16384 tokens in 1-16 sequences", ragged-sequences]
gens: [gfx950]
triton: 3.8.0
---

# GatedDeltaNet kernels in Gluon on gfx950

This page is drawn from 44 Gluon kernels of the library (154 Triton and Gluon kernels for gfx950: [kernels/kernel_library.md](kernels/kernel_library.md) §1) for the GatedDeltaNet (linear attention) layers of Qwen3.5 and Qwen3-Next as
sglang runs them: 18 kernels that advance each sequence by one token ("decode" below) update a 128 × 128 fp32 state
per value head on the VALU, 26 kernels that take each sequence in chunks of many tokens ("prefill"
below) run the recurrence on the matrix cores with sequential recurrence stages. All fuse the gated RMSNorm, all but one decode kernel the fp8 group
quant; that one writes the BF16 output, the one-token step of layers whose projections sglang runs unquantized. A
kernel is named below by the sizes it was built for: "the 2-row kernel" is Qwen3.5's decode kernel for 2 rows unless
Qwen3-Next is named, and the prefill designs carry the labels of §8. A speedup is the stored value of the ruler ([tuning.md](tuning.md) §1) for one size against the baseline of §1,
sglang's own path; a call time is read on the ruler's timer, its per-call time alone, unless a line names another
timer; that timer reads every kernel at about 4 µs or more ([profiled_timer_floor.md](../pitfalls/profiled_timer_floor.md)).

## Route here when

- You write or tune a GDN kernel, one token per sequence step (§2–§6) or chunks of many tokens
  (§7–§12), or another recurrence that keeps a matrix state per sequence slot (`run`, the kernel's host entry function ([authoring.md](authoring.md) §4), scratch and validation: [authoring.md](authoring.md)).
- You choose, per row count (one token per row), CTAs that split a head's state or own whole heads, and where the second launch goes.
- You fuse an fp8 group quant into a kernel that also makes BF16 or fp32 MFMA results (FP16_OVFL; [numerics.md](numerics.md) §2).
- You pick a chunk size and schedule for a (tokens, sequences) size, need an fp32 MFMA accumulator as the next
  MFMA's operand without a trip through LDS, or must handle ragged sequences without host reads.

## 1. What the op computes, and the kernels measured here

Per row of the one-token step (one token with its own state slot; slot −1 is padding: no state change, zero core output):

1. Causal width-4 convolution of the Q/K/V channels with the slot's 3-token history (plus `conv_bias` where the model
   has it), SiLU, BF16; the history becomes `[h1, h2, x]`, compared with the reference to within its BF16 rounding.
2. q and k L2-normalized, `x · rsqrt(Σx² + 1e-6)`; q times `scale`.
3. β = sigmoid(b) rounded to BF16; decay g = exp(−exp(A_log) · softplus(a + dt_bias)).
4. On the fp32 state S of the (slot, value head), 128 × 128 in `[value, key]` order (64 KiB):
   `S ← g·S;  u = β(v − S k);  S ← S + u kᵀ;  core = S q`, rounded to BF16.
5. Gated RMSNorm with silu(z), BF16; then per value head (the fp8 group: its 128 values) E4M3FN codes, one fp32 scale.

Where the reference rounds differs per model and regime: read the reference implementation the kernel is compared
with. Qwen3.5 decode rounds each conv tap product to BF16, Qwen3-Next sums
exact products; both prefills round the normalized q and k to BF16 and multiply the core S q by `scale`, where decode
multiplies q ([numerics.md](numerics.md) for rounding points). Prefill runs the recurrence over each sequence of a
ragged batch (`cu_seqlens`) in the chunked WY form, as one kernel's docstring states it:

```python
# docstring of a Qwen3.5 prefill kernel's chunk-factor function: G relative decay, c cumulative decay
    With cumulative decay c and relative decay G, solve
      (I + tril(beta * G * (K @ K.T), -1)) [W, U]
          = [beta * c * K, beta * V].
    Replay uses R = U.T - S @ W.T, then
      Y = S @ (c * Q).T + R @ response,
      S_next = c_last * S + R @ Ktail.
    Products of decays are formed directly, without dividing tiny prefixes.
```

Here `response` in the replay equation is the transpose of the chunk's causal decayed attention, tril(Q Kᵀ ⊙ G)ᵀ (the untransposed matrix has i ≥ j), `Ktail` is K scaled by each
token's decay to the chunk end, G(last, j), W and U solve the triangular system, R = Uᵀ − S Wᵀ is the residual.

| model, op | kernels | sizes | launches per call | writes |
|---|---|---|---|---|
| Qwen3.5 decode with the fp8 group quant | 8 | 1, 2, 4, 8, 16, 32, 128, 256 rows, one kernel each | 2 | `values`, `scales` |
| Qwen3.5 decode, BF16 output | 1 | 64 rows | 2 | `normalized` |
| Qwen3.5 prefill with the fp8 group quant | 15 | 21: 512–16384 tokens in 1–8 sequences | 6–7 | `normalized`, `values`, `scales` |
| Qwen3-Next decode with the fp8 group quant | 9 | 1–256 rows (9 sizes; one template kernel covers all nine) | 2 up to 8 rows, 1 from 16 | `normalized`, `values`, `scales` |
| Qwen3-Next prefill with the fp8 group quant | 11 | 13: 1024–16384 tokens in 1–16 sequences | 5–8 | `normalized`, `values`, `scales` |

- All five also update `conv_state` and `delta_state` in place. Fill the outputs with NaN before a check, so a skipped
  row fails ([authoring.md](authoring.md) §8; a copy of the 8-row kernel skipping half its `scales`: 64 mismatches *(field, E16)*). A timer that replays the state in place times a kernel that stores only the state segments that changed (a second 64-row BF16-output kernel, added to the library after its growth and not among this page's 44, does; the 64-row kernel of this page stores every segment) faster than a step with new tokens takes ([state_kept_across_calls.md](../pitfalls/state_kept_across_calls.md), form 3).
- Qwen3.5 has 16 value heads and 4 key heads, Qwen3-Next 8 and 4; key and value width is 128; value head h reads key
  head h // (value heads / key heads). Qwen3.5's projections are contiguous: `projected_qkvz` holds every Q, every K,
  every V, then every Z, `projected_ba` every B, then every A. Qwen3-Next's are grouped by key head: its q, its k, then
  the v and the z of its value heads; its B's, then its A's. `conv_state`, `conv_weight` and `conv_bias` keep every Q,
  every K, every V in both models, so a Qwen3-Next kernel reads conv channel c from another projection column.
  Both prefills and Qwen3-Next decode add `conv_bias`; Qwen3.5 decode has none. The decode state is 1 MiB read and
  1 MiB written per row (Qwen3.5), 512 KiB each way (Qwen3-Next).
- The Qwen3.5 decode baseline makes five calls: sglang's projection split, conv update, recurrent decode and gated
  norm, then aiter's per-1×128 quant; both Qwen3-Next baselines end with the same aiter quant, the Qwen3.5 prefill
  baseline with sglang's `per_token_group_quant_fp8`.
- Decode uses no MFMA *(field, E2)*: per (row, head) two 128 × 128 matrix-vector products and a rank-1 update on a
  64 KiB state no other row reads, 0.875 flop per byte (estimated). All 26 prefill kernels use both the BF16 and the
  fp32 MFMA *(field, E2)*. No kernel uses atomics. Two runs must give the same bytes; five calls on fresh copies of the inputs were byte-identical at each kernel's first and last size (61 kernel-sizes, 305 calls; the 13 sizes between them not run) *(field, E6)*.
- The numbers of the 20 Qwen3-Next kernels below are from one run on one MI355X. A repeat measurement of each read
  its speedup within 4.7% of that run at the kernel's smallest size, but none of the 20 passed that repeat: each
  failed the accuracy check or exited nonzero *(field, E2)*.

## 2. One token per step: split each head's state at small row counts, own whole heads above

With few rows the kernels cut each head's 128 value rows into tiles, one CTA per tile (the 2-row kernel: 17.7–18.6% less time than
whole-head tiles at 2 rows, 11.6–11.9% more at 16, 3 runs *(field, E5)*; its whole-head tile overtook between 8 and 16 rows on the ruler's timer (rocprofiler-sdk tracer) and in a cold-state graph replay alike, 1 run; against the faster of the two whole-head kernels built for 8 and for 16 rows the split kernel was 1.18–1.34× faster at 1–4 rows in the replay (the ruler's timer read all three near its two-launch floor, §3) and slower from 8 rows on both *(field, E25)*; cause not measured). Both families
switch to whole heads, with the epilogue fused, where rows × value heads reaches 128: Qwen3.5 at 8 rows, Qwen3-Next at 16
(from the grids). In Qwen3.5 the tile is 8 × rows value rows at 1, 2 and 4 rows: 256 recurrence CTAs, a target of one per CU
([tuning.md](tuning.md) §6), a derived target (at 2 rows 128–512 CTAs all took 8.60–8.71 µs on the ruler's timer, near its per-launch floor, 3 runs *(field, E5)*; in a cold-state graph replay 512 CTAs took 0.94–1.04× 256's time in four kernel and row-count pairs, 1 run *(field, E25)*):

| rows | output | recurrence CTA | CTAs | VGPRs | speedup |
|---|---|---|---|---|---|
| 1 | fp8 | 2 waves, 8 value rows, `waves_per_eu=1` | 256 | 35 | 2.464 |
| 2 | fp8 | 2 waves, 16 value rows, `waves_per_eu=1` | 256 | 61 | 2.507 |
| 4 | fp8 | 4 waves, 32 value rows | 256 | 61 | 2.422 |
| 8 | fp8 | 4 waves, whole head | 128 | 142 | 2.351 |
| 16 | fp8 | 4 waves, whole head | 256 | 146 | 2.336 |
| 32 | fp8 | 4 waves, whole head | 512 | 126 | 1.651 |
| 64 | BF16 | 16 waves, whole head, `waves_per_eu=6` | 1024 | 47 | 1.461 |
| 128 | fp8 | 16 waves, whole head, `waves_per_eu=8` | 2048 | 55 | 1.426 |
| 256 | fp8 | as at 128 rows | 4096 | 54 | 1.282 |

CTA shapes from the launches; VGPRs of the recurrence kernel, none spilling *(field, E2)*; speedup: one run of the ruler per size; a later run of it read 2.478, 2.409, 2.405 and 2.360 at 1, 4, 8 and 16 rows *(field, E1)*.

- On a stream probe, not a GDN kernel, the rate follows bytes in flight from 256 CTAs up (256 CTAs × 4 waves × 8
  loads/lane read 1.07× of 2048 × 4 × 1), but at 3 MB in flight 96 × 4 × 8 ran at 0.80× and 192 one-wave CTAs × 16 loads
  at half of 384 × 4 × 2, 3 runs *(field, E5)*; at 1–4 rows state bytes grow 4× while time stays 8.62–9.14 µs (t4/t1
  1.001–1.022, 4 runs) *(field, E5)*; latency-bound is inferred, launch latency not isolated (§5).
- From 64 rows the recurrence is a 16-wave CTA with 16 state values per thread under `waves_per_eu` 6 (64 rows) or 8,
  which Triton passes as `amdgpu-waves-per-eu` min = max = N, a hard budget a probe spills to meet (S11). At 8 it
  compiles to 55 (128 rows) and 54 (256 rows) VGPRs, no spill *(field, E2)*, "a fixed, spill-free 16-wave layout" in the kernel's own words;
  the 64-VGPR cap (512 / 8) and two CTAs per CU are derived. At 4–32 rows it is unset. Forcing it can cost:
  `waves_per_eu=4` at 16 and 64 rows (the Qwen3-Next template's rejected setting, "about 19%" in its comment
  *(field, E3)*) pinned both kernels at 128 VGPRs, spilling 9 and 20, and made the call 1.60× (16 rows) and 1.60–1.63× (64 rows) as long, 2 runs *(field, E18)*.
- Qwen3-Next's per-size kernels use the same two regimes: "split" at 1–8 rows (one- or two-wave producers own 4–8
  value rows: 256 programs at 1 and 2 rows, the template 512 at 2; 512 and 1024 at 4 and 8), "fused" from 16 rows (one
  4- or 8-wave workgroup per (row, key head) owns both value heads' states and the group's 512 history channels: 64
  workgroups at 16 rows, 1024 at 256). One run of the ruler, best per-size kernel and the template
  *(field, E4)*; a repeat measurement read them within 4.7% (§1) *(field, E2)*:

| rows | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| schedule | split | split | split | split | fused | fused | fused | fused | fused |
| per-size kernel | 2.443 | 2.538 | 2.370 | 2.506 | 3.222 | 3.350 | 2.462 | 1.782 | 1.425 |
| template | 2.390 | 2.450 | 2.370 | 2.479 | 3.101 | 3.091 | 2.401 | 1.755 | 1.457 |

One per-size kernel covers 4 and 8 rows and one 128 and 256; the template is one configurable kernel for all nine
counts. A third design at 128 and 256 rows (state loaded first, barrier-guarded conv) measured 1.466 and 1.151 in the
same run *(field, E4)*. Use the kernels built for the neighbouring sizes ([kernels/kernel_library.md](kernels/kernel_library.md)) as candidates, not a nearest-size rule: at 24 rows the log-nearest, the 32-row kernel, took 16.02–16.23 µs against the 16-row kernel's 15.48–15.59 µs, and at 48 rows 23.25–23.50 µs against the 128-row kernel's 21.53–21.60 µs, in two runs of two rounds per arm *(field, E7)*; at 3 and 6 rows the nearest was 0.2–2.3% slower, at 12 and 96 rows the faster *(field, E7)*. The template keeps one exact-match config row per count. Before copying a kernel to another row count, check what it assumes:

- The 64-, 128- and 256-row kernels map programs as `token = (pid // (VH * 8)) * 8 + pid % 8` and do not
  check that the row count is a multiple of 8: at 12 rows, 32 of the 192 (row, head) pairs would not be computed and
  32 programs would read rows past the input (derived). The 32-row kernel falls back to groups of 1.
- The 16-row kernel passes the pool size as a `gl.constexpr` for a `gl.assume`. The pool size changes from call to
  call, so each new pool size compiles again, inside a capture too (four pool sizes, four binaries; with a runtime
  pool size, one *(field, E16)*): keep it runtime ([authoring.md](authoring.md) §5).

## 3. One token per step: put the launch boundary where ownership ends

Two facts place it: the gated RMSNorm and the fp8 group each span one whole value head, so an independent tile CTA cannot finish the epilogue without obtaining whole-head statistics (a same-launch coordination or recomputation variant was not tested); and the conv history may be overwritten only after every
reader of the old history has read it, while with recomputed inputs (§4) many CTAs read the same channels.

| ownership | where the history shift and the epilogue go | kernels |
|---|---|---|
| a head split over CTAs; its key head's history read by many CTAs | launch 1: conv + gates + recurrence on a tile, BF16 core to scratch (on the 1-row kernel bitwise equal to an fp32 scratch of the rounded core *(field, E17)*); launch 2: norm + quant per head, and the history shift by programs or waves that each own whole channels (moved into launch 1, it broke `conv_state` on 10–11 of 20 inputs in each of 3 runs, `delta_state` off by up to 0.95–1.46, while the unmodified kernel was exact on all 20 *(field, E5)*; what an unforced launch boundary costs was not measured) | Qwen3.5 1–4 rows, Qwen3-Next 1–8 rows |
| one owner per channel in launch 1 | launch 1 shifts each channel's history right after reading it ("All history reads are complete; retire writes before activation arithmetic", the 8-row kernel's comment); launch 2 owns whole heads and fuses norm + quant | Qwen3.5 8–256 rows |
| one workgroup owns a (row, key head) | one launch: the shift happens inside the conv where each channel has one owner (16, 64, 128, 256 rows), after a `gl.barrier()` inside the conv (the third design of §2), or after the CTA-wide core exchange where a channel has two owners (32 rows) | Qwen3-Next 16–256 rows |

- In a one-launch design, prove single ownership in the layout, `gl.static_assert(VEC * 64 * NW <= conv_width)` for a
  `BlockedLayout([VEC], [64], [NW], [0])` conv layout, so no lane or wave is replicated (one Qwen3-Next fused kernel has VEC = 1; at 32 rows the
  template's VEC 2 × 64 × NW 8 = 1024 slots for 512 channels fails it), or put `gl.barrier()` between the history
  loads and stores. On gfx950 `gl.barrier()` lowers to a workgroup release fence, `s_barrier` and an acquire
  fence, so it also orders the CTA's global accesses (S16); the barriers Triton's Membar pass inserts between
  conflicting LDS accesses cover LDS only (S16). Without either, a replicated layout can go wrong: forced inline history
  retirement at 32 rows in the Qwen3-Next template (two owners per channel) left `delta_state` wrong on 10 of 10 inputs (max |d| 3.24)
  and was at most 1.5% faster, 3 runs *(field, E5)*; its author noted 12% faster, max |d| 0.8 *(field, E3)*.
- At the per-launch floor of the ruler's timer a ratio charges a boundary more than a graph replay pays: a chained trivial kernel added
  4.04–4.08 µs to that timer's reading under Kineto, 1.77–1.84 µs to a bare replay, a fifth of two such
  kernels' 8.25–8.40 µs, 5 runs *(field, E10)*;
  the 2-row kernel's two launches take 8.55–8.87 µs, 3 runs *(field, E5)* ([profiled_timer_floor.md](../pitfalls/profiled_timer_floor.md); [authoring.md](authoring.md) §7).

## 4. One token per step: recompute shared inputs, keep reductions inside a wave

- Recompute instead of staging: split-schedule tile CTAs convolve/normalize their key head's whole Q/K (256 channels); history is read-only, costing reads and saving a launch. In a one-row Triton toy (16 key heads × 4 tile CTAs, 4-tap conv + SiLU + L2 norm, outputs identical) recompute took 0.693–0.775× the staged version's replay time in three runs but 1.014× (9.967 against 9.832 µs) in a fourth, and 0.589–0.612× on the ruler's timer (Kineto) in all four: the trade depends on the timer and the run; the GDN kernels themselves were not re-run *(field, E24)*.
- Exchange q and k between the CTA's waves through 1–2 KiB of LDS (q and k, 2 × 128 fp32, at 1 and 2 rows; q, k, v and β
  at 4 rows) and read them back in the state layout; the 2-row kernel's late q read "to shorten its register
  lifetime" compiles as an early one *(field, E17)*. Summing a `[1, BK]` slice over its unit axis yields the vector in
  `SliceLayout(0, state_layout)` with no data movement (compiled with Triton 3.8.0 for gfx950: four `ds_read_b128` per
  vector, no lane exchange; `qk_shared.index(1).load(gl.SliceLayout(0, state_layout))` is rejected by Triton 3.8.0 for gfx950 at the descriptor `index` step: "rank must be equal to or one less than the shape size. Got 2 and 1"):
```python
# the 2-row kernel; the update and the state store sit between the read of k and the read of q
        # Hold Q in LDS until state writeback to shorten its register lifetime.
        qk_shared = gl.allocate_shared_memory(
            gl.float32, (2, BK), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), qk,
        )
        k = gl.sum(qk_shared.slice(1, 1, 0).load(state_layout), 0)
        q = gl.sum(qk_shared.slice(0, 1, 0).load(state_layout), 0)
        result = gl.sum(state * q[None, :], 1)
```

- Lay the state out with 4 contiguous keys per thread (16-byte accesses) and the key axis on lanes and registers only,
  so `S k` and `S q` reduce inside a wave; take q and k from `SliceLayout(0, …)` and v, u and the core from
  `SliceLayout(1, …)` of the state layout so the broadcasts cost nothing. Examples: `BlockedLayout([1, 4], [8, 8],
  [2, 1], [1, 0])` at 2 rows, `([2, 4], [4, 16], [16, 1], [1, 0])` at 128–256 rows.
- Order for latency: issue the state load first ("Expose the large state fetch before preparing scalar gates"), fetch
  epilogue inputs early but defer their math, normalize q late: reversed, the 32-row kernel lost 0.8–1.6% (two changes in one arm, within the 1–2% A-vs-A spread, §13) and the 8-row kernel
  2.1–2.9% (the q load and normalization moved together), 3 runs *(field, E5)*; "fetch epilogue inputs early" was not varied. The Qwen3-Next template calls the position of the state load "the single most important
  knob for register pressure": before the conv at 1 and 32 rows, after the q/k normalization at 128–256 rows (early: +77
  VGPRs at 128 rows, ≤ 7 at 1 and 32 *(field, E17)*). At 16 and 64 rows its fused-heads path ignores the config's
  cache-policy and gate-position fields and hard-codes `.cg`/`.cs`: A/B those in the code.
- Bound what stays live across the reductions, and check it compiled. The template takes the accumulate form
  `gl.fma(residual, k, state * decay)` at 16 and 64 rows for a 64-VGPR tile, "the difference between 4 and 3 waves/SIMD"
  in its comment, yet it compiles to 138 VGPRs against 134 for the recompute form at 16 rows and 131 against 132 at 64, 3
  waves/SIMD both *(field, E17)*; it rounds `state * decay`, so it is a numerics choice too. The Qwen3-Next 64-row kernel loads a
  64-column state prefix before the conv and the rest after it, so "only a register-owned column prefix stays live
  across convolution", and the Qwen3-Next 16-row kernel stores three state quadrants before the epilogue's layout conversion and the
  last after it. The template's 4-warp 128-row instance computes decay and β after the conv because before it
  "costs enough registers to halve occupancy", yet both orders compile to 200 VGPRs *(field, E17)*.
- Take scalar math off the recurrence's path: the 8-row kernel computes decay and β for all heads in one extra program of the conv
  launch, "so their transcendental work can overlap the independent convolution tiles"; per update program instead,
  the 8-row call took 6.4–7.2% longer, 3 runs (overlap not measured); the 1-row kernel evaluates the four exponentials of the
  value SiLU, decay and β "in neighboring lanes" of one `[BV, 4]` tensor, one `gl.exp` and division, picked by
  `gl.gather`, 4 `v_exp_f32` against 7, no faster than scalar (B/A 0.986–1.013), 3 runs *(field, E5)*.
- Program order: five whole-head kernels (16–256 rows) number programs in token groups, `token = (pid // (VH * G)) * G +
  pid % G`, G a multiple of 8, so all heads of a row share `pid % 8` and an XCD's L2 (1.75–1.80× on a probe; its
  comment: "Interleave heads over token groups for state-pool traffic"); the 16-row kernel also XOR-permutes each row's heads,
  `head = head ^ ((7 * token) & 15)` ([tuning.md](tuning.md) §12). In the 128-row kernel it beat `token = pid // VH` by only 0.6–1.1%, 3 runs *(field, E5)*.
- Check each variant's VGPR count ([debugging.md](debugging.md)) and A/B it ([tuning.md](tuning.md) §2, §10): at equal
  VGPRs (160) the template's late state load at 32 rows took 1.23× as long *(field, E17)*.

## 5. One token per step: the state's loads and stores

The Qwen3.5 kernels use five combinations; state stores, and loads up to 32 rows, are 16 bytes per lane:

| rows | state load | state store |
|---|---|---|
| 1, 32 | `gl.load`, no hint | `gl.store`, no hint |
| 2 | `buffer_load`, `.cs` | `buffer_store`, `.cs` |
| 4, 8 | `buffer_load`, `.cg` | `buffer_store`, `.cs` |
| 16 | `gl.load`, `.cg` | `gl.store`, `.cs` |
| 64–256 | `gl.load`, no hint | `buffer_store`, `.wt` |

Both choices were ablated on the 2-row kernel: B against the unmodified A, 4 rounds per arm alternated on one
GPU, one run *(field, E9)*. The instructions are from compiling its recurrence kernel with Triton 3.8.0 for gfx950:

| change to the state access | state instructions per thread | VGPRs | mean speedup A → B (4 rounds) | kernel time A → B |
|---|---|---|---|---|
| none (as built: `buffer_load`/`buffer_store`, `cache=".cs"`) | 4 `buffer_load_dwordx4 … offen sc0 nt`, 4 `buffer_store_dwordx4 … offen sc0 nt` | 61 | — | — |
| `cache=".cs"` removed | the same 8 instructions without cache bits | 61 | 2.452 → 2.442 | 8.75 → 8.79 µs |
| `gl.load`/`gl.store` with `cache_modifier=".cs"` | 4 `global_load_dwordx4 … nt`, 4 `global_store_dwordx4 … nt`, 64-bit address in a VGPR pair | 62 | 2.429 → 2.432 | 8.83 → 8.81 µs |

- These two single-run ablations do not establish a reproducible speed advantage; the A-vs-A spread at this size was 0.4% and 1.7% over 3 runs, 1.9% and 2.0% over 8, in two sessions *(field, E9)*. The call moves 4
  MiB of state in 8.65–9.14 µs, 0.459–0.485 TB/s, 6.8–7.2% of a 1 GiB non-temporal read stream's 6.72–6.78 TB/s, and takes
  8.62–9.14 µs from 1 to 4 rows while state bytes double and quadruple (t2/t1 0.951–1.022, t4/t1 1.001–1.022,
  4 runs) *(field, E5)*: latency rather than proportional state streaming is inferred; the latency source was not isolated. At 2 rows pick buffer or pointer access for the addressing, not for speed.
- At 128–256 rows the state stream is most of the work (256 MiB each way at 256 rows) and the store hint moves the time:
  without `.wt`, or as `gl.store` with `.cs`, the state store cost 2.6–3.6% (one run); load hints were not tried *(field, E9)*.
  A/B the hints rather than copy a kernel's choice ([gluon/memory.md](gluon/memory.md)). On a 1 GiB stream probe (2048 programs × 4 waves, one contiguous range per program) `.cg`/`.cs` loads ran 13.0–13.7% faster than plain ones, `.cs` stores 5.5–7.2% slower, `.wt` stores 0.4–1.7% slower through `gl.store` (below every store without cache bits) and within 0.5% through `buffer_store`, 3 runs *(field, E5)*; on other fp32 streams the load gain held at 1 and 4 GiB (6.4–13.2%) but not at 256 MiB (−3.5 to +1.2%), and store hints ran from 18% slower to 18% faster *(field, E28)*. `.cs`
  puts `sc0 nt` on a buffer op and only `nt` on `gl.load`/`gl.store` (S18), which then stream at one rate (within 1.2%)
  *(field, E5)* ([../hardware/gfx950/memory_and_sync.md](../hardware/gfx950/memory_and_sync.md) §2).
- Buffer addressing: the descriptor covers 2^31 − 2 bytes and element offsets are scaled to bytes in 32 bits (S19), so
  kernels that fold the slot into the offset test the pool size in bytes on the host and rebase a 64-bit scalar head
  pointer for larger pools; a pool of at most 2047 slots, 2047 MiB of Qwen3.5 state,
  never takes the wide path.
- Padding rows: put one uniform `if slot >= 0:` around every access to either pool, the history shift included, and
  write a zero core otherwise. The norm and quant still run on that zero core and write the row's outputs, as
  the 8-row kernel does: zero-valued codes, roughly half −0 on the sampled random inputs, and scale float32(1e-10) × float32(1/448) *(field, E17)*.

## 6. One token per step: the fused fp8 group quant and the FP16_OVFL mode bit

One helper ends every head in all eight Qwen3.5 fp8 decode kernels (the same function text in each): it rounds the head
to BF16, takes the amax of its non-NaN values floored at 1e-10, scales by one reciprocal of `amax · (1/448)`, clamps to
±448 with an inline `v_med3_f32` (`gl.clamp` lowers to the same, S20) and converts. The recipe and its division
forms: [numerics.md](numerics.md) §2.

```python
value = value.to(gl.bfloat16, fp_downcast_rounding="rtne").to(gl.float32)  # quantize the BF16-rounded head
magnitude = gl.where(value == value, gl.abs(value), 0.0)  # a NaN does not enter the amax
maximum = gl.maximum(gl.max(magnitude, 0), 1.0e-10)
group_scale = maximum * (1.0 / 448.0)
inverse_scale = 1.0 / group_scale  # one reciprocal per head
scaled = value * inverse_scale
clipped = gl.inline_asm_elementwise(
    "v_med3_f32 $0, $1, $2, $3",
    constraints="=v,v,v,v",
    args=[scaled, -448.0, 448.0],
    dtype=gl.float32,
    is_pure=True,
    pack=1,
)
```

- Keep the head's 128 values in one wave (`BlockedLayout([4], [64], [2], [0])` gives each of two waves a full copy), so
  amax and the RMS are wave reductions with no LDS exchange.
- Quantize the BF16-rounded output, as the references do; compute the scale from it, not from the fp32 precursor.
- Round before a cross-wave `convert_layout`: "Preserve the BF16 boundary and halve the cross-wave LDS transfer".

**FP16_OVFL.** In 38 E4M3FN/E5M2 cast variants from fp32, fp16 and bf16 compiled for gfx950, Triton wrote MODE.FP16_OVFL = 1 once, before the first VALU of the function containing the cast (S21): the kernel when inlined, the callee in six noinline variants. The variants cover default and rtne `.to`, E5M2 rtz from fp32 and fp16, an implicit cast in `gl.store`, an explicit cast then `buffer_store` and a runtime branch; a direct wider-value `buffer_store` and E4M3FN rtz did not compile, and seven controls (upcast, bitcast, MFMA, asm, BF16) wrote no MODE *(field, E20)*. Other kernel entries were not compiled. The 2-row kernel's second launch, the norm and quant kernel, sets it before its first `v_cvt_pk_bf16_f32`; BF16 finite overflows then saturate to ±0x7F7F *(field, E17)*. The Qwen3.5 kernels leave it set and pass an accuracy check whose inputs are drawn finite. The Qwen3-Next 1-row kernel
and the Qwen3-Next template at 1 row map |x| ≥ 3.396e38 to ±Inf before the BF16 convert and keep Triton's fp8 store;
the template at 2, 4, 8 and 32 rows clamps and converts with an inline `v_cvt_scalef32_pk_fp8_f32`, so no Triton
cast sets the bit, and switches it in value-producing asm, clear at entry, set after the last BF16 rounding; the Qwen3-Next 32-row kernel sets
it in the convert's own asm; the template's BF16 results saturate at 16, 64, 128 and 256 rows *(field, E17)*.
Rule: [numerics.md](numerics.md) §2; asm and ordering risk: [gluon/wave_ops.md](gluon/wave_ops.md) §3.2, §7. Keep the
two choices separate knobs: "collapsing the two into one flag silently changes results", in the template's words.

## 7. Chunks of many tokens: the pipeline and its launch boundaries

Every prefill kernel runs, in launch order (the Qwen3-Next compact-factor pair and 64-token snapshot scan of §8 add a
conv-history launch, the snapshot scan a chunk-offset one):

1. A first kernel zeroes the state slot of each sequence that has no initial state, from the device flag (`if not gl.load(INITIAL + seq):` in the kernel, no host read): all
   26 kernels capture globally and replay with new inputs matching the reference; this does not independently test flag zeroing, ragged handling or stream-only synchronization *(field, E16)*. Nine Qwen3.5 kernels then run a pack pass, a copy of the
   contiguous projection into Qwen3-Next's grouped order (§1), 10 KiB per token each way (estimated). The six MAP
   and RSUM kernels read the contiguous layout directly: the pass is avoidable; what it costs is not measured.
2. Prepare, parallel over token tiles × channel groups: conv + SiLU + q/k L2 norm to BF16, gates.
3. Chunk factors, parallel over chunk × head: Gram products, decay products, the unit-triangular solve, WY factors.
4. Long sequences only (summary designs, §8): each run of chunks summarized as an affine map `S ↦ S·A + B`, in parallel.
5. Serial pass over chunks or summaries, parallel over sequence × head × value strip; per-sequence designs add outputs.
6. Parallel epilogue: outputs rebuilt from incoming states where step 5 made none, then gated RMSNorm and fp8 quant.

- Every boundary changes the parallel decomposition; with no atomics (§1), stream order is the only synchronization.
- The kernels write the conv history after its readers: 19 in their last launch; SEQ16 (§8) in the one-wave program that
  convolves a sequence's first tokens, its only reader, after a `gl.barrier()` whose removal left `conv_state` exact
  in all 20 runs (2 runs), as it must in one wave *(field, E19)*; the Qwen3-Next reverse-summary pair behind `if NW > 1`, false at their size;
  three Qwen3-Next kernels in a launch of their own. Write the history only after every reader of the old one has read it: in a toy kernel a history entry stored before the reads was read in place of the old one in every repeat, and at four waves readers that no barrier ordered before the writer read 192 wrong values, at one wave none *(field, E8)*. Put a barrier where readers and the writer sit in different waves; SEQ16's one-wave program stayed correct without one.
- A fresh sequence starts from a zero history as well as a zero state, but the kernel of step 1 zeroes only
  `delta_state`: mask every read of the old history with the flag. A sequence of one or two tokens keeps old entries:
  its new history is the last three of [old history, or zeros when fresh; its tokens]. The 2049/2 size
  always holds a one-token sequence when, as in these measurements, one sequence has exactly `max_seq_len` tokens.
- Ragged batches without host reads: grids come from shapes only *(field, E17)*; a chunk program finds its sequence by
  binary search over `cu_seqlens`, the slot formula `begin // BT + sequence` "reserves enough space for every ragged
  tail", and programs past the device-written chunk count do nothing. The search runs in the kernel: each step
  loads `gl.load(starts + mid) // BT + mid`, the first slot of sequence `mid`, and moves both bounds with `gl.where`.
- Scratch is `torch.empty` per call, so it holds undefined bytes, and large: 173–294 MB of fp32 at
  8–16K tokens in the summary designs (MAP at 10000 tokens: 246 MB of factors, 36 MB of maps) *(field, E17)*; buffers
  with disjoint live ranges are aliased. Write every element a later launch reads in the same call, padding rows and
  ragged tails included: a NaN left in old memory survives a zero β or a decay of 1 (0 × NaN is NaN). One Qwen3-Next
  reverse-summary kernel zeroes each chunk record's padding tokens; a compact-factor kernel states the rule for its
  whole pipeline, "Every consumed scratch element is initialized within the invocation, including ragged tails."
  ([authoring.md](authoring.md) §6).
- While a stream is capturing, the sglang integration of the Qwen3.5 and Qwen3-Next prefill kernels declines the call
  and sglang's own path runs (called with mocked
  sglang layers *(field, E16)*); a capture in a running server was not tested. Eager launches pay about
  1.0 µs of host time per tensor argument, which a graph-replay timer omits *(field, E10)* ([authoring.md](authoring.md) §7).

## 8. Chunks of many tokens: chunk size and schedule per size

Measure candidate designs at the specific (tokens, sequences, `max_seq_len`) size; the boundaries between designs below are not selection rules confirmed by measurement. Labels: D, the ≤ 1024-token source;
CKPT, the checkpoint scan; SEQ32L, SEQ32G and SEQ16, one program per
(sequence, head, value strip) over 32-token sequence-local, 32-token global (plus a scalar per-token recurrence at
ragged edges) and 16-token global chunks (masked so each segment's coefficients stay in its sequence); MAP and RSUM,
summaries of runs of chunks. SEQ16 and RSUM each have two sources, a and b in the table. The last column times another design's kernel, or the other strip, at the row's size,
2 runs, accuracy unchecked *(field, E19)*: the row's own kernel wins or ties except at 8192/1 (RSUM 0.73–0.74× MAP's time).

- **Short-scan candidates, not a universal ≤2048-token rule**: D and CKPT store chunk incoming states and reconstruct outputs in parallel. At (512,1,512), (1024,2,512), (512,2,256), (1024,1,1024), (1025,1,1025), both were faster than the tested RSUM and MAP under eager single-call device-event timing *(field, E21)*. CKPT beat D at all five; CKPT against SEQ32L was unresolved at the second and third sizes, not shown equal.
  One study of the five sizes, 16 balanced adjacent pairs per comparison, each source on its own host branch with no config search; all 25 size-arms passed the accuracy bounds against an fp64 sequential reference on every full tensor, and the two-sequence inputs had equal lengths. 2048/1, other ragged inputs and graph or serving timing were not covered. The table's 2048/1 RSUM/CKPT 1.99× comes from a separate measurement that did not check the transplanted kernel's accuracy *(field, E19)*.
- **More tokens with `max_seq_len` 2048 (2–8 sequences)**: one program per (sequence, head, value strip) walks all
  chunks and produces the outputs; BV makes sequences × 16 heads × 128/BV = 256, one program per CU: SEQ32L at 2
  sequences (BV 16), SEQ32G at 3–4 (BV 32), SEQ16 at 5–8 (BV 64). The sizes that fill 256 programs
  measure 1.374–1.475; the two with fewer sequences than their kernel assumes (160 and 192 programs) 0.997 and 1.029,
  below the bar of these runs ([tuning.md](tuning.md) §1), no win over sglang's own path. Their
  longest sequence is 2048 tokens, as at the full sizes, and the time does not follow the work: about half the work
  took 0.97× (SEQ32G) and 0.85–0.86× (SEQ16 a) the full size's time, 2 runs *(field, E19)*, which does not separate the
  serial chain from the program count; the other strip was slower (table).
- **Long sequences, `max_seq_len` 4096 (Qwen3-Next) or 5000 (Qwen3.5) and above, 1–4 sequences**: a per-sequence walk
  is a long serial chain, and at one sequence 16 × 128/BV programs are too few. Build each run of chunks' affine map in
  parallel, from an identity seed (MAP: the rows `[I; 0]` run through the recurrence) or composed right to left from the
  run's last chunk (RSUM); chain the maps in a short serial pass (4–15 dependent 128 × 128 products per row strip,
  derived from the chunk counts); then replay all chunks in parallel from their incoming states. At one sequence
  this beats a per-sequence walk, SEQ16 (table); 2–4 sequences and Qwen3-Next were not timed.
- **Keep the serial chain overlapped**; undoing each knob cost, at one size, 2 runs: in CKPT
  at 2048/1, unroll 1 for `loop_unroll_factor=8` 6.3–7.0%, no `amdgpu-sched-strategy` `iterative-ilp` 11.8–13.3%
  ([tuning.md](tuning.md) §10), one 128-deep MFMA chain for two independent 64-deep ones 3.1–4.6% *(field, E19)*; in
  SEQ16 b at 16384/8, the next chunk's operands loaded after the current chunk's math 2.0–2.4%, `.cg` on the value tile
  dropped −0.2 to +0.3% *(field, E19)* ([gluon/memory.md](gluon/memory.md) §12). SEQ32G stores each half table once
  final, "to shorten register lifetimes": 190 VGPRs at M = 8192, 194 with all stores at the end, untimed *(field, E19)*.

Qwen3.5, speedup from one run of the ruler per size *(field, E11)*; a later run of it read 2.452 at 1024 / 1 and 2.681 and 1.927 at 512 / 1 and 1024 / 2 *(field, E11)*; serial-pass programs derived from the grids:

| tokens / sequences | max_seq_len | source | design | chunk (tokens) | serial programs | speedup | other kernel here, its time ÷ the row's |
|---|---|---|---|---|---|---|---|
| 512 / 1 | 512 | D | D | 64 | 128 | 2.638 | — |
| 512 / 2 | 256 | D | D | 64 | 256 | 2.629 | — |
| 1024 / 1 | 1024 | D | D | 64 | 256 | 2.423 | CKPT 1.01× |
| 1024 / 2 | 512 | D | D | 64 | 512 | 1.889 | — |
| 1025 / 1 | 1025 | CKPT | CKPT | 64 | 128 | 2.256 | — |
| 2048 / 1 | 2048 | CKPT | CKPT | 64 | 128 | 2.092 | RSUM 1.99× |
| 2049 / 2 | 2048 | SEQ32L | SEQ32L | 32 | 256 | 1.374 | — |
| 4096 / 2 | 2048 | SEQ32L | SEQ32L | 32 | 256 | 1.452 | SEQ16 1.93–1.95× |
| 4097 / 3 | 2048 | SEQ32G | SEQ32G | 32 | 192 | 1.029 | BV 16 1.32× |
| 8192 / 4 | 2048 | SEQ32G | SEQ32G | 32 | 256 | 1.424 | — |
| 8193 / 5 | 2048 | SEQ16 a | SEQ16 | 16 | 160 | 0.997 | BV 32 1.15× |
| 15360 / 8 | 2048 | SEQ16 a | SEQ16 | 16 | 256 | 1.473 | — |
| 15361 / 8 | 2048 | SEQ16 b | SEQ16 | 16 | 256 | 1.454 | — |
| 16384 / 8 | 2048 | SEQ16 b | SEQ16 | 16 | 256 | 1.475 | RSUM 1.28–1.29× |
| 5000 / 1 | 5000 | RSUM a | RSUM | 32-token tiles, 512-token chunks | 128 | 1.199 | — |
| 15000 / 3 | 5000 | RSUM a | RSUM | 32 / 1024 | 384 | 1.198 | — |
| 8192 / 1 | 8192 | MAP | MAP | 16-token blocks, 512-token chunks | 128 | 1.193 | RSUM 0.73–0.74×, SEQ16 2.71× |
| 10000 / 1 | 10000 | MAP | MAP | 16 / 640 | 128 | 1.188 | — |
| 16384 / 1 | 16384 | RSUM b | RSUM | 32 / 1024 | 128 | 1.741 | SEQ16 3.97× |
| 16384 / 2 | 8192 | RSUM b | RSUM | 32 / 1024 | 256 | 1.339 | — |
| 16384 / 2 | 10000 | RSUM b | RSUM | 32 / 1024 | 256 | 1.444 | — |

Rows with the same source run byte-identical code,
yet one source's ratio moves across sizes: D 2.65 at 512 × 1 against 1.91 at 1024 × 2 (1.38–1.39×), RSUM b
1.73–1.74 at 16384 × 1 against 1.36 at 16384 × 2 with `max_seq_len` 8192 (1.27–1.28×), 2 runs *(field, E19)*: re-measure at every
size. No prefill `run` reads `max_seq_len`, only takes it.

Qwen3-Next: one run of the ruler, constexpr kernel / runtime-count twin *(field, E11)*; a repeat measurement read
them within 4.7% (§1) *(field, E2)*. Every sequence at `max_seq_len` 1024 is 1024 tokens long. The multi-schedule
rows are one source and one twin.

| tokens / sequences | max_seq_len | kernels | schedule | chunk | speedup, per size |
|---|---|---|---|---|---|
| 1024 / 1; 2048 / 2 | 1024 | snapshot-scan kernel | serial scan with per-chunk state snapshots, parallel output | 64 | 1.573; 1.434 |
| 3072 / 3; 4096 / 4 | 1024 | multi-schedule source / twin | snapshot scan, parallel output | 32 | 1.592 / 1.554; 1.504 / 1.501 |
| 8192 / 8; 12288 / 12; 16384 / 16 | 1024 | as above | direct recurrence | 32 | 1.490 / 1.484; 1.288 / 1.282; 1.463 / 1.460 |
| 13312 / 13; 15360 / 15 | 1024 | compact-factor source / twin | compact factors, operands rebuilt in the serial loop | 32 | 1.232 / 1.233; 1.328 / 1.329 |
| 4096 / 1 | 4096 | multi-schedule source / twin | block summaries (identity and zero seeds) | 32 | 1.718 / 1.719 |
| 5120 / 1; 10240 / 2 | 5120 | multi-schedule source / twin | block summaries | 32 | 1.727 / 1.741; 1.308 / 1.311 |
| 16384 / 4 | 5120 | reverse-summary source / twin | segment summaries composed in reverse | 32 | 1.294 / 1.283 |

The snapshot-scan kernel takes 64-token chunks at 1–2 sequences; the multi-schedule source picks its schedule from the size: the
snapshot scan at 3–4 sequences up to 4096 tokens, block summaries at 1–2 sequences here, the direct recurrence at 8 and
more. The twins, the token and sequence counts as `gl.constexpr` against runtime values, differ by 0.1–2.4% at a size in one
run *(field, E11)*, about the run-to-run spread of the 2-row decode kernel, a different kernel (0.4–1.7% over 3 runs, 1.9–2.0%
over 8, two sessions) *(field, E9)*; no repeated run measures it at these sizes, so A/B the two before choosing.

Before you reuse a prefill kernel at another size, read which branch its sizes take. Above 4096 tokens SEQ32L
static-asserts that `normalized` has the conv buffer's 3072-element row stride, which the (tokens, 16, 128)
output it is called with lacks: compiled with Triton 3.8.0 for gfx950 at that output's stride, that path fails its `static_assert`. The Qwen3-Next snapshot-scan kernel
finds a token's sequence in seven unrolled halvings, right for at most 128 sequences, where the op it was written for allows up to 2047 in a batch:
it fails at 129 sequences, where eight halvings pass *(field, E17)*. SEQ32G's path up to 4096 tokens and CKPT's
up to 1024 do not run at their sizes: no run checked or timed them.

## 9. Chunks of many tokens: the intra-chunk math on the matrix cores

- Put the Gram products K Kᵀ and Q Kᵀ on the BF16 MFMA (`instr_shape=[16, 16, 32]`, `k_width` 8: eight consecutive K per
  lane, [gluon/mfma.md](gluon/mfma.md) §5): the prefill references round q and k to BF16, and the product of two finite BF16 values is exact in fp32 when it lies in fp32's normal range: 2^20 random BF16 pairs with exponents within ±63 all matched the fp64 product *(field, E17)*. The fp32 accumulation still rounds.
- Share the Gram products among the value heads of a key head (four in Qwen3.5, two in Qwen3-Next): K Kᵀ and Q Kᵀ depend
  only on the key head, so SEQ32L computes them once per (chunk, key head) ("share four-head Gram matrices"), SEQ16 b and
  SEQ32G once per head pair; only β and the decays differ per value head. In SEQ32L at 4096/2 sharing
  saved 1–2% of the call (means 1.011 and 1.021 in 2 runs) and 84 VGPRs *(field, E19)*.
- State algebra, solves and tables run on the fp32 MFMA (`v_mfma_f32_16x16x4_f32`, some `32x32x2`) *(field, E2)*: 32
  cycles for 1024 MACs, against 16 cycles for 8192 MACs on the BF16 16x16x32 (S37), 16× fewer MACs per cycle (15.7–15.8×
  measured, 4 runs *(field, E13)*). Keep the serial pass to few fp32 products per chunk (CKPT: two per 64 tokens);
  moving output products to BF16 (§10) is untimed; [gluon/mfma.md](gluon/mfma.md) covers the fp32 input precision.
- The unit-lower-triangular inverse `(I + A)⁻¹` of a 16–64-token chunk: solve small diagonal blocks (4 × 4, 8 × 8 or
  16 × 16) by forward substitution with `gl.gather` pivots and join them with fp32 MFMAs; or, on a 16 × 16 block,
  multiply out `(I+N)(I+N²)(I+N⁴)(I+N⁸)`, which is `(I − N)⁻¹` for a strictly lower N (N¹⁶ = 0): pass N = −A, as RSUM b
  negates the Gram term before the inverse below and the Qwen3-Next snapshot-scan kernel documents ("Inverse of I-P for a strictly lower triangular
  16-by-16 P"), and join 16 × 16 inverses with fp32 MFMAs for longer chunks. Keep each small solve in one
  wave: `gl.gather` lowers to lane shuffles only when the gathered axis stays in a warp, else through LDS (S38).
```python
# RSUM b: `lower` is N, the negated strictly lower Gram term of a 16 × 16 block; _fp32_mfma(a, b, acc) is acc + a @ b
power = gl.convert_layout(lower, MMA)
inverse = power + (rr[:, None] == cc[None, :]).to(gl.float32)  # I + N
for stage in gl.static_range(3):
    power = _fp32_mfma(power, power, gl.zeros((16, 16), gl.float32, MMA), MMA)  # N², N⁴, N⁸
    inverse = _fp32_mfma(power, inverse, inverse, MMA)  # (I + power) @ inverse
```

- The kernels skip the zero halves of triangular operands: K blocks taken with `gl.amd.slice`, a register-only view (S39),
  only nonzero blocks multiplied, only tiles on or below the diagonal stored and loaded. In SEQ32L at 4096/2
  that cut 184 to 136 MFMAs, not the call time (B/A 0.966–1.008, 2 runs) *(field, E19)*; the stores were not varied.
- Store each factor in the order its consumer's MFMA operand holds it, so the consumer loads 16-byte vectors and
  converts with `assert_trivial=True`, which fails the compile ("layout conversion from … is not trivial",
  printing both linear layouts) when the conversion moves data across lanes or waves (S42).

## 10. Chunks of many tokens: compensated products, and decay products over intervals

**Compensated products.** Where the product of an fp32 operand and a BF16 operand is rounded to BF16 anyway, split the
fp32 operand into BF16 terms, run the BF16 MFMA once per term, smallest first, and accumulate in fp32. On random finite 16×32 @ 32×16 tiles (4,194,304 outputs), three terms (RTZ, RTZ, RNE) matched the fp64 result rounded to BF16 in 100% of the outputs, two RNE terms in 99.76%, one term in 58.2% *(field, E17)*: sampled agreement, not a proof of correct rounding for every product. Five Qwen3.5
kernels do, three saying "Only output reconstruction uses compensated BF16 products: chunk solves, the persistent
recurrence, and mutable state remain FP32". CKPT's query product takes two RTNE terms at its sizes, three RTZ terms
in its other branch: on 300000 sampled fp32 values with exponents within ±40, two RTNE terms had at most 7.605e-6 relative split error, below the estimated 2^-16 bound and the BF16 output step; three RTZ terms reconstructed every sampled value exactly *(field, E17)*. Values outside that exponent range were not sampled. Where both operands are fp32-derived,
the D kernels store both as BF16 high/low pairs and keep all four cross products. No Qwen3-Next prefill kernel uses
them ([numerics.md](numerics.md) §5). (`gl.amd.cdna4` re-exports `mfma`, `buffer_load` and `buffer_store`, S40.)

**Scan each interval, not two prefixes.** Build each relative decay Γ(i, j) over (j, i] directly, not as `prefix_i /
prefix_j`: on log-decays uniform in [−6, 0] the ratio underflows to NaN or 0 at 60% of normal-valued entries, the masked
add-scan stays within 3.8e-6 *(field, E17)*. MAP runs `gl.associative_scan` with a multiply combiner down the columns of
`gl.where(i[:, None] > j[None, :], sg[:, None], 1.0)`; the Qwen3-Next snapshot-scan kernel scans the log-decays of each interval with an add
combiner and exponentiates once, `gl.exp(gl.associative_scan(intervals, 0, _add))` over `gl.where(si[:, None] > sj[None, :], sg[:, None], 0.0)`: "Direct interval scans also handle zero decay without inf-inf or 0/0". Other
kernels split the chunk into two 16-token halves with a separable cross block, or keep dyadic levels of prefix and
suffix products and read Γ(i, j) = suffix_L[j] · prefix_L[i] at L = `31 − clz(i ^ j)`, the highest bit in which i and j
differ. `gl.associative_scan` lowers only a `BlockedLayout` operand (S38): on a `SliceLayout` vector, such as the
per-token gates of a 2-D tile, the compile fails with "TODO: unsupported scan layout" (compiled with Triton 3.8.0 for
gfx950; [debugging.md](debugging.md) §4): convert to a blocked layout, scan, convert back, as MAP does. Padding
tokens get decay 1 and β 0. One recurrence step with decay 1 and β 0 on a random finite 128 × 128 state changed no bit, but a NaN in v made its rows NaN *(field, E17)*: the update is neutral only with finite intermediates, since β = 0 times a non-finite prediction or residual is NaN. Which
exponential keeps subnormal decays (`gl.exp`) and which flushes them (`gl.exp2`): [numerics.md](numerics.md).

## 11. Chunks of many tokens: relabel layouts instead of converting them

Each chunk step feeds an fp32 MFMA accumulator (the state, or an inverse block) into the next MFMA as the A operand.
For its `[16, 16, 4]` accumulator as A, `k_width = 1` takes two register↔lane transpositions through LDS (one wave: 4 `ds_write_b32` + 4 `ds_read_b32`, 1 KiB *(field, E17)*); a transposed `[32, 32, 2]` accumulator needs only one and stays in-wave with warps on M (S38). With 4 waves, the `[16, 16, 4]` case took two transpositions and 16 KiB of LDS at 128 × 32, transposed or not; the transposed `[32, 32, 2]` one at 256 × 64 one transposition and no LDS, the untransposed one four and 64 KiB; a sweep of 432 builds stayed in-wave whenever fewer than two transpositions were needed and the warps did not change *(field, E22)*. Ways around it (Triton 3.8.0, gfx950):
- **Where the warps go decides it.** On 64 × 64 fp32 `[16, 16, 4]` tiles with 4 waves, only a `transposed=True` `[4, 1]`
  accumulator is a free `k_width = 4` A operand and a `transposed=False` `[1, 4]` one a free B operand; every other tested mix
  of transposition, warps (`[4, 1]`, `[1, 4]`, `[2, 2]`), operand and `k_width` in `{1, 4}` goes through LDS (§12).
- **`k_width = 4` operands (conversion candidate, not a speed preference).** With warps only along M, the
  transposed fp32 accumulator already is the `k_width = 4` A operand: the conversion to `DotOperandLayout(0, mma, 4)` passes `assert_trivial=True`, while `k_width = 1` fails it, and so does `k_width = 4` with `[2, 2]` or `[1, 4]` warps; it needs no permutation code. CKPT's scan (`[1, 4]` warps, `transposed=False`) stages the state through LDS and reads it back as a `k_width = 4` operand; RSUM's summaries use `k_width = 4` with `[2, 2]` warps.
- **Relabel K (SEQ16).** Permute K inside every 16-block (k = 4y + z → 4z + y) on both operands,
  the columns of A (below) and the rows of B (`(rows // 16, 4, 4, columns)` permuted by `(0, 2, 1, 3)`): the product is unchanged, the conversion passes
  `assert_trivial=True` (S42), and the one-wave code is that of `k_width = 4`. The kernel holds the state K-permuted
  for the whole sequence (permuting it again lands it in the A layout) and un-permutes it for the final store. In a
  probe of its 64 × 128 `[4, 1]` state, the direct `k_width = 1` conversion costs 8 `ds_write_b128` + 8 `ds_read_b128`
  and 16 KB of LDS; relabeled, it has no LDS access or lane shuffle, only 32 `v_mfma_f32_16x16x4_f32`.
```python
# SEQ16, the columns of A, x of shape (rows, columns): swap the two base-4 digits of k inside every 16-block
    return gl.reshape(
        gl.permute(gl.reshape(x, (rows, columns // 16, 4, 4)), (0, 1, 3, 2)),
        (rows, columns),
    )
```

- **Two permlane-sized steps (SEQ32G, on its 16 × 16 inverse blocks, not the state).** Go through a middle
  `DistributedLinearLayout` one transposition from each end, SEQ32G's `reg_bases=[[0, 1], [0, 8]]`: on a one-wave 16
  × 16 tile the steps lower to 2 `v_permlane32_swap` and 2 `v_permlane16_swap`, no LDS. The gfx950 backend uses these
  swaps for one (register, lane 4 or 5) transposition or a 3-cycle through lanes 4 and 5 (S43); helper:
  [gluon/layouts.md](gluon/layouts.md) §7.

In one wave, a product of two 16 × 16 fp32 accumulators held in the same transposed layout takes 69 instructions with
`k_width = 4` or the relabel, 79 with the permlane route (four `v_permlane*_swap`) and 92 with a direct `k_width = 1`
conversion *(field, E17)*; all four use 1 KB of LDS, since B is not free there (the direct conversion moves both
operands through it, the others only B). In a layout-conversion probe, not a GDN kernel, a lane-moving conversion cost
0.6–1.5 µs per 16 KB tile with all programs running, 3 runs *(field, E5)* ([gluon/layouts.md](gluon/layouts.md) §8; the D
kernels' hand-written `v_permlane16_swap`: [gluon/wave_ops.md](gluon/wave_ops.md)). Fewer instructions did not mean less time at every occupancy: a 256-step, 4-wave chain on 64 × 16 tiles at 16 programs per CU took 337.2 µs with `k_width = 4` against 256.8 µs with the direct `k_width = 1` conversion, 1.313× (8 balanced repeat clusters, one run) *(field, E23)*, but at 4 programs per CU `k_width = 4` was faster for all five tile shapes tried (0.54–0.89×), and in a one-wave chain at every occupancy tried (0.54–0.80×), 1 run *(field, E26)*; no GDN kernel was timed either way.

## 12. Chunks of many tokens: LDS staging, conv windows and masks

- What the kernels stage through LDS, in shared layouts they choose (alternatives not timed):
  - V: value-major loads widened to fp32 into `SwizzledSharedLayout(4, 1, 8, [0, 1])`, read in the MFMA layout;
  - the state, when the warps split the accumulator's columns: every wave then needs whole state rows as its A operand
    (compiled with Triton 3.8.0 for gfx950, a 16 × 128 state strip in a transposed `[1, 4]` layout, as SEQ32L holds it,
    sends even the `k_width = 4` conversion through LDS: two `ds_write_b128`, eight `ds_read_b128`, three
    `s_barrier`). SEQ32L writes a `SharedLinearLayout` that keeps 4-float vectors and XOR-swizzles banks. Liveness
    is use-based ([gluon/memory.md](gluon/memory.md) §10): to keep a buffer out of later conversion scratch, call
    `_keep_alive()` after those conversions, as the Qwen3-Next 16-row kernel does after its epilogue ("avoid a shared-memory reuse
    barrier"); SEQ32L calls it right after its load. Compiled with Triton 3.8.0 for gfx950 at 16 heads, 2 sequences, 32-token chunks, 16-row value strips, a core row stride of 2048, no per-sequence chunk-offset table, 4 warps, `waves_per_eu=0` and FP fusion off, SEQ32L's recurrence kernel without that call gives the same 1108 instructions in the same order and the same resources (8192 B of LDS, 152 VGPRs, no spills): at this specialization the call changes nothing.
- Conv windows: keep a tile's token rows in each lane's registers (`BlockedLayout([1, 2], [1, 64], [1, 1], [1, 0])`) and
  take the four taps as register slices, `gl.amd.slice(raw, [BT, 128], [j, 0])` for tap j, "Four overlapping convolutions share their seven input
  rows". The slice keeps the source layout, so its offsets must fall on the layout's CTA tile (S39): with rows on
  lanes, e.g. `BlockedLayout([1, 2], [2, 32], [1, 1], [1, 0])`, the compile fails with "No source register holds the
  element for destination index [1, 0]" (compiled with Triton 3.8.0 for gfx950). Or stage the window in LDS and
  `.gather` the shifted rows. Prove a tile interior (all taps in one sequence) at run time and take a mask-free
  branch for it: in CKPT at 2048/1, every tile on the masked branch made the call 0.9–2.4% slower
  (2 runs) *(field, E19)*. [gluon/memory.md](gluon/memory.md) covers both.
- Make masks fold at compile time. A runtime mask on a pointer `gl.load` puts a branch around every access (S45).
  SEQ32G's table builder masks rows with `(chunk * BT + ri < M) | complete_chunks`, where `complete_chunks: gl.constexpr
  = M % BT == 0`. Compiled with Triton 3.8.0 for gfx950, at M = 8192 the builder issues 24 `global_load_dwordx4`
  and uses 190 VGPRs; at the ragged M = 4097 it issues 195 `global_load_ushort` behind exec-mask branches, and under the
  launch's `waves_per_eu=2`, a 256-VGPR cap, it uses 256 and spills 64 *(field, E2)*; without the cap it takes 322
  (66 of them AGPRs) and spills nothing ([tuning.md](tuning.md) §8). Its three masked operand loads as `buffer_load`
  drop their branches and, under the cap, the spill (243 VGPRs), but stay 2-byte loads. Specializing folds the mask only
  when M % BT == 0, and M already is a constexpr: at a ragged M, have the producer write zeros into the tail rows of the
  chunk-padded scratch (the packed q/k scratch holds whole 32-token chunks; its producer's store is masked at ragged M) and
  load without a mask. In SEQ32G at 4097/3 the builder then issues 24 `global_load_dwordx4` and fits 190 VGPRs at
  `waves_per_eu=2` without a spill; the call took 10% less (B/A 0.895–0.902, 2 runs), passing accuracy *(field, E19)*.

## 13. What the measurements say

- Each speedup, one per kernel and size, describes a whole design, not one knob. Decode ratios are highest up to 16
  rows (Qwen3.5 2.336–2.507 in one run, 2.360–2.507 in a later one *(field, E1)*) and 64 rows (Qwen3-Next, 2.370–3.350 in one run *(field, E4)*), where one or
  two launches replace a five-call baseline and the state stream is small (on the ruler's timer (rocprofiler-sdk tracer) launch floors are 96–97% of both sides' time at 2 rows, where the ratio, 2.49, is the launch ratio 5/2, and 80% at 16 rows, 1 run *(field, E27)*); at 128–256 rows they fall to 1.282–1.426 and
  1.457–1.782; at 256 rows both sides stream the state near a read stream's rate (1.30–1.31 there, 2.47–2.48 at
  2 rows, where the kernel's two launches take 8.55–8.87 µs and the baseline 21.20–21.78 µs; 3 runs *(field, E5)*).
- At the 2-row decode size an identical copy's 4-run mean ratio read 0.25–0.27% off the original's (single runs spread
  1.9–2.0%; two sessions); over 7 sizes the largest such offset was 0.4% and 1.3% (fp8 GEMM m = 16) *(field, E9)*: these A/A offsets call for matched controls and uncertainty estimates, not a universal 1–2% detection floor. The same protocol resolved a roughly 1% real change with four alternated runs at a different size ([tuning.md](tuning.md) §2). Ablations: one token per step §2–§5, chunks §8, §9, §12. In
  one run the hand-written Qwen3-Next decode kernels led the template by 0–8% up to 64 rows *(field, E4)*.

## Evidence

Triton paths are `triton-lang/triton@c01b677` (tag v3.8.0); `hardware/...` paths are pages of this book. "The ruler" and "the ruler's timer", its per-call time alone, are those of [tuning.md](tuning.md) §1; "(Kineto)" and "(tracer)", a rocprofiler-sdk kernel tracer, name the profiler that read the kernel durations where the record does. A speedup is the ruler's stored value against the baseline its line names.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush), speedup over sglang's five-call path] Qwen3.5 GDN decode, speedups from three runs: with the fp8 quant 1 row 2.464, 4 rows 2.422, 8 rows 2.351, 16 rows 2.336, 32 rows 1.651 in one (a later run of the ruler 2.478, 2.409, 2.405 and 2.360 at 1–16 rows); 2 rows 2.507 and, with BF16 output, 64 rows 1.461 in a second; 128 rows 1.426 and 256 rows 1.282 in a third
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel run once on the ruler (Kineto) with its accuracy check, on its own Triton cache, resources and opcodes read from each cached `.amdgcn`; for the Qwen3-Next kernels also the measured speedup at each kernel's smallest size against the run of E4 and E11] the 18 GDN decode kernels: no `v_mfma` and no `v_dot2`; Qwen3.5 recurrence VGPRs 35, 61, 61, 142, 146, 126, 47, 55, 54 at 1, 2, 4, 8, 16, 32, 64, 128, 256 rows, no spills, the 128-row kernel 16 warps at `waves_per_eu` 8; all 26 prefill kernels contain an fp32 MFMA (`16x16x4`, or `32x32x2` in 4 Qwen3.5 and 2 Qwen3-Next kernels) and a BF16 one; SEQ32G's chunk-table builder 190 VGPRs at M = 8192, 256 VGPRs with 64 spilled at M = 4097 under `waves_per_eu=2`, 322 and no spill without it; the 20 Qwen3-Next kernels: every measured speedup within 4.7% of the earlier run's (largest deviation of the ratio from 1, 0.04654), and every one of them failed this run, by an accuracy failure or a nonzero exit (the 44 GDN kernels)
- E3 [2026-11, gfx950 (MI355X), Triton ?, N=?, not stated: the kernel author's note in the source] `waves_per_eu=4` at 16 and 64 rows about 19% slower than unset, VGPRs pinned at 128 by spilling; inline history retirement forced at 32 rows 12% faster and wrong, max |d| 8e-1 on the state (comments in the Qwen3-Next template's config)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush), speedup over the Qwen3-Next baseline of §1] Qwen3-Next GDN decode, per-size kernels: 1 row 2.443, 2 rows 2.538, 4 and 8 rows (one kernel) 2.370 and 2.506, 16 rows 3.222, 32 rows 3.350, 64 rows 2.462, 128 and 256 rows (one kernel) 1.782 and 1.425, the third design at 128 and 256 rows 1.466 and 1.151; the template at 1–256 rows 2.390, 2.450, 2.370, 2.479, 3.101, 3.091, 2.401, 1.755, 1.457; one run
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (N=4 for the 1–4-row call times), the ruler (Kineto) with its accuracy check, on the decode kernels and modified copies, 4 alternated rounds per arm, kernel time the mean of the rounds' median kernel-side times, variants that fail accuracy run in-process on random inputs against the unmodified kernel's output; Gluon probe kernels: a 1 GiB tensor streamed once per launch (2048 programs × 4 warps, 16 bytes per lane unless stated), median of 9 rounds × 10 launches, variants interleaved; a [64, 64] 32-bit tile converted A → B → A 1024 times in 1024 programs, median of 7 rounds × 3 launches; ranges over the runs] the 2-row kernel at 2 rows with 16-row value tiles (256 CTAs) 8.59–8.73 µs against 10.55–10.62 µs with whole-head tiles of the same kernel (128 rows, 32 CTAs, 267 VGPRs), 17.7–18.6% less, at 16 rows 14.76–14.78 against 13.19–13.23 µs; tiles of 8, 16 and 32 rows at 2 rows 8.640–8.655, 8.631–8.665 and 8.601–8.714 µs; its history shift moved into launch 1: `conv_state` wrong on 10–11 of 20 inputs per run, `delta_state` off by up to 0.95–1.46, the unmodified kernel exact on all 20 in every run; kernels run at other row counts: at 24 rows the 16-row kernel 1.821–1.833 and the 4-row kernel 1.809–1.823, at 192 rows the 128-row kernel 1.339–1.342 and the 4-row kernel 1.042–1.043; the Qwen3-Next template at 32 rows with inline history retirement: `delta_state` wrong on 10 of 10 inputs, max |d| 3.243 in every run, warm-cache graph replay B/A 0.986–0.999 (7.48–7.62 against 7.58–7.63 µs); the 32-row kernel, state load after the gates with the SiLU early: B/A per pair 1.000–1.018 (17.91–18.04 → 18.16–18.27 µs); the 8-row kernel, q loaded and normalized before the state load 1.018–1.039 (9.47–9.49 → 9.67–9.76 µs), decay and β in each update program 1.062–1.074 (9.47–9.53 → 10.13–10.16 µs); the 1-row kernel against scalar exponentials (7 against 4 `v_exp_f32`) B/A per pair 0.986–1.013 (8.65–8.73 against 8.64–8.71 µs); 8 readers of one chunk, ids congruent mod 8 12.1–12.4 µs against 21.6–22.0 µs consecutive (1.75–1.80×); the 128-row kernel with `token = pid // VH` B/A per pair 1.005–1.011 (45.62–45.93 → 46.09–46.21 µs); state read plus write 5.85–5.89 TB/s by the 128-row kernel (45.6–45.9 µs) and 6.57–6.59 TB/s by the 256-row kernel (81.5–81.7 µs), 1.06–1.20 of a 5.48–5.51 TB/s `.cg` read stream, the 256-row baseline 106.4–106.7 µs (0.91–0.92 of it; ratio 1.30–1.31), at 2 rows 8.7–8.8 and 21.6–21.7 µs (0.087–0.088 and 0.035; ratio 2.47–2.48); the 2-row kernel at 2 rows: two kernels in every Kineto trace, median call time per timing 8.55–8.87 µs (run means 8.65, 8.75, 8.68 µs), its baseline 21.20–21.78 µs, 4 timings per run; the 1-, 2- and 4-row kernels 8.62–9.04, 8.81–9.14 and 8.81–9.13 µs in three runs and 9.09, 8.65 and 9.10 µs in a fourth (t2/t1 0.951–1.022, t4/t1 1.001–1.022), the 2-row call 0.46–0.485 TB/s against 6.72–6.78 TB/s for a 1 GiB `.cg` Gluon read stream (0.068–0.072 of it); 256 programs × 4 warps × 8 loads per lane 7129–7144 GB/s, 2048 × 4 × 1 6674–6684 GB/s (1.067–1.070×); at 3 MB in flight 96 × 4 × 8 3945–3995 GB/s, 384 × 4 × 2 4920–4962 GB/s, 192 × 1 warp × 16 2454–2467 GB/s (0.80–0.81× and 0.495–0.499× of 384 × 4 × 2); reads: no hint 5993–6013 GB/s, `.cg`/`.cs` 6771–6837 GB/s (1.130–1.137×), buffer over pointer 1.002–1.005 hinted and 1.002–1.003 unhinted; writes: no hint 5792–5889 GB/s, `.cs` 5404–5510 (0.928–0.945×; buffer over pointer 0.997–1.012), `.wt` 0.998–1.005× through `buffer_store` (above 1.00 in two runs) and 0.983–0.996× through `gl.store`, below every `gl.store` without cache bits (none, `.wb`, `.cg`) in each run; lane-moving conversions 574–1522 ns per conversion against 73–74 ns for the loop alone, the 96-`ds_bpermute` path 661–663 ns, the vectorized LDS round trip 574–581 ns (Qwen3.5 GDN decode kernels; the Qwen3-Next template; probe kernels)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, each kernel called 5 times at its first and last size on fresh copies of the inputs, outputs compared byte by byte; sources scanned for atomics] all 44 GDN kernels: no atomic call; 61 kernel-sizes, 305 calls, every repeat byte-identical to the first; the 13 sizes between a kernel's first and last not run (the 18 decode and 26 prefill kernels)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the ruler (Kineto), 2 rounds per arm, arms alternated in a fixed order, one GPU] Qwen3.5 GDN fp8 decode kernels run at row counts between their sizes, the log-nearest kernel against the other neighbour, call time (first run / second): 3 rows the 4-row kernel 8.97 / 8.87 against the 2-row 8.90 / 8.85 µs; 6 rows the 8-row 9.18 / 9.18 against the 4-row 9.10 / 8.98; 12 rows the 16-row 10.37 / 10.35 against the 8-row 10.54 / 10.62; 24 rows the 32-row 16.23 / 16.02 against the 16-row 15.59 / 15.48; 48 rows the 32-row 23.25 / 23.50 against the 128-row 21.53 / 21.60; 96 rows the 128-row 36.39 / 36.38 against the 32-row 39.26 / 39.92
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 100 repeats per arm against a CPU model, compiled and run on one GPU] a toy Gluon kernel shifting a 3-entry history of 64 channels that every wave reads and wave 0 writes, a spin delaying the readers: the new entry stored before the reads was read in place of the old one in every repeat, 64 wrong values at one wave, 256 at four; no barrier between the reads and the writes: 192 wrong at four waves in every repeat, none at one wave; a full-workgroup barrier before the writes: none wrong at either (toy kernel)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 sessions for the A/A part, N=1 for the state-access A/Bs, the ruler (Kineto), one GPU, 4 runs of A alternated with 4 of B in one session] A/A, a kernel against an identical copy, speedups: the 2-row kernel at 2 rows 2.4222, 2.4105, 2.4522, 2.4595, the copy 2.4254, 2.4118, 2.4434, 2.4374: 1.73% spread over the first 3, 2.03% over the 8, the 4-run means 0.27% apart; over 7 sizes the largest gap between the two 4-run means 1.31% (fp8 GEMM m = 16, its single runs spread 2.4%); second session: 2.4914, 2.4960, 2.5015, 2.5065, the copy 2.4728, 2.4816, 2.4956, 2.5204: 0.41% over the first 3, 1.92% over the 8, means 0.25% apart, largest gap over the 7 sizes 0.42% (fp8 GEMM m = 16, single runs spread 2.5%); state-access A/Bs: `.cs` removed from the 2-row kernel's state `buffer_load`/`buffer_store`: speedup 2.452 → 2.442, 8.748 → 8.789 µs (pairs 0.99–1.015); the state as `gl.load`/`gl.store` with `.cs`: 2.429 → 2.432, 8.827 → 8.813 µs (pairs 0.983–1.02); `.wt` dropped from the state `buffer_store` of the 128-row and 256-row kernels: 45.77 → 46.97 and 81.79 → 83.90 µs, the store as `gl.store` with `.cs`: 45.80 → 47.04 and 81.81 → 84.78 µs
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (slope and two-kernel readings repeated in four more runs), Gluon probe kernels on one GPU: chained trivial 1-warp kernels in a CUDA graph, timed by the ruler's timer (Kineto, `GPU_FORCE_QUEUE_PROFILING=1`) and by the bare replay's wall clock, eager launches by the host clock] the reading of the ruler's timer grows 4.062 µs per added kernel (Kineto sum 3.902), a bare replay 1.815 µs (grid 1) and 1.844 µs (grid 256); two chained kernels read 8.246 µs on the ruler's timer; eager host time 1.004 µs per tensor argument, while the ruler's timer reads 4.162–4.178 µs for kernels with 1, 8 and 32 tensors; the repeats, with and without queue profiling forced: the ruler's timer 4.045–4.082 µs per kernel, bare replay 1.766–1.771 µs, two chained kernels 8.25–8.40 µs (toy kernels)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush), speedup over the baseline of §1] Qwen3.5 GDN prefill, speedups per size as in §8, from five runs: D 1024/1 2.423, D 512/1 2.638 and 1024/2 1.889, RSUM b 16384/2 at `max_seq_len` 8192 1.339 in the first (a later run of the ruler: D 1024/1 2.452, D 512/1 2.681 and 1024/2 1.927); RSUM b 16384/2 at `max_seq_len` 10000 1.444, MAP 10000/1 1.188, CKPT 1025/1 2.256, RSUM b 16384/1 1.741, RSUM a 5000/1 1.199 and 15000/3 1.198, MAP 8192/1 1.193, SEQ16 b 15361/8 1.454 and 16384/8 1.475, CKPT 2048/1 2.092 in the second; D 512/2 2.629 in the third; SEQ32L 2049/2 1.374 and 4096/2 1.452 in the fourth; SEQ32G 4097/3 1.029 and 8192/4 1.424, SEQ16 a 8193/5 0.997 and 15360/8 1.473 in the fifth; Qwen3-Next GDN prefill, one run, source / twin: snapshot scan 1024/1 1.573, 2048/2 1.434; multi-schedule 3072/3 1.592 / 1.554, 4096/4 1.504 / 1.501, 8192/8 1.490 / 1.484, 12288/12 1.288 / 1.282, 16384/16 1.463 / 1.460; compact factors 13312/13 1.232 / 1.233, 15360/15 1.328 / 1.329; multi-schedule 4096/1 1.718 / 1.719, 5120/1 1.727 / 1.741, 10240/2 1.308 / 1.311; reverse summaries 16384/4 1.294 / 1.283
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, a register-operand loop of one MFMA shape per kernel on all CUs] BF16 `16x16x32` 2447–2468 TFLOP/s, fp32 `16x16x4` 155.2–156.1 TFLOP/s, 15.71–15.84× (Gluon probe kernel)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the prefill graph capture N=3), the ruler (Kineto) and its accuracy check with poisoned scratch, on the kernels named and on all 26 GDN prefill kernels; JIT binaries counted around a CUDA-graph capture (relaxed mode); the sglang integration of the prefill kernels called with mocked sglang layers] a copy of the 8-row kernel that skips the `scales` store of odd rows: a numeric mismatch at the first captured call, 64 non-finite mismatches in `scales`; the unmodified kernel: no finding; the 16-row kernel with the pool size as a `gl.constexpr`: 1, 2, 3, 4 binaries after pool sizes 33, 34, 35 and, under capture, 36; with a runtime pool size 1, 1, 1, 1; no capture error; the 26 prefill kernels, each with the fresh-state zeroing kernel of §7: all timed (capture under `capture_error_mode='global'`, replay with new inputs), no capture, accuracy or mutation finding, exit 0; the mocked Qwen3.5 and Qwen3-Next prefill integrations ran eager, declined the call under capture and ran eager again (call counters 1, 1, 2)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, kernels compiled for gfx950, VGPRs, waves per SIMD and instructions read from the `.amdgcn`; probes and the kernels run on the GPU against fp64, torch or the op's reference; one A/B on the ruler (Kineto) of 4 rounds] the 2-row kernel with q read together with k: identical instructions, 61 VGPRs; the Qwen3-Next template, state load early against late: +7 VGPRs at 1 row (65 as built), 0 at 32 (160), +77 at 128, +12 at 256, no spills; accumulate against recompute form in its fused-heads path: 138 against 134 VGPRs at 16 rows, 131 against 132 at 64, 3 waves per SIMD, no spills; decay and β before the conv at 128 rows: 200 VGPRs either way, 2 waves per SIMD; at 32 rows the late state load 10.28 µs against 8.39 µs early, 160 VGPRs each (speedup 3.18 → 2.58; the other 8 sizes 0.997–1.006); one-wave product of two 16 × 16 fp32 accumulators: 69 instructions with `k_width = 4`, 69 relabeled, 79 with two permlane steps (4 swaps), 92 direct, 1024 B of LDS each; relabeled result 9.727e-8 relative to fp64 (`k_width = 4` the same, plain route 1.007e-7); `k_width = 1` from the transposed accumulator 4 `ds_write_b32` + 4 `ds_read_b32`, 1024 B, no `ds_bpermute`, a one-transposition step 2 `v_permlane32_swap`; 64 × 64 tiles, 4 waves: only (transposed, `[4, 1]`, A) and (not transposed, `[1, 4]`, B) at `k_width = 4` pass `assert_trivial`; SEQ16's 64 × 128 `[4, 1]` state: direct 8 `ds_write_b128` + 8 `ds_read_b128`, 16384 B, relabeled no LDS, 32 `v_mfma_f32_16x16x4_f32`; 2^20 random BF16 pairs with exponents within ±63: every fp32 product equals the fp64 one; 16×32 @ 32×16 tiles (4194304 elements), the fp32 operand split into BF16 terms: equal to the correctly rounded BF16 in 58.2% (high term), 99.76% (two RNE terms), 100% (RTZ, RTZ, RNE); 300000 random fp32 values with exponents within ±40: two RNE terms split error at most 7.605e-6 relative, three RTZ terms exact; 10000 chunks of 64 tokens, log-decays uniform in [−6, 0]: the prefix ratio NaN at 4,242,274 and 0 at 4,179,454 of 14,078,262 entries whose fp64 value is a normal fp32, a masked add-scan and one exponential within 3.76e-6 relative; padded rows of the 2-row and 8-row kernels: codes zero-valued, about half of the bytes 0x80, every scale 0x2a7b5124 = 2.23214296e-13; the Qwen3-Next snapshot-scan kernel within the accuracy bounds at 128 sequences, at 129 normalized excess 161, with eight halvings within them at 129; one recurrence step on a random 128 × 128 fp32 state with decay 1 and β 0: no bit changed, v NaN at rows 3 and 17 made those rows NaN; the 1-row kernel with its BF16 core scratch replaced by an fp32 scratch of the BF16-rounded core: outputs bitwise equal on 10 of 10 inputs, 4 of them not padding; with the unrounded core equal on the 6 padding inputs only; fp32 `.to(float8e4nv)`, `.to(float8e5)` and an fp32 store into an e4m3 pointer: one MODE bit-23 write (value 1) at instruction 6, before the first VALU; the 2-row kernel's norm and quant kernel: bit 23 set at instruction 6, before its first `v_cvt_pk_bf16_f32` (instruction 173); with the bit set fp32 → BF16 of finite overflowing values 0x7F7F/0xFF7F and of Inf 0x7F80/0xFF80, with it clear 0x7F80/0xFF80; the Qwen3-Next 1-row kernel's IEEE BF16 rounding next to a Triton fp8 store: no bit mismatch against torch on 60416 values; the template's config rows: IEEE fix-up at 1 row, native convert at 2, 4, 8 and 32, neither at 16, 64, 128 and 256; 16 re-splits of `cu_seqlens` at the 8 multi-sequence prefill sizes with unequal lengths changed no grid; 12000 random ragged batches: chunk slots distinct and below `cdiv(m, BT) + batch`; fp32 `torch.empty` scratch per call at the 8–16K-token summary sizes 173.1 (RSUM a, 15000/3) to 294.0 MB (MAP, 10000/1), MAP's factors there 246.2 MB and maps 35.7 MB (Qwen3.5 and Qwen3-Next GDN kernels; probe kernels)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the ruler (Kineto), 4 alternated rounds per arm on one GPU, A = the Qwen3-Next template, B = a copy whose 16- and 64-row configs get `waves_per_eu=4`] B takes 1.598 and 1.603× (16 rows) and 1.595 and 1.629× (64 rows) as long as A (speedup ratio 0.62 and 0.62 at 16 rows, 0.63 and 0.61 at 64), both B kernels at 128 VGPRs with 9 and 20 spilled, 2.6% and 5.3% more instructions
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the ruler (Kineto) with its accuracy check, on the Qwen3.5 prefill kernels, modified copies and kernels run at another kernel's size, 4 rounds per arm alternated, time = the median kernel-side call time; compiles for gfx950 read from the `.amdgcn`; values given first run / second] 2048/1: CKPT 107.3 / 106.9 µs, RSUM b run at that size 214.2 / 212.9 µs; SEQ16 b over MAP at 8192/1 2.712 / 2.708×, over RSUM b at 16384/1 3.977 / 3.968×; mean call time, the size's own kernel against another design's kernel run there (the transplanted kernels' accuracy not checked): 1024/1 D 67.4 / 67.2 against CKPT 68.3 / 67.9 µs, 4096/2 SEQ32L 193.7 / 192.9 against SEQ16 b 374.5 / 375.5 µs, 8192/1 MAP 495.3 / 497.2 against RSUM b 363.9 / 366.0 µs, 16384/8 SEQ16 b 507.4 / 507.1 against RSUM 651.1 / 652.3 µs; t(4097/3)/t(8192/4) 0.974 / 0.974 (SEQ32G), t(8193/5)/t(15360/8) 0.855 / 0.856 (SEQ16 a), work ratios 0.50 and 0.53; the other value strip, BV 16 at 4097/3 1.317 / 1.317× and BV 32 at 8193/5 1.153 / 1.153× the kernel's own time; SEQ16 b without the `gl.barrier()` before its conv-history write (the one-wave boundary program, 74 VGPRs either way), 10 runs per arm at 15361/8 and 16384/8: no accuracy finding in either arm, `conv_state` exact in all 10 B runs / all 10; SEQ16 b at 16384/8, B/A per pair: next chunk's loads after the math (218 against 222 VGPRs) 1.0196, 1.0211, 1.0199, 1.0233 / 1.0223, 1.0242, 1.0224, 1.0231, `.cg` dropped on the value tile 0.9986, 1.0032, 0.9984, 1.0011 / 1.0023, 1.0007, 1.0006, 1.0012; CKPT at 2048/1, B/A call time per pair: one 128-deep chain (576 MFMAs, 108 against 114 VGPRs) 1.0453, 1.0405, 1.0313, 1.0459 / 1.0386, 1.0426, 1.0426, 1.0421; unroll 1 (64 MFMAs against 576) 1.0675–1.0694 / 1.0629–1.0701; `iterative-ilp` (present in A's LLVM IR) removed 1.1183–1.1334 / 1.1224–1.1294; masked branch forced 1.0237, 1.0166, 1.0124, 1.0202 / 1.0087, 1.0158, 1.0144, 1.0113 (9 dwordx4 and 27 ushort loads, 58 VGPRs in both arms); SEQ32G's chunk-table builder at M = 8192, half tables stored once final 190 VGPRs, all stores at the end 194, no spill (compile only, the same in every session); speedups D 2.646 / 2.654 at 512/1 and 1.913 / 1.906 at 1024/2, RSUM b 1.737 / 1.730 at 16384/1 and 1.359 / 1.360 at 16384/2; SEQ32L at 4096/2: K Kᵀ and Q Kᵀ in each value head (the chunk-factor kernel 160 against 136 MFMAs, 254 against 170 VGPRs), accuracy passed, B/A 0.9777, 1.0235, 1.0225, 1.0221 / 1.0173, 1.0236, 1.0233, 1.0210 (means 1.011 / 1.021); the zero blocks multiplied too (184 against 136 MFMAs, 182 against 170 VGPRs) 0.9655, 1.0063, 1.0060, 1.0077 / 1.0005, 1.0059, 1.0062, 1.0017; SEQ32G at 4097/3 with zeros stored into the packed q/k scratch's tail rows and unmasked loads: builder 24 `global_load_dwordx4` and 3 ushort loads, 190 VGPRs, no spill, against 0, 195 and 256 + 64 spilled, `waves_per_eu` 2 in both, accuracy passed in every B run, B/A 0.8957, 0.8965, 0.9024, 0.8965 / 0.8966, 0.8957, 0.8983, 0.8950 (Qwen3.5 GDN prefill kernels)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compiled without a GPU: 38 fp8 cast variants and 7 controls, MODE writes read from the `.amdgcn`] each of the 38 E4M3FN/E5M2 casts from fp32, fp16 and bf16 (default and rtne `.to`, E5M2 rtz from fp32 and fp16, an implicit cast in `gl.store`, an explicit cast then `buffer_store`, a runtime branch) wrote MODE.FP16_OVFL = 1 once, before the first VALU of the function holding the cast: the kernel when inlined, the callee in the six noinline variants; no later setreg; a direct wider-value `buffer_store` and E4M3FN rtz did not compile; the seven controls (upcast, bitcast, MFMA, asm, BF16) wrote no MODE (toy Gluon kernels)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 study of 5 sizes, 16 adjacent pairs (8 AB, 8 BA) per comparison, eager single calls timed by device events (host enqueue and allocation gaps included), 99.5% bootstrap intervals per comparison with A/A guards] sizes (tokens, sequences, max_seq_len) (512, 1, 512), (1024, 2, 512), (512, 2, 256), (1024, 1, 1024), (1025, 1, 1025); D, CKPT, RSUM b, MAP, SEQ32L, each on its own host branch without tuning: all 25 size-arms passed the accuracy bounds against an fp64 sequential reference on every full mutable tensor (2 seeds or state modes × 2 scratch and output poisons, 500 comparisons), unindexed pool slots untouched, outputs independent of the poison; two-sequence inputs of equal length only; of 35 A/B comparisons, D and CKPT each faster than RSUM and MAP at all five sizes, CKPT faster than D at all five, CKPT against SEQ32L unresolved at the second and third (Qwen3.5 GDN prefill kernels)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compiled without a GPU: 7 four-wave builds; an earlier sweep of 432 builds, compiled twice] the `16x16x4` accumulator as the A operand at `k_width = 1`: two transpositions and 16 KiB of LDS at 128 × 32, transposed or not; the transposed `32x32x2` accumulator at 256 × 64: one, no LDS, 32 permlane32 instructions; untransposed: four, 64 KiB of LDS; in the sweep, conversions stayed in-wave whenever fewer than two transpositions were needed and the warps did not change (toy Gluon kernels)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 8 balanced repeat clusters] a 256-step chain of products on 64 × 16 fp32 tiles, 4 waves, 16 programs per CU: `k_width = 4` operands 337.2 µs against 256.8 µs with the direct `k_width = 1` conversion (1.313×), values correct, largest relative error 2.587e-6 (toy Gluon kernel)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, a one-row Triton toy, 16 key heads × 4 tile CTAs (4-tap conv, SiLU, L2 norm), outputs compared; CUDA-graph replay time and the ruler's timer (Kineto)] recompute over staged replay time 0.709, 0.693 and 0.775 in three runs, 1.014 in the fourth (9.967 against 9.832 µs, relative L2 difference of the outputs 0); on the ruler's timer 0.589–0.612 in all four (5.117 against 8.691 µs in the fourth) (toy kernel)
- E25 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 5 rounds per arm, median of the per-round ratios; the ruler's timer (tracer) and a cold-state CUDA-graph replay] Qwen3.5 decode, time ratios (timer / replay): the 2-row kernel with whole-head value tiles over its 16-row tiles 1.071 / 1.093 at 8 rows, 0.870 / 0.866 at 16; the faster of the 8-row and 16-row kernels over the 2-row kernel in the replay 1.345, 1.305, 1.183 at 1, 2, 4 rows (timer: all three 8.93–9.24 µs there), below 1 from 8 rows (timer 0.673–0.899, replay 0.716–0.931); only the value tile changed, replay 512 over 256 CTAs: the 1-row kernel at 2/4 rows 0.958/0.982, the 2-row kernel 1.040/0.939
- E26 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, CUDA-graph replay, 7 rounds, median of the per-round ratios] E23's 256-step chain, `k_width = 4` over direct `k_width = 1` time: at 4 programs per CU 0.543–0.890 in all five tile and wave shapes; 1 wave 32 × 16 0.543, 0.708, 0.803 at 4, 8 and 16 programs per CU (toy Gluon kernel)
- E27 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, per-kernel times on the ruler's timer (tracer), 5 rounds per arm, medians] Qwen3.5 decode, the kernel against the five-call baseline, a trivial kernel 4.26 µs: 2 rows, 2 launches against 5, ratio 2.486, launch floors 0.959 and 0.966 of each side's time; 16 rows floors 0.797 and 0.800
- E28 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, CUDA graph of 10 launches per arm, 7 rounds rotated, an arm's GB/s over the same round's unhinted arm, median] fp32 Gluon stream probes at 256 MiB, 1 GiB and 4 GiB, one range per program or grid-stride: hinted loads 1.064–1.132 at 1 and 4 GiB, 0.965–1.012 at 256 MiB; `.cs` and `.wt` stores 0.822–1.178; every output checked (probe kernels)
- S11 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:467 `waves_per_eu` N becomes `amdgpu-waves-per-eu` = "N,N" (:462-468)
- S16 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp:719 a barrier on local and global spaces becomes release fence, `s_barrier`, acquire fence (:719-736); TritonGPUToLLVM.cpp:113 Membar analysis runs before lowering (:113-115)
- S18 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:492 buffer-op `.cs` sets sc0 and nt; pointer loads and stores get only the nontemporal flag for `.cs` (:305-336)
- S19 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:104 `num_records` INT_MAX − 1 bytes; element offset times element size in 32 bits and masked lanes selected to an out-of-range offset :282-283
- S20 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp:433 `ClampFOp` lowers to `ROCDL::FMed3Op` (:433-451)
- S21 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp:2470 an fp8 destination type sets the FP8 clamping attribute on the function (:2470-2484); ElementwiseOpToLLVM.cpp:468-498 `adjustModeRegister` then inserts `llvm.amdgcn.s.setreg` of MODE bit 23 = 1 at the function entry, called from TritonGPUToLLVM.cpp:250
- S37 hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:81 Table 28: F32 16x16x4 32 cycles; BF16 16x16x32 16 cycles at :101
- S38 triton-lang/triton@c01b677:lib/Analysis/Utility.cpp:1096 a gather is warp-local only if no warp or block basis touches the gather axis; :1311-1321 `cvtNeedsWarpShuffle`: fewer than two mixed transpositions; :526-531 `ScanLoweringHelper::isSupported` accepts only a blocked encoding; otherwise lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp:474 emits "TODO: unsupported scan layout"
- S39 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/slice.py:12 `gl.amd.slice`: register-only slice, no cross-thread movement (:12-23)
- S40 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py:12 the cdna4 module re-exports `buffer_load`, `buffer_store` and `mfma` (:12-18)
- S42 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245 `assert_trivial` raises "layout conversion from … is not trivial" with both linear layouts unless the conversion is trivial (:245-248)
- S43 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertLayoutOpToLLVM.cpp:44 permlane swap only for exactly one (register, lane 4 or 5) transposition or a 3-cycle through lanes 4 and 5 (:44-87)
- S45 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MaskedOpsToLLVM.cpp:81 a mask that is not constant true becomes a conditional branch around the load (:81-104)
