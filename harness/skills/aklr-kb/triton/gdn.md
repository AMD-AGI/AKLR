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

At 43e112e AKLR had 44 Gluon kernel files for the GatedDeltaNet (linear attention) layers of Qwen3.5 and Qwen3-Next: 18
files that advance each sequence by one token (contracts `*gdn_decode*`, "decode" below) update a 128 × 128 fp32 state
per value head on the VALU, 26 files that take each sequence in chunks of many tokens (`*gdn_prefill*`, "prefill"
below) run the recurrence on the matrix cores with sequential recurrence stages. All fuse the gated RMSNorm, all but one decode file the fp8 group
quant. Dispatch served only the 24 Qwen3.5 files at 43e112e; since d908d01 it also serves 14 Qwen3-Next files. Later, #75
added a 64-row file to the Qwen3.5 FP8 decode op (9 files) and 11 to its BF16 op (12: rows 1, 2, 4, 8, 16, 32, 64 twice,
96, 128, 192, 256), #65 five Qwen3-Next decode files (14 over 36 Points); none of them stores a speedup at 0e02a81.
Since #76 the BF16 op alone serves the one-token step of layers whose projections sglang runs unquantized (`_decode_unquantized` in
`aklr/plugins/sglang/models/qwen3_5/api.py`). File counts and tables below are those of 43e112e.

## Route here when

- You write or tune a GDN kernel for an AKLR schema, one token per sequence step (§2–§6) or chunks of many tokens
  (§7–§12), or another recurrence that keeps a matrix state per sequence slot (`run`, scratch and validation: [authoring.md](authoring.md)).
- You choose, per row count (one token per row), CTAs that split a head's state or own whole heads, and where the second launch goes.
- You fuse an fp8 group quant into a kernel that also makes BF16 or fp32 MFMA results (FP16_OVFL; [numerics.md](numerics.md) §2).
- You pick a chunk size and schedule for a (tokens, sequences) Point, need an fp32 MFMA accumulator as the next
  MFMA's operand without a trip through LDS, or must handle ragged sequences without host reads.

## 1. What the op computes, and the kernels AKLR has

Per row of the one-token contracts (one token with its own state slot; slot −1 is padding: no state change, zero core output) (S1):

1. Causal width-4 convolution of the Q/K/V channels with the slot's 3-token history (plus `conv_bias` where the contract
   has it), SiLU, BF16; the history becomes `[h1, h2, x]` (S1), checked to within its BF16 rounding (S2).
2. q and k L2-normalized, `x · rsqrt(Σx² + 1e-6)`; q times `scale`.
3. β = sigmoid(b) rounded to BF16; decay g = exp(−exp(A_log) · softplus(a + dt_bias)).
4. On the fp32 state S of the (slot, value head), 128 × 128 in `[value, key]` order (64 KiB):
   `S ← g·S;  u = β(v − S k);  S ← S + u kᵀ;  core = S q`, rounded to BF16.
5. Gated RMSNorm with silu(z), BF16; then per value head (the fp8 group: its 128 values) E4M3FN codes, one fp32 scale.

Where the reference rounds differs per model and regime: read the contract's description and the reference function
(Qwen3.5's `reference_prefill` has no docstring). Qwen3.5 decode rounds each conv tap product to BF16, Qwen3-Next sums
exact products; both prefills round the normalized q and k to BF16 and multiply the core S q by `scale`, where decode
multiplies q (S1; [numerics.md](numerics.md) for rounding points). Prefill runs the recurrence over each sequence of a
ragged batch (`cu_seqlens`) in the chunked WY form, as one kernel's docstring states it (S3):

```python
# 7a69337f81190952.json#kernel.py:251-257 (Qwen3.5 prefill): G relative decay, c cumulative decay
    With cumulative decay c and relative decay G, solve
      (I + tril(beta * G * (K @ K.T), -1)) [W, U]
          = [beta * c * K, beta * V].
    Replay uses R = U.T - S @ W.T, then
      Y = S @ (c * Q).T + R @ response,
      S_next = c_last * S + R @ Ktail.
    Products of decays are formed directly, without dividing tiny prefixes.
```

Here `response` in the replay equation is the transpose of the chunk's causal decayed attention, tril(Q Kᵀ ⊙ G)ᵀ (the untransposed matrix has i ≥ j), `Ktail` is K scaled by each
token's decay to the chunk end, G(last, j), W and U solve the triangular system, R = Uᵀ − S Wᵀ is the residual (S3).

| model, contract | files at 43e112e | Points | launches per call | writes |
|---|---|---|---|---|
| Qwen3.5 `gdn_decode_group_fp8_quant@1.0` | 8 | 1, 2, 4, 8, 16, 32, 128, 256 rows, one file each | 2 | `values`, `scales` |
| Qwen3.5 `gdn_decode@1.0` (BF16 output) | 1 | 64 rows | 2 | `normalized` |
| Qwen3.5 `gdn_prefill_group_fp8_quant@1.0` | 15 | 21: 512–16384 tokens in 1–8 sequences | 6–7 | `normalized`, `values`, `scales` |
| Qwen3-Next `qwen3_next_gdn_decode_group_fp8_quant@1.0` | 9 | 1–256 rows (9 Points; one template file lists all nine) | 2 up to 8 rows, 1 from 16 | `normalized`, `values`, `scales` |
| Qwen3-Next `qwen3_next_gdn_prefill_group_fp8_quant@1.0` | 11 | 13: 1024–16384 tokens in 1–16 sequences | 5–8 | `normalized`, `values`, `scales` |

- All five also update `conv_state` and `delta_state` in place (S2, S48). Bench fills the outputs with NaN, so a skipped
  row fails ([authoring.md](authoring.md) §8; a `4f25` copy skipping half its `scales`: 64 mismatches *(field, E16)*).
- Qwen3.5 has 16 value heads and 4 key heads, Qwen3-Next 8 and 4; key and value width is 128; value head h reads key
  head h // (value heads / key heads). Qwen3.5's projections are contiguous: `projected_qkvz` holds every Q, every K,
  every V, then every Z, `projected_ba` every B, then every A. Qwen3-Next's are grouped by key head: its q, its k, then
  the v and the z of its value heads; its B's, then its A's. `conv_state`, `conv_weight` and `conv_bias` keep every Q,
  every K, every V in both models, so a Qwen3-Next kernel reads conv channel c from another projection column (S48).
  Both prefills and Qwen3-Next decode add `conv_bias`; Qwen3.5 decode has none. The decode state is 1 MiB read and
  1 MiB written per row (Qwen3.5), 512 KiB each way (Qwen3-Next).
- The Qwen3.5 decode baseline makes five calls: sglang's projection split, conv update, recurrent decode and gated
  norm, then aiter's per-1×128 quant; both Qwen3-Next baselines end with the same aiter quant, the Qwen3.5 prefill
  baseline with sglang's `per_token_group_quant_fp8` (S4).
- Decode uses no MFMA *(field, E2)*: per (row, head) two 128 × 128 matrix-vector products and a rank-1 update on a
  64 KiB state no other row reads, 0.875 flop per byte (estimated). All 26 prefill files use both the BF16 and the
  fp32 MFMA *(field, E2)*. No file uses atomics. Bitwise determinism is required (S2); five calls on fresh copies of the inputs were byte-identical at each file's first and last listed Point (61 file-Points, 305 calls; the 13 Points between them not run) *(field, E6)*.
- The 20 Qwen3-Next files store no speedup at 43e112e, so dispatch there served none of them at the default
  `AKLR_MIN_SPEEDUP` (`aklr/ops/README.md` § Environment variables); their numbers below are from the CI run recorded
  in d908d01, 43e112e's child. A later validate of each read its speedup within 4.7% of that run at the schema's lowest
  Point, but none of the 20 passed: each failed accuracy or exited nonzero *(field, E2)*.
- Files are named by their uuid's leading hex digits: `<model package>/impls/triton/<scenario>/<uuid>.json`, sources
  inside ([authoring.md](authoring.md) §3); the Qwen3.5 prefill files sit in nine scenario directories by
  `max_seq_len`, and `ls aklr/plugins/sglang/models/qwen3_*/impls/triton/*/<digits>*.json` finds one.

## 2. One token per step: split each head's state at small row counts, own whole heads above

With few rows the files cut each head's 128 value rows into tiles, one CTA per tile (`29913388`: 17.7–18.6% less time than
whole-head tiles at 2 rows, 11.6–11.9% more at 16, 3 runs *(field, E5)*; crossover and cause not measured). Both families
switch to whole heads, with the epilogue fused, where rows × value heads reaches 128: Qwen3.5 at 8 rows, Qwen3-Next at 16
(from the grids). In Qwen3.5 the tile is 8 × rows value rows at 1, 2 and 4 rows: 256 recurrence CTAs, a target of one per CU
([tuning.md](tuning.md) §6), a derived target (at 2 rows 128–512 CTAs all took 8.60–8.71 µs, 3 runs *(field, E5)*):

| rows | file | recurrence CTA | CTAs | VGPRs | CI |
|---|---|---|---|---|---|
| 1 | `2c2a09e2` | 2 waves, 8 value rows, `waves_per_eu=1` | 256 | 35 | 2.464 |
| 2 | `29913388` | 2 waves, 16 value rows, `waves_per_eu=1` | 256 | 61 | 2.507 |
| 4 | `409cb22b` | 4 waves, 32 value rows | 256 | 61 | 2.422 |
| 8 | `4f257a3e` | 4 waves, whole head | 128 | 142 | 2.351 |
| 16 | `a4f80b6a` | 4 waves, whole head | 256 | 146 | 2.336 |
| 32 | `20f63094` | 4 waves, whole head | 512 | 126 | 1.651 |
| 64, BF16 op | `cd00c50e` | 16 waves, whole head, `waves_per_eu=6` | 1024 | 47 | 1.461 |
| 128 | `fc261961` | 16 waves, whole head, `waves_per_eu=8` | 2048 | 55 | 1.426 |
| 256 | `c2c3a42a` | as at 128 rows | 4096 | 54 | 1.282 |

CTA shapes from the launches (S5–S10); VGPRs of the recurrence kernel, none spilling *(field, E2)*; CI: stored speedups at the commits E1 names, 43e112e for 1, 4, 8, 16 and 32 rows; since d908d01 the 1-, 4-, 8- and 16-row files store 2.478, 2.409, 2.405 and 2.360 *(field, E1)*.

- On a stream probe, not a GDN kernel, the rate follows bytes in flight from 256 CTAs up (256 CTAs × 4 waves × 8
  loads/lane read 1.07× of 2048 × 4 × 1), but at 3 MB in flight 96 × 4 × 8 ran at 0.80× and 192 one-wave CTAs × 16 loads
  at half of 384 × 4 × 2, 3 runs *(field, E5)*; at 1–4 rows state bytes grow 4× while time stays 8.62–9.14 µs (t4/t1
  1.001–1.022, 4 runs) *(field, E5)*; latency-bound is inferred, launch latency not isolated (§5).
- From 64 rows the recurrence is a 16-wave CTA with 16 state values per thread under `waves_per_eu` 6 (64 rows) or 8,
  which Triton passes as `amdgpu-waves-per-eu` min = max = N, a hard budget a probe spills to meet (S11). At 8 it
  compiles to 55 (128 rows) and 54 (256 rows) VGPRs, no spill *(field, E2)*, "a fixed, spill-free 16-wave layout" (S10);
  the 64-VGPR cap (512 / 8) and two CTAs per CU are derived. At 4–32 rows it is unset. Forcing it can cost:
  `waves_per_eu=4` at 16 and 64 rows (the Qwen3-Next template's rejected setting, "about 19%" in its comment
  *(field, E3)*) pinned both kernels at 128 VGPRs, spilling 9 and 20, and made the call 1.60× (16 rows) and 1.60–1.63× (64 rows) as long, 2 runs *(field, E18)*.
- Qwen3-Next's per-size files use the same two regimes (S12): "split" at 1–8 rows (one- or two-wave producers own 4–8
  value rows: 256 programs at 1 and 2 rows, the template 512 at 2; 512 and 1024 at 4 and 8), "fused" from 16 rows (one
  4- or 8-wave workgroup per (row, key head) owns both value heads' states and the group's 512 history channels: 64
  workgroups at 16 rows, 1024 at 256). The CI run recorded in d908d01, best per-size file and the template
  *(field, E4)*; a later validate read them within 4.7% (§1) *(field, E2)*:

| rows | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| schedule | split | split | split | split | fused | fused | fused | fused | fused |
| per-size file | `b31b` 2.443 | `bba1` 2.538 | `3719` 2.370 | `3719` 2.506 | `35b9` 3.222 | `38a7` 3.350 | `0c0b` 2.462 | `0678` 1.782 | `0678` 1.425 |
| template `f440` | 2.390 | 2.450 | 2.370 | 2.479 | 3.101 | 3.091 | 2.401 | 1.755 | 1.457 |

A third design at 128 and 256 rows (`c784`, state loaded first, barrier-guarded conv) measured 1.466 and 1.151 in the
same run *(field, E4)*. Use the Point files ([kernels/kernel_library.md](kernels/kernel_library.md)) as candidates, not a nearest-file rule: at 24 rows the log-nearest `20f6` took 16.02–16.23 µs against `a4f8`'s 15.48–15.59 µs, and at 48 rows 23.25–23.50 µs against `fc26`'s 21.53–21.60 µs, in two runs of two rounds per arm *(field, E7)*; at 3 and 6 rows the nearest was 0.2–2.3% slower, at 12 and 96 rows the faster *(field, E7)*. The template keeps one exact-match config row per count (S13). Before copying a file to another row count, check what it assumes:

- `fc26` (128 rows), `c2c3` (256) and `cd00` (64) map programs as `token = (pid // (VH * 8)) * 8 + pid % 8` and do not
  check that the row count is a multiple of 8: at 12 rows, 32 of the 192 (row, head) pairs would not be computed and
  32 programs would read rows past the input (derived). `20f6` (32 rows) falls back to groups of 1 (S10, S9).
- `a4f8` (16 rows) passes the pool size as `SLOTS: gl.constexpr` for a `gl.assume` (S9). `n_slots` is an `any_size` axis
  (S2), so each new pool size compiles again, inside a capture too (four pool sizes, four binaries; with a runtime
  `SLOTS`, one *(field, E16)*): keep it runtime ([authoring.md](authoring.md) §5).

## 3. One token per step: put the launch boundary where ownership ends

Two facts place it (S5, S7): the gated RMSNorm and the fp8 group each span one whole value head, so an independent tile CTA cannot finish the epilogue without obtaining whole-head statistics (a same-launch coordination or recomputation variant was not tested); and the conv history may be overwritten only after every
reader of the old history has read it, while with recomputed inputs (§4) many CTAs read the same channels.

| ownership | where the history shift and the epilogue go | files |
|---|---|---|
| a head split over CTAs; its key head's history read by many CTAs | launch 1: conv + gates + recurrence on a tile, BF16 core to scratch (on `2c2a` bitwise equal to an fp32 scratch of the rounded core *(field, E17)*); launch 2: norm + quant per head, and the history shift by programs or waves that each own whole channels (moved into launch 1, it broke `conv_state` on 10–11 of 20 inputs in each of 3 runs, `delta_state` off by up to 0.95–1.46, while the committed file was exact on all 20 *(field, E5)*; what an unforced launch boundary costs was not measured) | Qwen3.5 1–4 rows, Qwen3-Next 1–8 rows |
| one owner per channel in launch 1 | launch 1 shifts each channel's history right after reading it ("All history reads are complete; retire writes before activation arithmetic", S7); launch 2 owns whole heads and fuses norm + quant | Qwen3.5 8–256 rows |
| one workgroup owns a (row, key head) | one launch: the shift happens inside the conv where each channel has one owner (16, 64, 128, 256 rows), after a `gl.barrier()` inside the conv (`c784`), or after the CTA-wide core exchange where a channel has two owners (32 rows, `38a7`) (S14, S15) | Qwen3-Next 16–256 rows |

- In a one-launch design, prove single ownership in the layout, `gl.static_assert(VEC * 64 * NW <= conv_width)` for a
  `BlockedLayout([VEC], [64], [NW], [0])` conv layout, so no lane or wave is replicated (S14 has VEC = 1; at 32 rows the
  template's VEC 2 × 64 × NW 8 = 1024 slots for 512 channels fails it, S13), or put `gl.barrier()` between the history
  loads and stores (S15). On gfx950 `gl.barrier()` lowers to a workgroup release fence, `s_barrier` and an acquire
  fence, so it also orders the CTA's global accesses (S16); the barriers Triton's Membar pass inserts between
  conflicting LDS accesses cover LDS only (S16). Without either, a replicated layout can go wrong: forced inline history
  retirement at 32 rows in `f440` (two owners per channel) left `delta_state` wrong on 10 of 10 inputs (max |d| 3.24)
  and was at most 1.5% faster, 3 runs *(field, E5)*; its author noted 12% faster, max |d| 0.8 *(field, E3)*.
- At AKLR's per-launch floor CI's ratio charges a boundary more than a graph replay pays: a chained trivial kernel added
  4.04–4.06 µs to the bench's reading under Kineto, before #33 (S17), 1.77–1.84 µs to a bare replay, a fifth of two such
  kernels' 8.25–8.31 µs, 3–4 runs *(field, E10)*;
  the 2-row file's two launches take 8.55–8.87 µs, 3 runs *(field, E5)* ([authoring.md](authoring.md) §7).

## 4. One token per step: recompute shared inputs, keep reductions inside a wave

- Recompute instead of staging: split-schedule tile CTAs convolve/normalize their key head's whole Q/K (256 channels); history is read-only, costing reads and saving a launch (S6). In a one-row Triton toy (16 key heads × 4 tile CTAs, 4-tap conv + SiLU + L2 norm, outputs identical) recompute took 0.693–0.775× the staged version's replay time in three runs but 1.014× (9.967 against 9.832 µs) in a fourth, and 0.589–0.610× on AKLR's timer (Kineto, before #33) in all four: the trade depends on the ruler and the run; committed GDN files were not re-run *(field, E24)*.
- Exchange q and k between the CTA's waves through 1–2 KiB of LDS (q and k, 2 × 128 fp32, at 1 and 2 rows; q, k, v and β
  at 4 rows; S5, S6, S8) and read them back in the state layout; the 2-row file's late q read "to shorten its register
  lifetime" compiles as an early one *(field, E17)*. Summing a `[1, BK]` slice over its unit axis yields the vector in
  `SliceLayout(0, state_layout)` with no data movement (compiled with Triton 3.8.0 for gfx950: four `ds_read_b128` per
  vector, no lane exchange; `qk_shared.index(1).load(gl.SliceLayout(0, state_layout))` is rejected by Triton 3.8.0 for gfx950 at the descriptor `index` step: "rank must be equal to or one less than the shape size. Got 2 and 1"):
```python
# 299133886d479017.json#kernel.py:130-134, 144-145 (Qwen3.5, 2 rows; the update and the state store sit between)
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
  [2, 1], [1, 0])` at 2 rows (S6), `([2, 4], [4, 16], [16, 1], [1, 0])` at 128–256 rows (S10).
- Order for latency: issue the state load first ("Expose the large state fetch before preparing scalar gates"), fetch
  epilogue inputs early but defer their math, normalize q late (S9, S7): reversed, `20f6` lost 0.8–1.6% (two changes in one arm, within the 1–2% A-vs-A spread, §13) and `4f25`
  2.1–2.9% (the q load and normalization moved together), 3 runs *(field, E5)*; "fetch epilogue inputs early" was not varied. The Qwen3-Next template calls the position of the state load "the single most important
  knob for register pressure": before the conv at 1 and 32 rows, after the q/k normalization at 128–256 rows (early: +77
  VGPRs at 128 rows, ≤ 7 at 1 and 32 *(field, E17)*). At 16 and 64 rows its `_fused_heads` path ignores the config's
  `state_load_cache`, `state_store_cache` and `gate_position` and hard-codes `.cg`/`.cs` (S13): A/B those in the code.
- Bound what stays live across the reductions, and check it compiled. The template takes the accumulate form
  `gl.fma(residual, k, state * decay)` at 16 and 64 rows for a 64-VGPR tile, "the difference between 4 and 3 waves/SIMD"
  (S13, S25), yet it compiles to 138 VGPRs against 134 for the recompute form at 16 rows and 131 against 132 at 64, 3
  waves/SIMD both *(field, E17)*; it rounds `state * decay`, so it is a numerics choice too. `0c0b` (64 rows) loads a
  64-column state prefix before the conv and the rest after it, so "only a register-owned column prefix stays live
  across convolution", and `35b9` (16 rows) stores three state quadrants before the epilogue's layout conversion and the
  last after it (S49). The template's 4-warp 128-row instance computes decay and β after the conv because before it
  "costs enough registers to halve occupancy" (S13), yet both orders compile to 200 VGPRs *(field, E17)*.
- Take scalar math off the recurrence's path: `4f25` computes decay and β for all heads in one extra program of the conv
  launch, "so their transcendental work can overlap the independent convolution tiles" (S7); per update program instead,
  the 8-row call took 6.4–7.2% longer, 3 runs (overlap not measured); at 1 row `2c2a` evaluates the four exponentials of the
  value SiLU, decay and β "in neighboring lanes" of one `[BV, 4]` tensor, one `gl.exp` and division, picked by
  `gl.gather` (S5), 4 `v_exp_f32` against 7, no faster than scalar (B/A 0.986–1.013), 3 runs *(field, E5)*.
- Program order: five whole-head files (16–256 rows) number programs in token groups, `token = (pid // (VH * G)) * G +
  pid % G`, G a multiple of 8, so all heads of a row share `pid % 8` and an XCD's L2 (1.75–1.80× on a probe; its
  comment: "Interleave heads over token groups for state-pool traffic"); `a4f8` also XOR-permutes each row's heads (S9;
  [tuning.md](tuning.md) §12). In `fc26` it beat `token = pid // VH` by only 0.6–1.1%, 3 runs *(field, E5)*.
- Check each variant's VGPR count ([debugging.md](debugging.md)) and A/B it ([tuning.md](tuning.md) §2, §10): at equal
  VGPRs (160) the template's late state load at 32 rows took 1.23× as long *(field, E17)*.

## 5. One token per step: the state's loads and stores

The Qwen3.5 files use five combinations (S5–S10); state stores, and loads up to 32 rows, are 16 bytes per lane:

| rows | state load | state store |
|---|---|---|
| 1, 32 | `gl.load`, no hint | `gl.store`, no hint |
| 2 | `buffer_load`, `.cs` | `buffer_store`, `.cs` |
| 4, 8 | `buffer_load`, `.cg` | `buffer_store`, `.cs` |
| 16 | `gl.load`, `.cg` | `gl.store`, `.cs` |
| 64–256 | `gl.load`, no hint | `buffer_store`, `.wt` |

Both choices were ablated on the 2-row file `29913388`: B against the committed A, 4 rounds per arm alternated on one
GPU, one run *(field, E9)*. The instructions are from compiling `_conv_recurrence` with Triton 3.8.0 for gfx950:

| change to the state access | state instructions per thread | VGPRs | mean speedup A → B (4 rounds) | kernel time A → B |
|---|---|---|---|---|
| none (committed: `buffer_load`/`buffer_store`, `cache=".cs"`) | 4 `buffer_load_dwordx4 … offen sc0 nt`, 4 `buffer_store_dwordx4 … offen sc0 nt` | 61 | — | — |
| `cache=".cs"` removed | the same 8 instructions without cache bits | 61 | 2.452 → 2.442 | 8.75 → 8.79 µs |
| `gl.load`/`gl.store` with `cache_modifier=".cs"` | 4 `global_load_dwordx4 … nt`, 4 `global_store_dwordx4 … nt`, 64-bit address in a VGPR pair | 62 | 2.429 → 2.432 | 8.83 → 8.81 µs |

- These two single-run ablations do not establish a reproducible speed advantage; the A-vs-A spread at this Point was 0.4% and 1.7% over 3 runs, 1.9% and 2.0% over 8, in two sessions *(field, E9)*. The call moves 4
  MiB of state in 8.65–9.14 µs, 0.459–0.485 TB/s, 6.8–7.2% of a 1 GiB non-temporal read stream's 6.72–6.78 TB/s, and takes
  8.62–9.14 µs from 1 to 4 rows while state bytes double and quadruple (t2/t1 0.951–1.022, t4/t1 1.001–1.022,
  4 runs) *(field, E5)*: latency rather than proportional state streaming is inferred; the latency source was not isolated. At 2 rows pick buffer or pointer access for the addressing, not for speed.
- At 128–256 rows the state stream is most of the work (256 MiB each way at 256 rows) and the store hint moves the time:
  without `.wt`, or as `gl.store` with `.cs`, the state store cost 2.6–3.6% (one run); load hints were not tried *(field, E9)*.
  A/B the hints rather than copy a file's choice ([gluon/memory.md](gluon/memory.md)). On a stream probe `.cg`/`.cs`
  loads ran 13.0–13.7% faster than plain ones, `.cs` stores 5.5–7.2% slower, `.wt` stores 0.4–1.7% slower through
  `gl.store` (below every store without cache bits) and within 0.5% through `buffer_store`, 3 runs *(field, E5)*. `.cs`
  puts `sc0 nt` on a buffer op and only `nt` on `gl.load`/`gl.store` (S18), which then stream at one rate (within 1.2%)
  *(field, E5)* ([../hardware/gfx950/memory_and_sync.md](../hardware/gfx950/memory_and_sync.md) §2).
- Buffer addressing: the descriptor covers 2^31 − 2 bytes and element offsets are scaled to bytes in 32 bits (S19), so
  files that fold the slot into the offset test the pool size in bytes on the host and rebase a 64-bit scalar head
  pointer for larger pools (S8); the contracts cap the pool at 2047 slots (S2), 2047 MiB of Qwen3.5 state, so AKLR
  never takes the wide path.
- Padding rows: put one uniform `if slot >= 0:` around every access to either pool, the history shift included, and
  write a zero core otherwise (S6). The norm and quant still run on that zero core and write the row's outputs, as
  `4f25` does (S7): zero-valued codes, roughly half −0 on the sampled random inputs, and scale float32(1e-10) × float32(1/448) *(field, E17)*.

## 6. One token per step: the fused fp8 group quant and the FP16_OVFL mode bit

One helper ends every head in all eight Qwen3.5 fp8 decode files (the same function text in each): it rounds the head
to BF16, takes the amax of its non-NaN values floored at 1e-10, scales by one reciprocal of `amax · (1/448)`, clamps to
±448 with an inline `v_med3_f32` (`gl.clamp` lowers to the same, S20) and converts (S6). The recipe and its division
forms: [numerics.md](numerics.md) §2.

- Keep the head's 128 values in one wave (`BlockedLayout([4], [64], [2], [0])` gives each of two waves a full copy), so
  amax and the RMS are wave reductions with no LDS exchange (S5).
- Quantize the BF16-rounded output, as the references do (S1); compute the scale from it, not from the fp32 precursor.
- Round before a cross-wave `convert_layout`: "Preserve the BF16 boundary and halve the cross-wave LDS transfer" (S7).

**FP16_OVFL.** In 38 E4M3FN/E5M2 cast variants from fp32, fp16 and bf16 compiled for gfx950, Triton wrote MODE.FP16_OVFL = 1 once, before the first VALU of the function containing the cast (S21): the kernel when inlined, the callee in six noinline variants. The variants cover default and rtne `.to`, E5M2 rtz from fp32 and fp16, an implicit cast in `gl.store`, an explicit cast then `buffer_store` and a runtime branch; a direct wider-value `buffer_store` and E4M3FN rtz did not compile, and seven controls (upcast, bitcast, MFMA, asm, BF16) wrote no MODE *(field, E20)*. Other fp8 casts and kernel entries were not compiled. The 2-row file's `_normalize_update_history` sets it before its first `v_cvt_pk_bf16_f32`; BF16 finite overflows then saturate to ±0x7F7F *(field, E17)*. The Qwen3.5 files leave it set and pass the contracts, whose inputs are drawn finite. `b31b`
and the Qwen3-Next template at 1 row map |x| ≥ 3.396e38 to ±Inf before the BF16 convert and keep Triton's fp8 store
(S23); the template at 2, 4, 8 and 32 rows clamps and converts with an inline `v_cvt_scalef32_pk_fp8_f32`, so no Triton
cast sets the bit, and switches it in value-producing asm, clear at entry, set after the last BF16 rounding; `38a7` sets
it in the convert's own asm (S24); the template's BF16 results saturate at 16, 64, 128 and 256 rows *(field, E17)*.
Rule: [numerics.md](numerics.md) §2; asm and ordering risk: [gluon/wave_ops.md](gluon/wave_ops.md) §3.2, §7. Keep the
two choices separate knobs: "collapsing the two into one flag silently changes results" (S25).

## 7. Chunks of many tokens: the pipeline and its launch boundaries

Every prefill file runs, in launch order (`010e`, `70f4`, `11f6` add a conv-history launch, `11f6` a chunk-offset one):

1. `_initialize_fresh` zeroes the state slot of each sequence that has no initial state, from the device flag (S26): all
   26 files capture globally and replay with new inputs matching the reference; this does not independently test flag zeroing, ragged handling or stream-only synchronization *(field, E16)*. Nine Qwen3.5 files then run `pack`, a copy of the
   contiguous projection into Qwen3-Next's grouped order (§1), 10 KiB per token each way (estimated; S26). The six MAP
   and RSUM files read the contiguous layout directly (S32): the pass is avoidable; what it costs is not measured.
2. Prepare, parallel over token tiles × channel groups: conv + SiLU + q/k L2 norm to BF16, gates.
3. Chunk factors, parallel over chunk × head: Gram products, decay products, the unit-triangular solve, WY factors.
4. Long sequences only (summary designs, §8): each run of chunks summarized as an affine map `S ↦ S·A + B`, in parallel.
5. Serial pass over chunks or summaries, parallel over sequence × head × value strip; per-sequence designs add outputs.
6. Parallel epilogue: outputs rebuilt from incoming states where step 5 made none, then gated RMSNorm and fp8 quant.

- Every boundary changes the parallel decomposition; with no atomics (§1), stream order is the only synchronization.
- The files write the conv history after its readers: 19 in their last launch; SEQ16 (§8) in the one-wave program that
  convolves a sequence's first tokens, its only reader, after a `gl.barrier()` (S27) whose removal left `conv_state` exact
  in all 20 runs (2 runs), as it must in one wave *(field, E19)*; `98af`/`ed2a` behind `if NW > 1`, false at their Point
  (S47); three Qwen3-Next files in a launch of their own. Write the history only after every reader of the old one has read it: in a toy kernel a history entry stored before the reads was read in place of the old one in every repeat, and at four waves readers that no barrier ordered before the writer read 192 wrong values, at one wave none *(field, E8)*. Put a barrier where readers and the writer sit in different waves; SEQ16's one-wave program stayed correct without one.
- A fresh sequence starts from a zero history as well as a zero state, but `_initialize_fresh` zeroes only
  `delta_state`: mask every read of the old history with the flag. A sequence of one or two tokens keeps old entries:
  its new history is the last three of [old history, or zeros when fresh; its tokens] (S27, S1). The 2049/2 Point
  always holds a one-token sequence: AKLR draws one of exactly `max_seq_len` tokens and splits the rest (S50).
- Ragged batches without host reads: grids come from shapes only *(field, E17)*; a chunk program finds its sequence by
  binary search over `cu_seqlens`, the slot formula `begin // BT + sequence` "reserves enough space for every ragged
  tail", and programs past the device-written chunk count do nothing (S28).
- Scratch is `torch.empty` per call (`aklr/ops/README.md` § The kernel section and ABI) and large: 173–294 MB of fp32 at
  8–16K tokens in the summary designs (MAP at 10000 tokens: 246 MB of factors, 36 MB of maps) *(field, E17)*; buffers
  with disjoint live ranges are aliased. Write every element a later launch reads in the same call, padding rows and
  ragged tails included: a NaN left in old memory survives a zero β or a decay of 1 (0 × NaN is NaN). `98af` zeroes each
  chunk record's padding tokens; `010e` states the rule for its whole pipeline (S51; [authoring.md](authoring.md) §6).
- Under a stream capture the Qwen3.5 and Qwen3-Next prefill routes return `out_of_contract` (S29; called with mocked
  sglang layers *(field, E16)*); a capture in a running server was not tested. Eager launches pay about
  1.0 µs of host time per tensor argument, which the graph bench omits *(field, E10)* ([authoring.md](authoring.md) §7).

## 8. Chunks of many tokens: chunk size and schedule per Point

Measure candidate designs at the specific (tokens, sequences, `max_seq_len`) Point; the committed boundaries below are not selection rules confirmed by measurement. Labels: D, the ≤ 1024-token source
(`1fd6`, `61ed`, `1d55`); CKPT, the checkpoint scan (`19b2`, `3828`); SEQ32L, SEQ32G and SEQ16, one program per
(sequence, head, value strip) over 32-token sequence-local, 32-token global (plus a scalar per-token recurrence at
ragged edges) and 16-token global chunks (masked so each segment's coefficients stay in its sequence); MAP and RSUM,
summaries of runs of chunks. The last column times another design's kernel, or the other strip, in the file's scenario,
2 runs, accuracy unchecked *(field, E19)*: the committed file wins or ties except at 8192/1 (RSUM 0.73–0.74× MAP's time).

- **Short-scan candidates, not a universal ≤2048-token rule**: D and CKPT store chunk incoming states and reconstruct outputs in parallel (S33, S34). At (512,1,512), (1024,2,512), (512,2,256), (1024,1,1024), (1025,1,1025), both were faster than the tested RSUM and MAP under eager single-call device-event timing *(field, E21)*. CKPT beat D at all five; CKPT against SEQ32L was unresolved at the second and third Points, not shown equal.
  One study of the five Points, 16 balanced adjacent pairs per comparison, each source file on its own host branch with no config search; all 25 Point-arms passed the contract's accuracy bounds against an fp64 sequential reference on every full tensor, and the two-sequence inputs had equal lengths. 2048/1, other ragged inputs and graph or serving timing were not covered. The table's 2048/1 RSUM/CKPT 1.99× comes from a separate measurement that did not check the placed kernel's accuracy *(field, E19)*.
- **More tokens with `max_seq_len` 2048 (2–8 sequences)**: one program per (sequence, head, value strip) walks all
  chunks and produces the outputs; BV makes sequences × 16 heads × 128/BV = 256, one program per CU: SEQ32L at 2
  sequences (BV 16), SEQ32G at 3–4 (BV 32), SEQ16 at 5–8 (BV 64) (S27, S30, S31). The Points that fill 256 programs
  measure 1.374–1.475; the two with fewer sequences than their file assumes (160 and 192 programs) 0.997 and 1.029,
  below the default `AKLR_MIN_SPEEDUP` (`aklr/ops/README.md` § Environment variables), so sglang serves them. Their
  longest sequence is 2048 tokens, as at the full Points, and the time does not follow the work: about half the work
  took 0.97× (`2ece`) and 0.85–0.86× (`c53b`) the full Point's time, 2 runs *(field, E19)*, which does not separate the
  serial chain from the program count; the other strip was slower (table).
- **Long sequences, `max_seq_len` 4096 (Qwen3-Next) or 5000 (Qwen3.5) and above, 1–4 sequences**: a per-sequence walk
  is a long serial chain, and at one sequence 16 × 128/BV programs are too few. Build each run of chunks' affine map in
  parallel, from an identity seed (MAP: the rows `[I; 0]` run through the recurrence) or composed right to left from the
  run's last chunk (RSUM); chain the maps in a short serial pass (4–15 dependent 128 × 128 products per row strip,
  derived from the chunk counts); then replay all chunks in parallel from their incoming states (S32). At one sequence
  this beats a per-sequence walk, SEQ16 (table); 2–4 sequences and Qwen3-Next were not timed.
- **Keep the serial chain overlapped** (S33, S27, S30); undoing each knob cost, at one Point, 2 runs: in CKPT (`3828`)
  at 2048/1, unroll 1 for `loop_unroll_factor=8` 6.3–7.0%, no `amdgpu-sched-strategy` `iterative-ilp` 11.8–13.3%
  ([tuning.md](tuning.md) §10), one 128-deep MFMA chain for two independent 64-deep ones 3.1–4.6% *(field, E19)*; in
  `24ed` at 16384/8, the next chunk's operands loaded after the current chunk's math 2.0–2.4%, `.cg` on the value tile
  dropped −0.2 to +0.3% *(field, E19)* ([gluon/memory.md](gluon/memory.md) §12). `2ece` stores each half table once
  final, "to shorten register lifetimes": 190 VGPRs at M = 8192, 194 with all stores at the end, untimed *(field, E19)*.

Qwen3.5, CI at the commits E11 names *(field, E11)*; since d908d01 `1fd6` stores 2.452 and `1d55` 2.681 and 1.927 *(field, E11)*; serial-pass programs derived from the grids:

| tokens / sequences | max_seq_len | file | design | chunk (tokens) | serial programs | CI | other kernel here, its time ÷ the file's |
|---|---|---|---|---|---|---|---|
| 512 / 1 | 512 | `1d5568a7` | D | 64 | 128 | 2.638 | — |
| 512 / 2 | 256 | `61ed32f6` | D | 64 | 256 | 2.629 | — |
| 1024 / 1 | 1024 | `1fd6adb2` | D | 64 | 256 | 2.423 | CKPT 1.01× |
| 1024 / 2 | 512 | `1d5568a7` | D | 64 | 512 | 1.889 | — |
| 1025 / 1 | 1025 | `19b20285` | CKPT | 64 | 128 | 2.256 | — |
| 2048 / 1 | 2048 | `382853eb` | CKPT | 64 | 128 | 2.092 | RSUM 1.99× |
| 2049 / 2 | 2048 | `0a07f9f9` | SEQ32L | 32 | 256 | 1.374 | — |
| 4096 / 2 | 2048 | `0a07f9f9` | SEQ32L | 32 | 256 | 1.452 | SEQ16 1.93–1.95× |
| 4097 / 3 | 2048 | `2ecece96` | SEQ32G | 32 | 192 | 1.029 | BV 16 1.32× |
| 8192 / 4 | 2048 | `2ecece96` | SEQ32G | 32 | 256 | 1.424 | — |
| 8193 / 5 | 2048 | `c53b6d87` | SEQ16 | 16 | 160 | 0.997 | BV 32 1.15× |
| 15360 / 8 | 2048 | `c53b6d87` | SEQ16 | 16 | 256 | 1.473 | — |
| 15361 / 8 | 2048 | `24ede969` | SEQ16 | 16 | 256 | 1.454 | — |
| 16384 / 8 | 2048 | `24ede969` | SEQ16 | 16 | 256 | 1.475 | RSUM 1.28–1.29× |
| 5000 / 1 | 5000 | `12b560c8` | RSUM | 32-token tiles, 512-token chunks | 128 | 1.199 | — |
| 15000 / 3 | 5000 | `12b560c8` | RSUM | 32 / 1024 | 384 | 1.198 | — |
| 8192 / 1 | 8192 | `5ead0483` | MAP | 16-token blocks, 512-token chunks | 128 | 1.193 | RSUM 0.73–0.74×, SEQ16 2.71× |
| 10000 / 1 | 10000 | `7a69337f` | MAP | 16 / 640 | 128 | 1.188 | — |
| 16384 / 1 | 16384 | `47092631` | RSUM | 32 / 1024 | 128 | 1.741 | SEQ16 3.97× |
| 16384 / 2 | 8192 | `b8c54bde` | RSUM | 32 / 1024 | 256 | 1.339 | — |
| 16384 / 2 | 10000 | `3b99d196` | RSUM | 32 / 1024 | 256 | 1.444 | — |

`5ead`/`7a69`, `3b99`/`4709`/`b8c5` and `19b2`/`3828` are byte-identical sources and D is one source in three scenarios,
yet one source's ratio moves across Points: D (`1d55`) 2.65 at 512 × 1 against 1.91 at 1024 × 2 (1.38–1.39×), RSUM
1.73–1.74 at 16384 × 1 (`4709`) against 1.36 at 16384 × 2 (`b8c5`, 1.27–1.28×), 2 runs *(field, E19)*: re-measure at every
Point. No prefill `run` reads `max_seq_len`, only takes it (S32): the attr only picks the scenario.

Qwen3-Next: one CI run, recorded in d908d01, constexpr file / runtime-count twin *(field, E11)*; a later validate read
them within 4.7% (§1) *(field, E2)*. Every sequence of the 1024 scenario is 1024 tokens long. `162f`, `c3a7` and
`4489` are one source, and `a99f`, `f137` and `d8f9` its runtime-count twin.

| tokens / sequences | max_seq_len | files | schedule | chunk | CI, per Point |
|---|---|---|---|---|---|
| 1024 / 1; 2048 / 2 | 1024 | `11f62a16` | serial scan with per-chunk state snapshots, parallel output | 64 | 1.573; 1.434 |
| 3072 / 3; 4096 / 4 | 1024 | `448925c3` / `d8f9e1b5` | snapshot scan, parallel output | 32 | 1.592 / 1.554; 1.504 / 1.501 |
| 8192 / 8; 12288 / 12; 16384 / 16 | 1024 | as above | direct recurrence | 32 | 1.490 / 1.484; 1.288 / 1.282; 1.463 / 1.460 |
| 13312 / 13; 15360 / 15 | 1024 | `70f4be35` / `010eab66` | compact factors, operands rebuilt in the serial loop | 32 | 1.232 / 1.233; 1.328 / 1.329 |
| 4096 / 1 | 4096 | `162f43f1` / `a99f22dc` | block summaries (identity and zero seeds) | 32 | 1.718 / 1.719 |
| 5120 / 1; 10240 / 2 | 5120 | `c3a70b84` / `f1374046` | block summaries | 32 | 1.727 / 1.741; 1.308 / 1.311 |
| 16384 / 4 | 5120 | `ed2a8611` / `98af1fd2` | segment summaries composed in reverse | 32 | 1.294 / 1.283 |

The `11f6` file takes 64-token chunks at 1–2 sequences (S35); the `4489` source picks its schedule from the Point: the
snapshot scan at 3–4 sequences up to 4096 tokens, block summaries at 1–2 sequences here, the direct recurrence at 8 and
more (S36). The twins, `M` and `BATCH` as `gl.constexpr` against runtime values, differ by 0.1–2.4% at a Point in one
CI run *(field, E11)*, about the run-to-run spread of the 2-row decode file, a different kernel (0.4–1.7% over 3 runs, 1.9–2.0%
over 8, two sessions) *(field, E9)*; no repeated run measures it at these Points, so A/B the two before choosing.

Before you reuse a prefill file at another Point, read which branch its Points take. Above 4096 tokens `0a07`
static-asserts that `normalized` has the conv buffer's 3072-element row stride, which AKLR's `(n_tokens, 16, 128)`
output lacks: compiled with Triton 3.8.0 for gfx950 at AKLR's stride, that path fails its `static_assert` (S31). `11f6`
finds a token's sequence in seven unrolled halvings, right for at most 128 sequences where the contract allows 2047
(S35): it fails at 129 sequences, where eight halvings pass *(field, E17)*. `2ece`'s path up to 4096 tokens and CKPT's
up to 1024 do not run at their Points: no CI run checked or timed them (S30, S33).

## 9. Chunks of many tokens: the intra-chunk math on the matrix cores

- Put the Gram products K Kᵀ and Q Kᵀ on the BF16 MFMA (`instr_shape=[16, 16, 32]`, `k_width` 8: eight consecutive K per
  lane, [gluon/mfma.md](gluon/mfma.md) §5): the prefill references round q and k to BF16 (S1), and the product of two finite BF16 values is exact in fp32 when it lies in fp32's normal range: 2^20 random BF16 pairs with exponents within ±63 all matched the fp64 product *(field, E17)*. The fp32 accumulation still rounds.
- Share the Gram products among the value heads of a key head (four in Qwen3.5, two in Qwen3-Next): K Kᵀ and Q Kᵀ depend
  only on the key head, so `0a07` computes them once per (chunk, key head) ("share four-head Gram matrices"), `24ed` and
  `2ece` once per head pair; only β and the decays differ per value head (S31, S27, S30). In `0a07` at 4096/2 sharing
  saved 1–2% of the call (means 1.011 and 1.021 in 2 runs) and 84 VGPRs *(field, E19)*.
- State algebra, solves and tables run on the fp32 MFMA (`v_mfma_f32_16x16x4_f32`, some `32x32x2`) *(field, E2)*: 32
  cycles for 1024 MACs, against 16 cycles for 8192 MACs on the BF16 16x16x32 (S37), 16× fewer MACs per cycle (15.7–15.8×
  measured, 4 runs *(field, E13)*). Keep the serial pass to few fp32 products per chunk (CKPT: two per 64 tokens);
  moving output products to BF16 (§10) is untimed; [gluon/mfma.md](gluon/mfma.md) covers the fp32 input precision.
- The unit-lower-triangular inverse `(I + A)⁻¹` of a 16–64-token chunk: solve small diagonal blocks (4 × 4, 8 × 8 or
  16 × 16) by forward substitution with `gl.gather` pivots and join them with fp32 MFMAs; or, on a 16 × 16 block,
  multiply out `(I+N)(I+N²)(I+N⁴)(I+N⁸)`, which is `(I − N)⁻¹` for a strictly lower N (N¹⁶ = 0): pass N = −A, as `b8c5`
  negates the Gram term before `_inverse16` and `11f6` documents ("Inverse of I-P for a strictly lower triangular
  16-by-16 P"), and join 16 × 16 inverses with fp32 MFMAs for longer chunks (S46, S35). Keep each small solve in one
  wave: `gl.gather` lowers to lane shuffles only when the gathered axis stays in a warp, else through LDS (S38).
- The files skip the zero halves of triangular operands: K blocks taken with `gl.amd.slice`, a register-only view (S39),
  only nonzero blocks multiplied, only tiles on or below the diagonal stored and loaded (S31, S33). In `0a07` at 4096/2
  that cut 184 to 136 MFMAs, not the call time (B/A 0.966–1.008, 2 runs) *(field, E19)*; the stores were not varied.
- Store each factor in the order its consumer's MFMA operand holds it, so the consumer loads 16-byte vectors and
  converts with `assert_trivial=True` (S30), which fails the compile ("layout conversion from … is not trivial",
  printing both linear layouts) when the conversion moves data across lanes or waves (S42).

## 10. Chunks of many tokens: compensated products, and decay products over intervals

**Compensated products.** Where the product of an fp32 operand and a BF16 operand is rounded to BF16 anyway, split the
fp32 operand into BF16 terms, run the BF16 MFMA once per term, smallest first, and accumulate in fp32. On random finite 16×32 @ 32×16 tiles (4,194,304 outputs), three terms (RTZ, RTZ, RNE) matched the fp64 result rounded to BF16 in 100% of the outputs, two RNE terms in 99.76%, one term in 58.2% *(field, E17)*: sampled agreement, not a proof of correct rounding for every product. Five Qwen3.5
files do, three saying "Only output reconstruction uses compensated BF16 products: chunk solves, the persistent
recurrence, and mutable state remain FP32" (S34). CKPT's `_mm_query` takes two RTNE terms at its Points, three RTZ terms
in its other branch (S34): on 300000 sampled fp32 values with exponents within ±40, two RTNE terms had at most 7.605e-6 relative split error, below the estimated 2^-16 bound and the BF16 output step; three RTZ terms reconstructed every sampled value exactly *(field, E17)*. Values outside that exponent range were not sampled. Where both operands are fp32-derived,
the "D" files store both as BF16 high/low pairs and keep all four cross products (S34). No Qwen3-Next prefill file uses
them ([numerics.md](numerics.md) §5). (`gl.amd.cdna4` re-exports `mfma`, `buffer_load` and `buffer_store`, S40.)

**Scan each interval, not two prefixes.** Build each relative decay Γ(i, j) over (j, i] directly, not as `prefix_i /
prefix_j`: on log-decays uniform in [−6, 0] the ratio underflows to NaN or 0 at 60% of normal-valued entries, the masked
add-scan stays within 3.8e-6 *(field, E17)*. MAP runs `gl.associative_scan` with a multiply combiner down the columns of
`gl.where(i[:, None] > j[None, :], sg[:, None], 1.0)` (S32); `11f6` scans the log-decays of each interval with an add
combiner and exponentiates once: "Direct interval scans also handle zero decay without inf-inf or 0/0" (S35). Other
files split the chunk into two 16-token halves with a separable cross block (S30), or keep dyadic levels of prefix and
suffix products and read Γ(i, j) = suffix_L[j] · prefix_L[i] at L = `31 − clz(i ^ j)`, the highest bit in which i and j
differ (S33). `gl.associative_scan` lowers only a `BlockedLayout` operand (S38): on a `SliceLayout` vector, such as the
per-token gates of a 2-D tile, the compile fails with "TODO: unsupported scan layout" (compiled with Triton 3.8.0 for
gfx950; [debugging.md](debugging.md) §4): convert to a blocked layout, scan, convert back, as MAP does (S32). Padding
tokens get decay 1 and β 0 (S32). One recurrence step with decay 1 and β 0 on a random finite 128 × 128 state changed no bit, but a NaN in v made its rows NaN *(field, E17)*: the update is neutral only with finite intermediates, since β = 0 times a non-finite prediction or residual is NaN. Which
exponential keeps subnormal decays (`gl.exp`) and which flushes them (`gl.exp2`): [numerics.md](numerics.md).

## 11. Chunks of many tokens: relabel layouts instead of converting them

Each chunk step feeds an fp32 MFMA accumulator (the state, or an inverse block) into the next MFMA as the A operand.
For its `[16, 16, 4]` accumulator as A, `k_width = 1` takes two register↔lane transpositions through LDS (one wave: 4 `ds_write_b32` + 4 `ds_read_b32`, 1 KiB *(field, E17)*); a transposed `[32, 32, 2]` accumulator needs only one and stays in-wave with warps on M (S38). With 4 waves, the `[16, 16, 4]` case took two transpositions and 16 KiB of LDS at 128 × 32, transposed or not; the transposed `[32, 32, 2]` one at 256 × 64 one transposition and no LDS, the untransposed one four and 64 KiB; a sweep of 432 builds stayed in-wave whenever fewer than two transpositions were needed and the warps did not change *(field, E22)*. Ways around it (Triton 3.8.0, gfx950):
- **Where the warps go decides it.** On 64 × 64 fp32 `[16, 16, 4]` tiles with 4 waves, only a `transposed=True` `[4, 1]`
  accumulator is a free `k_width = 4` A operand and a `transposed=False` `[1, 4]` one a free B operand; every other tested mix
  of transposition, warps (`[4, 1]`, `[1, 4]`, `[2, 2]`), operand and `k_width` in `{1, 4}` goes through LDS (§12).
- **`k_width = 4` operands (conversion candidate, not a speed preference).** With warps only along M, the
  transposed fp32 accumulator already is the `k_width = 4` A operand: the conversion to `DotOperandLayout(0, mma, 4)` passes `assert_trivial=True`, while `k_width = 1` fails it, and so does `k_width = 4` with `[2, 2]` or `[1, 4]` warps; it needs no permutation code. CKPT's scan (`[1, 4]` warps, `transposed=False`) stages the state through LDS and reads it back as a `k_width = 4` operand (S33); RSUM's summaries use `k_width = 4` with `[2, 2]` warps (S32).
- **Relabel K (SEQ16).** Permute K inside every 16-block (k = 4y + z → 4z + y) on both operands,
  `_permute_columns_for_a` on A, `_permute_rows_for_b` on B: the product is unchanged, the conversion passes
  `assert_trivial=True` (S27, S42), and the one-wave code is that of `k_width = 4`. The file holds the state K-permuted
  for the whole sequence (permuting it again lands it in the A layout) and un-permutes it for the final store. In a
  probe of its 64 × 128 `[4, 1]` state, the direct `k_width = 1` conversion costs 8 `ds_write_b128` + 8 `ds_read_b128`
  and 16 KB of LDS; relabeled, it has no LDS access or lane shuffle, only 32 `v_mfma_f32_16x16x4_f32`.
- **Two permlane-sized steps (SEQ32G, on its 16 × 16 inverse blocks, not the state).** Go through a middle
  `DistributedLinearLayout` one transposition from each end, 2ece's `reg_bases=[[0, 1], [0, 8]]` (S30): on a one-wave 16
  × 16 tile the steps lower to 2 `v_permlane32_swap` and 2 `v_permlane16_swap`, no LDS. The gfx950 backend uses these
  swaps for one (register, lane 4 or 5) transposition or a 3-cycle through lanes 4 and 5 (S43); helper:
  [gluon/layouts.md](gluon/layouts.md) §7.

In one wave, a product of two 16 × 16 fp32 accumulators held in the same transposed layout takes 69 instructions with
`k_width = 4` or the relabel, 79 with the permlane route (four `v_permlane*_swap`) and 92 with a direct `k_width = 1`
conversion *(field, E17)*; all four use 1 KB of LDS, since B is not free there (the direct conversion moves both
operands through it, the others only B). In a layout-conversion probe, not a GDN kernel, a lane-moving conversion cost
0.6–1.5 µs per 16 KB tile with all programs running, 3 runs *(field, E5)* ([gluon/layouts.md](gluon/layouts.md) §8; the "D"
files' hand-written `v_permlane16_swap`: [gluon/wave_ops.md](gluon/wave_ops.md)). Fewer instructions did not mean less time: a 256-step, 4-wave chain on 64 × 16 tiles at 16 programs per CU took 337.2 µs with `k_width = 4` against 256.8 µs with the direct `k_width = 1` conversion, 1.313× (8 balanced repeat clusters, one run) *(field, E23)*; no GDN kernel was timed either way.

## 12. Chunks of many tokens: LDS staging, conv windows and masks

- What the files stage through LDS, in shared layouts they choose (alternatives not timed):
  - V: value-major loads widened to fp32 into `SwizzledSharedLayout(4, 1, 8, [0, 1])`, read in the MFMA layout (S27);
  - the state, when the warps split the accumulator's columns: every wave then needs whole state rows as its A operand
    (compiled with Triton 3.8.0 for gfx950, a 16 × 128 state strip in a transposed `[1, 4]` layout, as `0a07` holds it,
    sends even the `k_width = 4` conversion through LDS: two `ds_write_b128`, eight `ds_read_b128`, three
    `s_barrier`). `0a07` writes a `SharedLinearLayout` that keeps 4-float vectors and XOR-swizzles banks (S31). Liveness
    is use-based ([gluon/memory.md](gluon/memory.md) §10): to keep a buffer out of later conversion scratch, call
    `_keep_alive()` after those conversions, as Qwen3-Next `35b9` does after its epilogue ("avoid a shared-memory reuse
    barrier", S49); `0a07` calls it right after its load (S31). Compiled with Triton 3.8.0 for gfx950 at `HEADS=16`, `BATCH=2`, `BT=32`, `BV=16`, `CORE_STRIDE=2048`, `SequenceChunks=None`, 4 warps, `waves_per_eu=0` and FP fusion off, `0a07`'s `_chunk_recurrence` without that call gives the same 1108 instructions in the same order and the same resources (8192 B of LDS, 152 VGPRs, no spills): at this specialization the call changes nothing.
- Conv windows: keep a tile's token rows in each lane's registers (`BlockedLayout([1, 2], [1, 64], [1, 1], [1, 0])`) and
  take the four taps as register slices with `gl.amd.slice`, "Four overlapping convolutions share their seven input
  rows" (S32). The slice keeps the source layout, so its offsets must fall on the layout's CTA tile (S39): with rows on
  lanes, e.g. `BlockedLayout([1, 2], [2, 32], [1, 1], [1, 0])`, the compile fails with "No source register holds the
  element for destination index [1, 0]" (compiled with Triton 3.8.0 for gfx950). Or stage the window in LDS and
  `.gather` the shifted rows (S33). Prove a tile interior (all taps in one sequence) at run time and take a mask-free
  branch for it (S33): in CKPT (`3828`) at 2048/1, every tile on the masked branch made the call 0.9–2.4% slower
  (2 runs) *(field, E19)*. [gluon/memory.md](gluon/memory.md) covers both.
- Make masks fold at compile time. A runtime mask on a pointer `gl.load` puts a branch around every access (S45).
  SEQ32G's table builder masks rows with `(chunk * BT + ri < M) | complete_chunks`, where `complete_chunks: gl.constexpr
  = M % BT == 0` (S30). Compiled with Triton 3.8.0 for gfx950, at M = 8192 the builder issues 24 `global_load_dwordx4`
  and uses 190 VGPRs; at the ragged M = 4097 it issues 195 `global_load_ushort` behind exec-mask branches, and under the
  launch's `waves_per_eu=2`, a 256-VGPR cap (S30), it uses 256 and spills 64 *(field, E2)*; without the cap it takes 322
  (66 of them AGPRs) and spills nothing ([tuning.md](tuning.md) §8). Its three masked operand loads as `buffer_load`
  drop their branches and, under the cap, the spill (243 VGPRs), but stay 2-byte loads. Specializing folds the mask only
  when M % BT == 0, and M already is a constexpr: at a ragged M, have the producer write zeros into the tail rows of the
  chunk-padded scratch (`packed_qk` holds whole 32-token chunks; its producer's store is masked at ragged M, S30) and
  load without a mask. In `2ece` at 4097/3 the builder then issues 24 `global_load_dwordx4` and fits 190 VGPRs at
  `waves_per_eu=2` without a spill; the call took 10% less (B/A 0.895–0.902, 2 runs), passing accuracy *(field, E19)*.

## 13. What the measurements say

- Each stored ratio, one per file and Point, describes a whole design, not one knob. Decode ratios are highest up to 16
  rows (Qwen3.5 2.336–2.507 at 43e112e, 2.360–2.507 since d908d01 *(field, E1)*) and 64 rows (Qwen3-Next, 2.370–3.350 in one run *(field, E4)*), where one or
  two launches replace a five-call baseline and the state stream is small; at 128–256 rows they fall to 1.282–1.426 and
  1.457–1.782; at 256 rows (`c2c3`) both sides stream the state near a read stream's rate (1.30–1.31 there, 2.47–2.48 at
  2 rows, where AKLR's two kernels take 8.55–8.87 µs and the baseline 21.20–21.78 µs; 3 runs *(field, E5)*).
- At the 2-row decode Point an identical copy's 4-run mean ratio read 0.25–0.27% off the original's (single runs spread
  1.9–2.0%; two sessions); over 7 Points the largest such offset was 0.4% and 1.3% (fp8 GEMM m = 16) *(field, E9)*: these A/A offsets call for matched controls and uncertainty estimates, not a universal 1–2% detection floor. The same protocol resolved a roughly 1% real change with four alternated runs at a different Point ([tuning.md](tuning.md) §2). Ablations: one token per step §2–§5, chunks §8, §9, §12. In
  one run the hand-written Qwen3-Next decode files led the template by 0–8% up to 64 rows *(field, E4)*.

## Evidence

AKLR paths are on `AMD-AGI/AKLR@0e02a81` unless a line names another commit; a committed kernel is cited as `<schema path>#<source file>:<line>`, and every schema cited here holds the same sources at 43e112e. Triton paths are `triton-lang/triton@c01b677` (tag v3.8.0). Each E line names the commit or CI run its numbers come from; those marked only #89 were taken for #89 with AKLR's code as of 43e112e, whose bench timed calls with Kineto (AKLR's rocprofiler-sdk collector since #33).

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] Qwen3.5 GDN decode, stored speedups: FP8 op 1 row 2.464, 4 rows 2.422, 8 rows 2.351, 16 rows 2.336, 32 rows 1.651 at 43e112e (since d908d01 2.478, 2.409, 2.405 and 2.360 at 1–16 rows, 32 rows unchanged); 2 rows 2.507 at 90b9d50; 128 rows 1.426 and 256 rows 1.282 at a29581b; BF16 op 64 rows 1.461 at 90b9d50; the values of d908d01 unchanged at 0e02a81 (`2c2a09e220f2706c`, `409cb22b39b3fa00`, `4f257a3e566b20d1`, `a4f80b6a3877b48e`, `20f63094b85c00a2`, `299133886d479017`, `fc261961f0ad572d`, `c2c3a42a3909684a` in gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072, `cd00c50e595b05bc` in gdn_decode_n_v_heads16_key_dim128_value_dim128_conv_dim3072; AKLR@43e112e CI run 41991630592, AKLR@d908d01 CI run 42000899019, AKLR@90b9d50 CI run 41976547645, AKLR@a29581b CI run 41197527705)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every Triton schema validated once with `python -m aklr validate` on its own Triton cache, resources and opcodes read from each cached `.amdgcn`; for the Qwen3-Next schemas also the measured speedup at each schema's lowest Point against the stored one] the 18 GDN decode schemas: no `v_mfma` and no `v_dot2` in AKLR's kernels; Qwen3.5 recurrence VGPRs 35, 61, 61, 142, 146, 126, 47, 55, 54 at 1, 2, 4, 8, 16, 32, 64, 128, 256 rows, no spills, `fc26` (128 rows) 16 warps at `waves_per_eu` 8; all 26 prefill files contain an fp32 MFMA (`16x16x4`, or `32x32x2` in 4 Qwen3.5 and 2 Qwen3-Next kernels) and a BF16 one; `2ecece96` `_build_chunk_tables_paired` 190 VGPRs at M = 8192, 256 VGPRs with 64 spilled at M = 4097 under `waves_per_eu=2`, 322 and no spill without it; the 20 Qwen3-Next schemas: every measured speedup within 4.7% of the stored one (largest deviation of the ratio from 1, 0.04654), and every one of them failed validate, by an accuracy failure or a nonzero exit: 0678, 0c0b, 35b9, 3719, 38a7, b31b, bba1, c784, f440, 010e, 11f6, 162f, 4489, 70f4, 98af, a99f, c3a7, d8f9, ed2a, f137 (the 44 GDN schemas; stored values of AKLR@d908d01, CI run 42000899019; #89)
- E3 [2026-11, gfx950 (MI355X), Triton ?, N=?, not stated: the kernel author's note in the source] `waves_per_eu=4` at 16 and 64 rows about 19% slower than unset, VGPRs pinned at 128 by spilling; inline history retirement forced at 32 rows 12% faster and wrong, max |d| 8e-1 on the state (Qwen3-Next template `f44007374038041f`; AMD-AGI/AKLR@0f2036b:aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_decode_config.py:91-96, :98-117)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] Qwen3-Next GDN decode, per-size files: 1 row `b31b` 2.443, 2 `bba1` 2.538, 4 and 8 `3719` 2.370 and 2.506, 16 `35b9` 3.222, 32 `38a7` 3.350, 64 `0c0b` 2.462, 128 and 256 `0678` 1.782 and 1.425, `c784` 1.466 and 1.151; template `f440` at 1–256 rows 2.390, 2.450, 2.370, 2.479, 3.101, 3.091, 2.401, 1.755, 1.457; one run, recorded in d908d01, the child of 43e112e, unchanged at 0e02a81 (qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048; AKLR@d908d01 CI run 42000899019)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (N=4 for the 1–4-row call times), `python -m aklr validate` on committed decode files and modified copies, 4 alternated rounds per arm, kernel time the mean of the rounds' median candidate times, variants that fail accuracy run in-process on random inputs against the committed file's output; Gluon probe kernels: a 1 GiB tensor streamed once per launch (2048 programs × 4 warps, 16 bytes per lane unless stated), median of 9 rounds × 10 launches, variants interleaved; a [64, 64] 32-bit tile converted A → B → A 1024 times in 1024 programs, median of 7 rounds × 3 launches; ranges over the runs] `29913388` at 2 rows with 16-row value tiles (256 CTAs) 8.59–8.73 µs against 10.55–10.62 µs with whole-head tiles of the same kernel (128 rows, 32 CTAs, 267 VGPRs), 17.7–18.6% less, at 16 rows 14.76–14.78 against 13.19–13.23 µs; tiles of 8, 16 and 32 rows at 2 rows 8.640–8.655, 8.631–8.665 and 8.601–8.714 µs; its history shift moved into launch 1: `conv_state` wrong on 10–11 of 20 inputs per run, `delta_state` off by up to 0.95–1.46, the committed file exact on all 20 in every run; files run at other row counts: at 24 rows `a4f8` 1.821–1.833 and `409c` 1.809–1.823, at 192 rows `fc26` 1.339–1.342 and `409c` 1.042–1.043; `f440` at 32 rows with inline history retirement: `delta_state` wrong on 10 of 10 inputs, max |d| 3.243 in every run, warm-cache graph replay B/A 0.986–0.999 (7.48–7.62 against 7.58–7.63 µs); `20f6` (32 rows) state load after the gates with the SiLU early: B/A per pair 1.000–1.018 (17.91–18.04 → 18.16–18.27 µs); `4f25` (8 rows) q loaded and normalized before the state load 1.018–1.039 (9.47–9.49 → 9.67–9.76 µs), decay and β in each update program 1.062–1.074 (9.47–9.53 → 10.13–10.16 µs); `2c2a` (1 row) against scalar exponentials (7 against 4 `v_exp_f32`) B/A per pair 0.986–1.013 (8.65–8.73 against 8.64–8.71 µs); 8 readers of one chunk, ids congruent mod 8 12.1–12.4 µs against 21.6–22.0 µs consecutive (1.75–1.80×); `fc26` (128 rows) with `token = pid // VH` B/A per pair 1.005–1.011 (45.62–45.93 → 46.09–46.21 µs); state read plus write 5.85–5.89 TB/s by `fc26` (45.6–45.9 µs) and 6.57–6.59 TB/s by `c2c3` (81.5–81.7 µs), 1.06–1.20 of a 5.48–5.51 TB/s `.cg` read stream, the `c2c3` baseline 106.4–106.7 µs (0.91–0.92 of it; ratio 1.30–1.31), at 2 rows 8.7–8.8 and 21.6–21.7 µs (0.087–0.088 and 0.035; ratio 2.47–2.48); `29913388` at 2 rows: two kernels in every Kineto trace, median call time per validate 8.55–8.87 µs (run means 8.65, 8.75, 8.68 µs), its baseline 21.20–21.78 µs, 4 validates per run; the 1-, 2- and 4-row files 8.62–9.04, 8.81–9.14 and 8.81–9.13 µs in three runs and 9.09, 8.65 and 9.10 µs in a fourth (t2/t1 0.951–1.022, t4/t1 1.001–1.022), the 2-row call 0.46–0.485 TB/s against 6.72–6.78 TB/s for a 1 GiB `.cg` Gluon read stream (0.068–0.072 of it); 256 programs × 4 warps × 8 loads per lane 7129–7144 GB/s, 2048 × 4 × 1 6674–6684 GB/s (1.067–1.070×); at 3 MB in flight 96 × 4 × 8 3945–3995 GB/s, 384 × 4 × 2 4920–4962 GB/s, 192 × 1 warp × 16 2454–2467 GB/s (0.80–0.81× and 0.495–0.499× of 384 × 4 × 2); reads: no hint 5993–6013 GB/s, `.cg`/`.cs` 6771–6837 GB/s (1.130–1.137×), buffer over pointer 1.002–1.005 hinted and 1.002–1.003 unhinted; writes: no hint 5792–5889 GB/s, `.cs` 5404–5510 (0.928–0.945×; buffer over pointer 0.997–1.012), `.wt` 0.998–1.005× through `buffer_store` (above 1.00 in two runs) and 0.983–0.996× through `gl.store`, below every `gl.store` without cache bits (none, `.wb`, `.cg`) in each run; lane-moving conversions 574–1522 ns per conversion against 73–74 ns for the loop alone, the 96-`ds_bpermute` path 661–663 ns, the vectorized LDS round trip 574–581 ns (Qwen3.5 GDN decode `299133886d479017`, `2c2a09e220f2706c`, `409cb22b39b3fa00`, `4f257a3e566b20d1`, `a4f80b6a3877b48e`, `20f63094b85c00a2`, `fc261961f0ad572d`, `c2c3a42a3909684a`; Qwen3-Next `f44007374038041f`; probe kernels; #89)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, each schema called 5 times at its first and last listed Point on fresh copies of the inputs, outputs compared byte by byte; sources scanned for atomics] all 44 GDN files: no atomic call; 61 file-Points, 305 calls, every repeat byte-identical to the first; the 13 Points between a file's first and last not run (the 18 decode and 26 prefill schemas as at AKLR@43e112e; #89)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `python -m aklr validate`, 2 rounds per arm, arms alternated in a fixed order, one GPU] Qwen3.5 GDN FP8 decode files run at row counts between their Points, the log-nearest file against the other neighbour, call time (first run / second): 3 rows `409c` 8.97 / 8.87 against `2991` 8.90 / 8.85 µs; 6 rows `4f25` 9.18 / 9.18 against `409c` 9.10 / 8.98; 12 rows `a4f8` 10.37 / 10.35 against `4f25` 10.54 / 10.62; 24 rows `20f6` 16.23 / 16.02 against `a4f8` 15.59 / 15.48; 48 rows `20f6` 23.25 / 23.50 against `fc26` 21.53 / 21.60; 96 rows `fc26` 36.39 / 36.38 against `20f6` 39.26 / 39.92 (`409cb22b39b3fa00`, `299133886d479017`, `4f257a3e566b20d1`, `a4f80b6a3877b48e`, `20f63094b85c00a2`, `fc261961f0ad572d`; #89)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 100 repeats per arm against a CPU model, compiled and run on one GPU] a toy Gluon kernel shifting a 3-entry history of 64 channels that every wave reads and wave 0 writes, a spin delaying the readers: the new entry stored before the reads was read in place of the old one in every repeat, 64 wrong values at one wave, 256 at four; no barrier between the reads and the writes: 192 wrong at four waves in every repeat, none at one wave; a full-workgroup barrier before the writes: none wrong at either (toy kernel; #89)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 sessions for the A/A part, N=1 for the state-access A/Bs, `python -m aklr validate`, one GPU, 4 runs of A alternated with 4 of B in one session] A/A, the committed schema against an identical copy: `29913388` at 2 rows 2.4222, 2.4105, 2.4522, 2.4595, the copy 2.4254, 2.4118, 2.4434, 2.4374: 1.73% spread over the first 3, 2.03% over the 8, the 4-run means 0.27% apart; over 7 Points the largest gap between the two 4-run means 1.31% (fp8 GEMM m = 16, its single runs spread 2.4%); second session: 2.4914, 2.4960, 2.5015, 2.5065, the copy 2.4728, 2.4816, 2.4956, 2.5204: 0.41% over the first 3, 1.92% over the 8, means 0.25% apart, largest gap over the 7 Points 0.42% (fp8 GEMM m = 16, single runs spread 2.5%); state-access A/Bs: `.cs` removed from `29913388`'s state `buffer_load`/`buffer_store`: speedup 2.452 → 2.442, 8.748 → 8.789 µs (pairs 0.99–1.015); the state as `gl.load`/`gl.store` with `.cs`: 2.429 → 2.432, 8.827 → 8.813 µs (pairs 0.983–1.02); `.wt` dropped from the state `buffer_store` of `fc261961` (128 rows) and `c2c3a42a` (256 rows): 45.77 → 46.97 and 81.79 → 83.90 µs, the store as `gl.store` with `.cs`: 45.80 → 47.04 and 81.81 → 84.78 µs (`299133886d479017`, `fc261961f0ad572d`, `c2c3a42a3909684a`; #89)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (slope readings repeated in three more runs), Gluon probe kernels on one GPU: chained trivial 1-warp kernels in a CUDA graph, timed by AKLR's `bench_cudagraph` before #33 (Kineto, `GPU_FORCE_QUEUE_PROFILING=1`) and by the bare replay's wall clock, eager launches by the host clock] AKLR's bench reading grows 4.062 µs per added kernel (Kineto sum 3.902), a bare replay 1.815 µs (grid 1) and 1.844 µs (grid 256); two chained kernels read 8.246 µs on AKLR's bench; eager host time 1.004 µs per tensor argument, while AKLR's bench reads 4.162–4.178 µs for kernels with 1, 8 and 32 tensors; the repeats: bench 4.046 and 4.051 µs per kernel, bare replay 1.771 and 1.766 µs, two chained kernels 8.29–8.31 µs (toy kernels; #89)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] Qwen3.5 GDN prefill, stored speedups per Point as in §8: `1fd6` 1024/1 2.423, `1d55` 512/1 2.638 and 1024/2 1.889, `b8c5` 16384/2 1.339 at 43e112e (since d908d01 `1fd6` 2.452, `1d55` 2.681 and 1.927, `b8c5` unchanged); `3b99` 16384/2 1.444, `7a69` 10000/1 1.188, `19b2` 1025/1 2.256, `4709` 16384/1 1.741, `12b5` 5000/1 1.199 and 15000/3 1.198, `5ead` 8192/1 1.193, `24ed` 15361/8 1.454 and 16384/8 1.475, `3828` 2048/1 2.092 at 369a3c0; `61ed` 512/2 2.629 at 62a5fe8; `0a07` 2049/2 1.374 and 4096/2 1.452 at 5f63e8a; `2ece` 4097/3 1.029 and 8192/4 1.424, `c53b` 8193/5 0.997 and 15360/8 1.473 at 3fb9033; Qwen3-Next GDN prefill, one run recorded in d908d01: `11f6` 1024/1 1.573, 2048/2 1.434; `4489` / `d8f9` 3072/3 1.592 / 1.554, 4096/4 1.504 / 1.501, 8192/8 1.490 / 1.484, 12288/12 1.288 / 1.282, 16384/16 1.463 / 1.460; `70f4` / `010e` 13312/13 1.232 / 1.233, 15360/15 1.328 / 1.329; `162f` / `a99f` 4096/1 1.718 / 1.719; `c3a7` / `f137` 5120/1 1.727 / 1.741, 10240/2 1.308 / 1.311; `ed2a` / `98af` 16384/4 1.294 / 1.283; all as stored at 0e02a81 except the three values noted (gdn_prefill_group_fp8_quant_* and qwen3_next_gdn_prefill_group_fp8_quant_* directories; AKLR@43e112e CI run 41991630592, AKLR@d908d01 CI run 42000899019, AKLR@369a3c0 CI run 41753978959, AKLR@62a5fe8 CI run 41778680267, AKLR@5f63e8a CI run 41956295934, AKLR@3fb9033 CI run 41812750710)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, a register-operand loop of one MFMA shape per kernel on all CUs] BF16 `16x16x32` 2447–2468 TFLOP/s, fp32 `16x16x4` 155.2–156.1 TFLOP/s, 15.71–15.84× (Gluon probe kernel; #89)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the prefill graph capture N=3), `python -m aklr validate` with poisoned scratch, and of all 26 GDN prefill files; JIT binaries counted around a CUDA-graph capture (relaxed mode); the prefill routes called with mocked sglang layers] a `4f25` copy that skips the `scales` store of odd rows: `numeric_mismatch` at `first_capture/0`, 64 non-finite mismatches in `scales`; the committed file: no finding; `a4f8` with `SLOTS: gl.constexpr`: 1, 2, 3, 4 binaries after pool sizes 33, 34, 35 and, under capture, 36; with a runtime `SLOTS` 1, 1, 1, 1; no capture error; the 26 prefill files that carry `_initialize_fresh`: all benched (capture under `capture_error_mode='global'`, replay with new inputs), no capture, accuracy or mutation finding, exit 0; the mocked Qwen3.5 and Qwen3-Next prefill routes ran eager, returned `out_of_contract` under capture and ran eager again (call counters 1, 1, 2) (`4f257a3e566b20d1`, `a4f80b6a3877b48e`, the 26 prefill schemas; routes in aklr/plugins/sglang/models/qwen3_5/api.py at 0e02a81; #89)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, kernels compiled for gfx950, VGPRs, waves per SIMD and instructions read from the `.amdgcn`; probes and committed files run on the GPU against fp64, torch or the contract's reference; one A/B with `python -m aklr validate` of 4 rounds] `29913388` with q read together with k: identical instructions, 61 VGPRs; template `f440` state load early against late: +7 VGPRs at 1 row (65 committed), 0 at 32 (160), +77 at 128, +12 at 256, no spills; accumulate against recompute form in `_fused_heads`: 138 against 134 VGPRs at 16 rows, 131 against 132 at 64, 3 waves per SIMD, no spills; decay and β before the conv at 128 rows: 200 VGPRs either way, 2 waves per SIMD; at 32 rows the late state load 10.28 µs against 8.39 µs early, 160 VGPRs each (speedup 3.18 → 2.58; the other 8 Points 0.997–1.006); one-wave product of two 16 × 16 fp32 accumulators: 69 instructions with `k_width = 4`, 69 relabeled, 79 with two permlane steps (4 swaps), 92 direct, 1024 B of LDS each; relabeled result 9.727e-8 relative to fp64 (`k_width = 4` the same, plain route 1.007e-7); `k_width = 1` from the transposed accumulator 4 `ds_write_b32` + 4 `ds_read_b32`, 1024 B, no `ds_bpermute`, a one-transposition step 2 `v_permlane32_swap`; 64 × 64 tiles, 4 waves: only (transposed, `[4, 1]`, A) and (not transposed, `[1, 4]`, B) at `k_width = 4` pass `assert_trivial`; SEQ16's 64 × 128 `[4, 1]` state: direct 8 `ds_write_b128` + 8 `ds_read_b128`, 16384 B, relabeled no LDS, 32 `v_mfma_f32_16x16x4_f32`; 2^20 random BF16 pairs with exponents within ±63: every fp32 product equals the fp64 one; 16×32 @ 32×16 tiles (4194304 elements), the fp32 operand split into BF16 terms: equal to the correctly rounded BF16 in 58.2% (high term), 99.76% (two RNE terms), 100% (RTZ, RTZ, RNE); 300000 random fp32 values with exponents within ±40: two RNE terms split error at most 7.605e-6 relative, three RTZ terms exact; 10000 chunks of 64 tokens, log-decays uniform in [−6, 0]: the prefix ratio NaN at 4,242,274 and 0 at 4,179,454 of 14,078,262 entries whose fp64 value is a normal fp32, a masked add-scan and one exponential within 3.76e-6 relative; padded rows of `29913388` and `4f25`: codes zero-valued, about half of the bytes 0x80, every scale 0x2a7b5124 = 2.23214296e-13; `11f6` within the contract at 128 sequences, at 129 normalized excess 161, with eight halvings within it at 129; one recurrence step on a random 128 × 128 fp32 state with decay 1 and β 0: no bit changed, v NaN at rows 3 and 17 made those rows NaN; `2c2a` (1 row) with its BF16 core scratch replaced by an fp32 scratch of the BF16-rounded core: outputs bitwise equal on 10 of 10 inputs, 4 of them not padding; with the unrounded core equal on the 6 padding inputs only; fp32 `.to(float8e4nv)`, `.to(float8e5)` and an fp32 store into an e4m3 pointer: one MODE bit-23 write (value 1) at instruction 6, before the first VALU; the 2-row `_normalize_update_history`: bit 23 set at instruction 6, before its first `v_cvt_pk_bf16_f32` (instruction 173); with the bit set fp32 → BF16 of finite overflowing values 0x7F7F/0xFF7F and of Inf 0x7F80/0xFF80, with it clear 0x7F80/0xFF80; `b31b`'s `_round_bf16_ieee` next to a Triton fp8 store: no bit mismatch against torch on 60416 values; the template's committed rows: IEEE fix-up at 1 row, native convert at 2, 4, 8 and 32, neither at 16, 64, 128 and 256; 16 re-splits of `cu_seqlens` at the 8 multi-sequence prefill Points with unequal lengths changed no grid; 12000 random ragged batches: chunk slots distinct and below `cdiv(m, BT) + batch`; fp32 `torch.empty` scratch per call at the 8–16K-token summary Points 173.1 (`12b5`, 15000/3) to 294.0 MB (`7a69`, 10000/1), `7a69` factors 246.2 MB and maps 35.7 MB (Qwen3.5 `299133886d479017`, `4f257a3e566b20d1`, `2c2a09e220f2706c`, `7a69337f81190952`, `12b560c8`; Qwen3-Next `f44007374038041f`, `b31bfeccf6dcfd7c`, `11f62a16b4778338`; probe kernels; #89)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `python -m aklr validate`, 4 alternated rounds per arm on one GPU, A = the committed template `f440`, B = a copy whose 16- and 64-row configs get `waves_per_eu=4`] B takes 1.598 and 1.603× (16 rows) and 1.595 and 1.629× (64 rows) as long as A (speedup ratio 0.62 and 0.62 at 16 rows, 0.63 and 0.61 at 64), both B kernels at 128 VGPRs with 9 and 20 spilled, 2.6% and 5.3% more instructions (`f44007374038041f`; #89)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `python -m aklr validate` on committed Qwen3.5 prefill files, modified copies and committed kernels placed in another file's scenario, 4 rounds per arm alternated, time = the median candidate call time; compiles for gfx950 read from the `.amdgcn`; values given first run / second] 2048/1: CKPT `3828` 107.3 / 106.9 µs, the RSUM source `4709` placed in that scenario 214.2 / 212.9 µs; SEQ16 (`24ed` source) over MAP `5ead` at 8192/1 2.712 / 2.708×, over RSUM `4709` at 16384/1 3.977 / 3.968×; mean call time, the committed file against another design's kernel in its scenario (placed kernels' accuracy not checked): 1024/1 D `1fd6` 67.4 / 67.2 against CKPT 68.3 / 67.9 µs, 4096/2 SEQ32L `0a07` 193.7 / 192.9 against SEQ16 `24ed` 374.5 / 375.5 µs, 8192/1 MAP `5ead` 495.3 / 497.2 against RSUM `b8c5` 363.9 / 366.0 µs, 16384/8 SEQ16 `24ed` 507.4 / 507.1 against RSUM 651.1 / 652.3 µs; t(4097/3)/t(8192/4) 0.974 / 0.974 (`2ece`), t(8193/5)/t(15360/8) 0.855 / 0.856 (`c53b`), work ratios 0.50 and 0.53; the other value strip, BV 16 at 4097/3 1.317 / 1.317× and BV 32 at 8193/5 1.153 / 1.153× the committed time; `24ed` without the `gl.barrier()` before `_write_conv_state` (`_prepare_boundaries`, one wave, 74 VGPRs either way), 10 validates per arm at 15361/8 and 16384/8: no accuracy finding in either arm, `conv_state` exact in all 10 B runs / all 10; `24ed` at 16384/8, B/A per pair: next chunk's loads after the math (218 against 222 VGPRs) 1.0196, 1.0211, 1.0199, 1.0233 / 1.0223, 1.0242, 1.0224, 1.0231, `.cg` dropped on the value tile 0.9986, 1.0032, 0.9984, 1.0011 / 1.0023, 1.0007, 1.0006, 1.0012; `3828` at 2048/1, B/A call time per pair: one 128-deep chain (576 MFMAs, 108 against 114 VGPRs) 1.0453, 1.0405, 1.0313, 1.0459 / 1.0386, 1.0426, 1.0426, 1.0421; unroll 1 (64 MFMAs against 576) 1.0675–1.0694 / 1.0629–1.0701; `iterative-ilp` (present in A's LLVM IR) removed 1.1183–1.1334 / 1.1224–1.1294; masked branch forced 1.0237, 1.0166, 1.0124, 1.0202 / 1.0087, 1.0158, 1.0144, 1.0113 (9 dwordx4 and 27 ushort loads, 58 VGPRs in both arms); `2ece` `_build_chunk_tables_paired` at M = 8192, half tables stored once final 190 VGPRs, all stores at the end 194, no spill (compile only, the same in every session); speedups `1d55` 2.646 / 2.654 at 512/1 and 1.913 / 1.906 at 1024/2, `4709` 1.737 / 1.730 at 16384/1, `b8c5` 1.359 / 1.360 at 16384/2; `0a07` at 4096/2: K Kᵀ and Q Kᵀ in each value head (`_chunk_factors` 160 against 136 MFMAs, 254 against 170 VGPRs), accuracy passed, B/A 0.9777, 1.0235, 1.0225, 1.0221 / 1.0173, 1.0236, 1.0233, 1.0210 (means 1.011 / 1.021); the zero blocks multiplied too (184 against 136 MFMAs, 182 against 170 VGPRs) 0.9655, 1.0063, 1.0060, 1.0077 / 1.0005, 1.0059, 1.0062, 1.0017; `2ece` at 4097/3 with zeros stored into `packed_qk`'s tail rows and unmasked loads: builder 24 `global_load_dwordx4` and 3 ushort loads, 190 VGPRs, no spill, against 0, 195 and 256 + 64 spilled, `waves_per_eu` 2 in both, accuracy passed in every B run, B/A 0.8957, 0.8965, 0.9024, 0.8965 / 0.8966, 0.8957, 0.8983, 0.8950 (Qwen3.5 GDN prefill `382853eb770e30b3`, `47092631ad3a6e79`, `24ede969ed66a87a`, `5ead0483`, `1fd6adb2b91cf5c2`, `0a07f9f9342c6cf1`, `b8c54bde37ee77c7`, `2ecece9650b95f71`, `c53b6d87`, `1d5568a78971fb62`; #89)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compiled without a GPU: 38 fp8 cast variants and 7 controls, MODE writes read from the `.amdgcn`] each of the 38 E4M3FN/E5M2 casts from fp32, fp16 and bf16 (default and rtne `.to`, E5M2 rtz from fp32 and fp16, an implicit cast in `gl.store`, an explicit cast then `buffer_store`, a runtime branch) wrote MODE.FP16_OVFL = 1 once, before the first VALU of the function holding the cast: the kernel when inlined, the callee in the six noinline variants; no later setreg; a direct wider-value `buffer_store` and E4M3FN rtz did not compile; the seven controls (upcast, bitcast, MFMA, asm, BF16) wrote no MODE (toy Gluon kernels; #89)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 study of 5 Points, 16 adjacent pairs (8 AB, 8 BA) per comparison, eager single calls timed by device events (host enqueue and allocation gaps included), 99.5% bootstrap intervals per comparison with A/A guards] Points (tokens, sequences, max_seq_len) (512, 1, 512), (1024, 2, 512), (512, 2, 256), (1024, 1, 1024), (1025, 1, 1025); D `1d5568a7`, CKPT `382853eb`, RSUM `47092631`, MAP `5ead0483`, SEQ32L `0a07f9f9`, each on its own host branch without tuning: all 25 Point-arms passed the contract's accuracy bounds against an fp64 sequential reference on every full mutable tensor (2 seeds or state modes × 2 scratch and output poisons, 500 comparisons), unindexed pool slots untouched, outputs independent of the poison; two-sequence inputs of equal length only; of 35 A/B comparisons, D and CKPT each faster than RSUM and MAP at all five Points, CKPT faster than D at all five, CKPT against SEQ32L unresolved at the second and third (Qwen3.5 GDN prefill `1d5568a78971fb62`, `382853eb770e30b3`, `47092631ad3a6e79`, `5ead0483`, `0a07f9f9342c6cf1`; #89)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compiled without a GPU: 7 four-wave builds; an earlier sweep of 432 builds, compiled twice] the `16x16x4` accumulator as the A operand at `k_width = 1`: two transpositions and 16 KiB of LDS at 128 × 32, transposed or not; the transposed `32x32x2` accumulator at 256 × 64: one, no LDS, 32 permlane32 instructions; untransposed: four, 64 KiB of LDS; in the sweep, conversions stayed in-wave whenever fewer than two transpositions were needed and the warps did not change (toy Gluon kernels; #89)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 8 balanced repeat clusters] a 256-step chain of products on 64 × 16 fp32 tiles, 4 waves, 16 programs per CU: `k_width = 4` operands 337.2 µs against 256.8 µs with the direct `k_width = 1` conversion (1.313×), values correct, largest relative error 2.587e-6 (toy Gluon kernel; #89)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, a one-row Triton toy, 16 key heads × 4 tile CTAs (4-tap conv, SiLU, L2 norm), outputs compared; CUDA-graph replay time and AKLR's timer before #33 (Kineto)] recompute over staged replay time 0.709, 0.693 and 0.775 in three runs, 1.014 in the fourth (9.967 against 9.832 µs, relative L2 difference of the outputs 0); on AKLR's timer 0.589–0.610 in all four (5.117 against 8.691 µs in the fourth) (toy kernel; #89)
- S1 aklr/plugins/sglang/models/qwen3_5/reference.py:113 decode rounding points (to :121): each conv tap product to BF16, SiLU output, beta, core, normalized output, then E4M3FN; padding rows; q times `scale` :148; `reference_prefill` :19 has no docstring, rounds q and k to BF16 :77-78, multiplies the core by `scale` :88, starts a fresh sequence from zero history :63, history = the window's last three :74. aklr/plugins/sglang/models/qwen3_next/reference.py:4-11: q and k rounded to BF16 in prefill only (:132-133), core times `scale` :153, decode q times `scale` :197
- S2 aklr/plugins/sglang/models/qwen3_5/contract.py:116 `conv_state` bound 0/0 (to :122); `CompareSpec(determinism="bitwise")` :189; `n_slots` 1–2047, `any_size`, :74; Qwen3-Next: aklr/plugins/sglang/models/qwen3_next/contract.py:99-105, :236, :283
- S3 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/7a69337f81190952.json#kernel.py:249 WY-form docstring of `_block_factors` (to :258); W and U from the inverse :343-351; `Ktail` from the last row of the relative decay :352-355, :365; `response` = tril(Q Kᵀ ⊙ G) :367-380
- S4 aklr/plugins/sglang/models/qwen3_5/baseline.py:127 decode baseline: `fused_qkvzba_split_reshape_cat_contiguous` :154, `causal_conv1d_update` :162, `fused_recurrent_gated_delta_rule_packed_decode` :171, `_layer_norm_fwd` :183, aiter `per_1x128` :227; the prefill baseline (:28) ends with `per_token_group_quant_fp8` :123; aklr/plugins/sglang/models/qwen3_next/baseline.py:38 aiter `per_1x128` in `_normalize_quantize`, used by prefill :112 and decode :170
- S5 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2c2a09e220f2706c.json#kernel.py:8 "The second launch normalizes the BF16 core output and advances convolution history, after every first-launch history reader has completed" (:8-9); `_value_and_gates` "Evaluate value SiLU, decay, and BF16 beta in neighboring lanes" :77-110; q/k in a `(2, BK)` fp32 LDS buffer :138-140; state `gl.load` :182, `gl.store` :202, no hint; norm layout and "stays wave-local" :214-215; `value_tile = 8` :312; launches :315-327
- S6 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/299133886d479017.json#kernel.py:116 state `buffer_load(..., cache=".cs")` (:116-118), store :141-143; `_store_quantized_head` :17-35, the same function text (md5) in the other seven fp8 decode files; state layout :87; `if slot >= 0:` :95, zero core otherwise :94, :146; history shift under `if slot >= 0:` in the epilogue :189-204; Q/K conv in the recurrence CTA :99-105; q in LDS :130-145; launches :234-246
- S7 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/4f257a3e566b20d1.json#kernel.py:77 history shift before arithmetic (:77-80); coefficient program "Queue the coefficient waves first ..." :44-60 on the grid `(tokens, cdiv(channels, 64) + 1)` :200; padding still normalizes and quantizes :110-121; epilogue loads early :133; state `.cg` :146, `.cs` :160; late Q :152; BF16 before the cross-wave convert :162-164; launches :200-212
- S8 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/409cb22b39b3fa00.json#kernel.py:206 64-bit head rebase for pools over 2 GiB (:206-216); host byte test :318-319; state `.cg` :218, `.cs` :231; grid :327; Q, K, V and β in a `(4 * D,)` fp32 LDS buffer :143-146
- S9 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/20f63094b85c00a2.json#kernel.py:167 "Expose the large state fetch before preparing scalar gates"; epilogue inputs early :159; "Interleave heads over token groups for state-pool traffic." and the mapping :136-139; state `gl.load` :171, `gl.store` :194; launches :233-245, token group 32, 16 or 1 by divisibility :243; a4f80b6a3877b48e.json#kernel.py:254 `gl.load(..., cache_modifier=".cg")`, store `.cs` :259, launches :298-312; `SLOTS: gl.constexpr` :221, `gl.assume(slot < SLOTS)` :244, `SLOTS=slots` :310; token groups :227, XOR head permutation :229-231
- S10 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/fc261961f0ad572d.json#kernel.py:8 "fixed, spill-free 16-wave layout"; `token = (pid // (VH * 8)) * 8 + pid % 8` :182 on the grid `(tokens * v_heads,)` :300; layout :186; state load :202, `.wt` store :221; `num_warps=16`, `waves_per_eu=6 if tokens == 64 else 8` :321-322; the same in c2c3a42a3909684a.json#kernel.py:320-321 (mapping :180) and gdn_decode_n_v_heads16_key_dim128_value_dim128_conv_dim3072/cd00c50e595b05bc.json#kernel.py:299-300 (mapping :159)
- S11 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:467 `waves_per_eu` N becomes `amdgpu-waves-per-eu` = "N,N" (:462-468)
- S12 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#kernel.py:5 "split" from 1 to 8 rows, "fused" from 16 (:5-11); per-size grids: b31bfeccf6dcfd7c.json#kernel.py:463-466, bba17a5608327b11.json#kernel.py:359-367, 371905e46b19ee36.json#kernel.py:450-463, 35b9cfd56833844f.json#kernel.py:508-511, 06780bcbbb3e3c70.json#kernel.py:176-182
- S13 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_decode_config.py:76 `state_prefetch`, "the single most important knob for register pressure" (:76-83); `update_form` :85-89, "accumulate" at 16 and 64 rows :237, :264; "VEC * 64 * NW <= conv_width" and the 32-row two-owner case :100-106; `CONFIGS` :187, `state_tiling="heads"` at 16 and 64 rows :236, :263; exact-match `lookup` :323; kernel.py:977-992 `_fused_heads` launched without the cache and gate-position fields, `.cg` loads :728-737 and `.cs` stores :867-872 hard-coded; decay and β after the conv, "halve occupancy on the 4-warp instance" :482-484
- S14 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/06780bcbbb3e3c70.json#kernel.py:70 `gl.static_assert(NW * 64 <= conv_width)` on `BlockedLayout([1], [64], [NW], [0])` :78; "no replicated lanes/waves" :79-80; inline shift :48-51
- S15 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/c7840603329d5ed5.json#kernel.py:40 `gl.barrier()` between the history loads (:24-28) and stores (:41-44); 38a742cfd908c997.json#kernel.py:312 `packed_history = T == 32`, history retired after "the full-CTA core redistribution" :447-454
- S16 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp:719 a barrier on local and global spaces becomes release fence, `s_barrier`, acquire fence (:719-736); TritonGPUToLLVM.cpp:113 Membar analysis runs before lowering (:113-115)
- S17 aklr/core/utils/testing.py:164 `bench_cudagraph`: a call's time is the sum of the device durations of the kernels in its profiler scope, from AKLR's rocprofiler-sdk collector (`_collect_scope_times`, :139-160), the flush outside the scope (:206-210)
- S18 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:492 buffer-op `.cs` sets sc0 and nt; pointer loads and stores get only the nontemporal flag for `.cs` (:305-336)
- S19 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:104 `num_records` INT_MAX − 1 bytes; element offset times element size in 32 bits and masked lanes selected to an out-of-range offset :282-283
- S20 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp:433 `ClampFOp` lowers to `ROCDL::FMed3Op` (:433-451)
- S21 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp:2470 an fp8 destination type sets the FP8 clamping attribute on the function (:2470-2484); ElementwiseOpToLLVM.cpp:468-498 `adjustModeRegister` then inserts `llvm.amdgcn.s.setreg` of MODE bit 23 = 1 at the function entry, called from TritonGPUToLLVM.cpp:250
- S23 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/b31bfeccf6dcfd7c.json#kernel.py:306 `_round_bf16_ieee` (:306-313), used at :365 because `_finish_owned` passes `NATIVE_FP8=False` (:425-428), so the clamped Triton fp8 store :323 runs and the native branch :324-339 does not
- S24 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_primitives.py:476 native `store_fp8` branch, clamp then `v_cvt_scalef32_pk_fp8_f32` (:476-493); `ieee_pid` and `fp8_mode_scale`, `is_pure=False` (:741-754); kernel.py:304-305 and :430-431 `ieee_pid` at entry, :166-167 `fp8_mode_scale` before the quotient; `native_fp8_cvt=True` at 2, 4, 8, 32 rows gdn_decode_config.py:210, :221, :232, :258; 38a742cfd908c997.json#kernel.py:58-61 the setreg inside the convert asm, its `_ieee_pid`/`_fp8_scale` guarded off at its 32-row fp8 Point (:257, :476)
- S25 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_primitives.py:451 "deliberately a *separate* knob" (:450-453); accumulate form and "64 VGPRs at 8 warps, which is the difference between 4 and 3 waves/SIMD" :648-657
- S26 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#initial_state.py:9 `_initialize_fresh` zeroes a slot of the pool it is given when the device flag says fresh (:9-26); packing.py:4 "The conversion is part of candidate execution, including graph timing."; `_pack` writes Qwen3-Next's grouped order :15-43
- S27 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#kernel.py:236 `gl.barrier()` then `_write_conv_state` (:236-237); old-history reads masked with the fresh flag :184-217; `_write_conv_state` keeps old entries for short sequences :134-152; `_permute_columns_for_a` :246-252, `_permute_rows_for_b` :256-262; state kept permuted :490, `assert_trivial=True` :554-557, un-permuted at the store :581; next chunk's Q/K and V loaded before its math :499-516, :524-544, `.cg` on V :515, :543; V through `SwizzledSharedLayout(4, 1, 8, [0, 1])` :530-536; BV 64 :471; `_chunk_coefficients[chunks, heads // 2]` :723, Grams once per head pair :363-367; `_delta_chunks[heads, 2, batch]` :739; `_prepare_boundaries` launched with `num_warps=1` :706-722
- S28 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/010eab6660c4e415.json#kernel.py:189 binary search over chunk bases `starts[s] // BT + s` (:189-196); "reserves enough space for every ragged tail" :199; 448925c31bc8ea55.json#kernel.py:1008 `chunks = triton.cdiv(m, bt) + batch`, :370-371 programs past the device-written chunk count do nothing
- S29 aklr/plugins/sglang/models/qwen3_5/api.py:161 prefill falls back while a stream is capturing
- S30 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2ecece9650b95f71.json#kernel.py:1812 `_chunk_recurrent` grid `(batch, heads, 128 // bv)`, bv 32 :1738; short path `if m <= 4096` :1757; `_build_chunk_tables_paired[chunks, heads // 2]` :1779 with `waves_per_eu=2` :1792; `_half_decay_scan`, two 16-token halves and a separable cross block :595-641; `_inverse_left_operand` :695-720, both callers pass `SHORT=True` (:1006, :1075), so its first middle layout :701-707 runs; `_stream_chunk_tables` "Store finished FP32 half tables early to shorten register lifetimes" :1002; `complete_chunks: gl.constexpr = M % BT == 0` and the row mask :1277-1278, operand loads :1279-1280, :1320; packed table reshaped and converted with `assert_trivial=True` :1430-1437; `packed_qk` in whole 32-token chunks :1743, its store masked at ragged M :475-484
- S31 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/0a07f9f9342c6cf1.json#kernel.py:1333 BV 16 below 4097 tokens, grid :1422-1427; `_upper_dense_product` skips the zero block :449-466; `_chunk_factors` "share four-head Gram matrices" :851 on `[chunks, heads // 4]` :1404; recurrence in a transposed `[1, 4]` layout at BV 16 :1057-1072; `_state_projection_operand` with `SharedLinearLayout` and `_keep_alive()` right after the only load :993-1033; `large = m > 4096` :1332 with `normalized` the caller's buffer :1339 and `NORM_STRIDE == (8 + HEADS) * 128` asserted :1261-1263
- S32 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/7a69337f81190952.json#kernel.py:792 MAP chunk `max(256, cdiv(m, batch * 16 * block) * block)`, propagation grid :861; scratch `factors` and `maps` :744-755; reads the contiguous projection directly :89-99; register window :73-75, taps `gl.amd.slice(raw, [BT, 128], [j, 0])` :122, :142; gates converted to a blocked layout for the scan :301-305; masked product scan :310-319; `max_seq_len` taken by `run` and not read :924 (each of the 26 prefill kernel.py files names it once); 3b99d19659746a36.json#kernel.py:807 RSUM chunk 1024 or 512, boundary grid :873, padding decay 1 and beta 0 :289-290, `k_width = 4` with `[waves // 2, 2]` warps in the summaries :410-421 (b8c54bde37ee77c7 and 47092631ad3a6e79 are the same source)
- S33 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/382853eb770e30b3.json#kernel.py:1614 checkpoint scan grid, 64-token chunks :1538, 16 value rows :1539; `m <= 1024` selects the short path :1541-1547; `_convolution_interior` :59-73, branch :301; conv window in LDS read with `Window.gather` :254; `_decay_products` dyadic levels :670-697 with `31 - clz` :687; tiles stored only where `key_tile <= query_tile` :1008; the scan's `transposed=False`, `[1, 4]` layout :1294-1296, state and update through LDS read as `DotOperandLayout(0, mma, 4)` :1234, :1243-1247, :1266-1267; two 64-deep MFMA chains :1249-1255; `loop_range(..., loop_unroll_factor=8)` :1313; `iterative-ilp` :1627
- S34 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len1024_n_v_heads16_key_dim128_value_dim128_conv_dim3072/1fd6adb2b91cf5c2.json#kernel.py:19 "Only output reconstruction uses compensated BF16 products" (:19-21); four products :1302-1312; chunk 64 :1546, value block 8 at 1024 tokens else 16 :1615-1616 (same source as 61ed32f62fa797df and 1d5568a78971fb62); 19b20285ceb08b14.json#kernel.py:574 `_mm_query` (:574-599), same source as 382853eb770e30b3
- S35 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/11f62a16b4778338.json#kernel.py:791 `C=32 if batch >= 16 or 5 <= batch <= 7 else 64`; sequence search in `for depth in gl.static_range(0, 7)` :76-81 (1–2047 sequences in aklr/plugins/sglang/models/qwen3_next/contract.py:180); `_inverse16` "Inverse of I-P for a strictly lower triangular 16-by-16 P" :193-203, `_inverse32` joins two :218-226; "Direct interval scans also handle zero decay without inf-inf or 0/0." and the masked add scan under `gl.exp` :370-372
- S36 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/448925c31bc8ea55.json#kernel.py:1003 `snapshot_outputs = 3 <= batch <= 4 and m <= 4096`; `hierarchical` :1005
- S37 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:81 Table 28: F32 16x16x4 32 cycles; BF16 16x16x32 16 cycles at :101
- S38 triton-lang/triton@c01b677:lib/Analysis/Utility.cpp:1096 a gather is warp-local only if no warp or block basis touches the gather axis; :1311-1321 `cvtNeedsWarpShuffle`: fewer than two mixed transpositions; :526-531 `ScanLoweringHelper::isSupported` accepts only a blocked encoding; otherwise lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp:474 emits "TODO: unsupported scan layout"
- S39 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/slice.py:12 `gl.amd.slice`: register-only slice, no cross-thread movement (:12-23)
- S40 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py:12 the cdna4 module re-exports `buffer_load`, `buffer_store` and `mfma` (:12-18)
- S42 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245 `assert_trivial` raises "layout conversion from … is not trivial" with both linear layouts unless the conversion is trivial (:245-248)
- S43 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertLayoutOpToLLVM.cpp:44 permlane swap only for exactly one (register, lane 4 or 5) transposition or a 3-cycle through lanes 4 and 5 (:44-87)
- S45 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MaskedOpsToLLVM.cpp:81 a mask that is not constant true becomes a conditional branch around the load (:81-104)
- S46 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#kernel.py:281 `_inverse_unit_lower_16`: 4 × 4 diagonal blocks by `gl.gather` pivots, then three fp32 MFMAs (:280-300); gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/7a69337f81190952.json#kernel.py:338-342 15 rank-1 eliminations; gdn_prefill_group_fp8_quant_max_seq_len8192_n_v_heads16_key_dim128_value_dim128_conv_dim3072/b8c54bde37ee77c7.json#kernel.py:224-232 `_inverse16`, three squarings, fed the negated lower part :325-335
- S47 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_max_seq_len5120_n_v_heads8_key_dim128_value_dim128_conv_dim2048/98af1fd2ff621082.json#kernel.py:115 "Only this CTA reads history for this sequence/channel group. Finish its readers before updating any cache element." then `gl.barrier()` and the stores (:115-131); the same in ed2a8611dd433d8a
- S48 aklr/plugins/sglang/models/qwen3_next/contract.py:9 projections grouped by key head (:9-13, :88-97), conv channels every Q, then K, then V :71, :100; `conv_bias` in both contracts :108; outputs `normalized`, `values`, `scales` :125-153; aklr/plugins/sglang/models/qwen3_next/reference.py:41-49 `_ungroup`; aklr/plugins/sglang/models/qwen3_5/contract.py:114-115 contiguous projections; `conv_bias` in prefill :151, absent from the decode inputs :220-238; outputs :168-187, :263, :275
- S49 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/35b9cfd56833844f.json#kernel.py:374 "Only a register-owned column prefix stays live across convolution." (:374-379), tail and merge :417-422, off at its 16-row Point (`STATE_PREFIX` :333); update in accumulate form :428-431; three quadrant stores before the epilogue conversion, the last after it :433-443 (`PHASED_STORE` at 16 rows :338); `_keep_alive()` after the epilogue, "Separate convolution storage from the epilogue's conversion scratch to avoid a shared-memory reuse barrier." :452-454; 0c0b7dfe26861b02.json#kernel.py:258 the same prefix load (:258-262, :316-321) with `state_prefix = 64` :478
- S50 aklr/core/contract/semantics/_distributions.py:276 `BoundedCuSeqlensDistribution.sample`: one sequence of exactly `max_seqlen`, the rest split at random (:276-292)
- S51 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_max_seq_len5120_n_v_heads8_key_dim128_value_dim128_conv_dim2048/98af1fd2ff621082.json#kernel.py:107 "Initialize padding on every invocation so all downstream loads are safe." and the zeros stored (:107-110); qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/010eab6660c4e415.json#kernel.py:10 "Every consumed scratch element is initialized within the invocation, including ragged tails." (:10-11)
