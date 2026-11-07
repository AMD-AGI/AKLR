---
title: "GatedDeltaNet decode and chunked prefill in Gluon on gfx950 (Qwen3.5, Qwen3-Next)"
kind: topic
ops: [gdn]
dtypes: [bf16, fp8, fp32]
regimes: [decode, prefill]
gens: [gfx950]
triton: 3.8.0
---

# GatedDeltaNet kernels in Gluon on gfx950

AKLR has 44 Gluon kernel files for the GatedDeltaNet (linear attention) layers of Qwen3.5 and Qwen3-Next: 18 decode
files update a 128 × 128 fp32 state per value head on the VALU, 26 chunked-prefill files run the recurrence on the
matrix cores with sequential recurrence stages. All fuse the gated RMSNorm, all but one decode file the fp8 group quant.
The schedules below are shape-specific source templates, not universal crossover or performance rules.
Original kernels, contracts and compiler references are listed under Evidence.

## Route here when

- You write or tune a GDN decode or prefill kernel for an AKLR schema, or another recurrence that keeps a matrix state
  per sequence slot (`run`, scratch and validation: [authoring.md](authoring.md)).
- You choose, per decode row count, CTAs that split a head's state or own whole heads, and where the second launch goes.
- You fuse an fp8 group quant into a kernel that also makes BF16 or fp32 MFMA results (FP16_OVFL; [numerics.md](numerics.md) §2).
- You pick a prefill chunk size and schedule for a (tokens, sequences) Point, need an fp32 MFMA accumulator as the next
  MFMA's operand without a trip through LDS, or must handle ragged sequences without host reads.

## 1. What the op computes, and the kernels AKLR has

Per decode row (one token with its own state slot; slot −1 is padding: no state change, zero core output) (S1):

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

```text
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

| model, contract | files | Points | launches per call | writes |
|---|---|---|---|---|
| Qwen3.5 `gdn_decode_group_fp8_quant@1.0` | 8 | 1, 2, 4, 8, 16, 32, 128, 256 rows, one file each | 2 | `values`, `scales` |
| Qwen3.5 `gdn_decode@1.0` (BF16 output) | 1 | 64 rows | 2 | `normalized` |
| Qwen3.5 `gdn_prefill_group_fp8_quant@1.0` | 15 | 21: 512–16384 tokens in 1–8 sequences | 6–7 | `normalized`, `values`, `scales` |
| Qwen3-Next `qwen3_next_gdn_decode_group_fp8_quant@1.0` | 9 | 1–256 rows (9 Points; one template file lists all nine) | 2 up to 8 rows, 1 from 16 | `normalized`, `values`, `scales` |
| Qwen3-Next `qwen3_next_gdn_prefill_group_fp8_quant@1.0` | 11 | 13: 1024–16384 tokens in 1–16 sequences | 5–8 | `normalized`, `values`, `scales` |

- All five also update `conv_state` and `delta_state` in place (S2, S48). Write every output row, including padded rows; every in-place state change must follow the contract.
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
- Decode uses no MFMA : per (row, head) two 128 × 128 matrix-vector products and a rank-1 update on a
  64 KiB state no other row reads, 0.875 flop per byte (estimated). All 26 prefill files use both the BF16 and the
  fp32 MFMA. No file uses atomics. Bitwise determinism is required (S2); no file uses arrival-order atomic sums.
- Stored performance is version-bound. Re-read a file's current workloads and validate its full outputs
  and in-place states before using a template; stored CI metadata does not certify a new specialization.

- Files are named by their uuid's leading hex digits: `<model package>/impls/triton/<scenario>/<uuid>.json`, sources
  inside ([authoring.md](authoring.md) §3); the Qwen3.5 prefill files sit in nine scenario directories by
  `max_seq_len`, and `ls aklr/plugins/sglang/models/qwen3_*/impls/triton/*/<digits>*.json` finds one.

## 2. Decode: split each head's state at small row counts, own whole heads above

With few rows, split each head's 128 value rows into tiles to expose more CTAs. With more rows, whole-head
ownership permits fused normalization/quantization and avoids the split hand-off. The sources switch where
rows × value heads reaches 128 (Qwen3.5: 8 rows; Qwen3-Next: 16), but this is not a universal crossover.
Qwen3.5 uses 8 × rows value rows per CTA at 1/2/4 rows, giving 256 recurrence CTAs. Compare tile sizes at
the actual Point: a 256-CTA target alone did not isolate the measured trade.

| rows | file | recurrence CTA | CTAs | VGPRs |
|---|---|---|---|---|
| 1 | `2c2a09e2` | 2 waves, 8 value rows, `waves_per_eu=1` | 256 | 35 |
| 2 | `29913388` | 2 waves, 16 value rows, `waves_per_eu=1` | 256 | 61 |
| 4 | `409cb22b` | 4 waves, 32 value rows | 256 | 61 |
| 8 | `4f257a3e` | 4 waves, whole head | 128 | 142 |
| 16 | `a4f80b6a` | 4 waves, whole head | 256 | 146 |
| 32 | `20f63094` | 4 waves, whole head | 512 | 126 |
| 64, BF16 op | `cd00c50e` | 16 waves, whole head, `waves_per_eu=6` | 1024 | 47 |
| 128 | `fc261961` | 16 waves, whole head, `waves_per_eu=8` | 2048 | 55 |
| 256 | `c2c3a42a` | as at 128 rows | 4096 | 54 |

CTA shapes from the launches (S5–S10); VGPRs of the recurrence kernel, none spilling.

- Small decode is not necessarily state-bandwidth bound. The measured 1–4-row calls stayed near 9 µs
  while state bytes grew fourfold; this suggests latency sensitivity but does not isolate its source.
  Streaming bytes-in-flight estimates do not determine the GDN grid.

- From 64 rows the recurrence is a 16-wave CTA with 16 state values per thread under `waves_per_eu` 6 (64 rows) or 8,
  which Triton passes as `amdgpu-waves-per-eu` min = max = N, a hard budget a probe spills to meet (S11). At 8 it
  compiles to 55 (128 rows) and 54 (256 rows) VGPRs, no spill, "a fixed, spill-free 16-wave layout" (S10);
  the 64-VGPR cap (512 / 8) and two CTAs per CU are derived. At 4–32 rows it is unset. Forcing it can cost:
  `waves_per_eu=4` at 16 and 64 rows in the Qwen3-Next template pinned the tested kernels at 128 VGPRs,
  spilled registers and made the full call about 60% slower. Inspect the compiled specialization, not an
  occupancy comment or target budget alone.

- Qwen3-Next's per-size files use the same two regimes (S12): "split" at 1–8 rows (one- or two-wave producers own 4–8
  value rows: 256 programs at 1 and 2 rows, the template 512 at 2; 512 and 1024 at 4 and 8), "fused" from 16 rows (one
  4- or 8-wave workgroup per (row, key head) owns both value heads' states and the group's 512 history channels: 64
  workgroups at 16 rows, 1024 at 256). The CI run main recorded in d908d01, best per-size file and the template
| rows | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| schedule | split | split | split | split | fused | fused | fused | fused | fused |
| per-size file | `b31b` 2.443 | `bba1` 2.538 | `3719` 2.370 | `3719` 2.506 | `35b9` 3.222 | `38a7` 3.350 | `0c0b` 2.462 | `0678` 1.782 | `0678` 1.425 |
| template `f440` | 2.390 | 2.450 | 2.370 | 2.479 | 3.101 | 3.091 | 2.401 | 1.755 | 1.457 |

`c784` is a third whole-head design with state loaded first and barrier-guarded convolution.
Use these files as candidates, not a nearest-file dispatch rule: the log-nearest source lost to another
candidate at 24 and 48 rows. The template uses exact-match configuration rows (S13).

Before copying a file to another row count, check what it assumes:

- `fc26` (128 rows), `c2c3` (256) and `cd00` (64) map programs as `token = (pid // (VH * 8)) * 8 + pid % 8` and do not
  check that the row count is a multiple of 8: at 12 rows, 32 of the 192 (row, head) pairs would not be computed and
  32 programs would read rows past the input (derived). `20f6` (32 rows) falls back to groups of 1 (S10, S9).
- `a4f8` (16 rows) passes the pool size as `SLOTS: gl.constexpr` for a `gl.assume` (S9). `n_slots` is an `any_size` axis
  (S2), so each new pool size compiles again, inside a capture too (four pool sizes, four binaries; with a runtime
  `SLOTS`, one): keep it runtime ([authoring.md](authoring.md) §5).

## 3. Decode: put the launch boundary where ownership ends

Two facts place it (S5, S7): the gated RMSNorm and the fp8 group each span one whole value head, so an independent tile CTA cannot finish the epilogue without obtaining whole-head statistics (a same-launch coordination or recomputation variant was not tested); and the conv history may be overwritten only after every
reader of the old history has read it, while with recomputed inputs (§4) many CTAs read the same channels.

| ownership | where the history shift and the epilogue go | files |
|---|---|---|
| a head split over CTAs; its key head's history read by many CTAs | launch 1: conv + gates + recurrence on a tile, BF16 core to scratch; launch 2: norm + quant per head, and the history shift by programs or waves that each own whole channels | Qwen3.5 1–4 rows, Qwen3-Next 1–8 rows |
| one owner per channel in launch 1 | launch 1 shifts each channel's history right after reading it ("All history reads are complete; retire writes before activation arithmetic", S7); launch 2 owns whole heads and fuses norm + quant | Qwen3.5 8–256 rows |
| one workgroup owns a (row, key head) | one launch: the shift happens inside the conv where each channel has one owner (16, 64, 128, 256 rows), after a `gl.barrier` inside the conv (`c784`), or after the CTA-wide core exchange where a channel has two owners (32 rows, `38a7`) (S14, S15) | Qwen3-Next 16–256 rows |

- In a one-launch design, prove single ownership in the layout, `gl.static_assert(VEC * 64 * NW <= conv_width)` for a
  `BlockedLayout([VEC], [64], [NW], [0])` conv layout, so no lane or wave is replicated (S14 has VEC = 1; at 32 rows the
  template's VEC 2 × 64 × NW 8 = 1024 slots for 512 channels fails it, S13), or put `gl.barrier` between the history
  loads and stores (S15). On gfx950 `gl.barrier` lowers to a workgroup release fence, `s_barrier` and an acquire
  fence, so it also orders the CTA's global accesses (S16); the barriers Triton's Membar pass inserts between
  conflicting LDS accesses cover LDS only (S16). A replicated history layout without a reader/writer
  boundary can overwrite another wave's old history; this occurred in the 32-row `f440` specialization.
  A race need not fail every input, and correctness with one wave is not proof for a cross-wave layout.

- Price a boundary with the deployment's ruler. AKLR sums profiler kernel durations ; bare graph
  replay includes different costs. Do not remove a required reader boundary just to save a launch.

## 4. Decode: recompute shared inputs, keep reductions inside a wave

- **Recompute shared inputs as a trade.** Split tile CTAs convolve and normalize their key head's whole
  Q/K (256 channels), while history remains read-only (S6). This avoids a staging launch but duplicates work.
  Synthetic replay results changed sign across measurements, so no universal replay gain is established.

- Exchange q and k between the CTA's waves through 1–2 KiB of LDS (q and k, 2 × 128 fp32, at 1 and 2 rows; q, k, v and β
  at 4 rows; S5, S6, S8) and read them back in the state layout; the 2-row file's late q read "to shorten its register
  lifetime" compiles as an early one. Summing a `[1, BK]` slice over its unit axis yields the vector in
  `SliceLayout(0, state_layout)` with no data movement (compiled with Triton 3.8.0 for gfx950: four `ds_read_b128` per
  vector, no lane exchange; `qk_shared.index(1).load(gl.SliceLayout(0, state_layout))` is rejected in this matched Triton 3.8.0/gfx950 build at the descriptor `index` step: "rank must be equal to or one less than the shape size. Got 2 and 1"):
```python
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
- **Schedule for latency and liveness.** Try the large state load before scalar gates, load epilogue
  inputs early and defer their arithmetic (S9, S7). Moving Q normalization also changes liveness, so an
  observed timing difference is not a measurement of overlap alone. In the template, early state loading
  adds substantial registers at 128 rows but not at every Point.
  The 16/64-row `_fused_heads` branch ignores cache/gate-position configuration fields and hard-codes
  `.cg`/`.cs` (S13): change the actual branch, not an unused table entry.

- **Check comments against compiled liveness.** Accumulate versus recompute update forms alter both
  rounding and register lifetime. The template's “64 VGPR” comment does not match its tested 16/64-row
  binaries; moving gates before convolution also did not halve occupancy in the tested 128-row build.
  `0c0b` loads a state prefix before conv and the rest later; `35b9` retires three quadrants before the
  epilogue conversion (S49). Inspect resources and time each specialization.

- **Take scalar math off the recurrence path.** `4f25` computes decay and beta in an extra role of the
  convolution grid (S7); moving them back into each update program was slower at its 8-row Point, but
  dynamic overlap was not measured. `2c2a` evaluates value SiLU, decay and beta in neighboring lanes
  of a `[BV, 4]` tensor (S5); fewer exponential instructions did not establish a speed advantage.

- Program order: five whole-head files (16–256 rows) number programs in token groups, `token = (pid // (VH * G)) * G +
  pid % G`, G a multiple of 8, so all heads of a row share `pid % 8` under the modulo-8 placement model; its
  comment says "Interleave heads over token groups for state-pool traffic"; `a4f8` also XOR-permutes each row's heads (S9;
  [tuning.md](tuning.md) §12). In `fc26` it beat `token = pid // VH` by only 0.6–1.1%.
- Check each variant's VGPR count ([debugging.md](debugging.md)) and A/B it ([tuning.md](tuning.md) §2, §10): at equal
  VGPRs (160) the template's late state load at 32 rows took 1.23× as long.

## 5. Decode: the state's loads and stores

The Qwen3.5 files use five combinations (S5–S10); state stores, and loads up to 32 rows, are 16 bytes per lane:

| rows | state load | state store |
|---|---|---|
| 1, 32 | `gl.load`, no hint | `gl.store`, no hint |
| 2 | `buffer_load`, `.cs` | `buffer_store`, `.cs` |
| 4, 8 | `buffer_load`, `.cg` | `buffer_store`, `.cs` |
| 16 | `gl.load`, `.cg` | `gl.store`, `.cs` |
| 64–256 | `gl.load`, no hint | `buffer_store`, `.wt` |

At 2 rows, buffer versus pointer access changed address formation but did not establish a reproducible
speed advantage. Prefer the form whose index bounds and scalar-base rebasing are easiest to prove.
At 128/256 rows the tested `.wt` buffer store was faster than unhinted or `.cs` pointer stores, but that is
shape-specific. Load and store hints have different effects: `.cs` sets `sc0 nt` on a buffer operation and
only `nt` on pointer loads/stores (S18). Inspect ISA and compare the actual state stream
(`gluon/memory.md`, `hardware/gfx950/memory_and_sync.md`).

- Buffer addressing: the descriptor covers 2^31 − 2 bytes and element offsets are scaled to bytes in 32 bits (S19), so
  files that fold the slot into the offset test the pool size in bytes on the host and rebase a 64-bit scalar head
  pointer for larger pools (S8); the contracts cap the pool at 2047 slots (S2), 2047 MiB of Qwen3.5 state, so AKLR
  never takes the wide path.
- Padding rows: put one uniform `if slot >= 0:` around every access to either pool, the history shift included, and
  write a zero core otherwise (S6). The norm and quant still run on that zero core and write the row's outputs, as
  `4f25` does (S7): zero-valued codes, roughly half −0 on the sampled random inputs, and scale float32(1e-10) × float32(1/448).

## 6. Decode: the fused fp8 group quant and the FP16_OVFL mode bit

One helper ends every head in all eight Qwen3.5 fp8 decode files (the same function text in each): it rounds the head
to BF16, takes the amax of its non-NaN values floored at 1e-10, scales by one reciprocal of `amax · (1/448)`, clamps to
±448 with an inline `v_med3_f32` (`gl.clamp` lowers to the same, S20) and converts (S6). The recipe and its division
forms: [numerics.md](numerics.md) §2.

- Keep the head's 128 values in one wave (`BlockedLayout([4], [64], [2], [0])` gives each of two waves a full copy), so
  amax and the RMS are wave reductions with no LDS exchange (S5).
- Quantize the BF16-rounded output, as the references do (S1); compute the scale from it, not from the fp32 precursor.
- Round before a cross-wave `convert_layout`: "Preserve the BF16 boundary and halve the cross-wave LDS transfer" (S7).

**FP16_OVFL.** In Triton 3.8.0's gfx950 lowering, an fp8 destination marks the containing function and
sets MODE bit 23 at that function's entry (S21). This can precede earlier BF16 casts in source order;
with noinline helpers, the relevant entry can be the callee rather than the kernel. Implicit fp8 stores
and runtime branches can also trigger it. Inspect the compiled cast path rather than assume every fp8
instruction, upcast, bitcast or inline-asm conversion has the same side effect.

The 2-row `_normalize_update_history` sets it before its first `v_cvt_pk_bf16_f32`; BF16 finite overflows then saturated to ±0x7F7F. The Qwen3.5 sources leave it set; finite workload inputs do not establish IEEE overflow behavior. `b31b`
and the Qwen3-Next template at 1 row map |x| ≥ 3.396e38 to ±Inf before the BF16 convert and keep Triton's fp8 store
(S23); the template at 2, 4, 8 and 32 rows clamps and converts with an inline `v_cvt_scalef32_pk_fp8_f32`, so no Triton
cast sets the bit, and switches it in value-producing asm, clear at entry, set after the last BF16 rounding; `38a7` sets
it in the convert's own asm (S24); the template's BF16 results saturate at 16, 64, 128 and 256 rows.
Rule: [numerics.md](numerics.md) §2; asm and ordering risk: [gluon/wave_ops.md](gluon/wave_ops.md) §3.2, §7. Keep the
two choices separate knobs: "collapsing the two into one flag silently changes results" (S25).

## 7. Prefill: the pipeline and its launch boundaries

Every prefill file runs, in launch order (`010e`, `70f4`, `11f6` add a conv-history launch, `11f6` a chunk-offset one):

1. `_initialize_fresh` zeroes state slots marked fresh by a device flag (S26). Nine Qwen3.5 files also
   repack contiguous projections into Qwen3-Next's grouped order, while MAP/RSUM read contiguous inputs
   directly (S32). Avoiding the repack is a candidate, not an established timing gain.

2. Prepare, parallel over token tiles × channel groups: conv + SiLU + q/k L2 norm to BF16, gates.
3. Chunk factors, parallel over chunk × head: Gram products, decay products, the unit-triangular solve, WY factors.
4. Long sequences only (summary designs, §8): each run of chunks summarized as an affine map `S ↦ S·A + B`, in parallel.
5. Serial pass over chunks or summaries, parallel over sequence × head × value strip; per-sequence designs add outputs.
6. Parallel epilogue: outputs rebuilt from incoming states where step 5 made none, then gated RMSNorm and fp8 quant.

- Every boundary changes the parallel decomposition; with no atomics (§1), stream order is the only synchronization.
- **Commit conv history only after old-history readers finish.** Some sources use the final launch;
  SEQ16 does it in its per-sequence one-wave convolution role (S27); three Qwen3-Next sources use a
  separate history launch. Where readers span waves, prove exclusive ownership or insert a required
  barrier. Early commit can change output even if final conv and delta states appear correct.
  One-wave correctness without a barrier does not establish a cross-wave barrier requirement or safety.

- A fresh sequence starts from a zero history as well as a zero state, but `_initialize_fresh` zeroes only
  `delta_state`: mask every read of the old history with the flag. A sequence of one or two tokens keeps old entries:
  its new history is the last three of [old history, or zeros when fresh; its tokens] (S27, S1). The 2049/2 Point
  always holds a one-token sequence: AKLR draws one of exactly `max_seq_len` tokens and splits the rest (S50).
- Ragged batches without host reads: grids come from shapes only; a chunk program finds its sequence by
  binary search over `cu_seqlens`, the slot formula `begin // BT + sequence` "reserves enough space for every ragged
  tail", and programs past the device-written chunk count do nothing (S28).
- Scratch is `torch.empty` per call (`aklr/ops/README.md` § The kernel section and ABI) and large: 173–294 MB of fp32 at
  8–16K tokens in the summary designs (MAP at 10000 tokens: 246 MB of factors, 36 MB of maps); buffers
  with disjoint live ranges are aliased. Write every element a later launch reads in the same call, padding rows and
  ragged tails included: a NaN left in old memory survives a zero β or a decay of 1 (0 × NaN is NaN). `98af` zeroes each
  chunk record's padding tokens; `010e` states the rule for its whole pipeline (S51; [authoring.md](authoring.md) §6).
- The Qwen3.5 prefill plugin falls back during stream capture (S29). A standalone kernel graph is not
  proof that the serving route uses it under capture. Eager launch/allocation costs and graph kernel sums
  are distinct timing rulers (`authoring.md` §7).

## 8. Prefill: chunk size and schedule per Point

Measure candidate designs at the specific (tokens, sequences, `max_seq_len`) Point; the committed boundaries below are not independently confirmed selection rules. Labels: D, the ≤ 1024-token source
(`1fd6`, `61ed`, `1d55`); CKPT, the checkpoint scan (`19b2`, `3828`); SEQ32L, SEQ32G and SEQ16, one program per
(sequence, head, value strip) over 32-token sequence-local, 32-token global (plus a scalar per-token recurrence at
ragged edges) and 16-token global chunks (masked so each segment's coefficients stay in its sequence); MAP and RSUM,
summaries of runs of chunks. Choose by actual sequence lengths, serial-pass parallelism and scratch cost.
The tables are source maps, not certified best-design boundaries.

- **Short-scan candidates.** D and CKPT store incoming chunk states and reconstruct output in parallel
  (S33, S34). On five tested short Points from 512 to 1025 tokens, both beat MAP/RSUM in eager
  single-call device-event timing, and CKPT beat D. Those equal-length or single-sequence inputs do not
  establish a universal ≤2048 rule, arbitrary-ragged performance or graph/service benefit.

- **More tokens with `max_seq_len` 2048 (2–8 sequences)**: one program per (sequence, head, value strip) walks all
  chunks and produces the outputs; BV makes sequences × 16 heads × 128/BV = 256, one program per CU: SEQ32L at 2
  sequences (BV 16), SEQ32G at 3–4 (BV 32), SEQ16 at 5–8 (BV 64) (S27, S30, S31). The Points that fill 256 programs
  measure 1.374–1.475; the two with fewer sequences than their file assumes (160 and 192 programs) 0.997 and 1.029,
  below the default `AKLR_MIN_SPEEDUP` (`aklr/ops/README.md` § Environment variables), so sglang serves them. Their
  longest sequence is 2048 tokens, as at the full Points, and the time does not follow the work: about half the work
  took 0.97× (`2ece`) and 0.85–0.86× (`c53b`) the full Point's time, which does not separate the
  serial chain from the program count; the other strip was slower (table).
- **Long sequences, `max_seq_len` 4096 (Qwen3-Next) or 5000 (Qwen3.5) and above, 1–4 sequences**: a per-sequence walk
  is a long serial chain, and at one sequence 16 × 128/BV programs are too few. Build each run of chunks' affine map in
  parallel, from an identity seed (MAP: the rows `[I; 0]` run through the recurrence) or composed right to left from the
  run's last chunk (RSUM); chain the maps in a short serial pass (4–15 dependent 128 × 128 products per row strip,
  derived from the chunk counts); then replay all chunks in parallel from their incoming states (S32). At one sequence
  this beats a per-sequence walk, SEQ16 (table); 2–4 sequences and Qwen3-Next were not timed.
- **Keep the serial chain overlapped** (S33, S27, S30); undoing each knob cost, at one Point: in CKPT (`3828`)
  at 2048/1, unroll 1 for `loop_unroll_factor=8` 6.3–7.0%, no `amdgpu-sched-strategy` `iterative-ilp` 11.8–13.3%
  ([tuning.md](tuning.md) §10), one 128-deep MFMA chain for two independent 64-deep ones 3.1–4.6%; in
  `24ed` at 16384/8, the next chunk's operands loaded after the current chunk's math 2.0–2.4%, `.cg` on the value tile
  dropped −0.2 to +0.3%  ([gluon/memory.md](gluon/memory.md) §12). `2ece` stores each half table once
  final, "to shorten register lifetimes": 190 VGPRs at M = 8192, 194 with all stores at the end, untimed.

Qwen3.5 source configurations; serial-pass programs derived from the grids:

| tokens / sequences | max_seq_len | file | design | chunk (tokens) | serial programs |
|---|---|---|---|---|---|
| 512 / 1 | 512 | `1d5568a7` | D | 64 | 128 |
| 512 / 2 | 256 | `61ed32f6` | D | 64 | 256 |
| 1024 / 1 | 1024 | `1fd6adb2` | D | 64 | 256 |
| 1024 / 2 | 512 | `1d5568a7` | D | 64 | 512 |
| 1025 / 1 | 1025 | `19b20285` | CKPT | 64 | 128 |
| 2048 / 1 | 2048 | `382853eb` | CKPT | 64 | 128 |
| 2049 / 2 | 2048 | `0a07f9f9` | SEQ32L | 32 | 256 |
| 4096 / 2 | 2048 | `0a07f9f9` | SEQ32L | 32 | 256 |
| 4097 / 3 | 2048 | `2ecece96` | SEQ32G | 32 | 192 |
| 8192 / 4 | 2048 | `2ecece96` | SEQ32G | 32 | 256 |
| 8193 / 5 | 2048 | `c53b6d87` | SEQ16 | 16 | 160 |
| 15360 / 8 | 2048 | `c53b6d87` | SEQ16 | 16 | 256 |
| 15361 / 8 | 2048 | `24ede969` | SEQ16 | 16 | 256 |
| 16384 / 8 | 2048 | `24ede969` | SEQ16 | 16 | 256 |
| 5000 / 1 | 5000 | `12b560c8` | RSUM | 32-token tiles, 512-token chunks | 128 |
| 15000 / 3 | 5000 | `12b560c8` | RSUM | 32 / 1024 | 384 |
| 8192 / 1 | 8192 | `5ead0483` | MAP | 16-token blocks, 512-token chunks | 128 |
| 10000 / 1 | 10000 | `7a69337f` | MAP | 16 / 640 | 128 |
| 16384 / 1 | 16384 | `47092631` | RSUM | 32 / 1024 | 128 |
| 16384 / 2 | 8192 | `b8c54bde` | RSUM | 32 / 1024 | 256 |
| 16384 / 2 | 10000 | `3b99d196` | RSUM | 32 / 1024 | 256 |

`5ead`/`7a69`, `3b99`/`4709`/`b8c5` and `19b2`/`3828` share source text, yet different Points can have
very different costs. Re-measure at every Point. Prefill `run` takes `max_seq_len` but does not read it (S32);
that attribute selects the scenario, not the runtime loop bound.

Qwen3-Next source pairs differ in constexpr versus runtime token/sequence counts. Every sequence in the
1024 scenario is 1024 tokens; `162f`, `c3a7`, `4489` share one source, and `a99f`, `f137`, `d8f9` its runtime twin.

| tokens / sequences | max_seq_len | files | schedule | chunk |
|---|---|---|---|---|
| 1024 / 1; 2048 / 2 | 1024 | `11f62a16` | serial scan with per-chunk state snapshots, parallel output | 64 |
| 3072 / 3; 4096 / 4 | 1024 | `448925c3` / `d8f9e1b5` | snapshot scan, parallel output | 32 |
| 8192 / 8; 12288 / 12; 16384 / 16 | 1024 | as above | direct recurrence | 32 |
| 13312 / 13; 15360 / 15 | 1024 | `70f4be35` / `010eab66` | compact factors, operands rebuilt in the serial loop | 32 |
| 4096 / 1 | 4096 | `162f43f1` / `a99f22dc` | block summaries (identity and zero seeds) | 32 |
| 5120 / 1; 10240 / 2 | 5120 | `c3a70b84` / `f1374046` | block summaries | 32 |
| 16384 / 4 | 5120 | `ed2a8611` / `98af1fd2` | segment summaries composed in reverse | 32 |

The `11f6` file takes 64-token chunks at 1–2 sequences (S35); the `4489` source picks its schedule from the Point: the
snapshot scan at 3–4 sequences up to 4096 tokens, block summaries at 1–2 sequences here, the direct recurrence at 8 and
more (S36). The twins' annotations alone do not establish binary reuse: all other specialization-key inputs must
remain unchanged. Compare both before choosing; a single stored timing difference is not equivalence.

Before you reuse a prefill file at another Point, read which branch its Points take. Above 4096 tokens `0a07`
static-asserts that `normalized` has the conv buffer's 3072-element row stride, which AKLR's `(n_tokens, 16, 128)`
output lacks: compiled with Triton 3.8.0 for gfx950 at AKLR's stride, that path fails its `static_assert` (S31). `11f6`
finds a token's sequence in seven unrolled halvings, right for at most 128 sequences where the contract allows 2047
(S35): it fails at 129 sequences, where eight halvings pass. `2ece`'s path up to 4096 tokens and CKPT's
up to 1024 do not run at their Points: no CI run checked or timed them (S30, S33).

## 9. Prefill: the intra-chunk math on the matrix cores

- Put the Gram products K Kᵀ and Q Kᵀ on the BF16 MFMA (`instr_shape=[16, 16, 32]`, `k_width` 8: eight consecutive K per
  lane, [gluon/mfma.md](gluon/mfma.md) §5): the prefill references round q and k to BF16 (S1). For finite BF16 operands whose exact product is within the normal FP32 range, the product is exactly representable in FP32. The FP32 accumulation still rounds.
- Share the Gram products among the value heads of a key head (four in Qwen3.5, two in Qwen3-Next): K Kᵀ and Q Kᵀ depend
  only on the key head, so `0a07` computes them once per (chunk, key head) ("share four-head Gram matrices"), `24ed` and
  `2ece` once per head pair; only β and the decays differ per value head (S31, S27, S30). In `0a07` at 4096/2 sharing
  reduced duplicated Gram work and the tested register count; the whole-call gain was only about 1–2%.
- State algebra, solves and tables run on the fp32 MFMA (`v_mfma_f32_16x16x4_f32`, some `32x32x2`) : 32
  cycles for 1024 MACs, against 16 cycles for 8192 MACs on the BF16 16x16x32 (S37), 16× fewer nominal MACs per cycle. Keep the serial pass to few fp32 products per chunk (CKPT: two per 64 tokens);
  moving output products to BF16 (§10) is untimed; [gluon/mfma.md](gluon/mfma.md) covers the fp32 input precision.
- The unit-lower-triangular inverse `(I + A)⁻¹` of a 16–64-token chunk: solve small diagonal blocks (4 × 4, 8 × 8 or
  16 × 16) by forward substitution with `gl.gather` pivots and join them with fp32 MFMAs; or, on a 16 × 16 block,
  multiply out `(I+N)(I+N²)(I+N⁴)(I+N⁸)`, which is `(I − N)⁻¹` for a strictly lower N (N¹⁶ = 0): pass N = −A, as `b8c5`
  negates the Gram term before `_inverse16` and `11f6` documents ("Inverse of I-P for a strictly lower triangular
  16-by-16 P"), and join 16 × 16 inverses with fp32 MFMAs for longer chunks (S46, S35). Keep each small solve in one
  wave: `gl.gather` lowers to lane shuffles only when the gathered axis stays in a warp, else through LDS (S38).
- The files skip the zero halves of triangular operands: K blocks taken with `gl.amd.slice`, a register-only view (S39),
  only nonzero blocks multiplied, only tiles on or below the diagonal stored and loaded (S31, S33). In `0a07` at 4096/2
  that cut 184 to 136 MFMAs, not the call time (B/A 0.966–1.008); the stores were not varied.
- Store each factor in the order its consumer's MFMA operand holds it, so the consumer loads 16-byte vectors and
  converts with `assert_trivial=True` (S30), which fails the compile ("layout conversion from … is not trivial",
  printing both linear layouts) when the conversion moves data across lanes or waves (S42).

## 10. Prefill: compensated products, and decay products over intervals

**Compensated products.** Where FP32 × BF16 is rounded to BF16 anyway, split the FP32 operand into BF16
high/residual terms and accumulate their BF16 MFMA products in FP32, smallest first. This reduces split
error but is not an arbitrary-input correct-rounding guarantee; underflow, overflow and cancellation need
coverage. Five Qwen3.5 sources restrict compensation to output reconstruction; solves, recurrence and
mutable state remain FP32 (S34). CKPT uses two RNE terms in its active branch and three RTZ terms elsewhere.
The D sources keep BF16 high/low pairs for both FP32-derived operands and all four cross products (S34).
No Qwen3-Next prefill source uses this recipe (`numerics.md` §5).

**Scan each interval, not two prefixes.** Form Γ(i,j) over (j,i] directly. Dividing tiny cumulative
prefixes can yield 0/0, NaN or zero even when the relative interval decay is representable.

MAP runs `gl.associative_scan` with a multiply combiner down the columns of
`gl.where(i[:, None] > j[None, :], sg[:, None], 1.0)` (S32); `11f6` scans the log-decays of each interval with an add
combiner and exponentiates once: "Direct interval scans also handle zero decay without inf-inf or 0/0" (S35). Other
files split the chunk into two 16-token halves with a separable cross block (S30), or keep dyadic levels of prefix and
suffix products and read Γ(i, j) = suffix_L[j] · prefix_L[i] at L = `31 − clz(i ^ j)`, the highest bit in which i and j
differ (S33). `gl.associative_scan` lowers only a `BlockedLayout` operand (S38): on a `SliceLayout` vector, such as the
per-token gates of a 2-D tile, the compile fails with "TODO: unsupported scan layout" (compiled with Triton 3.8.0 for
gfx950; [debugging.md](debugging.md) §4): convert to a blocked layout, scan, convert back, as MAP does (S32). Padding
tokens get decay 1 and β 0 (S32), but neutral padding also requires finite arithmetic intermediates:
0 × NaN is NaN, including a nonfinite prediction or residual multiplied by beta zero. Which
exponential keeps subnormal decays (`gl.exp`) and which flushes them (`gl.exp2`): [numerics.md](numerics.md).

## 11. Prefill: relabel layouts instead of converting them

Each chunk step feeds an fp32 MFMA accumulator (the state, or an inverse block) into the next MFMA as the A operand.
A direct `k_width=1` conversion from a `[16,16,4]` accumulator can require two register/lane
transpositions through LDS. A transposed `[32,32,2]` mapping can remain within a wave with one
transposition. Neither is a universal rule: shape, transposition and warp axes determine the conversion.
Inspect the actual layout and `assert_trivial` result (Triton 3.8.0, gfx950).

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

In the one-wave 16×16 example, kw4/relabel uses fewer instructions than direct kw1, but B still needs
LDS, so the whole product is not LDS-free. Fewer movement instructions are not a speed guarantee:
a measured four-wave 64×16 chain was about 31% slower with kw4 than direct kw1. GDN end-to-end
benefit requires a full-call comparison.

## 12. Prefill: LDS staging, conv windows and masks

- What the files stage through LDS, in shared layouts they choose (alternatives not timed):
  - V: value-major loads widened to fp32 into `SwizzledSharedLayout(4, 1, 8, [0, 1])`, read in the MFMA layout (S27);
  - the state, when the warps split the accumulator's columns: every wave then needs whole state rows as its A operand
    (compiled with Triton 3.8.0 for gfx950, a 16 × 128 state strip in a transposed `[1, 4]` layout, as `0a07` holds it,
    sends even the `k_width = 4` conversion through LDS: two `ds_write_b128`, eight `ds_read_b128`, three
    `s_barrier`). `0a07` writes a `SharedLinearLayout` that keeps 4-float vectors and XOR-swizzles banks (S31). Liveness
    is use-based ([gluon/memory.md](gluon/memory.md) §10): to keep a buffer out of later conversion scratch, call
    `_keep_alive` after those conversions, as Qwen3-Next `35b9` does after its epilogue ("avoid a shared-memory reuse
    barrier", S49); `0a07` calls it right after its load (S31). For the tested `0a07` specialization, removing its early `_keep_alive` did not change assembly or
    resources. This does not establish a general redundant-call rule: inspect allocator liveness for the
    exact layout and later conversion scratch.

- Conv windows: keep a tile's token rows in each lane's registers (`BlockedLayout([1, 2], [1, 64], [1, 1], [1, 0])`) and
  take the four taps as register slices with `gl.amd.slice`, "Four overlapping convolutions share their seven input
  rows" (S32). The slice keeps the source layout, so its offsets must fall on the layout's CTA tile (S39): with rows on
  lanes, e.g. `BlockedLayout([1, 2], [2, 32], [1, 1], [1, 0])`, the compile fails with "No source register holds the
  element for destination index [1, 0]" (compiled with Triton 3.8.0 for gfx950). Or stage the window in LDS and
  `.gather` the shifted rows (S33). Prove a tile interior (all taps in one sequence) at run time and take a mask-free
  branch for it (S33): in CKPT (`3828`) at 2048/1, every tile on the masked branch made the call 0.9–2.4% slower
. [gluon/memory.md](gluon/memory.md) covers both.
- Make masks fold at compile time. A runtime mask on a pointer `gl.load` puts a branch around every access (S45).
  SEQ32G's table builder masks rows with `(chunk * BT + ri < M) | complete_chunks`, where `complete_chunks: gl.constexpr
  = M % BT == 0` (S30). Compiled with Triton 3.8.0 for gfx950, at M = 8192 the builder issues 24 `global_load_dwordx4`
  and uses 190 VGPRs; at the ragged M = 4097 it issues 195 `global_load_ushort` behind exec-mask branches, and under the
  launch's `waves_per_eu=2`, a 256-VGPR cap (S30), it uses 256 and spills 64; without the cap it takes 322
  (66 of them AGPRs) and spills nothing ([tuning.md](tuning.md) §8). Its three masked operand loads as `buffer_load`
  drop their branches and, under the cap, the spill (243 VGPRs), but stay 2-byte loads. Specializing folds the mask only
  when M % BT == 0, and M already is a constexpr: at a ragged M, have the producer write zeros into the tail rows of the
  chunk-padded scratch (`packed_qk` holds whole 32-token chunks; its producer's store is masked at ragged M, S30) and
  load without a mask. In `2ece` at 4097/3 the builder then issues 24 `global_load_dwordx4` and fits 190 VGPRs at
  `waves_per_eu=2` without a spill; the call took 10% less (B/A 0.895–0.902), passing accuracy.

## 13. Choosing and measuring a template

Stored speedups describe whole calls against model-specific baselines, not isolated knobs. Decode
launch savings and large state streaming have different costs; prefill additionally depends on serial
chain depth, ragged masks and number of sequence/head strips. Do not extrapolate a nearest-file choice
or a fixed detection floor from another Point. Use matched inputs, same-day identical-arm controls and
uncertainty on mean differences (`tuning.md` §2); validate all in-place states as well as output.

## Evidence

- S1 aklr/plugins/sglang/models/qwen3_5/reference.py:113 decode rounding points (to :121): each conv tap product to BF16, SiLU output, beta, core, normalized output, then E4M3FN; padding rows; q times `scale` :148; `reference_prefill` :19 has no docstring, rounds q and k to BF16 :77-78, multiplies the core by `scale` :88, starts a fresh sequence from zero history :63, history = the window's last three :74. aklr/plugins/sglang/models/qwen3_next/reference.py:4-11: q and k rounded to BF16 in prefill only (:126-127), core times `scale` :147, decode q times `scale` :191
- S2 aklr/plugins/sglang/models/qwen3_5/contract.py:116 `conv_state` bound 0/0 (to :122); `CompareSpec(determinism="bitwise")` :189; `n_slots` 1–2047, `any_size`, :74; Qwen3-Next: aklr/plugins/sglang/models/qwen3_next/contract.py:91-97, :228, :275
- S3 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/7a69337f81190952.json#kernel.py:249 WY-form docstring of `_block_factors` (to :258); W and U from the inverse :343-351; `Ktail` from the last row of the relative decay :352-355, :365; `response` = tril(Q Kᵀ ⊙ G) :367-380
- S4 aklr/plugins/sglang/models/qwen3_5/baseline.py:127 decode baseline: `fused_qkvzba_split_reshape_cat_contiguous` :154, `causal_conv1d_update` :162, `fused_recurrent_gated_delta_rule_packed_decode` :171, `_layer_norm_fwd` :183, aiter `per_1x128` :227; the prefill baseline (:28) ends with `per_token_group_quant_fp8` :123; aklr/plugins/sglang/models/qwen3_next/baseline.py:30 aiter `per_1x128` in `_normalize_quantize`, used by prefill :104 and decode :162
- S5 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2c2a09e220f2706c.json#kernel.py:8 "The second launch normalizes the BF16 core output and advances convolution history, after every first-launch history reader has completed" (:8-9); `_value_and_gates` "Evaluate value SiLU, decay, and BF16 beta in neighboring lanes" :77-110; q/k in a `(2, BK)` fp32 LDS buffer :138-140; state `gl.load` :182, `gl.store` :202, no hint; norm layout and "stays wave-local" :214-215; `value_tile = 8` :312; launches :315-327
- S6 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/299133886d479017.json#kernel.py:116 state `buffer_load(..., cache=".cs")` (:116-118), store :141-143; `_store_quantized_head` :17-35, the same function text (md5) in the other seven fp8 decode files; state layout :87; `if slot >= 0:` :95, zero core otherwise :94, :146; history shift under `if slot >= 0:` in the epilogue :189-204; Q/K conv in the recurrence CTA :99-105; q in LDS :130-145; launches :234-246
- S7 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/4f257a3e566b20d1.json#kernel.py:77 history shift before arithmetic (:77-80); coefficient program "Queue the coefficient waves first..." :44-60 on the grid `(tokens, cdiv(channels, 64) + 1)` :200; padding still normalizes and quantizes :110-121; epilogue loads early :133; state `.cg` :146, `.cs` :160; late Q :152; BF16 before the cross-wave convert :162-164; launches :200-212
- S8 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/409cb22b39b3fa00.json#kernel.py:206 64-bit head rebase for pools over 2 GiB (:206-216); host byte test :318-319; state `.cg` :218, `.cs` :231; grid :327; Q, K, V and β in a `(4 * D,)` fp32 LDS buffer :143-146
- S9 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/20f63094b85c00a2.json#kernel.py:167 "Expose the large state fetch before preparing scalar gates"; epilogue inputs early :159; "Interleave heads over token groups for state-pool traffic." and the mapping :136-139; state `gl.load` :171, `gl.store` :194; launches :233-245, token group 32, 16 or 1 by divisibility :243; a4f80b6a3877b48e.json#kernel.py:254 `gl.load(..., cache_modifier=".cg")`, store `.cs` :259, launches :298-312; `SLOTS: gl.constexpr` :221, `gl.assume(slot < SLOTS)` :244, `SLOTS=slots` :310; token groups :227, XOR head permutation :229-231
- S10 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/fc261961f0ad572d.json#kernel.py:8 "fixed, spill-free 16-wave layout"; `token = (pid // (VH * 8)) * 8 + pid % 8` :182 on the grid `(tokens * v_heads,)` :300; layout :186; state load :202, `.wt` store :221; `num_warps=16`, `waves_per_eu=6 if tokens == 64 else 8` :321-322; the same in c2c3a42a3909684a.json#kernel.py:320-321 (mapping :180) and gdn_decode_n_v_heads16_key_dim128_value_dim128_conv_dim3072/cd00c50e595b05bc.json#kernel.py:299-300 (mapping :159)
- S11 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:467 `waves_per_eu` N becomes `amdgpu-waves-per-eu` = "N,N" (:462-468)
- S12 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#kernel.py:5 "split" from 1 to 8 rows, "fused" from 16 (:5-11); per-size grids: b31bfeccf6dcfd7c.json#kernel.py:463-466, bba17a5608327b11.json#kernel.py:359-367, 371905e46b19ee36.json#kernel.py:450-463, 35b9cfd56833844f.json#kernel.py:508-511, 06780bcbbb3e3c70.json#kernel.py:176-182
- S13 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_decode_config.py:76 `state_prefetch`, "the single most important knob for register pressure" (:76-83); `update_form` :85-89, "accumulate" at 16 and 64 rows :237, :264; "VEC * 64 * NW <= conv_width" and the 32-row two-owner case :100-106; `CONFIGS` :187, `state_tiling="heads"` at 16 and 64 rows :236, :263; exact-match `lookup` :323; kernel.py:977-992 `_fused_heads` launched without the cache and gate-position fields, `.cg` loads :728-737 and `.cs` stores :867-872 hard-coded; decay and β after the conv, "halve occupancy on the 4-warp instance" :482-484
- S14 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/06780bcbbb3e3c70.json#kernel.py:70 `gl.static_assert(NW * 64 <= conv_width)` on `BlockedLayout([1], [64], [NW], [0])` :78; "no replicated lanes/waves" :79-80; inline shift :48-51
- S15 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/c7840603329d5ed5.json#kernel.py:40 `gl.barrier` between the history loads (:24-28) and stores (:41-44); 38a742cfd908c997.json#kernel.py:312 `packed_history = T == 32`, history retired after "the full-CTA core redistribution" :447-454
- S16 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp:719 a barrier on local and global spaces becomes release fence, `s_barrier`, acquire fence (:719-736); TritonGPUToLLVM.cpp:113 Membar analysis runs before lowering (:113-115)
- S18 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:492 buffer-op `.cs` sets sc0 and nt; pointer loads and stores get only the nontemporal flag for `.cs` (:305-336)
- S19 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:104 `num_records` INT_MAX − 1 bytes; element offset times element size in 32 bits and masked lanes selected to an out-of-range offset :282-283
- S20 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp:433 `ClampFOp` lowers to `ROCDL::FMed3Op` (:433-451)
- S21 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp:2470 an fp8 destination type sets the FP8 clamping attribute on the function (:2470-2484); ElementwiseOpToLLVM.cpp:468-498 `adjustModeRegister` then inserts `llvm.amdgcn.s.setreg` of MODE bit 23 = 1 at the function entry, called from TritonGPUToLLVM.cpp:250
- S23 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/b31bfeccf6dcfd7c.json#kernel.py:306 `_round_bf16_ieee` (:306-313), used at :365 because `_finish_owned` passes `NATIVE_FP8=False` (:425-428), so the clamped Triton fp8 store :323 runs and the native branch :324-339 does not
- S24 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_primitives.py:476 native `store_fp8` branch, clamp then `v_cvt_scalef32_pk_fp8_f32` (:476-493); `ieee_pid` and `fp8_mode_scale`, `is_pure=False` (:741-754); kernel.py:304-305 and :430-431 `ieee_pid` at entry, :166-167 `fp8_mode_scale` before the quotient; `native_fp8_cvt=True` at 2, 4, 8, 32 rows gdn_decode_config.py:210, :221, :232, :258; 38a742cfd908c997.json#kernel.py:58-61 the setreg inside the convert asm, its `_ieee_pid`/`_fp8_scale` guarded off at its 32-row fp8 Point (:257, :476)
- S25 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_primitives.py:451 "deliberately a *separate* knob" (:450-453); accumulate form and "64 VGPRs at 8 warps, which is the difference between 4 and 3 waves/SIMD" :648-657
- S26 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#initial_state.py:9 `_initialize_fresh` zeroes a slot of the pool it is given when the device flag says fresh (:9-26); packing.py:4 "The conversion is part of candidate execution, including graph timing."; `_pack` writes Qwen3-Next's grouped order :15-43
- S27 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#kernel.py:236 `gl.barrier` then `_write_conv_state` (:236-237); old-history reads masked with the fresh flag :184-217; `_write_conv_state` keeps old entries for short sequences :134-152; `_permute_columns_for_a` :246-252, `_permute_rows_for_b` :256-262; state kept permuted :490, `assert_trivial=True` :554-557, un-permuted at the store :581; next chunk's Q/K and V loaded before its math :499-516, :524-544, `.cg` on V :515, :543; V through `SwizzledSharedLayout(4, 1, 8, [0, 1])` :530-536; BV 64 :471; `_chunk_coefficients[chunks, heads // 2]` :723, Grams once per head pair :363-367; `_delta_chunks[heads, 2, batch]` :739; `_prepare_boundaries` launched with `num_warps=1` :706-722
- S28 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/010eab6660c4e415.json#kernel.py:189 binary search over chunk bases `starts[s] // BT + s` (:189-196); "reserves enough space for every ragged tail" :199; 448925c31bc8ea55.json#kernel.py:1008 `chunks = triton.cdiv(m, bt) + batch`, :370-371 programs past the device-written chunk count do nothing
- S29 aklr/plugins/sglang/models/qwen3_5/api.py:150 prefill falls back while a stream is capturing
- S30 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2ecece9650b95f71.json#kernel.py:1812 `_chunk_recurrent` grid `(batch, heads, 128 // bv)`, bv 32 :1738; short path `if m <= 4096` :1757; `_build_chunk_tables_paired[chunks, heads // 2]` :1779 with `waves_per_eu=2` :1792; `_half_decay_scan`, two 16-token halves and a separable cross block :595-641; `_inverse_left_operand` :695-720, both callers pass `SHORT=True` (:1006, :1075), so its first middle layout :701-707 runs; `_stream_chunk_tables` "Store finished FP32 half tables early to shorten register lifetimes" :1002; `complete_chunks: gl.constexpr = M % BT == 0` and the row mask :1277-1278, operand loads :1279-1280, :1320; packed table reshaped and converted with `assert_trivial=True` :1430-1437; `packed_qk` in whole 32-token chunks :1743, its store masked at ragged M :475-484
- S31 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/0a07f9f9342c6cf1.json#kernel.py:1333 BV 16 below 4097 tokens, grid :1422-1427; `_upper_dense_product` skips the zero block :449-466; `_chunk_factors` "share four-head Gram matrices" :851 on `[chunks, heads // 4]` :1404; recurrence in a transposed `[1, 4]` layout at BV 16 :1057-1072; `_state_projection_operand` with `SharedLinearLayout` and `_keep_alive` right after the only load :993-1033; `large = m > 4096` :1332 with `normalized` the caller's buffer :1339 and `NORM_STRIDE == (8 + HEADS) * 128` asserted :1261-1263
- S32 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/7a69337f81190952.json#kernel.py:792 MAP chunk `max(256, cdiv(m, batch * 16 * block) * block)`, propagation grid :861; scratch `factors` and `maps` :744-755; reads the contiguous projection directly :89-99; register window :73-75, taps `gl.amd.slice(raw, [BT, 128], [j, 0])` :122, :142; gates converted to a blocked layout for the scan :301-305; masked product scan :310-319; `max_seq_len` taken by `run` and not read :924 (each of the 26 prefill kernel.py files names it once); 3b99d19659746a36.json#kernel.py:807 RSUM chunk 1024 or 512, boundary grid :873, padding decay 1 and beta 0 :289-290, `k_width = 4` with `[waves // 2, 2]` warps in the summaries :410-421 (b8c54bde37ee77c7 and 47092631ad3a6e79 are the same source)
- S33 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/382853eb770e30b3.json#kernel.py:1614 checkpoint scan grid, 64-token chunks :1538, 16 value rows :1539; `m <= 1024` selects the short path :1541-1547; `_convolution_interior` :59-73, branch :301; conv window in LDS read with `Window.gather` :254; `_decay_products` dyadic levels :670-697 with `31 - clz` :687; tiles stored only where `key_tile <= query_tile` :1008; the scan's `transposed=False`, `[1, 4]` layout :1294-1296, state and update through LDS read as `DotOperandLayout(0, mma, 4)` :1234, :1243-1247, :1266-1267; two 64-deep MFMA chains :1249-1255; `loop_range(..., loop_unroll_factor=8)` :1313; `iterative-ilp` :1627
- S34 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len1024_n_v_heads16_key_dim128_value_dim128_conv_dim3072/1fd6adb2b91cf5c2.json#kernel.py:19 "Only output reconstruction uses compensated BF16 products" (:19-21); four products :1302-1312; chunk 64 :1546, value block 8 at 1024 tokens else 16 :1615-1616 (same source as 61ed32f62fa797df and 1d5568a78971fb62); 19b20285ceb08b14.json#kernel.py:574 `_mm_query` (:574-599), same source as 382853eb770e30b3
- S35 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/11f62a16b4778338.json#kernel.py:791 `C=32 if batch >= 16 or 5 <= batch <= 7 else 64`; sequence search in `for depth in gl.static_range(0, 7)` :76-81 (1–2047 sequences in aklr/plugins/sglang/models/qwen3_next/contract.py:172); `_inverse16` "Inverse of I-P for a strictly lower triangular 16-by-16 P" :193-203, `_inverse32` joins two :218-226; "Direct interval scans also handle zero decay without inf-inf or 0/0." and the masked add scan under `gl.exp` :370-372
- S36 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/448925c31bc8ea55.json#kernel.py:1003 `snapshot_outputs = 3 <= batch <= 4 and m <= 4096`; `hierarchical` :1005
- S37 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:81 Table 28: F32 16x16x4 32 cycles; BF16 16x16x32 16 cycles at :101
- S38 triton-lang/triton@c01b677:lib/Analysis/Utility.cpp:1096 a gather is warp-local only if no warp or block basis touches the gather axis; :1311-1321 `cvtNeedsWarpShuffle`: fewer than two mixed transpositions; :526-531 `ScanLoweringHelper::isSupported` accepts only a blocked encoding; otherwise lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp:474 emits "TODO: unsupported scan layout"
- S39 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/slice.py:12 `gl.amd.slice`: register-only slice, no cross-thread movement (:12-23)
- S40 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py:12 the cdna4 module re-exports `buffer_load`, `buffer_store` and `mfma` (:12-18)
- S42 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245 `assert_trivial` raises "layout conversion from … is not trivial" with both linear layouts unless the conversion is trivial (:245-248)
- S43 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertLayoutOpToLLVM.cpp:44 permlane swap only for exactly one (register, lane 4 or 5) transposition or a 3-cycle through lanes 4 and 5 (:44-87)
- S45 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MaskedOpsToLLVM.cpp:81 a mask that is not constant true becomes a conditional branch around the load (:81-104)
- S46 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#kernel.py:281 `_inverse_unit_lower_16`: 4 × 4 diagonal blocks by `gl.gather` pivots, then three fp32 MFMAs (:280-300); gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/7a69337f81190952.json#kernel.py:338-342 15 rank-1 eliminations; gdn_prefill_group_fp8_quant_max_seq_len8192_n_v_heads16_key_dim128_value_dim128_conv_dim3072/b8c54bde37ee77c7.json#kernel.py:224-232 `_inverse16`, three squarings, fed the negated lower part :325-335
- S47 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_max_seq_len5120_n_v_heads8_key_dim128_value_dim128_conv_dim2048/98af1fd2ff621082.json#kernel.py:115 "Only this CTA reads history for this sequence/channel group. Finish its readers before updating any cache element." then `gl.barrier` and the stores (:115-131); the same in ed2a8611dd433d8a
- S48 aklr/plugins/sglang/models/qwen3_next/contract.py:8 projections grouped by key head (:8-12, :80-89), conv channels every Q, then K, then V :63, :92; `conv_bias` in both contracts :100; outputs `normalized`, `values`, `scales` :117-145; aklr/plugins/sglang/models/qwen3_next/reference.py:35-43 `_ungroup`; aklr/plugins/sglang/models/qwen3_5/contract.py:114-115 contiguous projections; `conv_bias` in prefill :151, absent from the decode inputs :220-238; outputs :168-187, :263, :275
- S49 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/35b9cfd56833844f.json#kernel.py:374 "Only a register-owned column prefix stays live across convolution." (:374-379), tail and merge :417-422, off at its 16-row Point (`STATE_PREFIX` :333); update in accumulate form :428-431; three quadrant stores before the epilogue conversion, the last after it :433-443 (`PHASED_STORE` at 16 rows :338); `_keep_alive` after the epilogue, "Separate convolution storage from the epilogue's conversion scratch to avoid a shared-memory reuse barrier." :452-454; 0c0b7dfe26861b02.json#kernel.py:258 the same prefix load (:258-262, :316-321) with `state_prefix = 64` :478
- S50 aklr/core/contract/semantics/_distributions.py:276 `BoundedCuSeqlensDistribution.sample`: one sequence of exactly `max_seqlen`, the rest split at random (:276-292)
- S51 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_max_seq_len5120_n_v_heads8_key_dim128_value_dim128_conv_dim2048/98af1fd2ff621082.json#kernel.py:107 "Initialize padding on every invocation so all downstream loads are safe." and the zeros stored (:107-110); qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/010eab6660c4e415.json#kernel.py:10 "Every consumed scratch element is initialized within the invocation, including ragged tails." (:10-11)
