---
title: "Decode attention in Gluon on gfx950: DSA sparse MLA and Qwen3.5 gated attention"
kind: topic
ops: [attention]
dtypes: [bf16, fp8]
regimes: [decode]
gens: [gfx950]
triton: 3.8.0
---

# Decode attention in Gluon (gfx950)

AKLR serves two decode attention ops with Gluon kernels on MI355X: DSA sparse MLA decode, where each token attends the
2048 fp8 KV rows its top-k slots name, and Qwen3.5 gated attention decode, where 8 query heads share one fp8 KV head
and the output gate and fp8 group quantization are fused in. All but the 256-token 8-head DSA file split each token's
keys over programs, write per-split records to scratch and merge them in a later launch.

## Route here when

- You write or tune a Triton or Gluon decode attention kernel for an AKLR schema: gathered (sparse) MLA, paged GQA
  decode, or attention with a fused prologue or epilogue.
- You choose the split count, the head tile, the producer grid or the merge of a split-KV decode; you gather KV rows
  by slot and stage them in LDS for the MFMA; you choose the precision of QK, of P·V or of the split partials.

**Not in AKLR's Triton kernels:** softmax attention for prefill or extend, decode with more than one KV head per rank,
bf16 KV pools; start those from `../attention/INDEX.md` (measured on FlyDSL kernels). Linear attention: `gdn.md`.

**Somewhere else if:** the mechanisms, with numbers measured on FlyDSL kernels → `../attention/INDEX.md`, first
`../attention/decode_split_kv.md`; a launch's cost under each ruler → `authoring.md` §7; loads, async copies, waits,
barriers → `gluon/memory.md`; layouts → `gluon/layouts.md`; MFMA → `gluon/mfma.md`; inline assembly →
`gluon/wave_ops.md`; the contracts' numerics → `numerics.md`; the ruler and A/B protocol → `tuning.md`; every AKLR
Triton kernel → `kernels/kernel_library.md`. The configurations below are shape-specific source examples for
Triton 3.8.0 on gfx950, not universal tuning rules. Original kernel and compiler citations are listed under Evidence.

## 1. Decisions at a glance

| You need | Starting mechanism | § |
|---|---|---|
| reproducible split-KV decode | neutral records for empty splits, fixed-order merge in a later launch, no float atomics | 2 |
| producer loop | peel the first tile; specialize full-tile masks; guard all-empty running maxima | 2, 7 |
| grid and head tile | tune head grouping and split count together at the actual batch/length distribution | 3, 4, 7 |
| gathered KV rows | vectorized 32-bit buffer offsets within the contract's byte bound | 5 |
| async copies into swizzled LDS | pre-swizzle global columns and copy into a linear alias | 8 |
| one KV head shared by query heads | compare per-head VALU against grouped-head MFMA; tiny batches need their own comparison | 6 |
| QK and P·V precision | exact fp8 upcast or unit-scale fp8 MFMA; bf16/fp16/hi+lo probabilities according to the output bound | 10 |
| gate and quantization | fuse into the merge while the output is in registers | 9 |
| Q/K norm, mRoPE and cache write | one-wave role grid, wave-local XOR partner gather | 11 |
| cross-wave softmax | defer denominator, redistribute scores or use a two-level reduction | 12 |

## 2. Split the keys, merge in a later launch

Each producer program owns a key range (a split) of one token for a group of heads, runs an online softmax over it,
and writes a record: the unnormalized numerator Σ exp(s − m)·v, the running max m and the denominator Σ exp(s − m).
One tile's step in the 32-row gated-attention file (S1): max and sum are `SliceLayout(1, mma)` vectors, one value per
head row, and the numerator stays in the MFMA accumulator. The caller peels the first tile (`FIRST`: no rescale) and
compiles the masks only into the variant for a row's last, partial tile (`FULL_TILE=False`):

```python
key = gl.convert_layout(gl.permute(key, (1, 0)), b_layout).to(gl.bfloat16)
scores = gl.amd.cdna4.mfma(
    q, key, gl.full((q.type.shape[0], TILE), 0, gl.float32, mma)
)
scores = scores * key_scale * attention_scale
score_pos = tile * TILE + gl.arange(0, TILE, gl.SliceLayout(0, mma))
if not FULL_TILE:
    scores = gl.where(score_pos[None, :] < length, scores, -float("inf"))
if FIRST:
    next_max = gl.max(scores, 1)
else:
    next_max = gl.maximum(running_max, gl.max(scores, 1))
    correction = gl.exp(running_max - next_max)
probability = gl.exp(scores - next_max[:, None])
if not FULL_TILE:
    probability = gl.where(score_pos[None, :] < length, probability, 0.0)
...
if FIRST:
    running_sum = gl.sum(probability, 1)
else:
    running_sum = running_sum * correction + gl.sum(probability, 1)
    numerator = numerator * correction[:, None]
```

P·V follows, `gl.amd.cdna4.mfma(gl.convert_layout(probability.to(gl.bfloat16), a_layout), value, numerator)`, V read
through padded LDS (§8); K and V are gathered by slot above (§5); these `gl.max` and `gl.sum` cross waves (§12). A
later launch reads the records of all splits of a (token, head), weighs split s by exp(m_s − M), M the largest m_s,
sums in a fixed order and divides once, as the 24-token DSA merge does (S2):

```python
maximum = gl.max(maximums, 0)
safe_max = gl.where(maximum == -float("inf"), 0, maximum)
correction = _softmax_exp(maximums - safe_max[None, :])
denominator = gl.sum(denominators * correction, 0)
...
numerator = gl.sum(partials * correction[:, :, None], 0)
inverse_denominator = 1.0 / gl.where(denominator > 0, denominator, 1)
```

`_softmax_exp(x)` is `gl.exp2(x * 1.4426950408889634)`. Its lowering can be shorter than `gl.exp`,
but exponent/denormal behavior and actual attention accuracy must be checked (`numerics.md` §6).

- **Merge in a later launch.** None of AKLR's 17 Gluon attention files (8 DSA, 9 Qwen3.5) makes one workgroup wait
  for another or reduces in the last workgroup to arrive (why: `../attention/decode_split_kv.md` §2, §3). What a
  separate merge launch costs depends on who times it (`authoring.md` §7): CI sums the Kineto durations of the call's
  kernels, gaps between graph nodes excluded , and there an added launch reads at least AKLR's dispatch floor
  (`../pitfalls/aklr_dispatch_floor.md`), 4.06 µs for a trivial kernel; a bare replayed CUDA graph pays 1.77–1.84 µs
  of wall time per extra chained kernel. Put follow-on work into the merge (§9).
- **No atomics.** Both contracts require `determinism: bitwise` (S4); a fixed split layout and fixed-order
  merge avoid arrival-order float sums. The accuracy bound is a separate requirement (§10).
- **Every split writes a record, empty or not:** max −inf, denominator 0, numerator 0 (S1).
  This makes scratch from `torch.empty` safe for changed lengths under graph replay, provided every consumed
  record is rewritten within the call. Lengths and slots remain device inputs; configuration uses batch shape (S5).

- **Lay the records out for both launches.** Simplest, Qwen3.5's: numerators `[rows, heads, splits, 256]` bf16 and
  statistics `[rows, heads, splits, 2]` fp32, the numerator converted to
  `BlockedLayout([1, 8], [8, 8], [1, gl.num_warps()], [1, 0])` so each lane stores 16 bytes (S1). The 64-head DSA
  producers store straight from the transposed 32 × 32 accumulator in `[channel // 8][head][channel % 8]` order, read
  back 8 channels per lane, and at 64 heads XOR the channel group with `split * 4`, for a reason the source does not
  state (S2). The 12- and 16-token files pad a record to 544 channels (S6), the 16-token one keeping the statistics in
  the padding (S7); the 256-row Qwen3.5 file interleaves 4 heads for its 4-wave finish (S8). These record layouts have no isolated performance comparison.
- **Guard the running max in the producer as well as in the merge.** DSA's contract marks an unused slot −1, and its
  reference outputs 0 for a token that reads no row (S9): a kernel must survive empty tiles, splits and tokens. Every
  DSA producer takes its exponentials against `safe_max = gl.where(m == -float("inf"), 0, m)` (S10, S11): without it,
  a split whose first two tiles are empty rescales by exp(−inf − (−inf)) = NaN in its token; the merges guard their
  max likewise. Otherwise an all-empty token produces NaN; guard the final denominator so it outputs zero.
  Distinct-slot benchmark inputs may not exercise empty tiles (S12), so cover −1 tails spanning a tile, a split
  and a whole token. Qwen3.5's contract requires at least one key per row (S13); its producer only runs tiles
  containing a key (S1), which is why its valid-row path does not need the same running-max guard.

## 3. DSA sparse MLA decode: plugin, Points, grid

The plugin replaces sglang 0.5.22's `DeepseekSparseAttnBackend._forward_triton_decode` with
`dsa_sparse_mla_decode@1.0`; speculative decoding stays on sglang (S14). The baseline is sglang's split-K Triton
kernel `triton_sparse_mla_decode_splitk` (S15). q is `[tokens, heads, 576]` fp8 e4m3fn (512 latent channels, then 64
RoPE channels), the pool `[n_slots, 576]` fp8, 2048 slots per token (−1: none), the output `[tokens, heads, 512]`
bf16; there are no scales. Two scenarios, one file per Point:

| Heads | Tokens | File | Heads per program × splits | Producer: programs × warps | `waves_per_eu` | LDS (B) | VGPRs | Merge: programs × warps |
|---|---|---|---|---|---|---|---|---|
| 64 | 8 | `0bb2f7cbe49d1be3` | 16 × 8 | 256 × 4 | 2 | 147,696 | 150 (22 of them AGPRs) | 512 × 4 |
| 64 | 12 | `1148d1c8e30397f7` | 32 × 8 | 192 × 8 | – | 156,160 | 88 | 768 × 1 |
| 64 | 16 | `f60f22e6f14448c6` | 32 × 8 | 256 × 8 | 2 | 156,160 | 94 | 1024 × 1 |
| 64 | 24 | `2ee56a6dbabc48a6` | 64 × 8 | 192 × 8 | 2 | 163,840 | 228 | 1536 × 1 |
| 64 | 32 | `2f7000a8186a8ae8` | 32 × 4 (two 256-key tiles each) | 256 × 8 | 1 | 157,696 | 214 | 512 × 4 |
| 8 | 32 | `43ffd5e3e5b67298` | 8 (16 MFMA rows) × 16 | 512 × 4 | – | 49,216 | 154 | 256 × 2 |
| 8 | 128 | `ae32cc132679692c` | 8 × 2 | 256 × 4 | – | 79,872 | 199 | 512 × 2 |
| 8 | 256 | `fbe6ddab5dac5958` | 8, one program per token | 256 × 4 | 1 | 83,968 | 202 | none |

LDS and registers of the compiled producers, one variant each, no spills
  VGPRs being `.vgpr_count`, VGPRs and AGPRs together (`kernels/kernel_library.md` §1); grids, warps and
`waves_per_eu` as traced at launch; heads per program × splits from the host code. A file is
`<file>.json` in the directory of S16 (64 heads) or S11 (8 heads), its sources the `content` of each `kernel.sources`
entry (`aklr/ops/README.md` § The kernel section and ABI). The 8-token file and the three 8-head files add a
`q.to(torch.bfloat16)` launch in `run` (S17, S11, S18, S19).

- **Tune the grid within a Point.** The committed 64-head producers use 192–256 eight-wave programs.
  Fewer or more splits can be slower: splits also change tiles per producer, merge width and sometimes layout.
  LDS occupancy estimates do not establish the timing cause. Compare neighbors on the same shape and inputs.

- **Do not start a 64-head kernel from the 8-token file.** It differs from the 12–32-token files in many choices:
  16-head groups of 4 warps, rows async-copied into LDS, bf16 hi + lo P, fp32 partials and the extra `q` cast. It
  stores 1.030, at the edge of what dispatch serves (`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` §
  Environment variables); its baseline differs from theirs, and no Point attributes the gap to one choice.
- **With 8 heads the MFMA did not bound the 32- and 128-token producers.** Each 576-byte row feeds 8 query rows, about
  30 FLOP per gathered byte (estimated); doubling the QK MFMAs cost 0.6–1.0% at 32 tokens and 4.2–4.7% at 128
  the gather plausibly bounds them (inferred). The three store 1.050–1.196. At 256 tokens
  one program per token runs all 32 key tiles, with no partials or merge (S18); at 128 tokens two programs per token and
  a 2-way merge. The 128- and 256-token files' staging (async panels per 64 keys, a live-tile bitmap from a wave ballot,
  double-buffered panels, raw `s_waitcnt vmcnt(n)`, S11): `gluon/wave_ops.md` §6, `gluon/memory.md` §11; at 128 tokens
  each raw wait counts the loads issued after its panel, so dropping one leaves a panel in flight (compiled).

## 4. DSA: head tile, split count, `waves_per_eu`

- **Widen heads to reuse gathered rows, but price the coupled grid.** A 32-head tile reads each KV row twice
  for 64 heads; a 64-head tile can read it once. The 24-token source chooses 64 heads × 8 splits (192 programs).
  Its measured whole-call advantage over the tested 32-head/4-split design was about 8–9% on MI355X;
  that comparison also changes tiles and merge width, so it does not isolate the saved reads.
  Widening was slower at 12 and 16 tokens, and a tested 32-head design was faster at 32 tokens.
- **Respect the LDS limit when widening.** The original 16-token 64-head specialization requests 164848 B,
  above the 160 KiB limit. A fallback needs both fitting LDS and correctly sized independent packed statistics.

- **Head tile and split count are coupled in the source.** The host code sets both from the token count,
  `head_tile = 64 if rows == 24 and heads == 64 else 32` and `splits = 8 if rows <= 16 or head_tile == 64 else 4`,
  and launches `_attention_split[rows, heads // head_tile, splits]` (S16). The 64-head path is written for one
  256-key tile per program: it reads `cache_shared`, which the kernel allocates only when `TILES == 1` (S20). Four
  splits at 24 tokens give two tiles per program, and the producer fails to compile
  (`NameError: cache_shared is not defined`), without a GPU too (Triton 3.8.0, gfx950). A second tile
  cannot sit beside the first, which already takes 160 KiB with P (§5): start from the 32-token file.
- **Two tiles per program: refill under P·V.** The 32-token file holds a tile as two `[256, 256]` LDS panels. During
  tile 0's P·V, per 64-key slab, it loads P and V from LDS, gathers tile 1's rows, runs the MFMAs, and after a
  `gl.barrier` stores them over the slab; tile 1 then rescales by exp(m_old − m_new) (S21). The refill has no established whole-call gain; the other 64-head files hold one tile per program.
- **Readers of the same rows on one XCD.** Workgroups go round-robin to the 8 XCDs and their L2s (`tuning.md` §12).
  The 8-token file sets `row = program_id % 8`, so a token's programs share an XCD (S17); in the 3-D grids a (token,
  split)'s two head blocks share one XCD at 16 and 32 tokens, 4 apart at 12 (inferred). No Point measures it; in a
  shared-reader microbenchmark this placement helped, but it does not establish actual attention L2 causality.
- **Treat `waves_per_eu` as a register-allocation knob.** The 24-token producer uses 160 KiB LDS and can fit only
  one eight-wave workgroup per CU; the hint cannot increase that LDS-limited residency (S22).
  Inspect register counts and time the full call rather than infer a gain from the hint alone.

## 5. DSA: gathering the 2048 rows

- **32-bit offsets over the whole pool.** The contract bounds the pool to 2^31 − 2 bytes (S9), the range of the buffer
  descriptor Triton 3.8.0 builds, so a `gl.amd.cdna4.buffer_load` with 32-bit offsets reaches every row, and a masked
  lane gets an out-of-range offset and reads 0 (S23). The 64-head files clamp a −1 slot to row 0 and mask it; the
  32-token 8-head file masks its panel copies (`slots[:, None] >= 0`, S19); the 128- and 256-token 8-head files turn
  it into an `INT_MIN` element offset and copy with no mask, which stays out of range only for 1-byte elements (S11, `gluon/memory.md` §4).
- **16 bytes per lane, a whole tile in flight.** The 12- to 24-token producers gather each 256-row × 512-byte tile into
  registers as one tensor, in `BlockedLayout([1, 16], [4, 16], [8, 1], [1, 0])` (12 tokens: `[8, 8], [4, 2]`, S6), the
  32-token one in halves and refill slabs; 16 fp8 values per lane in one `buffer_load_dwordx4`, then to LDS (S24):

```python
indices = (split * TILES + tile) * 256 + key_offsets
slot = gl.load(Slots + row * TOPK + indices)
valid = slot >= 0
if CACHE_BUFFER:
    base = gl.maximum(slot, 0) * WIDTH
    ...
    latent = gl.amd.cdna4.buffer_load(Cache, base[:, None] + channels[None, :], valid[:, None], 0.0)
    ...
    cache_shared.store(latent)
```

  That is 128 KiB per program, 16 loads of 16 bytes per lane, all issued before the first `ds_write_b128` (24-token
  producer, compiled for gfx950). Earlier contiguous bf16 streams at 256×4 read 2.24–2.29→7.13–7.17 TB/s when independent 16-byte loads per lane increased 1→8 (3 runs) . A new synthetic distinct-random-row fp8 gather directly measured U8/U1 gains 3.171× for 256-byte rows and 2.959× for 576-byte-stride rows reading only 512 bytes (256 programs×4 warps, 12 randomized blocks; paired 95% CIs 3.1698–3.1729/2.9568–2.9613); the matched bf16 stream read 2.240→7.073 TB/s .
  Gather gains fell to 1.825/1.718× at 512 programs and 1.172/1.109× at 1024. These are synthetic probes, not attention/DSA measurements; repeated indices, larger U and a universal “many loads” benefit remain untested.
- **One LDS tile serves QK and P·V.** In MLA the 512 latent channels of a row are the key (with the 64 RoPE channels)
  and the value: the producer stores the tile once and reads it as the QK operand through a `.permute((1, 0))` view and
  as the P·V operand directly (S25; a stand-in matched torch ). It loads the 64 RoPE channels in
  `BlockedLayout([32, 1], [2, 32], [HEAD_TILE // 32, 256 // HEAD_TILE], [1, 0])`, where the B operand of the
  `[32, 32, 64]` scaled MFMA wants each element: the `gl.convert_layout` that follows compiles with
  `assert_trivial=True` (Triton 3.8.0, gfx950, both head tiles), with no LDS and no lane shuffle.
- **At the 160 KiB limit, allocate late:** the allocator overlaps buffers whose live ranges do not meet
  (`gluon/memory.md` §10). The 24-token producer keeps the `[256, 512]` fp8 tile (128 KiB) through P·V and allocates
  its `[64, 256]` bf16 P (32 KiB) after the last use of the query staging and reduction buffers (S20): 163,840 B
  ; allocated before the second key reduction, P takes it to 165,888 B with the 2 KiB scratch (compiled).
- **The bench has no index locality.** Each token's 2048 entries are distinct and uniformly sampled from −1 and the pool's slots (S12),
  so the 8-head files' live-tile skipping does no work on the bench. Test the locality levers of
  `../attention/decode_split_kv.md` §7 (measured on FlyDSL and Triton kernels) with production-like indices.

## 6. Qwen3.5 gated attention decode: plugin, Points, launches

The plugin replaces sglang 0.5.22's `Qwen3_5AttentionDecoderLayer.self_attention` on HIP decode with
`gated_attention_decode_group_fp8_quant@1.0`, between sglang's `qkv_proj` and an `o_proj` that takes the op's fp8
values and per-128 scales; it falls back on another attention backend, on KV scales other than 1 and in the other
cases `_attention_falls_back` lists (S26). The baseline is six library calls: the fused QK norm and gate split, mRoPE,
the fp8 KV-cache write, aiter's `paged_attention_ragged`, `fused_sigmoid_mul` and the per-1×128 quantization (S27). 8
query heads share one KV head of 256 channels in fp8 pools; the bench gives each row 4096 keys on average, with
lengths 54.5× apart in a CPU reconstruction of the workload draw at 256 rows (S12). Three launches:

1. **Prologue**, a task grid of one-wave programs, each owning one job of a row: key/value cache work (separate or
   combined in channel quarters), one or more query heads (RMSNorm, RoPE), or gate copies, cut differently per Point (§11). It
   ends before attention starts because attention reads the slots it writes.
2. **Partials**, the split-KV producer of the table below.
3. **Finish**: the merge, the gate and the fp8 group quantization (§9).

| Rows | File | Producer | Programs × warps | Splits × tile (keys) | Mean tiles per split (estimated) | LDS (B) | VGPRs | CI speedup |
|---|---|---|---|---|---|---|---|---|
| 1 | `f5ba496e04f8e989` | one program per (row, head, split), fp32 multiply-adds, no MFMA | 256 × 4 | 32 × 64 | 2 | 1,024 | 130 | 2.182 |
| 2 | `42a24a8f5e63bd0c` | the same attention code as 1 row | 512 × 4 | 32 × 64 | 2 | 1,024 | 130 | 1.699 |
| 4 | `3d0f64ffff6790c8` | all 8 heads in a 16-row MFMA; K and V gathered into registers; fp16 P·V | 128 × 4 | 32 × 64 | 2 | 32,768 | 182 | 1.671 |
| 8 | `10df1ae2dc9d3c69` | async K and V into LDS, no V copy permitted outstanding at the actual pre-QK wait ; bf16 hi + lo P (§10) | 256 × 4 | 32 × 64 | 2 | 34,816 | 125 | 1.803 |
| 16 | `6dd321aac1b1d85a` | K into registers, V through padded LDS; bf16 partials | 512 × 4 | 32 × 64 | 2 | 16,624 | 136 | 1.708 |
| 32 | `d9cbde237fd34de9` | as 16 rows, with a 16-row head tile | 512 × 4 | 16 × 128 | 2 | 33,264 | 188 | 1.789 |
| 64 | `eeb171871a296094` | the same attention code as 32 rows; 8-row head tile | 512 × 4 | 8 × 128 | 4 | 33,264 | 192 | 1.401 |
| 128 | `14e91857b4038e02` | async Q, K and V with pre-swizzled offsets; one contiguous chunk per split | 512 × 4 | 4 × 128 | 8 | 69,632 | 147 (63 of them AGPRs) | 1.204 |
| 256 | `dfb1ab1cdd5bf263` | as 128 rows | 512 × 4 | 2 × 128 | 16 | 71,680 | 172 | 1.066 |

CI's stored speedups ; the compiled producers: 4 warps each, LDS and VGPRs as listed, no spills, no MFMA at
1–2 rows ; programs and splits × tile from the launch code (S5, S29, S8), checked by the launch trace (field, E15). Mean tiles per split =
4096 / (splits × tile). A file is `<file>.json` in the directory of S5, holding `kernel.py` (`run`), `prologue.py` and
`attention.py`; the 4-row one allocates no LDS, its 32 KiB being compiler scratch (compiled), for converting K, V and P
to operand layouts (inferred, S28). The head tile is the rows of the query, score and
accumulator tensors on the `[16, 16, 32]` MFMA, `head_tile = 16 if SPLITS == 16 else HEADS` (S29): 16 pads the 8 heads
with 8 zero rows (S1); 8 leaves copies in the lanes of the missing rows and, in a one-wave stand-in, issues the same
MFMA, load, store and `v_exp_f32` instructions as 16 rows: it saves none (compiled; `gluon/mfma.md` §4).

- **Per-head at 1–2 rows, combined heads in MFMA above: implementation, not a universal optimum** (grids/layouts: S5/S28).
  At the 4-row Point, `3d0f` MFMA took 21.59–21.77 µs versus per-head 26.79–26.95, per-head/MFMA time ratio 1.235–1.248 (4 runs, 16 pairs). At 8 rows its branches observed paired ratios 1.975–2.019 (3 runs, 12 pairs; MFMA 27.74–28.11 versus per-head 55.08–55.89 µs).
  At 2 rows the same-file branches observed MFMA/per-head 1.025–1.067 (3 runs, 12 pairs), while committed `42a2` versus replacement MFMA `attention.py` observed 0.988–1.005 (2 runs, 8 pairs; merge changes too). At 1 row the earlier comparison observed run-mean MFMA/per-head 0.982–1.001 (4 runs, 16 pairs). All timed GQA arms passed accuracy .
  Fixed AB order and cross-process dependence leave mean-difference uncertainty unresolved, especially near zero; neither a general 2–4-row crossover nor a small 1-row advantage is established. Shared-KV traffic and its contribution were not separately measured.
- **Where the gain is.** Up to 32 rows a call reads 2–64 MiB of KV (estimated), and the op runs 3 kernels where the
  baseline's six library calls run 7 (S27): at AKLR's dispatch floor of about 4 µs per kernel
  (`../pitfalls/aklr_dispatch_floor.md`) the 4 saved are 16 µs (estimated), 47% of the measured gain at 32 rows and 86%
  or more at 1 to 16 . At 256 rows × 4096 keys (N=4 runs), the partials kernel's 512 MiB call took
  78.34–79.42 µs (6.76–6.85 TB/s), baseline paged attention 144.53–145.13 µs (3.699–3.715 TB/s), against a flushed
  random-permutation 256-byte-row gather at 5.109–5.400 TB/s. Kernels were profiled unflushed with replacement-drawn
  slots: rates are not like for like and establish no gather bound; 64/128 rows untested here .
  The separate field speedup remains 1.066 . In a Triton toy, one row 55× longer than seven others,
  fixed 4 splits/row (32 programs) took 50.74–51.21 µs versus 6.13–10.56 µs for 62 equal chunks, 4.806–8.356×
  (N=4 replayed-graph runs); the long-row tail was not measured on Qwen3.5 files .
- **Copy a file only at its own Point.** Below 256 rows the 256-row file's prologue writes Q and the gate interleaved
  as `uint32` pairs and returns strided views (S30), while its attention, which also accepts 128 rows, reads both as
  contiguous rows (S8): at 128 rows it raised no error, its values at −1.3 dB against a 15 dB bound .

## 7. Qwen3.5: the split count by rows

From 16 rows the committed split count keeps `rows × splits = 512`, nominally two four-wave producers per CU (S29), not a measured residency/optimality rule:
`splits = 32 if rows <= 16 else (16 if rows <= 32 else 8)`, `tile = 128 if rows >= 32 else 64`, grid
`(rows, splits)`. The 128- and 256-row file continues the rule with 4 and 2 splits (S8). At 32 rows, against the
committed 16 splits with nothing else changed in the sources, in two sessions :

| Splits at 32 rows | Programs ÷ CUs | Mean tiles per split (estimated) | Speedup vs baseline | Call time | Speedup ÷ committed |
|---|---|---|---|---|---|
| 8 | 1 | 4 | 1.129–1.132 | 69.8–69.9 µs | 0.63 |
| 16 (committed) | 2 | 2 | 1.780–1.791 | 44.1–44.3 µs | 1 |
| 32 | 4 | 1 | 1.690–1.693 | 46.6–46.8 µs | 0.95 |

Call time is the three kernels' device time per call. The file ties its head tile to the split count (S29), so both
changed arms also ran an 8-row head tile: the table measures the pair, not the split count alone.

- **At the older 32-row Point, half/double splits took 1.58/1.06× as long**, losing about 37%/5% of the speedup, not increasing call time by those percentages . Splits also change the head tile, tiles per producer and merge records; CU residency and in-flight-load causality were not measured, and a synthetic stream is not a decode measurement.
- **Measure both neighbours at each Point; neither a 512-program grid nor a two-tile floor is universally optimal.** New accurate r3 comparisons use N=6 counterbalanced outer rounds per arm (3 forward/3 reverse) :
  - Doubling splits at 4/8/64/128/256 rows gave modified/committed call-time ratios 0.7863/0.8142/0.9731/0.7818/0.7052; paired-log 99% CIs 0.782904–0.789708/0.804810–0.823643/0.970714–0.975498/0.780215–0.783308/0.704074–0.706232. Both order strata agree; the 4/8-row doubled arms have one tile per split.
  - At 16 rows half/double = 1.441/1.033; double CI 1.018381–1.046948 supports slower, not a guaranteed ≥2% effect. The committed arm was fastest among tested arms. At 32 retain the older 16-split comparison (half/double 1.58/1.06); no 256-row half or exhaustive optimum search was run.
  Head-tile changes following the file rule, tiles per split and merge width remain confounders; these measurements neither isolate a CU-grid mechanism nor establish a global optimum.
- **Which tiles a split takes.** The 1- to 64-row files deal a row's tiles out in turn, split s taking s, s + SPLITS, …
  (`for tile in range(split + SPLITS, gl.cdiv(length, TILE), SPLITS)` after the peeled first tile, S1), so a row's
  splits differ by at most one tile. The 128- and 256-row files give each split one contiguous chunk of
  `gl.cdiv(length, SPLITS * BLOCK) * BLOCK` keys, leaving the last split empty for rows of up to 384 keys at 4 splits,
  128 at 2 (S8); the 64-head DSA producers also give each split a contiguous slot range (S24); no Point compares the
  two. An empty split still writes its neutral record, and at 32 and 64 rows skips the Q load (`QUERY_GUARD`, S1, S29).
- **Where V is gathered is a per-Point knob, unablated:** "Stage 0 issues V before QK; stages 1/2 bracket the sum
  reduction" (S1); the first tile and the rest use stages 0 and 2 at 16 rows, 1 and 1 at 32, 0 and 0 at 64 (S29). The
  8-row file loads the next strided tile's slots behind the V copy (`PREFETCH_STAGE`, S31). No attention Point compares
  async copies with register gathers; on a 64×64 bf16 tile stream async copies ran from 4.3% below to 0.9% above register-only
  loads (1.3–4.3% below at 2048 programs in all 4 runs), and in an 8-token MoE router contiguous register loads and layout
  conversions in place of async copies changed call time by −0.1% to +1.5% (8.30–8.67 µs across arms, 4 runs)  (`gluon/memory.md` §8).
- **`waves_per_eu` is a register choice here, not a grid one.** The files set 2 at 16 and 32 rows and 3 at 128 rows
  (S29, S32). Compiled with Triton 3.8.0 for gfx950: 188 VGPRs with it and 186 without at 32 rows; at 128 rows 147
  registers, 84 VGPRs and 63 AGPRs, with 113 `v_accvgpr` moves (3039 instructions against 2983), and 164 VGPRs
  without, occupancy 3 for both, though 69,632 B of LDS let only 2 waves per SIMD reside (estimated). Untimed.

## 8. Qwen3.5: K and V into LDS

| Rows | K | V |
|---|---|---|
| 1–4 (S28) | gathered into registers | gathered into registers |
| 8 (S31) | async copy into a swizzled LDS tile | async copy into a second LDS tile; actual `EARLY_VALUE=True` pre-QK wait permits no V copy outstanding  |
| 16–64 (S37) | gathered into registers, converted to the QK operand | gathered into registers, stored to padded LDS, read as the P·V operand |
| 128–256 (S8) | async copy at pre-swizzled offsets (Q joins K's first group) | async copy at pre-swizzled offsets; pre-QK ISA permits eight LDS-direct copies outstanding, dynamic overlap unmeasured  |

- **Two K/V commit groups do not guarantee V–QK overlap: the actual 8-row launch is a counterexample.** With `EARLY_VALUE=True` in `10df1ae2`, the pre-QK `vmcnt(4)` leaves only ordinary index-prefetch loads permitted outstanding, no LDS-direct V copy; one group uses `vmcnt(0)`. One-group/two-group call-time ratio ≈1.019 (2 runs, 8 pairs; 22.33–22.35 versus 21.93–21.94 µs; accuracy passed) is not evidence of V overlap or a resolved mean difference. At 128 rows (`14e91857`) run-mean ratios were 1.167–1.169 (2 runs, 8 pairs; accuracy passed); at 256 (`dfb1ab1c`) 1.102–1.108 (6 runs, 24 pairs; accuracy passed in 5 runs/20 pairs, failed in the initial run). Corrected ISA at 128/256 permits eight LDS-direct copies outstanding at `vmcnt(8)`, versus zero with one group; actual dynamic overlap and causal contribution to speed were not measured . The 256-row file after its pre-swizzled offsets (S8):

```python
if FULL_TILE:
    gl.amd.cdna4.async_copy.buffer_load_to_shared(key_copy, key_ptr, key_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(value_copy, value_ptr, value_offsets)
else:
    ...
gl.amd.cdna4.async_copy.commit_group()
# At this 256-row specialization, ISA permits eight V copies outstanding; not a dynamic overlap measurement.
gl.amd.cdna4.async_copy.wait_group(1)
```

  The partial tile passes the mask `valid[:, None]` and no `other`: masked lanes read 0 with no branch. `other` adds a
  branch per copy and, by build, a store of `other` (S33): the MI355X image's Triton 3.8.0 made none, so masked rows
  kept stale LDS (`gluon/memory.md` §4). The copy needs 128 or 32 bits per lane and coalesced writes; `wait_group(n)`
  returns when at most n committed groups of this wave are outstanding (S34); Membar puts an `s_barrier` after it unless
  a barrier comes first (S35; a 4-warp stand-in with its own `gl.barrier()` after the wait still got one; the 8-row file also had one barrier after `wait_group(1)` in the compiled launch; `gluon/memory.md` §8, §11).
- **Pre-swizzle the global side, copy into a linear alias.** A direct-to-LDS copy writes LDS lane by lane in order
  (`../hardware/gfx950/memory_and_sync.md` §3); into a `SwizzledSharedLayout` with `max_phase` above 1, Triton swizzles
  the source instead, one `ds_bpermute_b32` per copy (S33; compiled at `max_phase` 8). The 128- and 256-row files and
  the DSA 8-head files XOR the global column with the consumer's swizzle, copy into a linear alias and restate the
  16-byte runs with `gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])` (S8, S11); the 8-row file pays the
  shuffles (S31). Code and cost: `gluon/memory.md` §5, §9; the swizzle to match: `gluon/layouts.md` §4. Untimed.
- **Read async-copied tiles with a plain `.load()`.** The 128- and 256-row files (Q, K; S8) and the DSA 8-head files
  (S11) use `gl.amd.cdna4.async_copy.load_shared_relaxed`, whose tag makes Membar drop the barrier between that read and
  any async copy into the same buffer, in either order (S34, S36). Compiled, a straight-line kernel copying K and V as
  two groups and reading K after `wait_group(1)` gets the same `s_waitcnt vmcnt(4)`, `s_barrier` and instructions with a
  plain `.load()`. With the relaxed read, the next copy into a buffer relies on the barrier after `wait_group(0)` (S8):
  a K prefetch placed above that wait compiles ahead of the K reads with no `s_barrier`, so it can overwrite K while
  other waves still read it (inferred; `gluon/memory.md` §11).
- **V through padded LDS.** The 16–64-row files store the gathered V `[TILE, 256]` in the padded layout that
  `gl.amd.cdna4.compute_efficient_padded_shared_layout(b_layout, [TILE, 256], gl.float8e4nv, is_k_contig=False)`
  returns, and read it back as the P·V operand (S37; a stand-in matched torch ). Every producer that
  stages V in an LDS tile (all DSA producers, the Qwen3.5 ones from 8 rows), padded or swizzled, reads it with gfx950's
  transposing `ds_read_b64_tr_b8`, 16 to 80 per producer ; the instruction is in `../attention/lds.md` §1.
- **Cache bits are chosen per Point.** Qwen3.5 loads V `.cg` at 32 and 64 rows, K `.cg` at 64 (S29); the 64-head DSA
  producers store partials `.cs`, read `.cg` by the 8- to 16-token merges and unhinted at 24 and 32 tokens (S17, S6,
  S16); the 8-head files copy the RoPE panel `.cv`, store any partials `.wt` (S11). Without V's `.cg` at 32 rows or K's
  at 64 the call ran 0.6–1.9% faster : test them at your Point (`gluon/memory.md` §6, `tuning.md` §11).

## 9. Qwen3.5: the gate and fp8 group quantization in the merge launch

The finish merges the splits of a (row, head), then applies the gate and quantizes each group of 128 channels while
the merged values are in registers. The 16-row file's epilogue (S37):

```python
attention = attention.to(gl.bfloat16).to(gl.float32)
offset = row * HEADS * 256 + group * 128 + d
gate = gl.load(Gate + offset).to(gl.float32)
sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
gated = (attention * sigmoid).to(gl.bfloat16)
gl.store(Gated + offset, gated)
values = gated.to(gl.float32)
amax = gl.max(gl.abs(values), 0)
quant_scale = gl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
quant = gl.clamp(gl.div_rn(values, quant_scale), -448.0, 448.0)
gl.store(Quantized + offset, quant)
gl.store(Scales + row * HEADS * 2 + group, quant_scale)
```

- **Give one wave whole groups,** so the amax reduces in registers: in one wave it takes no LDS or `s_barrier`, over two
  waves an LDS round trip and an `s_barrier` (compiled); the finishes map groups per Point (S29, S8).
- **Keep the reference's rounding points.** The reference rounds the attention output and the gated product to bf16
  (S38). Some sources additionally round sigmoid to bf16, unlike the reference; this is a numerical
  approximation, not an exact rewrite. Removing output/product rounding can change fp8 codes even when
  the loose output bound still holds. Quantization forms
  (`gl.div_rn` here, `values * (1 / scale)` at 128 and 256 rows, S32), the clamp and the contract: `numerics.md` §2.
- **Drop the bf16 `Gated` store when you copy this.** Every finish writes it, and `run` discards it (S39).
- **Source order is not instruction order.** Independent gate/statistics loads can be moved by the
  compiler; inspect the actual finish ISA before attributing a scheduling gain (S8).

## 10. Precision of QK, P·V and the partials

| Kernels | QK | P for P·V | Partial numerators |
|---|---|---|---|
| DSA 64 heads, 12–32 tokens | `mfma_scaled` e4m3 × e4m3, unit scales, `[32, 32, 64]` | bf16 (fp16 at 12 tokens) | bf16, statistics fp32 |
| DSA 64 heads, 8 tokens | `mfma_scaled` `[16, 16, 128]`, RoPE padded to K = 128 | bf16 hi + lo: 16 heads' hi over their lo on one `[32, 32, 16]` MFMA | fp32 |
| DSA 8 heads | exact bf16 copy of q × fp8 K upcast to bf16, `[16, 16, 32]` | bf16 | bf16 (none at 256 tokens) |
| Qwen3.5, 1–2 rows | fp32 multiply-adds | fp32 | fp32 |
| Qwen3.5, 4 rows | bf16 MFMA, K upcast from fp8 | fp16, V upcast to fp16 | fp32 |
| Qwen3.5, 8 rows | bf16 MFMA | bf16 hi + lo in the 8 spare rows of the 16-row MFMA | fp32 |
| Qwen3.5, 16–256 rows | bf16 MFMA | bf16 | bf16, statistics fp32 |

The compiled instruction forms are DSA 64 heads at 12–32 tokens scaled
fp8 `32x32x64` and bf16 `32x32x16` (f16 at 12), 8 tokens scaled `16x16x128` and bf16 `32x32x16`; 8 heads bf16 `16x16x32`
only; Qwen3.5 none at 1–2 rows, bf16 `16x16x32` from 4 (f16 too at 4). Partial dtypes, hi + lo P: from the sources.

- **QK product precision.** Finite e4m3 × e4m3 products are representable in fp32. A bf16 × bf16 product is exact in fp32 only when its exponent lies in fp32 range and it is representable without subnormal rounding; accumulation can still round,
  and e4m3 values convert exactly to bf16 and fp16. With no scales, `mfma_scaled` fills unit E8M0 scales (S40), one
  `0x7F` register for both: the 64-head DSA QK is a plain fp8 product, `sm_scale` applied to the fp32 scores after it
  (S10). The 8-token file pads the 64 RoPE channels to the instruction's K = 128 (S17):
  a K = 64 operand with `k_width` 32 returned acc / 2 + a·b (`gluon/mfma.md` §10). `allow_flush_denorm=True` (S16)
  changes no instruction of the 24-token producer or merge (compiled with Triton 3.8.0 for gfx950; `numerics.md` §6).
- **Choose P against the bound.** The DSA and Qwen3.5 contracts permit lossy baseline outputs (S9, S13),
  but every precision change must satisfy its output and scale bounds. fp16 P keeps 11 significant bits
  for P ≥ 2^−14 with V exact in fp16; bf16 hi + lo about 16  (below; `numerics.md` §5).
- **Hi + lo P: the residual as extra MFMA rows.** Split each probability into a bf16 high part and a bf16 residual,
  stack them as row pairs, run one P·V MFMA on twice the head rows and add each pair at the end. Compiles with Triton
  3.8.0 for gfx950 (8 heads, 64 keys: eight `v_mfma_f32_16x16x32_bf16`, as many as a 16-row bf16 P·V):

```python
hi = p.to(gl.bfloat16)
lo = (p - hi.to(gl.float32)).to(gl.bfloat16)
pair = gl.reshape(gl.permute(gl.join(hi, lo), (0, 2, 1)), (2 * HEADS, KEYS))  # row 2h: hi, 2h + 1: lo
acc = gl.amd.cdna4.mfma(gl.convert_layout(pair, gl.DotOperandLayout(0, mma, 8)), v, acc)
out = gl.sum(gl.reshape(acc, (HEADS, 2, 256)), axis=1)  # hi·V + lo·V
```

  The extra rows cost no instruction only where they would be padding: 8 heads on the 16-row MFMA. The 8-row file loads
  query row r from head r // 2, keeps rows 2h and 2h + 1 in one thread (non-transposed MFMA), splits off the duplicate
  score row, runs the softmax once per head (§12), rescales both rows of a pair and adds the pairs before the store
  (S31). The 8-token DSA file stacks 16 heads' hi over their lo on `[32, 32, 16]` (S17): as many instructions as the
  `[16, 16, 32]` its 16 heads would fill, at twice the MACs each (`../hardware/gfx950/matrix_core.md`).
- **Partials in bf16 halve the numerator traffic** at one more rounding; the statistics stay fp32 and the finish widens the
  partials before its fp32 sums (S37); a stand-in so built about doubled the error against float64 .

## 11. Qwen3.5: the prologue

The first launch RMS-normalizes Q and K, applies mRoPE, writes K and V into the e4m3 pools and copies the gate, in
one-wave programs that each own one job of a row (S41). The 32-row file's head job begins:

```python
variance = gl.sum(source * source, axis=0) / 256.0
normalized = (
    source * gl.rsqrt(variance + eps) * (1.0 + weight)
).to(gl.bfloat16)
normalized_float = normalized.to(gl.float32)

half: gl.constexpr = rotary_dim // 2
partner = gl.gather(normalized_float, offsets ^ half, axis=0)
```

- **A head per wave or sub-wave**, so the RMS sum stays in the wave: `BlockedLayout([1], [64], [1], [0])` for one head
  (S41), threads per warp `[heads_per_wave, 64 // heads_per_wave]` for 2 or 4 (S30).
- **Long jobs on low program ids:** the source schedules longer cache-scatter jobs before query-only jobs
  (S42, S32). This is a candidate to reduce a queued-grid tail, not a measured Qwen3.5 scheduling gain.
  At 8 rows four programs per token each recompute the whole K RMS but write only their quarter; only the
  first quarter rotates (S42).

- **NeoX partner by XOR:** channel c < 64 pairs with c ^ 32. A 64-value one-wave gather compiles to one
  `ds_bpermute_b32`; the 256-value normalized-head gather in the 32-row prologue compiles to four, with no LDS
  (`gluon/wave_ops.md` §5). `gl.gather` needs an index of the source's rank and returns the index's layout (S43): a
  2-D source takes `gl.broadcast((columns ^ 32)[None, :], source)[0]` (S44). One 64-lane load of 32 cosines then 32
  sines and the same gather give every lane both (S42). With 4 heads per wave the 256-row file pairs the halves in
  registers with `gl.amd.slice`, `split` and `join` (S30); `gl.amd.slice` takes whole CTA tiles only (S45).
- **mRoPE axis from constants.** Frequency f reads `positions[1]` when `f % 3 == 1 and f < 33`, `positions[2]` when
  `f % 3 == 2 and f < 30`, else `positions[0]` (S13). Encodings include two `uint32` masks the host builds from
  `mrope_section` (S41), the same masks as Python ints, written signed as 0x92492492 does not fit an int32 (S44), and a
  `uint64` table of two-bit axis ids (S30); each gives the rule, and every file's key cache follows it.
- **The contract's rounding points.** The RMSNorm result rounds to bf16 before RoPE, the 192 unrotated channels included
  (S30, S38). The cache value is `gl.div_rn(x, scale)` rounded to bf16, which the fp8 store then converts (S30); with
  the unit scales `run` passes, the division is exact (S39), and every file's caches equal that chain bitwise: without
  one bf16 rounding or the other, 948 or 1028 of 130,816 key bytes differ.
- **Prove every int32 index intermediate and final valid index, not just buffer size.** A large bf16/fp32
  allocation can still have legal int32 element indices; an int8 `row * stride` above 2^31 can wrap and read
  the wrong mapped location. Buffer ops additionally impose their descriptor's byte range (§5).
- **Drop dead outputs when adapting a source:** bf16 K and some V copies are written but never read (S39, S41, S30).

## 12. Softmax across waves: the reductions and the P hand-off

With the head rows in one MFMA tile the producers spread a score tile's keys over the warps: `warps_per_cta=[1, 4]` on
Qwen3.5's `[16, TILE]` tile (S1), `[HEAD_TILE // 32, 256 // HEAD_TILE]` over 8 warps in DSA (S25). A row max or sum then
crosses waves through LDS and `s_barrier`. Compiled with Triton 3.8.0 for gfx950, on a `[16, 64]` fp32 tile in the
`[16, 16, 32]` MFMA layout, row max and sum take 2 LDS writes, 2 reads and 3 `s_barrier` with `[1, 4]` warps and none
with `[4, 1]`, 4 `s_barrier` per tile in an online-softmax loop; DSA's `[32, 256]` over `[1, 8]` warps (`HEAD_TILE` 32)
takes 7. There the bf16 P conversion to the P·V operand is one more round trip: a `ds_write_b64`, an `s_barrier` and two
`ds_read2st64_b64` (2 KiB); DSA stages P through its own buffer (S20). The files cut the crossings three ways, none
timed against plain `gl.max` and `gl.sum`:

- **Defer the denominator** (128 and 256 rows): carry the fp32 terms,
  `denominator_terms * alpha[:, None] + probabilities`, sum them once after the loop: only the max crosses waves (S8).
- **Move the softmax into a wave** (8 rows): convert the scores to `BlockedLayout([1, 1], [2, 32], [4, 1], [1, 0])` over
  `[8, TILE]`, a head per half-wave (S31). Compiled from a blocked `[8, 64]` stand-in, that conversion is the one LDS
  round trip (2 KiB, one `s_barrier`); max and sum use DPP and `v_permlane16_swap`.
- **Reduce in two levels by hand** (DSA): each warp reduces its keys in registers and writes its partial into a 2 KiB
  fp32 scratch whose halves hold max and sum; a fold layout reads them back and finishes in registers (S46): at
  `HEAD_TILE` 32, 2 `s_barrier` against 7 for `gl.max` and `gl.sum` (compiled).

## Evidence

- S1 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/d9cbde237fd34de9.json#attention.py:59 one producer tile: QK (:59-62), scale and mask (:63-66), running max and rescale factor (:67-74), rescale of sum and numerator (:79-83), V through padded LDS and P·V (:88-99); keys on the warps (:28-31); "Stage 0 issues V before QK; stages 1/2 bracket the sum reduction" (:54); max and sum as `SliceLayout(1, mma)` vectors (:136-138); a split with keys loads Q, rows past 8 masked to 0 (:139-146), peels its first tile (:147-158) and deals itself the rest cyclically, `FULL_TILE` per tile (:159-171); every split writes its record (:172), the bf16 numerator at 16 bytes per lane (:173-179); records from `torch.empty` (:360-365)
- S2 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:540 the merge: max, weights, fixed-order sums, the guarded reciprocal (:560), 8 channels per lane (:525-527); `_write_partial` (:233) stores the records `[channel // 8][head][channel % 8]` from the accumulator (:250-254), XORing the channel group with `split * 4` at 64 heads (:252-253), which the merge undoes (:546-554)
- S4 aklr/core/contract/semantics/_compare.py:142 `runs_agree`: `bitwise` means every run's output byte-identical to the first's
- S5 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/f5ba496e04f8e989.json#attention.py:152 batch-size tuning only, capacity not a loop bound; 64-key tiles and 32 splits (:153); grid `(rows, heads, splits)` (:157)
- S6 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/1148d1c8e30397f7.json#kernel.py:240 `BALANCED_GATHER` layout, set by `rows == 12` at the launch (:574); records of 544 channels at ≤ 16 tokens (:549); partials stored `.cs` (:62-64); merge loads `.cg` at ≤ 16 tokens (:595)
- S7 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/f60f22e6f14448c6.json#kernel.py:343 statistics stored after channel 512 inside the record when `PACK_STATS` (:343-344), on at 16 tokens with 544-channel records (:683, :691); `stats = partial.view(torch.float32)` (:716)
- S8 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/dfb1ab1cdd5bf263.json#attention.py:46 pre-swizzled K and V offsets into linear aliases (:46-60); the copies and commit groups (:67-80); `wait_group(1)`, `wait_group(0)` (:82, :116-117); `load_shared_relaxed` of Q and K (:84-89); the deferred denominator (:108-112, :247); the swizzled buffers (:172-181); one contiguous chunk per split (:184-187); 4 heads interleaved in 32-column chunks (:231-239, :281-288); Q and the gate read as contiguous rows (:159-161, :271-274) at 128 or 256 rows (:329); the finish's layout and early gate (:256, :272); splits by rows (:334)
- S9 aklr/plugins/sglang/layers/dsa_backend/contract.py:62 the pool bound `n_slots * kv_dim <= 2**31 - 2`; a slot −1 is none and a token that reads no row outputs 0 (:42-44), as the reference.py beside it computes: every −1 masked, a token with none 0 (reference.py:46-48); the baseline's fp8 softmax weights, put at about 33 dB in a comment (:100), against `SqnrBound(min_sqnr_db=20.0)` (:102)
- S10 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:413 `sm_scale` applied to the fp32 scores after the MFMA; the producer's `safe_max` (:425), used for P (:426) and the rescale (:436)
- S11 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/ae32cc132679692c.json#kernel.py:276 `INT_MIN` row offsets; the live-tile bitmap from a wave ballot (:270); unmasked panel copies at pre-swizzled column offsets into a linear view (:77-84), the RoPE panel `.cv` (:83); double-buffered panels with a runtime `.index(phase)` (:194); raw `s_waitcnt vmcnt(n)` (:147, used at :161-166); `load_shared_relaxed` reads (:109, :137, :177); `safe_maximum` (:127-131); partials stored `.wt` (:294); `q.to(torch.bfloat16)` in `run` (:409)
- S12 aklr/core/contract/semantics/_distributions.py:175 distinct slots drawn as the top indices of a row of N(0, 1) draws; sequence-length weights exp(clip(N(0, 1), ±2)) (:245)
- S13 aklr/plugins/sglang/models/qwen3_5/contract.py:329 every row attends at least one slot; the pool bound (:332); the mRoPE axis rule (:297-300); the baseline's 21.8–23.1 dB at the schemas' Points (:402); values ≥ 15 dB (:412), scales ≥ 20 dB (:418), `bitwise` (:422)
- S14 aklr/plugins/sglang/layers/dsa_backend/api.py:19 the route on `_forward_triton_decode` for sglang 0.5.22; speculative decoding falls back (:30)
- S15 aklr/plugins/sglang/layers/dsa_backend/baseline.py:14 the baseline, sglang's `triton_sparse_mla_decode_splitk`
- S16 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:584 head tile and splits from the token count; the 3-D producer launch (:593); `allow_flush_denorm=True` (:616, :637); `waves_per_eu=2 if rows >= 24 else 0` (:617); the merge loads partials `.cg` only at ≤ 16 tokens (:634)
- S17 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/0bb2f7cbe49d1be3.json#kernel.py:311 `_launch_geometry`: 16-head groups, 8 splits, `waves_per_eu` 2 at 8 tokens; QK `[16, 16, 128]` (:68) with the RoPE block padded to 128 (:80), masked past 64 (:107); hi and residual P stacked by `join`/`permute`/`reshape` (:483-492) for a `[32, 32, 16]` P·V (:72), split and added (:561-570); `row = linear % ROWS` (:515) on the flat grid (:892); partials stored `.cs` (:606), merged from `.cg` loads (:471); `q.to(torch.bfloat16)` in `run` (:939)
- S18 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/fbe6ddab5dac5958.json#kernel.py:703 one program per token at 256 tokens, no partials or merge (:709-723); `q.to(torch.bfloat16)` in `run` (:762)
- S19 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/43ffd5e3e5b67298.json#kernel.py:118 panel copies masked with `slots[:, None] >= 0` (:122, :130); successor panels as separate allocations (:227-233); 64-key blocks and 16 splits at 32 tokens (:375-376); `q.to(torch.bfloat16)` in `run` (:416)
- S20 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:316 `cache_shared` allocated only when `TILES == 1`, two `[256, 256]` panels otherwise (:309); read by the 64-head path (:376); P allocated after the key reductions (:450; reductions at :418, :430)
- S21 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2f7000a8186a8ae8.json#kernel.py:256 `_panel_products_refill`: per 64-key slab, P and V loads, the next tile's gathers, MFMAs, `gl.barrier`, stores (:277-289); the two-tile loop (:413) and its rescale (:502-506); called for tile 0 when `REFILL` (:536-551), on at 32 tokens only (:729); `waves_per_eu` 1 at 32 tokens (:717), passed to the producer at :781
- S22 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:468 `waves_per_eu=N` becomes `"amdgpu-waves-per-eu"="N,N"`
- S23 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:283 a masked lane's byte offset becomes 0x80000000 (:280); the descriptor's `num_records` is INT_MAX − 1 bytes (:104)
- S24 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:337 the gather of contiguous slot ranges: slots, `max(slot, 0) * WIDTH`, the 16-byte `buffer_load`, `cache_shared.store` (:358); the load layout (:291)
- S25 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:306 `direct_rope_layout`; `gl.convert_layout(rope, score_rhs)` (:373); the tile read as the QK operand through `.permute((1, 0))` (:119) and as V (:134); the score layout over the warps (:296-298)
- S26 aklr/plugins/sglang/models/qwen3_5/api.py:250 the route on `Qwen3_5AttentionDecoderLayer.self_attention`, HIP decode only (:258); the fallbacks (:297); `o_proj` on the op's fp8 values and scales (:294)
- S27 aklr/plugins/sglang/models/qwen3_5/baseline.py:278 the baseline: QK norm and gate, mRoPE, the KV write, `paged_attention_ragged` (:344), `fused_sigmoid_mul` (:365), the per-1×128 quantization (:366)
- S28 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/3d0f64ffff6790c8.json#attention.py:110 the `[16, 16, 32]` MFMA layout; 16 query rows for 8 heads (:114-117); conversions of K, P and V to the operand layouts (:139, :154-155), no LDS allocation; fp16 P and V (:148-157)
- S29 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/d9cbde237fd34de9.json#attention.py:358 splits and tile by rows; the grid (:367); `head_tile = 16 if SPLITS == 16 else HEADS` (:116); K `.cg` at 64 rows, V `.cg` from 32 (:372-373); `QUERY_GUARD` and the V stages by rows (:375-376); `waves_per_eu=2 if rows != 64 else 0` (:379); one wave per group in the 32- and 64-row finishes (:248, :291)
- S30 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/dfb1ab1cdd5bf263.json#prologue.py:19 RoPE by `gl.amd.slice`/`split`/`join` (:19-44); XOR gather (:47-50); `uint64` axis table (:66-77); bf16 before RoPE (:129-132); cache rounding (:139-141); dead K store (:171); 2 or 4 query heads per wave (:189-192); Q and the gate interleaved as `uint32` pairs below 256 rows (:217-223) and returned as strided views (:259-264); "Narrow only when every indexed buffer fits in signed 32-bit elements" (:251-257)
- S31 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/10df1ae2dc9d3c69.json#attention.py:126 the K copy without `other`, into the swizzled tile of :104; the V copy (:140); `wait_group(1)` and `gl.barrier` (:152-153); the slot prefetch behind the V copy (:26, :143-150); the non-transposed MFMA (:44-49); query row r from head r // 2 (:63-65); the half-wave softmax layout (:84-87) and the duplicate row split off (:185-187); the rescale on both rows of a pair (:207-210); hi + lo P (:231-236); pairs added before the store (:256-258)
- S32 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/14e91857b4038e02.json#attention.py:311 the group scale and `values * (1 / scale)` in the 128-row finish; `waves_per_eu=3 if small_batch else 0` (:358); its prologue.py: "Issue the longer cache-scatter programs first in the single launch." (:71), Q and the gate allocated contiguous (:190-191)
- S33 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:789 a swizzled direct-to-LDS destination makes the lowering swizzle the source, with `shuffleIdx` across lanes (:419); a masked copy has no branch only without `other` and with warp-uniform predicates (:848); with `other`, a branch (:859) and a store of `other` into LDS (:870)
- S34 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:71 `buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="")`: 128 or 32 bits per lane, coalesced writes, a swizzle within a warp (:91-94); `commit_group` (:127); `wait_group(num_outstanding)`, until at most that many committed groups are outstanding (:137-140); `load_shared_relaxed` (:151) tags its load `ttg.amdg.syncedViaAsyncWait` (:164)
- S35 triton-lang/triton@c01b677:lib/Analysis/Membar.cpp:292 a barrier after an async wait unless a sync point follows before the next memory access
- S36 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp:11 Membar drops the dependency between an async copy and a local load synced via an async wait on the same buffer, in either order
- S37 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/6dd321aac1b1d85a.json#attention.py:195 the gate and group-quantization epilogue; V through padded LDS (:87-94); bf16 partials widened in the finish (:239)
- S38 aklr/plugins/sglang/models/qwen3_5/reference.py:283 the attention output rounded to bf16; the gated product, the sigmoid unrounded (:284); the group scale (:287); RMSNorm rounded to bf16 before RoPE, the unrotated channels included (:228-234)
- S39 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/6dd321aac1b1d85a.json#kernel.py:58 `run` calls the attention without using its returned gated tensor; it keeps only Q and the gate of the prologue's outputs (:43) and passes unit KV scales (:19-22, :54-55)
- S40 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_ops.py:69 `mfma_scaled` without scales fills unit E8M0 scales (0x7F)
- S41 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/d9cbde237fd34de9.json#prologue.py:121 the prologue's task grid of K, V and query-head programs; the launch with `num_warps=1` (:191); one head per wave (:19); RMSNorm with `1 + w`, rounded to bf16 (:31-35); the partner by `gl.gather` at `offsets ^ half` (:37-38); the mRoPE axis from two `uint32` masks (:43-51) built on the host from `mrope_section` (:184-190); a V copy no caller reads (:99)
- S42 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/10df1ae2dc9d3c69.json#prologue.py:106 four K/V programs per token first (:106-110); cos and sin from one 64-wide load, halves swapped by `gl.gather` (:22-26); each quarter redoes the RMS, writes its 64 channels, only part 0 rotates (:66-81); `div_rn` then bf16 for the cache (:85-87)
- S43 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:559 `gather`: equal ranks (:566), result in the index's layout (:579)
- S44 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/42a24a8f5e63bd0c.json#prologue.py:20 broadcast XOR index and `gl.gather` (:20-21); signed mRoPE masks (:26-29)
- S45 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/slice.py:12 `gl.amd.slice`: register-only, extent and offsets aligned to the layout's CTA tiling (:17-18)
- S46 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#decode_key_reduce.py:9 `_key_reduce`: in-warp reduction (:11-22), per-warp partials into the max or the sum half of the scratch (:23-24), fold layout and final reduction (:25-42), conversion to the MFMA row layout (:43-46); the 2 KiB scratch at kernel.py:326
