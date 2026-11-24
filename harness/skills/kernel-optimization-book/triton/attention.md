---
title: "Attention with one query token per sequence in Gluon on gfx950: split-KV, DSA sparse MLA and Qwen3.5 gated attention"
kind: topic
ops: [attention, mla]
dtypes: [bf16, fp8]
regimes: ["one query token per sequence", split-kv, sparse, gqa]
gens: [gfx950]
triton: 3.8.0
---

# Attention with one query token per sequence in Gluon (gfx950)

This page covers two attention ops with one query token per sequence, written as Gluon kernels for MI355X, 17
kernels in all, one per batch size: DSA sparse MLA (8 kernels), where each token attends the
2048 fp8 KV rows its top-k slots name, and Qwen3.5 gated attention (9
kernels), where 8 query heads share one fp8 KV head and the output gate and fp8 group quantization are fused in. All but
the 256-token 8-head DSA kernel split each token's keys over programs, write per-split records to scratch and merge them
in a later launch. The 17 belong to the library (154 Triton and Gluon kernels for gfx950: [kernels/kernel_library.md](kernels/kernel_library.md) §1). Not covered here, from the library after its growth: Qwen3-Next gated attention (31 kernels) and DSA kernels at 16 heads (8) and at 8 heads (9 more); that page lists them. Qwen3.5's K and V staging, finish and prologue (§8, §9, §11) are on [attention/gated_attention_launches.md](attention/gated_attention_launches.md).

## Route here when

- You write or tune a Triton or Gluon attention kernel with one query token per sequence: gathered
  (sparse) MLA, paged GQA, or attention with a fused prologue or epilogue.
- You choose the split count, the head tile, the producer grid or the merge of a split-KV attention; you gather KV rows
  by slot and stage them in LDS for the MFMA; you choose the precision of QK, of P·V or of the split partials.

**Not among the library's kernels:** softmax attention over many query rows per sequence (as in prefill or extend), one
query token with more than one KV head per rank, bf16 KV pools; start those from `../attention/INDEX.md` (measured on FlyDSL kernels). Linear attention: `gdn.md`.

**Somewhere else if:** the mechanisms, with numbers measured on FlyDSL kernels → `../attention/INDEX.md`, first
`../attention/split_kv.md`; a launch's cost under each timer → `authoring.md` §7; loads, async copies, waits,
barriers → `gluon/memory.md`; layouts → `gluon/layouts.md`; MFMA → `gluon/mfma.md`; inline assembly →
`gluon/wave_ops.md`; the ops' numerics → `numerics.md`; the ruler ([tuning.md](tuning.md) §1) and the A/B protocol →
`tuning.md`; the library by family → `kernels/kernel_library.md`. Numbers here were measured on these Gluon kernels
unless the sentence says otherwise; a speedup is the ruler's stored value for one size against the op's baseline
([tuning.md](tuning.md) §1).

## 1. Decisions at a glance

| You need | What these one-query-token attention kernels do | § |
|---|---|---|
| a split-KV attention, bitwise reproducible and safe under graph replay | a producer writes (numerator, max, denominator) per split, an empty split a neutral one; a later launch merges them in a fixed order; no atomics in the 17 kernels, each byte-identical over 40 runs; replayed graphs on new slots and lengths give the eager result (run on the 24-token DSA and 32-row Qwen3.5 kernels) | 2 |
| a producer loop | the first tile peeled (one `v_exp_f32` fewer in Qwen3.5's 32-row tile, 4 against 5), the length masks only in the partial tile's variant (12 `v_cndmask` in the full one against 60), the running max guarded where tiles can be empty (DSA) | 2, 7 |
| a producer grid | measure neighbours within each size: DSA's unmodified 192/256/256 programs beat modified 96/128/512 at 12/16/32 tokens, but splits also change tiles/merge and other knobs; Qwen's 512-program rule and two-tile floor are not universal optima, with faster doubled splits at 4/8/64/128/256 rows; residency causality unmeasured *(field, E18)* | 3, 7 |
| a head tile when a row serves many heads | choose it with split count: 64 heads were faster at 24 tokens; widening lost at 12/16 tokens (at 16 both arms the 24-token kernel), and 32 heads were faster at 32 tokens in one session; grid, the P·V mode and refill confound the comparisons, fewer KV reads are not a measured cause *(field, E5, E18)* | 4 |
| to gather rows by slot | 32-bit `buffer_load` offsets reach every row of a pool within 2^31 − 2 bytes, as every pool these kernels were checked on is; 16 bytes per lane, masked lanes read 0 | 5 |
| a gathered tile in swizzled LDS through async copies | XOR the global column offsets with the consumer's swizzle, copy into a linear alias of the buffer | 8: [launches](attention/gated_attention_launches.md) §1 |
| GQA with one KV head | the kernels use per-head programs at 1–2 rows and combine 8 heads in MFMA at larger rows; same-kernel comparisons favor MFMA at the measured 4- and 8-row sizes, not a universal crossover; 2-row branch and whole-kernel replacement comparisons disagree, and the small 1-row difference is unresolved; per-size ratios and run counts in §6 *(field, E2, E16)* | 6 |
| fp8 QK | the scaled MFMA with unit scales on fp8 q (DSA 64 heads); the bf16 MFMA on K upcast from fp8, exact for all 254 finite e4m3 codes, against bf16 q (Qwen3.5 from 4 rows; DSA 8 heads, after an exact cast in the host code) | 10 |
| P·V precision | bf16 P; fp16 P with exact fp16 V for 11 bits where P ≥ 2^−14; bf16 hi + lo P for about 16 | 10 |
| work after attention (gate, quantization) | in the merge launch, where the merged row is in registers | 9: [launches](attention/gated_attention_launches.md) §2 |
| Q/K norm, mRoPE and the cache write before attention | one launch of one-wave programs, cache jobs on the lowest ids, the RoPE partner by an XOR `gl.gather` (`ds_bpermute`, no LDS), constexpr mRoPE masks (as the 32-row kernel; untimed) | 11: [launches](attention/gated_attention_launches.md) §3 |
| a softmax over keys spread across warps | defer the denominator, move the softmax into a wave, or reduce in two levels by hand: fewer `s_barrier` per tile, compiled, none timed | 12 |

## 2. Split the keys, merge in a later launch

Each producer program owns a key range (a split) of one token for a group of heads, runs an online softmax over it,
and writes a record: the unnormalized numerator Σ exp(s − m)·v, the running max m and the denominator Σ exp(s − m).
One tile's step in the 32-row gated-attention kernel: max and sum are `SliceLayout(1, mma)` vectors, one value per
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
    # ...
    if FIRST:
        running_sum = gl.sum(probability, 1)
    else:
        running_sum = running_sum * correction + gl.sum(probability, 1)
        numerator = numerator * correction[:, None]
```

P·V follows, `gl.amd.cdna4.mfma(gl.convert_layout(probability.to(gl.bfloat16), a_layout), value, numerator)`, V read
through padded LDS ([attention/gated_attention_launches.md](attention/gated_attention_launches.md) §1); K and V are gathered by slot above (§5); these `gl.max` and `gl.sum` cross waves (§12). A
later launch reads the records of all splits of a (token, head), weighs split s by exp(m_s − M), M the largest m_s,
sums in a fixed order and divides once, as the 24-token DSA merge does:

```python
    maximum = gl.max(maximums, 0)
    safe_max = gl.where(maximum == -float("inf"), 0, maximum)
    correction = _softmax_exp(maximums - safe_max[None, :])
    denominator = gl.sum(denominators * correction, 0)
    # ...
    numerator = gl.sum(partials * correction[:, :, None], 0)
    inverse_denominator = 1.0 / gl.where(denominator > 0, denominator, 1)
```

Merged so in float64, the 32-row Qwen3.5 kernel's records give the softmax over all keys with maximum per-head relative L2 error 5.042e-4 in the tested 2049–4096-key rows
*(field, E13)*. `_softmax_exp(x)` is `gl.exp2(x * 1.4426950408889634)`: 2 VALU against 9 for `gl.exp(x)` with one value
per lane, compiled with Triton 3.8.0 for gfx950 (`numerics.md` §6). The gated-attention kernels use `gl.exp`; no size compares the two.

- **Merge in a later launch.** None of the 17 kernels (8 DSA, 9 Qwen3.5) makes one workgroup wait for another or
  reduces in the last workgroup to arrive (why: `../attention/split_kv.md` §2, §3). What a separate merge launch
  costs depends on who times it (`authoring.md` §7): the ruler's timer, which sums the device durations of the call's
  kernels, gaps between graph nodes excluded, reads an added
  launch at no less than its floor (`../pitfalls/profiled_timer_floor.md`), about 4.05 µs for a trivial
  kernel, measured under Kineto; a bare replayed CUDA graph pays 1.77–1.84 µs of wall time per extra chained
  kernel *(field, E9)*. Put follow-on work into the merge ([attention/gated_attention_launches.md](attention/gated_attention_launches.md) §2).
- **No atomics.** Both ops are held to `bitwise` (`numerics.md` §1): every run's output byte-identical to the first's.
  None of the 17 kernels has an atomic, and 40 runs of each at its size were byte-identical *(field, E15)*. The
  accuracy bound is separate and loose (§10).
- **Every split writes a record, empty or not:** max −inf, denominator 0 and numerator 0 for a split with no keys, which
  the merge weighs by exp(−inf) = 0. Scratch from `torch.empty` then needs no zeroing, and a graph replay with
  other lengths reads no stale record, as run on the 32-row Qwen3.5 and 24-token DSA kernels *(field, E13)*; lengths and
  slots are read on the device, the configuration from the batch shape. The 32-row Qwen3.5 producer:

  ```python
      running_max = gl.full((head_tile,), -float("inf"), gl.float32, gl.SliceLayout(1, mma))
      running_sum = gl.full((head_tile,), 0, gl.float32, gl.SliceLayout(1, mma))
      numerator = gl.full((head_tile, 256), 0, gl.float32, mma)
      if split * TILE < length:
          # ... a split with keys runs its tiles
      # Every split publishes fresh neutral records, including inactive splits.
      numerator = gl.convert_layout(numerator.to(gl.bfloat16), query_layout)
      record = (row * HEADS + h) * SPLITS + split
      # ... numerator stored at Partial + record[:, None] * 256 + d[None, :], rows past HEADS masked
      h_stats = gl.arange(0, head_tile, gl.SliceLayout(1, mma))
      stat_record = (row * HEADS + h_stats) * SPLITS + split
      gl.store(Stats + stat_record * 2, running_max, h_stats < HEADS)
      gl.store(Stats + stat_record * 2 + 1, running_sum, h_stats < HEADS)
  ```
- **Lay the records out for both launches.** Simplest, Qwen3.5's: numerators `[rows, heads, splits, 256]` bf16 and
  statistics `[rows, heads, splits, 2]` fp32, the numerator converted to
  `BlockedLayout([1, 8], [8, 8], [1, gl.num_warps()], [1, 0])` so each lane stores 16 bytes (`query_layout` above). The 64-head DSA
  producers store straight from the transposed 32 × 32 accumulator in `[channel // 8][head][channel % 8]` order, read
  back 8 channels per lane, and at 64 heads XOR the channel group with `split * 4`, for a reason the source does not
  state:

  ```python
      matrix: gl.constexpr = gl.amd.AMDMFMALayout(
          version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 8]
      )
      local_heads = gl.arange(0, HEAD_TILE, gl.SliceLayout(1, matrix))
      channels = START + gl.arange(0, 256, gl.SliceLayout(0, matrix))
      record = (row * (HEADS // HEAD_TILE) + head_block) * SPLITS + split
      groups = channels[None, :] // 8
      if HEAD_TILE == 64:
          groups = groups ^ split * 4
      offsets = (groups * HEAD_TILE + local_heads[:, None]) * 8 + channels[None, :] % 8
      gl.amd.cdna4.buffer_store(
          numerator.to(gl.bfloat16), Partial + record * RECORD_VALUE * HEAD_TILE, offsets, cache=".cs"
      )
  ```

  The 12- and 16-token kernels pad a record to 544 channels, the 16-token one keeping the statistics in
  the padding; the 256-row Qwen3.5 kernel interleaves 4 heads for its 4-wave finish. None is measured.
- **Guard the running max in the producer as well as in the merge.** DSA marks an unused slot −1, and its
  reference outputs 0 for a token that reads no row: a kernel must survive empty tiles, splits and tokens. Every
  DSA producer takes its exponentials against `safe_max = gl.where(m == -float("inf"), 0, m)`, as the 24-token one:

  ```python
          if tile == 0:
              next_max = tile_max
          else:
              next_max = gl.maximum(maximum, tile_max)
          safe_max = gl.where(next_max == -float("inf"), 0, next_max)
          probability = gl.where(valid_scores[None, :], _softmax_exp(scores - safe_max[:, None]), 0)
          # ...
          if tile == 0:
              denominator = tile_sum
          else:
              correction = _softmax_exp(maximum - safe_max)
              denominator = denominator * correction + tile_sum
  ```

  Not this: `next_max` in place of `safe_max` in the exponentials. Without the guard,
  a split whose first two tiles are empty rescales by exp(−inf − (−inf)) = NaN in its token; the merges guard their
  max likewise (all 9 merges, read from the sources); without it a token with no row is NaN, and the 24-token merge
  turns 0/0 into 0 *(field, E13)*. A benchmark whose slots are distinct cannot catch a
  missing guard: no tile is empty, and the 24-token kernel without both
  guards passes its accuracy check there *(field, E13)*. Test against the reference with −1 tails covering whole tiles, a split
  and a token; they catch both. The 32-row Qwen3.5 kernel has no guard and stays within the bounds on 1 to 4096 keys
  *(field, E13)*: every row attends at least one slot, and a split runs only tiles that hold a key.

## 3. DSA sparse MLA, one query token over 2048 gathered rows: op, sizes, grid

The op is what sglang 0.5.22's `DeepseekSparseAttnBackend._forward_triton_decode` computes, speculative decoding
left out. The baseline is sglang's split-K Triton
kernel `triton_sparse_mla_decode_splitk`. q is `[tokens, heads, 576]` fp8 e4m3fn (512 latent channels, then 64
RoPE channels), the pool `[pool rows, 576]` fp8, 2048 slots per token (−1: none), the output `[tokens, heads, 512]`
bf16; there are no scales. Two head counts, 64 and 8, one kernel per token count (not in the table: 16 heads, and more 8-head
token counts with two kernels each at 32, 128 and 256 tokens, none of them with a measured speedup):

| Heads | Tokens | Heads per program × splits | Producer: programs × warps | `waves_per_eu` | LDS (B) | VGPRs | Merge: programs × warps | Speedup over the baseline |
|---|---|---|---|---|---|---|---|---|
| 64 | 8 | 16 × 8 | 256 × 4 | 2 | 147,696 | 150 (22 of them AGPRs) | 512 × 4 | 1.030 |
| 64 | 12 | 32 × 8 | 192 × 8 | – | 156,160 | 88 | 768 × 1 | 1.699 |
| 64 | 16 | 32 × 8 | 256 × 8 | 2 | 156,160 | 94 | 1024 × 1 | 1.840 |
| 64 | 24 | 64 × 8 | 192 × 8 | 2 | 163,840 | 228 | 1536 × 1 | 2.188 |
| 64 | 32 | 32 × 4 (two 256-key tiles each) | 256 × 8 | 1 | 157,696 | 214 | 512 × 4 | 1.966 |
| 8 | 32 | 8 (16 MFMA rows) × 16 | 512 × 4 | – | 49,216 | 154 | 256 × 2 | 1.062 |
| 8 | 128 | 8 × 2 | 256 × 4 | – | 79,872 | 199 | 512 × 2 | 1.196 |
| 8 | 256 | 8, one program per token | 256 × 4 | 1 | 83,968 | 202 | none | 1.050 |

The speedups are the ruler's stored values against that baseline ([tuning.md](tuning.md) §1), its kernel durations
read from Kineto *(field, E1)*; a second run of the ruler read
1.737, 1.839, 2.159 and 1.978 for the 64-head kernels at 12–32 tokens *(field, E1)*. LDS and registers of the compiled producers, one variant each, no spills
*(field, E8)*, VGPRs being `.vgpr_count`, VGPRs and AGPRs together (`kernels/kernel_library.md` §1); grids, warps and
`waves_per_eu` as traced at launch *(field, E15)*; heads per program × splits from the host code. The 8-token kernel and the three 8-head kernels add a
`q.to(torch.bfloat16)` launch in the host code.

- **Measured DSA grids, not an isolated CU-residency rule.** At 12/16/32 tokens the accurate unmodified 64-head kernels' 192/256/256 eight-wave producers beat modified 96/128/512: modified/unmodified call times 1.414/1.456/1.098, paired-log 99% CIs 1.402751–1.425598/1.441523–1.470504/1.094254–1.102277 (one session, 6 counterbalanced rounds per arm, 3 forward and 3 reverse) *(field, E18)*. At 24 tokens, in three earlier runs, 384 same-head producers ran 1.08–1.09× slower than 192 *(field, E2)*. Splits change tiles and merge width; the 16-token variant also changes the P·V mode (64-key panels) and the packed statistics, the 32-token variant the P·V mode (serial 32-key slabs) and the refill. LDS predicts at most one such workgroup per CU, but residency and its timing contribution were not measured, nor were four-wave or Qwen3.5 residency and untested sizes (`../attention/split_kv.md` §4).
- **Do not start a 64-head kernel from the 8-token kernel.** It differs from the 12–32-token kernels in many choices:
  16-head groups of 4 warps, rows async-copied into LDS, bf16 hi + lo P, fp32 partials and the extra `q` cast. It
  measured 1.030 *(field, E1)*; its baseline differs from theirs, and no size attributes the gap to one choice.
- **With 8 heads the MFMA did not bound the 32- and 128-token producers.** Each 576-byte row feeds 8 query rows, about
  30 FLOP per gathered byte (estimated); doubling the QK MFMAs cost 0.6–1.0% at 32 tokens and 4.2–4.7% at 128 (3 runs)
  *(field, E5)*; the gather plausibly bounds them (inferred). The three measured 1.050–1.196 *(field, E1)*. At 256 tokens
  one program per token runs all 32 key tiles, with no partials or merge; at 128 tokens two programs per token and
  a 2-way merge. The 128- and 256-token kernels' staging (async panels per 64 keys, a live-tile bitmap from a wave ballot,
  double-buffered panels, raw `s_waitcnt vmcnt(n)`): `gluon/wave_ops.md` §6, `gluon/memory.md` §11; at 128 tokens
  each raw wait counts the loads issued after its panel, so dropping one leaves a panel in flight (compiled).

## 4. DSA: head tile, split count, `waves_per_eu`

- **The 24-token kernel widens the head tile instead of adding programs.** A 32-head tile reads each gathered row once per
  head block, twice per token with 64 heads. The 24-token kernel puts all 64 heads in one tile, 8 splits: 192 programs, as
  the 32-head rule (4 splits) would, with half the row reads (estimated), and ran 1.08–1.09× faster than that tiling
  *(field, E5)*. It measured DSA's highest speedup, 2.188 (2.159 in the second run) *(field, E1)*, but a ratio across sizes also ranks the baseline
  (`tuning.md` §1), 1.72–1.73 µs per token at 24, 1.37 at 32 (3 runs) *(field, E5)*: measure both tilings.
- **Choose head tile and splits within a size, not from baseline ratios across sizes.** At 24 tokens (3 runs, 4 pairs each) 64 heads took 18.66–18.80 µs against 20.27–20.31 for 32 heads, 1.08–1.09× *(field, E5)*. At the other sizes, one session of 6 counterbalanced rounds per arm (3 forward, 3 reverse), paired-log 99% CIs *(field, E18)*:
  - At 12 tokens widening 32→64 heads also changes 192→96 producers; 64/32 call time = 1.154, CI 1.144109–1.163215.
  - At 16 tokens both arms are the 24-token kernel run at 16 tokens with correctly sized packed statistics, 64 heads on 128 producers against 32 on 256: 64/32 = 1.222 (17.58/14.39 µs), CI 1.213658–1.229784. The 16-token kernel's own 64-head path does not fit: 164,848 B of LDS exceed 160 KiB.
  - At 32 tokens both use 256 producers; 32/64 = 0.9737 (21.67/22.25 µs), inverse paired-log CI 0.970162–0.977209: 32 heads were faster in this session, both order strata agreeing. This also changes the P·V mode and the refill; at 12 and 16 tokens the grid halves too. Fewer KV reads and CU residency are unmeasured mechanisms, not measured causes.
- **Head tile and split count are coupled in the source.** The host code sets both from the token count,
  `head_tile = 64 if rows == 24 and heads == 64 else 32` and `splits = 8 if rows <= 16 or head_tile == 64 else 4`,
  and launches the producer on the grid `(rows, heads // head_tile, splits)`. The 64-head path is written for one
  256-key tile per program: it reads the whole-tile LDS buffer (`cache_shared` in §5's excerpt), which the kernel
  allocates only with one tile per program. Four
  splits at 24 tokens give two tiles per program, and the producer fails to compile
  (`NameError: cache_shared is not defined`) *(field, E6)*, without a GPU too (Triton 3.8.0, gfx950). A second tile
  cannot sit beside the first, which already takes 160 KiB with P (§5): start from the 32-token kernel.
- **Two tiles per program: refill under P·V.** The 32-token kernel holds a tile as two `[256, 256]` LDS panels. During
  tile 0's P·V, per 64-key slab, it loads P and V from LDS, gathers tile 1's rows, runs the MFMAs, and after a
  `gl.barrier()` stores them over the slab; tile 1 then rescales by exp(m_old − m_new). Without it the call moved
  by −0.7% to +0.7% over 3 runs, no measurable gain *(field, E2)*; the other 64-head kernels hold one tile per program.
- **Readers of the same rows on one XCD.** Workgroups go round-robin to the 8 XCDs and their L2s (`tuning.md` §12).
  The 8-token kernel sets `row = program_id % 8`, so a token's programs share an XCD; in the 3-D grids a (token,
  split)'s two head blocks share one XCD at 16 and 32 tokens, 4 apart at 12 (inferred). No size measures it; in a probe at 2048 programs, 8 readers of one 64–512 KB chunk ran 1.78–2.12× faster on one XCD over 3 runs, but 1.31× with 16 KB chunks and 1.54× at 1024 programs (one run) *(field, E12, E20)*.
- **`waves_per_eu` moved the 24-token call by under 1%.** The kernel sets 2; unset, the call took 18.71 µs against
  18.89 µs (−0.96%, speedup ratio 1.008), faster in each of the 4 alternated pairs *(field, E6)*, compiled to 222 VGPRs
  against 228 *(field, E8)*. Its 163,840 B of LDS fit one 8-wave workgroup per CU, so the hint (S22) cannot raise
  residency, only move register allocation (`tuning.md` §8); the 32-token kernel's `waves_per_eu=1` compiles as none.

## 5. DSA: gathering the 2048 rows

- **32-bit offsets over the whole pool.** Every pool these kernels were checked on is within 2^31 − 2 bytes, the range
  of the buffer descriptor Triton 3.8.0 builds (65536 to 524288 rows of 576 bytes at the measured sizes
  *(field, E1)*). Within that range a `gl.amd.cdna4.buffer_load` with 32-bit offsets reaches every row, and a masked
  lane gets an out-of-range offset and reads 0 (S23); a pool that can outgrow it needs a host gate on its byte extent
  and another path above it (`gluon/memory.md` §3). The 64-head kernels clamp a −1 slot to row 0 and mask it; the
  32-token 8-head kernel masks its panel copies (`slots[:, None] >= 0`); the 128- and 256-token 8-head kernels turn
  it into an `INT_MIN` element offset and copy with no mask, which stays out of range only for 1-byte elements (`gluon/memory.md` §4):

  ```python
      row_offsets = gl.where(selected >= 0, selected * KV_ROW, -2147483648).to(gl.int32)
  ```
- **16 bytes per lane, a whole tile in flight.** The 12- to 24-token producers gather each 256-row × 512-byte tile into
  registers as one tensor, in `BlockedLayout([1, 16], [4, 16], [8, 1], [1, 0])` (12 tokens: `[8, 8], [4, 2]`), the
  32-token one in halves and refill slabs; 16 fp8 values per lane in one `buffer_load_dwordx4`, then to LDS:

  ```python
          indices = (split * TILES + tile) * 256 + key_offsets
          slot = gl.load(Slots + row * TOPK + indices)
          valid = slot >= 0
          if CACHE_BUFFER:
              base = gl.maximum(slot, 0) * WIDTH
          # ...
                  latent = gl.amd.cdna4.buffer_load(Cache, base[:, None] + channels[None, :], valid[:, None], 0.0)
          # ...
              cache_shared.store(latent)
  ```

  That is 128 KiB per program, 16 loads of 16 bytes per lane, all issued before the first `ds_write_b128` (24-token producer, compiled for gfx950). A contiguous bf16 stream at 256 programs × 4 warps read 2.24–2.29 → 7.13–7.17 TB/s as the independent 16-byte loads per lane went 1 → 8 (3 runs) *(field, E10)*. A synthetic fp8 gather of distinct random rows ran 3.171× faster with 8 loads per lane than with 1 for 256-byte rows and 2.959× for 576-byte-stride rows reading 512 bytes (256 programs × 4 warps, 12 randomized blocks, paired 95% CIs 3.1698–3.1729 and 2.9568–2.9613; the matched bf16 stream 2.240 → 7.073 TB/s), 1.825× and 1.718× at 512 programs, 1.172× and 1.109× at 1024 *(field, E19)*. These are synthetic probes, not attention or DSA measurements; repeated indices and more than 8 loads per lane were not tested.
- **One LDS tile serves QK and P·V.** In MLA the 512 latent channels of a row are the key (with the 64 RoPE channels)
  and the value: the producer stores the tile once and reads it as the QK operand through a `.permute((1, 0))` view and
  as the P·V operand directly (a stand-in matched torch *(field, E13)*). It loads the 64 RoPE channels in
  `BlockedLayout([32, 1], [2, 32], [HEAD_TILE // 32, 256 // HEAD_TILE], [1, 0])`, where the B operand of the
  `[32, 32, 64]` scaled MFMA wants each element: the `gl.convert_layout` that follows compiles with
  `assert_trivial=True` (Triton 3.8.0, gfx950, both head tiles), with no LDS and no lane shuffle.
- **At the 160 KiB limit, allocate late:** the allocator overlaps buffers whose live ranges do not meet
  (`gluon/memory.md` §10). The 24-token producer keeps the `[256, 512]` fp8 tile (128 KiB) through P·V and allocates
  its `[64, 256]` bf16 P (32 KiB) after the last use of the query staging and reduction buffers: 163,840 B
  *(field, E8)*; allocated before the second key reduction, P takes it to 165,888 B with the 2 KiB scratch (compiled).
- **The ruler's inputs have no index locality.** Each token's 2048 entries are distinct and uniformly sampled from −1 and the pool's slots,
  so the 8-head kernels' live-tile skipping does no work on them. Test the locality levers of
  `../attention/split_kv.md` §7 (measured on FlyDSL and Triton kernels) with production-like indices.

## 6. Qwen3.5 gated attention, one query token, 8 heads per KV head: op, sizes, launches

The op is the body of sglang 0.5.22's `Qwen3_5AttentionDecoderLayer.self_attention` with one query token per
sequence, between sglang's `qkv_proj` and an `o_proj` that takes the op's fp8
values and per-128 scales; the kernels cover KV scales of 1 only. The baseline is six separate calls: the fused QK norm and gate split, mRoPE,
the fp8 KV-cache write, aiter's `paged_attention_ragged`, `fused_sigmoid_mul` and the per-1×128 quantization. 8
query heads share one KV head of 256 channels in fp8 pools; the ruler's inputs give each row 4096 keys on average, with
lengths 54.5× apart in a CPU reconstruction of the workload draw at 256 rows. Three launches:

1. **Prologue**, a task grid of one-wave programs, each owning one job of a row: key/value cache work (separate or
   combined in channel quarters), one or more query heads (RMSNorm, RoPE), or gate copies, cut differently per size ([attention/gated_attention_launches.md](attention/gated_attention_launches.md) §3). It
   ends before attention starts because attention reads the slots it writes.
2. **Partials**, the split-KV producer of the table below.
3. **Finish**: the merge, the gate and the fp8 group quantization ([attention/gated_attention_launches.md](attention/gated_attention_launches.md) §2).

| Rows | Producer | Programs × warps | Splits × tile (keys) | Mean tiles per split (estimated) | LDS (B) | VGPRs | Speedup over the baseline |
|---|---|---|---|---|---|---|---|
| 1 | one program per (row, head, split), fp32 multiply-adds, no MFMA | 256 × 4 | 32 × 64 | 2 | 1,024 | 130 | 2.182 |
| 2 | the same attention code as 1 row | 512 × 4 | 32 × 64 | 2 | 1,024 | 130 | 1.699 |
| 4 | all 8 heads in a 16-row MFMA; K and V gathered into registers; fp16 P·V | 128 × 4 | 32 × 64 | 2 | 32,768 | 182 | 1.671 |
| 8 | async K and V into LDS, no V copy left outstanding at the pre-QK wait *(field, E16)*; bf16 hi + lo P (§10) | 256 × 4 | 32 × 64 | 2 | 34,816 | 125 | 1.803 |
| 16 | K into registers, V through padded LDS; bf16 partials | 512 × 4 | 32 × 64 | 2 | 16,624 | 136 | 1.708 |
| 32 | as 16 rows, with a 16-row head tile | 512 × 4 | 16 × 128 | 2 | 33,264 | 188 | 1.789 |
| 64 | the same attention code as 32 rows; 8-row head tile | 512 × 4 | 8 × 128 | 4 | 33,264 | 192 | 1.401 |
| 128 | async Q, K and V with pre-swizzled offsets; one contiguous chunk per split | 512 × 4 | 4 × 128 | 8 | 69,632 | 147 (63 of them AGPRs) | 1.204 |
| 256 | as 128 rows | 512 × 4 | 2 × 128 | 16 | 71,680 | 172 | 1.066 |

Speedups are the ruler's stored values against that baseline ([tuning.md](tuning.md) §1), as in §3 *(field, E3)*; a second run of the ruler read 2.221, 1.685, 1.668, 1.842, 1.729, 1.785,
1.398, 1.206 and 1.085 *(field, E3)*. The compiled producers: 4 warps each, LDS and VGPRs as listed, no spills, no MFMA
at 1–2 rows *(field, E8)*; programs and splits × tile from the launch code, checked by the launch trace *(field, E15)*. Mean tiles per split =
4096 / (splits × tile). Each kernel has three sources, the host code, the prologue and the
attention; the 4-row one allocates no LDS, its 32 KiB being compiler scratch (compiled), for converting K, V and P
to operand layouts (inferred). The head tile is the rows of the query, score and
accumulator tensors on the `[16, 16, 32]` MFMA, `head_tile = 16 if SPLITS == 16 else HEADS`: 16 pads the 8 heads
with 8 zero rows; 8 leaves copies in the lanes of the missing rows and, in a one-wave stand-in, issues the same
MFMA, load, store and `v_exp_f32` instructions as 16 rows: it saves none (compiled; `gluon/mfma.md` §4).

- **Per-head at 1–2 rows, combined heads in MFMA above: implementation, not a universal optimum.**
  At 4 rows, the 4-row kernel's MFMA attention took 21.59–21.77 µs versus per-head 26.79–26.95, per-head/MFMA time ratio 1.235–1.248 (4 runs, 16 pairs). At 8 rows its branches observed paired ratios 1.975–2.019 (3 runs, 12 pairs; MFMA 27.74–28.11 versus per-head 55.08–55.89 µs).
  At 2 rows the same-kernel branches observed MFMA/per-head 1.025–1.067 (3 runs, 12 pairs), while a replacement MFMA attention source versus the unmodified 2-row kernel observed 0.988–1.005 (2 runs, 8 pairs; merge changes too). At 1 row the earlier comparison observed run-mean MFMA/per-head 0.982–1.001 (4 runs, 16 pairs). All timed GQA arms passed accuracy *(field, E2, E16)*.
  Fixed AB order and cross-process dependence leave mean-difference uncertainty unresolved, especially near zero; neither a general 2–4-row crossover nor a small 1-row advantage is established. Shared-KV traffic and its contribution were not separately measured.
- **Where the gain is.** Up to 32 rows a call reads 2–64 MiB of KV (estimated), and the op runs 3 kernels where the
  baseline's six calls run 7: at the floor of a timer that sums Kineto kernel durations, as the ruler's timer does, about 4 µs per kernel
  (`../pitfalls/profiled_timer_floor.md`), the 4 saved are 16 µs (estimated), 47% of the gain in summed kernel durations at 32 rows and
  86% or more at 1 to 16 *(field, E9, E15)*. At 256 rows × 4096 keys (N=4 runs), the partials kernel's 512 MiB call took
  78.34–79.42 µs (6.76–6.85 TB/s), baseline paged attention 144.53–145.13 µs (3.699–3.715 TB/s), against a flushed
  random-permutation 256-byte-row gather at 5.109–5.400 TB/s. Kernels were profiled unflushed with replacement-drawn
  slots: rates are not like for like and establish no gather bound; 64/128 rows untested here *(field, E14)*.
  The 256-row kernel measured 1.066 (1.085 in the second run) *(field, E3)*. In a Triton toy, one row 55× longer than seven others,
  fixed 4 splits/row (32 programs) took 50.74–51.21 µs versus 6.13–10.56 µs for 62 equal chunks, 4.806–8.356×
  (N=4 replayed-graph runs); the long-row tail was not measured on Qwen3.5 kernels *(field, E14)*.
- **Copy a kernel only at its own size.** Below 256 rows the 256-row kernel's prologue writes Q and the gate interleaved
  as `uint32` pairs and returns strided views, while its attention, which also accepts 128 rows, reads both as
  contiguous rows: at 128 rows it raised no error, its values at −1.3 dB against a 15 dB bound *(field, E13)*.

## 7. Qwen3.5: the split count by rows

From 16 rows the kernels' split count keeps `rows × splits = 512`, nominally two four-wave producers per CU, not a measured residency/optimality rule:
`splits = 32 if rows <= 16 else (16 if rows <= 32 else 8)`, `tile = 128 if rows >= 32 else 64`, grid
`(rows, splits)`. The 128- and 256-row kernels continue the rule with 4 and 2 splits. At 32 rows, against the
unmodified 16 splits with nothing else changed in the sources, in two sessions *(field, E4)*:

| Splits at 32 rows | Programs ÷ CUs | Mean tiles per split (estimated) | Speedup vs baseline | Call time | Speedup ÷ unmodified |
|---|---|---|---|---|---|
| 8 | 1 | 4 | 1.129–1.132 | 69.8–69.9 µs | 0.63 |
| 16 (unmodified) | 2 | 2 | 1.780–1.791 | 44.1–44.3 µs | 1 |
| 32 | 4 | 1 | 1.690–1.693 | 46.6–46.8 µs | 0.95 |

Call time is the three kernels' device time per call. The kernel ties its head tile to the split count, so both
changed arms also ran an 8-row head tile: the table measures the pair, not the split count alone.

- **At 32 rows half and double splits took 1.58× and 1.06× as long**, losing about 37% and 5% of the speedup, not increasing call time by those percentages *(field, E4)*. Splits also change the head tile, tiles per producer and merge records; CU residency and in-flight-load causality were not measured, and a synthetic stream is not a measurement of this attention kernel.
- **Measure both neighbours at each size; neither a 512-program grid nor a two-tile floor is universally optimal.** One session of 6 counterbalanced rounds per arm (3 forward, 3 reverse) *(field, E18)*:
  - Doubling splits at 4/8/64/128/256 rows gave modified/unmodified call-time ratios 0.7863/0.8142/0.9731/0.7818/0.7052; paired-log 99% CIs 0.782904–0.789708/0.804810–0.823643/0.970714–0.975498/0.780215–0.783308/0.704074–0.706232. Both order strata agree; the 4/8-row doubled arms have one tile per split.
  - At 16 rows half/double = 1.441/1.033; double CI 1.018381–1.046948 supports slower, not a guaranteed ≥2% effect. The unmodified arm was the fastest tested. At 32 rows see the table above (half/double 1.58/1.06); no 256-row half and no exhaustive search was run.
  Head-tile changes following the kernel's rule, tiles per split and merge width remain confounders; these measurements neither isolate a CU-grid mechanism nor establish a global optimum.
- **Which tiles a split takes.** The 1- to 64-row kernels deal a row's tiles out in turn, split s taking s, s + SPLITS, …
  (`for tile in range(split + SPLITS, gl.cdiv(length, TILE), SPLITS)` after the peeled first tile), so a row's
  splits differ by at most one tile. The 128- and 256-row kernels give each split one contiguous chunk of
  `gl.cdiv(length, SPLITS * BLOCK) * BLOCK` keys, leaving the last split empty for rows of 1–384, 513–768 and 1025–1152
  keys at 4 splits, of up to 128 at 2; the 64-head DSA producers also give each split a contiguous slot range; no size compares the
  two. An empty split still writes its neutral record, and at 32 and 64 rows skips the Q load.
- **Where V is gathered is a per-size knob, unablated:** the producer's own comment reads "Stage 0 issues V before QK; stages 1/2 bracket the sum
  reduction"; the first tile and the rest use stages 0 and 2 at 16 rows, 1 and 1 at 32, 0 and 0 at 64. The
  8-row kernel loads the next strided tile's slots behind the V copy ([attention/gated_attention_launches.md](attention/gated_attention_launches.md) §1). No attention size compares
  async copies with register gathers; on a 64×64 bf16 tile stream of 4-warp programs at 1024 and 2048 programs (16–32 waves per CU) async copies ran from 4.3% below to 0.9% above register-only loads (1.3–4.3% below at 2048 programs in all 4 runs) *(field, E14)*, but at 256 programs (4 waves per CU) with one or two tiles in flight they ran 15–48% behind *(field, E21)*, and in an 8-token MoE router contiguous register loads and layout conversions in place of async copies changed call time by −0.1% to +1.5% (8.30–8.67 µs across arms, 4 runs) *(field, E14)* (`gluon/memory.md` §8).
- **`waves_per_eu` is a register choice here, not a grid one.** The kernels set 2 at 16 and 32 rows and 3 at 128 rows.
  Compiled with Triton 3.8.0 for gfx950: 188 VGPRs with it and 186 without at 32 rows; at 128 rows 147
  registers, 84 VGPRs and 63 AGPRs, with 113 `v_accvgpr` moves (3039 instructions against 2983), and 164 VGPRs
  without, occupancy 3 for both, though 69,632 B of LDS let only 2 waves per SIMD reside (estimated). Untimed.

## 8. Qwen3.5: K and V into LDS

Where each row count gathers K and V (registers, async copies into a swizzled LDS tile, padded LDS), the commit groups
and what the wait before QK leaves in flight, pre-swizzled offsets, relaxed reads and the cache bits per size:
[attention/gated_attention_launches.md](attention/gated_attention_launches.md) §1.

## 9. Qwen3.5: the gate and fp8 group quantization in the merge launch

The finish launch after the merge (the gate, the fp8 quantization per group of 128 channels, the reference's rounding
points): [attention/gated_attention_launches.md](attention/gated_attention_launches.md) §2.

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

All passed the accuracy check *(field, E1, E3)*. Compiled MFMAs *(field, E8)*: DSA 64 heads at 12–32 tokens scaled
fp8 `32x32x64` and bf16 `32x32x16` (f16 at 12), 8 tokens scaled `16x16x128` and bf16 `32x32x16`; 8 heads bf16 `16x16x32`
only; Qwen3.5 none at 1–2 rows, bf16 `16x16x32` from 4 (f16 too at 4). Partial dtypes, hi + lo P: from the sources.

- **QK product precision.** Finite e4m3 × e4m3 products are representable in fp32. A bf16 × bf16 product is exact in fp32 only when its exponent lies in fp32 range and it is representable without subnormal rounding; accumulation can still round,
  and e4m3 values convert exactly to bf16 and fp16. With no scales, `mfma_scaled` fills unit E8M0 scales (S40), one
  `0x7F` register for both: the 64-head DSA QK is a plain fp8 product, the softmax scale applied to the fp32 scores after it,
  on an instruction that ran at twice the bf16 rate in a probe, 4899–4949 against 2435–2468 TFLOP/s (ratio
  2.00–2.02 over 4 runs) *(field, E11)*. The 8-token kernel pads the 64 RoPE channels to the instruction's K = 128:
  a K = 64 operand with `k_width` 32 returned acc / 2 + a·b (`gluon/mfma.md` §10). `allow_flush_denorm=True`
  changes no instruction of the 24-token producer or merge (compiled with Triton 3.8.0 for gfx950; `numerics.md` §6).
- **Choose P against the bound.** The bounds are loose because the baselines are lossy: a port of DSA's sglang kernel
  (fp8 softmax weights) sits at 36.5 dB against a 20 dB bound, an emulation of Qwen3.5's baseline at 22.3–22.6 dB
  against 15 dB on the values *(field, E13)*. bf16 P and bf16 partials pass both. fp16 P keeps 11 significant bits
  for P ≥ 2^−14 with V exact in fp16; bf16 hi + lo about 16 *(field, E13)* (below; `numerics.md` §5).
- **Hi + lo P: the residual as extra MFMA rows.** Split each probability into a bf16 high part and a bf16 residual,
  stack them as row pairs, run one P·V MFMA on twice the head rows and add each pair at the end. A stand-in of this form
  compiles with Triton 3.8.0 for gfx950 (8 heads, 64 keys: eight `v_mfma_f32_16x16x32_bf16`, as many as a 16-row bf16 P·V).
  The 8-row Qwen3.5 kernel, whose launch sets the three constexprs below for its 8 heads (`TILE` 64, `lhs` the A operand
  layout of the `[16, 16, 32]` MFMA; row 2h holds head h's high part, row 2h + 1 its residual):

  ```python
          prob_hi = probabilities.to(gl.bfloat16)
          prob_lo = (probabilities - prob_hi.to(gl.float32)).to(gl.bfloat16)
          if COMPRESS_SOFTMAX:
              # Reconstitute M16 as adjacent high/residual probability rows.
              probability_pair = gl.join(prob_hi, prob_lo).permute((0, 2, 1)).reshape((16, TILE))
              numerator = gl.amd.cdna3.mfma(gl.convert_layout(probability_pair, lhs), value, numerator)
          # ... the arms of other row counts, the end of the tile loop, the record indices
      if FOLD_PROB:
          if INTERLEAVED:
              folded = gl.sum(numerator.reshape((HEADS, 2, 256)), axis=1)
          # ... the arm of other row counts, the store of the folded rows
  ```

  The extra rows cost no instruction only where they would be padding: 8 heads on the 16-row MFMA. The 8-row kernel loads
  query row r from head r // 2, keeps rows 2h and 2h + 1 in one thread (non-transposed MFMA), splits off the duplicate
  score row, runs the softmax once per head (§12), rescales both rows of a pair and adds the pairs before the store.
  The 8-token DSA kernel stacks 16 heads' hi over their lo on `[32, 32, 16]`: as many instructions as the
  `[16, 16, 32]` its 16 heads would fill, at twice the MACs each (`../hardware/gfx950/matrix_core.md`).
- **Partials in bf16 halve the numerator traffic** at one more rounding; the statistics stay fp32 and the finish widens the
  partials before its fp32 sums; a stand-in so built about doubled the error against float64 *(field, E13)*.

## 11. Qwen3.5: the prologue

The first launch (Q/K RMSNorm, mRoPE, the fp8 KV-cache write and the gate copy, in one-wave programs):
[attention/gated_attention_launches.md](attention/gated_attention_launches.md) §3.

## 12. Softmax across waves: the reductions and the P hand-off

With the head rows in one MFMA tile the producers spread a score tile's keys over the warps: `warps_per_cta=[1, 4]` on
Qwen3.5's `[16, TILE]` tile, `[HEAD_TILE // 32, 256 // HEAD_TILE]` over 8 warps in DSA. A row max or sum then
crosses waves through LDS and `s_barrier`. Compiled with Triton 3.8.0 for gfx950, on a `[16, 64]` fp32 tile in the
`[16, 16, 32]` MFMA layout, row max and sum take 2 LDS writes, 2 reads and 3 `s_barrier` with `[1, 4]` warps and none
with `[4, 1]`, 4 `s_barrier` per tile in an online-softmax loop; DSA's `[32, 256]` over `[1, 8]` warps (`HEAD_TILE` 32)
takes 7. There the bf16 P conversion to the P·V operand is one more round trip: a `ds_write_b64`, an `s_barrier` and two
`ds_read2st64_b64` (2 KiB); DSA stages P through its own buffer. The kernels cut the crossings three ways, none
timed against plain `gl.max` and `gl.sum`:

- **Defer the denominator** (128 and 256 rows): carry the fp32 terms,
  `denominator_terms * alpha[:, None] + probabilities`, sum them once after the loop: only the max crosses waves.
- **Move the softmax into a wave** (8 rows): convert the scores to `BlockedLayout([1, 1], [2, 32], [4, 1], [1, 0])` over
  `[8, TILE]`, a head per half-wave. Compiled from a blocked `[8, 64]` stand-in, that conversion is the one LDS
  round trip (2 KiB, one `s_barrier`); max and sum use DPP and `v_permlane16_swap`.
- **Reduce in two levels by hand** (DSA): each warp reduces its keys in registers and writes its partial into a 2 KiB
  fp32 scratch whose halves hold max and sum; a fold layout reads them back and finishes in registers: at
  `HEAD_TILE` 32, 2 `s_barrier` against 7 for `gl.max` and `gl.sum` (compiled). That reduction in the 24-token kernel's source, the branches for a 64-head tile
  left out; `storage` is a `[HEAD_TILE, 2 * (256 // HEAD_TILE)]` fp32 LDS buffer:

  ```python
  @gluon.jit
  def _key_reduce(value, storage, HEAD_TILE: gl.constexpr, MAXIMUM: gl.constexpr, COMPACT_FOLD: gl.constexpr):
      KEY_WAVES: gl.constexpr = 256 // HEAD_TILE
      if HEAD_TILE == 32:
          local = gl.reshape(value, [HEAD_TILE, KEY_WAVES, 32])
          if MAXIMUM:
              partial = gl.max(local, 2)  # each warp reduces its own keys in registers
          else:
              partial = gl.sum(local, 2)
      # ...
      shared = storage.slice(0 if MAXIMUM else KEY_WAVES, KEY_WAVES, dim=1)  # the max or the sum half
      shared.store(partial)
      if HEAD_TILE == 32:
          if COMPACT_FOLD:
              fold_layout: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2], [1, 8], [0, 1])
          else:
              fold_layout: gl.constexpr = gl.BlockedLayout([1, 8], [32, 2], [1, 8], [0, 1])
      # ...
      partials = shared.load(fold_layout)  # read back in the fold layout
      if MAXIMUM:
          result = gl.max(partials, 1)  # finished in registers
      else:
          result = gl.sum(partials, 1)
      matrix: gl.constexpr = gl.amd.AMDMFMALayout(
          version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[HEAD_TILE // 32, KEY_WAVES]
      )
      return gl.convert_layout(result, gl.SliceLayout(1, matrix))
  ```

## Evidence

Triton paths are `triton-lang/triton@c01b677` (tag v3.8.0). "The ruler" is the one [tuning.md](tuning.md) §1 defines; Kineto read its kernel durations here, and the call times quoted are medians over replays; an "unmodified" kernel or arm is the kernel as §3 and §6 describe it. The evidence of §8, §9 and §11 (E7, E17, E22 among it) is on [attention/gated_attention_launches.md](attention/gated_attention_launches.md), under the same ids.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, speedups as tuning.md §1 defines them] DSA sparse MLA with one query token per sequence, accuracy passed at each: 64 heads (pool of 65536 rows) 1.030 at 8 tokens, 1.699 at 12, 1.840 at 16, 2.188 at 24, 1.966 at 32; in a second run of the ruler 1.03, 1.737, 1.839, 2.159 and 1.978; 8 heads 1.062 at 32 tokens (pool 65536 rows), 1.196 at 128 (262144), 1.050 at 256 (524288)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (DSA) and N=4 (Qwen3.5) runs, the ruler, 4 alternated pairs per arm in each run, one GPU] DSA 64 heads at 24 tokens (192 producers of 8 warps, 163,840 B of LDS each): the 32-head tile at 8 splits (384 programs, 1 tile each, an 8-way merge) 22.07, 22.03 and 21.87 µs against 20.27, 20.30 and 20.31 µs at 4 splits (192 programs, 2 tiles each, a 4-way merge), ratios 1.089, 1.085 and 1.077, the two arms timed in separate A/B sessions whose unmodified arms agree (18.66–18.80 and about 18.69–18.85 µs); no 4-wave producer and no other token count timed. 32 tokens, refill off (`v_mfma` 82 → 82, accuracy passing): 21.68 → 21.83, 21.72 → 21.77 and 21.62 → 21.47 µs (B/A 1.007, 1.002, 0.993), the off arm's rounds 21.71–21.93, 21.69–21.87 and 21.36–21.52 µs against 21.63–21.74, 21.67–21.78 and 21.59–21.70. Qwen3.5 at 4 rows: the MFMA attention 21.77, 21.64, 21.69 and 21.59 µs against 26.88, 26.79, 26.95 and 26.94 µs for the same kernel forced to a program per (row, head, split) (B/A 1.235–1.248, all 16 pairs 1.230–1.255); at 1 row the per-head attention 16.56, 16.42, 16.44 and 16.57 µs against 16.41, 16.22, 16.45 and 16.28 µs with the 4-row kernel's MFMA attention and merge (MFMA/per-head 0.982–1.001, pairs 0.976–1.014, the MFMA faster in 14 of 16), which resolves neither a small advantage nor a crossover; at 256 rows the gate load and sigmoid moved below the merge: finish 6.13, 6.18, 6.20 and 6.11 µs against 6.21, 6.18, 6.26 and 6.25 µs unmodified (0.978–1.001, pairs 0.965–1.024, the moved order faster in 14 of 16), outputs bitwise equal, the gate load still issued before the statistics load (instruction 24 against 26; unmodified 19 and 28)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, speedups as tuning.md §1 defines them] Qwen3.5 gated attention with one query token per row (qkv_dim 4608, 1 KV head, head_dim 256, 4096 keys per row on average), accuracy passed at each: 2.182 at 1 row, 1.699 at 2, 1.671 at 4, 1.803 at 8, 1.708 at 16, 1.789 at 32, 1.401 at 64, 1.204 at 128, 1.066 at 256; in a second run of the ruler 2.221, 1.685, 1.668, 1.842, 1.729, 1.785, 1.398, 1.206 and 1.085
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, 4 alternated pairs per arm, one GPU; at 32 rows also an earlier batch of 4 runs per arm] Qwen3.5 gated attention with one query token per row, only the split-count line of the host code changed, the head tile following it: at 32 rows 16 → 8 splits, speedup 1.790 → 1.129 (B/A 0.631, pairs 0.627–0.634), call 44.18 → 69.77 µs; 16 → 32 splits, 1.791 → 1.693 (B/A 0.945), call 44.10 → 46.56 µs, each B run below each A run; the 32-split producer 190 VGPRs, occupancy 2, the unmodified one 188; at 64 rows 8 → 4 splits, 1.408 → 0.901 (B/A 0.640), call 84.23 → 131.6 µs; 8 → 16 splits with a 16-row head tile, 1.407 → 1.447 (B/A 1.028, pairs 1.025–1.034), call 84.21 → 81.86 µs; the earlier batch at 32 rows: 1.788 → 1.132 (44.13 → 69.88 µs) and 1.780 → 1.690 (44.25 → 46.76 µs)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (DSA) and N=4 (Qwen3.5) runs, the ruler, 4 alternated pairs per arm in each run, one GPU] DSA 64 heads at 24 tokens: unmodified 18.80, 18.78 and 18.66 µs; the 32-head tile with 4 splits (192 programs either way, 2 tiles each, a 4-way merge) 20.27, 20.30 and 20.31 µs (B/A 1.078, 1.081, 1.088); rows gathered per call 28.31 against 56.62 MB (computed from the host lines); the baseline 25.63–25.73, 27.02–27.11, 41.28–41.47 and 43.67–43.83 µs at 12, 16, 24 and 32 tokens (1.72–1.73 µs per token at 24, 1.37 at 32). 8 heads with the QK MFMAs issued twice (`v_mfma` 68 → 96 and 34 → 48), the whole call timed: 18.37, 18.45 and 18.50 → 18.52, 18.63 and 18.60 µs at 32 tokens (B/A 1.006–1.010), 36.41, 36.55 and 36.41 → 38.10, 38.07 and 38.01 µs at 128 (1.042–1.047); about 30 FLOP per gathered byte, computed; accuracy passed in every arm. Qwen3.5 at 256 rows, K and V (with Q) in one group waited with `wait_group(0)` before QK: 229.3, 230.2, 231.2 and 231.2 µs against 207.7, 207.8, 209.8 and 209.5 µs unmodified (B/A 1.104, 1.108, 1.102, 1.103; all 16 pairs 1.102–1.112); accuracy passed in 3 of the 4 runs, the other run's report also carrying two status flags of the ruler
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, 4 alternated runs per arm, one GPU; the split change also compiled without a GPU] DSA 64 heads at 24 tokens, the kernel's only size: `waves_per_eu=2 if rows >= 24 else 0` → `waves_per_eu=0` on the 8-warp producer launch: call 18.89 → 18.71 µs (time ratio 0.9904, speedup B/A 1.008), the unset arm faster in all 4 pairs, the arms' run ranges not recorded; KV splits 8 → 4 with the 64-head tile kept: the producer fails to compile, `CompilationError` from `NameError('cache_shared is not defined')`, without a GPU and in each of the 4 timed runs (4 of 4 in an earlier batch too); the 32-head tile with 4 splits compiles and passes, 2.034 and 2.039
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel run once on its own Triton cache, the cached `.amdgcn` and `.json` parsed; LDS liveness variants compiled without a GPU] producers of the 8 DSA and 9 Qwen3.5 kernels, each a single compiled variant with no spills: LDS bytes, VGPRs and AGPRs as in §3 and §6 (DSA 64 heads at 8, 12, 16, 24 and 32 tokens 147,696 B / 150 with 22 AGPRs, 156,160 / 88, 156,160 / 94, 163,840 / 228, 157,696 / 214; DSA 8 heads at 32, 128 and 256 tokens 49,216 / 154, 79,872 / 199, 83,968 / 202; Qwen3.5 at 1–256 rows 4 warps each, 1,024 / 130, 1,024 / 130, 32,768 / 182, 34,816 / 125, 16,624 / 136, 33,264 / 188, 33,264 / 192, 69,632 / 147 with 63 AGPRs, 71,680 / 172); `ds_read_b64_tr_b8` 64, 32, 32, 32, 64 in the 8- to 32-token 64-head producers, 32, 16, 16 in the 8-head ones, 16, 32, 64, 64, 80, 64 in the Qwen3.5 ones at 8–256 rows, none at 1–4 rows; MFMA opcodes: `scale_f32_32x32x64_f8f6f4` with `f32_32x32x16_bf16` at 16–32 tokens and `f32_32x32x16_f16` at 12, `scale_f32_16x16x128_f8f6f4` with `f32_32x32x16_bf16` at 8, `f32_16x16x32_bf16` alone in the 8-head producers and the Qwen3.5 ones from 8 rows, with `f32_16x16x32_f16` in the 4-row one, none in the 1- and 2-row ones; the 24-token producer with `waves_per_eu` 2: 228 VGPRs, unset 222, 163,840 B either way; the 12- to 32-token 64-head producers 8 warps each, and the 24-token one at 165,888 B with P allocated before the second key reduction
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=5, N chained trivial Gluon kernels timed by the wall time of CUDA-graph replays with CUDA events (gaps included) and by the ruler's timer (Kineto, queue profiling forced), least-squares slope over N] one more chained kernel in a bare replayed graph: 1.766–1.771 µs in four runs, 1.815 µs (grid 1) and 1.844 µs (grid 256) in the fifth; on the ruler's timer 4.045–4.065 µs per added kernel; the reduce kernel of the N 2624 × K 6144 bf16 GEMM's split-K added 4.2 µs (12.57 against 8.37 µs) (trivial one-wave kernels, grids 1 and 256)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, 1 GiB streaming read probe, `buffer_load` `.cg` 16 B per lane, median of 7 rounds × 5 launches] 256 programs × 4 warps with 1, 2, 4 and 8 independent loads in flight per lane: 2.26, 3.87, 5.74 and 7.14 TB/s; 2.29, 3.90, 5.70 and 7.13; 2.29, 3.92, 5.76 and 7.14 in the three runs; one and eight loads re-measured in each run 2.26, 2.24, 2.27 and 7.17, 7.15, 7.13 TB/s; 512 programs with one load 3.87–3.88 TB/s, as 256 with two (ratio 1.000–1.010); an earlier probe 2.31–7.17 TB/s (contiguous bf16 stream, Gluon probe kernel)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, probe: 2048 one-wave programs, a 64 × 64 fp32 accumulator, 4096 chained MFMAs, median of 7 rounds × 3 launches] bf16 `mfma` 2447–2468 TFLOP/s at [16, 16, 32] and 2435–2452 at [32, 32, 16], e4m3 `mfma_scaled` with unit scales 4899–4934 at [16, 16, 128] and 4907–4949 at [32, 32, 64], ratio 1.999–2.004 and 2.008–2.020; an earlier probe 2448–2451 and 4922–4923 (Gluon probe kernel)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, probe: 2048 programs × 4 warps, 8 readers per chunk, a 1 GiB read before each launch, median of 9 launches, 3 alternated rounds per run] readers with program ids congruent mod 8 against consecutive ids, per round: 64 KB chunks 11.7–12.1 against 21.3–21.6 µs, 512 KB chunks 74.1–76.8 against 154.4–160.5 µs; the median of the 3 rounds' ratios 1.790, 1.781 and 1.797 (64 KB) and 2.121, 2.085 and 2.102 (512 KB) in the three runs, the lowest single round 1.765 and 2.066; program i ran on XCD i mod 8 on the default stream in every grid probed (on a new stream the mapping rotates by a constant); an earlier probe 1.78 and 2.12 (Gluon probe kernel)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, kernels run on the GPU: the unmodified kernels and ablated copies whole against a float64 copy of the op's reference, stand-in Triton and Gluon kernels against torch or float64] Qwen3.5 32 rows, rows of 1–4096 keys: records merged in float64 within 5.04e-4 of the float64 softmax (rows of 2049–4096 keys); 209 empty splits, each max −inf, denominator 0, numerator 0; no max guard, values 36.09 dB and scales 50.87 dB (bounds 15, 20); without the bf16 rounding of the attention output and the gated product 35.13 dB, 2060 of 65,536 codes changed, mismatches against the reference 2155 → 2545. DSA 24 tokens: output within 1.84e-3 (bf16 rounding of the reference alone 1.76e-3); −1 tails over tiles, a split, a single key and a whole token: 0 non-finite values, the all −1 token exactly 0, 55.71 dB (bound 20); both guards removed: that token NaN in every head and channel, and the accuracy check on the ruler's distinct slots still passes; at 32 tokens with only the producer guard removed: NaN in the token whose first split has two empty tiles, no other. Both kernels: bitwise over 20 runs; NaN- and zero-filled scratch bitwise equal; a graph captured on one set of slots or lengths and replayed on another equal to eager. The 256-row kernel at 128 rows: no error, values −1.31 dB (the 128-row kernel 34.96 dB, the 256-row one at 256 rows 34.73 dB). All 9 prologues with unit scales: key and value caches equal the reference's chain bitwise (0 of 130,816 bytes); without the bf16 rounding before RoPE 948 key bytes differ, before the fp8 store 1028. Stand-ins: P·V: largest relative error of fp16 P 4.865e-4 over P ≥ 2^−14 (282 of 131,072 values below), of bf16 hi + lo P 7.606e-6; a port of sglang's DSA kernel (p × 448 in e4m3), N(0, 1) inputs, 64 heads: 36.57 and 36.54 dB at 8 and 32 tokens; a float64 emulation of Qwen3.5's baseline 22.33, 22.44 and 22.62 dB at 1, 8 and 32 rows; bf16 P: 54.3 dB (DSA) and 38.7 dB (Qwen3.5 values); bf16 partials with fp32 statistics in a split-K merge: error against float64 8.0e-4 → 1.51e-3 (DSA, 8 splits) and 7.5e-4 → 1.79e-3 (Qwen3.5, 16 splits); one fp8 [64, 128] tile in 8192 B of LDS read for QK through `.permute((1, 0))` and for P·V: within 1.756e-3 of torch; V in `compute_efficient_padded_shared_layout` at TILE 32 and 64, read with `ds_read_b64_tr_b8`: P·V within an absolute 1.9e-6 of torch
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, Triton probes timed by CUDA-graph replay or kernel time without a flush; a tile stream in GB/s, median of rounds; the router by the ruler, 4 alternated rounds] At 256 Qwen3.5 rows (512 MiB of K and V per call, slots drawn with replacement) the partials kernel 78.6, 78.3, 78.9 and 79.42 µs (6.83, 6.85, 6.80 and 6.76 TB/s) and the baseline's paged attention 144.8, 145.0, 145.1 and 144.53 µs (3.71, 3.70, 3.70 and 3.71 TB/s), against a flushed gather of a random permutation of 256-byte rows at 5.40, 5.37, 5.11 and 5.12 TB/s; 8 rows, one 55× longer than the others, 4 splits per row (32 programs) 51.19, 50.95, 51.21 and 50.74 µs against 7.34, 6.56, 6.13 and 10.56 µs for 62 equal chunks (6.97×, 7.76×, 8.36×, 4.81×), replayed; a 16384-program 1-D grid: program id against start rank, correlation 0.997–0.998, and 256 jobs 40× longer than the rest finished in 178.2, 178.4, 178.1 and 180.1 µs on the lowest ids against 243.4, 243.5, 243.6 and 244.7 µs on the highest; a 64 × 64 bf16 tile stream at 1024 and 2048 programs: async copies 5.50–5.93 TB/s, register-only loads 5.61–6.01, async over register-only 0.957–0.987 at 2048 programs and 0.983–1.008 at 1024 (two async modes); an 8-token MoE router with register loads in place of async copies 8.417 against 8.383, 8.410 against 8.421, 8.422 against 8.295 and 8.667 against 8.595 µs (B/A 1.004, 0.999, 1.015, 1.008) (the Qwen3.5 256-row kernel; the rest toy Triton and Gluon kernels)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, each kernel at its size: 40 runs compared byte by byte, launches traced, kernel time as the sum of Kineto kernel durations, median of 5; sources scanned for atomics] all 17 kernels (8 DSA, 9 Qwen3.5) byte-identical over 40 runs, none with an atomic, semaphore or spin; the DSA producer and merge grids, warps and `waves_per_eu` of §3 at all 8 kernels; Qwen3.5 3 kernels at all 9 row counts against the baseline's 7, device time 15.6, 21.4, 20.6, 20.3, 26.9 and 43.7 µs against 28.8, 29.6, 30.8, 37.2, 45.6 and 77.8 µs at 1, 2, 4, 8, 16 and 32 rows in the newer run, the older within 1 µs
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 runs for the 4-row kernel's branches at 2 and 8 rows, N=2 for the unmodified 2-row kernel against its replacement and for the 8- and 128-row group comparisons, N=6 at 256 rows; 4 pairs per run in a fixed A/B order, the ruler; ISA of the compiled launches] no same-size A/A, so no mean-difference uncertainty. GQA: at 8 rows the same kernel's branches, per-head over MFMA 1.975–2.019 (MFMA 27.74–28.11 against per-head 55.08–55.89 µs); at 2 rows MFMA over per-head 1.025–1.067; a replacement MFMA attention source over the unmodified 2-row kernel 0.988–1.005 (the merge changes too); every timed arm accurate. Commit groups, one over two: 8 rows 22.33–22.35 against 21.93–21.94 µs (about 1.019); 128 rows 138.9–139.2 against 118.7–119.2 µs (run means 1.167–1.169); 256 rows 1.102–1.108 over 6 runs and 24 pairs (the 4 runs of E5 among them), accuracy passed in 5 of the 6 runs. Compiled pre-QK waits: the 8-row kernel (its early-V switch on) `vmcnt(4)` with two groups and `vmcnt(0)` with one, no LDS-direct copy outstanding in either; 128 and 256 rows `vmcnt(8)` and `vmcnt(0)`, 8 and 0 copies outstanding
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=6 counterbalanced rounds per arm in one session (3 forward, 3 reverse), the ruler, paired-log 99% t CIs of call time] 27 arms (54 kernels) compiled, every timed arm accurate. DSA 64 heads, modified over unmodified: 96/128/512 producers against 192/256/256 at 12/16/32 tokens 1.414/1.456/1.098; the 64-head over the 32-head tile 1.154 at 12 tokens and 1.222 at 16 (17.58 against 14.39 µs; both arms the 24-token kernel run at 16 tokens with correctly sized packed statistics, 64 heads on 128 producers against 32 on 256, since the 16-token kernel's own 64-head path needs 164,848 B of LDS); 32 over 64 heads 0.9737 at 32 tokens (21.67 against 22.25 µs). Qwen3.5, doubled splits over unmodified 0.7863/0.8142/0.9731/0.7818/0.7052 at 4/8/64/128/256 rows; at 16 rows half 1.441, double 1.033. CIs in §3, §4 and §7
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=12 randomized timing blocks in one session, 78 configurations with 7 same-kernel duplicate controls per block, paired-log 95% CIs] synthetic gathers of distinct random fp8 rows at 256- and 576-byte strides (the 576-byte rows read 512 bytes), 4 warps per program, U = 1, 2, 4 or 8 independent 16-byte loads per lane: U8 over U1 3.171× and 2.959× at 256 programs (CIs 3.1698–3.1729 and 2.9568–2.9613), 1.825× and 1.718× at 512, 1.172× and 1.109× at 1024; the matched contiguous bf16 stream 2.240 → 7.073 TB/s; all 78 configurations accurate; the compiled stream and gather issue U `dwordx4` loads before the wait, no scratch; duplicate-control CIs within ±1% (Gluon probe kernels)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, E12's probe (8 readers per chunk, 4 warps, a 1 GiB read before each launch, median of 9 launches), 7 alternated rounds, both arms one binary] consecutive ids over ids congruent mod 8: 64 KB chunks at 2048 programs 1.776 (22.20 against 12.48 µs), as in E12; 16 KB chunks at 2048 programs 1.309 (9.36 against 7.16 µs); 64 KB chunks at 1024 programs 1.536 (Gluon probe kernel)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 sessions × 36 alternated rounds, each the median of 10 launches that read 1 GiB, rates paired within a round] E14's 64 × 64 bf16 tile stream of 4-warp programs at 256 programs, async over register-only rate, median of 108 pairs: one tile in flight 0.525, two 0.849; at 1024 programs, 32 × 64 tiles 0.935 and 0.971 (toy Gluon kernel)
- S22 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:468 `waves_per_eu=N` becomes `"amdgpu-waves-per-eu"="N,N"`
- S23 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:283 a masked lane's byte offset becomes 0x80000000 (:280); the descriptor's `num_records` is INT_MAX − 1 bytes (:104)
- S40 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_ops.py:69 `mfma_scaled` without scales fills unit E8M0 scales (0x7F)
