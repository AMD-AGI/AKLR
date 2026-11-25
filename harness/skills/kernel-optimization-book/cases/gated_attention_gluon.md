---
title: "Qwen3.5 gated attention in Gluon, one query token per sequence: QKV dim 4608, 8 query heads on 1 fp8 KV head of dim 256, a paged KV cache the op writes, fp8 group-quantized output — a one-wave task grid, split-KV partials and a fixed-order merge that gates and quantizes, three launches at 1-256 rows"
kind: case
ops: [attention, norm]
dtypes: [bf16, fp8, fp32]
regimes: ["one query token per sequence", "rows 1-256 (a kernel per count: 1, 2, 4, 8, 16, 32, 64, 128, 256)", split-kv, gqa]
accuracy: bitwise
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup over the baseline's six calls, per size, the ruler's stored value in one reference run", value: "2.182 (1 row), 1.699 (2), 1.671 (4), 1.803 (8), 1.708 (16), 1.789 (32), 1.401 (64), 1.204 (128), 1.066 (256)", gpu: MI355X, workload: "one query token per row, 8 query heads on 1 fp8 KV head of dim 256, 4096 keys per row on average, at 1, 2, 4, 8, 16, 32, 64, 128 and 256 rows", date: "2026-11"}
  - {metric: "the same speedup in a second run of the ruler", value: "2.221 (1 row), 1.685 (2), 1.668 (4), 1.842 (8), 1.729 (16), 1.785 (32), 1.398 (64), 1.206 (128), 1.085 (256)", gpu: MI355X, workload: "the same nine sizes", date: "2026-11"}
  - {metric: "runs with byte-identical outputs, per size", value: "40 of 40 at each of the 9 sizes", gpu: MI355X, workload: "each kernel at its row count", date: "2026-11"}
---

# Qwen3.5 gated attention in Gluon: one query token per sequence over a paged fp8 KV cache the op writes

Nine Gluon kernels for MI355X, one per row count from 1 to 256, a family of the Triton and Gluon kernel library
that [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps; that page says which family to start
from. Each call is three launches: a prologue, split-KV partials and a finish.
[triton/attention](../triton/attention.md) §2, §6, §7, §10 and §12 explains the producer and its merge, the split
count and the precisions, and
[triton/attention/gated_attention_launches](../triton/attention/gated_attention_launches.md) the K and V staging,
the finish and the prologue. This page has the family's kernels by size, their code, and what was measured.

The op:

One token per row, 8 query heads on 1 KV head of dim 256: Q and K RMS-normalized, mRoPE on 64 channels, K and V written
to the e4m3 paged cache, attention over the row's slots, `sigmoid(gate)`, then e4m3 per 128 channels with fp32 scales; 9
kernels (three source files each: the entry, the prologue, the attention), one size each, 1–256 rows. The baseline chains SGLang's fused QK
norm, Triton mRoPE and FP8 KV write, aiter's `paged_attention_ragged`, `fused_sigmoid_mul` and aiter's per-1 × 128
quantization. Triton side: [triton/attention](../triton/attention.md) §6, §7 and [gated-attention launches](../triton/attention/gated_attention_launches.md) §1, §2.

## Applies when

- The op is the body of sglang 0.5.22's `Qwen3_5AttentionDecoderLayer.self_attention` with one query token per
  sequence, between `qkv_proj` and an `o_proj` that takes the op's fp8 values and per-128 scales. The kernels cover
  KV scales of 1 only. A speedup is against the baseline's chain of six calls named above, which runs 7 kernels
  where the op runs 3 *(field, E1)*.
- Shapes and dtypes: QKV dim 4608 in bf16, a row holding per query head 256 query channels then 256 gate channels,
  then the key and the value; 8 query heads share 1 KV head of dim 256; the K and V pools are `[slots, 1, 256]` in
  e4m3, and a row brings the slot it writes and the list of slots it attends; the output is e4m3 values
  `[rows, 8, 256]` with one fp32 scale per group of 128 channels, 16 a row.
- A kernel is written for one row count: 1, 2, 4, 8, 16, 32, 64, 128 or 256 (§ Mechanism); another count has no
  kernel. The configuration is chosen from the batch shape, and lengths and slots are read on the device.
- Built with Triton 3.8.0: the 16–64-row kernels take their padded V layout from
  `compute_efficient_padded_shared_layout` and the 256-row prologue uses `gl.amd.slice`, which Triton 3.7 lacks
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §1).
- The family is held to `bitwise` accuracy: every run on the same inputs gives the same bytes as the first. No
  kernel has an atomic, and 40 runs of each at its size were byte-identical *(field, E1)*.
  The outputs' error bounds hold besides, 15 dB on the fp8 values and 20 dB on their scales, where the 32-row
  kernel read 36.09 and 50.87 dB on rows of 1–4096 keys *(field, E2)*.

## Counter-signals

- **The lowest measured sizes** are 256 rows (1.066; 1.085 in the second run of the ruler) and 128 rows (1.204;
  1.206) *(field, E3)*.
- **Up to 16 rows most of the gain is kernels saved at the timer's floor**, about 4 µs per kernel on a timer that
  sums Kineto kernel durations, as the ruler's timer does (`../pitfalls/profiled_timer_floor.md`): the 4 kernels
  saved are 16 µs (estimated), 86% or more of the gain in summed kernel durations at 1 to 16 rows and 47% at 32
  *(field, E4, E1)* ([triton/attention](../triton/attention.md) §6).
- **A kernel runs at its own row count only.** The 256-row kernel's attention also accepts 128 rows, but below 256
  rows its prologue writes Q and the gate interleaved as `uint32` pairs and returns strided views, which that
  attention reads as contiguous rows: at 128 rows it raised no error, and its values read −1.3 dB against a 15 dB
  bound *(field, E2)*. The 16-, 32- and 64-row kernels share one tile function, and their
  sources hold the other sizes' branches too: follow your size's branch.
- **The split counts are the kernels' rule, not measured optima.** With the split count doubled the call took
  0.7863, 0.8142, 0.9731, 0.7818 and 0.7052 times as long at 4, 8, 64, 128 and 256 rows (one session, 6
  counterbalanced rounds per arm) *(field, E5)*. At 16 rows the kernel's count was the fastest
  of the three tested (half 1.441, double 1.033), and at 32 rows half and double took 1.58 and 1.06 times as long
  *(field, E5, E6)*. Where the kernel ties the head tile to the split
  count (16–64 rows: a 16-row head tile at 16 splits, else 8 rows), an arm that moves to or from 16 splits changes
  both; tiles per split and merge width change in every arm; no exhaustive search was run.
- **No guard for a row without keys.** The 32-row kernel guards the running max neither in the producer nor in the
  merge. What was run is rows of 1 to 4096 keys, where it stays within the bounds
  *(field, E2)*: every row attends at least one slot, and a split runs only tiles that hold a
  key.
- **Unequal row lengths were not isolated on these kernels.** The split count is fixed per row count, and the
  ruler's inputs give each row 4096 keys on average; the tail of one long row was timed on a Triton toy only
  *(field, E7)* ([triton/attention](../triton/attention.md) §6).
- **Relaxed reads leave a barrier to the kernel.** The 128- and 256-row kernels read Q and K with
  `load_shared_relaxed`, whose tag makes Membar drop the barrier between that read and any async copy into the same
  buffer (S1, S2).
  The next copy into a buffer then relies on the barrier after `wait_group(0)`: a K prefetch placed above that wait
  compiles ahead of the K reads with no `s_barrier`, so it can overwrite K while other waves still read it
  (inferred).
- **Per-size choices that no size isolates.** Where V is gathered and async copies against register gathers are
  compared at no size, and `waves_per_eu` (2 at 16 and 32 rows, 3 at 128) is untimed
  ([triton/attention](../triton/attention.md) §7); the deferred denominator at 128 and 256 rows and the softmax
  moved into a wave at 8 rows are not timed against plain `gl.max` and `gl.sum`
  ([triton/attention](../triton/attention.md) §12). Without V's `.cg` at 32 rows or K's at 64 the call ran 0.6–1.9%
  faster in two runs, the noise floor not measured *(field, E8)*.
- **Per-head programs at 1–2 rows and heads combined on MFMA above are the implementation, not a measured
  crossover**: same-kernel comparisons favour MFMA at 4 rows (per-head 1.235–1.248 times the MFMA time) and at 8 rows
  (1.975–2.019), at 2 rows the same-kernel branches and a whole-kernel replacement disagree, and the small 1-row
  difference is unresolved *(field, E9, E10)*.
- **Written and not read.** The prologues write a bf16 K, and in some kernels a V copy, that nothing reads
  ([triton/attention/gated_attention_launches](../triton/attention/gated_attention_launches.md) §3); every finish
  also makes the dead `Gated` store of entry 6.

## Mechanism

| Rows | Kernels | Decomposition (3 launches: prologue, attention, finish) |
|---|---|---|
| 1–2 | a kernel per count (1), (2) | one CTA per (row, head, split), 32 splits, FMA dot products, no MFMA |
| 4 | one | one program per (row, split), 4 × 32, all 8 query heads in it on MFMA |
| 8 | one | MFMA partials, async copies into 34,816 bytes of LDS |
| 16–64 | a kernel per count (16), (32), (64) | a (row, split) grid, rows × splits = 512 (32, 16 and 8 splits), tiles dealt to the splits cyclically; K and V gathered into registers, V through a padded LDS layout; a 16-row head tile at 16 splits, else 8 |
| 128–256 | a kernel per count (128), (256) | a 1-D grid of 4 and 2 splits (rows × splits = 512), one contiguous chunk per split, async copies into LDS |

- **Techniques.** The prologue is a per-token task grid of one-wave CTAs (K norm, RoPE and cache write; V cache write; Q
  heads and the gate copy; the kernels for 1, 2, 8, 32, 64 and 128 rows give the cache-write tasks the low program ids). The
  attention launch is flash-decoding ([../attention/split_kv.md](../attention/split_kv.md) §5): per
  split an unnormalized numerator with its (max, sum); the finish merges in a fixed order, rounds to bf16, applies the
  gate and quantizes, bit for bit as emulated in the 32-row kernel ([triton/numerics](../triton/numerics.md) §2). At 256 rows the 1-D
  grid is a bijection whose blocks of 128 CTAs hold both splits of 64 rows. Enumerated for rows 1–1024 and 2,
  3, 4, 8, 16 or 32 splits, the map is a bijection exactly when the row count is a multiple of 64, each block of
  64 × SPLITS consecutive CTAs then covering every split of the same 64 rows (6144 grids; no proof for other split
  counts) *(field, E12)*. Every split writes a record, inactive
  ones neutral, and a replayed graph matched the eager call (32-row kernel). `run` is two calls, one into the prologue module and one into the attention module.
- **Resources** *(field, E13)*: 27 compiled kernels; `num_warps` 1 (9), 2 (1), 4 (17); `waves_per_eu` 2 in 2, 3 in 1; VGPR
  25–192, AGPRs in 1 (63); LDS 0–70 KiB; no spills; bf16 16 × 16 × 32 MFMA in 7, fp16 in 1.

The prologue's task grid is cut differently per size, and the launch ends before attention starts because attention
reads the slots it writes. K and V are gathered into registers at 1–4 rows (no MFMA runs at 1–2), and reach the MFMA
by async copies into two swizzled LDS tiles at 8 rows, from registers with V through padded LDS at 16–64 rows, and
by async copies at pre-swizzled offsets at 128–256 rows
([triton/attention/gated_attention_launches](../triton/attention/gated_attention_launches.md) §1). QK is fp32
multiply-adds at 1–2 rows and a bf16 MFMA on K upcast from fp8 from 4 rows; P for P·V is fp32 at 1–2 rows, fp16 at 4
(V upcast to fp16), bf16 hi + lo at 8 (entry 3) and bf16 from 16; the partial numerators are fp32 up to 8 rows and
bf16 with fp32 statistics from 16 ([triton/attention](../triton/attention.md) §10); every size passed the accuracy
check *(field, E3)*.

## Code

### 1. One tile of the online softmax: the first tile peeled, the masks only in the partial tile (partials, 32 rows)

The tile step of the 32-row kernel's partials launch; the 16- and 64-row kernels run the same function. A program
owns one split of a row's keys for all 8 heads and runs an online softmax over its tiles: the scores come from one
bf16 MFMA of the query rows against the gathered K tile, the max and the sum are `SliceLayout(1, mma)` vectors, one
value per head row, and the numerator stays in the MFMA accumulator. The caller peels a split's first tile
(`FIRST`: no rescale of the old state) and compiles the length masks only into the variant for a row's last,
partial tile (`FULL_TILE=False`); compiled, the peeled tile has 4 `v_exp_f32` against 5 and the full tile 12
`v_cndmask` against 60 ([triton/attention](../triton/attention.md) §1). `mma` is the transposed `[16, 16, 32]` MFMA
layout with `warps_per_cta=[1, gl.num_warps()]`, `[1, 4]` in this launch, and `b_layout` its B operand layout; with a
tile's keys spread over the 4 warps these `gl.max` and `gl.sum` cross waves through LDS and `s_barrier`
([triton/attention](../triton/attention.md) §12). `TILE` is the keys of a tile (128 at 32 rows), `length` the row's
key count, `key_scale` the K pool's scale and `attention_scale` the op's score scale; the `# ...` is where this size
gathers V, and P·V follows on `gl.amd.cdna4.mfma` with V read through padded LDS.

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

Explained in [triton/attention](../triton/attention.md) §2.

### 2. Every split stores a record, a split with no keys a neutral one (partials, 32 rows)

The same program around its tile loop. The state starts neutral, max −inf, sum 0 and numerator 0, and the tiles run
only under `if split * TILE < length`, so a split with no keys stores that neutral record, which the merge weighs
by exp(−inf) = 0. Scratch from `torch.empty` then needs no zeroing and a graph replayed with other lengths reads no
stale record: in a run on rows of 1–4096 keys 209 splits were empty and each held that record, NaN- and
zero-filled scratch gave bitwise equal outputs, and a graph captured on one set of lengths and replayed on another
equalled the eager call *(field, E2)*. `head_tile` is the head rows of the MFMA tile, 16 at 16 splits and else the 8
heads, the rows past `HEADS` masked out of the stores; `Partial` holds the numerators `[rows, heads, splits, 256]`
in bf16 and `Stats` the (max, sum) pairs `[rows, heads, splits, 2]` in fp32. `query_layout` is
`BlockedLayout([1, 8], [8, 8], [1, gl.num_warps()], [1, 0])`, in which each lane stores 16 bytes.

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

Explained in [triton/attention](../triton/attention.md) §2.

### 3. P as a bf16 high part and its bf16 residual in the spare rows of the 16-row MFMA (partials, 8 rows)

From the 8-row kernel's partials launch, which sets `COMPRESS_SOFTMAX`, `FOLD_PROB` and `INTERLEAVED` for its 8
heads. Each probability is split into a bf16 high part and a bf16 residual; `gl.join` stacks them as adjacent rows,
row 2h head h's high part and row 2h + 1 its residual, one P·V MFMA runs on the 16 rows, and after the tile loop
each pair is added (`folded`) before the store. The extra rows cost no instruction only where they would be
padding, as with 8 heads on the 16-row `[16, 16, 32]` MFMA. `TILE` is 64, `lhs` the A operand layout of that MFMA,
`value` the V tile and `numerator` the accumulator. In a stand-in, P·V with bf16 hi + lo P had a largest relative
error of 7.606e-6, against 4.865e-4 for fp16 P over P ≥ 2^−14 *(field, E2)*.

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

Explained in [triton/attention](../triton/attention.md) §10.

### 4. K and V copied into two LDS tiles in two commit groups, the next tile's slots after V's commit (partials, 8 rows)

From the tile loop of the 8-row kernel's partials launch, on its direct-to-LDS path (`DIRECT_LDS`): K is copied
into the swizzled tile `cache_shared` and committed, V into a second tile (`DUAL_LDS`) and committed, and
`wait_group(1)` lets QK read K while one group may still be in flight. Two groups do not by themselves keep V in
flight under QK. This size loads the next tile's slots after V's commit (`PREFETCH_STAGE == 2`; 1 is after the K
copy, 0 after the K read), so the compiled pre-QK wait, `vmcnt(4)`, leaves only those four loads outstanding and no
V copy; with stage 1 or 0 the same wait leaves the 4 V copies outstanding
*(field, E10, E14)*. One group instead of two
took about 1.019 times the call time (22.33–22.35 against 21.93–21.94 µs; 2 runs, 8 pairs), which resolves no mean
difference *(field, E10)*. `live` masks the tile's keys below the row's length, `rhs` is the B
operand layout of the `[16, 16, 32]` MFMA and `VALUE_CACHE` the cache modifier of V, empty at this size.

```python
        if DIRECT_LDS:
            # Omit `other` to use native masked zero filling on CDNA4.
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                cache_shared, Key, offsets, live[:, None])
            gl.amd.cdna4.async_copy.commit_group()
            if PREFETCH_INDICES and PREFETCH_STAGE == 1:
                # ... gl.load of the next tile's slots
            if DUAL_LDS:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    value_shared, Value, offsets, live[:, None], cache_modifier=VALUE_CACHE)
                gl.amd.cdna4.async_copy.commit_group()
                if PREFETCH_INDICES and PREFETCH_STAGE == 2:
                    # ... gl.load of the next tile's slots
                # K must be complete; V can remain in flight during QK.
                gl.amd.cdna4.async_copy.wait_group(1)
                gl.barrier()
                key = cache_shared.permute((1, 0)).load(rhs).to(gl.bfloat16)
```

Not this: `other` on a masked copy. It adds a branch per copy and, by build, a store of `other` into LDS
(S3); the MI355X image's Triton 3.8.0 made no such store, so masked
rows kept stale LDS.

Explained in [triton/attention/gated_attention_launches](../triton/attention/gated_attention_launches.md) §1.

### 5. K then V as two commit groups: QK waits for one (partials, 256 rows)

From the tile function of the 256-row kernel's partials launch, right after the offsets of entry 12. The K copy and
its commit form one group and the V copy a second; `wait_group(1)` returns when at most one committed group of the
wave is outstanding (S1), so QK reads K with V still in flight, and
V is waited for before P·V. On a split's first tile the Q copy is issued ahead of K's and shares its group. The
partial tile passes the mask `valid[:, None]` and no `other`: masked lanes read 0 with no branch. Compiled, the
pre-QK wait is `vmcnt(8)` and leaves eight direct-to-LDS copies outstanding; the overlap at run time and its share
of the gain were not measured *(field, E10)*.

```python
    if FULL_TILE:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(key_copy, key_ptr, key_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(value_copy, value_ptr, value_offsets)
    else:
        # ... the partial tile: the same copies and commit, masked with valid[:, None]
    gl.amd.cdna4.async_copy.commit_group()
    # Q/K are complete; V remains the one outstanding async group.
    gl.amd.cdna4.async_copy.wait_group(1)
```

Not this: K and V, with Q, in one group waited with `wait_group(0)` before QK. The call then took 1.102–1.108 times
as long at 256 rows (6 runs, 24 pairs; accuracy passed in 5 of the 6 runs) and 1.167–1.169 at 128 rows (2 runs, 8
pairs) *(field, E10)*.

Explained in [triton/attention/gated_attention_launches](../triton/attention/gated_attention_launches.md) §1.

### 6. The gate and the fp8 quantization of one group of 128 on the merged row (finish, 16 rows)

The function the 16-row kernel's finish calls once it has merged the splits of a (row, head) for one group of 128
channels, the merged values still in registers; the finish runs one program per (row, group). The attention output
and the gated product round to bf16 as in the reference; the sigmoid rounds to bf16 too, which the reference does
not, and the kernels pass *(field, E3)*. Without the two reference roundings the 32-row kernel
still passed and moved 2060 of 65,536 fp8 codes *(field, E2)*. The codes divide by the scale
with `gl.div_rn`, where the 128- and 256-row finishes multiply by a reciprocal; the 32- and 64-row finishes run
the same steps on a whole head and on four groups per program. `attention` is the merged fp32 row of the group, `d`
its 128 channel indices, and `Scales` holds two fp32 scales per head; the bf16 `Gated` store is dead, the host
code discarding it: drop it in a copy.

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

Explained in [triton/attention/gated_attention_launches](../triton/attention/gated_attention_launches.md) §2.

### 7. A head job: RMSNorm in one wave and the RoPE partner by an XOR gather (prologue, 32 rows)

The start of the job that normalizes one head in the 32-row kernel's prologue, where every program is one wave and
owns one job of a row: a query head, the key or the value. The head's 256 channels sit in one wave,
`BlockedLayout([1], [64], [1], [0])` with four values per lane, so the RMS sum stays in the wave. The NeoX partner
of channel c < 64 is channel c ^ 32, fetched with `gl.gather` on an XORed index; compiled, this 256-value gather is
four `ds_bpermute_b32` and no LDS. The RMSNorm result rounds to bf16 before RoPE, a rounding point of the
reference: without it 948 of 130,816 key bytes differ from the reference's chain
*(field, E2)*. `source` is the head's 256 inputs in fp32, `weight` its norm weight, `offsets`
the channel index and `rotary_dim` 64.

```python
    variance = gl.sum(source * source, axis=0) / 256.0
    normalized = (
        source * gl.rsqrt(variance + eps) * (1.0 + weight)
    ).to(gl.bfloat16)
    normalized_float = normalized.to(gl.float32)

    half: gl.constexpr = rotary_dim // 2
    partner = gl.gather(normalized_float, offsets ^ half, axis=0)
```

Explained in [triton/attention/gated_attention_launches](../triton/attention/gated_attention_launches.md) §3.

### 8. The mRoPE axis of a frequency from two constexpr masks (prologue, 32 rows)

Further down the same head job. Of the 32 frequencies, f reads `positions[1]` when `f % 3 == 1 and f < 33`,
`positions[2]` when `f % 3 == 2 and f < 30`, and `positions[0]` otherwise. The host builds the rule from
`mrope_section` as two `uint32` masks with one bit per frequency and passes them as the constexpr `axis_masks`; the
program tests the frequency's bit in each and loads its position from the row of the axis it finds, so the masks
hold the static layout and the positions stay runtime data. `frequency` is `offsets % 32`, `positions_stride` the
stride between the three rows of `positions`, `token` the row of the batch and `layout` the one-wave layout of
entry 7.

```python
        # Masks encode only the static section layout, never position data.
        height_mask = gl.full((), axis_masks[0], gl.uint32, layout)
        width_mask = gl.full((), axis_masks[1], gl.uint32, layout)
        height = ((height_mask >> frequency.to(gl.uint32)) & 1) != 0
        width = ((width_mask >> frequency.to(gl.uint32)) & 1) != 0
        axis = gl.where(height, 1, gl.where(width, 2, 0))
        position = gl.load(
            positions_ptr + axis * positions_stride + token
        ).to(gl.int64)
```

Explained in [triton/attention/gated_attention_launches](../triton/attention/gated_attention_launches.md) §3.

### 9. The split-to-row map of a 1-D grid (partials, 256 rows)

The first lines of the 256-row kernel's partials program, which runs on a 1-D grid of rows × `SPLITS` CTAs, 2
splits at 256 rows; § Mechanism has the enumeration of the map for 2 splits or more. With one split the map is a
bijection at every row count tested (1024 more grids), so the multiple-of-64 condition holds for 2 splits or more;
at 256 rows it is a bijection. The source expressions of the 256- and 128-row kernels matched the enumeration, and
five sample grids ran as enumerated *(field, E12)*. This does not widen what the
kernels accept: the 128- and 256-row kernels assert `rows in (128, 256)`, and at 16–64 rows the kernels launch a
(row, split) grid instead.

```python
# Flatten the launch into groups containing all splits of 64 rows.
cta = gl.program_id(0)
row = cta % 64 + cta // (64 * SPLITS) * 64
split = cta // 64 % SPLITS
```

Not this: the map at a row count that is not a multiple of 64 with 2 splits or more. At 96 rows and 2 splits it
misses 32 (row, split) pairs, launches 32 CTAs out of range and reaches row 127
*(field, E12)*.

### 10. The unit KV scale both launches read, built once per device (host code, every size)

These four lines are in the host code of each of the nine kernels. `run` passes the tensor as both the K and the V
scale to the prologue and to the attention call, the pools holding unscaled values; built in `run`, it would be
allocated and filled in every call, and `functools.cache` keeps one per device.

```python
@functools.cache
def _unit_scale(device):
    """The FP32 KV scale both kernels read: the pools hold unscaled values."""
    return torch.ones(1, dtype=torch.float32, device=device)
```

Explained in [triton/authoring](../triton/authoring.md) §5.

### 11. The host gate of the buffer path on the pools' byte extents (host code, 4 rows)

From the host code of the 4-row kernel. A buffer access takes 32-bit offsets against a descriptor of 2^31 − 2
bytes, so the pools are gated on their byte extent: the elements up to the end of the last row, from the shape and
the stride, times the element size. `buffer_kv` goes to the launch as a constexpr; when it is False the kernel
gathers K and V with `gl.load` on pointers. Where the caller bounds the pool below the range, that path is dead
code *(field, E15)*.

```python
key_extent = (key_cache.shape[0] - 1) * key_cache.stride(0) + 256  # elements up to the end of the last row
value_extent = (value_cache.shape[0] - 1) * value_cache.stride(0) + 256
buffer_kv = (  # False: the launch takes the kernel's pointer path
    key_extent * key_cache.element_size() <= 2**31 - 2
    and value_extent * value_cache.element_size() <= 2**31 - 2
)
```

Not this: a `buffer_load` on a pool past that range with no gate. Out of range, loads read 0 and stores are
dropped, with no fault *(field, E15)*.

Explained in [triton/gluon/memory](../triton/gluon/memory.md) §3.

### 12. Source offsets pre-swizzled for a copy into a linear alias of the LDS tile (partials, 256 rows)

From the tile function of the 256-row kernel's partials launch, ahead of the copies of entry 5. A direct-to-LDS
copy writes LDS lane by lane in order; into a `SwizzledSharedLayout` with `max_phase` above 1 the compiler swizzles
the source instead, one `ds_bpermute_b32` per copy instruction (S4): a test kernel with
this `[64, 256]` fp8 copy had 4 `ds_bpermute_b32` and 76 instructions into the swizzled buffer, none and 56
pre-swizzled *(field, E16)*. The kernel XORs the global column with the consumer's swizzle,
copies into a linear alias of the buffer (`_reinterpret` to `SwizzledSharedLayout(1, 1, 1, [1, 0])`) and restates
the 16-byte runs with `gl.max_contiguous(gl.multiple_of(...))`. K and V are
`SwizzledSharedLayout(16, 1, 8, [1, 0])` here (`QK_WIDTH` is 8, so `key_vec` is 16); the 128-row kernel XORs its
`vec = 32` K buffer with `* 32`. `slots` are the KV slots of the tile's keys, `tokens` the tile's rows and `dims`
the 256 channels.

```python
# Apply each consumer's swizzle in global source offsets so the LDS
# transfers themselves have a linear, lane-local destination layout.
linear_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
key_copy = key_shared._reinterpret(layout=linear_layout)
value_copy = value_shared._reinterpret(layout=linear_layout)
key_vec: gl.constexpr = 32 if QK_WIDTH == 16 else 16
key_offsets = slots[:, None] * 256 + (
    dims[None, :] ^ ((tokens[:, None] % 8) * key_vec)
)
value_offsets = slots[:, None] * 256 + (
    dims[None, :] ^ ((tokens[:, None] % 8) * 16)
)
# XOR only permutes aligned 16-byte runs; each run remains contiguous.
key_offsets = gl.max_contiguous(gl.multiple_of(key_offsets, [1, 16]), [1, 16])
value_offsets = gl.max_contiguous(gl.multiple_of(value_offsets, [1, 16]), [1, 16])
```

Not this: a multiplier that is not the consumer layout's `vec`. It compiles and loads the wrong chunks.

Explained in [triton/gluon/memory](../triton/gluon/memory.md) §9.

## Result

- **Speedups** over the baseline above: 1.066 (256 rows) to 2.182 (1 row); a later run of the ruler read 1.085 to 2.221
  *(field, E17)*.

Per size, the ruler's stored value against the baseline *(field, E3)*:

| Rows | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| Reference run of the ruler | 2.182 | 1.699 | 1.671 | 1.803 | 1.708 | 1.789 | 1.401 | 1.204 | 1.066 |
| Second run of the ruler | 2.221 | 1.685 | 1.668 | 1.842 | 1.729 | 1.785 | 1.398 | 1.206 | 1.085 |

[triton/attention](../triton/attention.md) §6 tables each size's partials launch as compiled (producer, programs ×
warps, splits × tile, mean tiles per split, LDS, VGPRs) and says where the gain is. At 256 rows × 4096 keys, over 4
runs, the partials kernel's call, 512 MiB of K and V, took 78.34–79.42 µs (6.76–6.85 TB/s) against 144.53–145.13 µs
(3.699–3.715 TB/s) for the baseline's paged attention, both profiled unflushed with slots drawn with replacement, so
the rates establish no gather bound; 64 and 128 rows were not timed this way *(field, E7)*.

The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1.

## Evidence

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, each kernel at its size: 40 runs compared byte by byte, launches traced, kernel time as the sum of Kineto kernel durations, median of 5; sources scanned for atomics] all 17 kernels (8 DSA, 9 Qwen3.5) byte-identical over 40 runs, none with an atomic, semaphore or spin; the DSA producer and merge grids, warps and `waves_per_eu` of [triton/attention](../triton/attention.md) §3 at all 8 kernels; Qwen3.5 3 kernels at all 9 row counts against the baseline's 7, device time 15.6, 21.4, 20.6, 20.3, 26.9 and 43.7 µs against 28.8, 29.6, 30.8, 37.2, 45.6 and 77.8 µs at 1, 2, 4, 8, 16 and 32 rows in the newer run, the older within 1 µs
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, kernels run on the GPU: the unmodified kernels and ablated copies whole against a float64 copy of the op's reference, stand-in Triton and Gluon kernels against torch or float64] Qwen3.5 32 rows, rows of 1–4096 keys: records merged in float64 within 5.04e-4 of the float64 softmax (rows of 2049–4096 keys); 209 empty splits, each max −inf, denominator 0, numerator 0; no max guard, values 36.09 dB and scales 50.87 dB (bounds 15, 20); without the bf16 rounding of the attention output and the gated product 35.13 dB, 2060 of 65,536 codes changed, mismatches against the reference 2155 → 2545. DSA 24 tokens: output within 1.84e-3 (bf16 rounding of the reference alone 1.76e-3); −1 tails over tiles, a split, a single key and a whole token: 0 non-finite values, the all −1 token exactly 0, 55.71 dB (bound 20); both guards removed: that token NaN in every head and channel, and the accuracy check on the ruler's distinct slots still passes; at 32 tokens with only the producer guard removed: NaN in the token whose first split has two empty tiles, no other. Both kernels: bitwise over 20 runs; NaN- and zero-filled scratch bitwise equal; a graph captured on one set of slots or lengths and replayed on another equal to eager. The 256-row kernel at 128 rows: no error, values −1.31 dB (the 128-row kernel 34.96 dB, the 256-row one at 256 rows 34.73 dB). All 9 prologues with unit scales: key and value caches equal the reference's chain bitwise (0 of 130,816 bytes); without the bf16 rounding before RoPE 948 key bytes differ, before the fp8 store 1028. Stand-ins: P·V: largest relative error of fp16 P 4.865e-4 over P ≥ 2^−14 (282 of 131,072 values below), of bf16 hi + lo P 7.606e-6; a port of sglang's DSA kernel (p × 448 in e4m3), N(0, 1) inputs, 64 heads: 36.57 and 36.54 dB at 8 and 32 tokens; a float64 emulation of Qwen3.5's baseline 22.33, 22.44 and 22.62 dB at 1, 8 and 32 rows; bf16 P: 54.3 dB (DSA) and 38.7 dB (Qwen3.5 values); bf16 partials with fp32 statistics in a split-K merge: error against float64 8.0e-4 → 1.51e-3 (DSA, 8 splits) and 7.5e-4 → 1.79e-3 (Qwen3.5, 16 splits); one fp8 [64, 128] tile in 8192 B of LDS read for QK through `.permute((1, 0))` and for P·V: within 1.756e-3 of torch; V in `compute_efficient_padded_shared_layout` at TILE 32 and 64, read with `ds_read_b64_tr_b8`: P·V within an absolute 1.9e-6 of torch
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, speedups as [triton/tuning](../triton/tuning.md) §1 defines them] Qwen3.5 gated attention with one query token per row (qkv_dim 4608, 1 KV head, head_dim 256, 4096 keys per row on average), accuracy passed at each: 2.182 at 1 row, 1.699 at 2, 1.671 at 4, 1.803 at 8, 1.708 at 16, 1.789 at 32, 1.401 at 64, 1.204 at 128, 1.066 at 256; in a second run of the ruler 2.221, 1.685, 1.668, 1.842, 1.729, 1.785, 1.398, 1.206 and 1.085
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=5, N chained trivial Gluon kernels timed by the wall time of CUDA-graph replays with CUDA events (gaps included) and by the ruler's timer (Kineto, queue profiling forced), least-squares slope over N] one more chained kernel in a bare replayed graph: 1.766–1.771 µs in four runs, 1.815 µs (grid 1) and 1.844 µs (grid 256) in the fifth; on the ruler's timer 4.045–4.065 µs per added kernel; the reduce kernel of the N 2624 × K 6144 bf16 GEMM's split-K added 4.2 µs (12.57 against 8.37 µs) (trivial one-wave kernels, grids 1 and 256)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=6 counterbalanced rounds per arm in one session (3 forward, 3 reverse), the ruler, paired-log 99% t CIs of call time] 27 arms (54 kernels) compiled, every timed arm accurate. DSA 64 heads, modified over unmodified: 96/128/512 producers against 192/256/256 at 12/16/32 tokens 1.414/1.456/1.098; the 64-head over the 32-head tile 1.154 at 12 tokens and 1.222 at 16 (17.58 against 14.39 µs; both arms the 24-token kernel run at 16 tokens with correctly sized packed statistics, 64 heads on 128 producers against 32 on 256, since the 16-token kernel's own 64-head path needs 164,848 B of LDS); 32 over 64 heads 0.9737 at 32 tokens (21.67 against 22.25 µs). Qwen3.5, doubled splits over unmodified 0.7863/0.8142/0.9731/0.7818/0.7052 at 4/8/64/128/256 rows; at 16 rows half 1.441, double 1.033. CIs in [triton/attention](../triton/attention.md) §3, [triton/attention](../triton/attention.md) §4 and [triton/attention](../triton/attention.md) §7
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, 4 alternated pairs per arm, one GPU; at 32 rows also an earlier batch of 4 runs per arm] Qwen3.5 gated attention with one query token per row, only the split-count line of the host code changed, the head tile following it: at 32 rows 16 → 8 splits, speedup 1.790 → 1.129 (B/A 0.631, pairs 0.627–0.634), call 44.18 → 69.77 µs; 16 → 32 splits, 1.791 → 1.693 (B/A 0.945), call 44.10 → 46.56 µs, each B run below each A run; the 32-split producer 190 VGPRs, occupancy 2, the unmodified one 188; at 64 rows 8 → 4 splits, 1.408 → 0.901 (B/A 0.640), call 84.23 → 131.6 µs; 8 → 16 splits with a 16-row head tile, 1.407 → 1.447 (B/A 1.028, pairs 1.025–1.034), call 84.21 → 81.86 µs; the earlier batch at 32 rows: 1.788 → 1.132 (44.13 → 69.88 µs) and 1.780 → 1.690 (44.25 → 46.76 µs)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, Triton probes timed by CUDA-graph replay or kernel time without a flush; a tile stream in GB/s, median of rounds; the router by the ruler, 4 alternated rounds] At 256 Qwen3.5 rows (512 MiB of K and V per call, slots drawn with replacement) the partials kernel 78.6, 78.3, 78.9 and 79.42 µs (6.83, 6.85, 6.80 and 6.76 TB/s) and the baseline's paged attention 144.8, 145.0, 145.1 and 144.53 µs (3.71, 3.70, 3.70 and 3.71 TB/s), against a flushed gather of a random permutation of 256-byte rows at 5.40, 5.37, 5.11 and 5.12 TB/s; 8 rows, one 55× longer than the others, 4 splits per row (32 programs) 51.19, 50.95, 51.21 and 50.74 µs against 7.34, 6.56, 6.13 and 10.56 µs for 62 equal chunks (6.97×, 7.76×, 8.36×, 4.81×), replayed; a 16384-program 1-D grid: program id against start rank, correlation 0.997–0.998, and 256 jobs 40× longer than the rest finished in 178.2, 178.4, 178.1 and 180.1 µs on the lowest ids against 243.4, 243.5, 243.6 and 244.7 µs on the highest; a 64 × 64 bf16 tile stream at 1024 and 2048 programs: async copies 5.50–5.93 TB/s, register-only loads 5.61–6.01, async over register-only 0.957–0.987 at 2048 programs and 0.983–1.008 at 1024 (two async modes); an 8-token MoE router with register loads in place of async copies 8.417 against 8.383, 8.410 against 8.421, 8.422 against 8.295 and 8.667 against 8.595 µs (B/A 1.004, 0.999, 1.015, 1.008) (the Qwen3.5 256-row kernel; the rest toy Triton and Gluon kernels)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the ruler, 3 alternated rounds per arm in each run, one GPU] Qwen3.5, `.cg` removed: V at 32 rows 44.3 → 43.4 and 44.1 → 43.5 µs, K at 64 rows 84.3 → 83.7 and 84.1 → 83.6 µs; noise floor not measured
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (DSA) and N=4 (Qwen3.5) runs, the ruler, 4 alternated pairs per arm in each run, one GPU] DSA 64 heads at 24 tokens (192 producers of 8 warps, 163,840 B of LDS each): the 32-head tile at 8 splits (384 programs, 1 tile each, an 8-way merge) 22.07, 22.03 and 21.87 µs against 20.27, 20.30 and 20.31 µs at 4 splits (192 programs, 2 tiles each, a 4-way merge), ratios 1.089, 1.085 and 1.077, the two arms timed in separate A/B sessions whose unmodified arms agree (18.66–18.80 and about 18.69–18.85 µs); no 4-wave producer and no other token count timed. 32 tokens, refill off (`v_mfma` 82 → 82, accuracy passing): 21.68 → 21.83, 21.72 → 21.77 and 21.62 → 21.47 µs (B/A 1.007, 1.002, 0.993), the off arm's rounds 21.71–21.93, 21.69–21.87 and 21.36–21.52 µs against 21.63–21.74, 21.67–21.78 and 21.59–21.70. Qwen3.5 at 4 rows: the MFMA attention 21.77, 21.64, 21.69 and 21.59 µs against 26.88, 26.79, 26.95 and 26.94 µs for the same kernel forced to a program per (row, head, split) (B/A 1.235–1.248, all 16 pairs 1.230–1.255); at 1 row the per-head attention 16.56, 16.42, 16.44 and 16.57 µs against 16.41, 16.22, 16.45 and 16.28 µs with the 4-row kernel's MFMA attention and merge (MFMA/per-head 0.982–1.001, pairs 0.976–1.014, the MFMA faster in 14 of 16), which resolves neither a small advantage nor a crossover; at 256 rows the gate load and sigmoid moved below the merge: finish 6.13, 6.18, 6.20 and 6.11 µs against 6.21, 6.18, 6.26 and 6.25 µs unmodified (0.978–1.001, pairs 0.965–1.024, the moved order faster in 14 of 16), outputs bitwise equal, the gate load still issued before the statistics load (instruction 24 against 26; unmodified 19 and 28)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 runs for the 4-row kernel's branches at 2 and 8 rows, N=2 for the unmodified 2-row kernel against its replacement and for the 8- and 128-row group comparisons, N=6 at 256 rows; 4 pairs per run in a fixed A/B order, the ruler; ISA of the compiled launches] no same-size A/A, so no mean-difference uncertainty. GQA: at 8 rows the same kernel's branches, per-head over MFMA 1.975–2.019 (MFMA 27.74–28.11 against per-head 55.08–55.89 µs); at 2 rows MFMA over per-head 1.025–1.067; a replacement MFMA attention source over the unmodified 2-row kernel 0.988–1.005 (the merge changes too); every timed arm accurate. Commit groups, one over two: 8 rows 22.33–22.35 against 21.93–21.94 µs (about 1.019); 128 rows 138.9–139.2 against 118.7–119.2 µs (run means 1.167–1.169); 256 rows 1.102–1.108 over 6 runs and 24 pairs (the 4 runs of E11 among them), accuracy passed in 5 of the 6 runs. Compiled pre-QK waits: the 8-row kernel (its early-V switch on) `vmcnt(4)` with two groups and `vmcnt(0)` with one, no LDS-direct copy outstanding in either; 128 and 256 rows `vmcnt(8)` and `vmcnt(0)`, 8 and 0 copies outstanding
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (DSA) and N=4 (Qwen3.5) runs, the ruler, 4 alternated pairs per arm in each run, one GPU] DSA 64 heads at 24 tokens: unmodified 18.80, 18.78 and 18.66 µs; the 32-head tile with 4 splits (192 programs either way, 2 tiles each, a 4-way merge) 20.27, 20.30 and 20.31 µs (B/A 1.078, 1.081, 1.088); rows gathered per call 28.31 against 56.62 MB (computed from the host lines); the baseline 25.63–25.73, 27.02–27.11, 41.28–41.47 and 43.67–43.83 µs at 12, 16, 24 and 32 tokens (1.72–1.73 µs per token at 24, 1.37 at 32). 8 heads with the QK MFMAs issued twice (`v_mfma` 68 → 96 and 34 → 48), the whole call timed: 18.37, 18.45 and 18.50 → 18.52, 18.63 and 18.60 µs at 32 tokens (B/A 1.006–1.010), 36.41, 36.55 and 36.41 → 38.10, 38.07 and 38.01 µs at 128 (1.042–1.047); about 30 FLOP per gathered byte, computed; accuracy passed in every arm. Qwen3.5 at 256 rows, K and V (with Q) in one group waited with `wait_group(0)` before QK: 229.3, 230.2, 231.2 and 231.2 µs against 207.7, 207.8, 209.8 and 209.5 µs unmodified (B/A 1.104, 1.108, 1.102, 1.103; all 16 pairs 1.102–1.112); accuracy passed in 3 of the 4 runs, the other run's report also carrying two status flags of the ruler
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the map enumerated on CPU, five sample grids run on the GPU] the 256-row kernel's split-to-row map over rows 1–1024 and SPLITS 1, 2, 3, 4, 8, 16, 32 (7168 grids): for SPLITS ≥ 2 a bijection exactly at multiples of 64, each block of 64 × SPLITS consecutive CTAs covering every split of the same 64 rows; for SPLITS = 1 a bijection at every row count; at 96 rows and 2 splits 32 pairs missed, 32 CTAs out of range, row 127 reached; the expressions of the 256- and 128-row kernels agree with the enumeration (Qwen3.5 gated attention decode)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E14 [2026-11, gfx950, Triton 3.8.0, N=1, the Qwen3.5 8-row kernel compiled for gfx950 without a launch, one constexpr changed at a time, the pre-QK wait read from the ISA] the Qwen3.5 8-row kernel: unmodified and with its early-V switch off the same ISA, `vmcnt(4)` with the 4 slot-prefetch loads and no V copy outstanding; with `PREFETCH_STAGE=1` or `0`, `vmcnt(4)` with the 4 V copies outstanding
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the 2 GiB-range runs N=2), GPU runs with values read back] an unmasked `buffer_load` of offsets 64–79 of a 64-element view returned 64.0–79.0, and the ISA held `s_mov_b32 s2, 0x7ffffffe`; fp32 loads at byte offsets 2^31 to 2^32 − 4 (three 64-lane windows) returned 0 and stores there left the tensor unchanged, with no fault; an offset reaching 2^32 wrapped onto the wrong element, for loads and stores; 1-byte loads at byte 2^31 − 3 and below read their data, at 2^31 − 2, 2^31 − 1, 2^31, 2^31 + 1, 2^31 + 4096 and 2^32 − 1 read 0; 1 KiB at byte 2^31 + 4096 of a 3 GiB tensor read 0 through `buffer_load` from the base, and its data through a scalar base moved 2 GiB (64-bit) or through `gl.load`; `global_load_to_shared` copied the 4 KiB at byte 2^31 + 8192 exactly where `buffer_load_to_shared` read 0; an int32 index product of 2^31 + 4096 read the wrapped element; of the host gates against 2^31 in the library (33 kernels), 46 take the buffer path at every size their kernel is built for and none takes the other path, 18 were not evaluated (test kernels on buffers up to 3 GiB; the gates counted in the library's sources)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compile-only, the copies also run] copies into `SwizzledSharedLayout(8, 1, 8)` (bf16 `[64, 64]`) and `(16, 1, 8)` (fp8 `[64, 256]`): 2 and 4 `ds_bpermute_b32` for 2 and 4 copies; the fp8 copy, wait, relaxed read and store: 4 `ds_bpermute_b32` and 76 instructions, pre-swizzled into a linear `_reinterpret` view 0 and 56; async copies from pre-swizzled global columns into a flat alias, read through `(8, 1, 8)`, `(8, 2, 8)` (int16 32×64), `(16, 4, 4)` (int8 64×64), `(16, 1, 8)` (int8 64×128): equal to the source on the MI355X; `compute_efficient_padded_shared_layout` for the bf16 A operand of a 16×16×32 MFMA v4, warps [2, 2], k_width 8: `[32, 64]` → `[[512, 16]]`, `[8, 64]` → `None`; copies laid out from `offset_bases` into the helper's layout, 4 warps, A operands bf16 `[32, 64]` and uint8 `[64, 128]`: only `buffer_load_dwordx4 … lds` (1 and 2 per lane), tile and MFMA operand exact on the MI355X; counted in the library's sources: XORed offsets in all 6 DSA copies (4 kernels), 8 of 11 gated-attention copies (2 kernels), 92 of 231 fp8 GEMM copies (10 kernels); the gated attention at 8 rows copies K and V into `max_phase` 8 layouts (test kernels and the library's sources)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] Qwen3.5 gated attention decode: in one run lowest 1.066 at 256 rows, highest 2.182 at 1 row; a later run of the ruler: 1.085 and 2.221 (QKV dim 4608, 1 KV head of dim 256, fp8 group-quantized output)
- S1 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:71 `buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="")`: 128 or 32 bits per lane, coalesced writes, a swizzle within a warp (:91-94); `commit_group` (:127); `wait_group(num_outstanding)`, until at most that many committed groups are outstanding (:137-140); `load_shared_relaxed` (:151) tags its load `ttg.amdg.syncedViaAsyncWait` (:164)
- S2 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp:11 Membar drops the dependency between an async copy and a local load synced via an async wait on the same buffer, in either order
- S3 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:789 a swizzled direct-to-LDS destination makes the lowering swizzle the source, with `shuffleIdx` across lanes (:419); a masked copy has no branch only without `other` and with warp-uniform predicates (:848); with `other`, a branch (:859) and a store of `other` into LDS (:870)
- S4 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:263 mask alignment caps `vec`; `other` packed per vector 697-705; swizzled `dest` → source swizzling 788-803; DMA `other` → branch + LDS store 840-874; redundant-thread store predicate 1646-1657; `async_wait` → `wait.asyncmark n` 2412-2431; commit → `asyncmark` 2503-2505
