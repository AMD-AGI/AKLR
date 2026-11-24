---
title: "Qwen3.5 gated attention in Gluon on gfx950: K and V into LDS, the fused merge launch, the prologue"
kind: topic
ops: [attention]
dtypes: [bf16, fp8]
regimes: ["one query token per sequence", gqa, split-kv]
gens: [gfx950]
triton: 3.8.0
---

# Qwen3.5 gated attention in Gluon: K and V into LDS, the merge launch, the prologue (gfx950)

Three parts of the Qwen3.5 gated-attention kernels that [../attention.md](../attention.md) §6 describes: 9 Gluon
kernels for MI355X, one per row count from 1 to 256, with one query token per sequence and 8 query heads on one fp8 KV
head, each a prologue launch, a split-KV partials launch and a finish launch. This page covers how the partials launch
gets K and V into LDS (§1), the gate and the fp8 group quantization fused into the finish (§2), and the prologue (§3).
The op, the sizes, the producer and its merge, the split count and the precisions are on that page, whose §8, §9 and
§11 point here. The 9 kernels belong to the library (154 Triton and Gluon kernels for gfx950:
[../kernels/kernel_library.md](../kernels/kernel_library.md) §1; this family: §12). The A/B call times come from runs
of the ruler ([../tuning.md](../tuning.md) §1); numbers were measured on these Gluon kernels unless the sentence says
otherwise.

## Route here when

- You stage gathered K and V rows in LDS for the MFMA: async copies in commit groups and what the wait before QK
  leaves in flight, pre-swizzled source offsets, relaxed reads, V through padded LDS, cache bits per size (§1).
- You put work that follows attention (an output gate, a group quantization) into the merge launch (§2).
- You write the launch before attention: Q/K RMSNorm, RoPE or mRoPE and the fp8 KV-cache write as a grid of one-wave
  programs (§3).

Elsewhere: the split-KV producer and its merge, the grids, the split count and the precision of QK and P·V,
[../attention.md](../attention.md) §2, §6, §7, §10; DSA sparse MLA, the same page's §3–§5; async copies, waits and LDS
in general, `../gluon/memory.md` §8–§11; the mechanisms measured on FlyDSL kernels, `../../attention/INDEX.md`.

## 1. K and V into LDS

| Rows | K | V |
|---|---|---|
| 1–4 | gathered into registers | gathered into registers |
| 8 | async copy into a swizzled LDS tile | async copy into a second LDS tile; with the next tile's slot prefetch after V's commit the pre-QK wait leaves no V copy outstanding *(field, E16, E22)* |
| 16–64 | gathered into registers, converted to the QK operand | gathered into registers, stored to padded LDS, read as the P·V operand |
| 128–256 | async copy at pre-swizzled offsets (Q joins K's first group) | async copy at pre-swizzled offsets; the compiled pre-QK wait leaves eight LDS-direct copies outstanding, overlap not measured *(field, E16)* |

- **Two K/V commit groups do not guarantee V–QK overlap: the 8-row kernel shows it.** There the next tile's slot prefetch is issued after V's commit (`PREFETCH_STAGE == 2` in the excerpt below), so the compiled pre-QK `vmcnt(4)` leaves only those prefetch loads outstanding, no LDS-direct V copy (the kernel's early-V switch is not read on this direct-to-LDS path); with `PREFETCH_STAGE=1` or `0` the same wait leaves the 4 V copies outstanding *(field, E16, E22)*; one group uses `vmcnt(0)`. One group took about 1.019× the call time of two (2 runs, 8 pairs; 22.33–22.35 against 21.93–21.94 µs; accuracy passed), no evidence of V overlap or a resolved mean difference. At 128 rows the run-mean ratios were 1.167–1.169 (2 runs, 8 pairs; accuracy passed); at 256 1.102–1.108 (6 runs, 24 pairs; accuracy passed in 5 of the 6 runs). There the compiled pre-QK `vmcnt(8)` leaves eight LDS-direct copies outstanding, none with one group; the overlap at run time and its share of the gain were not measured *(field, E16)*. The 8-row kernel's K and V copies, with the prefetch stages (0 is after the K read):

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

  The 256-row kernel after its pre-swizzled offsets:

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

  The partial tile passes the mask `valid[:, None]` and no `other`: masked lanes read 0 with no branch. `other` adds a
  branch per copy and, by build, a store of `other` (S33): the MI355X image's Triton 3.8.0 made none, so masked rows
  kept stale LDS (`../gluon/memory.md` §4). The copy needs 128 or 32 bits per lane and coalesced writes; `wait_group(n)`
  returns when at most n committed groups of this wave are outstanding (S34); Membar puts an `s_barrier` after it unless
  a barrier comes first (S35; a 4-warp stand-in with its own `gl.barrier()` after the wait still got one; the 8-row kernel also had one barrier after `wait_group(1)` in the compiled launch; `../gluon/memory.md` §8, §11).
- **Pre-swizzle the global side, copy into a linear alias.** A direct-to-LDS copy writes LDS lane by lane in order
  (`../../hardware/gfx950/memory_and_sync.md` §3); into a `SwizzledSharedLayout` with `max_phase` above 1, Triton swizzles
  the source instead, one `ds_bpermute_b32` per copy (S33; compiled at `max_phase` 8). The 128- and 256-row kernels and
  the DSA 8-head kernels ([../attention.md](../attention.md) §3) XOR the global column with the consumer's swizzle, copy into a linear alias and restate the
  16-byte runs with `gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])`; the 8-row kernel pays the
  shuffles. Code and cost: `../gluon/memory.md` §5, §9; the swizzle to match: `../gluon/layouts.md` §4. Untimed.
- **Read async-copied tiles with a plain `.load()`.** The 128- and 256-row kernels (Q, K) and the DSA 8-head kernels
  use `gl.amd.cdna4.async_copy.load_shared_relaxed`, whose tag makes Membar drop the barrier between that read and
  any async copy into the same buffer, in either order (S34, S36). Compiled, a straight-line kernel copying K and V as
  two groups and reading K after `wait_group(1)` gets the same `s_waitcnt vmcnt(4)`, `s_barrier` and instructions with a
  plain `.load()`. With the relaxed read, the next copy into a buffer relies on the barrier after `wait_group(0)`:
  a K prefetch placed above that wait compiles ahead of the K reads with no `s_barrier`, so it can overwrite K while
  other waves still read it (inferred; `../gluon/memory.md` §11).
- **V through padded LDS.** The 16–64-row kernels store the gathered V `[TILE, 256]` in the padded layout that
  `gl.amd.cdna4.compute_efficient_padded_shared_layout(b_layout, [TILE, 256], gl.float8e4nv, is_k_contig=False)`
  returns, and read it back as the P·V operand (a stand-in matched torch *(field, E13)*). Every producer that
  stages V in an LDS tile (all DSA producers, the Qwen3.5 ones from 8 rows), padded or swizzled, reads it with gfx950's
  transposing `ds_read_b64_tr_b8`, 16 to 80 per producer *(field, E8)*; the instruction is in `../../attention/lds.md` §1.
- **Cache bits are chosen per size.** Qwen3.5 loads V `.cg` at 32 and 64 rows, K `.cg` at 64; the 64-head DSA
  producers store partials `.cs`, read `.cg` by the 8- to 16-token merges and unhinted at 24 and 32 tokens;
  the 8-head kernels copy the RoPE panel `.cv`, store any partials `.wt`. Without V's `.cg` at 32 rows or K's
  at 64 the call ran 0.6–1.9% faster *(field, E7)*: test them at your size (`../gluon/memory.md` §6, `../tuning.md` §11).

## 2. The gate and fp8 group quantization in the merge launch

The finish merges the splits of a (row, head) ([../attention.md](../attention.md) §2), then applies the gate and quantizes each group of 128 channels while
the merged values are in registers. The 16-row kernel's epilogue:

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
  waves an LDS round trip and an `s_barrier` (compiled); the finishes map groups per size.
- **Keep the reference's rounding points.** The reference rounds the attention output and the gated product to bf16;
  the kernels also round the sigmoid to bf16, which the reference does not, and pass *(field, E3)*; without the
  two roundings the 32-row kernel still passed but moved 2060 of 65,536 fp8 codes *(field, E13)*. Quantization forms
  (`gl.div_rn` here, `values * (1 / scale)` at 128 and 256 rows), the clamp and the op's accuracy bounds: `../numerics.md` §2.
- **Drop the bf16 `Gated` store when you copy this.** Every finish writes it, and the host code discards it.
- **In the 256-row finish the compiler kept the gate load first.** The 256-row finish loads the gate before the statistics; moved
  below the merge, the gate load still issues first and the finish takes 0.98–1.00× as long over 4 runs *(field, E2)*.

## 3. The prologue

The first launch RMS-normalizes Q and K, applies mRoPE, writes K and V into the e4m3 pools and copies the gate, in
one-wave programs that each own one job of a row. The 32-row kernel's head job begins:

```python
    variance = gl.sum(source * source, axis=0) / 256.0
    normalized = (
        source * gl.rsqrt(variance + eps) * (1.0 + weight)
    ).to(gl.bfloat16)
    normalized_float = normalized.to(gl.float32)

    half: gl.constexpr = rotary_dim // 2
    partner = gl.gather(normalized_float, offsets ^ half, axis=0)
```

- **A head per wave or sub-wave**, so the RMS sum stays in the wave: `BlockedLayout([1], [64], [1], [0])` for one head,
  threads per warp `[heads_per_wave, 64 // heads_per_wave]` for 2 or 4.
- **Long jobs on low program ids:** two prologues' comments read "Schedule the longer KV-cache work before the query-only programs" and "Issue the
  longer cache-scatter programs first in the single launch". A 16384-program 1-D toy grid had id/start-rank
  correlation 0.9968–0.9982; its 256 jobs 40× longer than the rest finished in 178.1–180.1 µs on low ids versus
  243.4–244.7 µs on high ids (26.4–26.9% sooner, N=4). Resident-at-once grids and the Qwen3.5 prologue were not tested
  *(field, E14)*. At 8 rows four programs per token each load the whole K row for the RMS but
  write only their 64 channels of K, V and both pools; only the first quarter rotates.
- **NeoX partner by XOR:** channel c < 64 pairs with c ^ 32. A 64-value one-wave gather compiles to one
  `ds_bpermute_b32`; the 256-value normalized-head gather in the 32-row prologue compiles to four, with no LDS
  (`../gluon/wave_ops.md` §5). `gl.gather` needs an index of the source's rank and returns the index's layout (S43): a
  2-D source takes `gl.broadcast((columns ^ 32)[None, :], source)[0]`. One 64-lane load of 32 cosines then 32
  sines and the same gather give every lane both. With 4 heads per wave the 256-row kernel pairs the halves in
  registers with `gl.amd.slice`, `split` and `join`; `gl.amd.slice` takes whole CTA tiles only (S45).
- **mRoPE axis from constants.** Frequency f reads `positions[1]` when `f % 3 == 1 and f < 33`, `positions[2]` when
  `f % 3 == 2 and f < 30`, else `positions[0]`. Encodings include two `uint32` masks the host builds from
  `mrope_section`, the same masks as Python ints, written signed as 0x92492492 does not fit an int32, and a
  `uint64` table of two-bit axis ids; each gives the rule, and every kernel's key cache follows it. The two masks in
  the 32-row prologue, one bit per frequency:

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
- **The reference's rounding points.** The RMSNorm result rounds to bf16 before RoPE, the 192 unrotated channels included.
  The cache value is `gl.div_rn(x, scale)` rounded to bf16, which the fp8 store then converts; with
  the unit scales the host code passes, the division is exact, and every kernel's caches equal that chain bitwise: without
  one bf16 rounding or the other, 948 or 1028 of 130,816 key bytes differ *(field, E13)*.
- **Prove every int32 index intermediate and the final valid index, not just buffer size.** In two runs, bf16/fp32 buffers larger than 2^31 bytes gave identical `gl.load`/`tl.load` results with int32 and int64 element indices; int8 `row * stride = 2^31 + 4096` wrapped to −2147479552 and read 62 instead of −86 from a mapped view. `buffer_load` returned zero at all three tested out-of-byte-range locations *(field, E17)*. Stores, negative-index behavior and complex intermediate overflow were not tested by this comparison.
- **Drop the dead outputs when you copy one:** the bf16 K, and in some kernels a V copy, are written and never read.

## Evidence

Triton paths are `triton-lang/triton@c01b677` (tag v3.8.0). The E and S lines keep the ids they have on [../attention.md](../attention.md); a line both pages cite appears on both. "The ruler" is the one [../tuning.md](../tuning.md) §1 defines; Kineto read its kernel durations here, and the call times quoted are medians over replays; an "unmodified" kernel or arm is the kernel as [../attention.md](../attention.md) §6 describes it.

- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (DSA) and N=4 (Qwen3.5) runs, the ruler, 4 alternated pairs per arm in each run, one GPU] DSA 64 heads at 24 tokens (192 producers of 8 warps, 163,840 B of LDS each): the 32-head tile at 8 splits (384 programs, 1 tile each, an 8-way merge) 22.07, 22.03 and 21.87 µs against 20.27, 20.30 and 20.31 µs at 4 splits (192 programs, 2 tiles each, a 4-way merge), ratios 1.089, 1.085 and 1.077, the two arms timed in separate A/B sessions whose unmodified arms agree (18.66–18.80 and about 18.69–18.85 µs); no 4-wave producer and no other token count timed. 32 tokens, refill off (`v_mfma` 82 → 82, accuracy passing): 21.68 → 21.83, 21.72 → 21.77 and 21.62 → 21.47 µs (B/A 1.007, 1.002, 0.993), the off arm's rounds 21.71–21.93, 21.69–21.87 and 21.36–21.52 µs against 21.63–21.74, 21.67–21.78 and 21.59–21.70. Qwen3.5 at 4 rows: the MFMA attention 21.77, 21.64, 21.69 and 21.59 µs against 26.88, 26.79, 26.95 and 26.94 µs for the same kernel forced to a program per (row, head, split) (B/A 1.235–1.248, all 16 pairs 1.230–1.255); at 1 row the per-head attention 16.56, 16.42, 16.44 and 16.57 µs against 16.41, 16.22, 16.45 and 16.28 µs with the 4-row kernel's MFMA attention and merge (MFMA/per-head 0.982–1.001, pairs 0.976–1.014, the MFMA faster in 14 of 16), which resolves neither a small advantage nor a crossover; at 256 rows the gate load and sigmoid moved below the merge: finish 6.13, 6.18, 6.20 and 6.11 µs against 6.21, 6.18, 6.26 and 6.25 µs unmodified (0.978–1.001, pairs 0.965–1.024, the moved order faster in 14 of 16), outputs bitwise equal, the gate load still issued before the statistics load (instruction 24 against 26; unmodified 19 and 28)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, speedups as tuning.md §1 defines them] Qwen3.5 gated attention with one query token per row (qkv_dim 4608, 1 KV head, head_dim 256, 4096 keys per row on average), accuracy passed at each: 2.182 at 1 row, 1.699 at 2, 1.671 at 4, 1.803 at 8, 1.708 at 16, 1.789 at 32, 1.401 at 64, 1.204 at 128, 1.066 at 256; in a second run of the ruler 2.221, 1.685, 1.668, 1.842, 1.729, 1.785, 1.398, 1.206 and 1.085
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the ruler, 3 alternated rounds per arm in each run, one GPU] Qwen3.5, `.cg` removed: V at 32 rows 44.3 → 43.4 and 44.1 → 43.5 µs, K at 64 rows 84.3 → 83.7 and 84.1 → 83.6 µs; noise floor not measured
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel run once on its own Triton cache, the cached `.amdgcn` and `.json` parsed; LDS liveness variants compiled without a GPU] producers of the 8 DSA and 9 Qwen3.5 kernels, each a single compiled variant with no spills: LDS bytes, VGPRs and AGPRs as in `../attention.md` §3 and §6 (DSA 64 heads at 8, 12, 16, 24 and 32 tokens 147,696 B / 150 with 22 AGPRs, 156,160 / 88, 156,160 / 94, 163,840 / 228, 157,696 / 214; DSA 8 heads at 32, 128 and 256 tokens 49,216 / 154, 79,872 / 199, 83,968 / 202; Qwen3.5 at 1–256 rows 4 warps each, 1,024 / 130, 1,024 / 130, 32,768 / 182, 34,816 / 125, 16,624 / 136, 33,264 / 188, 33,264 / 192, 69,632 / 147 with 63 AGPRs, 71,680 / 172); `ds_read_b64_tr_b8` 64, 32, 32, 32, 64 in the 8- to 32-token 64-head producers, 32, 16, 16 in the 8-head ones, 16, 32, 64, 64, 80, 64 in the Qwen3.5 ones at 8–256 rows, none at 1–4 rows; MFMA opcodes: `scale_f32_32x32x64_f8f6f4` with `f32_32x32x16_bf16` at 16–32 tokens and `f32_32x32x16_f16` at 12, `scale_f32_16x16x128_f8f6f4` with `f32_32x32x16_bf16` at 8, `f32_16x16x32_bf16` alone in the 8-head producers and the Qwen3.5 ones from 8 rows, with `f32_16x16x32_f16` in the 4-row one, none in the 1- and 2-row ones; the 24-token producer with `waves_per_eu` 2: 228 VGPRs, unset 222, 163,840 B either way; the 12- to 32-token 64-head producers 8 warps each, and the 24-token one at 165,888 B with P allocated before the second key reduction
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, kernels run on the GPU: the unmodified kernels and ablated copies whole against a float64 copy of the op's reference, stand-in Triton and Gluon kernels against torch or float64] Qwen3.5 32 rows, rows of 1–4096 keys: records merged in float64 within 5.04e-4 of the float64 softmax (rows of 2049–4096 keys); 209 empty splits, each max −inf, denominator 0, numerator 0; no max guard, values 36.09 dB and scales 50.87 dB (bounds 15, 20); without the bf16 rounding of the attention output and the gated product 35.13 dB, 2060 of 65,536 codes changed, mismatches against the reference 2155 → 2545. DSA 24 tokens: output within 1.84e-3 (bf16 rounding of the reference alone 1.76e-3); −1 tails over tiles, a split, a single key and a whole token: 0 non-finite values, the all −1 token exactly 0, 55.71 dB (bound 20); both guards removed: that token NaN in every head and channel, and the accuracy check on the ruler's distinct slots still passes; at 32 tokens with only the producer guard removed: NaN in the token whose first split has two empty tiles, no other. Both kernels: bitwise over 20 runs; NaN- and zero-filled scratch bitwise equal; a graph captured on one set of slots or lengths and replayed on another equal to eager. The 256-row kernel at 128 rows: no error, values −1.31 dB (the 128-row kernel 34.96 dB, the 256-row one at 256 rows 34.73 dB). All 9 prologues with unit scales: key and value caches equal the reference's chain bitwise (0 of 130,816 bytes); without the bf16 rounding before RoPE 948 key bytes differ, before the fp8 store 1028. Stand-ins: P·V: largest relative error of fp16 P 4.865e-4 over P ≥ 2^−14 (282 of 131,072 values below), of bf16 hi + lo P 7.606e-6; a port of sglang's DSA kernel (p × 448 in e4m3), N(0, 1) inputs, 64 heads: 36.57 and 36.54 dB at 8 and 32 tokens; a float64 emulation of Qwen3.5's baseline 22.33, 22.44 and 22.62 dB at 1, 8 and 32 rows; bf16 P: 54.3 dB (DSA) and 38.7 dB (Qwen3.5 values); bf16 partials with fp32 statistics in a split-K merge: error against float64 8.0e-4 → 1.51e-3 (DSA, 8 splits) and 7.5e-4 → 1.79e-3 (Qwen3.5, 16 splits); one fp8 [64, 128] tile in 8192 B of LDS read for QK through `.permute((1, 0))` and for P·V: within 1.756e-3 of torch; V in `compute_efficient_padded_shared_layout` at TILE 32 and 64, read with `ds_read_b64_tr_b8`: P·V within an absolute 1.9e-6 of torch
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, Triton probes timed by CUDA-graph replay or kernel time without a flush; a tile stream in GB/s, median of rounds; the router by the ruler, 4 alternated rounds] At 256 Qwen3.5 rows (512 MiB of K and V per call, slots drawn with replacement) the partials kernel 78.6, 78.3, 78.9 and 79.42 µs (6.83, 6.85, 6.80 and 6.76 TB/s) and the baseline's paged attention 144.8, 145.0, 145.1 and 144.53 µs (3.71, 3.70, 3.70 and 3.71 TB/s), against a flushed gather of a random permutation of 256-byte rows at 5.40, 5.37, 5.11 and 5.12 TB/s; 8 rows, one 55× longer than the others, 4 splits per row (32 programs) 51.19, 50.95, 51.21 and 50.74 µs against 7.34, 6.56, 6.13 and 10.56 µs for 62 equal chunks (6.97×, 7.76×, 8.36×, 4.81×), replayed; a 16384-program 1-D grid: program id against start rank, correlation 0.997–0.998, and 256 jobs 40× longer than the rest finished in 178.2, 178.4, 178.1 and 180.1 µs on the lowest ids against 243.4, 243.5, 243.6 and 244.7 µs on the highest; a 64 × 64 bf16 tile stream at 1024 and 2048 programs: async copies 5.50–5.93 TB/s, register-only loads 5.61–6.01, async over register-only 0.957–0.987 at 2048 programs and 0.983–1.008 at 1024 (two async modes); an 8-token MoE router with register loads in place of async copies 8.417 against 8.383, 8.410 against 8.421, 8.422 against 8.295 and 8.667 against 8.595 µs (B/A 1.004, 0.999, 1.015, 1.008) (the Qwen3.5 256-row kernel; the rest toy Triton and Gluon kernels)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 runs for the 4-row kernel's branches at 2 and 8 rows, N=2 for the unmodified 2-row kernel against its replacement and for the 8- and 128-row group comparisons, N=6 at 256 rows; 4 pairs per run in a fixed A/B order, the ruler; ISA of the compiled launches] no same-size A/A, so no mean-difference uncertainty. GQA: at 8 rows the same kernel's branches, per-head over MFMA 1.975–2.019 (MFMA 27.74–28.11 against per-head 55.08–55.89 µs); at 2 rows MFMA over per-head 1.025–1.067; a replacement MFMA attention source over the unmodified 2-row kernel 0.988–1.005 (the merge changes too); every timed arm accurate. Commit groups, one over two: 8 rows 22.33–22.35 against 21.93–21.94 µs (about 1.019); 128 rows 138.9–139.2 against 118.7–119.2 µs (run means 1.167–1.169); 256 rows 1.102–1.108 over 6 runs and 24 pairs (the 4 runs of E5 on `../attention.md` among them), accuracy passed in 5 of the 6 runs. Compiled pre-QK waits: the 8-row kernel (its early-V switch on) `vmcnt(4)` with two groups and `vmcnt(0)` with one, no LDS-direct copy outstanding in either; 128 and 256 rows `vmcnt(8)` and `vmcnt(0)`, 8 and 0 copies outstanding
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, 64-element loads per dtype and index variant] bf16 and fp32 buffers of 2147500032 and 2147516416 bytes, index products 1073745920 and 536875008 elements: `gl.load` and `tl.load` with int32 and int64 indices match the reference; an int8 product of 2147487744 wraps to −2147479552 and returns 62 rather than the reference's −86; `buffer_load` with int32 and uint32 offsets returns zero at all three out-of-byte-range locations, in-range controls read data (toy Triton and Gluon kernels)
- E22 [2026-11, gfx950, Triton 3.8.0, N=1, the Qwen3.5 8-row kernel compiled for gfx950 without a launch, one constexpr changed at a time, the pre-QK wait read from the ISA] the Qwen3.5 8-row kernel: unmodified and with its early-V switch off the same ISA, `vmcnt(4)` with the 4 slot-prefetch loads and no V copy outstanding; with `PREFETCH_STAGE=1` or `0`, `vmcnt(4)` with the 4 V copies outstanding
- S33 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:789 a swizzled direct-to-LDS destination makes the lowering swizzle the source, with `shuffleIdx` across lanes (:419); a masked copy has no branch only without `other` and with warp-uniform predicates (:848); with `other`, a branch (:859) and a store of `other` into LDS (:870)
- S34 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:71 `buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="")`: 128 or 32 bits per lane, coalesced writes, a swizzle within a warp (:91-94); `commit_group` (:127); `wait_group(num_outstanding)`, until at most that many committed groups are outstanding (:137-140); `load_shared_relaxed` (:151) tags its load `ttg.amdg.syncedViaAsyncWait` (:164)
- S35 triton-lang/triton@c01b677:lib/Analysis/Membar.cpp:292 a barrier after an async wait unless a sync point follows before the next memory access
- S36 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp:11 Membar drops the dependency between an async copy and a local load synced via an async wait on the same buffer, in either order
- S43 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:559 `gather`: equal ranks (:566), result in the index's layout (:579)
- S45 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/slice.py:12 `gl.amd.slice`: register-only, extent and offsets aligned to the layout's CTA tiling (:17-18)
