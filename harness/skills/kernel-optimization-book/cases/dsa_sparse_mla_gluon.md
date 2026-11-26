---
title: "DSA sparse MLA in Gluon, one query token per sequence over 2048 gathered fp8 cache rows of 576 values, 64 or 8 heads, bf16 out — key splits written as records and merged in a fixed order by a separate launch; one workgroup per token with no merge at 8 heads and 256 tokens"
kind: case
ops: [mla, attention]
dtypes: [fp8, bf16]
regimes: ["tokens 8-32 (64 heads)", "tokens 32-256 (8 heads)"]
accuracy: bitwise
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup over SGLang's triton_sparse_mla_decode_splitk, per size, a reference run of the ruler", value: "64 heads 1.030 (8 tokens), 1.699 (12), 1.840 (16), 2.188 (24), 1.966 (32); 8 heads 1.062 (32 tokens), 1.196 (128), 1.050 (256)", gpu: MI355X, workload: "fp8 q (tokens, heads, 576) and cache (rows, 576), top-k 2048, bf16 output (tokens, heads, 512); caches of 65536 rows, 262144 at 128 tokens and 524288 at 256", date: "2026-11"}
---

# DSA sparse MLA in Gluon, one query token per sequence: 64 heads at 8–32 tokens, 8 heads at 32–256

Eight Gluon kernels for one op, the sparse MLA of SGLang's DSA backend with one query token per sequence, five for 64
heads and three for 8: a family of the Triton and Gluon kernel library that
[triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps. That page says which family to start
from; [triton/attention](../triton/attention.md) §2–5, §10 and §12 explain the techniques. This page has the family's
kernels by size, their code, and what was measured. The FlyDSL kernels of the op are on
[dsa_sparse_mla_8_heads.md](dsa_sparse_mla_8_heads.md) (8 heads) and
[dsa_sparse_mla_16_heads.md](dsa_sparse_mla_16_heads.md) (16 heads).

The op:

Each token's 64 or 8 heads attend 2048 rows of an fp8 latent cache (576 values: 512 latent and 64 RoPE) named by
top-k indices, q in e4m3, `kv_b_proj` absorbed, bf16 out; 8 kernels, one size each (the later additions are 9 more 8-head kernels
and a 16-head op, [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §2). The baseline is SGLang's
`triton_sparse_mla_decode_splitk`. Triton side: [triton/attention](../triton/attention.md) §3, §4, §5; op card:
[../attention/INDEX.md](../attention/INDEX.md).

## Applies when

- The op is the one above: q `[tokens, heads, 576]` fp8 e4m3 (512 latent channels, then 64 RoPE channels), the cache
  `[rows, 576]` fp8, 2048 slots per token (−1: none), the output `[tokens, heads, 512]` bf16, 0 for a token that reads
  no row; there are no scales. By the kernel sources' own descriptions it is GLM-5.2's attention at 64 heads per
  rank (TP1) or 8 (TP8).
- The path replaced is SGLang's `DeepseekSparseAttnBackend._forward_triton_decode` (sglang 0.5.22, speculative
  decoding left out); a speedup is against the op's baseline, SGLang's split-K Triton kernel
  `triton_sparse_mla_decode_splitk`.
- A kernel is written for one head count and one token count: 64 heads at 8, 12, 16, 24 and 32 tokens, 8 heads at 32,
  128 and 256 (§ Mechanism). Another count has no kernel in this family; the kernels added to the library since (16
  heads; at 8 heads more token counts, and a second kernel at each of 32, 128 and 256 tokens) are listed in
  [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §2 and not described here.
- The family is held to `bitwise` accuracy: every run on the same inputs gives the same bytes as the first. No kernel
  has an atomic, and 40 runs of each at its size were byte-identical *(field, E1)*. The output's error
  bound against the reference is separate; every size passed it *(field, E2)*.

## Counter-signals

- **The thinnest margins** are 64 heads at 8 tokens (1.030), 8 heads at 256 tokens (1.050) and 8 heads at 32 tokens
  (1.062) *(field, E2)*. These runs counted a size as won from 1.03 over the baseline, those after 2026-11-12
  from 1.05 ([triton/tuning](../triton/tuning.md) §1).
- **The 8-head kernels are not the fastest measured at their sizes.**
  [dsa_sparse_mla_8_heads.md](dsa_sparse_mla_8_heads.md) calls them the older Gluon kernels; its repeat measurement of
  them reads the same 1.062, 1.196 and 1.05 over the baseline *(field, E3)*. In one
  session of that page's ruler, a one-sided lower bound of the per-round time ratio over 16 rounds, an earlier build
  of its FlyDSL kernel measured 1.536, 1.319 and 1.198 over them at 32, 128 and
  256 tokens *(field, E4)*. The two Gluon kernels added to the library since at
  these sizes, that page's newer Gluon kernels, one for 32 tokens and one for 128 and 256, read the fp8 q directly
  and launch no `q.to`; in the same session they measured 1.143 and 1.113 over these kernels at 32 and 128 tokens and
  0.925 at 256, slower there *(field, E4)*. That page's § Result has the times and
  how each kernel splits a token's keys.
- **Do not start a 64-head kernel from the 8-token kernel.** It differs from the 12–32-token kernels in many choices:
  16-head groups of 4 warps, rows async-copied into LDS, bf16 hi + lo P, fp32 partials and the extra `q` cast. It
  measured 1.030 *(field, E2)*; its baseline differs from theirs, and no size attributes the gap to one choice.
- **Copy a size's whole configuration, and read only your size's branch.** A size's constants were chosen together:
  at 24 tokens, 4 splits in place of 8 and nothing else changed make the producer fail to compile (entry 7). The
  sources also keep arms and tables for token counts they are not written for: in the 8-token kernel one profile
  table gives `waves_per_eu` 1 at 8 tokens but is read only at other token counts, while the launch-geometry table
  gives the 2 that the 8-token launch passes ([triton/tuning](../triton/tuning.md) §14); entries 3 and 6 show arms
  that the 24-token kernel does not run at 24 tokens.
- **What the ruler's inputs do not exercise.** Each token's 2048 entries are distinct and uniformly sampled from −1
  and the cache's slots. No tile is empty, so a missing guard is not caught: the 24-token kernel without the guards
  of entries 1 and 3 passes its accuracy check there *(field, E5)*; test against the reference with −1 tails
  covering whole tiles, a split and a token. The entries have no index locality either, so the 8-head kernels'
  live-tile skipping does no work on them; test the locality levers of `../attention/split_kv.md` §7 (measured on
  FlyDSL and Triton kernels) with production-like indices.
- **Every cache the kernels were checked on is within the buffer range**, 2^31 − 2 bytes for the descriptor Triton
  3.8.0 builds: 65536 to 524288 rows of 576 bytes at the measured sizes *(field, E2)*. For a larger cache the
  24-token kernel takes its `gl.load` path (entry 5); the host code of the 256-token 8-head kernel asserts that the
  cache's last row ends within 2^31 bytes, and the `INT_MIN` offsets of entry 4 stay out of range only for 1-byte
  elements.
- **The 64-head grids are measured choices at 12–32 tokens; residency is not measured.** At 12, 16 and 32 tokens the
  kernels' 192, 256 and 256 producers beat 96, 128 and 512 (modified over unmodified call times 1.414, 1.456 and
  1.098, one session of 6 counterbalanced rounds per arm) *(field, E6)*, and at 24 tokens 384 same-head
  producers ran 1.08–1.09× slower than 192 (3 runs) *(field, E7)*. The changed arms also change tiles and merge
  width, at 16 tokens the P·V mode and the packed statistics, and at 32 tokens the P·V mode and the refill. LDS
  predicts at most one such workgroup per CU, but residency and its timing contribution were not measured
  ([triton/attention](../triton/attention.md) §3).
- **Settings without a measured gain.** Without `waves_per_eu=2` the 24-token call took 18.71 µs against 18.89 µs,
  faster in each of the 4 alternated pairs *(field, E8)*. Without the 32-token kernel's refill, which gathers
  tile 1's rows during tile 0's P·V, the call moved by −0.7% to +0.7% over 3 runs *(field, E7)*.
  `allow_flush_denorm=True`, set on the launches of the 12–32-token kernels, changes no instruction of the 24-token
  producer or merge (compiled with Triton 3.8.0 for gfx950). Removed at 12, 16, 24 and 32 tokens, it moved the
  speedup by a factor of 1.0007, 1.0033, 0.9986 and 0.9970 (without over with), accuracy passing in every compared
  arm and the 8 compiled kernels identical with and without it *(field, E9)*.
- **Raw `vmcnt` waits at 8 heads, 128 and 256 tokens.** These kernels wait for part of a commit group with an inline
  `s_waitcnt vmcnt(N)`. N counts the copy instructions issued after a panel and goes stale silently when those
  change, where `wait_group` recounts: at 128 tokens, dropping one prefetch leaves a panel in flight (compiled).
  Re-derive N from the compiled code after any change ([triton/gluon/memory](../triton/gluon/memory.md) §11,
  [triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §8).
- **The 24-token kernel is two source files**: its key reduction (entry 6) is a sibling module that the kernel
  imports relatively. Compiled outside a package, such a kernel fails with "attempted relative import with no known
  parent package"; put its sources into one package first ([triton/authoring](../triton/authoring.md) §10).

## Mechanism

| Tokens (heads) | Kernels | Launches | Decomposition |
|---|---|---|---|
| 8 (64) | one | 3 | `q.to(torch.bfloat16)`; a producer of 256 programs × 4 warps; a merge of 512 programs |
| 12–32 (64) | a kernel per count (12), (16), (24), (32) | 2 | 192–256 programs of 8 warps per (token, 32- or 64-head tile, key split), 256 gathered rows per LDS tile (two tiles per program at 32 tokens, which has 4 splits), QK on `mfma_scaled` e4m3 with unit scales; a merge launch |
| 32, 128 (8) | a kernel per count (32), (128) | 3 | `q.to`; a producer; a merge |
| 256 (8) | one | 2 | `q.to`; one workgroup per token over all 2048 keys, no merge |

- **Techniques.** Head tile and split count follow the token count; at 64 heads the grid is 192–256 programs, one
  LDS-heavy workgroup per CU ([../attention/split_kv.md](../attention/split_kv.md) §4, §5).
  Partials merged by a separate launch in a fixed order, scaled by `exp(m_s - max)` with a guarded `1 / den`
  (bitwise output, [triton/attention](../triton/attention.md) §2); enough tokens to fill the machine drop the merge. Four kernels
  (64 heads at 8 tokens; 8 heads at 32, 128 and 256) convert `q` in `run`, one more kernel per call.
- **Resources** *(field, E10)*: 15 compiled kernels; `num_warps` 1 (3), 2 (2), 4 (6), 8 (4); `waves_per_eu` 1 in 2, 2 in 3;
  VGPR 23–228, AGPRs in 1 (22); LDS 0–160 KiB; no spills; bf16 32 × 32 × 16 MFMA in 4, scaled 32 × 32 × 64 in 4, bf16
  16 × 16 × 32 in 3, scaled 16 × 16 × 128 in 1, fp16 32 × 32 × 16 in 1.
- **8 heads** ([triton/attention](../triton/attention.md) §3). At 32 tokens a token's keys are split 16 ways; at 128
  tokens two programs per token feed a 2-way merge; at 256 tokens one program per token runs all 32 key tiles, with
  no partials or merge. The 128- and 256-token kernels stage the rows in async panels per 64 keys, double-buffered,
  take a live-tile bitmap from a wave ballot and wait with raw `s_waitcnt vmcnt(n)`
  ([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §6, [triton/gluon/memory](../triton/gluon/memory.md) §11).
  The MFMA did not bound the 32- and 128-token producers: doubling the QK MFMAs cost 0.6–1.0% at 32 tokens and
  4.2–4.7% at 128 (3 runs) *(field, E11)*; the gather plausibly bounds them (inferred).
- **Precision** ([triton/attention](../triton/attention.md) §10). 64 heads at 12–32 tokens: QK on `mfma_scaled`
  e4m3 × e4m3 with unit scales, `[32, 32, 64]`; bf16 P for P·V (fp16 at 12 tokens); bf16 partial numerators, fp32
  statistics. 64 heads at 8 tokens: QK on `mfma_scaled` `[16, 16, 128]` with the RoPE channels padded to K = 128;
  bf16 hi + lo P, 16 heads' hi over their lo on one `[32, 32, 16]` MFMA; fp32 partial numerators. 8 heads: QK on the
  bf16 `[16, 16, 32]` MFMA (entry 8); bf16 P; bf16 partial numerators, none at 256 tokens.

Per kernel, [triton/attention](../triton/attention.md) §3 tabulates the heads per program × splits, the producer and
merge grids, `waves_per_eu`, LDS and registers.

## Code

The code to start from is the 64-head host code of the 24-token kernel ([triton/attention](../triton/attention.md)
§4). It takes the head tile and the split count from the token count (`rows` in the host code),
`head_tile = 64 if rows == 24 and heads == 64 else 32` and `splits = 8 if rows <= 16 or head_tile == 64 else 4`,
launches the producer on the grid `(rows, heads // head_tile, splits)`, sizes the partial records from the two
(entry 2) and gates the buffer path on the cache's size (entry 5). Entries 1–3 and 5–7 are from the 64-head kernels,
4 and 8 from the 8-head kernel for 256 tokens.

### 1. Merge the splits' records in a fixed order (64 heads, 24 tokens)

The merge launch reads the records of all splits of a (token, head): `maximums` and `denominators` hold each split's
running max m_s and its denominator, `partials` its numerator. It weighs split s by exp(m_s − M), M the largest m_s,
sums in a fixed order and divides once. `safe_max` and the guarded division make a token that reads no row come out
0: the 24-token merge turns 0/0 into 0 *(field, E5)*. `_softmax_exp(x)` is `gl.exp2(x * 1.4426950408889634)`,
2 VALU instructions against 9 for `gl.exp(x)` with one value per lane (compiled with Triton 3.8.0 for gfx950).

```python
    maximum = gl.max(maximums, 0)
    safe_max = gl.where(maximum == -float("inf"), 0, maximum)
    correction = _softmax_exp(maximums - safe_max[None, :])
    denominator = gl.sum(denominators * correction, 0)
    # ...
    numerator = gl.sum(partials * correction[:, :, None], 0)
    inverse_denominator = 1.0 / gl.where(denominator > 0, denominator, 1)
```

Not this: `maximum` in place of `safe_max`. Without the guard a token that reads no row is NaN; with this guard and
entry 3's both removed, the 24-token kernel gave NaN in every head and channel of an all −1 token *(field, E5)*.

Explained in [triton/attention](../triton/attention.md) §2.

### 2. A split's numerator stored straight from the MFMA accumulator (64 heads, 24 tokens)

The 64-head producers at 12–32 tokens store a split's numerator straight from the transposed 32 × 32 accumulator, as
bf16 in `[channel // 8][head][channel % 8]` order, and the merge reads it back 8 channels per lane; the 8-token
kernel's partials are fp32. In the 24-token kernel this store runs once per half of the 512 channels (`START` 0 and
256); `record` is the index of the (token, head block, split), and the host code sizes `Partial` from the head tile,
the split count and `RECORD_VALUE`, a record's channels per head: 512 here, 544 in the 12- and 16-token kernels,
which pad a record. With a 64-head tile the channel group is XORed with `split * 4`, for a reason the source does
not state; no record layout is measured.

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

Explained in [triton/attention](../triton/attention.md) §2.

### 3. The running max guarded in the producer (64 heads, 24 and 32 tokens)

DSA marks an unused slot −1 and its reference outputs 0 for a token that reads no row, so a kernel must survive
empty tiles, splits and tokens. Every DSA producer takes its exponentials against `safe_max`, the running max with
−inf replaced by 0; these lines are in the 24-token producer and, the same, in the 32-token one. `tile_max` and
`tile_sum` are the tile's row max and sum over its keys (entry 6), `valid_scores` marks the keys whose slot is not
−1, `maximum` and `denominator` are the split's running statistics. The `else` arms run from a program's second
tile on: the 32-token kernel has two tiles per program, the 24-token kernel one.

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

Not this: `next_max` in place of `safe_max` in the exponentials. A split whose first two tiles are empty then
rescales by exp(−inf − (−inf)) = NaN in its token; at 32 tokens with only this guard removed, the token whose first
split has two empty tiles was NaN, and no other *(field, E5)*.

Explained in [triton/attention](../triton/attention.md) §2.

### 4. A −1 slot as an out-of-range offset, with no mask (8 heads, 128 and 256 tokens)

The 128- and 256-token 8-head kernels turn a −1 slot into the element offset `INT_MIN` and copy the rows with no
mask: as an unsigned offset it is out of the buffer's range, so the fp8 gather needs no mask. The line is from the
256-token kernel's source: `selected` holds the slots of the program's tiles and `KV_ROW` the cache's row stride, so
`row_offsets` is each row's element offset from the cache's base. The other kernels mask: the 64-head ones clamp a
−1 slot to row 0 and mask it (entry 5), the 32-token 8-head kernel masks its panel copies.

```python
    row_offsets = gl.where(selected >= 0, selected * KV_ROW, -2147483648).to(gl.int32)
```

Not this: the same sentinel for 2- or 4-byte elements. The 32-bit multiply by the element size turns it into 0, and
those lanes read element 0 (S1): use a mask.

Explained in [triton/attention](../triton/attention.md) §5.

### 5. A 256-row tile gathered 16 bytes per lane, then stored to LDS (64 heads, 24 tokens)

The 24-token producer gathers its 256-row × 512-byte tile into registers as one tensor, 16 fp8 values per lane in one
`buffer_load_dwordx4`, then stores it to LDS: 128 KiB per program, 16 loads of 16 bytes per lane, all issued before
the first `ds_write_b128` (compiled for gfx950). `key_offsets` are the tile's 256 keys and `channels` its 512 latent
channels, in `BlockedLayout([1, 16], [4, 16], [8, 1], [1, 0])`; `WIDTH` is a row's 576 values; a −1 slot is clamped
to row 0 and masked. The excerpt passes `0.0` as `other`: masked lanes of a buffer load already read 0, and a passed
`other`, 0 included, costs a select per 32-bit word ([triton/gluon/memory](../triton/gluon/memory.md) §4; not
counted on this kernel); the 32-token kernel's panel gathers leave it out. `CACHE_BUFFER` is the host's gate,
`cache.numel() * cache.element_size() < 2**31`: the kernel takes the buffer path only when its cache is under 2^31
bytes, and loads with `gl.load` from 64-bit offsets otherwise. The tile is stored once and read as the QK operand
through a `.permute((1, 0))` view and as the P·V operand directly.

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

Not this: `buffer_load` with no gate on a cache that can outgrow the range. A buffer access returns data only for an
access, or each dword of a wider one, lying wholly below byte 2^31 − 2 from its scalar base; any other access
silently reads 0 *(field, E12, E13)*.

Explained in [triton/attention](../triton/attention.md) §5.

### 6. A row max or sum across 8 warps, reduced in two levels (64 heads, 32-head tile)

With the head rows in one MFMA tile the producers spread a score tile's 256 keys over the warps
(`warps_per_cta=[HEAD_TILE // 32, 256 // HEAD_TILE]`: `[1, 8]` at a 32-head tile, `[2, 4]` at a 64-head one), so a
row max or sum crosses waves through LDS and `s_barrier`. These producers reduce in two levels by hand: each warp
reduces its keys in registers and writes its partial into a 2 KiB fp32 scratch whose halves hold max and sum; a fold
layout reads the partials back and the reduction finishes in registers. At `HEAD_TILE` 32 that is 2 `s_barrier`
against 7 for `gl.max` and `gl.sum` (compiled); it was not timed against them. The excerpt is the 24-token kernel's
source with the branches for a 64-head tile left out: at 24 tokens that kernel has a 64-head tile and runs those
branches, not the arm shown, and the 12-token kernel, whose key-reduction module is the same, runs the arm shown
with `COMPACT_FOLD` False, the fold layout of the inner `else`. `storage` is a
`[HEAD_TILE, 2 * (256 // HEAD_TILE)]` fp32 LDS buffer, `MAXIMUM` selects the max or the sum and `COMPACT_FOLD` one of
two fold layouts.

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

Explained in [triton/attention](../triton/attention.md) §12.

### 7. The LDS tile that a constexpr chain defines on some branches only (64 heads, 24 tokens)

In the 24-token producer `TILES = cdiv(TOPK, SPLITS * 256)`: 8 splits give each program one 256-key tile (`TILES` = 1)
and the branch that allocates `cache_shared`; 4 splits give `TILES` = 2 and the branch that allocates `cache0` and
`cache1`, while the 64-head path still reads `cache_shared`. That path is written for one 256-key tile per program,
and the host code sets the head tile and the split count together (above). A second tile cannot sit beside the
first, which already takes 160 KiB with P: for two tiles per program start from the 32-token kernel. `VALUE` is the
512 latent channels and `LDS_VECTOR` the swizzled layout's `vec`, 16 in this launch.

```python
TILES: gl.constexpr = triton.cdiv(TOPK, SPLITS * 256)  # TOPK 2048: 8 splits give 1 tile, 4 splits 2
# ...
if TILES > 1:
    cache0 = gl.allocate_shared_memory(
        Cache.dtype.element_ty, [256, 256], gl.SwizzledSharedLayout(LDS_VECTOR, 1, 16, [1, 0])
    )
    cache1 = gl.allocate_shared_memory(
        Cache.dtype.element_ty, [256, 256], gl.SwizzledSharedLayout(LDS_VECTOR, 1, 16, [1, 0])
    )
else:  # (an `elif HEAD_TILE == 32:` before it allocates cache_shared in a padded layout)
    cache_shared = gl.allocate_shared_memory(
        Cache.dtype.element_ty, [256, VALUE], gl.SwizzledSharedLayout(LDS_VECTOR, 1, 16, [1, 0])
    )
for tile in gl.static_range(TILES):
    # ...
    if HEAD_TILE == 64:
        key_rope = gl.convert_layout(rope, score_rhs)
        scores = _wide_query_scores(
            Query,
            cache_shared,  # no branch defined it when TILES > 1
            # ...
```

Not this: 4 splits at 24 tokens with the 64-head tile kept. The producer fails to compile,
`NameError('cache_shared is not defined')`, without a GPU and in each of the 4 timed runs *(field, E8)*; the
16-token kernel guards the combination with `gl.static_assert(TILES == 1 or PANEL_CACHE)`, so that a lone change
fails with a stated condition, and the 24-token one has no such assert.

Explained in [triton/debugging](../triton/debugging.md) §6.

### 8. fp8 keys read from LDS in the operand's lane order and upcast to bf16 for QK (8 heads, 256 tokens)

The 8-head kernels run QK on the bf16 `[16, 16, 32]` MFMA: q is an exact bf16 copy of the fp8 q, made in the host
code, and K is upcast from fp8, exact for all 254 finite e4m3 codes. `_qk_panel`, here from the 256-token kernel,
reads one panel of K from LDS into `_QK_OPERAND`, a blocked layout whose per-lane run of 16 fp8 values equals the
bf16 operand's, converts it and hands it to the B operand layout with `k_width` 16; `score` is the fp32 accumulator.
That hand-over passes with `assert_trivial=True` (not set in the excerpt), the flag that makes `gl.convert_layout`
raise at compile time if it would move data between lanes or waves, so the convert stays in registers, as
`v_cvt_scalef32_pk_bf16_fp8` (compiled with Triton 3.8.0 for gfx950). The kernel calls it for five panels per tile:
four of 128 latent channels and one of the 64 RoPE channels.

```python
_QK_OPERAND = gl.constexpr(gl.BlockedLayout([16, 1], [4, 16], [1, 4], [1, 0]))  # 16 fp8 of K per lane, as the operand

@gluon.jit
def _qk_panel(query, key_smem, score):
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, score.type.layout, 16)
    key = gl.amd.cdna4.async_copy.load_shared_relaxed(key_smem.permute((1, 0)), _QK_OPERAND).to(gl.bfloat16)
    return gl.amd.cdna4.mfma(query, gl.convert_layout(key, rhs_layout), score)
```

Explained in [triton/gluon/mfma](../triton/gluon/mfma.md) §2.3.

## Result

- **Speedups** over the baseline above *(field, E14)*: 1.030 (64 heads, 8 tokens) to 2.188 (64 heads, 24 tokens).

Per size *(field, E2)*: 64 heads 1.030 at 8 tokens, 1.699 at 12, 1.840 at 16, 2.188 at 24 and 1.966 at 32; 8 heads
1.062 at 32 tokens, 1.196 at 128 and 1.050 at 256. A second run of the ruler read 1.737, 1.839, 2.159 and 1.978 for
the 64-head kernels at 12–32 tokens *(field, E2)*. The 24-token value is the highest, but a ratio across sizes
also ranks the baseline, which took 1.72–1.73 µs per token at 24 tokens and 1.37 at 32 (3 runs)
*(field, E11)*: compare two designs within a size.

The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1; the kernel
durations behind these values were read from Kineto.

## Evidence

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, each kernel at its size: 40 runs compared byte by byte, launches traced, kernel time as the sum of Kineto kernel durations, median of 5; sources scanned for atomics] all 17 kernels (8 DSA, 9 Qwen3.5) byte-identical over 40 runs, none with an atomic, semaphore or spin; the DSA producer and merge grids, warps and `waves_per_eu` of [triton/attention](../triton/attention.md) §3 at all 8 kernels; Qwen3.5 3 kernels at all 9 row counts against the baseline's 7, device time 15.6, 21.4, 20.6, 20.3, 26.9 and 43.7 µs against 28.8, 29.6, 30.8, 37.2, 45.6 and 77.8 µs at 1, 2, 4, 8, 16 and 32 rows in the newer run, the older within 1 µs
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, speedups as [triton/tuning](../triton/tuning.md) §1 defines them] DSA sparse MLA with one query token per sequence, accuracy passed at each: 64 heads (pool of 65536 rows) 1.030 at 8 tokens, 1.699 at 12, 1.840 at 16, 2.188 at 24, 1.966 at 32; in a second run of the ruler 1.03, 1.737, 1.839, 2.159 and 1.978; 8 heads 1.062 at 32 tokens (pool 65536 rows), 1.196 at 128 (262144), 1.050 at 256 (524288)
- E3 [2026-11, gfx950, Gluon, N=1, a repeat measurement, ruler not stated in the record] speedups over the baseline of the three older Gluon kernels: at (32, 65536) 1.062, at (128, 262144) 1.196, at (256, 524288) 1.05; accuracy passed
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=1, one-sided Student-t lower bound of the geometric mean of the per-round time ratios, 16 rounds per size of CUDA-graph replays with a 1 GiB read flush before each call, every participant timed once per round in a seeded order] one session on one GPU: an earlier build of the FlyDSL kernel of [dsa_sparse_mla_8_heads.md](dsa_sparse_mla_8_heads.md) and the two newer Gluon kernels, each paired with the baseline (sglang 0.5.22's `triton_sparse_mla_decode_splitk`) and with the older Gluon kernel of its size; medians of 16 rounds and lower bounds, the FlyDSL kernel / baseline / older Gluon kernel: 32 tokens 12.037 / 19.786 / 18.500 µs, bounds 1.642 and 1.536; 128 tokens 27.625 / 44.020 / 36.465 µs, 1.592 and 1.319; 256 tokens 47.363 / 59.979 / 56.798 µs, 1.265 and 1.198; the older Gluon kernels: three, one per size; the newer Gluon kernel for 32 tokens 16.151 µs, bounds over the baseline / the older Gluon kernel 1.224 / 1.143; the newer Gluon kernel for 128 and 256 tokens at 128 tokens 32.895 µs, 1.336 / 1.113, at 256 tokens 61.587 µs, 0.979 / 0.925, slower than both; all six accuracy passed, worst violation fraction 0.0
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, kernels run on the GPU: the unmodified kernels and ablated copies whole against a float64 copy of the op's reference, stand-in Triton and Gluon kernels against torch or float64] Qwen3.5 32 rows, rows of 1–4096 keys: records merged in float64 within 5.04e-4 of the float64 softmax (rows of 2049–4096 keys); 209 empty splits, each max −inf, denominator 0, numerator 0; no max guard, values 36.09 dB and scales 50.87 dB (bounds 15, 20); without the bf16 rounding of the attention output and the gated product 35.13 dB, 2060 of 65,536 codes changed, mismatches against the reference 2155 → 2545. DSA 24 tokens: output within 1.84e-3 (bf16 rounding of the reference alone 1.76e-3); −1 tails over tiles, a split, a single key and a whole token: 0 non-finite values, the all −1 token exactly 0, 55.71 dB (bound 20); both guards removed: that token NaN in every head and channel, and the accuracy check on the ruler's distinct slots still passes; at 32 tokens with only the producer guard removed: NaN in the token whose first split has two empty tiles, no other. Both kernels: bitwise over 20 runs; NaN- and zero-filled scratch bitwise equal; a graph captured on one set of slots or lengths and replayed on another equal to eager. The 256-row kernel at 128 rows: no error, values −1.31 dB (the 128-row kernel 34.96 dB, the 256-row one at 256 rows 34.73 dB). All 9 prologues with unit scales: key and value caches equal the reference's chain bitwise (0 of 130,816 bytes); without the bf16 rounding before RoPE 948 key bytes differ, before the fp8 store 1028. Stand-ins: P·V: largest relative error of fp16 P 4.865e-4 over P ≥ 2^−14 (282 of 131,072 values below), of bf16 hi + lo P 7.606e-6; a port of sglang's DSA kernel (p × 448 in e4m3), N(0, 1) inputs, 64 heads: 36.57 and 36.54 dB at 8 and 32 tokens; a float64 emulation of Qwen3.5's baseline 22.33, 22.44 and 22.62 dB at 1, 8 and 32 rows; bf16 P: 54.3 dB (DSA) and 38.7 dB (Qwen3.5 values); bf16 partials with fp32 statistics in a split-K merge: error against float64 8.0e-4 → 1.51e-3 (DSA, 8 splits) and 7.5e-4 → 1.79e-3 (Qwen3.5, 16 splits); one fp8 [64, 128] tile in 8192 B of LDS read for QK through `.permute((1, 0))` and for P·V: within 1.756e-3 of torch; V in `compute_efficient_padded_shared_layout` at TILE 32 and 64, read with `ds_read_b64_tr_b8`: P·V within an absolute 1.9e-6 of torch
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=6 counterbalanced rounds per arm in one session (3 forward, 3 reverse), the ruler, paired-log 99% t CIs of call time] 27 arms (54 kernels) compiled, every timed arm accurate. DSA 64 heads, modified over unmodified: 96/128/512 producers against 192/256/256 at 12/16/32 tokens 1.414/1.456/1.098; the 64-head over the 32-head tile 1.154 at 12 tokens and 1.222 at 16 (17.58 against 14.39 µs; both arms the 24-token kernel run at 16 tokens with correctly sized packed statistics, 64 heads on 128 producers against 32 on 256, since the 16-token kernel's own 64-head path needs 164,848 B of LDS); 32 over 64 heads 0.9737 at 32 tokens (21.67 against 22.25 µs). Qwen3.5, doubled splits over unmodified 0.7863/0.8142/0.9731/0.7818/0.7052 at 4/8/64/128/256 rows; at 16 rows half 1.441, double 1.033. CIs in [triton/attention](../triton/attention.md) §3, [triton/attention](../triton/attention.md) §4 and [triton/attention](../triton/attention.md) §7
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (DSA) and N=4 (Qwen3.5) runs, the ruler, 4 alternated pairs per arm in each run, one GPU] DSA 64 heads at 24 tokens (192 producers of 8 warps, 163,840 B of LDS each): the 32-head tile at 8 splits (384 programs, 1 tile each, an 8-way merge) 22.07, 22.03 and 21.87 µs against 20.27, 20.30 and 20.31 µs at 4 splits (192 programs, 2 tiles each, a 4-way merge), ratios 1.089, 1.085 and 1.077, the two arms timed in separate A/B sessions whose unmodified arms agree (18.66–18.80 and about 18.69–18.85 µs); no 4-wave producer and no other token count timed. 32 tokens, refill off (`v_mfma` 82 → 82, accuracy passing): 21.68 → 21.83, 21.72 → 21.77 and 21.62 → 21.47 µs (B/A 1.007, 1.002, 0.993), the off arm's rounds 21.71–21.93, 21.69–21.87 and 21.36–21.52 µs against 21.63–21.74, 21.67–21.78 and 21.59–21.70. Qwen3.5 at 4 rows: the MFMA attention 21.77, 21.64, 21.69 and 21.59 µs against 26.88, 26.79, 26.95 and 26.94 µs for the same kernel forced to a program per (row, head, split) (B/A 1.235–1.248, all 16 pairs 1.230–1.255); at 1 row the per-head attention 16.56, 16.42, 16.44 and 16.57 µs against 16.41, 16.22, 16.45 and 16.28 µs with the 4-row kernel's MFMA attention and merge (MFMA/per-head 0.982–1.001, pairs 0.976–1.014, the MFMA faster in 14 of 16), which resolves neither a small advantage nor a crossover; at 256 rows the gate load and sigmoid moved below the merge: finish 6.13, 6.18, 6.20 and 6.11 µs against 6.21, 6.18, 6.26 and 6.25 µs unmodified (0.978–1.001, pairs 0.965–1.024, the moved order faster in 14 of 16), outputs bitwise equal, the gate load still issued before the statistics load (instruction 24 against 26; unmodified 19 and 28)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, 4 alternated runs per arm, one GPU; the split change also compiled without a GPU] DSA 64 heads at 24 tokens, the kernel's only size: `waves_per_eu=2 if rows >= 24 else 0` → `waves_per_eu=0` on the 8-warp producer launch: call 18.89 → 18.71 µs (time ratio 0.9904, speedup B/A 1.008), the unset arm faster in all 4 pairs, the arms' run ranges not recorded; KV splits 8 → 4 with the 64-head tile kept: the producer fails to compile, `CompilationError` from `NameError('cache_shared is not defined')`, without a GPU and in each of the 4 timed runs (4 of 4 in an earlier batch too); the 32-head tile with 4 splits compiles and passes, 2.034 and 2.039
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (Kineto) on one GPU, 4 alternated runs per arm] DS512 (8 tokens), `enable_fp_fusion` False → True on its 3 launches: 1.147 → 1.146 (0.9994), 56.27 → 56.34 µs, accuracy ok in all 8 runs, A's and B's compiled launches (front split, expert, down reduce) identical in instructions and `.amdhsa_` fields (633, 1554 and 440 instructions, compiled offline with PyPI Triton 3.8.0); `allow_flush_denorm` removed from the split and merge launches of the DSA 64-head kernels at 12, 16, 24 and 32 tokens: B/A 1.0007, 1.0033, 0.9986, 0.9970, accuracy ok, the 8 compiled kernels identical with and without it
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (DSA) and N=4 (Qwen3.5) runs, the ruler, 4 alternated pairs per arm in each run, one GPU] DSA 64 heads at 24 tokens: unmodified 18.80, 18.78 and 18.66 µs; the 32-head tile with 4 splits (192 programs either way, 2 tiles each, a 4-way merge) 20.27, 20.30 and 20.31 µs (B/A 1.078, 1.081, 1.088); rows gathered per call 28.31 against 56.62 MB (computed from the host lines); the baseline 25.63–25.73, 27.02–27.11, 41.28–41.47 and 43.67–43.83 µs at 12, 16, 24 and 32 tokens (1.72–1.73 µs per token at 24, 1.37 at 32). 8 heads with the QK MFMAs issued twice (`v_mfma` 68 → 96 and 34 → 48), the whole call timed: 18.37, 18.45 and 18.50 → 18.52, 18.63 and 18.60 µs at 32 tokens (B/A 1.006–1.010), 36.41, 36.55 and 36.41 → 38.10, 38.07 and 38.01 µs at 128 (1.042–1.047); about 30 FLOP per gathered byte, computed; accuracy passed in every arm. Qwen3.5 at 256 rows, K and V (with Q) in one group waited with `wait_group(0)` before QK: 229.3, 230.2, 231.2 and 231.2 µs against 207.7, 207.8, 209.8 and 209.5 µs unmodified (B/A 1.104, 1.108, 1.102, 1.103; all 16 pairs 1.102–1.112); accuracy passed in 3 of the 4 runs, the other run's report also carrying two status flags of the ruler
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, probe kernels reading one 4.5 GiB uint8 tensor with `buffer_load` and with `gl.load`] `buffer_load` returns the data up to byte 2147483645 and 0 from byte 2147483646 = 2^31 − 2 on (also at 2^31, 2^31 + 64, 2^31 + 4096 and 2^32 − 1), where `gl.load` returns the data; the descriptor is built with `num_records` 0x7ffffffe (a probe, not a library kernel)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 repeats, `buffer_load` over a 4.25 GiB uint8 tensor from three scalar bases, `num_records` 0x7ffffffe] `_ubyte` and `_ushort` returned data up to byte 2^31 − 3, `_dword` and `_dwordx4` up to 2^31 − 5 and 0 from 2^31 − 4, a `dwordx4` across that edge its first 12 bytes; no width read data at or above 2^31 − 2; the edge moved with the scalar base (a probe, not a library kernel)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] DSA sparse MLA decode, both in one run: lowest 1.030 at 64 heads, 8 tokens, highest 2.188 at 64 heads, 24 tokens (64 heads, cache rows of 576 values, top-k 2048)
- S1 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:104 `num_records` = INT_MAX − 1; select unless `other` is a zero constant 121-122; DMA via async LDS intrinsic 126-149; 32-bit byte offset, masked lanes INT_MAX + 1 280-283
