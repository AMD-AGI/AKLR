---
title: "Sparse MLA of sglang's DSA backend with one query token per sequence, 8 heads, fp8 q and 576-byte KV rows, top-2048, at 32, 128 and 256 tokens — a wave-private buffer-to-LDS ring feeds fp8 MFMAs; 8, 2 or 1 splits per token, merged inside the launch by the last split to arrive"
kind: case
ops: [mla, attention]
dtypes: [fp8, bf16]
regimes: ["tokens 32", "tokens 128", "tokens 256"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1, AKLR validate measurement profile 19
measured:
  - {metric: "speedup over the contract's baseline (sglang's triton_sparse_mla_decode_splitk), AKLR validate published lower bound per Point, local run, not CI, of this file's predecessor d5943f5f096473af", value: "1.642 / 1.592 / 1.265 at 32 / 128 / 256 tokens", gpu: "MI355X", workload: "dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048, (n_tokens, n_slots) = (32, 65536), (128, 262144), (256, 524288)", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/dsa/tag_hash_fix/dsa_compare3/validation/report.json", date: "2026-11-09"}
  - {metric: "speedup over the older Gluon file main 0e02a81 served at the Point (since 6d7cfec served at 256 tokens only), same run, published lower bound per Point, of this file's predecessor d5943f5f096473af", value: "1.536 / 1.319 / 1.198 at 32 / 128 / 256 tokens", gpu: "MI355X", workload: "the same three Points", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/dsa/tag_hash_fix/dsa_compare3/validation/report.json", date: "2026-11-09"}
sources:
  - aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json
---

# Sparse MLA decode for sglang's DSA backend, 8 heads: one launch, splits merged by the last to arrive

One FlyDSL file, `8373b752b36cff1f`, lists three Points of `dsa_sparse_mla_decode@1.0`, the decode attention of
GLM-5.2 at 8 heads per rank (TP8). Its workloads carry no `speedup` yet, so dispatch serves them only once CI has
written one that qualifies (`aklr/ops/README.md` § Environment variables). It embeds its source as
`kernel.sources[0].content` (`kernel.py`); function names below are in it. The padded DeepSeek-V4 pool of
`flash_mla_with_kvcache@1.0` is a different contract, served by a two-launch kernel for 64 heads:
[mla_sparse_fp8_kv_pool.md](mla_sparse_fp8_kv_pool.md).

## Applies when

- Contract `dsa_sparse_mla_decode@1.0` (op type `dsa_backend`, `aklr/plugins/sglang/layers/dsa_backend/`), layout
  `row_major`: q (tokens, 8, 576) and the pool (n_slots, 576) float8_e4m3fn, unscaled, a row being 512 latent
  values then 64 RoPE values; top-k (tokens, 2048) int32, distinct, −1 for none; `sm_scale` fp32; output (tokens,
  8, 512) bf16, 0 for a token that reads no row. `compare`: `determinism: bitwise` (two runs bit-identical),
  `max_violation_fraction` 0, output SQNR ≥ 20 dB against the reference (S9).
- Points, three pairs and nothing between them: (n_tokens, n_slots) = (32, 65536), (128, 262144) and (256,
  524288), each with `sm_scale` 0.0625. Dispatch matches `n_tokens` exactly. `n_slots` is `any_size`, so once a
  Point is served, its token count is served at every pool size (`aklr/ops/README.md` § Inputs and workloads);
  bench times only pools of tokens × 2048 rows (36, 144 and 288 MiB).
- Not covered: the other 33 decode sizes sglang pads TP8 batches to and the 16-head schema name (other files,
  #66); another top-k or head count; draft steps of speculative decoding and a q that is not one contiguous
  (tokens, heads, 576) view, which the api sends back to sglang (S11).
- Yardstick at each Point: the faster of the baseline, `dsa_sparse_mla_decode_baseline` (sglang 0.5.22's
  `triton_sparse_mla_decode_splitk`, S10), and the file main serves there (S12). All are Gluon files (schema
  language `triton`). Since 6d7cfec (#102, CI run 42549655045, measurement profile 19) main serves PR #66's
  `a9d172eb1a6783bf` at 32 tokens and `04c92e850fb6e62f` at 128, with CI speedups over the baseline of 1.226 and
  1.343; at 256 tokens CI measured `04c92e850fb6e62f` at 0.99, so the older `fbe6ddab5dac5958` stays served there
  *(field, E5)*. Before it, at 0e02a81, the older files `43ffd5e3e5b67298`, `ae32cc132679692c` and
  `fbe6ddab5dac5958` were served at the three Points *(field, E6)*. Every one of them is faster than the baseline
  at its Point, so it is the yardstick (§ Result).

## Counter-signals

- **The pool sizes bench times are tied to the token count.** `_build` is a `functools.cache` keyed by
  `n_tokens` alone (S2): `sm_scale · log2 e`, the pool's size in bytes and the key sort's bucket shift are launch
  arguments, so a call at a pool size or scale not seen before runs the kernel already compiled for its token
  count. A bucket of the key sort is the smallest power of two of slots that covers the pool in 16 or 64 buckets
  (S5), so at a pool that is not 16 or 64 times a power of two the last buckets hold fewer keys or none. The
  speed at serving pool sizes is not measured.
- **Uniform indices.** Bench draws each token's 2048 slots uniformly over the pool. Serving top-k sets overlap
  between adjacent tokens, which moved gather cost on another sparse decode (`../attention/split_kv.md` §7);
  the key sort at 128 and 256 tokens was not measured on such indices.
- **At 256 tokens the margin owes nothing to the merge.** There this kernel and the file main serves,
  `fbe6ddab5dac5958`, both run one workgroup per token with no split (S13); the 1.198 there comes from the producer
  and from the bf16 copy of q that the served file launches first and this kernel does not need.
- **Half of every MFMA is padding.** 8 heads fill 8 of the 16 MFMA columns of both GEMMs; the other 8 carry zero q.
- **P·V is on the legacy fp8 MFMA**, which takes the bf16 cycle count for the same K (`../attention/mfma_layout.md`
  §8); only the latent part of QK uses the wide-K form. P is rounded to fp8 for P·V while its row sum stays fp32. It
  is scaled by 2^8 first (S4), so a weight keeps a normal e4m3 value down to 2^−14 of the running max and rounds to
  zero only under about 2^−18; unscaled the limits are 2^−6 and 2^−10, and a key e^−8 (about 2^−11.5) below the max
  would add nothing. Smaller weights still round to zero, so an input where each wave's other keys score about 12.5
  or more below one key whose latent values are zero misses the 20 dB bound: about 12 dB at a gap of 12 and 0 dB
  from 13 (estimated, a CPU model of the arithmetic). sglang's kernel, the contract's baseline, multiplies P by the
  fp8 maximum, 448, before its conversion, and meets the same limit within a factor of 1.75.
- **Measured as its predecessor.** Every time, bound and stress result on this page is of `d5943f5f096473af`. This
  file differs from it in two ways: P is scaled by 2^8 before its fp8 conversion (S4), and `sm_scale · log2 e`, the
  pool's size and the key sort's bucket shift are launch arguments rather than compiled in (S2). At the three
  benchmarked pools the bucket width is the same power of two as before (16384 slots at 128 tokens, 8192 at 256),
  so the key order is unchanged. This file has not been run on a GPU here; its speed and its output are CI's to
  measure.
- **Not CI, and one FlyDSL only.** The predecessor was validated on FlyDSL 0.3.2 alone, pinned by hand over the
  image's 0.3.4.1, with the baseline aiter installed without its dependencies. That is not what CI runs: CI
  installs the baseline aiter over its image and runs the FlyDSL that aiter pins
  (`../pitfalls/measured_a_different_build.md`).
- **What the cross-workgroup merge was run under.** Besides the bench's correctness steps, each try below was
  compared byte for byte with the same call alone, at all three Points: on a stream limited to 16 CUs and on one
  limited to 64, beside a second call on the same inputs, and in a replayed graph, 20 tries each, none differing
  and none hung *(field, E2)*; two calls at once on two streams with different inputs, unrestricted and with both
  streams on the same 64 CUs, 20 tries each, none differing *(field, E4)*. Nothing beyond two concurrent calls of
  this op was run, and no other kernel was resident beside it. No workgroup waits for another
  (`../attention/split_kv.md` §2), and each call gets its own `torch.empty` partials and arrival words (S8).
  The merge crosses XCDs only at 32 tokens, where the 8 splits of a token are consecutive workgroups on 8 XCDs
  (`../attention/grid_dispatch.md` §2); at 128 tokens the two splits of token t are workgroups t and t + 128, on
  one XCD.
- **Workspace reuse is tested for the op alone, not in a model's graph.** Arrival words are not cleared between
  calls, so whether a word belongs to this launch rests on its tag alone (§ Mechanism). One captured graph per
  Point, replayed 24 times with three different input sets copied into its buffers in turn, so every replay meets
  the words its predecessor left, matched an eager call on the same inputs at every replay *(field, E3)*. Inside
  a model's decode graph the call's workspace comes from the graph's memory pool, which other kernels may write
  between replays; that has not been run.
- **Two queues can share a tag.** A launch on a queue whose object sits at address a and one on a queue at b get
  equal tags exactly when the first's dispatch id minus the second's is δ = (b − a) · H2 · H1⁻¹ mod 2^56, one
  value fixed by the two addresses (S7). A flag buffer last written on one queue and read by a launch on the other
  at that distance would be misread: a split could merge before this launch's partials have landed, with no error.
  Not checked: where δ falls for the queues ROCm creates; whether one flag buffer moves between queues; δ = 0
  cases — a queue destroyed and recreated at the same address restarts its dispatch ids at 0, and a graph captured
  on one stream and replayed on another may carry the capture-time queue address; and whether the address
  `llvm.amdgcn.queue.ptr` returns on this build is non-zero and distinct per queue.
- **Cross-XCD visibility rests on scope bits, not fences.** Partials, maxima, sums and the merge's loads use
  `SC1` (aux 16) and the kernel has no fence. `SC1` stores were bitwise clean across XCDs in another decode
  kernel (`../attention/split_kv.md` §2, its E5); the ISA does not say which L2 case gfx950 is
  (`../hardware/gfx950/memory_and_sync.md` §2). The tests here do not add to that: the bench flushes 1 GiB before
  each call and the graph test copies the 36 MiB pool or more between replays, so none is known to have left a
  stale partial line in an XCD's L2.

## Mechanism

**Grid** (`_config`, S1). 4 waves (256 threads) per workgroup; the number of splits per token keeps the grid at
256 workgroups, one per CU of an MI355X:

| Point | splits per token | workgroups | keys per wave | 32-key chunks per wave | split merge |
|---|---|---|---|---|---|
| 32 tokens | 8 | 256 | 64 | 2 | in the launch, across XCDs |
| 128 tokens | 2 | 256 | 256 | 8 | in the launch |
| 256 tokens | 1 | 256 | 512 | 16 | none: the output is stored directly |

A workgroup takes 148–156 KiB of LDS (from the source's shapes: two ring slots of 4 waves × 32 rows × 592 B, plus
the sort's buffer at 128 and 256 tokens), so one fits per CU and the grid runs as one round. `_build` switches
features by split count (S2): split-fastest workgroup order and a last chunk scored in two halves at 8 splits; the
key sort, key validity from a ballot and `SC1` gathers at 2 and 1; `v_readlane` addressing, a peeled first chunk
and each (token, head)'s two maxima and sums side by side at 2; all of a chunk's `ds_bpermute`s in flight before
the first is used at 1.

**DMA ring** (S3). Two ring slots; each wave owns a 32-row region of each, rows pitched 592 B (576 + 16, so the
rows one K or Vᵀ read touches start in distinct bank groups). A chunk is 18 `buffer_load … lds` of 16 B per lane,
1 KiB each, and a 19th on 32 lanes (512 B), filling the region in byte order: consecutive lanes read consecutive
16 B of one row (`../attention/split_kv.md` §7), and a lane that falls on a row's 16-byte pad gets an
offset past the pool's size, so the buffer returns zeros and fetches nothing
(`../hardware/gfx950/memory_and_sync.md` §2–3). A
lane's row comes from the wave's index registers by `ds_bpermute`, or, from the third chunk on at 2 splits, by
`v_readlane` of the chunk's 32 keys; an index of −1 reads row 0 and its score is set to −inf. Chunks 0 and 1 are
issued first; chunk c is computed while c + 1 lands, and once every LDS read of c's slot has returned
(`s_waitcnt lgkmcnt(0)`), the DMA of c + 2 into that slot goes out before the last 16 P·V MFMAs. The loop has no
workgroup barrier: each wave waits on its own `vmcnt`. A 576-byte row starts at phase 0 or 64 and touches 5 lines.

**Key sort** (`sort_keys`, S5; 128 and 256 tokens). Each wave counting-sorts its keys by pool bucket (16 buckets
at 2 splits, 64 at 1; a bucket is a power of two of slots wide, so it is the slot shifted right by a launch
argument), chunk 0's keys, already in flight, kept first; the source's aim is that every CU sweeps the
pool in the same order. The order is fixed by the indices, so the result stays bitwise repeatable. Its effect was
not measured apart.

**fp8 MFMAs and the softmax** (S4). The token's q stays in registers as fp8 for the whole workgroup, the B operand,
heads 0–7 on lanes 0–7 of each 16-lane group. Per 16-key tile: 4 `mfma_scale_f32_16x16x128_f8f6f4` with unit E8M0
scales over the 512 latent values and 2 `mfma_f32_16x16x32_fp8_fp8` over the 64 RoPE values, A = key rows from
LDS. A lane holds one head with its keys on the element index, so the row max is an in-lane fold plus two
`shuffle_xor` (`../attention/mfma_layout.md` §4). `sm_scale · log2 e` is one multiply before `exp2`. The running
max and the accumulators are rescaled only when a ballot shows some lane's chunk max above the running max (a
wave-uniform branch, threshold 0, so a skipped rescale is a multiplication by exactly 1;
`../attention/softmax.md` §6.3). P is exp2(s − (m − 8)), the weight scaled by 2^8 (`P_LOG2_SCALE`; at most 256,
under e4m3's 448), summed in fp32 and packed to fp8 (`cvt_pk_fp8_f32`) as the B operand of P·V; the sum carries
the same factor and every output is divided by a sum, so it cancels. P·V is 32 `mfma_f32_16x16x32_fp8_fp8` per
chunk with A = Vᵀ read from the same ring rows by `ds_read_tr8_b64`, into 32 f32x4 accumulators (512 values × 16
head columns). A chunk issues 16 of those reads (28 at 2 splits) before the softmax, all 32 when its slot is
reused next, so they land under it. Nothing is converted to bf16; the older Gluon files copy q to bf16 in a
launch of their own first, and PR #66's read the fp8 q directly (S13).

**Workgroup merge** (S6). After its last chunk each wave writes its fp32 Oᵀ to its own ring region and its max and
sum to LDS; one barrier; 32 threads per head then combine the 4 waves in wave order, 16 output columns per
thread. At 256 tokens this is the output, stored as bf16 with `SC1` write-through stores.

**Split merge inside the launch** (S7; 32 and 128 tokens). Thread h·32 … h·32 + 31 of a workgroup hold head h, so
wave w holds heads 2w and 2w + 1, and each (token, wave) has its own 64-bit arrival word on its own 128-byte line.
Its low byte is the mask of splits that published; the other 56 bits hold the launch's tag, the low 56 bits of
`dispatch.id` (which counts per HSA queue) · H1 + the address of the launch's queue object
(`llvm.amdgcn.queue.ptr`) · H2, with two odd 64-bit constants `TAG_H1` and `TAG_H2`: tag bits 0–23 above the mask,
bits 24–55 in the high half. The low bits of a product depend only on the low bits of its factors, and an odd
multiplier permutes the residues modulo 2^56, so the tag changes with any change in the low 56 bits of either
input alone, and on one queue two dispatch ids give the same tag only 2^56 dispatches apart.

1. Each wave issues one `SC1` load of its arrival word, then stores its heads' normalized bf16 partials and one
   (max, sum) per head with `SC1` write-through stores.
2. If the word carries this launch's tag and every other split's bit, the wave merges at once and does not publish,
   so no other wave can complete the mask.
3. Otherwise it waits `vmcnt(0)`: its stores are acknowledged, and they are all the word vouches for, since the
   word covers only this wave's heads. That wait is the release; there is no fence and no workgroup barrier.
   Lane 0 then ORs (tag | its bit) into the word with one 64-bit atomic that returns the old word; the wave whose
   OR completes the mask merges.
4. A word that carries another launch's tag is claimed with one 64-bit compare-swap from the value read in step 1;
   a failed swap means a partner claimed it for this launch, and the wave ORs its bit as in step 3 (at 2 splits,
   a returned word already holding the partner's bit lets it merge without publishing).
5. The merge issues all its loads at once, with `SC1`: the splits' maxima and sums and 2 × 16 B of partials per
   split and thread, before its first wait. The merging wave's own max and sum come from registers. It weighs the
   splits in index order, so the output bits do not depend on which split merges, and stores bf16 with `SC1`.

No loop polls, nothing sleeps, and a merge is one round of loads by one wave; the 4 waves of a token merge
independently and may sit in different workgroups. `../attention/split_kv.md` §3 measured a "last workgroup
reduces" fold losing where a row's reduce ran as a loop of dependent memory round trips in one workgroup at a few
rows, or where the reduce's registers sat on a 16-head producer. Here the merge was not measured apart from the
rest of the kernel (§ Result).

## Result

AKLR validate, measurement profile 19, on an MI355X with FlyDSL 0.3.2 (local, not CI): graph replays with a 1 GiB
read flush before each call, every participant timed once per round in a seeded order, 16 rounds at each Point.
A speedup is the published one-sided Student-t lower bound, α 0.025 per look, of the geometric mean of the
per-round time ratios (the bound is on the mean of their logarithms; `TimingRecord.speedup_bounds` in
`aklr/core/validation/stages/_measure.py`); times are medians of the 16 rounds. The "baseline" participant is the
contract's baseline function; the `baseline` path the report prints is the served file it exports (S12). One
session on one GPU timed this file's predecessor, `d5943f5f096473af` (§ Counter-signals), and PR #66's two files,
with a base that lacks those two, so the file paired with it as served was the older Gluon file at each Point, the
one main served at 0e02a81. "This kernel" in the two tables below is that predecessor.

| Point | this kernel | baseline | older Gluon file | bound over baseline | bound over older file |
|---|---|---|---|---|---|
| 32 tokens *(field, E1)* | 12.04 µs | 19.79 µs | 18.50 µs | 1.642 | 1.536 |
| 128 tokens *(field, E1)* | 27.63 µs | 44.02 µs | 36.46 µs | 1.592 | 1.319 |
| 256 tokens *(field, E1)* | 47.36 µs | 59.98 µs | 56.80 µs | 1.265 | 1.198 |

The older files beat the baseline at every Point, so against them the per-Point speedup is 1.536, 1.319 and 1.198.
At 256 tokens that is still the yardstick on main since 6d7cfec; at 32 and 128 tokens the yardstick is now PR
#66's file, which this session timed but not paired with this kernel (below). All three Points passed accuracy
with a violation fraction of 0.

PR #66 added a newer revision of the older Gluon kernels, `a9d172eb1a6783bf` (32 tokens) and `04c92e850fb6e62f`
(128 and 256), which read the fp8 q directly; since 6d7cfec main serves them at 32 and 128 tokens (§ Applies
when). In the same session *(field, E1)*:

| Point | this kernel | PR #66's file | bounds of PR #66's file, over baseline / older file |
|---|---|---|---|
| 32 tokens | 12.04 µs | 16.15 µs | 1.224 / 1.143 |
| 128 tokens | 27.63 µs | 32.89 µs | 1.336 / 1.113 |
| 256 tokens | 47.36 µs | 61.59 µs | 0.979 / 0.925, reported `regressed` (demoted) |

The two were not timed against each other: the ratio of their medians, 1.34, 1.19 and 1.30, is estimated, and at
32 and 128 tokens it is the only figure against the file main now serves.

How the kernels split the work at each Point, from their sources (S13):

| Point | older Gluon file | PR #66's file | this kernel |
|---|---|---|---|
| 32 tokens | q copied to bf16; 16 splits per token; merge launch | 8 splits; merge launch | 8 splits; merged in the launch |
| 128 tokens | q copied to bf16; 2 splits; merge launch | 4 splits; merge launch | 2 splits; merged in the launch |
| 256 tokens | q copied to bf16; one workgroup per token | 2 splits; merge launch | one workgroup per token |

At 32 tokens PR #66's file splits a token 8 ways as this kernel does but merges in a second launch; the producers
differ as well, so the 1.34 is not the merge's alone. No ablation separates the merge, the ring, the fp8 q or the
key sort. Rows × 576 B over the median time is 3.1, 5.5 and 6.4 TB/s at 32, 128 and 256 tokens (estimated; rows
shared between tokens may hit in cache): at 2 chunks per wave the 32-token Point is far from the bandwidth the
256-token Point reaches (`../attention/split_kv.md` §6).

## Reproduce

On an MI355X, in a checkout where this file is not on the base branch, with FlyDSL 0.3.2:

```
python -m aklr validate aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json --base origin/main
```

It runs validate's two stages, times the kernel against the baseline and against the file main serves at each
Point, and writes `aklr_validation/report.json` and `aklr_validation/report.md` (`aklr/ops/README.md` §
Commands). Once the file is on the base, bench skips it with `--base` set, and without `--base` it is timed
against the baseline alone (`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). On main from
6d7cfec on, the files it is paired with are PR #66's at 32 and 128 tokens and `fbe6ddab5dac5958` at 256. E1's
setup pairs it with the older files instead: a local base commit without `a9d172eb1a6783bf.json` and
`04c92e850fb6e62f.json`, those two restored on top, and all three paths passed with `--base` at that commit. The
local result does not count; CI's measurement is the one dispatch reads. To read the kernel:
`python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["kernel"]["sources"][0]["content"])' <schema>`.

## Rollback

Delete the file in a PR: once served, its three Points fall back to the files main serves there (since 6d7cfec,
PR #66's at 32 and 128 tokens and `fbe6ddab5dac5958` at 256). Dispatch
stops serving a Point whose measured speedup falls under `AKLR_MIN_SPEEDUP` without a rollback
(`aklr/ops/README.md` § Environment variables). `AKLR_DISABLE=dsa_sparse_mla_decode@1.0` sends every head count
of the contract back to sglang's own kernel. Roll back for a wrong output CI does not catch: the workspace written
by other kernels inside a model's graph, a flag buffer shared between queues, more concurrent calls than the two
tested or other kernels resident beside it (§ Counter-signals), or a FlyDSL or LLVM upgrade.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` the schema embeds.

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR validate, measurement profile 19, 16 rounds per Point] one session on one GPU, base = main 0e02a81 without a9d172eb1a6783bf and 04c92e850fb6e62f, head = those two plus d5943f5f096473af, the predecessor of 8373b752b36cff1f; medians of 16 rounds and published lower bounds, this kernel / baseline / older file: 32 tokens 12.037 / 19.786 / 18.500 µs, bounds 1.642 and 1.536; 128 tokens 27.625 / 44.020 / 36.465 µs, 1.592 and 1.319; 256 tokens 47.363 / 59.979 / 56.798 µs, 1.265 and 1.198; older files, served at 0e02a81: 43ffd5e3e5b67298, ae32cc132679692c, fbe6ddab5dac5958; a9d172eb1a6783bf at 32 tokens 16.151 µs, 1.224 / 1.143; 04c92e850fb6e62f at 128 tokens 32.895 µs, 1.336 / 1.113, at 256 tokens 61.587 µs, 0.979 / 0.925, finding `regressed`; all six accuracy passed, worst violation fraction 0.0 (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/dsa/tag_hash_fix/dsa_compare3/validation/report.json; setup in local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/dsa/tag_hash_fix/README.txt)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=20, stress driver on main 0e02a81 plus d5943f5f096473af, the predecessor of 8373b752b36cff1f, each try byte-compared with the call alone, AKLR timing inputs] at each of the three Points: on a 16-CU stream, on a 64-CU stream, beside a second call on the same inputs, and in a graph replayed with the same inputs, 20 tries each, 0 differ, no hang (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/results/g4/dsa_d5943f5f096473af/stress.log; conditions in local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/README.txt)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=24, one captured graph per Point, three AKLR input sets with differing outputs copied into its buffers in turn, each replay byte-compared with an eager call on the same inputs] d5943f5f096473af, the predecessor of 8373b752b36cff1f: 24 replays at each of the three Points, 0 differ (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/graph/results/g2/dsa_d5943f5f096473af/graph.log)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=20, two calls at once on two streams, each on its own AKLR input set and outputs, each byte-compared with an eager call alone] d5943f5f096473af, the predecessor of 8373b752b36cff1f, at each of the three Points, with the two streams unrestricted and with both on the same first 64 CUs, 20 tries each, 0 differ (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/concurrent/results/g4/dsa_d5943f5f096473af/concurrent.log)
- E5 [2026-11, gfx950, Gluon, N=1, AKLR CI, measurement profile 19] published speedups over the baseline written by #102: a9d172eb1a6783bf at (32, 65536) 1.226; 04c92e850fb6e62f at (128, 262144) 1.343 and at (256, 524288) 0.99; accuracy passed (6d7cfec, CI run 42549655045)
- E6 [2026-11, gfx950, Gluon, N=1, AKLR CI] published speedups over the baseline of the older files, unchanged by #102: 43ffd5e3e5b67298 at (32, 65536) 1.062, ae32cc132679692c at (128, 262144) 1.196, fbe6ddab5dac5958 at (256, 524288) 1.05; accuracy passed (a29581b, CI run 41197527705)
- S1 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json#kernel.py:41 `_config`: (8, 4) up to 32 tokens, (2, 4) up to 128, (1, 4) above; grid `n_tokens * S` at :781
- S2 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json#kernel.py:95 `_build(n_tokens)` under `functools.cache` (:94); switches by split count at :106–117; `Shared` (ring, maxima and sums, sort buffer) at :120; split-fastest order at :133; `scale2`, `pool_bytes` and `bshift` kernel arguments at :128 and :775–777, passed by `run` at :812–814, the shift computed at :792
- S3 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json#kernel.py:212 pad lanes sent past `pool_bytes`; row pitch `RP` at :21; `raw_ptr_buffer_load_lds` at :244; ring schedule at :495–531
- S4 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json#kernel.py:312 scaled fp8 QK MFMA; RoPE MFMAs at :320; −inf for −1 at :346; `ds_read_tr8_b64` at :354; rescale ballot at :366; P scaled by 2^8 at :392 (`P_LOG2_SCALE` at :29); P to fp8 at :394; slot release after `lgkmcnt(0)` at :410; P·V MFMA at :414
- S5 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json#kernel.py:417 `sort_keys`; bucket count `NB` at :107; bucket by shift at :435
- S6 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json#kernel.py:556 the one workgroup barrier; direct bf16 store at one split at :599
- S7 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json#kernel.py:619 the tag's multiply-add; odd `TAG_H1`, `TAG_H2` at :34–35; `dispatch.id` at :616 and `queue.ptr` at :617; `tag_lo` and `tag_hi` at :622–623; arrival word load at :635; `merge` at :657, its loads in one batch at :695; `arrive_or` at :723; fast path at :736; `vmcnt(0)` at :740; compare-swap at :747
- S8 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/8373b752b36cff1f.json#kernel.py:794 partials, maxima and sums, and arrival words (:801) from `torch.empty` per call
- S9 aklr/plugins/sglang/layers/dsa_backend/contract.py:60 `n_slots` `any_size`; distinct indices from −1 at :91; `SqnrBound(min_sqnr_db=20.0)` at :110
- S10 aklr/plugins/sglang/layers/dsa_backend/baseline.py:14 `dsa_sparse_mla_decode_baseline` calls `triton_sparse_mla_decode_splitk` (:23)
- S11 aklr/plugins/sglang/layers/dsa_backend/api.py:35 speculative decoding falls back; `_whole_query` at :89
- S12 aklr/core/validation/stages/_bench/_stage.py:215 `_incumbents`: the qualifying base file with the highest published speedup; the record's `baseline` path is the served file (:200); aklr/core/validation/stages/_bench/_evaluation/_candidate.py:418 the baseline participant is `contract.baseline()`
- S13 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/43ffd5e3e5b67298.json#kernel.py:376 16 splits at 32 tokens, merge launch at :406, q copied to bf16 at :416; ae32cc132679692c.json#kernel.py:385 2 splits, merge at :401, q copied to bf16 at :409; fbe6ddab5dac5958.json#kernel.py:709 one workgroup per token at 256 tokens, q copied to bf16 at :762; a9d172eb1a6783bf.json#kernel.py:216 8 splits, merge launch at :227; 04c92e850fb6e62f.json#kernel.py:22 512 and 1024 keys per split at 128 and 256 tokens, `_finish` launch at :114; PR #66's a9d172eb1a6783bf.json#kernel.py:233 and 04c92e850fb6e62f.json#kernel.py:122 pass the fp8 `q` to `sparse_paged_mla` with `quantize_query=True`, which keeps its fp8 values (a9d172eb1a6783bf.json#kernel.py:231-232)
