---
title: "fp8 paged MQA logits (DeepSeek-V4 indexer), 64 heads, page 64 — 16 fp8 16x16x128 MFMAs per page with the head sum kept in registers, a merge-free split over pages, and a band-sorted grid from batch 64"
kind: case
ops: [mla, indexer_logits]
dtypes: [fp8]
regimes: ["one query token per sequence", "paged KV", "batch 1-4", "batch 8-32", "batch 64-256"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter (deepgemm_fp8_paged_mqa_logits through aklr/ops/mla/baseline.py), CI, measurement profile 19, per Point", value: "1.046–1.837; 61 of 61 Points ≥ 1.03", gpu: MI355X, workload: "fp8_paged_mqa_logits_page_size64_n_heads64 file 2c17b79278febb6a, batch 1–256 × 4–9 pool sizes, max_blocks 4097", run_ref: "227aa66 (#115, CI run 42599763780)", date: "2026-11-09"}
sources:
  - aklr/ops/mla/impls/flydsl/fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a.json
---

# fp8 paged MQA logits: whole pages per wave, no merge

One FlyDSL kernel for `fp8_paged_mqa_logits@1.0`, the DeepSeek-V4 indexer's logits
`logits[b, t] = Σ_h weights[b, h] · relu(q[b, h] · k_t) · scale_t`, added for DeepSeek-V4-Flash with one query token
per request, as in decode. The
schema embeds it as `kernel.sources[0].content` (`kernel.py`); the function names below are in that file. "batch"
is the Point's number of requests.

## Applies when

- Contract `fp8_paged_mqa_logits@1.0`, layout variant `row_major`, `page_size` 64, `preshuffled` true, 64 heads: q
  fp8 e4m3 (batch, 1, 64, 128); the key pool uint8 (n_pages, 64, 1, 132), a page being 64 tokens × 128 fp8 values in
  aiter's 16 × 16 preshuffled tiles followed by 64 fp32 scales; fp32 weights (batch, 64); int32 `context_lens` and
  `block_tables` (batch, max_blocks); fp32 logits (batch, max_seq_len) with max_seq_len = 64 · max_blocks.
- Points: batch 1, 2, 4, …, 256, each with 4 to 8 of eight recorded pool sizes (105428–123163 pages), and batch
  1, 16 and 64 with a 262,144-page pool (2.2 GB, one Point per body), every one with 4097-page tables (262,208
  logits per request): 61 Points. `n_pages`, `max_blocks` and `max_seq_len` are `any_size`: a batch's Points,
  if each of them qualifies, serve every pool size and table width within the contract's bounds.
- Compare: `determinism: bounded`, `max_violation_fraction` 0.0, logits within an SQNR of 30 dB of the reference
  (`SqnrBound`, `aklr/ops/mla/contract.py`).
- Yardstick: `fp8_paged_mqa_logits_baseline` (`aklr/ops/mla/baseline.py`), aiter's
  `deepgemm_fp8_paged_mqa_logits` (`aiter/ops/triton/attention/pa_mqa_logits.py`) with `Preshuffle`, `KVBlockSize`
  64 and the logits' width as `max_model_len`; `ChunkK`, `WavePerEU` and the scheduling arguments stay at aiter's
  defaults, because the baseline does not pass them (S2). On gfx950 that function takes its Gluon kernel when
  Triton is 3.5 or later (read from the source; the CI image's Triton was not checked).
- The schema name has one kernel file and none in another language. `9d22a1efb2186f81.json` in the same directory is
  a DeepSeek-V4-Pro definition (empty `sources`, never served or benched) whose 41 Points (pools of 6621–52289
  pages) are not this file's.

## Counter-signals

- **Batch 1 sits just above the threshold**: 1.046–1.110 over its nine pool sizes in one CI run of this file
  *(field, E5)*. In a CI run of the file this one replaced, the Point at 111670 pages read 1.024 and was not
  served *(field, E2)*. Which batch-1 Points dispatch serves depends on the run that wrote the file.
- **The bench's workload may not be serving's.** Every Point draws full contexts (`context_lens` = `max_seq_len`, the
  contract's stated shortcut) and block tables uniform over the whole pool, so requests share pages: at batch 256
  the tables name 8.5–9.4 entries per pool page (estimated: 256 · 4097 over 123,163 to 111,670 pages). The band
  path exists to catch those cross-request re-reads in cache (its comment in `_build` says so). Whether serving's
  block tables look like the bench's is not established. With each request owning its pages (contexts of
  8,192–49,152 tokens, batch 64–256), the band path ran at 0.88–0.94 of the slice path's speed with the pages
  scattered over the pool, and at 0.19–0.51 (0.27–0.47 of aiter's) with each request's pages one contiguous run
  *(field, E7)*: such a run lies in one or two of the 128 bands, so one or two workgroups walk the whole request. The
  gain of the per-batch slice counts there is not measured.
- **Logits past a request's last page are not written.** Pages at or past ⌈context / 64⌉ are not visited; −inf is
  written only to the tokens past the context inside the last page. The reference fills −inf up to `max_seq_len`;
  the bench's full contexts do not reach the difference, and the contract's comment says aiter leaves those logits
  unspecified as well.
- **The scale is applied after relu and the head sum.** That equals the reference, which scales the keys before
  relu, only for non-negative scales; the contract's distribution keeps every byte of the pool below 0x40, so its
  scales are non-negative.
- **Scratch outside the README's form.** `_build` allocates the band path's `ORD` / `OFF` and a warm-up buffer with
  `device="cuda"`, the current device, while its cache key holds no device (the README asks for an argument's
  device, `aklr/ops/README.md` § The kernel section and ABI). `sort_kernel` rewrites every entry `band_kernel` reads
  before it reads it, but every call of a shape shares them, so two streams calling one Point at once would race.
- **Widths no Point has.** Every Point is a 4097-page table, but dispatch serves any width, and `_build` picks the
  split and the body from it: under 1024 pages (`_BAND_MIN_BLOCKS`) batch 64–256 take `logits_kernel` instead of the
  band path, and batches outside `split_table` take the untuned split (`_FILL`). Those were measured outside CI
  only.
- **A shape compiles on its first call.** `_build` is cached by the exact (batch, max_blocks, max_seq_len,
  n_pages), while dispatch counts a batch warm after any of its shapes ran: a pool size or table width first seen
  inside a CUDA graph capture compiles there. sglang runs each captured shape eagerly first.
- **Code no Point reaches**: the `small` branch of `_slice_plan` (`logits_kernel` is launched only at batch ≥ 8, the
  batch ≤ 4 body is `small_kernel`), and the `split_table` entries for batch 64, 128 and 256 (the band path sets
  its own split from 1024 pages).
- **Tuning numbers in the comments** (2–4%, 12%, 3–12% and others) were measured outside CI and are not reproduced
  here; the module docstring itself describes only the kernel.
- **Pool size.** Byte offsets are 32-bit (`page id · 8448` in `Int32`) inside one buffer descriptor per pool up to
  2 GiB (254,200 pages); a larger pool gives each page its own descriptor (`_key_pool`, `_issue_page`), which the
  comment there puts at about 4 % at batch 1 (measured outside CI). The 262,144-page Points take that path.
- **A number from another ruler.** The commit that added the batch ≤ 4 body quotes batch 1, 2 and 4 as
  1.7×, 2.1× and 2.2× faster than aiter on AKLR's own clock *(field, E3)*; CI's lower bounds for that commit's file,
  `a989fd4b…`, at those Points were 1.041–1.101, 1.211–1.284 and 1.272–1.372 *(field, E1)*. What makes the gap is
  not established (`../pitfalls/speedup_from_another_ruler.md`).

## Mechanism

**One page, one wave.** A page holds token t, dim d at byte (t // 16) · 2048 + (d // 16) · 256 + (t % 16) · 16 +
d % 16, so two 16-byte loads give a lane the 32 contiguous dims of one token that a 16x16x128 fp8 operand takes
(the preshuffled operand of `../gemm/datapath.md` §10). Per page a wave issues one 4-byte scale load per
lane and eight 16-byte key loads per lane (`_issue_page`), then 16 `mfma_scale_f32_16x16x128_f8f6f4`: four 16-token
tiles × four 16-head groups, A = q, B = keys, unit E8M0 scales (127) (`../attention/mfma_layout.md` §8,
`../gemm/mainloop.md` §5; FlyDSL spelling: `../flydsl/attention.md` § mfma_layout §8). C
holds one token per lane and four heads per lane group. The request's
query (64 heads × 128 fp8) and weights stay in registers for the workgroup's life.

**Head sum in registers, no LDS** (`_page_compute`). relu, × weight, accumulated over the four head groups with
fma (the first group peeled, so no `+ 0.0`), summed over the lane's four heads, then over the four lane groups with
`permlane16_swap` and `permlane32_swap` (`_lane_group_sum`): a fixed tree, no atomics. A `sched_barrier(0)` between
token tiles keeps one tile's results live at a time. Lane group g keeps tile g, so lane l holds token l of the page;
it multiplies by that token's scale, writes −inf past the context, and the wave stores the page's 64 logits as one
256-byte non-temporal store (`../gemm/epilogue.md` §5).

**A split over pages with no merge.** A logit depends on one page, so workgroups split each request's table into
disjoint page ranges and each wave takes every fourth page of its range; nothing is combined
(`../attention/split_kv.md` §5: an output that is not a reduction splits with zero merge). The
host picks one of three bodies from the batch (`_build`):

| Batch | Body | Grid | How a wave gets its next page |
|---|---|---|---|
| 1, 2, 4 | `small_kernel` | 512 workgroups: 512, 256 or 128 slices × batch | Slice bounds from the compile-time table width, clamped to the request's page count, so the table read does not wait for `context_lens`; one dword load per lane holds the slice's 9, 17 or 33 page ids in one VGPR, read per page with `v_readlane_b32`; the next page's nine loads are issued before the current page's MFMAs (two pages in flight), the look-ahead past the slice aimed past the descriptor's `num_records`, where it is dropped; keys and scales loaded `nt` |
| 8, 16, 32 | `logits_kernel` | 256, 256 or 128 slices × batch: 2048, 4096, 4096 workgroups (batch 8's from `_FILL`) | Slice bounds from the runtime page count; the next page id is loaded at the top of each iteration, pinned there with `sched_barrier(0)` and carried across the loop |
| 64, 128, 256 | `sort_kernel`, then `band_kernel` (tables of 1024 pages or more) | batch prepass workgroups, then 128 × batch | The prepass buckets the request's table by band = page id · 128 // n_pages (LDS histogram with LDS atomics, Hillis–Steele scan, scatter of packed (page id, position) words into `ORD`, band bounds into `OFF`); workgroup (request, band) walks its bucket, request index fastest, and its block index is relabelled so XCD c takes the bands ≡ c mod 8 |

The batch ≤ 4 body keeps two pages in flight through its whole loop at 2 workgroups per CU (512 on 256 CUs);
`../attention/schedule_sync.md` §10 found a prologue depth of 2 paying on some grids and losing on others, and a
steady-state depth ≥ 2 slower, on split-KV producers with few query rows per head, so the depth is a
per-kernel result. It drops out-of-range
loads through the descriptor bound of `../gemm/datapath.md` §8. The band path puts co-resident
workgroups on a few bands of the pool and keeps a band on one XCD (`../attention/grid_dispatch.md` §2,
`../gemm/schedule_and_locality.md` §2). The prepass's LDS atomics fix only the order of entries inside a
bucket, which changes which wave computes a page, not any value.

**Compile and launch.** `_build` is a module-level `functools.cache` on (batch, max_blocks, max_seq_len, n_pages,
page size, preshuffle); it allocates `ORD` / `OFF` once per shape and calls `flyc.compile`, which runs the launch
once, here with zeroed context lengths so no page is walked (`../flydsl/attention.md` § measurement §6). `run`
calls the compiled object with raw addresses, the form that section recommends for hot calls.

## Result

The file is the only kernel of its schema name, so the yardstick at every Point is aiter's kernel as
`baseline.py` calls it. `speedup` is a one-sided lower bound of aiter / kernel time over CUDA-graph replays with a
1 GiB cache flush before each call, read at 16 rounds and, where a threshold is still undecided, again at 32, each
look at α = 0.025 (`aklr/core/validation/stages/_measure.py`, measurement profile 19); a Point is served from 1.05
(`AKLR_MIN_SPEEDUP`).

This file's CI runs and the predecessor repository's runs of E1 and E2 used FlyDSL 0.3.2, not the FlyDSL of their
CI image (`../pitfalls/measured_a_different_build.md`).

Per batch, the range over that batch's pool sizes, in one CI run of this file:

| Batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| Body | small | small | small | slices | slices | slices | band | band | band |
| Points | 9 | 8 | 6 | 6 | 9 | 8 | 7 | 4 | 4 |
| Speedup *(field, E5)* | 1.046–1.110 | 1.206–1.301 | 1.327–1.390 | 1.296–1.311 | 1.203–1.257 | 1.241–1.269 | 1.434–1.674 | 1.572–1.727 | 1.796–1.837 |

61 of the 61 Points are served, and all pass accuracy (`accuracy: passed` in each workload).
`a989fd4b5f22b3d2.json`, the file this one replaced, whose kernel differs from this one in more than the per-page
descriptors, read 1.041–1.884 over its 58 Points in CI *(field, E1)*. This file's validation against main, outside
CI, reads 1.030–1.809 over
its 61 Points, batch 1 at 1.030–1.134, the 262,144-page Points at 1.030 (batch 1), 1.240 (16) and 1.409 (64)
*(field, E4)*.

## Reproduce

On an MI355X, in a checkout that holds this file:

```
python -m aklr validate aklr/ops/mla/impls/flydsl/fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a.json
```

It runs the audit and bench stages and writes `aklr_validation/report.md` with the per-Point speedup
(`aklr/ops/README.md` § Commands). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). The local result does not count; CI's
measurement is the one dispatch reads.

## Rollback

Delete the schema file (in a PR): its Points go unserved, since no other file serves them.
`AKLR_DISABLE=fp8_paged_mqa_logits@1.0` drops this kernel and also the 32-head HIP one GLM-5.x's indexer takes
(`aklr/ops/mla/impls/hip/fp8_paged_mqa_logits_page_size64_n_heads32/`). A Point CI measures below 1.05 stops being served without a rollback, as the batch-1 Points can. Roll
back for a wrong result CI does not catch: a consumer that reads logits past a request's last page, a process that
calls the kernel on two devices or two streams at batch ≥ 64, or a FlyDSL or LLVM upgrade.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` the schema embeds; `<schema>` is
aklr/ops/mla/impls/flydsl/fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter 1.041–1.884 per Point, 58 of 58 ≥ 1.03; batch 1 1.041–1.101, 2 1.211–1.284, 4 1.272–1.372, 8 1.117–1.165, 16 1.212–1.264, 32 1.195–1.269, 64 1.643–1.714, 128 1.700–1.747, 256 1.794–1.884 (fp8_paged_mqa_logits_page_size64_n_heads64, 58 Points)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] the other committed measurement: 1.024–1.891 per Point, 57 of 58 ≥ 1.03; batch 1 by pool size 105428 1.055, 109404 1.035, 111670 1.024, 111761 1.073, 115603 1.035, 119349 1.030, 120631 1.037, 123163 1.031 (fp8_paged_mqa_logits_page_size64_n_heads64)
- E3 [2026-10, gfx950, FlyDSL ?, N=?, AKLR bench_cudagraph, not CI] batch 1, 2 and 4 "1.7x, 2.1x and 2.2x faster than aiter" after the depth-2 pipeline (fp8_paged_mqa_logits_page_size64_n_heads64)
- E4 [2026-11, gfx950, FlyDSL 0.3.4.1, N=1, `python -m aklr validate --base origin/main`, not CI] speedup over aiter 1.030–1.809 per Point, 61 of 61 passed; batch 1 1.030–1.134, 2 1.244–1.302, 4 1.294–1.341, 8 1.278–1.307, 16 1.217–1.261, 32 1.241–1.271, 64 1.409–1.700, 128 1.570–1.731, 256 1.696–1.809 (fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a, PR #96)
- E5 [2026-11, gfx950, FlyDSL 0.3.2 (aiter dbd8bf5's pin), N=1, AKLR CI, measurement profile 19] speedup over aiter 1.046–1.837 per Point, 61 of 61 ≥ 1.03, accuracy passed at every Point; batch 1 1.046–1.110, 2 1.206–1.301, 4 1.327–1.390, 8 1.296–1.311, 16 1.203–1.257, 32 1.241–1.269, 64 1.434–1.674, 128 1.572–1.727, 256 1.796–1.837 (fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a; 227aa66, #115, CI run 42599763780)
- E7 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=8 rounds in one process, AKLR's `bench_cudagraph` with a 1 GiB flush per call] 2c17b79278febb6a's kernel.py (band path) against a copy forced to the slice path and against aiter, batch 64, 128, 256, each request owning its pages, contexts uniform in 8,192–49,152 tokens: band / slice speed 0.882–0.937 with the pages scattered over the pool, 0.194–0.514 with one contiguous run per request, there band at 0.266–0.471 of aiter's speed; band and slice outputs bitwise equal inside the context
- S1 aklr/ops/mla/contract.py:410 `SqnrBound(min_sqnr_db=30.0)`; compare `BOUNDED` at :414; `context_lens` drawn at `max_seq_len` only at :396; pool bytes below 0x40 at :380
- S2 aklr/ops/mla/baseline.py:98 the baseline calls aiter with `Preshuffle`, `KVBlockSize` and the logits' width only, leaving `ChunkK`, `WavePerEU` and the scheduling arguments at aiter's defaults
- S3 aklr/ops/mla/reference.py:133 the reference scales the keys before relu
- S4 aklr/ops/mla/impls/flydsl/fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a.json#kernel.py:199 `_issue_page`; MFMA at :275; scale after the head sum at :288; −inf past the context at :290; nt store at :301; a pool's descriptor or a page's own at :188 (`_key_pool`) and :218
- S5 aklr/ops/mla/impls/flydsl/fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a.json#kernel.py:334 `split_table`; `band_table` at :354; `_BAND_MIN_BLOCKS` at :46; the untuned split (`_FILL`) at :416; `small` at :422; body chosen at :876
- S6 aklr/ops/mla/impls/flydsl/fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a.json#kernel.py:524 `small_kernel`: page ids by `v_readlane` at :628, look-ahead at :639, aimed at `kv_bytes` at :222
- S7 aklr/ops/mla/impls/flydsl/fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a.json#kernel.py:744 `sort_kernel`: LDS-atomic histogram at :782, scatter at :833; `ORD` / `OFF` allocated with `device="cuda"` at :850; `flyc.compile` at :917
- S8 aklr/ops/mla/impls/flydsl/fp8_paged_mqa_logits_page_size64_n_heads64/2c17b79278febb6a.json#kernel.py:172 the `small` branch of `_slice_plan`, which no launched kernel reaches
