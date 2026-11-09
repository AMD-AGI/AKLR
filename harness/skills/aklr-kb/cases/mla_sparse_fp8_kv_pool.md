---
title: "Sparse MLA decode on DeepSeek-V4's padded fp8 KV pool, 64 heads, 128 window slots plus 0, 512 or 8256 compressed-cache slots — one dequantised 32-key LDS tile feeds both GEMMs for all 64 heads; a second launch merges the partials"
kind: case
ops: [mla, attention]
dtypes: [fp8, bf16]
regimes: ["decode batch 1-16", "decode batch 32-64", "decode batch 128-256"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over the contract's baseline (sglang's TileLang dpsk_v4_fp8_attention_fwd), CI, per Point", value: "1.165–2.405; 61 of 61 Points ≥ 1.03", gpu: MI355X, workload: "flash_mla_with_kvcache_lse_sparse_lens_sinks_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128, batch 1–256 × 4–9 pool sizes", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
  - {metric: "speedup over the contract's baseline (sglang's TileLang dpsk_v4_fp8_attention_fwd), CI, per Point", value: "2.396–5.845; 61 of 61 Points ≥ 1.03", gpu: MI355X, workload: "flash_mla_with_kvcache_…_extra_page_size2_n_extra_indices8256, batch 1–256 × 4–9 pool sizes", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
  - {metric: "speedup over the contract's baseline (sglang's TileLang dpsk_v4_fp8_attention_fwd), CI, per Point", value: "2.362–5.078; 61 of 61 Points ≥ 1.03", gpu: MI355X, workload: "flash_mla_with_kvcache_…_extra_page_size64_n_extra_indices512, batch 1–256 × 4–9 pool sizes", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
sources:
  - aklr/ops/mla/impls/flydsl/flash_mla_with_kvcache_lse_sparse_lens_sinks_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128/97e39bdef97d3e9b.json
  - aklr/ops/mla/impls/flydsl/flash_mla_with_kvcache_lse_sparse_lens_extra_extra_idx_extra_lens_sinks_kv_cache_float8_e4m3fn_extra_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128_extra_page_size2_n_extra_indices8256/3b21a6af84ad0024.json
  - aklr/ops/mla/impls/flydsl/flash_mla_with_kvcache_lse_sparse_lens_extra_extra_idx_extra_lens_sinks_kv_cache_float8_e4m3fn_extra_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128_extra_page_size64_n_extra_indices512/7afdc9a6fcb42ad3.json
---

# Sparse MLA decode on the DeepSeek-V4 fp8 pool: one dequantised tile for all 64 heads

One FlyDSL source serves three schemas of `flash_mla_with_kvcache@1.0` for DeepSeek-V4-Flash decode (PR #71): the
three files embed the same `kernel.py` as `kernel.sources[0].content`, and the function names below are in it. They
are called here by their extra pool: "no extra" (window only), "extra 8256" (extra page size 2, 8256 indices) and
"extra 512" (extra page size 64, 512 indices). "batch" is the Point's rows, one decode token each.

## Applies when

- Contract `flash_mla_with_kvcache@1.0`, layout variant `padded_pages`, sparse mode (`kv_cache_format` `dsv4_fp8`):
  one query per row, 64 heads, q bf16 with 512 values per head, `sparse_lens`, attention sinks, lse returned. The
  pool is float8_e4m3fn bytes in pages of 256 slots; a slot is a 576-byte payload (448 fp8 e4m3 values, then 64 bf16
  rope values) plus an 8-byte scale slot (7 UE8M0 exponents, one per 64 values) stored after the page's payloads,
  each page padded to a multiple of 576 bytes (`Deepseekv4KvPages`). 128 indices per row into it; the extra-pool
  schemas add a second pool of the same format with its own indices and lengths.
- Points: batch 1, 2, 4, …, 256, each with 4 to 9 of nine recorded pool sizes (`n_pages` 10543–12317, paired with
  one `n_extra_pages` 105428–123163 in the extra-pool schemas): 61 Points per file. `n_pages` and `n_extra_pages`
  are var axes without `any_size`, so dispatch matches them exactly (`FastKeyCodec.match_key`): a pool of any other
  page count falls back with `no_workload`, though the kernel does not depend on it.
- Compare: `determinism: bounded`, `max_violation_fraction` 0.0; `output` atol = rtol = 1e-2, `lse` atol = rtol =
  1e-3 (`aklr/ops/mla/contract.py`). The reference defines the fp8 → bf16 step as the same bit assembly the kernel
  does, so that step matches it exactly; the error left to `compare` comes from bf16 P, bf16 partials and the bf16
  output.
- Yardstick: `flash_mla_with_kvcache_baseline` (`aklr/ops/mla/baseline.py`), which for this cache format calls
  sglang's TileLang `dpsk_v4_fp8_attention_fwd` (S3), not aiter. Each schema name has one
  kernel file and no file in another language. The extra-8256 directory also holds `69a38923eb0c686a.json`, a
  definition (empty `sources`, never served or benched) whose 19 Points (pools of 1501–5229 pages) are not this
  file's.

## Counter-signals

- **Only the recorded pool sizes are served.** The TP8 decode pool of the removed e2e stage's scenarios (12153 main,
  121529 extra pages) is a Point at batch 1, 2 and 4 only. A deployment whose pool differs by one page runs the
  TileLang kernel.
- **The thinnest margin is the window-only schema at batch ≥ 128**: 1.165–1.188 at batch 128 and 1.193–1.354 at
  batch 256 *(field, E1)*, against 1.582–2.405 at batch ≤ 64. The cause is not established.
- **The bench draws lengths and indices uniformly.** `sparse_lens` and `extra_sparse_lens` are uniform in
  [0, width] and the indices uniform over the whole pool (`IntRangeDistribution`). In serving they follow the
  model's top-k selection, whose index structure moved gather cost on another sparse kernel
  (`../attention/split_kv.md` §7); the speedup on serving inputs is not measured. At batch 1 one
  row's drawn lengths set the work: the nine batch-1 Points of extra 8256 read 2.809–5.845 *(field, E2)*, and each
  reproduces within 4.1% in the other committed run *(field, E4)*, so the spread follows each Point's inputs
  (seeded by its workload hash, `aklr/ops/README.md` § Inputs and workloads), not run noise.
- **`sm_scale` and `kv_scale` are compile-time constants.** `_build` is a `functools.cache` keyed by both values; a
  new value compiles a new kernel pair at the call that brings it, which may be inside a graph capture. The README
  has the first call of a Point compile every variant a scalar selects (`aklr/ops/README.md` § The kernel section
  and ABI). Every workload holds `sm_scale` 0.0442 and `kv_scale` 1.0.
- **Workspace kept per batch size.** `_workspace(batch, n1, n2, dev)` (`functools.cache`) holds a bf16 partial
  output and an fp32 partial lse for every slot a row can write, for the life of the process: estimated 0.26 MB,
  1.3 MB and 8.75 MB per row for no extra, extra 512 and extra 8256, so about 4.5 GB for extra 8256 at its nine batch
  sizes (estimated from the shapes). Every slot the combine reads is written by the same call's partial kernel,
  which meets the scratch rule (`../SKILL.md`, Stop table); two streams calling one Point at once would share it.
- **The pool is addressed from the start of its storage.** `_words` takes `torch.as_strided(raw, …, 0)`, offset 0
  of the storage, not `data_ptr()`, and the layout's storage test checks strides only
  (`aklr/core/contract/semantics/_layouts.py`). A pool view that starts inside a larger allocation would be read
  from the wrong base. The bench's pools start at offset 0; sglang's were not checked for this page.
- **Not the shape the decode cards recommend**, and nothing here prices the differences: the combine is a second
  launch with 4 waves and an LDS merge (`../attention/split_kv.md` §1 and
  `../gemm/grid_pricing.md` §2 found one wave per combine workgroup faster); a producer workgroup owns
  one 32-key tile, two above 1024 live keys, where §5 of the decode card asks for at least two per split; partials
  are bf16, which §5 says adds merge error.
- **Each call goes through the `@flyc.jit` launcher** `launch_all`, which re-binds its arguments on every call
  (`../flydsl/attention.md` § measurement §6, on FlyDSL 0.3.2). CI's ratio is GPU time in graph replays, so that
  host cost is outside it; an eager caller pays it.
- **64 heads are a constant** (`NHEAD`): the 128-head DeepSeek-V4-Pro definitions of this contract have no kernel.

## Mechanism

Two launches per call (`launch_all`): `part_kernel` on a grid of (batch, slots) and `comb_kernel` on (64 heads,
batch), each workgroup 256 threads, 4 waves.

**Slots.** The grid has one slot per 32 indices of each segment: 4 for the 128-wide window, plus 16 (extra 512) or
258 (extra 8256), fixed by the schema whatever the live lengths. Slot g of a segment covers keys [g·k, g·k + k): k
= 32, or 64 once that segment's live length passes 1024 (`KTIERS`, `_k_of`; only the 8256-wide segment can reach
it, `_tiers`). A slot whose first key is past the live length exits at once. Live partials stay packed at each
segment's base, so the workspace holds 4, 20 or 133 slots per row (`_slots`).

**Gather and dequantise into one LDS tile** (`stage`). Eight threads per key, 32 keys. Each key's index gives its
page (a shift by log2 of the page size) and slot; seven threads each read 64 fp8 bytes as four 16-byte global
loads plus that block's UE8M0 exponent, and build bf16 values by integer bit assembly: exponent and mantissa
shifted into place, exponent + scale − 7 added in one integer add per pair, sign moved to bit 15. The eighth
thread copies the 64 bf16 rope values. Everything lands in one bf16 LDS tile of 32 keys × 512 dims (32 KiB), stored
as 16 blocks of 32 keys × 32 dims. A key with a negative index or past the live length reads slot 0 and gets a
−1e30 bias in LDS, so its exp2 is 0 (the masking of `../attention/split_kv.md` §4). A payload is
4.5 lines of 128 bytes at phase 0 or 64, so each key touches 5 lines plus its scale (from the layout; §7 of the same
card).

**Logits, heads on the lane.** Each wave owns 16 heads and keeps their q in registers for the whole workgroup (16
loads of 16 bytes per lane). Per 16-key half, 16 `mfma_f32_16x16x32_bf16` with A = keys from LDS and B = q: C holds
one head per lane and the keys on the element index, so the row max and row sum are an in-lane fold over 8 keys
and two `shuffle_xor` (16, 32) (`../attention/mfma_layout.md` §3–4, `../attention/softmax.md`
§5). `sm_scale · kv_scale · log2 e` and the bias go in with one multiply and one add before `exp2`
(`../attention/softmax.md` §3).

**P·V from the same tile.** The values are the first 512 dims of the same latent, so P·V reads the tile again,
transposed, with `ds_read_b64_tr_b16` (`LDSReadTrans16_64b`): 64 `mfma_f32_16x16x16bf16_1k` per 32 keys and wave,
A = P as 4 bf16 per lane straight from the logits layout (keys already on the element index), B = Vᵀ, into 32 f32x4
accumulators per lane, 16 heads × 512 dims per wave (`../attention/lds.md` §1–3, `../flydsl/attention.md` § lds §2,
`../attention/mfma_layout.md` §4). One gather and one dequantisation feed both GEMMs and all 64 heads.

**Online softmax** (`step`). The running max starts at −1e20, so a block of masked keys gives P = 0, not NaN
(`../attention/softmax.md` §6.1). The accumulator holds heads on its element index, so `alpha =
exp2(m_old − m_new)` moves to that layout through 16 floats of the wave's own LDS (an `lgkmcnt(0)` wait, no
barrier) before it scales all 128 accumulators (`../attention/softmax.md` §6.2). Two barriers per
32-key step guard the tile. The row sum adds the fp32 P while P·V takes P rounded to bf16 (§5 of the softmax card
sums the rounded P). At the end a lane scales by 1/l and stores the partial as bf16, 2 bytes per store; lanes 0–15
store lse = (m + log2 l) · ln 2, or −1e30 for an empty slot.

**Combine** (`comb_kernel`). One workgroup per (head, row), 8 dims per lane. Wave w walks slots w, w + 4, … in
order, four per trip: first the max of the partial lse, then Σ exp(lse − max) · partial (bf16 read as fp32); wave 0
merges the four waves' (max, sum, vector) from LDS in wave order. It scales by 1/(1 + exp(sink − lse)) and by
`kv_scale`, writes bf16, and writes the lse without the sink; a row that read nothing gets output 0 and lse +inf.

Every reduction, in both kernels, runs in a fixed order, with no atomics. `run` calls `_build(sm_scale, kv_scale,
n1, ps1, n2, ps2)` and `_workspace`, both module-level `functools.cache` functions; a call without an extra pool
passes the window's tensors as the second segment with width 0, which compiles that segment away.

## Result

Each file is the only kernel of its schema name, so the yardstick at every Point is the defined baseline that bench
times, sglang's TileLang `dpsk_v4_fp8_attention_fwd`. `speedup` is a one-sided lower bound (α = 0.05) of baseline /
kernel time over 32 rounds of CUDA-graph replays with a 1 GiB cache flush before each call
(`aklr/core/validation/stages/_measure.py`); a Point is served from 1.03 (`AKLR_MIN_SPEEDUP`).

CI installs the repository with `--no-deps`, so these runs used the FlyDSL of the CI image, 0.3.4.1, not the
repository's pin 0.3.2 (`../pitfalls/measured_a_different_build.md`).

Per batch, the range over that batch's pool sizes (9, 9, 7, 6, 8, 8, 6, 4 and 4 Points):

| File | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | Served |
|---|---|---|---|---|---|---|---|---|---|---|
| no extra *(field, E1)* | 1.901–2.405 | 2.257–2.399 | 2.030–2.400 | 2.244–2.367 | 2.274–2.394 | 1.906–2.033 | 1.582–1.646 | 1.165–1.188 | 1.193–1.354 | 61 of 61 |
| extra 8256 *(field, E2)* | 2.809–5.845 | 4.069–5.486 | 3.772–5.269 | 3.342–3.967 | 2.811–3.162 | 2.467–2.734 | 2.396–2.625 | 2.428–2.553 | 2.482–2.560 | 61 of 61 |
| extra 512 *(field, E3)* | 2.620–3.493 | 2.674–3.571 | 3.300–3.585 | 3.114–3.414 | 2.688–2.790 | 2.362–3.614 | 2.738–3.385 | 3.429–3.910 | 4.645–5.078 | 61 of 61 |

183 of the 183 Points are served, and all pass accuracy (`accuracy: passed` in each workload). The other committed
measurement, on the same image, also served all 183 *(field, E4)*.

## Reproduce

On an MI355X, in a checkout that holds these files with these numbers (main at 0cef4f2e3), one schema at a time:

```
python -m aklr validate aklr/ops/mla/impls/flydsl/flash_mla_with_kvcache_lse_sparse_lens_sinks_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128/97e39bdef97d3e9b.json
```

and the same for the other two paths under `sources`. Each runs the audit and bench stages and writes
`aklr_validation/report.md` with the per-Point speedup (`aklr/ops/README.md` § Commands). The pools are large
(estimated from the shapes: 1.8 GB main pool at 12317 pages, 4.6 GB extra pool for extra 512), and bench draws them
again for every input set. Add `--base origin/main` only on a branch that edits the file: with the base set, bench
measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). The local result does not count; CI's
measurement is the one dispatch reads. To read the kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete a schema file (in a PR): its Points fall back to the caller's own kernel, sglang's TileLang attention, since
no other file serves them. `AKLR_DISABLE=flash_mla_with_kvcache@1.0` drops all three, and nothing else: the
contract's other files are definitions. A Point CI measures below 1.03 stops being served without a rollback. Roll
back for a wrong result CI does not catch, such as a pool view at a non-zero storage offset or a FlyDSL or LLVM
upgrade, or when the cached workspaces do not fit the server's memory budget.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` the schemas embed (the same text in all three);
lines are cited in the window-only file.

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over sglang's TileLang dpsk_v4_fp8_attention_fwd 1.165–2.405 per Point, 61 of 61 ≥ 1.03; batch 1 1.901–2.405, 2 2.257–2.399, 4 2.030–2.400, 8 2.244–2.367, 16 2.274–2.394, 32 1.906–2.033, 64 1.582–1.646, 128 1.165–1.188, 256 1.193–1.354 (flash_mla_with_kvcache_lse_sparse_lens_sinks_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128, 61 Points; 5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup 2.396–5.845 per Point, 61 of 61 ≥ 1.03; batch 1 2.809–5.845, 2 4.069–5.486, 4 3.772–5.269, 8 3.342–3.967, 16 2.811–3.162, 32 2.467–2.734, 64 2.396–2.625, 128 2.428–2.553, 256 2.482–2.560 (…_extra_page_size2_n_extra_indices8256, 61 Points; 5571f4589, CI run 40613440602)
- E3 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup 2.362–5.078 per Point, 61 of 61 ≥ 1.03; batch 1 2.620–3.493, 2 2.674–3.571, 4 3.300–3.585, 8 3.114–3.414, 16 2.688–2.790, 32 2.362–3.614, 64 2.738–3.385, 128 3.429–3.910, 256 4.645–5.078 (…_extra_page_size64_n_extra_indices512, 61 Points; 5571f4589, CI run 40613440602)
- E4 [2026-10, gfx950, FlyDSL 0.3.4.1, N=2, AKLR CI] the two committed measurements, in commit order: no extra 1.167–2.439 then 1.165–2.405, extra 8256 2.374–5.616 then 2.396–5.845, extra 512 2.363–5.019 then 2.362–5.078, all 183 Points ≥ 1.03 both times; per-Point ratio between the runs 0.924–1.070; extra 8256 at batch 1 by pool size 2.753, 5.616, 5.542, 5.027, 5.246, 5.523, 3.468, 4.491, 2.791 then 2.809, 5.845, 5.595, 5.071, 5.357, 5.526, 3.444, 4.494, 2.861 (the three schemas; 102877ca0, CI run 40154665266; 5571f4589, CI run 40613440602)
- S1 aklr/ops/mla/contract.py:346 compare `BOUNDED`; `output` tolerance at :315, `lse` at :326; `n_pages` declared without `any_size` at :114
- S2 aklr/ops/mla/reference.py:50 the reference's dequantisation is "the kernel's literal bf16 bit assembly, not fp8 scaling"
- S3 aklr/ops/mla/baseline.py:35 `flash_mla_with_kvcache_baseline` calls sglang's `dpsk_v4_fp8_attention_fwd` (:40) for the `dsv4_fp8` cache format
- S4 aklr/core/contract/fastkey/_codec.py:75 `match_key` drops only `any_size` axes from the dispatch key
- S5 aklr/ops/mla/impls/flydsl/flash_mla_with_kvcache_lse_sparse_lens_sinks_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128/97e39bdef97d3e9b.json#kernel.py:35 `KTIERS`; `_slots` at :74; `_workspace` at :84; `_build` keyed by `sm_scale`, `kv_scale` at :93
- S6 aklr/ops/mla/impls/flydsl/flash_mla_with_kvcache_lse_sparse_lens_sinks_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128/97e39bdef97d3e9b.json#kernel.py:171 `stage`: bias at :190, exponent arithmetic at :204, rope copy at :232
- S7 aklr/ops/mla/impls/flydsl/flash_mla_with_kvcache_lse_sparse_lens_sinks_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128/97e39bdef97d3e9b.json#kernel.py:274 logits MFMA; max and sum shuffles at :285; alpha through LDS at :300; transposed reads at :319 and P·V MFMA at :330
- S8 aklr/ops/mla/impls/flydsl/flash_mla_with_kvcache_lse_sparse_lens_sinks_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128/97e39bdef97d3e9b.json#kernel.py:374 `comb_kernel`: wave slices at :411, wave-order merge at :441, sink factor at :459
- S9 aklr/ops/mla/impls/flydsl/flash_mla_with_kvcache_lse_sparse_lens_sinks_kv_cache_float8_e4m3fn_padded_pages_query_length1_n_heads64_page_size256_n_indices128/97e39bdef97d3e9b.json#kernel.py:523 `_words` views the storage from offset 0 (:527); aklr/core/contract/semantics/_layouts.py:214 `Deepseekv4KvPages.storage_test` checks strides only
