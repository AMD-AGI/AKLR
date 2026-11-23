---
title: "Points with degenerate inputs let a kernel win on a path serving does not take"
kind: pitfall
scope: [AKLR Points and the contract distributions that draw their inputs, kernels with a fast path keyed on input values (per-request lengths, pool sizes), case pages that cite CI speedups]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "aklr/plugins/aiter/mla/impls/flydsl/flash_mla_with_kvcache_causal_bf16_lse_has_block_table_has_cu_seqlens_k_kv_cache_bfloat16_query_length8_n_heads96_page_size1/f6f727be6a8dd6e8.json"
  - "aklr/plugins/aiter/mla/plugin.py:279"
  - "aklr/core/contract/semantics/_distributions.py:297"
---

# Points with degenerate inputs

## Symptom

A kernel passes accuracy and CI stores a speedup at each of its Points, yet on the inputs serving sends it is
several times slower than the baseline. #201's FlashMLA kernel for a bf16 KV cache, 8 query tokens and 96 heads
(`f6f727be`) is stored at 1.25–2.56× over aiter's Gluon MLA decode at its 11 Points *(field, E2)*, and timed 1.4–1.6×
faster than it on a Point's inputs at batch 4 and 8, where every request holds 8 KV rows; at the same batch with 2.9K–7.3K KV rows
per request it was 4.6–8.7× slower, and both were within the contract's bounds *(field, E1)*.

## Root cause

- **The kernel bets on the Points' inputs.** It first computes every request as if it held exactly 8 KV rows
  starting at row 8·b, and stores that; a request whose rows differ sets `slow` and is recomputed serially in 8-row
  chunks, one wave per (request, query, 16 heads), with no split over the keys (S1).
- **Every Point gives every request exactly 8 rows.** Each workload sets `table_width` to 8 × batch, and the
  contract draws `cu_seqlens_k` with `min_seqlen` = `query_length` (8) and `total` = `table_width` (S2). The sampler
  gives each request `min_seqlen` rows and spreads only the spare rows, `total − requests · min_seqlen`, which is 0
  here (S3). So the Points time only the fast path, and accuracy holds because the slow path is right too.
- **Where it runs.** `n_pages` and `table_width` are var axes without `any_size`, so the file is served only at its
  listed Points (`aklr/plugins/aiter/README.md` § Inputs and workloads), pools of 8–112 one-row pages; a serving KV pool is
  far larger, so dispatch does not pick this file there. What misleads is the stored speedup, and any page or
  decision that quotes it.

## How to detect

- For each input a kernel branches on (lengths, offsets, page tables, pool size), read how the contract draws it at
  the Points: a distribution whose spare is 0, or a pool sized to exactly the rows the requests use, gives every
  call the same shape of input.
- Time the kernel and the baseline on the same batch with lengths drawn as serving sees them, both checked against
  the contract's reference (E1's second arm). A ratio that flips against the Points' is this trap.
- Look in the kernel for a fast path guarded by a test on the input's layout (`slow = …`, a length equal to a
  constant), and ask whether any Point reaches the other branch.

## Fix or workaround

- Give the Points realistic pool sizes and per-request lengths: the workload and its distribution decide what is
  timed (`aklr/plugins/aiter/README.md` § Inputs and workloads).
- A case page does not cite a speedup measured only on degenerate Points; it states the input condition, and the
  time on realistic inputs if measured.

## Scope

- Any contract whose distributions can collapse to one input shape at the Points; observed on one FlyDSL MLA
  decode kernel on MI355X.
- Not this trap: a Point set that misses a whole shape the deployment runs (`../attention/measurement.md` §8).

## Evidence

- E1 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=60 rounds per batch, the kernel and the contract's baseline (aiter's `mla_gluon`) alternating every round with the order swapped on odd rounds, a 1 GiB write before each call, CUDA-event time per call, median; both checked against the contract's reference under its bounds (output 1e-2, lse 1e-3)] f6f727be6a8dd6e8: at a Point's inputs (8 KV rows per request), batch 4 / 8: kernel 7.6 / 9.6 µs, baseline 10.3 / 15.2; at the same batches with each request's length drawn uniformly from 512–8192 rows (2.9K–7.3K per request), batch 4 / 8: kernel 1690.5 / 1710.5 µs, baseline 195.2 / 373.3; both passed in both arms (bf16 FlashMLA decode, 96 heads, 8 query tokens, page size 1)
- E2 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5, read from the workflow), N=1, AKLR CI, stored speedup per Point] f6f727be6a8dd6e8 at batch 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14 (n_pages = table_width = 8 × batch): 2.556, 2.435, 1.250, 1.262, 1.248, 1.680, 1.666, 1.666, 1.651, 1.893, 1.860 (AMD-AGI/AKLR@e193b1d9, CI run 44081029964)
- S1 aklr/plugins/aiter/mla/impls/flydsl/flash_mla_with_kvcache_causal_bf16_lse_has_block_table_has_cu_seqlens_k_kv_cache_bfloat16_query_length8_n_heads96_page_size1/f6f727be6a8dd6e8.json#kernel.py:114 `slow = (rt != b * CH) | (ln != fx.Int32(CH))`; the 8-row result stored first at :189; the serial 8-row chunks under `if slow:` at :190-198
- S2 aklr/plugins/aiter/mla/plugin.py:279 `cu_seqlens_k` drawn by `CuSeqlensDistribution(min_seqlen="query_length", total="table_width")`; `n_pages` (:112) and `table_width` (:118) are var axes without `any_size`
- S3 aklr/core/contract/semantics/_distributions.py:297 `spare = self.total - parts * self.min_seqlen`; each part gets `min_seqlen` plus its share of the spare (:305)
