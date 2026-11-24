---
title: "Benchmark sizes with degenerate inputs let a kernel win on a path serving does not take"
kind: pitfall
scope: [benchmark sizes and the distributions that draw their inputs, kernels with a fast path keyed on input values (per-request lengths, pool sizes), case pages that cite a speedup measured on such inputs]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
---

# Benchmark sizes with degenerate inputs

## Symptom

A kernel passes accuracy and a benchmark records a speedup at each of its sizes, yet on the inputs serving sends it is
several times slower than the baseline. A FlashMLA kernel for a bf16 KV cache, 8 query tokens and 96 heads
was recorded at 1.25–2.56× over aiter's Gluon MLA decode at its 11 benchmark sizes *(field, E2)*, and timed 1.4–1.6×
faster than it on a benchmark size's inputs at batch 4 and 8, where every request holds 8 KV rows; at the same batch with 2.9K–7.3K KV rows
per request it was 4.6–8.7× slower, and both were within the accuracy bounds *(field, E1)*.

## Root cause

- **The kernel bets on the benchmark's inputs.** It first computes every request as if it held exactly 8 KV rows
  starting at row 8·b, and stores that; a request whose rows differ sets `slow` and is recomputed serially in 8-row
  chunks, one wave per (request, query, 16 heads), with no split over the keys:

  ```python
  rt = fx.Int32(fx.ptr_load(fx.add_offset(cup, b)))  # request b's first KV row, from the cumulative lengths
  re = fx.Int32(fx.ptr_load(fx.add_offset(cup, b + fx.Int32(1))))
  ln = re - rt
  slow = (rt != b * CH) | (ln != fx.Int32(CH))  # CH = 8: anything but 8 rows starting at row 8 b
  # ...
  finish(chunk(0, slot_s, init, qpos + fx.Int32(1)))  # the 8-row result, computed and stored first
  if slow:  # every other request: recomputed serially, one 8-row chunk at a time
      for ci, st in range(fx.Int32(0), nch, fx.Int32(1), init=init):
          c = fx.Int32(ci)
          cidx = c * CH + key
          cidx = (cidx < nvalid).select(cidx, fx.Int32(0))
          slot = fx.Int32(fx.ptr_load(fx.add_offset(btp, rt + cidx)))
          nst = chunk(c, slot, st, nvalid)
          res = yield nst
      finish(res)
  ```
- **Every benchmark size gives every request exactly 8 rows.** Each workload sets the page table's width to 8 × batch, and
  the per-request lengths (`cu_seqlens_k`) are drawn with a minimum of the query length (8) and a total of the table's width. The sampler
  gives each request the minimum and spreads only the spare rows, total − requests · minimum, which is 0
  here. So the benchmark times only the fast path, and accuracy holds because the slow path is right too.
- **Where it runs.** The benchmark's pools hold 8–112 one-row pages; a serving KV pool is
  far larger. What misleads is the recorded speedup, and any page or
  decision that quotes it.

## How to detect

- For each input a kernel branches on (lengths, offsets, page tables, pool size), read how the benchmark draws
  it: a distribution whose spare is 0, or a pool sized to exactly the rows the requests use, gives every
  call the same shape of input.
- Time the kernel and the baseline on the same batch with lengths drawn as serving sees them, both checked against
  the reference (E1's second arm). A ratio that flips against the benchmark's is this trap.
- Look in the kernel for a fast path guarded by a test on the input's layout (`slow = …`, a length equal to a
  constant), and ask whether any benchmark size reaches the other branch.

## Fix or workaround

- Give the benchmark realistic pool sizes and per-request lengths: the workload and its distribution decide what is
  timed.
- A case page does not cite a speedup measured only on degenerate inputs; it states the input condition, and the
  time on realistic inputs if measured.

## Scope

- Any benchmark whose input distributions can collapse to one input shape; observed on one FlyDSL MLA
  decode kernel on MI355X.
- Not this trap: a set of benchmark sizes that misses a whole shape the deployment runs (`../attention/measurement.md` §8).

## Evidence

- E1 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=60 rounds per batch, the kernel and the baseline (aiter's `mla_gluon`) alternating every round with the order swapped on odd rounds, a 1 GiB write before each call, CUDA-event time per call, median; both checked against the reference under the bounds output 1e-2, lse 1e-3] the FlashMLA kernel: at a benchmark size's inputs (8 KV rows per request), batch 4 / 8: kernel 7.6 / 9.6 µs, baseline 10.3 / 15.2; at the same batches with each request's length drawn uniformly from 512–8192 rows (2.9K–7.3K per request), batch 4 / 8: kernel 1690.5 / 1710.5 µs, baseline 195.2 / 373.3; both passed in both arms (bf16 FlashMLA decode, 96 heads, 8 query tokens, page size 1)
- E2 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5), N=1, lower-bound speedup over aiter's `mla_gluon` per size, one run] the FlashMLA kernel at batch 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14 (pool pages = page table width = 8 × batch): 2.556, 2.435, 1.250, 1.262, 1.248, 1.680, 1.666, 1.666, 1.651, 1.893, 1.860
