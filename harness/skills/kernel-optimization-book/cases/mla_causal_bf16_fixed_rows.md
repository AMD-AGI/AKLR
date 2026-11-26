---
title: "Causal MLA over a paged bf16 KV pool, 8 query tokens per request, 96 heads — one wave per (request, query token, 16 heads) with a fast path for requests of exactly 8 KV rows, the only input its benchmark sizes give; any other request walks its rows serially in the same wave"
kind: case
ops: [mla]
dtypes: [bf16]
regimes: ["batch 1-14", "8 KV rows per request", "2.9K-7.3K KV rows per request"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=60
measured:
  - {metric: "lower-bound speedup over aiter's `mla_gluon`, per size, one run, on inputs where every request holds exactly 8 KV rows", value: "1.248–2.556", gpu: MI355X, workload: "batch 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14; pool pages = page table width = 8 × batch", date: "2026-11"}
  - {metric: "time per call, kernel / aiter's `mla_gluon` on the same inputs, median of 60 rounds, 8 KV rows per request", value: "7.6 / 10.3 µs at batch 4, 9.6 / 15.2 µs at batch 8: 1.4–1.6× faster", gpu: MI355X, workload: "batch 4 and 8, a benchmark size's inputs", date: "2026-11"}
  - {metric: "time per call, kernel / aiter's `mla_gluon` on the same inputs, median of 60 rounds, 2.9K–7.3K KV rows per request", value: "1690.5 / 195.2 µs at batch 4, 1710.5 / 373.3 µs at batch 8: 4.6–8.7× slower", gpu: MI355X, workload: "batch 4 and 8, each request's length drawn uniformly from 512–8192 rows", date: "2026-11"}
---

# Causal bf16 MLA, 8 query tokens per request: a fast path that assumes 8 KV rows per request

One FlyDSL kernel for causal MLA with FlashMLA's API over a paged bf16 KV pool, 8 query tokens per request and 96
heads: one launch, one wave per (request, query token, 16 heads). It is a counter-example: its fast path assumes that
every request holds exactly 8 KV rows, which is what each of its benchmark sizes gives it, and any other request is
recomputed serially in the same wave, 8 rows at a time. It was recorded at 1.25–2.56× over aiter's Gluon MLA decode
at its 11 benchmark sizes *(field, E1)* and was 4.6–8.7× slower than it at 2.9K–7.3K
KV rows per request *(field, E2)*. The trap is
[pitfalls/degenerate_points](../pitfalls/degenerate_points.md); the card on splitting a long KV range over
workgroups, which this kernel does not do, is [attention/split_kv](../attention/split_kv.md).

## Applies when

- **Op.** Causal MLA over a paged KV pool, FlashMLA's API: each request attends to its own KV rows, and query token
  i (0 to 7) of a request of L rows reads its first L − 8 + i + 1 rows. A score is q · row × `sm_scale` ×
  `kv_scale`; a value is the first 512 values of a row × `kv_scale`.
- **Operands.** `q` bf16 (batch, 8, 96, 576); the pool bf16, one 576-value row per page; `block_table`, a flat int32
  page table that holds every request's pool rows back to back; `cu_seqlens_k`, int32 cumulative lengths (batch + 1
  entries) that give each request's first entry in the table and its length. Outputs: `output` bf16
  (batch, 8, 96, 512) and `lse` fp32 (batch, 8, 96).
- **Sizes.** 11 benchmark sizes, batch 1–8, 10, 12 and 14, each with a pool of 8 × batch pages and a page table of
  the same width *(field, E1)*, so every request holds exactly 8 rows
  (§ Counter-signals).
- **Baseline.** aiter's Gluon MLA decode (`mla_gluon`).
- **Accuracy.** The kernel is held to `bounded` accuracy: each output stays within its error bound against the
  reference, and nothing is asked across runs. The timing run of § Result checked the kernel and the baseline under
  the bounds 1e-2 on `output` and 1e-3 on `lse` *(field, E2)*.

## Counter-signals

- **Every benchmark size takes the fast path.** Each size sets the page table's width to 8 × batch, and the
  per-request lengths (`cu_seqlens_k`) are drawn with a minimum of the query length, 8, and a total of the table's
  width. The sampler gives each request the minimum and spreads only the spare rows, total − requests × minimum,
  which is 0 here. So the benchmark times only the fast path, and accuracy holds because the serial path is right
  too. The recorded 1.25–2.56× *(field, E1)* is the fast path's.
- **At realistic lengths the kernel is several times slower than the baseline.** At batch 4 and 8, with each
  request's length drawn uniformly from 512–8192 rows (2.9K–7.3K per request), it took 1690.5 and 1710.5 µs per call
  against the baseline's 195.2 and 373.3, 4.6–8.7× slower, both within the accuracy bounds
  *(field, E2)*.
- **Where it runs.** The benchmark's pools hold 8–112 one-row pages; a KV pool in serving is far larger. What
  misleads is the recorded speedup, and any page or decision that quotes it: this page gives it only with its input
  condition and beside the times at realistic lengths (§ Result).
- **A request of 8 rows can take the serial path too.** The test is on where the request's rows start in the page
  table as well as on their count (§ Code): a request of 8 rows that does not start at entry 8b of the table,
  because an earlier request holds another count, is recomputed like any other.
- **Nothing splits a request's rows over waves** (§ Mechanism). The split-KV card's rule for rows of unequal length
  is a split of the batch's summed steps over one persistent workgroup per CU, one partial per row segment, merged
  in workgroup order (`../attention/split_kv.md` §4); it was measured on dense MLA kernels over an fp8 KV pool, not
  on this one.
- **P·V runs on `mfma_f32_16x16x16bf16_1k`.** For P·V with few query rows per head the book's rule is `16x16x32`,
  not that atom, measured on other MLA kernels (`../attention/mfma_layout.md` §5); the switch was not measured on
  this kernel.

## Mechanism

- **Grid.** One launch of 48 × batch one-wave workgroups: one wave per (request, query token, 16 of the 96 heads),
  whatever the requests' lengths. A wave holds the q of its 16 heads in registers (lane l holds head l mod 16 of
  them) and one online-softmax state: a running max and sum per head and 32 fp32x4 accumulators per lane, 16 heads ×
  512 values in all.
- **A chunk** is 8 KV rows. Each lane loads a distinct 16 B of one of the rows, nine times; 18
  `mfma_f32_16x16x32_bf16` with the rows as A and q as B give the scores with one head per lane and the rows on the
  element index (`../attention/mfma_layout.md` §4). The rows' first 512 values are written to LDS between two
  barriers and read back transposed (`LDSReadTrans16_64b`; `../attention/lds.md` §1–3, `../flydsl/attention.md`
  § lds §2) for the 32 `mfma_f32_16x16x16bf16_1k` of P·V. The scores are multiplied by `sm_scale` · `kv_scale` ·
  log2 e, the rows past the query's causal count are set to −1e30, and the chunk is folded into the state as one
  online-softmax step: the new max, `exp2(old max − new max)` on the sum and on the accumulators, `exp2` of the
  scores; each `exp2` carries `fastmath="afn"` (`../attention/softmax.md` §2, §6.2; `../flydsl/attention.md`
  § softmax §2; `../flydsl/language/limits.md` § Unclassified).
- **The fast path** runs first in every wave, whatever its request holds. It reads entries 8b to 8b + 7 of the page
  table for request b (clamped into the table and the pool), folds that one chunk with the causal count of an 8-row
  request, i + 1 rows for query token i, then normalizes and stores `output` and `lse`.
- **The serial path.** A wave whose request does not start at entry 8b of the table, or does not hold 8 rows, starts
  again from an empty state and folds, one after another in a run-time loop, the chunks that hold the rows its query
  may read: ⌈(L − 8 + i + 1) / 8⌉ chunks for a request of L rows, each trip loading its 8 page-table entries and then
  the rows. It then normalizes and stores `output` and `lse` again, over what the fast path stored. A long request
  is one sequential walk in each of its 48 waves.
- **Build.** The kernel is compiled for one batch, page-table length and pool size, with `sm_scale` and `kv_scale`
  folded in as constants, by a builder cached on those five values.

## Code

### 1. A fast path keyed on the request's rows, and the serial walk behind it

The kernel's test of a request's rows and the two paths behind it, a counter-example. It bets that every request
holds exactly 8 KV rows laid back to back in the page table, request b at entries 8b to 8b + 7, which is what each
of its benchmark sizes gives it; the first `finish(chunk(0, …))` computes and stores that 8-row result in every
wave. The bet fails for any other request, one of another length or one that starts elsewhere in the table: `slow`
is set and the wave starts over, folding the `nch` chunks of 8 rows that hold the `nvalid` rows its query may read,
one per trip of a run-time loop. `cup` is the cumulative lengths (`cu_seqlens_k`), `btp` the page table, `b` the
wave's request and `qpos` its query token, `rt` the index of the request's first entry in the page table
and `ln` its row count, `key` the lane's row within a chunk, `slot_s` the lane's pool row under
the 8-row guess and `init` the empty online-softmax state; `chunk` folds 8 rows into the state it is given, and
`finish` normalizes and stores the wave's `output` and `lse`. The fix the trap page states is on the benchmark:
sizes with realistic pool sizes and per-request lengths, so that both branches are timed
(`../attention/measurement.md` §8).

```python
rt = fx.Int32(fx.ptr_load(fx.add_offset(cup, b)))  # index of request b's first page-table entry, from the cumulative lengths
re = fx.Int32(fx.ptr_load(fx.add_offset(cup, b + fx.Int32(1))))
# ...
ln = re - rt
# ...
slow = (rt != b * CH) | (ln != fx.Int32(CH))  # CH = 8: anything but 8 rows starting at page-table entry 8 b
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

Not this: a fast path keyed on the layout that the benchmark's inputs have, with every other input left to a path
that no benchmark size reaches and that does not split the keys over waves. This kernel was recorded at 1.25–2.56×
over aiter's Gluon MLA decode at its 11 benchmark sizes *(field, E1)* and was
4.6–8.7× slower than it at 2.9K–7.3K KV rows per request *(field, E2)*.

Explained in [pitfalls/degenerate_points](../pitfalls/degenerate_points.md) § Root cause.

## Result

Two measurements against aiter's Gluon MLA decode (`mla_gluon`), on FlyDSL 0.3.2.

**The benchmark sizes, which reach only the fast path.** A lower-bound speedup (`../gemm/measurement.md` §1) per
size, one run. Pool pages = page table width = 8 × batch, so every request holds exactly 8 KV rows:

| Batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Speedup over aiter at 8 KV rows per request *(field, E1)* | 2.556 | 2.435 | 1.250 | 1.262 | 1.248 | 1.680 | 1.666 | 1.666 | 1.651 | 1.893 | 1.860 |

**Batch 4 and 8, each on both kinds of input,** on an MI355X. 60 rounds per batch, the kernel and the baseline
alternating every round with the order swapped on odd rounds, a 1 GiB write before each call; the median CUDA-event
time per call:

| KV rows per request *(field, E2)* | Batch | Kernel | aiter |
|---|---|---|---|
| 8, a benchmark size's inputs | 4 | 7.6 µs | 10.3 µs |
| 8, a benchmark size's inputs | 8 | 9.6 µs | 15.2 µs |
| 2.9K–7.3K, each length drawn uniformly from 512–8192 | 4 | 1690.5 µs | 195.2 µs |
| 2.9K–7.3K, each length drawn uniformly from 512–8192 | 8 | 1710.5 µs | 373.3 µs |

On a benchmark size's inputs the kernel was 1.4–1.6× faster than the baseline; at 2.9K–7.3K rows per request it was
4.6–8.7× slower *(field, E2)*. The kernel and the baseline passed the accuracy check
in both arms.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5), N=1, lower-bound speedup over aiter's `mla_gluon` per size, one run] the FlashMLA kernel at batch 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14 (pool pages = page table width = 8 × batch): 2.556, 2.435, 1.250, 1.262, 1.248, 1.680, 1.666, 1.666, 1.651, 1.893, 1.860
- E2 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=60 rounds per batch, the kernel and the baseline (aiter's `mla_gluon`) alternating every round with the order swapped on odd rounds, a 1 GiB write before each call, CUDA-event time per call, median; both checked against the reference under the bounds output 1e-2, lse 1e-3] the FlashMLA kernel: at a benchmark size's inputs (8 KV rows per request), batch 4 / 8: kernel 7.6 / 9.6 µs, baseline 10.3 / 15.2; at the same batches with each request's length drawn uniformly from 512–8192 rows (2.9K–7.3K per request), batch 4 / 8: kernel 1690.5 / 1710.5 µs, baseline 195.2 / 373.3; both passed in both arms (bf16 FlashMLA decode, 96 heads, 8 query tokens, page size 1)
