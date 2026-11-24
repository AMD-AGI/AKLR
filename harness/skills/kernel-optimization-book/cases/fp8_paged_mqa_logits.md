---
title: "fp8 paged MQA logits (DeepSeek-V4 indexer), 64 heads, page 64 — 16 fp8 16x16x128 MFMAs per page with the head sum kept in registers, a merge-free split over pages, and a band-sorted grid from batch 64"
kind: case
ops: [mla, indexer_logits]
dtypes: [fp8]
regimes: ["one query token per sequence", "paged KV", "batch 1-4", "batch 8-32", "batch 64-256"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's deepgemm_fp8_paged_mqa_logits, one-sided lower bound of the aiter / kernel time ratio over CUDA-graph replays, per size", value: "1.046–1.837; 61 of 61 sizes ≥ 1.03", gpu: MI355X, workload: "64 heads, page size 64, preshuffled fp8 keys, one query token per request; batch 1–256 × 4–9 pool sizes (105428–262144 pages), 4097-page block tables", date: "2026-11-09"}
---

# fp8 paged MQA logits: whole pages per wave, no merge

One FlyDSL kernel for the DeepSeek-V4 indexer's fp8 paged MQA logits, `logits[b, t] = Σ_h weights[b, h] · relu(q[b, h] ·
k_t) · scale_t`, added for DeepSeek-V4-Flash with one query token per request, as in decode. "batch" is the number of
requests in a call.

## Applies when

- The operands of aiter's `deepgemm_fp8_paged_mqa_logits` at page size 64 with preshuffled keys and 64 heads: q fp8 e4m3
  (batch, 1, 64, 128); the key pool uint8 (n_pages, 64, 1, 132), a page being 64 tokens × 128 fp8 values in aiter's 16 ×
  16 preshuffled tiles followed by 64 fp32 scales; fp32 weights (batch, 64); int32 `context_lens` and `block_tables`
  (batch, max_blocks); fp32 logits (batch, max_seq_len) with max_seq_len = 64 · max_blocks.
- Sizes: batch 1, 2, 4, …, 256, each with 4 to 8 of eight recorded pool sizes (105428–123163 pages), and batch 1, 16 and
  64 with a 262,144-page pool (2.2 GB, one size per body), every one with 4097-page tables (262,208 logits per request):
  61 sizes. A call may bring any pool size and table width: the kernel compiles for the call's exact (batch, table
  width, logits width, pool size) (§ Counter-signals).
- Accuracy: bounded, logits within an SQNR of 30 dB of the reference.
- Yardstick: aiter's `deepgemm_fp8_paged_mqa_logits` (`aiter/ops/triton/attention/pa_mqa_logits.py`) with `Preshuffle`,
  `KVBlockSize` 64 and the logits' width as `max_model_len`; `ChunkK`, `WavePerEU` and the scheduling arguments stay at
  aiter's defaults. On gfx950 that function takes its Gluon kernel when Triton is 3.5 or later (read from the source;
  the Triton of the measuring image was not checked).
- This is the one kernel for these operands, with none in another language. DeepSeek-V4-Pro's sizes of the same op, 41
  with pools of 6621–52289 pages, are not among the 61 and were not timed.

## Counter-signals

- **Batch 1 has the thinnest margin**: 1.046–1.110 over its nine pool sizes in one run *(field, E5)*. In a run of the
  kernel this one replaced, the size at 111670 pages read 1.024 *(field, E2)*. Which batch-1 sizes clear the bar (§
  Result) depends on the run.
- **The timed workload may not be serving's.** Every size draws full contexts (`context_lens` = `max_seq_len`) and block
  tables uniform over the whole pool, so requests share pages: at batch 256 the tables name 8.5–9.4 entries per pool
  page (estimated: 256 · 4097 over 123,163 to 111,670 pages). The band path exists to catch those cross-request re-reads
  in cache (a comment in its source says so). Whether serving's block tables look like the timed ones is not
  established. With each request owning its pages (contexts of 8,192–49,152 tokens, batch 64–256), the band path ran at
  0.88–0.94 of the slice path's speed with the pages scattered over the pool, and at 0.19–0.51 (0.27–0.47 of aiter's)
  with each request's pages one contiguous run *(field, E7)*: such a run lies in one or two of the 128 bands, so one or
  two workgroups walk the whole request. The gain of the per-batch slice counts there is not measured.
- **Logits past a request's last page are not written.** Pages at or past ⌈context / 64⌉ are not visited; −inf is
  written only to the tokens past the context inside the last page. A reference that fills −inf up to `max_seq_len`
  differs there; the timed full contexts do not reach the difference, and aiter leaves those logits unspecified as well
  (from a source comment, not checked here).
- **The scale is applied after relu and the head sum.** That equals the reference, which scales the keys before relu,
  only for non-negative scales; the timed inputs keep every byte of the pool below 0x40, so their scales are
  non-negative.
- **Scratch per shape, on the current device.** The build function allocates the band path's order and offsets buffers
  and a warm-up buffer with `device="cuda"`, the current device, while its cache key holds no device; scratch belongs on
  the device of an argument. The sort prepass rewrites every entry the band body reads before it reads it, but every
  call of a shape shares them, so two streams calling one shape at once would race.
- **Table widths not timed.** Every timed size is a 4097-page table, but a call may bring any width, and the split and
  the body are picked from it: under 1024 pages batch 64–256 take the slice body instead of the band path, and batches
  without a tuned slice count take an untuned one. Those were not measured with the ruler of § Result.
- **A shape compiles on its first call.** The compiled kernel is cached by the exact (batch, max_blocks, max_seq_len,
  n_pages), so a batch that has run at one shape is not warm at another: a pool size or table width first seen inside a
  CUDA graph capture compiles there. sglang runs each captured shape eagerly first.
- **Code no timed size reaches**: the slice body's variant with compile-time slice bounds (the slice body is launched
  only at batch ≥ 8; batch ≤ 4 has a body of its own), and the tuned slice counts for batch 64, 128 and 256 (the band
  path sets its own split from 1024 pages).
- **Tuning numbers in the kernel's source comments** (2–4%, 12%, 3–12% and others) were not measured with the ruler of §
  Result and are not reproduced here.
- **Pool size.** Byte offsets are 32-bit (`page id · 8448` in `Int32`) inside one buffer descriptor per pool up to 2 GiB
  (254,200 pages); a larger pool gives each page its own descriptor, which a comment in the source puts at about 4 % at
  batch 1 (not measured with the ruler of § Result). The 262,144-page sizes take that path.
- **A number from another ruler.** The change that added the batch ≤ 4 body was reported with batch 1, 2 and 4 as 1.7×,
  2.1× and 2.2× faster than aiter on another timer of CUDA-graph replays *(field, E3)*; the lower bounds of an earlier
  run for that build at those sizes were 1.041–1.101, 1.211–1.284 and 1.272–1.372 *(field, E1)*. What makes the gap is
  not established (`../pitfalls/speedup_from_another_ruler.md`).

## Mechanism

**One page, one wave.** A page holds token t, dim d at byte (t // 16) · 2048 + (d // 16) · 256 + (t % 16) · 16 + d % 16,
so two 16-byte loads give a lane the 32 contiguous dims of one token that a 16x16x128 fp8 operand takes (the preshuffled
operand of `../gemm/datapath.md` §10). Per page a wave issues one 4-byte scale load per lane and eight 16-byte key loads
per lane, then 16 `mfma_scale_f32_16x16x128_f8f6f4`: four 16-token tiles × four 16-head groups, A = q, B = keys, unit
E8M0 scales (127) (`../attention/mfma_layout.md` §8, `../gemm/mainloop.md` §5; FlyDSL spelling: `../flydsl/attention.md`
§ mfma_layout §8). C holds one token per lane and four heads per lane group. The request's query (64 heads × 128 fp8)
and weights stay in registers for the workgroup's life.

```python
# one page's loads; lane = thread % 64, r = lane % 16, g = lane // 16; pid is the page id
kv_rs, pbase = kv, pid * _PAGE_BYTES  # _PAGE_BYTES = 8448: 8192 key bytes, then 64 fp32 scales
if live is not None:  # a look-ahead past the slice: aimed past the descriptor's num_records, so dropped
    pbase = live.select(pbase, fx.Int32(kv_bytes))
page_scale = fx.Float32(  # lane l loads the scale of token l of the page
    rocdl.raw_ptr_buffer_load(
        T.f32, kv_rs, pbase + (_KEY_BYTES + lane * 4), zero_i32, aux=cpol
    )
)
kfrags = []
for j in range_constexpr(4):  # token tile j: two 16-byte loads give the lane 32 dims of token 16 * j + r
    koff = pbase + (j * 2048 + g * 512 + r * 16)
    klo = fx.Vector(
        rocdl.raw_ptr_buffer_load(T.i32x4, kv_rs, koff, zero_i32, aux=cpol)
    )
    khi = fx.Vector(
        rocdl.raw_ptr_buffer_load(T.i32x4, kv_rs, koff + 256, zero_i32, aux=cpol)
    )
    kfrags.append(klo.shuffle(khi, [0, 1, 2, 3, 4, 5, 6, 7]))
```

**Head sum in registers, no LDS.** relu, × weight, accumulated over the four head groups with fma (the first group
peeled, so no `+ 0.0`), summed over the lane's four heads, then over the four lane groups with `permlane16_swap` and
`permlane32_swap`: a fixed tree, no atomics. A `sched_barrier(0)` between token tiles keeps one tile's results live at a
time. Lane group g keeps tile g, so lane l holds token l of the page; it multiplies by that token's scale, writes −inf
past the context, and the wave stores the page's 64 logits as one 256-byte non-temporal store (`../gemm/epilogue.md`
§5).

```python
tiles = []
for j in range_constexpr(4):  # one token tile at a time
    rocdl.sched_barrier(0)
    kfrag = kfrags[j]
    total = None  # the first head group is peeled: a zero seed leaves `x + 0.0` adds fp32 cannot fold
    for hg in range_constexpr(4):  # 16 heads per MFMA, A = q, B = keys, unit E8M0 scales
        acc = fx.Vector(
            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [qfrag[hg], kfrag, zero4, 0, 0, 0, unity, 0, unity],
            )
        )
        relu = fx.Vector(acc.maximumf(zero4))
        total = (
            relu * wfrag[hg] if total is None else _fma4(relu, wfrag[hg], total)
        )
    tiles.append(_lane_group_sum(total.reduce("add")))  # two permlane swaps: sum over the four lane groups
value = tiles[0]
for j in range_constexpr(1, 4):
    value = (g == j).select(tiles[j], value)  # lane group g keeps tile g: lane l holds token l
value = value * page_scale  # the scale after relu and the head sum
token = pos * _PAGE + lane
value = (token < context).select(value, fx.Float32(float("-inf")))
out_rs = _rsrc(  # the page's own descriptor: a row of logits may pass 4 GB
    out, (fx.Int64(b) * max_seq + fx.Int64(_uniform(pos)) * _PAGE) * 4, _PAGE * 4
)
rocdl.raw_ptr_buffer_store(  # 64 lanes × 4 B, non-temporal
    fx.as_ir_value(value),
    out_rs,
    fx.as_ir_value(lane * 4),
    fx.as_ir_value(zero_i32),
    aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), _CPOL_NT),
)
```

**A split over pages with no merge.** A logit depends on one page, so workgroups split each request's table into
disjoint page ranges and each wave takes every fourth page of its range; nothing is combined (`../attention/split_kv.md`
§5: an output that is not a reduction splits with zero merge). The host picks one of three bodies from the batch:

| Batch | Body | Grid | How a wave gets its next page |
|---|---|---|---|
| 1, 2, 4 | small-batch body | 512 workgroups: 512, 256 or 128 slices × batch | Slice bounds from the compile-time table width, clamped to the request's page count, so the table read does not wait for `context_lens`; one dword load per lane holds the slice's 9, 17 or 33 page ids in one VGPR, read per page with `v_readlane_b32`; the next page's nine loads are issued before the current page's MFMAs (two pages in flight), the look-ahead past the slice aimed past the descriptor's `num_records`, where it is dropped; keys and scales loaded `nt` |
| 8, 16, 32 | slice body | 256, 256 or 128 slices × batch: 2048, 4096, 4096 workgroups (batch 8's from the untuned slice count) | Slice bounds from the runtime page count; the next page id is loaded at the top of each iteration, pinned there with `sched_barrier(0)` and carried across the loop |
| 64, 128, 256 | sort prepass, then band body (tables of 1024 pages or more) | batch prepass workgroups, then 128 × batch | The prepass buckets the request's table by band = page id · 128 // n_pages (LDS histogram with LDS atomics, Hillis–Steele scan, scatter of packed (page id, position) words into an order buffer, band bounds into an offsets buffer); workgroup (request, band) walks its bucket, request index fastest, and its block index is relabelled so XCD c takes the bands ≡ c mod 8 |

The batch ≤ 4 body keeps two pages in flight through its whole loop at 2 workgroups per CU (512 on 256 CUs);
`../attention/schedule_sync.md` §10 found a prologue depth of 2 paying on some grids and losing on others, and a
steady-state depth ≥ 2 slower, on split-KV producers with few query rows per head, so the depth is a
per-kernel result. It drops out-of-range
loads through the descriptor bound of `../gemm/datapath.md` §8.

```python
# small-batch body: the slice's page ids in one VGPR, lane l holding position wg_base + l
wg_base = _cut(sp, max_blocks, split)  # sp * max_blocks // split: needs no context length
pids = fx.Int32(
    rocdl.raw_ptr_buffer_load(T.i32, bt_rs, (wg_base + lane) * 4, zero_i32)
)
hi = _cut(sp + 1, max_blocks, split)
stop = (hi < npages).select(hi, npages)  # npages from context_lens: only the loop bound waits for it
pos0 = wg_base + wave
kf0, ps0 = _issue_page(  # the wave's first page: the nine loads above, with `nt`
    ctx, _readlane(pids, pos0 - wg_base), pos0 < stop, _KEY_CPOL
)
for page, state in range(
    wg_base + wave, stop, _WAVES, init=list(kf0) + [ps0]
):
    pos = fx.Int32(page)
    cur = [fx.Vector(fx.as_ir_value(state[j])) for j in range(4)]
    cur_scale = fx.Float32(fx.as_ir_value(state[4]))
    ahead = pos + _WAVES
    nkf, nps = _issue_page(  # the next page's loads go out before this page's MFMAs
        ctx, _readlane(pids, ahead - wg_base), ahead < stop, _KEY_CPOL
    )
    # Pin the look-ahead burst above this page's compute: left alone the
    # scheduler sinks it below the tile barriers and the pipeline
    # collapses back to one page in flight.
    rocdl.sched_barrier(0)
    _page_compute(ctx, cur, cur_scale, pos)
    yield list(nkf) + [nps]
```

The band path puts co-resident
workgroups on a few bands of the pool and keeps a band on one XCD (`../attention/grid_dispatch.md` §2,
`../gemm/schedule_and_locality.md` §2). The prepass's LDS atomics fix only the order of entries inside a
bucket, which changes which wave computes a page, not any value.

```python
# sort prepass, per page of the request: count, then scatter, by band = page id * split // n_pages
_lds_atomic_add(hist.h, (pid * split) // n_pages, one_i32)
...
slot = _lds_atomic_add(hist.h, (pid * split) // n_pages, one_i32)
rocdl.raw_ptr_buffer_store(  # the packed (page id, position) word, into the request's bucket
    fx.as_ir_value(pid + ii * _PSZ),
    ord_rs,
    fx.as_ir_value(slot * 4),
    fx.as_ir_value(zero_i32),
)

# band body, grid (batch, split): the low three bits of block_idx.x pick the XCD; with xcd_s = 1 and
# xcd_a = 8 they move from the request axis to the band axis, so XCD c takes the bands that are c mod 8
_c = fx.block_idx.x % 8
_blk = fx.block_idx.x // 8
_spy = fx.block_idx.y
b = _blk * 8 + (_c % xcd_s) * xcd_a + _spy % xcd_a  # request
sp = (_spy // xcd_a) * xcd_a + _c // xcd_s  # band
```

**Compile and launch.** The build function is a module-level `functools.cache` on (batch, max_blocks, max_seq_len,
n_pages, page size, preshuffle); it allocates the order and offsets buffers once per shape and calls `flyc.compile`,
which runs the launch once, here with zeroed context lengths so no page is walked (`../flydsl/attention.md` §
measurement §6). The entry point calls the compiled object with raw addresses, the form that section recommends for hot
calls.

## Result

The yardstick at every size is aiter's kernel, called as in § Applies when. The speedup is a one-sided lower bound of
aiter / kernel time over CUDA-graph replays with a 1 GiB cache flush before each call, read at 16 rounds and, where a
threshold is still undecided, again at 32, each look at α = 0.025; a size counts as won from 1.03.

The runs of E1, E2 and E5 used FlyDSL 0.3.2, not the FlyDSL of the image they ran in
(`../pitfalls/measured_a_different_build.md`).

Per batch, the range over that batch's pool sizes, in one run:

| Batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| Body | small | small | small | slices | slices | slices | band | band | band |
| Sizes | 9 | 8 | 6 | 6 | 9 | 8 | 7 | 4 | 4 |
| Speedup *(field, E5)* | 1.046–1.110 | 1.206–1.301 | 1.327–1.390 | 1.296–1.311 | 1.203–1.257 | 1.241–1.269 | 1.434–1.674 | 1.572–1.727 | 1.796–1.837 |

61 of the 61 sizes read 1.03 or more, and all pass the accuracy check. The kernel this one replaced, which differs from
it in more than the per-page descriptors, read 1.041–1.884 over its 58 sizes in an earlier run *(field, E1)*. A
separate run of this kernel, earlier than the table's and on FlyDSL 0.3.4.1, its ruler not stated in the record, reads
1.030–1.809 over its 61 sizes, batch 1 at 1.030–1.134, the 262,144-page sizes at 1.030 (batch 1), 1.240 (16) and 1.409
(64) *(field, E4)*.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, lower bound of the aiter / kernel time ratio in an earlier run, its rounds and flush not in the record] the kernel this one replaced: speedup over aiter 1.041–1.884 per size, 58 of 58 ≥ 1.03; batch 1 1.041–1.101, 2 1.211–1.284, 4 1.272–1.372, 8 1.117–1.165, 16 1.212–1.264, 32 1.195–1.269, 64 1.643–1.714, 128 1.700–1.747, 256 1.794–1.884 (64 heads, page size 64, 58 sizes)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, a separate run; lower bound of the aiter / kernel time ratio, its rounds and flush not in the record] the kernel this one replaced: 1.024–1.891 per size, 57 of 58 ≥ 1.03; batch 1 by pool size 105428 1.055, 109404 1.035, 111670 1.024, 111761 1.073, 115603 1.035, 119349 1.030, 120631 1.037, 123163 1.031 (64 heads, page size 64)
- E3 [2026-10, gfx950, FlyDSL ?, N=?, a timer of CUDA-graph replays, not the ruler of E1] batch 1, 2 and 4 "1.7x, 2.1x and 2.2x faster than aiter" after the depth-2 pipeline (64 heads, page size 64)
- E4 [2026-11, gfx950, FlyDSL 0.3.4.1, N=1, a separate run, earlier than E5; ruler not stated in the record] this kernel: speedup over aiter 1.030–1.809 per size, 61 of 61 passed; batch 1 1.030–1.134, 2 1.244–1.302, 4 1.294–1.341, 8 1.278–1.307, 16 1.217–1.261, 32 1.241–1.271, 64 1.409–1.700, 128 1.570–1.731, 256 1.696–1.809
- E5 [2026-11, gfx950, FlyDSL 0.3.2 (aiter dbd8bf5's pin), N=1, one-sided lower bound of the aiter / kernel time ratio over CUDA-graph replays with a 1 GiB cache flush before each call, read at 16 rounds and, where a threshold is still undecided, at 32, α = 0.025 per look] this kernel: speedup over aiter 1.046–1.837 per size, 61 of 61 ≥ 1.03, accuracy passed at every size; batch 1 1.046–1.110, 2 1.206–1.301, 4 1.327–1.390, 8 1.296–1.311, 16 1.203–1.257, 32 1.241–1.269, 64 1.434–1.674, 128 1.572–1.727, 256 1.796–1.837
- E7 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=8 rounds in one process, per-kernel times of CUDA-graph replays summed per call, with a 1 GiB flush per call] this kernel (band path) against a copy forced to the slice path and against aiter, batch 64, 128, 256, each request owning its pages, contexts uniform in 8,192–49,152 tokens: band / slice speed 0.882–0.937 with the pages scattered over the pool, 0.194–0.514 with one contiguous run per request, there band at 0.266–0.471 of aiter's speed; band and slice outputs bitwise equal inside the context
