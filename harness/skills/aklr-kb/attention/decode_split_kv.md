---
title: "Attention decode — split-KV grids, the combine, merging them, and pricing gathers"
kind: topic
lever: attn_decode_split_kv
ops: [attention]
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [decode, split-kv, mla, sparse]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:1010
  - ROCm/FlyDSL@dd8373349:kernels/attention/mla_fwd_decode.py:63
---

# Split-KV decode

> A decode step has few query rows, so the grid comes from splitting the KV range: a producer writes one
> partial output and LSE per split, and a combine merges them. FlyDSL's forward does this with a second
> launch (`flash_attn_gfx950.py:1010-1016`); its MLA decode takes the partial buffers as arguments
> (`mla_fwd_decode.py:63-64`). The rules below are general; constants from one model's shapes are left out.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.
> Kernel deltas are time (negative = faster); decode-step benchmark deltas are score (positive = better).

## Route here when
- A decode attention call is a producer plus a combine, and the pair is a visible share of the step.
- You choose the number of splits, tiles per workgroup, or the combine's grid.
- You want to merge the combine into the producer.
- A gather-based (sparse or paged) decode kernel does not respond to compute-side changes.

**Somewhere else if:** timing kernels of a few µs → `measurement.md`; XCD order of the
combine → `grid_dispatch.md` §3; GEMM-side split-K → `../gemm/small_m.md` §4.

## 1. The combine is a fixed cost
- *(field, E1)* One graph node cost 1.5–1.8 µs. *(field, E2)* Priced by repeated launches, the combine's body
  was 0.13–0.53 µs, so ~88% of the combine was dispatch. *(field, E3)* It was 13–16% of the attention call at the
  larger row counts and 46% at the smallest, not a rounding error.
- An empty kernel's duration is not what removing a launch saves: the producer's tail was overlapped by the next
  launch. Price a merge by launching the combine one extra time in the stream and reading the increase of the total
  (`../gemm/attribution.md` §3).
- Run the combine with one wave per workgroup (`../gemm/grid_pricing.md` §2) and number its
  workgroups like the producer's ownership (`grid_dispatch.md` §3), unless each output merges many
  partials. *(field, E42)* At 16–32 partials per (row, head), one 64-lane workgroup per (row, head) made the whole call
  10–32% slower than 512-lane workgroups whose waves split the partials; 1.5% slower at 11, level at 4.
- Too few combine workgroups: split the value dimension into disjoint column slices (bit-identical).
  *(field, E4)* Gains fell monotonically with the unsliced CTA count (−9.7% at 224, −3.7% at 512, −0.6% at
  1536) and never changed sign. Each slice recomputes the split weights from the LSE column, so more
  slices lose where that recomputation is scalar.

## 2. Merging producer and combine: a persistent, cooperative rendezvous
The merged form that paid has every producer workgroup of a row wait at a rendezvous and then share the
reduce. A workgroup that waits for another is right only while every workgroup it waits for is resident (Opening
window below), which a kernel cannot promise: a deployment may run it on a stream limited to part of the CUs or
beside another kernel. In a kernel for AKLR no workgroup waits on another: keep the separate combine (§1), or let
the last workgroup to arrive on an atomic counter do the merge (§3 prices that here). The steps below describe
the rendezvous as existing kernels write it.
1. Write the partials and LSE as write-through stores that are visible across XCDs (`SC1` = aux value 16,
   device scope, is enough across XCDs; values 16 and 17 were both bitwise clean *(field, E5)*. `0x11` adds `SC0`,
   i.e. system scope. DLC (value 4) and SC0 give no cross-XCD visibility: with value 5 (SC0|DLC) the reader saw
   the same stale elements as with no bits). The write-through is the release; add no fence.
2. Every wave waits `vmcnt(0)`, then a workgroup barrier (`s_waitcnt` covers only its own wave).
3. One thread per workgroup arrives on a per-row sense-reversing counter: the master adds
   `SENSE − (n − 1)` with `SENSE = −(1 << 31)`, the others add 1, so each launch adds exactly `SENSE` and
   graph replays need no reset. Give each counter its own 128 B line.
4. After the rendezvous, all `waves × n_groups` waves of the row share the reduce in round-robin.

**Opening window, both required.**
- Residency: `rows × n_groups ≤ num_cu`. The rendezvous ends only if all splits of a row are resident
  at once; at most one workgroup per CU guarantees it only while the kernel has every CU to itself. A stream
  limited to part of the CUs, or another kernel holding CUs, breaks it: *(field, E45)* a split-K GEMM fold that
  polled the same way returned millions of NaN on streams limited to 64 and 128 CUs. Bound the poll loop, so that a
  broken assumption produces a wrong answer instead of a hung device.
- Column length: `2 ≤ n_groups ≤ 16` on one shared counter. *(field, E6)* 32 workgroups on one counter: +3.25%.
  *(field, E43)* With one launch-tagged flag per workgroup polled by one
  wave, 32 per row: −3.6% against producer + combine; 16: −2.0%; 21–22: +0.7%. The residency rule is unchanged.

*(field, E7)* Bit-identical outputs; −1.0% to −7.6% against two kernels; +2.47% on the decode-step benchmark.

The poll: keep a nonzero sleep in the loop (*(field, E8)* zero sleep widened the run-to-run spread to 10.2%),
and make the polled load atomic or volatile; a plain load in a store-free loop is loop-invariant to LLVM
and gets hoisted. In the merged epilogue only the number of rounds and the store width per round matter.
*(field, E9)* Cutting the LSE reads per row by 37.5% changed nothing: they issue beside the longer partial loads.

## 3. Why a "last workgroup reduces" ticket fails here
- **Parallelism collapses.** Only the last arrival knows all splits are in, so the reduce of a row runs in
  one workgroup. With rows ≪ CUs that drops one to two orders of magnitude of parallelism. Its floor is
  (reduce work ÷ the largest tile one wave holds) × one memory round trip. *(field, E10)* On a Triton split-K
  decode: ~1 µs per loop iteration, ~9 µs of tail against the ~3 µs launch it removed, +79% to +510% even with a
  correct release.
- **The producer pays for the reduce.** The reduce's live registers sit on the whole producer, and the
  tail that the next launch used to overlap is now exposed. *(field, E11)* Without a fence: kernel +7% and +144%,
  decode-step benchmark −2.5%.
- **No agent-scope fence in a kernel that lives on L2 reuse.** A release/acquire fence writes back and
  invalidates the XCD's L2. *(field, E12)* Decode-step benchmark −62%; *(field, E13)* release 5.1 µs + acquire
  1.3 µs, 3–4× the whole epilogue. Get cross-CU visibility from per-instruction scope bits.
- In a backward pass, where thousands of output tiles reduce in parallel, the same pattern also lost to a separate
  reduce kernel (`backward.md` §6).

## 4. Grid size: fill the CUs once, with fat workgroups
- **Ragged groups.** Allow any `n_groups` with `ceil(n_tiles / n_groups)` tiles per workgroup; in the last
  round, point the short groups' index rows at −1 so the existing padding mask makes them −inf. P = 0 and
  the rescale factor is 1, so accumulators, running max and LSE are bit-identical. Cost: one compare and
  two selects.
- **Rows of unequal length, known only on the device: split the batch's summed steps evenly over `num_cu`
  persistent CTAs**, one partial per row segment, merged in CTA order; each CTA sums the rows' block counts itself
  and takes the equal share `[i·B/NCTA, (i+1)·B/NCTA)`. On dense MLA decodes: *(field, E38)* one chunk size for all
  rows took 35% off the time of per-row chunk sizes, and the exact split another 3%; *(field, E40)* the exact split
  took 24% off per-row chunks in a second kernel; *(field, E39)* in a third it was 6–9% faster than per-row splits
  dispatched longest row first, and 2 CTAs per CU was slower on every shape.
- **One workgroup per CU is a cliff.** *(field, E14)* Configurations at ≤ 1.0 CTA/CU won 1–16% (unless tiles
  per CTA grew to 16); every configuration just above 1.0 (1.09–1.31) lost 2.5–22%: some CU then runs two
  fat workgroups and that straggler sets the wall. Take the largest `n_groups` with
  `rows × n_groups ≤ num_cu`; on a tie take fewer groups (less padding). This picked the measured best on
  all nine sequence lengths tested.
- Cap tiles per workgroup and fall back to more groups beyond it. *(field, E14)* 16 tiles per CTA lost 2–20%
  where 11 won 8–11%.
- **The residency limit is a second cliff.** *(field, E15)* Two resident CTAs per CU: 512 → 528 CTAs was +3% work
  and +24% time.
- **The wall is the busiest workgroup's real tiles plus its footprint**, not slots per workgroup. Padding
  tiles move no data (the descriptor bound zeroes out-of-range rows) and run on non-critical workgroups.
  *(field, E16)* The first workgroup to carry an extra real tile cost 0.49 µs, each further workgroup carrying
  one 0.18 µs; *(field, E17, E18)* making
  dead tiles cheaper (smaller redirect window, zero-traffic out-of-range rows) lost 1–7%.
- **Attribute grid shape only at fixed total work.** *(field, E19)* A sequence-length sweep priced half-filled
  rounds at +11.6%; *(field, E20)* at fixed total work, 1.5 → 3.0 CTA/CU bought 0.4% while the combine got
  32% slower.
  The objective is producer plus combine.

## 5. Split rules
- **At least two tiles per split**, or the pipeline has nothing to overlap. *(field, E21)* On a Triton split-K
  decode, halving the tile and doubling the splits at 2 tiles per split: −11%; the same CTA count with 1 tile per
  split: +1.8%.
- **Tiles per split jump at divisibility boundaries.** `ceil(n / (splits × tile))` steps by a whole round.
  *(field, E22)* On a Triton split-K decode, widening the key set by 5 columns past a boundary cost 12–20%; by 64
  columns (which divided again) 5–12%.
- **Pin every other launch parameter when sweeping splits.** *(field, E23)* On a Triton split-K decode, a launcher
  heuristic dropped `num_warps` when the split count rose: +21.8%; with `num_warps` pinned the same split read
  −0.4%.
- **An output that is not a reduction splits with zero merge.** Index-score logits `[rows, Skv]` split
  along KV write disjoint column ranges, bit-identical. *(field, E24)* In a Gluon kernel, 1 → 32 splits: 820 → 73
  µs; with 1–2 waves per workgroup the optimum was about 4 workgroups per CU.
- bf16 partials add merge error; gate every split count on accuracy (`../gemm/small_m.md` §4).
- Changing the reduction order changes bits: compare against a reference whose order is shared by neither
  arm (`correctness.md`).

## 6. Price the producer
- Fit `t ≈ fixed + bytes / bandwidth`. *(field, E25)* `t ≈ (1.7–2.4 µs) + bytes / (6.0–6.2 TB/s)` over five
  shapes; the marginal bandwidth equalled the measured stream floor, so the body had no bandwidth left to buy.
  The fixed part is the dispatch plus the dependent first hop (index → KV address).
- Confirm the fit on a clean axis: fix the group count, CTA count and tiles per CTA, and sweep only the
  number of real tiles. *(field, E26)* Within ±5% of the per-tile fit `2.83 µs + 6.81 ns × tile`.
- A plain gather with the same row width and index pattern shows what the pattern allows (in Triton, random rows
  4.5 lines wide reached only ~45% of a stream copy *(field, E27)*), but it is not a ceiling: a tuned decode
  producer ran 5.9–9.5% faster than the plain gather built to bound it *(field, E28)*.
  - Use the marginal-bandwidth fit above and never HBM peak (`measurement.md` §5).
- **Price the partial write by skipping it.** *(field, E44)* On an fp8 MLA decode, skipping it took 4.5% off the
  call. Staging the bf16 partials in LDS and storing them 16 B per lane, contiguous across lanes, in place of 8 B per
  lane into head rows 1 KB apart, took 0.8–3.4% off, every shape faster (§7's rule for gathers, applied to stores).
- Before compute-side work, run the compute-additivity probe on the producer (`attribution.md` §9).
  *(field, E29)* Four times the P·V MFMAs cost +7%. *(field, E30)* Waves spent
  ~54% of their cycles in `SQ_WAIT_ANY` and ~20% issuing; *(field, E31)* deeper steady-state gather prefetch
  lost 4.6–18%: more random 128 B requests in flight only queued.

## 7. Gathers: unique lines and the index distribution
- **The currency is unique 128 B lines per row (L2 misses).** *(field, E32)* On a sparse gather prefill under
  uniform indices: all rows L2-resident −52.9%, all
  rows MALL-resident only −10.7%, unique lines per row 5 → 3 −27.3%; cutting TCP requests per row 7 → 5
  with unique lines unchanged: +0.9%.
- **Make each gather instruction contiguous across lanes** (lane j at byte 16j of the span), not one contiguous chunk per
  lane: same bytes and lines. *(field, E41)* With two waves issuing every gather: 6–7% less time; with six: flat.
- **A row whose width is not a multiple of 128 B touches an extra line**: lines per row =
  ⌈(phase + width) / 128⌉, so a 4.5-line row starting at phase 0 or 64 always touches 5. Only a layout change
  reduces that, and it must be measured.
  - *(field, E33)* In a gather-latency-bound Triton sparse-MLA prefill kernel whose production indices
    already overlapped heavily, padding rows to whole lines and splitting them into two line-aligned pools were
    all 2–5% slower.
- **Benchmark with production index structure.** *(field, E34)* Adjacent tokens of one request overlapped
  0.69–0.92, different requests 0. With the same 9.4 MB footprint: 12.35 µs for an unstructured
  distribution, 15.44 µs for the production structure. *(field, E35)* An ownership remap that put a
  request's tokens on one XCD measured up to 9% slower (producer alone up to 11%) on uniform random indices
  and 3–9% faster (producer alone 7–15%) on production indices. *(field, E33)* On a Triton sparse-MLA prefill,
  locality headroom of 26–29% seen on uniform sampling did not exist at production overlap. Re-measure ownership
  levers after any layout or grouping change.
- **Check a carried-over patch's enabling condition on today's grid.** *(field, E36)* A remap gated on
  `(rows × n_groups) % 8 == 0` returned early on three of four shapes once groups became ragged. The
  remainder-spreading form (`../gemm/schedule_and_locality.md` §2) needs no gate.

## 8. Measuring
Time decode kernels as graph replays with events, arms interleaved in one process. *(field, E37)* A
synchronize-bracketed wall carried ~27 µs of host time and turned a 16.8% kernel gap into 38%. Protocol:
`measurement.md`.

## Verify
- Combine numbering matches producer ownership; the combine has one wave per workgroup unless each output merges
  many partials (§1).
- No workgroup waits on another (no rendezvous, no polled flags): the merge is a separate combine launch or the
  last workgroup to arrive on an atomic counter (§1, §3).
- The grid has ≤ 1 CTA per CU and ≥ 2 tiles per split on every scored shape.
- The benchmark's indices reproduce production adjacency.

## Related
`grid_dispatch.md` · `backward.md` §6 · `measurement.md` ·
`../gemm/small_m.md` · `../gemm/epilogue.md` §5 · `../hardware/gfx950/memory_and_sync.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. `Triton`, `Gluon` or `HIP` in place of FlyDSL: the line measured a kernel of the same kind written in that
language, or bare graph nodes. "Decode-step score" is that campaign's ruler: the geomean over c8/c10/c14 of `BASE_US` /
(one decode step of absorb BMM + sparse-MLA producer and combine). The FlyDSL decode producer and combine are aiter's
`sparse_mla_decode` / `mla_decode_combine` (GLM-5.2 TP4/EP4, H=16, topk 2048, fp8 KV in 576 B rows; MI355X, 256 CUs);
"rN" is the campaign's round.

- E1 [2026-09, gfx950, HIP (empty kernels and a 1-element torch.add as graph nodes), N=?, HIP graph replay; the r12 floor with 20 replays, min of 5] cost of one graph node: empty-node floor 1.746/1.728/1.666 µs (r14), 1-element torch.add 1.8477 µs (r12), empty-kernel slope 1.4987–1.5709 µs per node (r16) (campaign 20260912_134949 r12, r14, r16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:320; :300; :344)
- E2 [2026-09, gfx950, FlyDSL 0.3.2, N=?, repeated-launch pricing: the combine launched 1, 2 and 3 times in one HIP graph, spread 0.03–0.14%] combine marginal cost 1.88/1.95/2.19 µs at seq 48/60/84, of which the body 0.134/0.218/0.526 µs: 88% dispatch (combine `mla_decode_combine`; r14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:320)
- E3 [2026-09, gfx950, FlyDSL 0.3.2, N=?, producer and combine captured in separate HIP graphs, ABBA, spread ≤1.7%] producer + combine: seq48 15.043 + 2.776 µs, seq60 16.821 + 2.997, seq84 22.127 + 3.414, seq8 4.484 + 3.802; the combine is 13.4–15.6% of a verify call (seq 48/60/84) and 46% of a draft call (seq8) (r1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:128)
- E4 [2026-09, gfx950, FlyDSL 0.3.2, N=14 at 1344 and 1536 CTAs (two independent 14-rep blocks), not stated for the other seven shapes, same-block ABBA with the combine timed behind the producer] combine Dv slices against unsliced, by the unsliced CTA count: 224 −9.72%, 512 −3.74%, 768 −1.19%, 1344 −0.74%, 1536 −0.58% (nine shapes, all negative); repeats at seq84/96 −0.74/−0.65% and −0.58/−0.39% (r8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/rounds/round-08/agent.txt:39)
- E5 [2026-09, gfx950, FlyDSL 0.3.2, N=?, `torch.equal` of partial, LSE and output against the two-kernel path] cache-policy value of the cooperative merge's partial/LSE stores: 16 (SC1) and 17 bitwise clean at the same cost; 0 and 5 (SC0|DLC) broke the same 1971 elements (seq60) (r14 H7/H8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:320)
- E6 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-block ABBA] cooperative merge at seq8 with `n_groups=32` (32 workgroups per counter): +3.25% against two kernels (r14 H3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:320)
- E7 [2026-09, gfx950, FlyDSL 0.3.2, N=4 for the score, ? for the kernel deltas, same-block ABBA for the kernels; decode-step score, median of 4 runs] cooperative merge against producer + combine: verify seq 48/60/84 −4.95/−5.46/−1.02% with outputs `torch.equal`, draft seq10 −6.06%, seq14 −7.60%; score 1.7190 → 1.76449/1.76010/1.76082/1.76205, median 1.7614 (+2.47%) (r14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:320)
- E8 [2026-09, gfx950, FlyDSL 0.3.2, N=?, ?] poll backoff `_BARRIER_SLEEP` 0/1/2/4: epilogue 1.537/1.513–1.583/1.524/1.486 µs, within noise; sleep 0 widened the fused arm's spread to 10.2% (r14 H5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:320)
- E9 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-block ABBA with a lazy control arm] DV-major epilogue slots cut the non-cached LSE dword reads per token 96 → 60 (−37.5%, a count): seq84 ship 22.1313 µs, DV-major 22.2499 (+0.536%), control 22.2428 (+0.504%), delta − control +0.03%, bitwise identical (r15; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/rounds/round-15/agent.txt:5)
- E10 [2026-09, gfx950, Triton (split-K decode in sglang), N=?, isolated probes against the shipping split + reduce pair] "last CTA reduces" ticket fold with `.wt` stores and a relaxed gpu-scope ticket (no fence in the ISA), bitwise correct over 12 runs: +78.7% to +510%; tail ~9 µs ≈ 8 loop iterations × ~1 µs (8.08 µs even with 8 pipeline stages), against a ~3 µs launch (GLM-5.2 TP8/EP1 sparse-MLA decode, `_sparse_mla_split_k_kernel` + `_sparse_mla_reduce_kernel`; sglang campaign 20260925_134931 r4; local:/workspace/code/gpt_oss_docker/sync/sglang_optimizer/sglang_campaigns/20260925_134931/rounds/round-04/agent.txt:3)
- E11 [2026-09, gfx950, FlyDSL 0.3.2, N=1, decode-step score, one run of the arm; kernel times from that run] ticket fold of the combine into the producer epilogue without a fence (only where split-major and seq % 8 == 0): score 1.54194 against the separate combine's mean 1.58145 (−2.5%); `mla_v8` 19.03 against 17.79 µs (+7%), `mla_d8` 16.40 against 6.72 µs (+144%) (r5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:180)
- E12 [2026-09, gfx950, FlyDSL 0.3.2, N=1, decode-step score, one run of the arm] the same fold with agent-scope release/acquire fences: score 0.60625 against 1.58145 (−61.7%); `mla_v8` 52.40 µs, `mla_d8` 24.41 µs (r5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:179)
- E13 [2026-09, gfx950, FlyDSL 0.3.2, N=?, ?] agent-scope fences in the cooperative epilogue: release 5.1 µs + acquire 1.3 µs, epilogue 7.97 µs, against a fence-free epilogue of 1.29–2.06 µs ("3–4×" is the round's own figure) (r14 H1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:320; :323)
- E14 [2026-09, gfx950, FlyDSL 0.3.2, N=5, same-block ABBA, min of 5–6, a ship copy at the tail measuring position bias 0.03–1.6%] ragged `n_groups` sweep at ng=32, 256 CUs, against ship: ≤1.0 CTA/CU −1.1% (seq24 n=10) to −16.2% (seq11), except 16 tiles per CTA (seq84 n=2 +20.2%, seq96 n=2 +1.9%); 1.09–1.31 CTA/CU +2.5% to +21.6%; 11 tiles per CTA at seq72/84 −11.1%/−8.1%; "largest n_groups with seq × n ≤ num_cu, fewer on a tie" picked the measured best on 9 seqs (11/20/24/32/40/48/60/72/84) (r9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/rounds/round-09/agent.txt:23; full table in myskill pitfalls/12-dsv4-sparse-mla-attn.md:728-746)
- E15 [2026-09, gfx950, FlyDSL 0.3.2, N=?, fixed ii=4, seq swept, time per tile] 448/480/496/512 CTAs 8.891/8.754/8.717/8.527 ns per tile, 528/544/576 CTAs 10.260/9.925/9.397: 512 → 528 CTAs, +3% work, +24% time, with 2 CTAs per CU resident (r6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:205)
- E16 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-block ABBA with a lazy control arm (+0.615%)] seq48 at fixed `n_groups=5` (240 CTAs, 7 slots), real tiles 30/31/32/33: 14.58/15.07/15.28/15.43 µs; the critical CTA's first extra real tile 0.49 µs, each further CTA 0.18 µs (r15 H3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:332)
- E17 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-block ABBA, control arm +0.01…+0.17%, outputs bitwise identical] padding-tile rows redirected to a 1/32/256/4096-row window: seq48 +3.23/+3.96/+3.78/+4.61%, seq42 +4.04/+4.27/+3.77/+5.20%, seq84 +1.06/+1.14/+1.39/+1.92% (r10; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:257)
- E18 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-block ABBA, control ±0.46%, outputs `torch.equal`] padding-tile gathers sent out of bounds (no traffic), producer: per-lane `v_cndmask` +6.69% (seq48), +2.44% (seq84), +1.50% (seq60, no padding); scalar `soffset` +3.84%, +1.31%, +0.11% (r12 H1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:296)
- E19 [2026-09, gfx950, FlyDSL 0.3.2, N=?, ii=4, seq swept 16–96 (128–768 CTAs), line through the 1.0/2.0/3.0 CTA/CU points] half-filled last rounds above the `2.83 µs + 6.81 ns × tile` line: seq48 +11.6%, seq60 +5.3%, seq84 +4.5% (r1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:128)
- E20 [2026-09, gfx950, FlyDSL 0.3.2, N=4, same-process ABBA, min of a 4-step palindrome, 60 s ramp, producer spread 0.11–0.92%] seq48 with the tile total fixed, ii 8/4/2 = 0.75/1.50/3.00 CTA/CU: producer 15.780/15.137/15.078 µs (1.5 → 3.0: −0.4%), combine 2.096/2.480/3.283 µs (+32%) (r2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/rounds/round-02/agent.txt:20)
- E21 [2026-09, gfx950, Triton (split-K decode in sglang), N=?, the real launcher, split + reduce pair] seq=6 (`BLOCK_K`, `kv_splits`) (64,16) → (32,32), 2 tiles per split: −10.97%; (64,32), the same 192 CTAs at 1 tile per split: +1.8% (GLM-5.2 TP8/EP1 c1 verify, H=8; sglang campaign 20260925_134931 r3; local:/workspace/code/gpt_oss_docker/sync/sglang_optimizer/sglang_campaigns/20260925_134931/rounds/round-03/agent.txt:24)
- E22 [2026-09, gfx950, Triton (`triton_sparse_mla_decode_splitk`), N=?, CUDA graph replay, arms rotated, min of N, A-vs-A floor 0.3%] page-table width 2048 → 2053 (+5 columns): ×1.128/×1.123 fp8 and ×1.201/×1.196 bf16 (bs 1/2); → 2112 (+64): ×1.049/×1.048 and ×1.124/×1.122 (H=8, d_v 512, 400k pages; sglang campaign 20260922_154212 r16; local:/workspace/code/gpt_oss_docker/sync/sglang_optimizer/sglang_campaigns/20260922_154212/memory.md:321)
- E23 [2026-09, gfx950, Triton (split-K decode in sglang), N=?, isolated graph race] seq=12, `kv_splits` 16 → 32: 11.7527 → 14.3177 µs (+21.8%); the −0.41% with `num_warps` forced to 4 and the attribution to the launcher dropping `num_warps` are recorded only in myskill pitfalls/12-dsv4-sparse-mla-attn.md:1906-1907 (sglang campaign 20260925_134931 r3; local:/workspace/code/gpt_oss_docker/sync/sglang_optimizer/sglang_campaigns/20260925_134931/rounds/round-03/deep_raw.jsonl:11357)
- E24 [2026-09, gfx950, Gluon (`_gluon_fp8_mqa_logits` in aiter), N=?, ?] DSA indexer logits, rows=33, ctx 156001, KV splits 1/8/16/32/64/192: 820.4/138.6/91.7/73.1/78.8/81.6 µs, best at 1056 workgroups ≈ 4 per CU (GLM-5.2 DSA indexer; auto-memory note, no run record kept; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsa_indexer_kv_split.md:39)
- E25 [2026-09, gfx950, FlyDSL 0.3.2, N=?, ?] producer time against billed bytes (KV rows at 5 × 128 B, Q and partial write-back) over 5 shapes (seq 18/42/48/60/84): marginal bandwidth 6.0–6.2 TB/s, the measured stream floor; intercept 1.7–2.4 µs (r10; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:258; table in myskill pitfalls/12-dsv4-sparse-mla-attn.md:833-842)
- E26 [2026-09, gfx950, FlyDSL 0.3.2, N=?, ?] fixed `n_groups`, CTA count and ii, real tiles 8/16/24/32: seq48 5.554/8.466/11.247/13.776 µs → 2.7 µs + 7.15 ns/tile, seq84 7.171/12.935/16.303/20.202 → 2.8 µs + 6.46 ns/tile, within ±5% of `2.83 µs + 6.81 ns × tile` (r12 H3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:299)
- E27 [2026-09, gfx950, Triton (a plain gather), N=?, ?] random 576 B rows from a 576 MB pool: ~45% of a streaming read; 576 B rows 17.55 µs, 512 B rows 11.80 µs (seq 48 × topk 2048; auto-memory note dated 2026-09-11, no run record; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_sparse_mla_flydsl_vs_triton.md:40)
- E28 [2026-09, gfx950, FlyDSL ?, N=?, ?] the FlyDSL decode producer 5.9–9.5% faster than a plain Triton gather on the same pool with the same instrument (GLM-5.2 sparse-MLA decode; a pre-campaign finding stated in campaign 20260911_130431's goal, no run record; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_130431/memory.md:61)
- E29 [2026-09, gfx950, FlyDSL 0.3.2, N=?, in-process ABBA additive probe] P·V MFMAs ×4 (16 → 64 `mfma32` per wave per tile, no extra memory traffic): producer +7.03/+7.75/+6.79% (three decode shapes, not named in the line; the round's barrier probe used seq 48/72/96) (campaign 20260911_143318 r10; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_143318/rounds/round-10/agent.txt:22)
- E30 [2026-09, gfx950, FlyDSL 0.3.2, N=?, rocprofv3 --pmc SQ_* counters, ~13 s per profile] wave cycles: `SQ_WAIT_ANY` 54.5% (seq48/60), 53.6% (seq84), 70.0% (seq8); ACTIVE_INST 15.6–20.2% (r2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/rounds/round-02/agent.txt:19)
- E31 [2026-09, gfx950, FlyDSL 0.3.2, N=?, ?] steady-state gather prefetch depth 1 → 2: seq48/60/84 +4.56/+6.01/+6.55%; depth 3 +11.98/+12.58/+18.05%; outputs bitwise identical (r1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:129; readings in rounds/round-01/deep_raw.jsonl:7027)
- E32 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-session probe, 0.39% spread; alias arms with an identical instruction stream] FlyDSL sparse-MLA prefill (T=4096, topk 2048, uniform random rows of a 576 MB pool), against 739.7 µs: all rows L2-resident (1.15 MB) 348.1 µs (−52.9%), MALL-resident (37 MB) 660.8 (−10.7%), unique lines per row 5 → 3 537.4 (−27.3%); row-aligned gather with TCP requests 7 → 5 per row 748.6 against 741.9 (+0.9%) (campaign 20260911_143318 r9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_143318/rounds/round-09/agent.txt:41)
- E33 [2026-09, gfx950, Triton (`_sparse_mla_fused_kernel` prefill in sglang), N=?, same-process ABBA, min of N, bitwise-equivalence check, A-vs-A 0.9988] production shape seq 32768, ctx 104000, topk 2048, H=16, ratio > 1 slower: rows padded to 640 B 1.0491, 512 + 64 B pools 1.0233, 512 + 128 B pools 1.0458; keeping 1/4 or 1/2 of the KV bytes 0.74/0.88; all rows L2-resident at production overlap 0.815 1.005–1.019; the −25.7%/−29% locality gains recorded earlier held only under uniform sampling (overlap 0.024) (GLM-5.2 TP4/EP4 prefill; sglang campaign 20260923_080725 r5; local:/workspace/code/gpt_oss_docker/sync/sglang_optimizer/sglang_campaigns/20260923_080725/memory.md:99)
- E34 [2026-09, gfx950, FlyDSL 0.3.2, N=?, live-server index capture (read-only overlay, CUDA graph off) and a same-process ABBA producer probe, min, spread ≤3.5%] production top-2048 sets: adjacent-token overlap 0.69–0.71 entering decode, 0.80–0.92 at steady state, cross-request overlap 0.000; producer seq48 at the same 9.4 MB footprint 12.35 µs (unstructured) against 15.44 µs (production structure) (GLM-5.2-MXFP4 TP4/EP4, EAGLE 6 draft tokens, 8 requests; r3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:141; overlap table in myskill pitfalls/12-dsv4-sparse-mla-attn.md:308-312)
- E35 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-block ABBA with a lazy control arm (−0.46…+0.58%), outputs `torch.equal`] request-to-XCD ownership remap on the ragged layout, full call / producer: uniform random +8.46/+9.65% (seq48), +5.13/+5.46% (seq84), +9.18/+10.65% (seq42), +1.76/−0.97% (seq18); production structure −5.60/−12.17%, −8.93/−13.71%, −6.97/−15.05%, −2.81/−6.96% (r11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/rounds/round-11/agent.txt:9)
- E36 [2026-09, gfx950, FlyDSL 0.3.2, N=?, host-gate check on the r9 grids] the remap's gate `(seq * n_groups) % 8` returned early on 3 of the 4 directive shapes (grids 240/240/252/198 CTAs) (r11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:282; myskill pitfalls/12-dsv4-sparse-mla-attn.md:928-932)
- E37 [2026-09, gfx950, FlyDSL ? (aiter FlyDSL decode) against Triton (sglang), N=?, rocprofv3 kernel trace against a `synchronize()`-bracketed perf_counter wall] decode seq=48: FlyDSL 21.54 µs (partial 17.94 + combine 3.60) against Triton 25.88 µs (split 20.64 + reduce 5.24), a 16.8% kernel gap; the wall read 48.7 µs and a 38% gap, ~27 µs of constant host time on both (GLM-5.2 TP4, H=16, topk 2048; auto-memory note dated 2026-09-11, no run record; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_sparse_mla_flydsl_vs_triton.md:18)
- E38 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop ablation ladder: each round against the version before it, the paired AKLR graph-timer timing it was kept on] a `(rows, ⌈2048/rows⌉)` grid with chunks sized per row → one chunk size for all rows, computed on the device from the summed `cache_seqlens`, over 256·m CTAs: gain +52.73% (time without it over time with it; one-sided 95% bounds +52.68…+52.78%), 20.5–45.6% less time per shape, 34.5% on the mean; that chunked grid → the exact even split of all rows' steps over 256 persistent CTAs (one per CU): gain +3.32% (+3.29…+3.36%), 0.3–6.9% less time per shape, 3.2% on the mean (dense MLA decode over paged fp8 KV, 96 heads, D 576/512, batch 8–112, cache_seqlens random 15–110k keys; local:/shared_nfs/kyle/aklr_mla/runs/L1/workspace/forge_experiments/ablation.md:13; local:/shared_nfs/kyle/aklr_mla/runs/L1/workspace/forge_experiments/ablation.md:18)
- E39 [2026-10, gfx950, FlyDSL 0.3.2, N=3–5, paired in-process Kineto A/B, median of per-round ratios of 10 calls; outputs passed the contract on 11 batch sizes] equal-block stream-K over 256 CTAs against per-row splits with in-kernel longest-row-first ranking: 0.907 / 0.928 / 0.938 at batch 8 / 32 / 112; 512 CTAs against the same base 1.022 / 0.942 / 0.960 (dense MLA decode, 96 heads, one query, fp8 KV in 576 B rows, page size 1, `cache_seqlens` uniform in [1, 112431]; MI355X, 256 CUs; local:/shared_nfs/kyle/aklr_mla/runs/L4/workspace/forge_experiments/lessons/iter_006.md:14; :15)
- E40 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm for the per-batch times (driver --bench-mode, warmup 3, iters 20, cache cleared); the gain from the campaign's 3-pair ablation] persistent partial kernel (256 CTAs, one per CU, each owning an equal contiguous range of the 32-key steps of all rows; partial slot `cta + row`; combine merges in CTA order) against a (row, split) grid of per-row chunks sized from each row's length: b8 0.441 → 0.289 ms, b112 3.768 → 3.147 ms, total 19.56 → 15.08 ms; ablation +32.31% when kept, +32.87% measured by the ablation (one-sided 95% bounds +32.80…+32.93%), 11/11 cases faster (96-head MLA decode, query length 1, fp8 KV, page size 1, batch 8–112, per-row lengths uniform in [1, 112431]; local:/shared_nfs/kyle/aklr_mla/runs/L2/workspace/forge_experiments/lessons/iter_002.md:22; local:/shared_nfs/kyle/aklr_mla/runs/L2/workspace/forge_experiments/ablation.md:14)
- E41 [2026-10, gfx950, FlyDSL 0.3.2, N=2–3 per arm, CUDA-graph replay of 10 calls, median of 15 replays, arms alternated; outputs bitwise identical] 576 B fp8 KV rows gathered by 12 lanes as 3 × 16 B loads: per-lane 48 B chunks (each instruction spans the whole row) → load c reading bytes 192c + 16j (each instruction 192 contiguous bytes): kernel with two dedicated gather waves b64 0.819 → 0.762, b80 1.000 → 0.938, b96 1.114 → 1.042, b112 1.488–1.500 → 1.392–1.401 ms; kernel whose six waves all gather, b8–b56: flat (96-head MLA decode, fp8 KV, page size 1; local:/shared_nfs/kyle/aklr_mla/runs/L2/workspace/forge_experiments/lessons/iter_011.md:28)
- E42 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, AKLR graph timer via `driver.py --bench-mode`, warmup 3, 20 iterations] combine with one 64-lane workgroup per (row, head) against 512-lane workgroups whose waves take every 8th partial (256-lane at more rows), whole call: b8 0.0742 against 0.0561 ms, b12 0.0771 / 0.0680, b16 0.0921 / 0.0838, b24 0.1129 / 0.1112, b60 0.2544 / 0.2553; 256 partials over the batch, so ~32 / 21 / 16 / 11 / 4 per row (96-head MLA decode, query length 1, fp8 KV, page size 1; local:/shared_nfs/kyle/moe/k3six/runs/mla_fp8/workspace/forge_experiments/lessons/iter_005.md:28)
- E43 [2026-10, gfx950, FlyDSL 0.3.2, N=3 pairs, forge ablation on AKLR's graph timer, noise floor 0.06%] producer with the merge fused (sc1 partial stores, one flag per workgroup holding `dispatch.id` xor `queue.ptr`, one wave polling the row's flags by ballot with `s_sleep`, an unbounded loop, no fences; one CTA per CU over 256 CUs) against producer + combine: +1.49% (time without over time with; one-sided 95% bounds +1.46…+1.53%), 8 of 9 cases faster; b8 (32 per row) +3.70%, b16 (16) +2.03%, b12 (21–22) −0.67% (same kernel family as E42; local:/shared_nfs/kyle/moe/k3six/runs/mla_fp8/workspace/forge_experiments/ablation.md:18; per-case gains in local:/shared_nfs/kyle/moe/k3six/runs/mla_fp8/workspace/forge_experiments/ablation.json; poll loop local:/shared_nfs/kyle/moe/k3six/runs/mla_fp8/workspace/kernel.py:1134)
- E44 [2026-10, gfx950, FlyDSL 0.3.2, N=1 for the skip (driver bench, warmup 3, 20 iterations), 3 pairs for the staging (forge ablation on AKLR's graph timer, noise floor 0.06%)] partial stores skipped (outputs wrong): 1.2841 against 1.3442 ms summed over 9 batches; bf16 partials staged in LDS (16 B chunks XOR-swizzled by head) and copied out 16 B per lane contiguous across lanes, in place of 8 B per lane into head rows 1 KB apart: +1.84% (+1.78…+1.91%), per case +0.80% (b60) to +3.54% (b8), 9 of 9 faster (same kernel family as E42; local:/shared_nfs/kyle/moe/k3six/runs/mla_fp8/workspace/forge_experiments/lessons/iter_005.md:3; :18; local:/shared_nfs/kyle/moe/k3six/runs/mla_fp8/workspace/forge_experiments/ablation.md:17)
- E45 [2026-11, gfx950, FlyDSL 0.3.2, N=1 per setting, eager calls on streams created with a CU mask, output against the same call alone] a one-launch split-K bf16 GEMM whose owner split polls its partners' flags and writes NaN once a bounded poll (~2 ms) gives up: m=256 3,151,872 NaN on 64 CUs and 2,340,864 on 128 CUs (35.5 and 12.4 ms per call), m=512 4,276,224 and 3,158,016 NaN; on 16 and 64 CUs the stress check failed at m=256 (gemm_n6144_k16384; local:/shared_nfs/kyle/moe/k3six/tasks2/step53b.out:12; :13; local:/shared_nfs/kyle/moe/k3six/tasks2/step73.out:9; poll bound local:/workspace/code/gpt_oss_docker/.claude/memory/project_k3_six_remaining.md:121)
