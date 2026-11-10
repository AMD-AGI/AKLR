---
title: "Attention — grid, dispatch order and XCD placement, audited before the kernel body"
kind: topic
lever: attn_grid_dispatch
ops: [attention, mla]
gens: [gfx950]
dtypes: [bf16]
regimes: [many query rows, dQ dK dV, split-kv combine, causal, sliding-window, rectangular, gqa]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:1000
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:4418
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_interface.py:1075
---

# Grid and dispatch for an attention kernel

> Which workgroup gets which `(batch, head, tile)`, in what order, and where the padding falls is index
> arithmetic outside the kernel body. On attention it often pays more than the body: *(field, E1, E2)* three
> index-only, bit-identical changes (XCD-major decode, LPT order, padding origin), kept together with an LDS
> hand-over and an LSE fold, gave +7% on a set of dQ/dK/dV kernels whose body tuning had returned ~1%.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- You start on an attention kernel, or the body has stopped paying.
- A causal, windowed or rectangular shape is slower than its FLOPs suggest.
- You are merging GQA heads into one workgroup, or changing the number of waves per workgroup.
- A consumer kernel (combine, reduce) reads what a producer kernel wrote.

**Somewhere else if:** few query rows per head, split KV and the combine → `split_kv.md`; dQ/dK/dV
workspace and fusion → `gradients.md`; generic round counting and tile pricing →
`../gemm/grid_pricing.md`.

## 1. The audit, in order
1. What do co-resident workgroups reuse? Make that axis adjacent inside an XCD (§2).
2. Is the work per workgroup uniform? If not, the dispatch order is a list schedule (§4).
3. Where does the tile padding fall — on the most or the least expensive tile (§5)?
4. Which axis does the grid spread over? Key the tile heuristic on it (§6).
5. How many `(batch, kv-head)` slices are live per XCD at once (§7)?

Judge changes at this layer by wall time, `SQ_WAIT_ANY` and `SQ_VALU_MFMA_COEXEC_CYCLES`. An L2 hit
rate can rise sharply and buy nothing.

## 2. XCD-major mapping
- Workgroup `i` runs on XCD `i % 8`, and each XCD has its own 4 MB L2 (`../gemm/schedule_and_locality.md` §2).
  FlyDSL's launcher states the same mapping for its forward grid (`flash_attn_interface.py:1075`).
- **Give each XCD a whole `(batch, kv-head)` block**, so the GQA workgroups that read the same K/V share one
  L2. *(field, E3, E4)* dq: L2 hit 86.5% → 94.8%, misses −62%, wall within noise (+0.54% kept together with an
  LDS hand-over); on the dK/dV kernel +2.81% on the whole backward (its own time −2.2%).
- **The fastest axis inside the block depends on the kernel.** Ask what co-resident workgroups reuse:
  the Q-outer dq kernel wanted the GQA sharers of one kv head adjacent, the KV-outer dK/dV kernel wanted
  adjacent q positions (q-split index fastest). *(field, E5)* The same remap with the inner order swapped:
  −2.5% instead of +2.4%. Do not copy the order from another kernel.
- **Use the remainder-spreading permutation.** `x*(G//8) + min(x, G%8) + bid//8` with `x = bid % 8`
  (`../gemm/schedule_and_locality.md` §2) is a bijection for any grid size `G`, so the
  `num_kv_heads % 8 == 0` gate goes away. *(field, E6)* With 6 kv heads, falling back to the plain decode for head
  counts not divisible by 8 left 9.1% of a windowed dK/dV body on the table.
- **Check bijectivity over every grid you launch.** The same binary is often launched on a narrower chunk
  grid than the deployment shape; a remap validated on `(B, Hkv)` faulted on a one-batch chunk.
- **Compute what `flat_id % 8` already is before changing anything.** A head-fast grid `(Hq, n_q, B)` with
  `Hq % 8 == 0` gives `flat_id % 8 == h % 8`; with `h_kv = h % Hkv` and `Hkv == 8`, every XCD already owns
  one kv head and any XCD remap is the identity. *(field, E7)* An XCD regrouping by kv head on such a kernel
  read −0.19% with memory counters bit-identical. FlyDSL's forward applies its head-slow remap to
  non-causal grids only (`flash_attn_utils.py:1000-1013`).
- **The same arithmetic sets the work per XCD; rows that share nothing want balance, not locality.** A row-fastest
  `(rows, splits)` grid with `rows % 8 == 0` runs every split of row r on XCD `r % 8`; rows of unequal length then
  load the XCDs unequally. Rotate the row by the split index (`row = (x + y) % rows`): *(field, E33, E34)* without
  it two split-KV decodes were 24% slower, on all 11 batch sizes.
- **A co-resident stream reverses the sign.** *(field, E8)* A contiguous per-XCD slice for a reduce launched
  alone cut its time 5.4% (a 1.8% on a second shape did not reproduce). The same slice for the same reduce
  running beside the body made it 12% slower, because it pinned
  the reduce's reads onto the L2 sets the body was writing through. Beside a co-resident body, offset the
  phase inside a slice instead of giving the stream a slice of its own.
- vL1D is not a reuse level for sharers. *(field, E9)* Placing the 8 GQA sharers of a forward tile on one CU
  cost 0.57%, with `TCP_TCC_READ_REQ` and `TCC_HIT` unchanged: one workgroup's own 32 KB K/V double buffer
  fills the vL1D.

## 3. A consumer's numbering must match the producer's ownership
A kernel that reads another kernel's output should give the reads of one row to the XCD that wrote it.
*(field, E10)* A split-KV combine with a 1-D row-major grid (`row * blocks_per_row + b`) spread each row's reads
over `min(blocks_per_row, 8)` L2 domains, while the producer owned `split * seq + token`. A 2-D grid
`(seq, blocks_per_row, 1)` makes the flat id `b * seq + row`, the producer's form: 18.6% and 7.0% faster on
two decode shapes, ~0.5% faster where the producer's own traffic had already evicted the partials. It is a pure
permutation. Before splitting any grid finer, check that the low 3 bits of the flat id did not change
with it. The grid-size cliff of a split-KV grid is in `split_kv.md` §4.

## 4. Dispatch order is a list schedule
- The dispatcher issues workgroups in block-id order, so the grid order is the schedule. Under a causal
  mask the work of a q-tile grows as `(q_tile + 1) * BLOCK_M / BLOCK_KV`: dispatch in descending q-tile
  order (longest processing time first). *(field, E11, E12)* dq +2.5% together with an LSE fold that gained
  nothing on its own (the reorder alone +1.1%, one read); forward +2.34%, but only with VGPR held below the
  occupancy step (a loop-carry split, or forcing waves_per_eu=3, +2.4%), because the plain reorder raised VGPR
  162 → 170 across it.
- FlyDSL's fp8 forward does this by reversing the q-block index under causal (`init_causal_lpt_order`,
  `flash_attn_utils.py:4418-4429`).
- **Do not assume in-order is already right.** *(field, E13)* In one code base the dK/dV kernel's in-order grid
  happened to be LPT and the dq kernel's was reversed. Read the decode of every kernel separately.
- **Choose the order from the profile shape.** Monotone → descending. Unimodal → start in the middle and
  walk out (`i → mid ± i/2`, O(1) index math, bit-identical). *(field, E14)* Rectangular causal plus a finite
  window gave a tent of trip counts (2 at the ends, 16 in the middle); ascending and descending were
  equally bad, middle-out cut body time 4.3% and wall 2.7%. Constant → nothing to win.
- **Do not make q-blocks the fastest axis under causal**; it clusters unequal work (FlyDSL records 7% for
  this, `flash_attn_utils.py:1005-1006`).
- **Trust a greedy "N slots per XCD" simulator for the first-order profile only.** *(field, E13, E12, E15)* It
  predicted the LPT gain (2.3% against 2.9% measured on the dq kernel; on a forward 2.29% against 2.34%), but two
  second-order phasings it priced 3.8% faster measured
  within noise and 0.4% slower. The model has no zero-work workgroups covering others and no cross-XCD DRAM
  contention.

## 5. Padding and empty work
- `num_q_tiles * BLOCK_M - Sq` rows of overshoot anchored at row 0 all land in the last tile, the one with
  the longest causal range. Move the origin down by `floor(pad / BLOCK_KV) * BLOCK_KV` so they land on
  tile 0. *(field, E16)* The dq kernel 1.0% faster (kv-block visits 7481 → 7396); the full backward moved within
  its noise. It needs `BLOCK_M % BLOCK_KV == 0`, the first
  tile clamped to row 0, and the store bounded by the rows the tile owns (shared rows are recomputed and
  written once, so the output stays deterministic).
- **Skip empty segments on the host.** A block-diagonal varlen pattern (context parallel) with most
  segments empty launches workgroups that only exit. *(field, E17)* 64 segments with one non-empty: 0.740 →
  0.153 ms by launching only the non-empty ones; the change takes that path only when at most 1/8 of the
  segments are non-empty.

## 6. Key the tile heuristic on the axis the grid spreads over
A KV-outer kernel's grid is `B · Hkv · ceil(Skv / BLOCK_KV) · q_split` (fills the CUs with Skv); a Q-outer
kernel's grid scales with Sq. *(field, E18)* Choosing the KV-outer kernel's tile and split from Sq cost 20% more
time (15.03 against 12.49 ms) at Sq = 2048, Skv = 16384.

Square shapes never expose this; put rectangular shapes in the acceptance set.

## 7. The first-order term is live `(batch, kv-head)` slices per XCD
At occupancy 1, more ready work per CU is not a gain. What matters is how many operand sets each XCD's L2
must hold at once. *(field, E19)* Running two batch-disjoint chunks concurrently, with bytes, slots and
workgroup count identical, made the dK/dV body 13–32% slower (min and average kernel time) and the B=4 wall
2.0% slower: a resident window of 256 consecutive block ids
shares an estimated ~2 MB of Q/dO plus 0.5–2 MB of K/V, and two chunks doubled that against a 4 MB L2. FlyDSL's forward
launcher comment reports the same term from the other side: the penalty for leaving its remap off tracks
the number of K/V streams per XCD, ~6% at 8 streams, ~3% at 4, nil at ≤ 2 (`flash_attn_interface.py:1075-1085`).
Keep one slice live per XCD; do not overlap chunks on separate streams to "fill" the machine.

## 8. Scan the batch axis before chasing tails
Before writing a persistent or work-stealing scheduler to collect the tail, fix the work per workgroup and
sweep only the batch. *(field, E20)* B = 1..16 read 1146.7 → 1195.7 TF/s with the scored shape already on the
asymptote, so grid quantization plus LPT tail was ≤ 0.2% and that family was closed. A sequence-length
sweep moves the grid and the per-workgroup work together; doubling S bought +0.7%, i.e. an estimated ~1.4% of
amortizable fixed cost.

Read a pool from two or more points of a curve, never from an end point.

## 9. Bytes per score and GQA sharer merging
- For a kernel that streams K/V once per workgroup, K/V bytes per score are `4D / (NUM_WAVES × ROWS_PER_WAVE)`:
  only the q rows one workgroup covers matter, not `BLOCK_N` or the number of heads it serves. *(field, E21)*
  2 → 4 heads per workgroup with `block_m` 128 → 64 (to keep 8 waves) left sectors and ISA bit-identical.
- Extra K/V bytes can cost cycles rather than energy. *(field, E22)* Doubling them cost 3.69% wall with MFMA-busy
  bit-identical and the clock flat; the cost sat in `SQ_WAVE_CYCLES` (+4.04%) and `SQ_WAIT_ANY` (+6.54%).
  Halving requests by merging sharers (below) paid through the clock instead: read sclk and `SQ_WAVE_CYCLES` for
  every byte change.
- **Two sharers per workgroup.** Run the same q-tile index of two q-heads of one kv-head in an 8-wave
  workgroup (low wave bits pick the row group, high bits the head), sharing one K/V LDS tile and barrier.
  *(field, E23)* Forward at the power cap: 2.04% faster, `TCP_TCC_READ_REQ` −49.5%, clock +8.7% at unchanged
  power, cycles +6.6%. The whole gain came through the clock.
- **Precondition: the merge must leave more than one workgroup per CU** (`../gemm/grid_pricing.md` §4).
  *(field, E24, E25)* In a VGPR-locked occupancy-2 backward, the same merge left one workgroup per CU with nothing
  to cover barriers and drains: dq −2.7% (the dq kernel itself an estimated −6.5%), an 8-wave dK/dV −4.3%, outputs
  bit-identical.
- **A 16-wave workgroup loses through barrier-domain width.** *(field, E26, E27)* A 16-wave workgroup with
  registers, waves per SIMD, bytes and instruction counts all identical was −3.1% (another reading of the same
  change −4.3%): 16 waves now wait for each other. Where the merge also lowered waves per SIMD (3 → 2) it lost 2.0%.
  *(field, E32)* Four sharers in an 8-wave workgroup (registers and waves per SIMD unchanged, 4 → 2 workgroups
  per CU) won 4.0%; eight sharers in a 16-wave workgroup were 1.1% slower than that.
- **20-second gate:** read `TCP_TCC_READ_REQ` on your kernel and on the reference. A ratio under 1.1 means
  the request path is already level and a merge has nothing to save.
- Changing waves per workgroup changes the rendezvous pool. *(field, E28)* Removing every barrier (diagnostic,
  wrong results) gained 0% with three 4-wave workgroups per CU, +2.4% (+1.25% on another build) with two 8-wave
  ones, and 0% with one
  16-wave one. It also changes how many DMA instructions each wave issues: recompute every `vmcnt` that
  was written as an instruction count.

## 10. Pipeline along the GQA head axis
A body that loops over the q-heads sharing one kv-head (measured on dK/dV bodies) has a second loop axis to
pipeline.
- Issue head `h+1`'s exp2/pack work inside head `h`'s GEMM2 MFMA run. *(field, E29)* +1.4% along the head
  axis; the same pipelining along the d-tile axis was negative.
- Issue the next head's LSE/delta loads at the last d-tile of the current head's GEMM2. *(field, E30)* +1.81%,
  register-neutral and bit-identical. On an occupancy-1 fused backward, carrying it across the q-block
  boundary (the last head prefetches the next q-block's head 0 through the loop's iter_args) added +0.29% /
  +0.55% *(field, E31)*; on the occupancy-2 body it was neutral (the next q-block's own DMA drain already covered
  head 0).
- This hides latency only. If the load's stall is a full VMEM issue queue, moving it earlier does nothing.

## Verify
- The `flat_id → (batch, head, tile)` decode is written down, with `flat_id % 8` evaluated for each shape class.
- Every remap is checked as a bijection over every launched grid width.
- The dispatch order was chosen from the measured trip-count profile.
- Rectangular, windowed and small-batch shapes are in the acceptance set.

## Related
`gradients.md` · `split_kv.md` · `attribution.md` ·
`../gemm/schedule_and_locality.md` §2 · `../gemm/grid_pricing.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. A version named only in a prompt or in prose is not taken. The month is the run's, or, where the number
is unverified, of the date it was reported. "Score" is the campaign's scored
bench; "best of 3" is its rule of keeping the best of three reads.

- E1 [2026-07, gfx950, FlyDSL ?, N=3, campaign score: full-backward conv TF/s, best of 3 reads per round] four kept rounds took the score from 789.6 to 844.6 conv TF/s (+7.0%): XCD-major dq decode together with an LDS hand-over 793.9 (+0.54%), XCD-major dK/dV decode 816.2 (+2.81%), descending q-tile dispatch together with a dq LSE fold into the MFMA C-init 836.6 (+2.50%), causal-aligned q-tile origin 844.6 (+0.96%); the LSE fold changed dQ's bits (dQ SNR 52.16 → 50.26 dB), so the four rounds together were not bit-identical (Primus-Turbo hd64 attention backward, odo + dq + dK/dV, B=1, Hq=128, Hkv=16, Sq=Skv=16384, causal, hardware exp)
- E2 [2026-07, gfx950, FlyDSL ?, N=?, ?] weeks of tuning inside the kernel body (tile size, double buffering, traversal direction) returned about 1%; unverified (Meta and Primus-Turbo hd64 attention backward)
- E3 [2026-07, gfx950, FlyDSL ?, N=3, campaign score as E1; L2 hit and misses from a rocprofv3 PMC profile] XCD-major dq decode alone: L2 hit 86.53 → 94.84%, misses −62%, score 789.1 against 787.6 (within noise); kept in the same round with a register-neutral LDS hand-over: 793.9 against 789.6 (+0.54%) (as E1)
- E4 [2026-07, gfx950, FlyDSL ?, N=3, campaign score as E1; kernel times and counters from one PMC profile] XCD-major dK/dV decode with the q-split index fastest: score 816.2 against 793.9 (+2.81%); dK/dV L2 hit 86.31 → 90.19%, SQ_WAIT_ANY −16.2%, dK/dV kernel 8.054 → 7.878 ms avg (as E1)
- E5 [2026-07, gfx950, FlyDSL ?, N=1 (kv-tile fastest) and 3 (q-split fastest), campaign score as E1; the +2.4% from an interleaved A/B, n=3] the dK/dV XCD-major decode with the kv-tile index fastest 767.1 (−2.5%); with the q-split index fastest 808.8 against 789.9 (+2.4%) (as E1)
- E6 [2026-08, gfx950, FlyDSL ?, N=?, per-kernel ledger for the body; the cell's wall from the scored bench, interleaved C-B-C-B-C-B-C] remainder-spreading contiguous per-XCD cut on the windowed body at Hkv=6: body 1.8121 → 1.6478 ms (−164 µs, −9.1%), wall 2.4723 → 2.3758 ms; a negative control that changed only the axis order read 1.8050; the plain modulo form faulted on the one-batch launch (gpt-oss attention backward, fused KV-outer dK/dV body emitting dQ partials, B=4, Hq=48, Hkv=6, Sq=Skv=4096, D=128, window 2047)
- E7 [2026-07, gfx950, FlyDSL 0.2.2, N=?, the campaign's bench] regrouping by kv head inside each XCD: −0.19%; that its memory counters stayed bit-identical is unverified (Primus-Turbo hd64 attention forward, B=1, Sq=Skv=16384, full causal; version printed in the same campaign)
- E8 [2026-08, gfx950, FlyDSL ?, N=2, per-kernel ledger, two reads per arm] the same contiguous per-XCD cut on the dQ reduce grid: reduce launched alone 217.6 / 214.0 → 204.4 / 203.7 µs (−5.4%, cell 3) and 100.6 / 101.5 → 99.3 / 99.3 µs (−1.8%, cell 0); reduce co-resident with the body 641 → 714 / 718 µs (+12%, cell 1); two rounds later the cell-0 gain did not reproduce (0.0999 → 0.1002 ms) (gpt-oss fused attention backward, windowed cells)
- E9 [2026-07, gfx950, FlyDSL 0.2.2, N=?, the campaign's bench and a PMC pair] the 8 GQA sharers of a `(kv_head, q_block)` placed on one CU: wall −0.57%, TCP_TCC_READ_REQ, TCP_PENDING_STALL_CYCLES and TCC_HIT bit-identical; one workgroup's double-buffered K/V is 32 KB (Primus-Turbo hd64 attention forward as E7)
- E10 [2026-09, gfx950, FlyDSL 0.3.2, N=?, kernel times from the campaign's decode attention bench; scored step three runs] 2-D row-minor combine grid: `mla_d8` (seq 8) 8.2545 → 6.7213 µs (−18.6%), `mla_d14` (seq 14) 10.6077 → 9.8624 µs (−7.0%), verify shapes (seq 48, 60, 84) about −0.5%; score 1.57713 → 1.58145 (GLM-5.2 sparse-MLA decode, aiter's FlyDSL `mla_decode_combine`; version printed the next round in the same venv)
- E11 [2026-07, gfx950, FlyDSL ?, N=3, campaign score as E1] descending q-tile dispatch on dq with an LSE fold into the MFMA C-init: 836.6 against 816.2 (+2.50%); the dispatch alone 827.0 (+1.14%, one read); the fold alone on the old order +0.0%; dq kernel 5.967 → 5.799 ms (as E1)
- E12 [2026-07, gfx950, FlyDSL ?, N=3, the campaign's bench, `bench tflops (n=3)`] descending q-block dispatch with a loop-carry split: 931.3 → 953.1 TF/s (+2.34%) at VGPR 162; descending alone raised VGPR 162 → 170 across the 3 → 2 waves-per-SIMD step; descending with `waves_per_eu` forced to 3 (VGPR 168) 953.4 (+2.4%), rejected only because the attribute reached every configuration; the offline greedy model predicted +2.29% for the reorder (Primus-Turbo hd64 attention forward as E7)
- E13 [2026-07, gfx950, FlyDSL ?, N=?, offline greedy list-schedule model, 64 slots per XCD (no GPU), against the round's measurement] the model put the dq pair-order-to-descending gain at +2.30% (makespan tail 2.34% → 0.04%), measured +2.9% (dq kernel 5.967 → 5.799 ms); the deployed dK/dV order simulated at a +0.19% tail, i.e. already LPT (as E1)
- E14 [2026-08, gfx950, FlyDSL ?, N=2, per-kernel ledger, two body reads per arm; wall four reads] middle-out dispatch of the live bands, whose trip counts run 2 at the ends and 16 in the middle: body 0.8149 → 0.7801 / 0.7747 ms (−4.3%), wall 1.2464 → 1.2054 / 1.2221 / 1.2138 / 1.2144 ms (−2.7%) (gpt-oss fused attention backward, windowed rectangular cell: B=4, Hq=64, Sq=1024, Skv=16384, D=128, window 2047)
- E15 [2026-08, gfx950, FlyDSL ?, N=12, body time over 12 cross-process reads (the second form's count is not stated)] two phasings of the live bands across family groups, each priced by the greedy model at body −3.8%: measured −0.25% (within noise) and +0.4% (as E14)
- E16 [2026-07, gfx950, FlyDSL ?, N=3, campaign score as E1; the dq kernel alone, interleaved, n=3] causal-aligned q-tile origin: score 844.6 against 836.6 (+0.96%); kv-block visits 7481 → 7396 per head; dq kernel 5.2686 → 5.2150 ms (−1.02%); the interleaved full bench read −0.1% at Sq=16384 and +0.75% at 8192, below its resolution (as E1)
- E17 [2026-08, gfx950, FlyDSL ?, N=?, ?] launching only the non-empty q segments of a block-diagonal varlen backward: 64 segments with one non-empty 0.740 → 0.153 ms; the host path is gated on `n_ne * 8 <= n_seg` (Primus-Turbo FlyDSL attention varlen backward, context-parallel block-diagonal pattern; Primus-Turbo a25b5e0d, its message and the gate in `primus_turbo/pytorch/kernels/attention/attention_flydsl_impl.py`)
- E18 [2026-07, gfx950, FlyDSL ?, N=?, ?] the dK/dV tile chosen from Sq on a rectangular shape: 15.03 ms, against 12.49 ms with the tile chosen from Skv (recorded as −19%); square shapes unchanged (Meta hd64 attention backward, KV-outer dK/dV, B=4, Sq=2048, Skv=16384, full causal; Primus-Turbo 85b28c41, its message; the full-causal column of the Meta table in `benchmark/ops/training/bench_flydsl_flash_attn_bwd.py` at c5fd3e7c)
- E19 [2026-09, gfx950, FlyDSL 0.2.4, N=1, one rocprofv3 kernel-trace run per arm (`R3_B=4 R3_N=9`), avg and min dispatch time; cell score from the campaign's same-process ABBA bench, two reads per arm] the two batch-disjoint chunks of the B=4 plan on two streams, bytes, slots and workgroup count unchanged: dK/dV body avg 2793 → 3697 µs (+32%), min 2463 → 2790 µs (+13%); B=4 cell 0.9853 / 0.9884 against 1.0100 / 1.0044 (−2.0%); a later round of the same campaign treats the kernel-trace average as polluted by concurrency (gpt-oss D=64 attention backward, B=4, Hq=64, Hkv=8, S=8192, causal; version printed in the same campaign)
- E20 [2026-08, gfx950, FlyDSL ?, N=1, one run per point, warms=5 reps=40] batch scan at fixed per-workgroup work: B=1, 2, 3, 4, 5, 6, 8, 16 → 1146.7, 1176.8, 1184.1, 1193.6, 1193.3, 1194.1, 1194.1, 1195.7 TF/s (B=12 1177.8, set aside as shared-node noise); S scan at B=4: S=8192 → 16384 1190.3 → 1198.6 TF/s (+0.7%) (gpt-oss D=64 attention forward, Hq=64, Hkv=8, S=8192, causal)
- E21 [2026-08, gfx950, FlyDSL ?, N=10, the campaign's screening ruler, n=10, and its scorer] `Q_HEADS_PER_WG` 2 → 4 with `block_m` 128 → 64: ISA bit-identical to the deployed build; screening 1.84095 against 1.84605 ms (+0.28%), scored 1.01954 against 1.020395 (−0.08%) (gpt-oss D=64 attention forward as E20, B=4)
- E22 [2026-08, gfx950, FlyDSL ?, N=3, the campaign's bench, three reads per arm; a rocprofv3 PMC pair of 14 dispatches each] a second K/V DMA per tile (buffer_load 4 → 8, the ISA otherwise identical): 1.9030 against 1.8353 ms (−3.69%), sclk 1620 against 1625 MHz at 1399–1401 W; SQ_VALU_MFMA_BUSY_CYCLES 2.45485e9 on both, SQ_WAVE_CYCLES +4.04%, SQ_WAIT_ANY +6.54% (gpt-oss D=64 attention forward as E21)
- E23 [2026-07, gfx950, FlyDSL 0.2.2, N=2, palindromic A B B A in one session; PMC pair; sclk and power pair] two GQA sharers per 8-wave workgroup: 1175.3 / 1179.2 against 1200.2 / 1202.4 TF/s (+2.04%); TCP_TCC_READ_REQ 2.734e8 → 1.381e8 (−49.5%), sclk 1609 → 1749 MHz (+8.7%) at 1342 → 1339 W, true cycles +6.6% (Primus-Turbo hd64 attention forward as E7)
- E24 [2026-07, gfx950, FlyDSL ?, N=3, the campaign's full-backward score, three reads] dq with two sharers per 8-wave workgroup (VGPR 252 → 250, spill 0, buffer_load 8 → 4 per wave-trip), outputs bit-identical: 819.6 against 842.5 (−2.7%, 3/3); the round's agent puts the dq kernel's own loss at ~6.5%, a figure no kernel timing in the record shows (gpt-oss hd64 attention backward, dq + dK/dV, B=2, Hq=64, Hkv=8, D=64, Sq=Skv=8192)
- E25 [2026-07, gfx950, FlyDSL ?, N=?, the campaign's full-backward score] dK/dV with BLOCK_KV=256 and an 8-wave workgroup (32 kv rows per wave, VGPR-neutral by construction): 795.5 against 829.5 (−4.3%), one workgroup per CU (as E24)
- E26 [2026-08, gfx950, FlyDSL ?, N=2, the campaign's bench, two reads per arm] a 16-wave workgroup (`Q_HEADS_PER_WG=4`, each K/V line filled twice) with an ISA identical to the 8-wave build (VGPR 128, spill 4, LDS 34048, 240 instructions, 4 waves per SIMD): 1.9011 / 1.9113 against 1.8479 / 1.8491 ms (−3.1%); the same 16-wave arm, built earlier in the campaign, 1.9207 against 1.8414 ms (−4.3%) (gpt-oss D=64 attention forward as E21)
- E27 [2026-09, gfx950, FlyDSL ?, N=2, palindromic B A A B in one session; the D=128 cell, unchanged by construction, as the same-binary control] GQA merge from 2 to 4 q-heads per workgroup at D=64: VGPR 168 → 190 (3 → 2 waves per SIMD), spill 22 → 0, VMEM loads 8 → 4 per trip, output bit-identical; D=64 A/B +2.03% slower against a −0.94% position bias on the control, a 2–3% loss; a same-node re-check of the D=64 cell −2.8% (gpt-oss D=64 attention forward, B=4, Hq=64, Hkv=8, S=8192, causal)
- E28 [2026-07, gfx950, FlyDSL 0.2.2, N=?, the campaign's bench; diagnostic arms with wrong results] deleting every barrier: 0% with 4-wave workgroups, +2.4% (1229.9 TF/s) with the merged 8-wave workgroup (+1.25% when re-measured on a later build), 0% with 16-wave workgroups (1202.1 against 1205.5); the record gives 2 and 1 workgroups per CU for the 8- and 16-wave builds, and 3 is also reported for the 4-wave one (Primus-Turbo hd64 attention forward as E7)
- E29 [2026-07, gfx950, FlyDSL ?, N=?, ?] issuing head h+1's exp2 inside head h's GEMM2 MFMA run is recorded at +1.4%, and the same along the d-tile axis as negative; unverified, and on 2026-07-16 the change was absent from the kernel a campaign was working on (Meta hd64 attention backward)
- E30 [2026-07, gfx950, FlyDSL ?, N=3, campaign score: hardware-exp conv TF/s, three reads] the next head's LSE/delta loads issued at the last d-tile of the current head's GEMM2: 1134.4 against 1114.2 (+1.81%), spill 8 unchanged, MFMA count unchanged, SNR 52.6 dB, deterministic; issued before GEMM2 instead −0.6% (spill 8 → 13) (Meta hd64 attention backward, dK/dV kernel, B=4, Hq=128, Hkv=16, D=64, Skv=16384, bottom-right causal, hardware exp)
- E31 [2026-08, gfx950, FlyDSL ?, N=2, same-process interleaved A/B, two sessions] carrying that prefetch across the q-block boundary through the loop's iter_args: +0.29% and +0.55% in two sessions; together with GEMM1 ks-outer issue 15/15 +0.57% (gpt-oss fused attention backward, dQ folded into the KV-outer dK/dV kernel, B=3, S=8192, Hq=64, Hkv=8, D=64, full causal)
- E32 [2026-09, gfx950, FlyDSL ?, N=3, campaign score: geometric mean of main_ms / candidate_ms over the d128 and d64 cells, three runs; a same-session null of the untouched tree read 1.006168] four GQA sharers per 8-wave workgroup at the harness's `block_m` 64 (VGPR 128, spill 2, scratch 12 B, LDS 34048 all unchanged; 4 → 2 workgroups per CU, total waves unchanged; K/V L2 → LDS traffic 16.9 → 8.5 GB): d64 cand_ms 1.84051 → 1.76759 (−4.0%), re-reads 1.76710 and 1.78059; eight sharers in a 16-wave workgroup 1.78758 ms, 1.1% slower than four (FlyDSL dualwave_swp attention forward, d64 B4 Hq64 Hkv8 S8192 D64 causal)
- E33 [2026-10, gfx950, FlyDSL 0.3.2, N=?, the campaign's ablation: paired timing on AKLR's graph timer, noise floor 0.12%] row rotated by split index against the unrotated `(rows, splits)` grid: +24.07% (one-sided 95% bounds +24.04 … +24.10%), 11 of 11 batch sizes faster (8–112, all multiples of 8); driver bench per case 2.3% (batch 8) to 30.7% (batch 96) less time (dense MLA decode, 96 heads, one query, fp8 KV, page size 1, `cache_seqlens` uniform in [1, 112431]; MI355X)
- E34 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm for the per-batch times (driver --bench-mode, warmup 3, iters 20, cache cleared); the gain from the campaign's 3-pair ablation; probe: partial-kernel µs, N=?] split-KV decode producer with grid `(rows, ns)`, row index fastest, rows 8–112 (all multiples of 8), per-row lengths uniform in [1, 112431]: `row = (bx + by) % rows` against `row = bx`: b8 0.4526 → 0.4407 ms, b16 0.7380 → 0.5590, b96 4.6846 → 3.2450, b112 4.5350 → 3.7692; ablation +23.65% when kept, +23.88% measured by the ablation (one-sided 95% bounds +23.73…+24.04%), 11/11 cases faster. An input-only probe on the unrotated kernel, rows permuted so every XCD gets equal keys: b8 −1.7%, b56 −6.9%, b96 −30.9%, b112 −21.1% partial-kernel time (96-head MLA decode, fp8 KV, page size 1)
