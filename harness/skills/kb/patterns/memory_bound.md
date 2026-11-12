---
title: "Memory-bound: the kernel is limited by global memory traffic"
kind: pattern
signals: ["bytes moved / wall time near the stream bandwidth measured on the same box",
  "wall time unchanged when MFMA work is added (compute-additivity probe)",
  "s_waitcnt vmcnt(0) in front of the MFMAs", "VMEM-load stalls (hotspot_analyzer.py)", "MemUnitStalled"]
ops: [gemm, moe, fused_moe, grouped_gemm, attention]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
---

# Memory-bound: limited by global memory traffic

## Symptom

- Bytes moved / wall time is near the stream bandwidth measured on the same box with the kernel's own read/write ratio
  (S1, S15); a rate above that ceiling means caches serve part of the traffic and the kernel is not at the HBM limit
  (S2). A copy ceiling does not judge a read-only kernel: on MI355X a cold read-only HIP stream with no L2 hits ran
  1.13–1.23× a 1:1 copy in steady state *(field, E2)*.
- MFMA work added with no extra traffic leaves wall time unchanged: compute is hidden (S3).
- The trace shows `s_waitcnt vmcnt(0)` in front of the MFMAs (S4), or `VMEM-load` stalls dominating at a high
  per-load rate (S5). A probe that deletes the hot loop's `ds_read`s and MFMAs (wrong output by design) is a
  diagnostic: it moved an M=8 o_proj (K=16384) less than its baseline spread, and short-K kernels at m 64–128 by much
  more; unless it keeps the loads live and the schedule fixed it bounds neither the memory nor the compute share (S6).
- How near is near: on fp8 × MXFP4 fused MoE at 32–112 tokens, kernels reading the weights of the experts their routes
  touch at 6.0–6.45 TB/s, measured 1.04–1.28× over aiter *(field, E1)*. On MI355X a read-only HIP kernel read expert
  bytes scattered the same way at 6.2–6.5 TB/s with ordinary loads and 6.8–7.0 TB/s with non-temporal loads
  *(field, E3)*: against the non-temporal roof those kernels had an estimated 7–12% left, more than a roof from
  ordinary loads shows.

## Likely causes

1. **Few MACs per operand byte.** Per K element a workgroup moves `tile_m · A_row_bytes + tile_n · B_col_bytes` for
   `tile_m · tile_n` MACs (S7); when the last dispatch round is full, intensity is the limit (S8).
2. **Many lines per instruction.** For data in L2 an instruction costs by the 128 B lines or 64-byte chunks it
   touches and by how its lanes share them, not by its bytes; streaming from HBM, loads cost their total traffic in
   whole 128 B lines, while stores still paid for scattered lines (S9). Operand rows off a 128 B line add lines (S10).
3. **Refetch across XCDs.** L2 is per XCD and workgroup ids go round-robin over the 8 XCDs, so a linear tile order
   spreads tiles that share a panel over all 8 L2s (S11).
4. **Latency or a compiler drain, not bandwidth.** `vmcnt(0)` in front of the MFMAs means either exposed global
   latency or a conservative drain the compiler inserted (S4).
5. **The C store,** when deleting the whole store is the only epilogue ablation that moves the time (S12).

## Rule out first

- **A warm ruler.** A replayed weight stays in the MALL/L2, while a step that touches each weight once finds it cold (S13): rotate an explicit
  number of B copies past the MALL (S14). Price headroom against a ceiling measured with the kernel's own read/write
  ratio (S15), not the datasheet; below about 1 GiB moved, also at the kernel's own byte count: on MI355X HIP streams
  that write peaked at 64–256 MiB, 18–26% above their rate at 1–4 GiB *(field, E4)*.
- **Bytes from a guess.** For a MoE, count the experts the timed inputs' routes touch, not the uniform estimate
  `E_local · (1 − (1 − k/E)^tokens)`: under expert parallelism the two differ (88 against 97 local experts at 112 tokens),
  and so does the floor *(field, E1)*.
- **A fixed per-tile cost:** TF/s rising with K is the same curve (S16); fit it (`gemm/grid_pricing.md` §6).
- **An idle grid:** with fewer tiles than CUs, each workgroup walks all of K (S17); see `low_cu_utilization.md`.
- **ATT charging:** an exposed global-load wait is charged to the `s_waitcnt vmcnt` in front of the consumer, not to
  the MFMA that reads the data (S18), so a high MFMA stall next to a small `VMEM-wait` share does not by itself show
  hidden memory latency.

## Candidate techniques

- Widen the tile, with `tile_m : tile_n` set by the operand widths — `gemm/grid_pricing.md` §1, §3 — on
  full rounds of a compute- or feed-bound tile at large M, to a tile that still fills whole rounds (S8, S19). A
  narrower tile where the last round is partial (S20) paid on the backward shapes of an MXFP4 training GEMM and lost
  in every case tried on two bf16 GEMMs at M 256–4096 (`gemm/grid_pricing.md` §1): race it. On a small-M row that
  streams B from HBM, a larger grid lost where split 1 already read B near a plain read of the same bytes and won
  where split 1 read well below that: compare the row with that read first (`gemm/grid_pricing.md` §1,
  `gemm/small_m.md` §2–3, `low_cu_utilization.md`).
- Operand rows on whole 128 B lines — `gemm/datapath.md` §6, §10 — for cause 2 (S10); on a grouped GEMM's LDS DMA
  reads, merging requests paid only where it also cut the lines touched (S21).
- XCD / L2 tile order — `gemm/schedule_and_locality.md` §2 — when co-resident workgroups reuse a slab; once one
  band's operand slice outgrows an XCD's L2, race it against the round-robin: clustering still paid there on one bf16
  GEMM in Triton (S22); judge by wall time and explain with stall counters such as `MemUnitStalled`, not L2 hit rate
  (S23).
- Prefetch, deeper stages — `gemm/datapath.md` §7, `../flydsl/guides.md` § prefetch-data-load — when independent
  work can fill the latency (S24); on one grouped GEMM whose working set thrashed L2 deeper prefetch bought nothing
  (S25), and a prefetch that pushes the kernel over a register tier lowers occupancy: on a HIP bf16 GEMM it lost 6.1%
  from 3 to 2 waves per SIMD and broke about even from 4 to 3 *(field, E7)*; re-read the counts and time it (S26).
- Cache policy on B and on C — `gemm/small_m.md` §6, `gemm/epilogue.md` §5 — a raced
  per-configuration axis whose sign depends on the shape (S27, S28).
- Dense small-M streaming rows (measured on one o_proj) — `gemm/small_m.md` §2–3 — there split-K and more blocks
  lose (S29).
- The C store — `gemm/epilogue.md` §2 — run its three ablations before changing it (S12).
- Other kernel families: attention gathers, unique lines and the index distribution —
  `attention/split_kv.md` §7; bytes per score and GQA sharer merging —
  `attention/grid_dispatch.md` §9; cache policy on MoE weights when few tiles share an expert's weights —
  `moe/grouped_sorted_tokens.md` §4.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, `python -m aklr validate --base origin/main`, not CI, on one MI355X: graph replay, 1 GiB flush per call, median candidate time; touched experts counted from the same validate inputs] weight bytes of the touched experts (2.19 MB per expert at inter 384, 17.5 MB at inter 3072: fp4 w1 and w2 plus E8M0 scales) over the candidate's time: 6.04–6.20 TB/s at 40–112 tokens (TP, 896 experts, 464–773 touched), 6.20–6.45 TB/s at 32–112 tokens (EP8, 112 of 896 experts local, 59–92 touched; at 112 tokens 88, against 97 by the uniform estimate); aiter 4.9–5.97 TB/s at the same Points; read roof 6.5 TB/s from a read-only kernel (Kimi-K3 SiTUv2 fused MoE, fp8 activations, model dim 3584, top-16)
- E2 [2026-11, gfx950 (MI355X), HIP 7.15, N=7 rounds × 5 calls per cell, rocprofv3 kernel-trace time and hipEvent, a 2 GiB read flush before every call, bandwidth as the slope of time over 2, 4 and 8 GiB] 16 B grid-stride streams: read-only 5.98 TB/s (kernel trace) and 5.95 (hipEvent) at 0.011% L2 hits, the fastest 1:1 copy 4.85 and 5.25: read-only over copy 1.23× and 1.13× (HIP microbenchmark)
- E3 [2026-11, gfx950 (MI355X), HIP 7.15, N=9 rounds × 15 calls, hipEvent (a lower bound on bandwidth), a 1 GiB read flush before every call] a read-only kernel with 16 B loads over the expert weights that random top-16 routes of 32–112 tokens touch, pools laid out as E1's: 6.79–7.02 TB/s with non-temporal loads, 6.16–6.48 TB/s with ordinary loads (HIP microbenchmark)
- E4 [2026-11, gfx950 (MI355X), HIP 7.15, N=7 rounds × 10 calls per cell, rocprofv3 kernel-trace time, a 2 GiB read flush before every call, best of three grid sizes] 16 B grid-stride streams from 32 MiB to 4 GiB: 1:1 copy, 2-read-1-write and write-only peaked at 5.90, 6.13 and 6.14 TB/s at 64–256 MiB and ran 4.81–5.22 TB/s at 1–4 GiB; read-only 5.33–5.71 at every size, 1.10–1.18× the copy at 1–4 GiB (HIP microbenchmark)
- E7 [2026-11, gfx950 (MI355X), HIP 7.15, N=11 rounds in one process, arms rotated, median of per-round ratios, θ = max(1%, 3 × SE)] a one-stage register prefetch of the next K step on a hand-written bf16 GEMM (8192×6144×1024, 128×128×32 tiles), placeholder registers putting it at a tier edge: 168 → 184 VGPRs, 3 → 2 waves per SIMD, +6.1%; 128 → 144 VGPRs, 4 → 3 waves, +2.1%, no clear difference (HIP microbenchmark)
- S1 gemm/attribution.md:45 apparent byte rate: kernel bytes / wall time vs stream bandwidth on the same box
- S2 gemm/attribution.md:48 a rate above stream bandwidth: caches serve part; not at the HBM limit
- S3 gemm/attribution.md:33 compute additivity: wall time does not move, compute is fully hidden
- S4 gemm/workflow.md:78 `vmcnt(0)` in front of the MFMAs: exposed global latency or a conservative drain
- S5 ROCm/FlyDSL@dd8373349:.claude/skills/kernel-trace-analysis/SKILL.md:336 hotspots dominated by VMEM-load at a high stall rate: memory-bound
- S6 gemm/small_m.md:293 compute-deletion probes are diagnostics, not an upper bound on compute's share
- S7 gemm/grid_pricing.md:212 bytes a workgroup moves per K element against the MACs it produces
- S8 gemm/grid_pricing.md:72 last round full: no idle slot, intensity is the limit, widen
- S9 gemm/datapath.md:170 in L2 an instruction costs by the 128 B lines it touches, or by 64-byte chunks where adjacent instructions read the rest; from HBM, loads cost total traffic only
- S10 gemm/datapath.md:291 keep every operand row on a 128 B line
- S11 gemm/schedule_and_locality.md:60 round-robin ids spread tiles sharing a panel over all 8 L2s
- S12 gemm/epilogue.md:62 only "delete the whole store" moving means the store is byte-bound
- S13 gemm/small_m.md:42 B is cold in decode; a benchmark replaying one weight keeps it in the MALL/L2
- S14 gemm/small_m.md:46 rotate enough distinct copies of B to exceed the MALL, explicit count
- S15 gemm/measurement.md:164 measured ceilings with the kernel's read/write ratio, not the datasheet
- S16 gemm/workflow.md:88 TF/s rising with K is not proof of memory-bound; fixed per-tile cost
- S17 gemm/grid_pricing.md:71 tiles < NCU: every workgroup walks all of K serially
- S18 gemm/attribution.md:209 ATT charges an exposed load wait to the `s_waitcnt`, not to the MFMA that consumes the data
- S19 gemm/grid_pricing.md:222 traffic per MAC is minimized at tile_m : tile_n = B_col_bytes : A_row_bytes
- S20 gemm/grid_pricing.md:73 partial last round: narrow the tile (or split)
- S21 gemm/datapath.md:181 on LDS DMA reads, merging requests paid only where it also cut the lines touched
- S22 gemm/schedule_and_locality.md:98 race the XCD order; the band's L2 footprint alone does not decide it
- S23 gemm/schedule_and_locality.md:129 judge by wall time; explain with stall counters (`MemUnitStalled`), not hit rate
- S24 gemm/datapath.md:211 prefetch pays only when independent work can fill the latency
- S25 gemm/datapath.md:214 prefetch hides latency, not capacity: deeper prefetch bought nothing on one grouped GEMM that thrashed L2
- S26 gemm/datapath.md:225 a prefetch over a register tier lowers occupancy; re-read the counts and time it
- S27 gemm/small_m.md:303 B's cache policy is a raced, per-configuration axis
- S28 gemm/epilogue.md:267 the store cache-policy sign is shape-dependent, race it
- S29 gemm/small_m.md:94 streaming rows: split-K and more blocks lost on an o_proj row at ~7.7 TB/s
