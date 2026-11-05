---
title: "Memory-bound: the kernel is limited by global memory traffic"
kind: pattern
signals: ["bytes moved / wall time near the stream bandwidth measured on the same box",
  "wall time unchanged when MFMA work is added (compute-additivity probe)",
  "s_waitcnt vmcnt(0) in front of the MFMAs", "VMEM-load stalls (hotspot_analyzer.py)", "MemUnitStalled"]
ops: [gemm, moe]
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
---

# Memory-bound: limited by global memory traffic

## Symptom

- Bytes moved / wall time is near the stream bandwidth measured on the same box (S1); a rate above it means caches
  serve part of the traffic and the kernel is not at the HBM limit (S2).
- MFMA work added with no extra traffic leaves wall time unchanged: compute is hidden (S3).
- The trace shows `s_waitcnt vmcnt(0)` in front of the MFMAs (S4), or `VMEM-load` stalls dominating at a high
  per-load rate (S5). At decode M, a probe that deletes the hot loop's `ds_read`s and MFMAs stays within noise (S6).
- How near is near: on fp8 × MXFP4 fused-MoE decode, kernels reading the weights of the experts their routes
  touch at 6.0–6.45 TB/s, against a 6.5 TB/s read roof measured on MI355X, measured 1.04–1.28× over aiter, and
  6.5 TB/s caps what is left at 5–8% *(field, E1)*.

## Likely causes

1. **Few MACs per operand byte.** Per K element a workgroup moves `tile_m · A_row_bytes + tile_n · B_col_bytes` for
   `tile_m · tile_n` MACs (S7); when the last dispatch round is full, intensity is the limit (S8).
2. **Many unique lines.** The cost unit is unique 128-byte lines touched per instruction, not bytes (S9); operand
   rows off a 128 B line add lines (S10).
3. **Refetch across XCDs.** L2 is per XCD and workgroup ids go round-robin over the 8 XCDs, so a linear tile order
   spreads tiles that share a panel over all 8 L2s (S11).
4. **Latency or a compiler drain, not bandwidth.** `vmcnt(0)` in front of the MFMAs means either exposed global
   latency or a conservative drain the compiler inserted (S4).
5. **The C store,** when deleting the whole store is the only epilogue ablation that moves the time (S12).

## Rule out first

- **A warm ruler.** A replayed weight stays in the MALL/L2 while decode reads it cold (S13): rotate an explicit
  number of B copies past the MALL (S14). Price headroom against a measured ceiling, not the datasheet (S15).
- **Bytes from a guess.** For a MoE, count the experts the timed inputs' routes touch, not the uniform estimate
  `E_local · (1 − (1 − k/E)^tokens)`: under expert parallelism the two differ (88 against 97 local experts at 112 tokens),
  and so does the floor *(field, E1)*.
- **A fixed per-tile cost:** TF/s rising with K is the same curve (S16); fit it (`gemm/grid_pricing.md` §6).
- **An idle grid:** with fewer tiles than CUs, each workgroup walks all of K (S17); see `low_cu_utilization.md`.
- **ATT charging:** an MFMA waiting for its operand is charged to the MFMA, not to an `s_waitcnt` (S18).

## Candidate techniques

- Widen the tile, with `tile_m : tile_n` set by the operand widths — `gemm/grid_pricing.md` §1, §3 — on
  full rounds (S8, S19); a partial last round wants a narrower tile instead (S20).
- Operand rows on whole 128 B lines — `gemm/datapath.md` §6, §10 — for cause 2 (S10); merging requests
  pays only when it also cuts unique lines (S21).
- XCD / L2 tile order — `gemm/schedule_and_locality.md` §2 — when co-resident workgroups reuse a slab;
  not once one band's operand slice outgrows an XCD's L2 (S22); judge by wall time and stall counters such as
  `MemUnitStalled`, not L2 hit rate (S23).
- Prefetch, deeper stages — `gemm/datapath.md` §7, `../flydsl/guides.md` § prefetch-data-load — when independent
  work can fill the latency (S24); not when the working set thrashes L2 (S25), nor across a register tier (S26).
- Cache policy on B and on C — `gemm/small_m.md` §6, `gemm/epilogue.md` §5 — a raced
  per-configuration axis whose sign depends on the shape (S27, S28).
- Decode streaming rows — `gemm/small_m.md` §2–3 — there split-K and more blocks lose (S29).
- The C store — `gemm/epilogue.md` §2 — run its three ablations before changing it (S12).
- Other kernel families: attention gathers, unique lines and the index distribution —
  `attention/flydsl_attn_decode_split_kv.md` §7; bytes per score and GQA sharer merging —
  `attention/flydsl_attn_grid_dispatch.md` §9; cache policy on MoE weights at decode sizes —
  `moe/flydsl_grouped_moe_inference.md` §4.

## Evidence

- E1 [2026-10, gfx950, FlyDSL ?, N=1, `python -m aklr validate --base origin/main` run locally on one MI355X: graph replay, 1 GiB flush per call, median candidate time; touched experts counted from the same validate inputs] weight bytes of the touched experts (2.19 MB per expert at inter 384, 17.5 MB at inter 3072: fp4 w1 and w2 plus E8M0 scales) over the candidate's time: 6.04–6.20 TB/s at 40–112 tokens (TP, 896 experts, 464–773 touched), 6.20–6.45 TB/s at 32–112 tokens (EP8, 112 of 896 experts local, 59–92 touched; at 112 tokens 88, against 97 by the uniform estimate); aiter 5.0–5.97 TB/s at the same Points; read roof 6.5 TB/s from a read-only kernel (Kimi-K3 SiTUv2 fused MoE, fp8 activations, model dim 3584, top-16; local:/shared_nfs/kyle/moe/k3rebase/val2/report.json; local:/shared_nfs/kyle/moe/kf/scan/touched.json; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_mi355_measured_ceilings.md)
- S1 gemm/attribution.md:42 apparent byte rate: kernel bytes / wall time vs stream bandwidth on the same box
- S2 gemm/attribution.md:44 a rate above stream bandwidth: caches serve part; not at the HBM limit
- S3 gemm/attribution.md:33 compute additivity: wall time does not move, compute is fully hidden
- S4 gemm/workflow.md:76 `vmcnt(0)` in front of the MFMAs: exposed global latency or a conservative drain
- S5 ROCm/FlyDSL@dd8373349:.claude/skills/kernel-trace-analysis/SKILL.md:336 hotspots dominated by VMEM-load at a high stall rate: memory-bound
- S6 gemm/small_m.md:184 at decode M the MFMA side is not the lever; the time is the B stream
- S7 gemm/grid_pricing.md:155 bytes a workgroup moves per K element against the MACs it produces
- S8 gemm/grid_pricing.md:68 last round full: no idle slot, intensity is the limit, widen
- S9 gemm/datapath.md:168 the cost unit is unique 128-byte lines touched per instruction
- S10 gemm/datapath.md:264 keep every operand row on a 128 B line
- S11 gemm/schedule_and_locality.md:64 round-robin ids spread tiles sharing a panel over all 8 L2s
- S12 gemm/epilogue.md:61 only "delete the whole store" moving means the store is byte-bound
- S13 gemm/small_m.md:39 B is cold in decode; a benchmark replaying one weight keeps it in the MALL/L2
- S14 gemm/small_m.md:42 rotate enough distinct copies of B to exceed the MALL, explicit count
- S15 gemm/measurement.md:147 measured ceilings with the kernel's read/write ratio, not the datasheet
- S16 gemm/workflow.md:86 TF/s rising with K is not proof of memory-bound; fixed per-tile cost
- S17 gemm/grid_pricing.md:67 tiles < NCU: every workgroup walks all of K serially
- S18 gemm/attribution.md:202 ATT charges an MFMA waiting for its operand to the MFMA
- S19 gemm/grid_pricing.md:170 traffic per MAC is minimized at tile_m : tile_n = B_col_bytes : A_row_bytes
- S20 gemm/grid_pricing.md:69 partial last round: narrow the tile (or split)
- S21 gemm/datapath.md:173 merging requests pays only when it also cuts unique lines
- S22 gemm/schedule_and_locality.md:99 gate the band on L2 bytes
- S23 gemm/schedule_and_locality.md:124 judge by wall time and stall counters (`MemUnitStalled`), not hit rate
- S24 gemm/datapath.md:202 prefetch pays only when independent work can fill the latency
- S25 gemm/datapath.md:202 prefetch hides latency, not capacity; L2 thrash
- S26 gemm/datapath.md:208 a prefetch over a register tier costs more than it hides
- S27 gemm/small_m.md:189 B's cache policy is a raced, per-configuration axis
- S28 gemm/epilogue.md:166 the store cache-policy sign is shape-dependent, race it
- S29 gemm/small_m.md:79 streaming rows: split-K and more blocks lose
