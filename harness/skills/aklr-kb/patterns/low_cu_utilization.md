---
title: "Low CU utilization: too few workgroups for 256 CUs, or a mostly empty last dispatch round"
kind: pattern
signals: ["tiles < NCU (256 on gfx950)", "last_round < slots in the dispatch-round arithmetic",
  "time flat across M while the launched grid is unchanged", "launched grid read from a kernel trace"]
ops: [gemm]
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
---

# Low CU utilization: too few workgroups, or a ragged last round

## Symptom

- The round arithmetic gives `tiles = ceil(M/tile_m) · ceil(N/tile_n) · split_k` below `NCU` (S1), or a
  `last_round` well short of `slots` (S2); `NCU` is 256 on gfx950: query it (S3).
- Time stays flat across M while the grid is unchanged (S4). Read the launched grid from a trace before any counter (S5).
- A persistent kernel whose tile count falls between two whole CU passes rounds its resident workgroups down (S6).

## Likely causes

1. **Too few tiles:** every workgroup walks all of K serially, so latency follows K, not M (S7).
2. **A ragged last round:** CUs idle through it (S8); "≥ 1024 workgroups and a multiple of 8" does not prevent it (S9).
3. **Fewer slots than assumed.** `wg_per_cu` comes from registers, LDS and wave slots (S10); a register-heavy GEMM
   is usually at 1, so `slots = 256` (S11). Occupancy moves in workgroup-sized steps (S12).
4. **A padded grid** that spills into one more round (S13).

## Rule out first

- **A streaming row.** Fewer tiles than CUs is necessary for starvation, not sufficient: a cold, read-once B stream
  can be HBM-bound with fewer blocks than CUs, and then every way to more blocks loses (S14); see `memory_bound.md`.
- **An idle-device ruler.** A single-shape tuner on an idle GPU overvalues split-K (S15); measure CU-fill levers
  under the real load mix (S16).
- **The compiler's occupancy comment.** `; Occupancy:` does not apply the workgroup quantization (S17).
- **The operating point.** Benchmark the deployment's per-call M and check the launched grid in a trace (S18).
- **The clock.** On a power-capped part, idle CUs come back as clock for the others, not as wall time (S19).

## Candidate techniques

- Narrow the tile so the last round fills — `gemm/grid_pricing.md` §1, §9 — a narrow tile wins only
  where its extra tiles still fit in the same number of rounds (S20), and it costs reuse (S21).
- Split-K — `gemm/grid_pricing.md` §2 — gated on wave quantization (S22), with `s = 1` kept in the race
  (S23); no split wins when the reduce costs more than the idle time it recovers (S24). Atomic split-K is not
  bitwise reproducible (S25): in AKLR reduce in a fixed order (S26).
- Tail split — `gemm/grid_pricing.md` §2 — split only the tiles of the ragged last round (S27), with
  each launch whole rounds by itself (S28).
- Persistent tiles — `gemm/grid_pricing.md` §10 — passing the gate is not a win: race it (S29).
- Small M — `gemm/small_m.md` §3–4 — count workgroups against the 256-CU round first (S30); split-K
  where narrow N leaves a single N-block, not on streaming rows or short K (S31, S32).
- Idle CUs load later K-tiles of B into L2 — `gemm/small_m.md` §6 — when the grid leaves CUs idle;
  race the first prefetched tile per M (S33).
- Other kernel families: attention's XCD-major mapping and live `(batch, kv-head)` slices per XCD —
  `attention/flydsl_attn_grid_dispatch.md` §2, §7; decode grid size and split rules —
  `attention/flydsl_attn_decode_split_kv.md` §4–5; grouped GEMM skew, longest-first and persistent grids —
  `moe/grouped_scheduling.md` §1, §5–6; MoE grids at decode sizes —
  `moe/grouped_moe_inference.md` §3.

## Evidence

- S1 gemm/grid_pricing.md:45 tiles = ceil(M/tile_m) · ceil(N/tile_n) · split_k
- S2 gemm/grid_pricing.md:48 last_round = tiles − (rounds − 1) · slots
- S3 gemm/grid_pricing.md:46 slots = NCU · wg_per_cu; NCU = 256 on gfx950, query it
- S4 gemm/grid_pricing.md:84 a small-M kernel ran flat across M at a fixed grid
- S5 gemm/grid_pricing.md:85 the grid alone explains it; read the launched grid from a trace first
- S6 gemm/grid_pricing.md:300 a tile count between two CU passes rounds the resident count down
- S7 gemm/grid_pricing.md:67 tiles < NCU: every workgroup walks all of K; latency set by K, not M
- S8 gemm/grid_pricing.md:69 last_round < slots: CUs idle through the last round
- S9 gemm/grid_pricing.md:88 "≥1024 workgroups and a multiple of 8" is not sufficient
- S10 gemm/grid_pricing.md:52 wg_per_cu from registers, LDS and wave slots
- S11 gemm/grid_pricing.md:61 a register-heavy GEMM is usually 1 workgroup per CU, slots = 256
- S12 gemm/grid_pricing.md:193 reachable occupancy is quantized by workgroup size
- S13 gemm/small_m.md:86 padding a grid turned one dispatch round into two
- S14 gemm/small_m.md:90 fewer tiles than CUs is necessary for starvation, not sufficient
- S15 gemm/small_m.md:60 a single-shape tuner on an idle GPU overvalues split-K
- S16 gemm/measurement.md:72 measure CU-fill levers under the real load mix
- S17 gemm/grid_pricing.md:203 the `; Occupancy:` comment does not apply the quantization
- S18 gemm/measurement.md:139 benchmark the deployment's per-call shape; check the grid in a trace
- S19 gemm/grid_pricing.md:294 on a power-capped part idle CUs come back as clock, not wall time
- S20 gemm/grid_pricing.md:285 a narrow tile wins only where its extra tiles fit the same rounds
- S21 gemm/grid_pricing.md:91 a narrower tile also costs reuse
- S22 gemm/grid_pricing.md:97 split-K gate: propose s only if it raises eff by a margin
- S23 gemm/grid_pricing.md:98 keep s = 1 in the candidate set
- S24 gemm/grid_pricing.md:103 if the reduce costs more than the idle time it recovers, no split wins
- S25 gemm/grid_pricing.md:122 atomic split-K is not bitwise reproducible
- S26 gemm/correctness.md:209 under AKLR's bitwise contracts, reduce the splits in a fixed order
- S27 gemm/grid_pricing.md:134 tail split: split only the tiles of the ragged last round
- S28 gemm/grid_pricing.md:136 head and tail launches must each be whole rounds
- S29 gemm/grid_pricing.md:309 passing the persistent gate is not a win: race it
- S30 gemm/small_m.md:86 count workgroups against the 256-CU dispatch round before anything else
- S31 gemm/small_m.md:110 split-K pays on a single N-block (narrow N) and latency-bound rows
- S32 gemm/small_m.md:111 split-K does not pay on streaming rows with many N-blocks or short K
- S33 gemm/small_m.md:229 idle CUs load later K-tiles of B into L2
