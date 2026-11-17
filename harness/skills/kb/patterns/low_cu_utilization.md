---
title: "Low CU utilization: too few workgroups for 256 CUs, or a mostly empty last dispatch round"
kind: pattern
signals: ["tiles < NCU (256 on gfx950)", "last_round < slots in the dispatch-round arithmetic",
  "time flat across M while the launched grid is unchanged", "launched grid read from a kernel trace"]
ops: [gemm, grouped_gemm, fused_moe, attention]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
---

# Low CU utilization: too few workgroups, or a ragged last round

## Symptom

- The round arithmetic gives `tiles = ceil(M/tile_m) · ceil(N/tile_n) · split_k` below `NCU` (S1), or a
  `last_round` well short of `slots` (S2); `NCU` is 256 on gfx950: query it (S3).
- Time stays flat across M while the grid is unchanged (S4). Read the launched grid from a trace before any counter (S5).
- A persistent kernel that gives every workgroup `ceil(tiles / NCU)` tiles rounds its resident workgroups down when the
  tile count falls between two whole CU passes (S6).

## Likely causes

1. **Too few tiles:** every workgroup walks all of K serially, so latency follows K, not M (S7).
2. **A ragged last round:** CUs idle through it (S8); "≥ 1024 workgroups and a multiple of 8" does not prevent it (S9).
3. **Fewer slots than assumed.** `wg_per_cu` comes from registers, LDS and wave slots (S10); a register-heavy GEMM
   is usually at 1, so `slots = 256` (S11). Occupancy moves in workgroup-sized steps (S12).
4. **A padded grid** that spills into one more round (S13).

## Rule out first

- **A streaming row.** Fewer tiles than CUs is necessary for starvation, not sufficient: a cold, read-once B stream
  can be HBM-bound with fewer blocks than CUs, and then more blocks do not pay (split-K and narrower tiles tie or lose)
  (S14) *(field, E1)*. Compare the row with a plain streaming read of the same bytes before ruling more blocks out:
  where split 1 read B well below it, split-K won *(field, E2)*; see `memory_bound.md`.
- **An idle-device ruler.** A single-shape tuner on an idle GPU overvalues split-K when the real step keeps the CUs
  the split would borrow busy with concurrent work (S15) *(field, E3)*, and it has also undervalued a tail split by
  letting the under-filled arm boost its clock (S16); measure CU-fill levers under the real load mix.
- **The compiler's occupancy comment.** `; Occupancy:` does not apply the workgroup quantization (S17).
- **The operating point.** Benchmark the deployment's per-call M and check the launched grid in a trace (S18).
- **The clock.** Turbo's pricing assumes that work freed on only some CUs returns less than it looks, because the
  idle CUs come back as clock for the others (S19). Whether they do depends on the run: an isolated loop let an
  under-filled arm boost its clock, while under the real mix both arms ran at one clock (S16); read each arm's clock.

## Candidate techniques

- Narrow the tile so the last round fills — `gemm/grid_pricing.md` §1, §9 — price it by rounds (S20), then race it: a
  narrow tile that needed an extra round usually lost, but won once where the wide grid's last round was nearly empty
  (S20); it costs reuse (S21).
- Split-K — `gemm/grid_pricing.md` §2 — gated on wave quantization (S22), with `s = 1` kept in the race
  (S23); no split wins when the reduce costs more than the idle time it recovers (S24). Atomic split-K is not
  bitwise reproducible (S25): in AKLR reduce in a fixed order (S26).
- Tail split — `gemm/grid_pricing.md` §2 — split only the tiles of the ragged last round (S27), with
  each launch whole rounds by itself (S28).
- Persistent tiles — `gemm/grid_pricing.md` §10 — passing the gate is not a win: race it (S29).
- Small M — `gemm/small_m.md` §3–4 — count workgroups against the 256-CU round first (S30); split-K
  where narrow N leaves a single N-block (S31) *(field, E2)*, not on a row whose split-1 kernel already reads B near
  HBM bandwidth (S32) *(field, E1)*; on short K, race it with the zeroing or reduction launch counted *(field, E2)*.
- Idle CUs load later K-tiles of B into L2 — `gemm/small_m.md` §6 — when the grid leaves CUs idle;
  race the first prefetched tile per M (S33).
- Other kernel families: attention's XCD-major mapping and live `(batch, kv-head)` slices per XCD —
  `attention/grid_dispatch.md` §2, §7; split-KV grid size and split rules —
  `attention/split_kv.md` §4–5; grouped GEMM skew, longest-first and persistent grids —
  `moe/grouped_scheduling.md` §1, §5–6; MoE grids that fill only a few M-blocks —
  `moe/grouped_sorted_tokens.md` §3.

## Evidence

- E1 [2026-11, gfx950 (MI355X), HIP 7.15, N=15, CUDA graph over 4 copies of B (805 MB), median per arm, threshold 2.1%] M = 8, N = 6144, K = 16384, 192 blocks reading B at 0.87 of a plain read of the same copies: split-K 2 +12.6% and 16-wide N tiles +13.9% time, split-K 4 +0.6%, a tie (hand-written bf16 MFMA GEMM, fp32 atomic output)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=9, CUDA graph over copies of B rotated through ≥ 768 MB (cold B), arms shuffled] best split-K against the best split-K 1 (M×N×K): single N-block 1×32×6144 and 8×64×8192 −81% to −88%; latency-bound 8×1024×1536 and 16×512×2048 −59% to −61%; 8×6144×16384 and 16×4096×14336, whose split 1 read B at 0.70 and 0.54 of a plain read, −14% to −41%; K = 512: 64×512×512 −16.6% without a `C.zero_()` node and +51% with it, 64×3584×512 split-K 2 +6.8% even without (Triton split-K GEMM, bf16 in)
- E3 [2026-11, gfx950 (MI355X), HIP 7.15, N=7, load conditions alternated by round] split-K 2 against split-K 1 on a 128-tile grid (128×4096×8192, 64×8192×2048): 47.6% and 40.8% less time on an idle GPU, 0.1–0.4% slower with a kernel on a second stream holding the other 128 CUs (hand-written bf16 MFMA GEMM, fp32 atomic output)
- S1 gemm/grid_pricing.md:46 tiles = ceil(M/tile_m) · ceil(N/tile_n) · split_k
- S2 gemm/grid_pricing.md:49 last_round = tiles − (rounds − 1) · slots
- S3 gemm/grid_pricing.md:47 slots = NCU · wg_per_cu; NCU = 256 on gfx950, query it
- S4 gemm/grid_pricing.md:112 N=6016 at 128×128 tiles and `split_k=1` launches 47 workgroups for every M ≤ 128
- S5 gemm/grid_pricing.md:113 read the launched grid from a trace before reading any counter
- S6 gemm/grid_pricing.md:378 with tiles_per_wg = ceil(tiles / ncu), a tile count between two CU passes rounds the resident count down
- S7 gemm/grid_pricing.md:71 tiles < NCU: every workgroup walks all of K; latency set by K, not M
- S8 gemm/grid_pricing.md:73 last_round < slots: CUs idle through the last round
- S9 gemm/grid_pricing.md:116 "≥1024 workgroups and a multiple of 8" is not sufficient
- S10 gemm/grid_pricing.md:53 wg_per_cu from registers, LDS and wave slots
- S11 gemm/grid_pricing.md:65 a register-heavy GEMM is usually 1 workgroup per CU, slots = 256
- S12 gemm/grid_pricing.md:245 reachable occupancy is quantized by workgroup size
- S13 gemm/small_m.md:99 padding a grid turned one dispatch round into two
- S14 gemm/small_m.md:105 fewer tiles than CUs is necessary for starvation, not sufficient
- S15 gemm/small_m.md:73 a single-shape tuner on an idle GPU overvalues split-K when concurrent work holds the CUs it borrows
- S16 gemm/measurement.md:79 measure CU-fill levers under the real load mix
- S17 gemm/grid_pricing.md:262 the `; Occupancy:` comment does not apply the quantization
- S18 gemm/measurement.md:154 benchmark the deployment's per-call shape; check the grid in a trace
- S19 gemm/grid_pricing.md:370 work freed on only some workgroups returns less than it looks; Turbo counts idle CUs as coming back as clock (`_nt4_wg_path`)
- S20 gemm/grid_pricing.md:359 a narrow tile usually pays only where its extra tiles fit the same rounds; 256×128 in three rounds once beat 256×256 in two
- S21 gemm/grid_pricing.md:120 a narrower tile also costs reuse
- S22 gemm/grid_pricing.md:126 split-K gate: propose s only if it raises eff by a margin
- S23 gemm/grid_pricing.md:127 keep s = 1 in the candidate set
- S24 gemm/grid_pricing.md:149 if the reduce costs more than the idle time it recovers, no split wins
- S25 gemm/grid_pricing.md:168 atomic split-K is not bitwise reproducible once an output receives three or more partials (two on a C not zeroed first)
- S26 gemm/correctness.md:227 under AKLR's bitwise contracts, reduce the splits in a fixed order
- S27 gemm/grid_pricing.md:182 tail split: split only the tiles of the ragged last round
- S28 gemm/grid_pricing.md:184 head and tail launches must each be whole rounds
- S29 gemm/grid_pricing.md:387 passing the persistent gate is not a win: race it
- S30 gemm/small_m.md:99 count workgroups against the 256-CU dispatch round before anything else
- S31 gemm/small_m.md:134 split-K pays on a single N-block (narrow N) and latency-bound rows
- S32 gemm/small_m.md:135 split-K does not pay on streaming rows with many N-blocks or short K
- S33 gemm/small_m.md:376 idle CUs load later K-tiles of B into L2
