---
title: "FlyDSL grouped GEMM — tile, occupancy and boundary bodies per pass, priced before building"
kind: topic
lever: flydsl_grouped_tile_pricing
ops: [grouped_gemm]
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, variable-k, small-m, ragged-tail, register-saturated]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2487
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2553
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:3921
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1689
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1716
---

# Pricing the tile, the occupancy and the boundary bodies of a FlyDSL grouped GEMM

> Grouped GEMMs on gfx950 reuse the dense 256×256 tile, and where it was measured their main loop ran at dense
> speed. What the group dimension changes is the price of everything around the loop: more tiles per FLOP,
> more ragged tiles, a per-tile prologue that looks up the group. This card prices the tile and the waves per
> SIMD per pass, the small-M dgrad tile, boundary half tiles and tail bodies, and the per-tile fixed cost.
> Round counting, split-K, the generic fixed-cost fit and the carry cost of a body variant are in
> `../gemm/grid_pricing.md`; register classes and the K loop are in
> `../gemm/mainloop.md`. "FP8" and "MX4" are Primus-Turbo's
> `primus_turbo/flydsl/grouped_gemm/grouped_gemm_{fp8,mxfp4}_kernel.py` at main `ca1fdd67`.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end. Use
> them for direction and order of magnitude, not as targets.

## Route here when
- You want a bigger tile, more waves per SIMD, or a smaller tile for a short pass.
- A dgrad with few tokens per group is far from roofline.
- Output dimensions are not multiples of 256 and the last tile of a group or of N is mostly padding.
- A grouped kernel runs its main loop at dense speed and still trails dense by 10% or more.

**Somewhere else if:** the lookup and the dead workgroups that add to the per-tile cost →
`flydsl_grouped_group_mapping.md`; tile order and tiles per workgroup → `flydsl_grouped_scheduling.md`;
short-contraction wgrad → `flydsl_grouped_wgrad.md`; MX scale layout → `flydsl_grouped_lowprec.md`.

## 1. The register wall comes before LDS
256×256 is the largest tile that fits, for both large-tile loop designs (`../gemm/mainloop.md` §1):
- **8 waves, 2 waves per SIMD** (one 512-thread workgroup per CU, at most 256 registers per wave). *(field, E1)* bf16
  with a 2×4 wave grid: 128 accumulator VGPRs plus ~96 operand VGPRs is 224 (228 measured); doubling either
  tile side needs ~352. *(field, E2)* MXFP8 NT sat at 246–254 of 256 VGPRs depending on the build, with no spill: a 16-wave
  version of the same tile would need four such waves per SIMD, about 1000 registers against 512.
- **4 waves, 1 wave per SIMD** (one 256-thread workgroup per CU). The fp32 accumulators of a tile are
  `BM × BN ≤ 4 waves × 64 lanes × 256 AGPRs = 65536`: 256×256 is exactly the AGPR file. An MXFP4
  BM = 384 would need ~620 registers per lane (estimated) against 512. In the measured MXFP4 grouped kernel BM must also
  be a multiple of 128 because of its scale-slab layout (`flydsl_grouped_lowprec.md` §4).
- **So the operand fill cannot shrink by tiling.** A `buffer_load … lds` costs the 128-byte lines it covers
  (`../gemm/datapath.md` §6); lines per unit of MFMA work go as `(BM + BN) / (BM · BN)` and do not
  depend on BLOCK_K. Only a larger tile lowers them, and the tile is at the wall. *(field, E3)* `BLOCK_N = 128` on
  an MXFP4 whole-loop NT body, tried to cut N padding, timed 4.9–10.2% slower on a build whose output was not
  yet correct: halving the N tile halves the reuse of each A tile, which cost more than the padding saved.
- **At the wall, anything that lengthens a live range spills.** *(field, E4)* MXFP8 NT at 246/256 arch VGPRs:
  folding the store earlier spilled 18 VGPRs (19% slower), a tile loop 25 (26% slower), holding an A fragment
  across iterations 44 (about 50% slower), each in an isolated timing. A second heavy body in the same kernel
  compiled to 256 VGPRs with 23 spills in the MFMA stream's accumulators; two copies of the same body did not
  spill, one full plus one paired body did.
- **Gate on a compile-only ISA dump** (20–60 s *(field, E5)*): `.vgpr_count`, `.agpr_count`, `vgpr_spill_count`,
  `group_segment_fixed_size`, and the `s_barrier` count. Do not time a candidate that spills
  (`../gemm/mainloop.md` §10).

## 2. Waves per SIMD: decide per pass
- On fp8 per-tensor, the 4-wave design (1 wave per SIMD) won every variable-K wgrad point against 8 waves (2
  per SIMD), a minority of the NN dgrad points and almost no forward points (`flydsl_grouped_wgrad.md` §8). It
  pays where both operands are transpose-read (`ds_read_b64_tr_b8`; mechanism in `flydsl_grouped_wgrad.md` §8);
  the forward pass reads both operands K-contiguous.
- The fastest grouped kernel measured is that wgrad: 4 waves, 1 wave per SIMD, 256 AGPR accumulators, a full
  160 KB of LDS, 0.13 synchronization instructions per MFMA against 0.93 for an 8-wave MXFP8 NT *(field, E6)*.
- **1 wave per SIMD with AGPR accumulators is a direction to test, not a rule.**
  - *(field, E7)* Moving a kernel to 4 waves while keeping the 8-wave phase skeleton (6.4 `s_barrier` per K
    iteration, `s_setprio`) lost 31% and 41%. Compile-only gate: at most 3 `s_barrier` per K iteration and no
    `s_setprio`; if the gate fails, the attempt says nothing about 4 waves.
  - *(field, E8)* An MXFP8 grouped kernel at 32 groups of 4096 rows, moved to AGPR-tied accumulators (arch VGPRs
    246 → 184, no spill), measured 0.03–0.17% slower: no gain in that regime.
  - Freed registers buy nothing by themselves (`../gemm/mainloop.md` §4).
- **On MXFP4 NT, more waves per SIMD by splitting the tile lost.** *(field, E9)* An 8-wave kernel at 2 waves per
  SIMD ran at 0.83–0.98× of the 4-wave whole loop when built as a tiled software pipeline and 0.78–0.92× when
  built as a whole loop; in the whole-loop case MFMA busy cycles were identical and MFMA utilization fell from
  57.3% to 48.2%. Each wave's MFMA stream halves, doubling the share of `ds_read` → MFMA bubbles, and the
  waves contend for the same LDS read port. Turbo keeps its MXFP4 wgrad at one wave per SIMD (`OCC = 1`, MX4:1689).
- **The 8-wave NT barriers that look removable are load-bearing for speed.** On MXFP8 NT the pre-MFMA rendezvous
  added no ordering the gates could see (the barrier after the MFMAs guards LDS reuse), but it kept the waves' MFMA
  bursts in phase; without it they drift and contend for issue and LDS. Whether an opening rendezvous orders
  anything differs per kernel (`flydsl_grouped_correctness.md` §8). *(field, E10, E11)* MXFP8 NT: removing the pre-MFMA
  rendezvous barriers was 3.3% slower overall (2.6–6.9% on all 8 NT configs), removing a steady-state barrier 6.1%
  (worst config 8.7%), merging MFMA phases 1.6%, all with SNR and determinism intact. Before deleting a barrier, ask
  whether it is an ordering constraint (`../gemm/mainloop.md` §2) or a phase-alignment device. Deleting
  the rendezvous of the last K steps, which issue no new fill but still make the fills issued two steps earlier
  visible across waves, raced (`flydsl_grouped_correctness.md` §8).

## 3. Small-M dgrad: when BLOCK_M = 128 pays
- With few tokens per group a dgrad grid of 256-row tiles leaves CUs idle. Halving BLOCK_M doubles the M tiles,
  and it pays only while the whole 128-row grid still fits in one round:
  `G · ceil(pm / 128) · ceil(N / 256) ≤ N_CU` with `pm = M_total // G` (the host sees no per-group length).
  Turbo's fp8 dgrad takes a fixed BLOCK_M = 128 config inside that gate and races 256-row candidates outside
  it (`_np_regime`, FP8:2487-2494; FP8:2553-2555).
- *(field, E12)* Past the gate it falls off a cliff: on fp8 NN dgrad, BLOCK_M = 128 lost once its grid needed 320
  tiles on 256 CUs, and at about three times that M the 256-row tile was 58% faster. *(field, E13, E14)* At production sizes
  BLOCK_M = 128 was 21–25% slower on fp8 NN dgrad and 25–35% slower on small fp8 forward shapes.
- Recompute the grid bound from BLOCK_M. *(field, E15)* An MXFP8 NT sweep whose bound was hard-coded for 256 rows
  launched half the tiles at BLOCK_M = 128 and read 9–35% "faster"; with `ceil(M_pad / BM) + G` it was 1.55×
  slower than the per-tensor reference kernel (1666 vs 2576 TF/s). Before believing a tile change, show the
  full grid ran (`flydsl_grouped_correctness.md`).

## 4. Boundary half tiles: what to delete, what to keep
When `OUT % BLOCK ≠ 0` and `OUT % BLOCK ≤ BLOCK / 2`, the back half of the last block is padding: its
quadrants' MFMAs and LDS reads are dead. Pick the variant with a scalar branch on `readfirstlane(block == last)`.
- **Start by deleting only `ds_read` + MFMA** and keep every g2s, `s_barrier` and `vmcnt` identical, so there
  is no new race surface. Stores of the skipped quadrants can stay unconditional; the descriptor drops them.
  *(field, E16)* MXFP8 wgrad: 6.25–8.16% of the MFMAs on paper, +1.3–4.1% measured (about half: barriers, fills and
  LDS round trips stay). NT, N side only: the worst config +1.9%.
- **Count on 20–50% of the MFMAs saved.** *(field, E17)* fp8 wgrad: a short-N body that saved 6.25% of the MFMAs
  returned 0.6–1.2%, because the L2 hit rate fell 6.4 points and MFMA-busy 5.4 points.
- **In a whole-loop asm kernel, keep each variant's register footprint.** Variants that pin register windows
  share one allocation. *(field, E18, E19)* fp8 wgrad: one variant that dropped a dead operand pool (and its `ds_read`s)
  on 12 of 276 tiles made the whole kernel spill 16 VGPRs and the gate_up wgrad 14.6% slower (−2.8% on the
  geomean over all 14 configs). On MXFP4, deleting
  MFMAs plus one g2s group kept the spill count at 0. Check the spill count after adding any variant.
- **Deleting the dead half's g2s: lower `vmcnt` by the number deleted, when those loads are the youngest in
  flight.** The in-flight allowance is "loads issued per iteration minus the earliest group".
  - *(field, E20, E21)* MXFP4 whole loop: `vmcnt(WLV − n_deleted)`, `WLV` being the full body's allowance, on the half-N body,
    +3.8% geomean, +4.7% worst config. MXFP8 NT (all pools prefetched two iterations ahead, the deleted pool issued
    last): drain 6 → 4, the down configs 0.985 → 1.005 against the reference.
  - *(field, E22)* The same surgery on MXFP8 wgrad, where the deleted pool is one iteration ahead and issued first:
    0.09% slower.
  - *(field, E23, E24)* fp8 NN dgrad (8 waves): the reduced counts raced at K = 1024. Keeping the loads and deleting only
    MFMAs and stores reused the full body's graded drain and beat the fully drained no-load body by 0.5 points.
    Prove any changed count by repetition at small K and wide N (`../gemm/correctness.md` §3).
- **The barrier pair around an empty phase can go with its quadrant.** *(field, E25, E26)* NT half-N body: 8 → 5
  barriers per K iteration, +1.0–1.5% on the down configs; on MXFP8 wgrad at 32 groups, the same change left the score
  flat (+0.01%) while the wgrad configurations read slower (geomean 1.1370 against 1.1393; the heavy gate_up one 1.320
  against 1.342–1.353).
- **The two M halves of a tile sit on different waves.** Idling `wave_m == 1` does not shorten the tile;
  re-assign the waves to live rows instead (`../gemm/grid_pricing.md` §8). *(field, E27)* MXFP4 wgrad:
  +0.4–3.1% on the deployment-like configs (median +1.7%). Turbo's MXFP4 wgrad serves a block ragged in both
  dimensions with the N re-tile (MX4:1716-1722).
- *(field, E28)* fp8 per-tensor wgrad, last M block with its upper `a` half all padding, `a_halves = 1`: +4.5–5.8%.
- **Is the boundary tile on the makespan?** A cheap boundary tile on wgrad finishes early, which is why the
  barrier trim above did nothing there. The M-side bodies above were measured at 18–35 dispatch rounds, while
  Turbo's fp8 wgrad keeps its M-side boundary body only below 8 rounds: "a cheap tile frees its CU early and
  desyncs the L2-slab phase cohorts (no resync at occ=1) -- pays off shallow, not deep" (FP8:3921-3925;
  `_WGRAD_AFF_ROUNDS`, FP8:4660). Measure at the deployed depth.
- **Leave boundary tiles where they are.** *(field, E29, E30)* Moving the half-price tiles (an N-first band walk, or the ends
  of the groups) cost 0.8–1.0 points of band locality. Peeling the boundary strip into a second dispatch was 2.9%
  slower overall and 4.6% on the worst config (MXFP8 NT): the strip loses the A slab it shared in L2. That is about
  edge strips. Running a ragged last dispatch round as a second launch of smaller tiles is a different lever, which
  won and lost on dense GEMMs (`../gemm/grid_pricing.md` §1–§2); *(field, E31)* under AKLR's timer a second
  launch after a dense GEMM costs ~4 µs (`../gemm/small_m.md` §1).

## 5. Tail bodies: compute the hit rate first
The carry cost of a body variant (code size paid by every shape in the module) is in
`../gemm/grid_pricing.md` §8. What the group dimension adds is that the hit rate depends on the
pass and on the token histogram:
- A ragged wgrad output (`OUT_M`, `OUT_N`) is a compile-time property: every call hits the tail body.
- A ragged forward/dgrad M tail exists per group and depends on the group lengths. *(field, E32)* Under the scored
  skew 19 of 529 M blocks (3.59%) had a remainder ≤ 128 rows; under balanced tokens, none. With the tail tile
  an estimated ~17% faster the estimated bound was 0.6–0.7 points on each of two cells. Measured, the two skewed cells gained 0.44
  and 0.30 points, the two balanced cells of the same module lost 0.31 and 0.34 to the carry, and the
  geomean moved −0.13%.
- Two tail bodies with 32 and 64 MFMAs per phase both made their tail tile an estimated ~17% faster: the saving is
  the tail tile's fixed cost, not its MFMAs.
- *(field, E33)* Disabling the half-N body of an MXFP4 kernel cost 1.79 points of the 8-cell geomean (NT and wgrad
  cells, 7 of 8 in the same direction); re-tiling it to quarter-N gained another 0.67–1.09 points per NT cell.

## 6. Short tiles: where grouped loses to dense
Fit `T_tile = F + n · P` from a K sweep, n K iterations at P each (`../gemm/grid_pricing.md` §6, which
writes it `a · n_kblocks + F`; its pitfalls:
`../gemm/attribution.md` §7). For grouped kernels the answer has been the same every time:
- *(field, E34)* MXFP4 grouped: P = 1.565 µs per K phase against an estimated 1.59 µs derived from the dense MXFP4 kernel's
  5405 TF/s. The steady state ran at dense speed; the whole loss against dense was in F = 7.81 µs per tile.
- *(field, E35)* MXFP8 NT against the fastest fp8 wgrad: 6192 tiles of 23 K iterations against 1008 of 134; F was
  15.7% of the wall against ≤3.7%, while MXFP8's P (1.164 µs) beat the reference's (≤1.35). Pushing F's
  share to the reference's level predicts the wall 779 → 673 µs, an estimated ratio of 1.06 — enough on its own. The
  same table carried the control: two of the three cells with 45 K iterations per tile passed, none of the
  nine with 23.
- *(field, E36)* That F split into launch 1.39, group decode 0.85, C store 2.37 and cold fill 0.80 µs.
- What the group dimension puts into F: the group lookup (`flydsl_grouped_group_mapping.md` §5), dead
  workgroups (§3 there), ragged tiles per group (§4 here), and a cold L2 for each group's slab
  (`flydsl_grouped_scheduling.md` §9).
- So the levers are F (lookup, decode, store overlap, boundary bodies, kernel-argument preload:
  `../gemm/small_m.md` §6) or longer tiles per workgroup (`flydsl_grouped_scheduling.md` §6–§7), not
  the main loop.
- For wgrad the grid is `G × tiles per group`, so G sets the last-round fill: 144 tiles per group fill 18.0
  rounds of 256 CUs at G = 32 and 13.5 at G = 24. Count rounds for every G you deploy
  (`../gemm/grid_pricing.md` §1).

## 7. Store-bound short contractions
A variable-K wgrad writes G full output slices whatever the token split, so groups with few tokens make it
store-bound. The short-contraction blocking with non-temporal C stores, and its crossover, which scales with
tiles per group rather than with a token count, are in `flydsl_grouped_wgrad.md` §6.

## Verify
- The compile-only dump shows no spill and the expected barrier count before any timing.
- The waves-per-SIMD choice is per pass, with the 4-wave sync gate passed before a 4-wave result is read.
- A small-M tile is used only inside the one-round gate, and the grid bound follows BLOCK_M.
- Every boundary variant keeps the race surface (or proves its changed counts at small K and wide N) and the
  spill count, and its hit rate is computed on the scored distribution.
- A K sweep separates F from P before work on the main loop.

## Related
`flydsl_grouped_group_mapping.md` · `flydsl_grouped_scheduling.md` · `flydsl_grouped_wgrad.md` ·
`flydsl_grouped_lowprec.md` · `flydsl_grouped_correctness.md` · `flydsl_grouped_measurement.md` ·
`../gemm/grid_pricing.md` · `../gemm/mainloop.md` · `../gemm/datapath.md` §6 ·
`../gemm/attribution.md` §7

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The Kimi-K3 forge run of E31 installed `flydsl==0.3.2`. The month is the run's where the record dates it;
where it does not, the line says which date it is. "Turbo" is Primus-Turbo; "gm" is a campaign's geomean score; gpt-oss
shapes are G=32 experts, 131072 routed tokens, hidden 2880 (padded to 2944).

- E1 [2026-08, gfx950, FlyDSL ?, N=?, ISA dump] bf16 256×256 tile on 2×4 waves: 8 accumulators × 16 VGPRs = 128 plus ~96 operand VGPRs = 224, measured vgpr 228; doubling either side ~352 by the same count (Turbo bf16 grouped GEMM, gpt-oss; record dated 2026-08-24; local:/workspace/code/gpt_oss_docker/.claude/memory/project_bf16_flydsl_grouped_campaign.md:528)
- E2 [2026-07, gfx950, FlyDSL ?, N=?, ISA dump] `kernel_grouped_mxfp8_nt_1`: num_vgpr 254, num_agpr 0, spill 0 (campaign 20260729_093030 round 12, as myskill records it); a 2026-08 build of the same kernel sat at 246/256 with 10 VGPRs to spare (`/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/memory.md:172`) (Turbo MXFP8 grouped NT, gpt-oss; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/01-tile-occupancy-register.md:211)
- E3 [2026-07, gfx950, FlyDSL ?, N=?, ?] `BLOCK_N = 128` on the MXFP4 whole-loop NT, to cut N padding: SNR 0 (the bare-asm layout assumed 4 N tiles) and −4.9 to −10.2% in time; record dated 2026-07-21 (Turbo MXFP4 grouped NT, gpt-oss; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_fwd_autotune.md:32)
- E4 [2026-08, gfx950, FlyDSL ?, N=?, compile-only ISA gate, then isolated timing probes against a baseline of about 0.835 ms] at 246/256 arch VGPRs: each quadrant's store moved right after its last MFMA phase, vgpr 256 / spill 18, 0.9951 ms (−19%, round 7); one workgroup walking 2 tiles, spill 25, 1.0528 ms (−26%, round 7); an A fragment kept live across the iteration, vgpr 256 / spill 44, 1.2504 against 0.8365 ms (−50%, round 6); the round-7 summary lists the three as +24 / +25 / +44 and −19 … −26%, its 24 being the VGPRs still live at the store fold point; one body 244 / spill 0, two identical bodies 244 / 0, one full plus one paired body 256 / spill 23 (round 14) (Turbo MXFP8 grouped NT, gpt-oss; campaign 20260805_150542; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/memory.md:169; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/memory.md:157; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/memory.md:260)
- E5 [2026-08, gfx950, FlyDSL ?, N=?, wall time of a compile-only ISA build] 20–60 s per ISA gate (campaign 20260805_150542; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/memory.md:210)
- E6 [2026-08, gfx950, FlyDSL ?, N=?, ISA instruction census of both kernels] synchronization and scheduling instructions per MFMA: 0.933 for the 8-wave MXFP8 NT against 0.126 for `kernel_grouped_tn_wgrad_4wave_0` (4 waves, 1 wave per SIMD, LDS 163840, 256 AGPR accumulators) (Turbo MXFP8 grouped NT against the fp8 per-tensor grouped wgrad, gpt-oss; campaign 20260805_150542 round 16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/goal.md:145)
- E7 [2026-08, gfx950, FlyDSL ?, N=2, campaign bench] 4-wave geometry with the 8-wave synchronization skeleton (6.39 `s_barrier` per K iteration, `s_setprio`): −31% (round 4) and −41% (round 16) (Turbo MXFP8 grouped NT, gpt-oss; campaign 20260805_150542; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/goal.md:447)
- E8 [2026-08, gfx950, FlyDSL ?, N=?, campaign bench, one reading per round] 4 waves with AGPR-tied accumulators (compile gate: arch VGPRs 246 → 184, spill 0): −0.03 / −0.10 / −0.16 / −0.17% in rounds 8–11 (Turbo MXFP8 grouped GEMM, G=32, 4096 tokens per expert; record dated 2026-08-17; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mx8tw_bal_campaign_and_mxfp4_pr.md:26)
- E9 [2026-07, gfx950, FlyDSL ?, N=?, steady-state median on two GPUs; rocprofv3 PMC at M=N=K=4096] 8-wave tiled software pipeline 3246–3641 TF/s = 0.83–0.98× the 4-wave whole loop (3703–4266); 8-wave whole loop: MFMA busy cycles identical (612.0B), MFMA utilization 57.3% → 48.2%, wall +19%, 0.78–0.92× (Turbo MXFP4 grouped NT; record dated 2026-07-22; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_fp4vs8_occ.md:39; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_fp4vs8_occ.md:46)
- E10 [2026-07, gfx950, FlyDSL ?, N=1, campaign score: geomean of t_tw / t_mx over 14 configurations, kernel-only] pre-MFMA rendezvous barriers removed (8 → 5 per K iteration): gm 1.06572 → 1.03039 (−3.3%), all 8 NT configurations −2.6 to −6.9%, SNR and determinism passing; MFMA phases merged (8 → 4 barriers per K iteration): 1.0610 → 1.04424 (−1.6%) (Turbo MXFP8 grouped NT, gpt-oss; campaign 20260729_093030 rounds 4 and 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/rounds/round-04/agent.txt:13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:168)
- E11 [2026-08, gfx950, FlyDSL ?, N=?, campaign bench] the steady-state barrier after c11 removed: gm −6.1%, min −8.7% (Turbo MXFP8 grouped NT, gpt-oss; campaign 20260805_150542; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/goal.md:525)
- E12 [2026-07, gfx950, FlyDSL ?, N=?, ?] dgrad `bm128` small-M branch: at M ≥ 1280 (N=8192) the 128-row grid is 320 tiles > 256 CUs and `bm128` loses; at M=4096 `bm256` +58%; no run record is kept, and the month is that of the myskill commit that holds it, 99db6ee (2026-07-05) (Turbo FlyDSL fp8 per-tensor grouped dgrad; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/06-autotune-dispatch.md:140)
- E13 [2026-08, gfx950, FlyDSL ?, N=?, isolated A/B at the deployed 4096 tokens per expert, outputs bitwise identical] NN dgrad `BLOCK_M=128`: −20.87% (`bm128_g8` −22.77%); a hands-on session of 2026-07-30 recorded −21 to −25% (palindrome A/B, byte-exact, `/workspace/code/gpt_oss_docker/.claude/memory/project_tw_nn_beats_mxfp8_campaign.md:38`) (Turbo fp8 per-tensor grouped NN dgrad, gpt-oss down projection; campaign 20260817_065611 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/goal.md:559)
- E14 [2026-07, gfx950, FlyDSL ?, N=?, ?] `BLOCK_M=128` on small forward shapes −25 to −35% (Turbo FlyDSL fp8 per-tensor grouped forward; record dated 2026-07-04; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_grouped_hipblaslt_gap.md:35)
- E15 [2026-07, gfx950, FlyDSL ?, N=?, gemm-only] grouped MX config sweep with `grid_upper` hard-coded for 256 rows: `BLOCK_M=128` read +9 to +35%; with `ceildiv(M_pad, bm) + G` it was 1.55× slower (1666 against the per-tensor kernel's 2576 TF/s); no run record is kept, and the month is that of the myskill commit that holds it, 9d3f961 (2026-07-05) (Turbo MXFP8 grouped GEMM; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:285)
- E16 [2026-07, gfx950, FlyDSL ?, N=1, campaign score of E10] wgrad M+N boundary quadrant skip: gm 1.05231 → 1.06572 (+1.27%), the six wgrad configurations +1.3–4.1% (down +3.6–4.1%, gate_up +1.3–3.1%) against 8.16% / 6.25% of the MFMAs on paper; NT N side only: min 0.97303 → 0.99180 (+1.9%) (Turbo MXFP8 grouped GEMM, gpt-oss; campaign 20260729_093030 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:130; paper counts myskill pitfalls/05:386)
- E17 [2026-08, gfx950, FlyDSL ?, N=1, one rocprofv3 pass with 5 counters] the down short-N remainder body: MFMA −6.25%, wall +0.6–1.2%, L2 hit −6.4 points, MFMA busy −5.4 points; no run record is kept, and the month is that of the myskill commit that recorded it, 523b563 (2026-08-13) (Turbo fp8 per-tensor grouped wgrad, 32 experts at 18 / 34.5 dispatch rounds; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/03-lds-l2-datapath.md:173)
- E18 [2026-08, gfx950, FlyDSL ?, N=1, campaign score `_bench_tw_vs_mx.py` (geomean of t_mx / t_tw over 14 configurations)] the dead b1 pool deleted from the (2,1) variant only: `vgpr_spill_count` 0 → 16, gm 0.9847 → 0.9573 (−2.8%), wgrad gate_up −14.6% (dead-pool tiles 12 of 276) (Turbo fp8 per-tensor grouped wgrad, 4-wave whole loop, gpt-oss; campaign 20260731_073404 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260731_073404/goal.md:360)
- E19 [2026-08, gfx950, FlyDSL ?, N=?, ISA dump] MXFP4 half-N body deleting MFMAs and one BR g2s group: vgpr 428 → 432, agpr 256, spill 0 (Turbo MXFP4 grouped GEMM, gpt-oss; campaign 20260803_155024 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/memory.md:168)
- E20 [2026-08, gfx950, FlyDSL ?, N=3, campaign score gm_ratio: geomean over 18 configurations of t_MXFP8 / t_MXFP4] the half-N body's deleted g2s group with the partial vmcnt tightened by its count (`_WLV − nsb`): gm +3.8%, min +4.7%, deterministic (Turbo MXFP4 grouped NT; campaign 20260803_155024 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/memory.md:170)
- E21 [2026-07, gfx950, FlyDSL ?, N=2, campaign score of E10] half-N body also dropping the B-side g2s with the drain 6 → 4: gm 1.07080 / 1.07084, min 1.00275 / 1.00288; myskill gives fwd down balanced 0.985 → 1.005 and dgrad down balanced 0.988 → 1.006 (Turbo MXFP8 grouped NT, gpt-oss; campaign 20260729_093030 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:168; myskill methodology/12:93)
- E22 [2026-07, gfx950, FlyDSL ?, N=1, campaign score of E10] the same surgery on wgrad (its A1 pool at distance 1, issued first): 1.06991 (−0.09%), the six wgrad configurations flat to −1.6% (campaign 20260729_093030 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:168)
- E23 [2026-07, gfx950, FlyDSL ?, N=?, determinism probe] the NN dgrad half-N no-load body with derived reduced wait counts raced at K=1024, 4096 tokens per group (SNR 75.8, not deterministic) (Turbo fp8 per-tensor grouped NN dgrad; hands-on session recorded 2026-07-30; local:/workspace/code/gpt_oss_docker/.claude/memory/project_tw_nn_beats_mxfp8_campaign.md:26)
- E24 [2026-08, gfx950, FlyDSL ?, N=1, campaign score `_bench_tw_vs_mx.py`] `nn_halfn_noload=False` (loads kept, MFMAs and stores deleted, against the fully drained no-load body): gm 0.99372 → 0.99600, dgrad 0.9890 → 0.9940 (Turbo fp8 per-tensor grouped NN dgrad, gpt-oss; campaign 20260731_073404 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260731_073404/memory.md:211)
- E25 [2026-07, gfx950, FlyDSL ?, N=?, ?] NT half-N body 8 → 5 barriers per K iteration: fwd/dgrad down balanced +1.0–1.5%, gm +0.13%; dated 2026-07-29 in the note (Turbo MXFP8 grouped NT; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:395)
- E26 [2026-07, gfx950, FlyDSL ?, N=1, campaign score of E10 against two baseline runs] the same gating on wgrad: gm 1.08500 against 1.08492 / 1.08459 (+0.01%), wgrad geomean 1.1370 against 1.1393, gate_up heavy 1.320 against 1.342 / 1.347 / 1.353 (Turbo MXFP8 grouped wgrad, gpt-oss G=32; campaign 20260729_093030 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:192)
- E27 [2026-08, gfx950, FlyDSL ?, N=5+3, campaign score (medians of 5 runs and of a 3-run same-tree control)] the M-side half tile with the waves re-assigned to live rows: gm 98.50 → 99.10; down wgrad 3921.6 / 3800.3 / 3840.3 → 4043.4 / 3872.6 / 3903.1 TF/s, gate_up 4078.2 / 3963.6 / 3929.4 → 4096.2 / 4033.5 / 3956.8 (+0.4 to +3.1%, median +1.7%) (Turbo MXFP4 grouped wgrad, gpt-oss deploy regime, 4096 tokens per expert; campaign 20260811_114814 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260811_114814/memory.md:198; myskill pitfalls/05:420-424)
- E28 [2026-07, gfx950, FlyDSL ?, N=?, drift-immune A/B with the new, old and ASM kernels interleaved per rep] `a_halves=1` on the last M block: gate_up balanced 1.111 → 1.164, skew 1.115 → 1.179; down 1.142 → 1.204, 1.106 → 1.155 (ratios against an ASM kernel, +4.5 to +5.8%) (Turbo fp8 per-tensor grouped wgrad 4-wave, gpt-oss; b0c02d64)
- E29 [2026-08, gfx950, FlyDSL ?, N=?, campaign score medians] moving the half-price tiles: an N-first band walk 98.03 against 99.07; the boundary tiles moved to the group ends −0.8 (Turbo MXFP4 grouped wgrad, gpt-oss deploy regime; campaign 20260811_114814 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260811_114814/memory.md:198; myskill pitfalls/05:426-428)
- E30 [2026-08, gfx950, FlyDSL ?, N=?, campaign score] a 512×128 tail strip as a second dispatch: gm 0.94075 against 0.96839 (−2.9%), min −4.6%; the extra launch measured 1.9 µs (Turbo MXFP8 grouped NT, gpt-oss; campaign 20260805_150542 round 15; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/memory.md:310; myskill pitfalls/05:512)
- E31 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver times, with a profiler split] on a dense GEMM, a one-workgroup second kernel after the unchanged GEMM took 4.12 µs and took m=7 from 10.48 to 14.43 µs (AKLR bf16 dense GEMM N=3584, K=7168, forge run iter 3; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3584_k7168/forge_experiments/lessons/iter_003.md:14)
- E32 [2026-09, gfx950, FlyDSL 0.2.4, N=5, campaign score: frozen-denominator geomean over 8 cells, each cell drift-corrected against 4 inert wgrad cells] a runtime M-tail body: B−A +0.44 (fc2_fwd@skew), +0.30 (fc1_dgrad@skew), −0.31 and −0.34 on the two balanced cells, gm −0.134 ± 0.045; under the scored skew 19 of 529 M blocks (3.59%) had a remainder ≤ 128 rows and balanced none (counted from a sampled distribution) (Turbo plain MXFP4 grouped NT, gpt-oss; campaign 20260925_090609 round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/rounds/round-08/agent.txt:11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/memory.md:303)
- E33 [2026-09, gfx950, FlyDSL 0.2.4, N=2, campaign score of E32] the half-N body disabled: gm 100.20 → 98.41 (−1.79, 7 of 8 cells the same direction); quarter-N: the four NT cells +0.67 to +1.09 in two runs (fc2_fwd +1.07 / +1.09, fc1_dgrad +0.67 / +1.07) (campaign 20260925_090609 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/memory.md:273; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/memory.md:277)
- E34 [2026-08, gfx950, FlyDSL ?, N=?, kernel trace of the main kernel only, six-point K sweep fit] F = 7.81 µs per tile, P = 1.565 µs per phase; the 1.59 µs is computed from the dense MXFP4 kernel's 5405 TF/s (Turbo MXFP4 grouped NT, gpt-oss; campaign 20260803_155024 round 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/memory.md:159)
- E35 [2026-08, gfx950, FlyDSL ?, N=?, one trace, arms interleaved; F from the four-part pricing of E36] forward down balanced: 6192 tiles of 23.0 K iterations against 1008 of 134.3; wall 779.3 against 712.7 µs; P 1.164 against ≤ 1.351 µs; F share 15.7% against ≤ 3.7%; in the 18-cell table the three n=45 cells read 0.984 / 1.032 / 1.013 (two passing) and the nine n=23 cells all ≤ 0.978 (Turbo MXFP8 grouped NT against the fp8 per-tensor 4-wave wgrad, gpt-oss; campaign 20260805_150542; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/goal.md:112; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/goal.md:69)
- E36 [2026-08, gfx950, FlyDSL ?, N=?, pricing probes kept out of the tree] F = launch 1.39 + group decode 0.85 + C store 2.37 + cold fill 0.80 µs (5.43 µs, closed to 0.4%) (Turbo MXFP8 grouped NT; campaign 20260805_150542; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/goal.md:284; probes local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/memory.md:169)
