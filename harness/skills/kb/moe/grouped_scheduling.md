---
title: "Grouped GEMM — ordering and placing tiles across groups"
kind: topic
lever: grouped_scheduling
ops: [grouped_gemm, fused_moe]
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: ["M split per group", transposed-operand, variable-k, skew, persistent]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py:688
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py:1639
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py:2694
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:102
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2024
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:3898
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:4211
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:4652
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:263
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:427
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:622
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:104
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:593
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1086
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1164
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1725
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1866
---

# Ordering and placing tiles across groups in a grouped GEMM

> Which XCD runs a tile, in which order the groups' tiles are dispatched, and how many tiles a workgroup walks
> decide how a grouped GEMM behaves when the token counts per group are skewed. The levers here are pure
> permutations of the tile index or of the grid; they leave every MFMA and every output byte unchanged, so
> they are cheap to try and easy to get backwards. XCD basics, 2D bands and bijection proofs are in
> `../gemm/schedule_and_locality.md` §2; dispatch order as a list schedule is §5
> below and `../attention/grid_dispatch.md` §4. "FP8", "BF16" and "MX4" are Primus-Turbo's
> `primus_turbo/flydsl/grouped_gemm/grouped_gemm_{fp8,bf16,mxfp4}_kernel.py` and "GH" is
> `primus_turbo/flydsl/utils/gemm_helper.py`, at main `ca1fdd67`.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end. Use
> them for direction and order of magnitude, not as targets.

## Route here when
- A grouped kernel is fast on balanced token counts and slow under skew, or the other way round.
- You are choosing an XCD remap, a band width or a dispatch order, or porting one from dense or another layout.
- You are placing the tiles of a MoE expert GEMM with several M-blocks per expert, or with thin experts
  (sorted-token specifics: `grouped_sorted_tokens.md` §2–§4).
- You are deciding between one tile per workgroup, a persistent grid and several tiles per workgroup.
- A grouped GEMM trails a dense GEMM of the same total size.

**Somewhere else if:** the device-side exit and lookup → `grouped_group_mapping.md` (exit on the raw
id and remap over the live count is §4 there); tile size and boundary bodies →
`grouped_tile_pricing.md`; split-K for deep groups → `grouped_variable_k.md`; ranking candidates on
balanced and skewed data → `grouped_host_autotune.md`.

## 1. First ask where the skew lives

| Layout | Skew changes | Order lever | XCD lever |
|---|---|---|---|
| M split per group (MoE expert GEMMs; forward, dgrad) | tiles per group | none while rounds are full | group-aligned runs (§3) |
| variable-K wgrad | contraction per tile | longest-first (§5), 1 tile per WG (§6) | round-robin (§2) |

M-split tiles do the same MFMA work and move the same bytes except ragged boundary tiles
(`grouped_tile_pricing.md` §4); variable-K
groups all have the same tile count.

- *(field, E1)* Copying the forward settings (GROUP_M = 2, eight contiguous XCD slices) to the variable-K wgrad of
  the same MXFP4 kernel ran wgrad at 0.38–0.55× speed: under skew the slices gave one XCD the four largest
  groups, 86% of the work.
- On wgrad, any partition coarser than one tile leaves up to one run of the heaviest group as imbalance. With
  276 tiles per group and runs of 8 tiles, one run per XCD is 23% of a group's tiles; times a hot group's 62%
  share of the work that is 14% of the launch. Only the hardware's own `bid % 8` (run length 1) is exact.
- Do not share one default configuration, or one race, between the M-split path (forward, dgrad) and the
  variable-K path (wgrad).

## 2. XCD placement: which way the remap should run
Dense GEMM sorts tile streams by whether they read **a slab of their own** or **a shared slice**, and in neither
class did the class alone fix the order: race a group's workgroups clustered on one XCD against the hardware
round-robin, and sweep the band width (`../gemm/schedule_and_locality.md` §2). For grouped layouts:
- **M-split tiles (MoE expert GEMMs, forward, dgrad) read a private slab.** Every M tile of group g streams
  `B[g]`. When `B[g]` exceeds an
  XCD's 4 MB L2, race putting co-resident workgroups that share a B column block on the same XCD (contiguous
  slices with a narrow band) against the identity order. *(field, E2)* fp8 NN dgrad, 8.7 MB slab: the shared-slice recipe (identity XCD order)
  lost 5.7% at GROUP_M = 4; on the NT forward of the same projection it lost 2.3% at GROUP_M = 2 and 5.4% at
  GROUP_M = 1. The slab size alone did not decide the sign: *(field, E3)* on the NN dgrad, 8 contiguous slices
  with GROUP_M = 4 gained 0.79% at K = 2944 and lost 1.91% at K = 5760 (17 MB slab). Treat K as part of the key.
- **Contiguous slices trade balance for locality.** *(field, E4)* MXFP8 NT: 8 slices made the heavy-skew down
  configs 2.8–3.0% slower, balanced unchanged. *(field, E5)* MXFP4 NT: 8 slices without an N band beat 1 slice
  with a 4-wide N band by 7–8% on the two worst (skewed) configs, and 1 slice ran the balanced down configs
  2.3–3.2% slower than 8; 2 slices sped the heavy configs of a four-config screen by 4.2–5.4% but dropped one
  wide-N short-K forward config from 1.373 to 1.223. Sweep per layout, and put both a narrow-N and a wide-N
  short-K shape in the screen: a screen with only a long-K shape missed that collapse.
- **Variable-K wgrad reads a shared slice.** Both operands are token-major slices of the group, shared by all
  of its output tiles, so XCD clustering fragments the sharing. *(field, E6, E7)* fp8 wgrad: identity order with
  GROUP_M = 2 beat XCD-affine rectangles and wide bands by 0.8–1.0%; GROUP_M = 1 lost 5.1% (co-resident
  workgroups no longer share rows). *(field, E8)* A contiguous remap applied inside each group, keeping the group order,
  still lost 0.6% overall and 2–3 points on two configs: round-robin keeps all eight XCDs on the same band.
  That measurement (G = 32, 144 tiles per group, ~18 rounds) is at the depth where Turbo's tuner leads with
  the affine rectangle (§3, last bullet); race both on the deployed shape.
- **A contiguous remap also undoes a longest-first order** on variable-K wgrad (measured: `grouped_variable_k.md`
  §5). A race scored on balanced data alone picks the losing arm; score on the worse of balanced and skewed.
- **Affinity that stays balanced.**
  - Renumber a group's tiles into XCD-affine rectangles only where the band does not already hand each XCD a
    compact sub-grid. Turbo decides that on the host from the tile geometry: `group_m × group_n` = 8 with both
    dividing the block grid (`_wgrad_band_is_xcd_aff`, FP8:1967-1971). *(field, E9)* fp8 wgrad at G = 4: with
    that gate (affine on gate_up only) +0.18% on the score, without it (both projections) −0.15%.
    *(field, E8)* At G = 32, the in-group contiguous remap above lost 0.6%.
  - Rotate which class serves the heavy side. When super-blocks of `gp` groups are dealt to the XCD classes and
    the heavy group of each super-block always lands on the same classes, advance the position by one per
    super-block (class stride `k+1`). The stride must be odd: ragged boundary tiles are cheaper, and an even
    stride freezes them on fixed classes. *(field, E10)* fp8 wgrad: +5.3% on a strongly skewed config (an offline
    greedy model predicted +5.0%), the mildly skewed configs unchanged, balanced slightly faster, one extra
    SGPR. Turbo rotates each group's tiles by an odd step coprime with the 16-tile CU-residue class
    (`_WG_TILE_ROT = 5`, FP8:2053, 4652) and gives its XCD-affine super-block exactly one group "so every XCD
    holds the same fraction of every group's K" (FP8:3898-3901).
  - Do not hand-tune the super-block span. *(field, E11)* fp8 wgrad, one projection: `gp` 4 / 8 / 1 against the
    automatic 2 were 0.5%, 9.2% and 5.8% slower.
- **Thin groups are a different regime.** Turbo's MXFP4 NT checks whether a group is at least
  `num_xcds × span` M blocks wide. Below that every XCD sees a different expert, and it switches to a blocking
  with non-temporal C stores, because "the write-only C stream evicts re-read weights" (MX4:1100, 1164-1170).

## 3. Two schemes both called "band-cyclic"
They permute different things. Name them apart.

| | XCD band deal | group interleave |
|---|---|---|
| Permutes | which XCD runs a tile | when a tile is dispatched |
| Rule | XCD x gets runs x, x+8, … of `band` ids (GH:2694) | band 0 of all groups, then band 1, … (FP8:2024) |
| In Turbo | bf16 forward/dgrad/wgrad, MXFP4 NT (forward and dgrad) | fp8 masked wgrad fallback (`_WG_INTERLEAVE`, FP8:102) |

**XCD band deal.** Contiguous slices and per-tile round-robin are the two ends of one axis, the run length.
Contiguous parks a run of small groups (no B reuse) on one XCD; per-tile scatters a band's reuse over eight.
- *(field, E12)* MXFP4 NT, 32 groups of 16 M blocks under uniform tokens. Run of 2 M blocks (one GROUP_M band):
  ~10% slower balanced. Run of 16 (one group): balanced 0–0.8% slower, heavy skew 4.5–6.1% faster. Run of 32:
  heavy skew only 3–4% faster. Net +1.6% geomean, +4.8% on the worst config; the heavy-vs-balanced penalty
  fell from 6.7–7.4% to 0–1.8%, with `SQ_INSTS_MFMA` and the L2 footprint bit-identical.
- Rule of thumb, from the shape of E12: a run is the M blocks of one group under uniform tokens, with at least 4
  runs per XCD. A run that crosses a group boundary pulls two slabs into one XCD. It is not universal: the bf16
  grouped GEMM of E14 was faster at a band of 32 tiles, the workgroups one XCD keeps resident, than at wider bands
  *(field, E14)*; sweep the run length per kernel. Turbo's MXFP4 NT uses the group-aligned run when it
  divides every group's M-block count and falls back to a narrow band otherwise (MX4:593-629).
- The band size matters more than the XCD count. *(field, E13)* bf16 wgrad: a band of one N row inside a group
  beat the identity order (worst config +1.05%); a band of a whole group's tiles left the worst config 15%
  below the narrow band under skew. Turbo's bf16 wgrad keeps "runs short enough to stay inside one expert, so
  skew spreads" (BF16:431). *(field, E14)* On bf16 forward/dgrad a band of M blocks raised the 24-config score
  by 1.4% (forward NT and dgrad NN, one bench run each), and by 2.6% stacked with a GROUP_M re-sweep and a wgrad
  tile-decode change.

**Group interleave.** It keeps every group in flight at once.
- *(field, E15)* fp8 per-tensor wgrad at identity XCD order: group-contiguous dispatch was 20%, 34% and 59%
  slower at 2:1, 4:1 and 32:1 skew than balanced. *(field, E16)* Interleaving cost 0.9% balanced and cut the 30:1 penalty
  from 46% to 26%.
- It also dispatches a hot group's later bands late: with three bands per group, the third starts 67% of the
  way through the stream. Combined with contiguous XCD slices it stacks the hottest groups on XCD 0.
  *(field, E17)* MXFP8 wgrad: interleave plus 8 contiguous slices was the skew collapse; group-major order with the
  identity XCD order took the heavy down config from 0.474 to 1.005 against its reference (+112%), the worst
  config of the table from 0.474 to 0.928, and balanced from 0.955 to 1.001. Group-major is longest-first
  whenever group sizes fall with group index.
- Do not interleave per tile.
  - *(field, E18)* Turbo bf16 wgrad with `group = tile % G`: 1081 → 833 TF/s balanced,
    1048 → 482 at 8:1 skew.
- Turbo keeps interleave only in its masked fallback; its 4-wave wgrad keeps group-major order under the
  hardware XCD split (`num_xcd = 1`, FP8:4654-4656) and, on launches of at least 8 dispatch rounds, leads its
  race with an XCD-affine rectangle inside one group (`_WGRAD_AFF_ROUNDS`, FP8:4657-4663, 4693-4720).

## 4. Band width (GROUP_M) against K
- Two terms pull against each other: B traffic falls as 1/GROUP_M (each band re-streams B), and the band's A
  footprint `GROUP_M × BLOCK_M × K × bytes` must fit an XCD's 4 MiB L2. Only K moves the second term.
- *(field, E19)* MXFP8 NT: at K = 2944 (A band 2.88 MiB at GROUP_M = 4) the peak was GROUP_M = 4; at K = 5760
  (5.62 MiB) it was 8 — GROUP_M 2/4/8/16 scored 1.010/1.036/1.045/1.031. "A band ≤ 4 MiB" alone would pick 2,
  the worst point. One global GROUP_M = 8 cost 0.21% on the geomean: 1.1% faster on one config and 3.0% slower
  on another K.
- Key the choice on K: switch to the wider band when `GROUP_M × BLOCK_M × K` crosses 4 MiB, with K in the
  autotune key (`grouped_host_autotune.md`). Sweep GROUP_M past 8 and again after any tile change; the best value
  moved with the shape. *(field, E20)* On one bf16 dense GEMM, 9 beat the values ≤ 8 that were tried by 1.2–5.8% at
  two m and 4 won at a third, where 9 was not tried (`../gemm/schedule_and_locality.md` §2); *(field, E45)* a Triton
  GEMM kept 9 ahead on only one shape and lost to 4 or 8 by 1.3–2.7% on two others.
- It does not carry across layouts. *(field, E21)* The same K rule on fp8 NN dgrad cost 1.2 points, likely because
  NN reuses an N stripe of `B[K,N]`, not an A slab.
- On MXFP4, keeping B resident with a 2D band lost to a 1D band that sweeps all of N for each row pair. Turbo
  records a 6-wide `group_n` band losing on all four NT cells; on wgrad the 1D band was 3.7% faster, with an
  L2 hit rate of 62.7% instead of 57.9% and 2.49 GB of HBM reads instead of 2.86, at identical request and MFMA
  counts (MX4:1102-1110, 1125-1138).
- *(field, E22)* bf16 forward/dgrad: GROUP_M 1/2/4/8 within 1% once the XCD band deal was in.

## 5. Longest-first inside a wave
- M-split tiles do the same MFMA work, so order changes are estimated at 0 where every dispatch round
  was full. Variable-K wgrad tiles cost what their group's tokens cost, so the dispatch order is a list
  schedule (`../attention/grid_dispatch.md` §4).
- Three ways to put the deepest group first, all computed on the device from the offsets:
  - Rank all groups in one wave: lane g holds group g's length, the stable descending rank takes G `readlane`
    compares, and slot → group is a one-hot ballot plus a popcount (`wave_rank_desc_stable`,
    `wave_lane_with_rank`, GH:688-709). bf16 wgrad maps `tile // TILES_PER_GROUP` through it (BF16:424-433);
    it needs G ≤ 64.
  - Rank groups by token span the same way in a split launch (FP8:4211-4230).
  - Pick one of the two walk directions from two half-sums of the offsets: reverse the group-major stream when
    the second half of the groups holds more tokens (MX4:1866-1876).
- *(field, E23)* MXFP4 wgrad: the direction choice gained 1.65% and 2.3% on the two cells of a 10× skew, others
  bit-identical. An unconditional reverse gained 3.4–6.0% there but was 7% and 19.8% slower on the configs
  whose histogram already descended, where group-major was already longest-first.

## 6. Persistent grid or one tile per workgroup
- With one workgroup per CU (the usual grouped tile), one tile per workgroup is only partly dynamic, although Turbo's
  comment calls the dispatcher the list scheduler (MX4:1112-1115). *(field, E48)* On MI355X a
  workgroup's XCD and its shader engine inside that XCD follow from its blockIdx; only inside that 8-CU engine does
  the next tile usually go to the CU that frees first. Random 20–100 µs workgroups ran 1.09× a device-wide list
  schedule, and 3.0× when the long ones shared blockIdx % 8. A persistent worker fixes the whole assignment at
  launch.
- *(field, E24)* fp8 per-tensor wgrad under heavy skew: a grid capped at 2× the CU count left a workgroup that drew
  a hot group's long tile with ~14 more tiles queued behind it. One tile per workgroup was ~26% faster on the
  heaviest config and 8% on the moderate one, unchanged elsewhere, byte-identical. *(field, E25)* On its scored shapes a grid
  of exactly the CU count was flat, and every grid between the CU count and the tile count was 1.3–4.9% slower.
- *(field, E26, E27, E28)* Truncating to a persistent grid lost on both MXFP8 kernels: NT 0.70–0.81× speed
  (twice; the second time with an O(1) lookup, so decode cost was not the reason), variable-K wgrad 11–18%
  slower under skew. The NT attempts showed 23–25 spilled VGPRs in the ISA each time:
  the loop keeps the group table, block id, grid size and live count alive across the main loop at 246/256
  arch VGPRs. At that wall, race the scheduler strategy (`../gemm/schedule_and_locality.md` §3) and
  check VGPR headroom before unrolling (`../gemm/mainloop.md` §7).
- *(field, E29)* bf16 wgrad: a persistent grid-stride loop was 8–18% slower and no flatter under skew.
- *(field, E30)* Persistent across groups (one workgroup walks one tile position through all G groups; grid
  divided by G), MXFP8 variable-K wgrad: its time against the reference kernel went from ~1.02× to 1.23–2.00×.
  276–448 tiles per group is 1.1–1.75 rounds on 256 CUs, and the 8× unrolled body thrashed the instruction cache.
- *(field, E31)* fp8 forward/dgrad: non-persistent was ~11% faster than a persistent tile loop, but only
  with the L2 tile swizzle kept (without it one small-K shape was 13% slower).
- When it can pay: free registers and a loop that prefetches the next tile across the boundary. *(field, E33, E34)* 1
  wave per SIMD with AGPR accumulators left 72 arch VGPRs free on one MXFP8 kernel but only 32 on plain MXFP4 NT,
  where 2 and 4 tiles per workgroup compiled without spills and still lost (§7). *(field, E35)* On a dense
  GEMM at one workgroup per CU, pairing two tiles alone was 0.78% slower, and pairing with the peeled tail's
  prefetch aimed at the next tile was 0.69% faster than one tile per workgroup: the value of a persistent
  worker is the next tile's fill overlapping this tile's epilogue (`../gemm/mainloop.md` §3,
  persistent carry), not the launch it saves. Gate on a compile-only spill check
  (`grouped_tile_pricing.md` §1).
- Repeating a ping-pong tile body breaks barrier parity. A body in which one wave group issues an extra `s_barrier`
  (`if wave_m == 1`) is consistent only once per workgroup; from the second tile the groups pair with the wrong
  barrier and the output is wrong. Balance the count: a matching `s_barrier` for the other group at the tile
  boundary, or one barrier that every wave takes in place of the conditional one, as Turbo's bf16 wgrad does when
  persistent (BF16:263-268); an unconditional `s_barrier` added at the boundary does not fix it. Another persistent
  form worth racing: *(field, E36)* a dense persistent grid with
  no workgroup barrier between tiles (C staged per wave, the next tile's stages filled by the last K-blocks) beat
  one tile per workgroup by 3–5% (`../gemm/grid_pricing.md` §10).
- A persistent grid capped below the CU count exists to leave CUs to a co-running kernel, not for speed (Turbo:
  BF16:622-625; FP8:2669-2670).
- Which variable-K loop body to run (chunked masked or exact-trip) is keyed on the per-group contraction
  `m_total / G`, not on `m_total` (`grouped_variable_k.md`).

## 7. Two tiles per workgroup
- What it buys is the per-workgroup fixed cost (launch, decode, table load, pipeline fill) shared by WGT tiles
  (tiles per workgroup; TOTAL below is the grid's tile count); what it gives up is the dispatcher's dynamic balance.
  Turbo ships `wg_tiles = 1` in all three of its MXFP4 wgrad tuples (MX4:1112-1122) and one tile per workgroup on
  MXFP4 NT (MX4:1086).
- If you pair, gate on the paired grid width, not on how many rounds a group spans: `TOTAL / WGT ≥ N_CU` with
  TOTAL divisible by WGT (MX4:1725), and take the two tiles from opposite ends of the stream, so a heavy one
  pairs with a light one.
- On MXFP4 variable-K wgrad the sign has followed the rest of the kernel; re-measure whenever the boundary
  bodies or the C store change.
  - *(field, E37)* With half-price M-tail bodies and whole-line `nt` folded stores in place, WGT = 2 gained 1.8–2.3
    points on the 10× skew and 0.7–1.0 on moderate skew; without them it lost 0.7–1.2% on heavy skew. WGT 3
    dropped the heavy config to 0.872 and 4 was worse; run-to-run spread on heavy tripled.
  - On another build, one tile per workgroup beat WGT = 2 on three of four cells (up to 2.7% under skew) and
    lost 1.05% on one balanced cell; WGT 4 and 8 lost up to 7.5% under skew and 11.9% balanced (Turbo's
    comment, MX4:1112-1121).
- *(field, E38, E34)* Plain MXFP4 NT, two tiles per workgroup with one emitted body: 1.1–3.7% slower on all four cells
  (0.9% on one cell in another build, 1.8% there with four tiles); +30 instructions, identical
  MFMA/load/store counts, VGPRs 480 → 512 (MX4:1086-1099). The loss was put down to static pairing locking in
  the imbalance of the cheap boundary tiles that the dispatcher would otherwise absorb; that was inferred, not
  measured.
- Do not pair remainder tiles into "equal-cost" tiles. *(field, E39)* fp8 wgrad, one workgroup running two
  half-height tiles back to back: MFMAs exactly 3.89% fewer, output bit-exact, wall 20% slower, L2 hit rate
  68.0% → 44.5%, DRAM reads +70%. A half-height body is 0.5× the MFMAs and 0.75× the bytes but 1.0× the K walk,
  so the pair lives ~1.5× as long; with 32 co-resident workgroups per XCD it straddles two runs, and the live
  slabs went from 12 to ~24. Co-residency is aligned in workgroup lifetime, not in MFMA count.

## 8. Launch skew
Turbo's MXFP4 NT makes workgroup `bid < 128` run `bid` iterations of `s_sleep 2` (65–128 clocks each in the
CDNA4 ISA) before starting and lets the rest start at once (MX4:104-130). Against a ramp over all 256 ranks,
its paired rounds record the NT total 0.88% less time (better in 6 of 6 rounds), but no delay at all took
0.84% less time against the same ramp: over no skew, the 128-rank ramp is level on the total and differs only
on single dgrad cells (MX4:102-115). Measure a launch skew against no delay, not against another ramp.
*(field, E40)* On a dense persistent grid, delaying half of each XCD's workgroups paid 1–3% at 6–20 tiles per workgroup
and nothing at ~60 (`../gemm/grid_pricing.md` §10).

## 9. Why a grouped GEMM trails one large GEMM
- In one launch, CUs run different groups at the same time and their weight slabs compete for L2 (inferred, not
  measured with L2 counters); per-group launches keep one slab hot but pay launch and ramp per group.
- The group-major order with the identity XCD order approximates "one group at a time" inside one launch,
  which is why it is the base the other levers in this card are measured against.

## Verify
- The pass is classified (skew in tile count, or in contraction per tile) before any order change.
- Every remap is a bijection over the live tile count and is scored on balanced and skewed data separately.
- Run lengths of an XCD band deal were swept per kernel, with the one-group run among the candidates.
- Persistent or multi-tile variants passed a compile-only spill check before timing.
- Order changes are checked bit-identical: they must not change a single output byte.

## Related
`grouped_group_mapping.md` · `grouped_tile_pricing.md` · `grouped_variable_k.md` ·
`grouped_host_autotune.md` · `grouped_measurement.md` ·
`../gemm/schedule_and_locality.md` §2 · `../gemm/grid_pricing.md` §1, §10 ·
`../attention/grid_dispatch.md` §4

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The Kimi-K3 forge runs (E20, E36, E40) installed `flydsl==0.3.2`. The month is the run's where the record
dates it; where it does not, the line says which date it is. "Turbo" is Primus-Turbo; "gm" is a campaign's geomean
score; gpt-oss shapes are G=32 experts, 131072 routed tokens, hidden 2880 (padded to 2944).

- E1 [2026-08, gfx950, FlyDSL ?, N=2, per-configuration timing with the race disabled, each configuration measured twice (repeatability < 0.15%)] the NT constants (GROUP_M 2, 8 contiguous XCD slices) applied to the wgrad path of the same kernel: wgrad moderate/heavy at 0.532 / 0.549 / 0.382 / 0.387 of MXFP8's speed, gm 1.0406; the 86% share of the four largest groups is computed from the heavy histogram (Turbo MXFP4 grouped GEMM, gpt-oss)
- E2 [2026-08, gfx950, FlyDSL ?, N=?, isolated 20-arm sweep at the deployed 4096 tokens per expert, outputs bitwise identical] NN dgrad: identity XCD order with GROUP_M 4 (`x1g4`) −5.72% against the incumbent arm, over which `x8g4` read +2.48%; NT forward of the same chain, against its incumbent `x8g4`: `x1g2` −2.26%, `x1g1` −5.35% (Turbo fp8 per-tensor grouped GEMM, gpt-oss down projection padN chain, B[g] 8.67 MB)
- E3 [2026-09, gfx950, FlyDSL ?, N=12, 3 pairs × both orders × 3 draws (11 of 12 reads negative on GateUP), no run reference] `xcd8_gm4` on NN dgrad: Down (K=2944, B[g] 8.67 MB) +0.79%, GateUP (K=5760, B[g] 16.96 MB) −1.91%; dated 2026-09-04 (Turbo fp8 per-tensor grouped NN dgrad, gpt-oss)
- E4 [2026-07, gfx950, FlyDSL ?, N=1, campaign score: geomean of t_tw / t_mx over 14 configurations, kernel-only, against two same-session baseline runs] XCD affinity (xcd 8) for narrow bands: fwd down heavy 1.098 / 1.084 → 1.065, dgrad down heavy 1.089 / 1.094 → 1.059; gm 1.1065 against 1.11046 / 1.10536 (Turbo MXFP8 grouped NT, gpt-oss)
- E5 [2026-08, gfx950, FlyDSL ?, N=2, per-configuration timing (ms) with the race disabled, each value measured twice; then fast-probe ms screens and the campaign score] before the dead-id fix of `grouped_group_mapping.md` §4: (GROUP_M 2, 8 slices, no N band) against (2, 1 slice, 4-wide N band) on the two worst heavy configurations 1.3447 against 1.4453 ms (−7.0%) and 1.1993 against 1.3005 (−7.8%); after it, one slice ran the balanced down configurations 2.3–3.2% slower than 8, 2 slices sped the heavy configurations of a four-configuration screen by 4.2–5.4%, and (2, 2, 0) over all 18 configurations scored 1.43661 against 1.4486 for 8 slices, with forward gate_up balanced falling from 1.373 to 1.223 (Turbo MXFP4 grouped NT, gpt-oss)
- E6 [2026-08, gfx950, FlyDSL ?, N=?, isolated sweep of 25+ (num_xcd, GROUP_M) arms at the deployed 4096 tokens per expert, then three distributions, outputs bitwise identical] wgrad winner `x1g2` +0.78–0.98% over XCD-affine rectangles and wide bands; balanced +1.00, heavy +0.95, canon +0.76% (Turbo fp8 per-tensor grouped wgrad, gpt-oss down projection, 144 tiles per group)
- E7 [2026-08, gfx950, FlyDSL ?, N=?, ?] GROUP_M 1 is recorded at −5.06% in the sweep of E6; the campaign record does not give the arm
- E8 [2026-08, gfx950, FlyDSL ?, N=1, campaign score (geomean of t_mx / t_tw over 14 configurations)] the XCD remap applied to the in-group tile index, group order kept: gm 0.99372 → 0.98762, wgrad down balanced 0.962 → 0.941, gate_up heavy 0.995 → 0.965 (Turbo fp8 per-tensor grouped wgrad, gpt-oss)
- E9 [2026-08, gfx950, FlyDSL ?, N=3, campaign score: geomean over 6 configurations of t_fwd / t_wgrad in one process, against 2 same-day baseline runs; wgrad wall per configuration] XCD-affine tile renumbering for gate_up only (down kept its band by a host-side geometry gate, `group_m*group_n == 8` with both factors dividing the block grid, which drops the affine candidate): bench 1.07790 / 1.07623 / 1.07839 against 1.07695 / 1.07591, the harness's gain +0.18%; ungated (both projections) 1.07534 against 1.07695 (−0.15%), kernel-only gate_up −0.87 to −1.86%, down −0.40 to +0.39% (Turbo fp8 per-tensor grouped wgrad, G=4, M=131072; the gate as it stands in Turbo ca1fdd67: `_wgrad_band_is_xcd_aff`, grouped_gemm_fp8_kernel.py:1967-1971)
- E10 [2026-08, gfx950, FlyDSL ?, N=3+5, interleaved A/B of the campaign score (3 base, 5 candidate reads)] skew-gated per-XCD group-position advance: down.strong1.3 1.1655 / 1.1682 / 1.1685 → 1.2294–1.2319 (+5.3%, disjoint, 3 of 3 adjacent pairs same sign), offline per-XCD model predicted +5.0%, balanced down mean 0.9934 against 0.9902, +1 SGPR, VGPR unchanged, 0 spill (Turbo fp8 per-tensor grouped wgrad, gpt-oss, skews 1.07^i / 1.3^i / one-hot)
- E11 [2026-08, gfx950, FlyDSL ?, N=?, ?] super-block span `gp` on the down projection against the automatic 2: gp=4 −0.52%, gp=8 −9.23%, gp=1 −5.8% (gate_up: automatic 1, others −1.03 to −1.81%); the run is not named; one arm of the large-M campaign read gp=1 at −5.68% (Turbo fp8 per-tensor grouped wgrad, 32 experts)
- E12 [2026-08, gfx950, FlyDSL ?, N=3, palindromic fast probes with the baseline twice, then the campaign score of E1's campaign three times] band-cyclic XCD runs: span 2 +9–11% balanced; span 16 +0.0–0.8% balanced, −1.8–2.7% moderate, −4.5–6.1% heavy; span 32 −3.1–4.4% heavy (fast-probe time changes); bench 1.47328 / 1.47023 / 1.46943 against 1.4495 (+1.48%), min +4.78%; the heavy-against-balanced penalty 6.7–7.4% → 0–1.8%, MFMA and L2 footprint bit-identical (Turbo MXFP4 grouped NT, 16 M blocks per group)
- E13 [2026-08, gfx950, FlyDSL ?, N=?, the campaign's scoring script on an isolated tree, all three changed files synced] band of one N row (`xcd_band_remap_pid(…, 8, N_BLOCKS_N)`) on wgrad: gm_tf 1161.93 → 1165.93 (+0.34%), worst configuration 1031.2 → 1042.0 (+1.05%); a whole group's tiles as the band: gm 1126.59, worst 880 (Turbo bf16 grouped GEMM, gpt-oss; 2026-08-24)
- E14 [2026-08, gfx950, FlyDSL ?, N=3, campaign score gm_tf over 24 configurations; the stacked change read three times, each step on the way once] NT band-cyclic XCD remap (8 XCDs, band 32): bench 1110.15 → 1122.92; NN the same: → 1125.74 (+1.4% for the two); wgrad tile decode at GROUP_M 4: → 1139.70; NT GROUP_M 8 → 4: → 1141.61; the stacked change 1138.5 against 1109.3 (+2.63%), reads 1141.5 / 1138.5 / 1138.3, every arm bit-identical; a fast probe of the NT forward at GROUP_M 4 (one reading per point, outputs identical) put band 32 0.4–1.6% above bands of 92 and 96 tiles (one GROUP_M super-tile per run) in all four cells (gate_up and down, balanced and skew 8) (Turbo bf16 grouped GEMM, gpt-oss)
- E15 [2026-06, gfx950, FlyDSL ?, N=?, ?] group-contiguous dispatch of the masked variable-K wgrad under skew: 2:1 −20%, 4:1 −34%, 32:1 −59%; unverified (Turbo FlyDSL fp8 per-tensor grouped wgrad)
- E16 [2026-07, gfx950, FlyDSL ?, N=?, ?] band-cyclic group interleave: balanced 2143 (−0.9%), 30:1 skew 1162 → 1592 (−46% → −26%); unverified (Turbo FlyDSL fp8 per-tensor grouped wgrad, `num_xcd=1`)
- E17 [2026-07, gfx950, FlyDSL ?, N=1, campaign score: geomean of t_tw / t_mx over 14 configurations, kernel-only] wgrad group-major with identity XCD order instead of band-cyclic interleave plus 8 XCD slices: gm 0.8062 → 0.9864, min 0.4743 → 0.9277, wgrad down heavy 0.474 → 1.005; the round-3 note gives balanced 0.955 → 1.001 (Turbo MXFP8 grouped wgrad, gpt-oss)
- E18 [2026-08, gfx950, FlyDSL ?, N=?, TF/s per configuration] `group_idx = tile % G` against group-major: GateUP balanced 1081 → 833, skew 8 1048 → 482; Down 1026 → 806, 942 → 460 (Turbo bf16 grouped wgrad, gpt-oss; record dated 2026-08-24)
- E19 [2026-07, gfx950, FlyDSL ?, N=1–3, the campaign score of E4 over all 14 configurations per point] GROUP_M on `dgrad gate_up heavy` (K=5760): 2/4/8/16 = 1.010 / 1.036–1.041 / 1.045–1.047 / 1.031; a forced global GROUP_M 8: gm 1.10404 (−0.21%), `dgrad gate_up heavy` 1.036 → 1.047, `fwd gate_up balanced` 1.122 → 1.086 (Turbo MXFP8 grouped NT, gpt-oss)
- E20 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver graph-timer medians (warmup 3, 20 iterations), FlyDSL cache cleared before each sweep] on dense bf16 GEMMs, GROUP_M 9 against smaller values: m=2732 1.129–1.133 against 1.173–1.194 ms, m=9397 3.440–3.442 against 3.483–3.538 ms (AKLR bf16 dense GEMM N=7168, K=35840, forge run)
- E21 [2026-07, gfx950, FlyDSL ?, N=?, campaign score] cand[0]'s GROUP_M narrowed 8 → 4 at K < 4096 on the NN dgrad: dgrad down (K=2944) 0.990 → 0.978 (Turbo fp8 per-tensor grouped NN dgrad, gpt-oss)
- E22 [2026-08, gfx950, FlyDSL ?, N=?, TF/s per projection; run-to-run band about 1%] with the band-cyclic XCD remap over M blocks in place, GROUP_M 1/2/4/8: GateUP 1217 / 1215 / 1230 / 1214, Down 1226 / 1227 / 1226 / 1226 (Turbo bf16 grouped GEMM forward/dgrad, gpt-oss; record dated 2026-08-24)
- E23 [2026-08, gfx950, FlyDSL ?, N=3+5, campaign score (geomean of MXFP4 deploy TF/s against a reference), medians of 3 and 5 runs] longest-group-first direction chosen at run time: canon gate_up wgrad 4008 → 4074 TF/s (+1.65%), canon down 3854 → 3942 (+2.3%), the other distributions within noise; an unconditional reverse scored gm 93.04 with canon +3.4% / +6.0% and moderate / heavy −7% / −19.8% (Turbo MXFP4 grouped wgrad, gpt-oss deploy regime, 4096 tokens per expert)
- E24 [2026-07, gfx950, FlyDSL ?, N=?, t_mx / t_tw, drift-immune interleaved] grid capped at 2× the CU count against one tile per workgroup: wgrad gate_up heavy 0.76 → 0.96, moderate 0.89 → 0.96, balanced and down configurations flat, byte-exact (Turbo fp8 per-tensor grouped wgrad `kernel_grouped_tn_wgrad_4wave`, gpt-oss; 2026-07-30)
- E25 [2026-08, gfx950, FlyDSL ?, N=2, production entry with each geometry raced separately; campaign bench] grid = CU count against one tile per workgroup −0.03% / −0.24%; intermediate grids 512 / 768 / 2304 (down) and 2208 / 2944 / 4416 (gate_up) −1.3 … −4.9% (Turbo fp8 per-tensor grouped wgrad, gpt-oss deploy regime)
- E26 [2026-07, gfx950, FlyDSL ?, N=?, a probe, single process interleaved, 3 shapes × 2 distributions × 3 configurations] persistent NT with the lane-resident lookup in place: every point 0.70–0.81×, nearly point for point the earlier round's result (Turbo MXFP8 grouped NT, gpt-oss)
- E27 [2026-07, gfx950, FlyDSL ?, N=?, ?] a persistent variable-K wgrad (grid truncated to the CU count plus a stride loop): skew 11–18% worse (new/old), reverted; recorded in the goal of the MXFP8 campaign as a result of the session before it; that campaign's first round calls it consistent with its own finding that tile order, not persistence, fixed the skew (Turbo MXFP8 grouped wgrad `_build_grouped_mxfp8_wgrad_kernel`, gpt-oss)
- E28 [2026-08, gfx950, FlyDSL ?, N=1, ISA dump and an isolated timing] one body, half grid, stride = grid: `vgpr 256 / spill 25 / sgpr 106`, −26%; the four attempts' spills are listed as 25 / 25 / 23 against a 246/256 wall (Turbo MXFP8 grouped NT)
- E29 [2026-08, gfx950, FlyDSL ?, N=?, TF/s per configuration, SNR 55.6] persistent grid 256 + grid stride against one tile per workgroup: GateUP 887 / 863 against 1081 / 1048 (balanced / skew 8), Down 936 / 887 against 1026 / 943 (Turbo bf16 grouped wgrad, gpt-oss; record dated 2026-08-24)
- E30 [2026-07, gfx950, FlyDSL ?, N=?, ?] persistent across groups (grid G·TILES → TILES): correct (SNR 28.14) but MX/TW time from ~1.02× to 1.23–2.00×; unverified (Turbo MXFP8 grouped variable-K wgrad)
- E31 [2026-06, gfx950, FlyDSL ?, N=?, ?] non-persistent forward/dgrad about 11% faster than persistent; without the L2 swizzle one small-K shape (gpt-oss down) −13%; unverified (Turbo FlyDSL fp8 per-tensor grouped NT)
- E33 [2026-08, gfx950, FlyDSL ?, N=?, ISA] 72 free arch VGPRs on the MXFP8 NT kernel at one wave per SIMD with AGPR accumulators (Turbo MXFP8 grouped NT)
- E34 [2026-09, gfx950, FlyDSL 0.2.4, N=?, ISA gate, then ABBA wall clock] k = 2 / 4 tiles per workgroup on plain MXFP4 NT: `next_free_vgpr` 480 → 512, `accum_offset` 224 → 256, spill 0; fc2_fwd balanced −0.93% (k=2), −1.79% (k=4) (Turbo plain MXFP4 grouped NT, gpt-oss)
- E35 [2026-10, gfx950, FlyDSL ?, N=?, campaign bench, linear1 time per call] on a dense A6W4 GEMM, two m-tiles chained in one workgroup 522.6 → 526.7 µs (+0.78%); with the peeled tail's prefetch aimed at the next tile 519.0 / 519.1 µs (−0.69%) (Turbo A6W4 dense forward `single_l1` 16384×21504×3072)
- E36 [2026-10, gfx950, FlyDSL 0.3.2, N=1, own CUDA-graph harness, all arms in one process, 10 launches per graph, median of 10 replays] on a dense bf16 GEMM, the same 256×256 body as a persistent 256-workgroup grid against one tile per workgroup: −2.8% to −5.2% time at about 6.5 / 19 / 63 tiles per workgroup (AKLR bf16 dense GEMM N=3072, K=512, forge run)
- E37 [2026-08, gfx950, FlyDSL ?, N=8+11, interleaved samples, paired difference of the campaign score] the WGT = 2 gate opened on gate_up wgrad: canon +1.8–2.3 points, moderate +0.7–1.0, heavy +0.3–0.5; WGT 3 dropped heavy to 0.872 and 4 was worse; the heavy run-to-run spread grew about 3×, and before the half-price M tails and folded `nt` stores the same change lost 0.7–1.2% on heavy (round 6: heavy gate_up 3959 → 3911 TF/s) (Turbo MXFP4 grouped wgrad, gpt-oss deploy regime)
- E38 [2026-09, gfx950, FlyDSL 0.2.4, N=?, four-arm round-robin (wgt1 / wgt2 lgkm fence / wgt2 vmcnt(0) fence / wgt1), same-arm drift 0.22–0.98%] two tiles per workgroup, one emitted body: fc2_fwd@bal +2.04 / +2.70%, fc1_dgrad@bal +2.29 / +1.10%, fc2_fwd@skew +2.65 / +1.71%, fc1_dgrad@skew +3.63 / +3.72% time; instructions 3404 → 3434, `v_mfma` 880, `buffer_load` 202, `buffer_store` 116, `next_free_vgpr` 480 → 512 (Turbo plain MXFP4 grouped NT, gpt-oss)
- E39 [2026-08, gfx950, FlyDSL ?, N=1, one rocprofv3 pass with 5 counters] one workgroup running two adjacent short-M (`a_halves=1`) tiles: MFMA −3.89%, bit-exact, wall −20%, L2 hit 68.0% → 44.5%, DRAM +70% with requests −1.7%; unverified (Turbo fp8 per-tensor grouped wgrad, gpt-oss with 32 experts at 18 / 34.5 dispatch rounds)
- E40 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder (3 pairs) and own CUDA-graph harness] on a dense bf16 GEMM, a start delay on half of each XCD's workgroups: m=35107 +2.92%, m=104093 +1.06%, m=342073 +0.02% (about 6.5 / 19 / 63 tiles per workgroup) (AKLR bf16 dense GEMM N=3072, K=512)
- E45 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=7 rounds, CUDA graphs without a cache flush, ratio per round against GROUP_M 9; A-vs-A ≤ 0.27% on the cells quoted] bf16 dense NT GEMM, 256 × 256 × 64 tiles, contiguous-slice XCD remap: m = 9397 (N 7168, K 35840) GROUP_M 1–8 0.53–3.83% slower than 9; m = 2732 (same N, K) 4–8 within 0.3% of 9; 8192³ 8 / 4 1.53% / 1.31% faster than 9; 16384 × 7168 × 2048 4 2.66% faster
- E48 [2026-11, gfx950 (MI355X), HIP 7.15, N=5 launches per pattern, per-wave `s_memrealtime` with the hardware CU and XCC ids, default stream] 2048 workgroups, one per CU, each spinning a set time: XCC = blockIdx % 8 in 25 of 25 launches; inside an XCD the k-th and (k+4)-th workgroups shared a shader engine, and workgroups started in blockIdx order, mostly on the first CU to free inside their shader engine; lengths 20–100 µs at random 620.2–621.4 µs against 571.1 µs for a device-wide list schedule (1.086×); 160 µs on blockIdx % 8 == 0 and 20 µs elsewhere 1282.5 against 422.0 µs (3.04×)
