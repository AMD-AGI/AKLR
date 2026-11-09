---
title: "Grouped GEMM — mapping tiles to groups on the device"
kind: topic
lever: grouped_group_mapping
ops: [grouped_gemm]
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, variable-k, skew, empty-groups, large-g]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm1.py:122
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm1.py:748
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/gemm1.py:217
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_utils.py:28
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:97
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:122
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:357
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:386
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:427
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2072
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:3442
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:507
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:654
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp8_kernel.py:474
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:584
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py:712
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/prims.py:404
---

# Mapping tiles to groups in a grouped GEMM

> Every workgroup of a grouped GEMM has to find out, on the device, which group its tile belongs to and which
> rows that group owns. This card covers where the group metadata lives, how the grid is sized without knowing
> the group lengths, how a tile finds its group and what each lookup costs, and how descriptors are built from
> the answer. Descriptor rebase, wave-uniform pinning and XCD basics are the same as for dense GEMM and are
> linked. "FP8", "BF16", "MX8" and "MX4" are Primus-Turbo's
> `primus_turbo/flydsl/grouped_gemm/grouped_gemm_{fp8,bf16,mxfp8,mxfp4}_kernel.py`, "GH" is
> `primus_turbo/flydsl/utils/gemm_helper.py` and "PR" is `utils/prims.py`, all at main `ca1fdd67`; "MOE1" is
> FlyDSL's `kernels/moe/mxfp_moe/gemm1.py`.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end. Use
> them for direction and order of magnitude, not as targets.

## Route here when
- You are writing the prologue of a grouped kernel: reading the offsets, sizing the grid, deciding who exits.
- A grouped kernel issues the same MFMAs as a reference but runs slower, worst on narrow-N or high-G shapes.
- A grouped kernel faults or returns garbage only when a group is empty, or only at large G.
- You are choosing between a pre-built tile table, an SGPR table and a lane-resident table for the lookup.

**Somewhere else if:** the order and XCD placement of tiles across groups → `grouped_scheduling.md`;
tile size, occupancy and boundary bodies → `grouped_tile_pricing.md`; variable-K wgrad →
`grouped_variable_k.md`; MoE inference with sorted token ids → `grouped_sorted_tokens.md`.

## 1. Group metadata stays on the device
- The kernel takes `group_offs[G+1]` (exclusive prefix of the group lengths) as a device tensor and derives
  everything else inside: live tile count, per-group tile prefix, each tile's group and row range. Turbo
  computes the offsets from `group_lens` with a device op (`group_offs_from_lens`,
  `pytorch/kernels/grouped_gemm/grouped_gemm_utils.py:28-33`).
- The host reads no group length on the steady-state path (no D2H sync, graph-capturable; where Turbo's host
  syncs sit: `grouped_host_autotune.md` §1). It knows `M_total` and `G`, which is all the grid needs
  (§3). Any decision read from the offsets inside the kernel must be a pure function of them so every
  workgroup reaches the same answer.
- Read `int64` offsets as their low 32-bit word: pass `group_offs.view(int32)` and read element `2g`
  (`_load_go`, FP8:122-127; the table loads use `stride=2`, FP8:360, 370-371). That holds while token offsets
  stay below 2^31. Byte offsets do not (§7).
- Nothing in the mapping may assume equal group lengths or lengths that are multiples of BLOCK_M. Real routing
  gives skewed histograms that change every step and contain empty groups. A per-group M baked in as a
  compile-time constant, or an `assert M_g % BLOCK_M == 0`, is wrong for MoE.

## 2. Tiles are cut per group
- A forward or dgrad tile is a BLOCK_M slice of **one** group's rows and reads that group's weights. A row
  block that crosses a group boundary multiplies the second group's rows by the first group's weights: wrong
  results, no fault. A kernel that assigns one group per fixed BLOCK_M row block silently requires every
  group to start on that grid. Turbo's bf16 kernels build one `(group, first row, rows)` entry per M tile
  instead (BF16:513-526).
- Cutting at group boundaries adds at most one short tile per group, so for any lengths the M-tile count is
  at most `ceil(M_total/BLOCK_M) + G` (`m_tile_upper_bound`, BF16:507-510).
- Run any super-tile swizzle on the index **inside** the group, after the group is found, so no super-tile
  spans two weight slabs. Turbo's `_grouped_block_mn` decodes `local = tile - tile_start` and falls back to
  row-major for a group with fewer M blocks than `group_m` (FP8:1886-1935). Band widths:
  `grouped_scheduling.md` §4.
- Variable-K wgrad is cut the other way: every group owns the same output tiles, and the skew sits in each
  tile's contraction length (`grouped_variable_k.md`).

## 3. A static upper-bound grid, trimmed on the device

```
grid = (ceil(M_total / BLOCK_M) + G) * n_blocks        # host: M_total, G and N only
live = sum_g ceil(M_g / BLOCK_M) * n_blocks            # device: from group_offs
```

| Dead workgroups leave by | What each one costs |
|---|---|
| `s_endpgm` once `live` is known (FP8:386-396, MX8:486-494, MX4:614-620) | its dispatch and the scan before it |
| a structured `if bid < bound:` around the body (MOE1:770) | the same |
| zero-row tiles: `rows = 0`, descriptor bound 0 (BF16:519-526) | **the whole K loop**: MFMAs still issue |
| a persistent loop `for t in range(bid, live, grid_dim.x)` (FP8:888-890) | none (see scheduling §6) |

The asm exit is `s_cmp_lt_u32 bid, live; s_cbranch_scc1 1f; s_endpgm; 1:` with both operands in SGPRs. A
zero-row tile reads zeros and has its stores dropped, so it is correct, but it runs to the end of its K loop.

- The dead share is at most `G / (ceil(M_total/BLOCK_M) + G)` of the M tiles, reached when every group is
  BLOCK_M-aligned: 32 of 544 M blocks (5.9%) for 32 groups of 4096 rows at BLOCK_M = 256. With zero-row tiles
  that share of workgroups runs the full K loop for nothing (dead quadrants likewise still issue their MFMAs,
  `../gemm/grid_pricing.md` §9).
- With the lane-resident table, `live` comes from the wave scan of §5, which needs a full EXEC mask; the exit
  compares wave-uniform SGPR values, so the whole wave leaves together. Put the exit before the first operand
  load.

## 4. Exit on the raw block id; remap over the live count
The order is `if bid >= live: exit`, then `pid = remap(bid, live)`. FlyDSL's MXFP4 MoE stage 1 computes its
bound from `cumsum[0]`, tests the raw `bx`, then, when `xcd_swizzle > 0`, applies its XCD remap over that bound
(MOE1:748-774). Turbo exits on the raw id and then remaps over the live tile count
(`xcd_remap_pid_u(t, total_tiles, ...)`, FP8:386-412; `xcd_band_remap_pid`, MX4:614-629).
- **Remap over the grid bound and exit on the remapped id, and the dead ids all land on one XCD.** A
  contiguous remap gives each XCD a slice of `[0, grid)`; the dead range is the top of the last slice, and the
  hardware's static `bid % 8` split cannot rebalance it. *(field, E1)* 32 dead M blocks of 544 with 12 N blocks:
  XCDs 0–6 each ran 816 tiles and XCD 7 ran 432, a makespan 816/768 = 6.25% over ideal. Fixing the order on
  MXFP4 grouped NT raised the score from 1.4292 to 1.4486 (+1.4%), with the tile order inside each XCD
  unchanged.
- **Remap over `live` but exit on the remapped id, and ids at or above `live` map into the live range**: one
  tile is computed twice, another never.
- Signature of a static XCD imbalance, without reading code *(field, E2)*: `SQ_INSTS_MFMA` identical and L2
  requests within 0.007% (hit rate within 0.7 points) across token distributions, `GRBM_GUI_ACTIVE` 13% apart,
  `SQ_BUSY_CU_CYCLES` only 0.7% apart.
  CUs were idle, not slow.
- Check the remap is a bijection for every `live` you can launch (`../gemm/schedule_and_locality.md`
  §2). Which remap to use: `grouped_scheduling.md` §2–§3.

## 5. Tile → group lookup: four forms and their price

| Form | Per-tile cost | Limit |
|---|---|---|
| pre-built tile table (torch ops on the device before the launch), one `(group, first row, rows)` row per M tile (BF16:513-526) | three loads | extra device ops per call |
| SGPR table: `s_buffer_load`, SALU scan and pick (GH:712-738; FP8:357-366) | O(G) SALU on `lgkmcnt` | SGPR budget |
| lane-resident table, wave scan, ballot (GH:741-775; FP8:367-380; MX4:584-637) | O(1) | full EXEC to scan |
| per-tile O(G) compare/select over all groups (MX8:474-527) | O(G) VALU | grows with G |

The bf16 table is built on the device with `cumsum` + `searchsorted` and is as long as the grid bound. Turbo's
fp8 kernel uses the SGPR table only for G ≤ 8 (`_SGPR_GO_MAX_G`, FP8:97-100) and the lane-resident table above
that; the lane table takes `ceil((G+1)/64)` VGPRs.

**Lane-resident table** (the default for a new kernel):
- Entry i sits in lane `i % 64` of chunk `i // 64`. Each lane computes its group's tile count,
  `ceil(len_g / BLOCK_M) * n_blocks`; lanes past G hold 0.
- Inclusive wave scan in six DPP steps: `row_shr` 1, 2, 4, 8, then `row_bcast15` with row mask 0xA and
  `row_bcast31` with row mask 0xC, `bound_ctrl` set so shifted-in lanes add 0 (`_wave_prefix_add_i32`,
  PR:404-411). It needs a full EXEC mask, so run it at kernel entry, before any branch.
- The owner of tile t is the number of lanes whose inclusive prefix is ≤ t: one `ballot` plus a popcount
  (`_wave_count_le_i32`, PR:420-430). On a monotone table that count is the group index. An empty group ties
  with its successor and is skipped; lanes past G hold the total and never count.
- `v_readlane` with a wave-uniform lane brings the group's entries into SGPRs, so the address arithmetic
  after the lookup stays scalar.
- *(field, E3)* MXFP8, 14-config table: +1.72% geomean, all of it from the 8 forward/dgrad configs (+2.8–5.0%
  each); the 6 variable-K wgrad configs were unchanged (their group index is a division). *(field, E4)* The chain it
  replaced (serial carry, two LDS tables, a publish barrier, a G-wide compare tree) was worth 5.3% of cycles at
  G = 32. *(field, E5)* MXFP4 NT: +5.4% on an 18-config geomean whose 6 wgrad configs do not run this
  prologue (+7.2% on the worst config); prologue 2153 → 761 instructions, SGPRs 97 → 58, per-tile fixed cost
  12.46 → 10.47 µs and steady-state cost per K phase unchanged in a K-sweep fit of the whole op (scale
  preshuffle included). *(field, E7)* fp8 per-tensor NN dgrad: about 5% faster on all four configs.

**SGPR table.** Loads go through the scalar cache and a consumer waits on `lgkmcnt` (`_sload_i32`, PR:434-445).
The cost is the budget of at most 102 SGPRs per wave (CDNA4 ISA): *(field, E8)* two `[G+1]` tables at G = 32
(66 live scalars on top of the kernel's own) overflowed it.

**Per-tile O(G) scan.** *(field, E9, E10)* On MXFP8 NT at G = 32, replacing an O(G) scan at entry and another per tile
with O(1) lookups halved the VALU instructions (225M → 110M), cut `v_cmp` + `v_cndmask` per tile from 299 to
74 and removed a 16-byte spill; the production version was 1.3–5.4% faster on 8 configs. The signature: the
same `SQ_INSTS_MFMA` as a reference, more VALU and VMEM per tile, a larger per-tile fixed cost in a K sweep
(`grouped_tile_pricing.md` §6) with the per-phase cost unchanged, and the worst result on the
narrowest-N shape, which has the fewest N blocks to share the prologue.

**Do not copy the whole table into every lane.** *(field, E11)* A persistent fp8 kernel that held all `[G+1]`
entries in each thread's VGPRs across its tile loop was 1.8% slower at G = 32 and about 4.5× slower at
G = 256: that form costs G+1 VGPRs per lane, against `ceil((G+1)/64)` for the lane-resident table.

**Monotone group carry (persistent loops).** A workgroup that walks tiles in increasing order only moves
forward through the groups: carry the group index and its bounds, and read the table only when a tile crosses
into the next group. *(field, E12)* fp8 per-tensor NT persistent kernel: +7.7–9.6% on one shape, +2.0–2.8% on
another; the gain grew with the number of experts per unit of K, from about 0 with 8 experts to about 2× at
the highest ratio measured.

**Variable-K wgrad needs no search.** The tile count per group is a compile-time constant, so
`group = tile // TILES_PER_GROUP`, or a rank permutation of it (`grouped_scheduling.md` §5); the tile
then reads its group's two offsets (MX4:1878-1886).

In FlyDSL: `../flydsl/moe.md` § grouped_group_mapping §5.

## 6. Pin every lookup result with `readfirstlane`
- A value that feeds a buffer descriptor (base, record count) or a body-selecting branch must be provably
  wave-uniform. FlyDSL pins the expert id it reads per M block (`v_readfirstlane_b32`, MOE1:122); Turbo pins
  `total_tiles` (FP8:388-389), the row count its bf16 descriptors are sized by (BF16:654-657), and
  every rebased descriptor base and record count (GH:124-128).
- Unpinned, a descriptor lands in VGPRs and every buffer access goes behind a waterfall loop
  (`../gemm/datapath.md` §8). Turbo's bf16 kernel records the other failure: a per-lane row count
  in a descriptor gave a non-deterministic wrong answer rather than a fault (BF16:654-656). Pin the record count
  as well as the base: *(field, E13)* pinning only the base gave different wrong rows on each run of one shape.
- Choose a boundary-body variant with a scalar branch on `readfirstlane(block == last)`, so the whole wave
  takes one body (`grouped_tile_pricing.md` §4).

## 7. 64-bit group bases, 32-bit in-tile offsets
The mechanism (fold the base into the descriptor's 64-bit base, bound `num_records` from it) is in
`../gemm/datapath.md` §8. What the group dimension adds:
- The group base is `m_start × row_stride`, a cumulative offset that outgrows int32 long before one group
  does. *(field, E14)* At G = 256, 1536 rows per group and an 8192-wide output, `m_total × OUT_M` is 3.2e9 > 2^31,
  and the last groups' gradients came out wrong with no error.
  - Fold `m_start × stride` into the 64-bit base and
    bound the descriptor by the group's own span `M_g × stride`, never by the cumulative `m_end × stride`
    (FP8:427-428; `_wgrad_rebase`, FP8:2072-2085).
- Clamp a record count at zero whenever a tile offset is subtracted from it (a signed max, FP8:3442-3451).
  For an empty group, `M_g × stride − tile_offset` is negative; the descriptor builder reads it as unsigned,
  saturates at `0xFFFFFFFF`, and the loads are no longer bounded: a memory fault.
- FlyDSL's bf16-weight MoE kernel folds the per-expert weight base the same way, because the whole `[E, N, K]`
  tensor overflows the 32-bit record count at large expert counts; its MXFP4 and int4 weights keep a
  whole-tensor descriptor (`kernels/moe/moe_2stage_a16wmix/gemm1.py:217-223`).

In FlyDSL: `../flydsl/moe.md` § grouped_group_mapping §7.

## 8. Empty groups
- Forward and dgrad: an empty group contributes zero M tiles. The prefix scan and the ballot lookup skip it
  with no special case (§5), and the record-count clamp of §7 keeps its descriptors bounded.
- Variable-K wgrad: an empty group still owns its whole output slice (zeros at beta=0, untouched at beta=1);
  how Turbo writes it and what the store costs: `grouped_variable_k.md` §11.
- Put empty groups at the start, middle and end, and one group holding almost all tokens, into every gate
  (`grouped_correctness.md`).

## Verify
- The steady-state call path reads no group length on the host (no `.item()`, `.cpu()`, `.tolist()`).
- The exit compares the raw block id with the live count, and the remap's total is the live count.
- The prologue ISA has no G-long chain of `v_cmp`/`v_cndmask` and no spill; with an SGPR table, the SGPR
  count is checked at the largest G.
- Every descriptor field comes from a pinned value: no `s_and_saveexec` loop around the hot loop's loads.
- Record counts are clamped at zero and bounded per group; G ≥ 256 and empty groups are in the gate.

## Related
`grouped_scheduling.md` · `grouped_tile_pricing.md` · `grouped_variable_k.md` ·
`grouped_correctness.md` · `grouped_host_autotune.md` · `grouped_sorted_tokens.md` ·
`../gemm/datapath.md` §8 · `../gemm/schedule_and_locality.md` §2, §7 ·
`../gemm/grid_pricing.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The month is the run's where the record dates it; where it does not, the line says which date it is. "Turbo"
is Primus-Turbo; "gm" is a campaign's geomean score.

- E1 [2026-08, gfx950, FlyDSL ?, N=3, campaign score gm_ratio: geomean over 18 configurations of t_MXFP8 / t_MXFP4, drift-immune interleaved A/B, per-call CUDA-event median] exit on the raw block id and remap over the live tile count: three bench runs 1.44669 / 1.45059 / 1.45115 against 1.42992 / 1.42841 (+1.42%; the goal file gives the first two, 1.4292 → 1.4467 / 1.4506); before the change XCDs 0–6 ran 816 tiles each and XCD 7 ran 432 (32 dead M blocks of 544, counted from the grid; 816/768 = +6.25%); SNR 55.58–55.60 dB, deterministic (Turbo MXFP4 grouped NT, forward and dgrad, gpt-oss G=32, M=131072, N 5760 / 2944, K 2944; campaign 20260803_155024 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/rounds/round-04/agent.txt:3)
- E2 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC, mean of the main kernel's last 8 dispatches] forward down, balanced against heavy tokens: SQ_INSTS_MFMA 36,175,872 on both, TCC_REQ +0.007%, L2 hit rate 79.49% against 78.78% (−0.7 points), HBM read latency +1.9%, SQ_BUSY_CU_CYCLES +0.67%, GRBM_GUI_ACTIVE +13.1% (kernel and campaign of E1, after its round-4 change; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/goal.md:74)
- E3 [2026-07, gfx950, FlyDSL ?, N=2, campaign score: geomean of t_tw / t_mx over 14 configurations, kernel-only (scale preshuffle excluded)] lane-resident DPP wave-scan group lookup in the MXFP8 NT kernel: gm 1.08715 → 1.10632, 1.10531 on a rerun; min 1.01324 → 1.04390 / 1.04090; the 8 forward/dgrad configurations +2.8–5.0% each, the 6 wgrad configurations unchanged (Turbo MXFP8 grouped GEMM against the per-tensor fp8 one, gpt-oss G=32, M=131072; campaign 20260729_093030 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:212)
- E4 [2026-07, gfx950, FlyDSL ?, N=?, ?] myskill records that the chain the lane-resident scan replaced (serial carry, two [G+1] tables in LDS, a publish barrier, a G-wide compare tree) was worth 5.3% of the cycles at G=32; the measurement behind it is not named, and the month is the one its section heading gives (MXFP8 grouped NT; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/12-mxfp8-grouped.md:98)
- E5 [2026-08, gfx950, FlyDSL ?, N=?, the campaign score of E1 (18 configurations: forward, dgrad and wgrad × 3 token distributions × 2 projections) and the ISA dump; F and P from a six-point K sweep of the whole op] the lane-resident scan ported into the MXFP4 NT prologue (the wgrad prologue was already O(1)): prologue 2153 → 761 instructions, SGPRs 97 → 58, per-tile fixed cost F 12.46 → 10.47 µs with P 1.6126 → 1.6153 µs, gm 1.2868 → 1.3568 (+5.4%), min 1.0976 → 1.1771 (+7.2%); the next round found that the sweep timed the whole op, scale preshuffle included, and refit the GEMM kernel alone at F = 7.81 µs; the +9.40% the records also give for this change is round 3's candidate (1.42866) against the round-0 baseline (1.30593), on a tree that carried this scan, the split NT swizzle constants of round 2 and the half-N boundary body of round 3 together (Turbo MXFP4 grouped NT; campaign 20260803_155024 round 1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/memory.md:126)
- E7 [2026-07, gfx950, FlyDSL ?, N=?, `_bench_tw_vs_mx.py` (t_mx / t_tw, drift-immune interleaved)] lane-DPP O(1) group lookup moved into the per-tensor fp8 NN dgrad kernel: dgrad from about −12% to about −7% against MXFP8 on all four dgrad configurations, gm 0.888 → 0.9025, SNR 54.7 dB, byte-deterministic (Turbo fp8 per-tensor grouped NN dgrad `kernel_grouped_nn_persistent`, gpt-oss; a hands-on session recorded 2026-07-29 to 07-31; local:/workspace/code/gpt_oss_docker/.claude/memory/project_tw_nn_beats_mxfp8_campaign.md:19)
- E8 [2026-07, gfx950, FlyDSL ?, N=?, ISA dump] two [G+1] prefix tables held as SGPRs at G=32 (66 live scalars) overflowed the ~102-SGPR budget and were moved to LDS (MXFP8 grouped NT; campaign 20260729_093030 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:122; myskill pitfalls/05:360)
- E9 [2026-07, gfx950, FlyDSL ?, N=?, rocprofv3 PMC and ISA dump of an O(1) probe against the O(G) kernel] SQ_INSTS_VALU 225.3M → 109.8M, buffer_load per tile 307 → 213, v_cmp + v_cndmask per tile 299 → 74, private segment 16 B → 0 (MXFP8 grouped NT, G=32; campaign 20260729_093030 round 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:109; myskill pitfalls/05:333-338)
- E10 [2026-07, gfx950, FlyDSL ?, N=2, campaign score of E3] the production version (one O(G) scan, SGPR prefix compare with a log-G tree reduce, two prefix tables in LDS): the 8 forward/dgrad configurations +1.3–5.4% (campaign 20260729_093030 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:118)
- E11 [2026-07, gfx950, FlyDSL ?, N=?, drift-immune separate-process A/B, outputs byte-identical across arms] the whole [G+1] offset and tile-cumsum table kept in each thread's registers across the persistent tile loop (`hoist_lookup`): G=32 −1.8%, G=64 −4.2%, G=128 −47.8%, G=256 −77.7% (Turbo FlyDSL fp8 per-tensor grouped forward `compile_fp8_grouped_gemm`, K=2880; record dated 2026-07-29; local:/workspace/code/gpt_oss_docker/.claude/memory/project_triton_gg_fp8_tile_cumsum.md:27)
- E12 [2026-07, gfx950, FlyDSL ?, N=?, drift-immune same-process position-balanced interleave of the commit against its parent, outputs byte-identical] monotone group carry in the persistent NT path: Shape A (gate_up forward / down dgrad, K=2880, N=5760) +9.22% balanced, +9.62% skewed, +7.72% with empty groups; Shape B (K=5760, N=2880) +2.80 / +2.04 / +2.65%; across models myskill records Kimi-K2 down +107.84% and Mixtral (8 experts) about 0% (Turbo fp8 per-tensor grouped NT; 4e15c6ce; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_nt_fwd_dgrad_opt.md:22)
- E13 [2026-09, gfx950, FlyDSL ?, N=3, repeated runs of one shape] a descriptor whose base was pinned with readfirstlane but whose extent was a per-lane value: the same shape gave 512, 160 and 320 bad rows across runs; the month is that of the commit (Turbo bf16 grouped GEMM; ed8d7af4 (#486))
- E14 [2026-06, gfx950, FlyDSL ?, N=?, ?] persistent wgrad with `m_start × OUT_M` in int32: at G=256, M_g=1536, OUT_M=8192, `m_total × OUT_M` = 3.2e9 > 2^31 and the last groups' gradients were silently wrong; no run record is kept, and the month is that of the myskill commit that first recorded it, 2563ed1 (Turbo FlyDSL fp8 per-tensor grouped wgrad; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/04-race-vmcnt-correctness.md:125)
