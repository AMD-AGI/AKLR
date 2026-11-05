---
title: "GEMM — the epilogue: what a C store costs on gfx950 and how to make it cheaper"
kind: topic
lever: gemm_epilogue
ops: [gemm]
gens: [gfx950]
dtypes: [fp8, mxfp4, mxfp8, bf16]
regimes: [prefill, training, beta-accumulate, fused-epilogue]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/gemm_helper.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp8_kernel.py
---

# The GEMM epilogue

> "GH" is `primus_turbo/flydsl/utils/gemm_helper.py`, and "FP8"/"FP4"/"MX8" are the dense fp8, MXFP4 and
> MXFP8 kernels under `primus_turbo/flydsl/gemm/`, all at Primus-Turbo main `7615a5a2`. "PS" and "A16" are
> FlyDSL's `kernels/gemm/preshuffle_gemm.py` and `kernels/gemm/gemm_a16w16_gfx950.py`. Numbers marked
> `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end. Use them for
> direction and order of magnitude, not as targets.

## Route here when
- The C store, a beta=1 read-back or a fused activation/quantization takes a visible share of the kernel.
- You are choosing a store width, a cross-lane reorder, or an LDS-staged (CShuffle) epilogue.
- An occupancy-1 kernel has an epilogue that nothing overlaps.

## 1. The facts that set the price

- **MFMA output layout.** In a 16×16 accumulator tile, a lane holds 4 consecutive rows of one column, and
  neighbouring columns sit in neighbouring lanes. A naive bf16 store therefore writes 16 lanes × 2 B =
  **32 B per request**.
- **The gfx950 write request is 64 B, not the 128 B cache line.** *(field, E1)* On a grouped fp8 GEMM,
  `TCP_TCC_WRITE_REQ` matched bytes/32 for scalar stores and bytes/64 for paired stores exactly. Widening
  further to `dwordx2` halved the issued store instructions but left the requests unchanged, and wall time got
  0.85 pp worse. So pairing two columns into one dword already reaches the request floor. Wider stores only
  save instructions.
- **Give each workgroup C column blocks that start and end on 128 B lines.** *(field, E2, E3)* The same bytes stored at a
  144 B instead of a 128 B column pitch cost 2–3% at M=24–96; owning a line-aligned 64-column block plus a slice
  of a separate leftover-column strip (B rows fetched to match) read −0.4% to +3.1% per shape, 11 of 13 faster, in
  a round that also changed the C store's cache policy in some M buckets and dropped an epilogue barrier (the
  barrier alone −0.1% to +0.3%).
- **Pairing pays `64 B ÷ (W × 2 B)` for a W-wide fragment.** For 32-wide MFMA fragments, the scalar store
  already covers 64 B, so pairing has no request savings left there. *(field, E4, E5)* It gained +1.3% on a
  grouped fp8 GEMM (a 32×32×64 atom on the transposed 4(M)×2(N) wave grid, the only grid that keeps the
  pairing), against +2.8–5.0% on 16-wide fragments of the same kernel.
- **At the power cap, store savings show up as clock, not as wall time inside a tile.** *(field, E6)* Deleting a
  grouped fp8 GEMM's whole epilogue at the power cap moved a same-process wall probe by 0 ± 1 pp, while the
  benchmark gained. Price store changes with `TCP_TCC_WRITE_REQ` (the mechanism) plus the real benchmark (the
  size), not with an in-process wall probe.

## 2. Decide whether the store matters before changing it

Run three ablations of the epilogue:
1. **Delete the whole store.** This gives the total cost.
2. **Same bytes, fewer segments.** This measures the request cost.
3. **Same segments, fewer VALU instructions.** This measures the issue cost.

Only (1) moving means the store is byte- or bandwidth-bound, and width or reorder work will not help.
*(field, E7)* In one grouped MXFP4 NT GEMM the time changes were (1) −17.7% to −20.3%, (2) +0.8%, (3) −0.5%,
so its store was byte-bound. Check that an ablation arm writes the same `M×N×2` bytes to the same rows.
*(field, E7)* A probe that collapsed four rows into one segment also cut the written bytes about 6×, and read
as a false request win.

*(field, E8, E9)* The "best" epilogue does not carry over between shape families. A permlane wide store was
optimal on dense 8192²×4096 (16 KB rows, `N % 256 == 0`). The same store was 0.7–4.1% slower on a grouped MXFP4
NT GEMM with 5888 B output rows.

## 3. Store classes, from free to expensive

FlyDSL's guide describes the two ends (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §7)): a **direct store** from the MFMA
layout, and **CShuffle**, which writes the accumulators to LDS row-major, re-reads them as `(MLane=8,
NLane=32)` after a barrier and stores 4-element vectors, for `tile_n ≥ 128`. Turbo keeps one store class per technique
between those two ends (GH:778-1525). Prefer the cheapest class that reaches the 64 B floor.

| Class | How a lane gets adjacent columns | Cost | Where |
|---|---|---|---|
| Scalar | none: `buffer_store_short` per value, column step in the immediate | baseline, 32 B requests | FlyDSL PS direct store; Turbo `StoreCPerTensor`, GH:778 |
| **Row-permuted feed** | permute the **global rows the B fill fetches** so a lane's two n-fragments become adjacent output columns | **zero instructions**: only the prologue's per-lane global offsets change | `StoreCPerTensorPairCol`, GH:1061; `compute_global_swizzle_pair`, GH:256 |
| DPP pair | `v_cvt_pk_bf16_f32` + one `quad_perm` DPP swap + `v_perm_b32` | 1 cvt + 1 DPP + 1 perm per pair | `StoreCPerTensorPairN`, GH:1113 |
| Row merge | one `v_permlane16_swap_b32` per packed fragment pair moves the second fragment into the other 32-lane half, so one store covers a whole 64 B row run | 1 permlane per pair | `StoreCPerTensorRowN`, GH:1004 |
| Quad/fold | the row permutation folds 4 or 8 columns into one lane: one `dwordx2`/`dwordx4` | offsets only | `StoreCPerTensorQuadN`, GH:1251 |
| Line via LDS | transpose through LDS with `ds_read_b64_tr_b16` into a whole 128 B line per fragment row; scratch borrowed from an operand buffer | LDS traffic + fence | `StoreCPerTensorLineN`, GH:1180 |
| CShuffle | LDS staging, then vector stores | LDS + barriers | FlyDSL A16 `write_cshuffle_vec_to_global`, `c_shuffle_epilog`; Turbo `StoreCPerTensorCShuffle`, GH:1335 |
| Transposed accumulator | swap the MFMA operands so the accumulator holds Cᵀ; two `permlane16_swap`s give 8 contiguous columns, one `dwordx4` | 2 permlanes per 8 columns | FP4 `store_tacc_wide`, FP4:513 |

How to choose:
- **First, see whether the reorder can be an address change.** When the output column is the B operand's
  slow axis (NT: output column = B row), permute B's rows in the G2S fill using the involution
  `W·t + m ↔ 2·m + t` within each `2W`-row block. The fill already takes arbitrary per-lane global offsets,
  and the LDS swizzle key still follows the LDS row, so the reader does not change. *(field, E10)* This gave
  +2.4–2.8% on a grouped fp8 NT GEMM (three token distributions): `buffer_store_short` 160→32,
  `buffer_store_dword` 0→64, and `v_cvt_pk` 160→96. The MFMA, barrier, VGPR and LDS counts were unchanged.
- **When the output column is the operand's fast axis (NN/TN), use the row merge.** The transpose read's
  lane→column map is fixed by hardware, and a 16 B fill chunk holds 16 whole columns, so neither can be
  permuted. Pack each row pair (`v_cvt_pk_bf16_f32`) first, then swap the packed dwords. *(field, E11)* Swapping
  unpacked f32 values on a grouped fp8 GEMM took 64 permlanes and gained +0.13%; swapping packed pairs took 32
  and gained +0.43–0.83% (three token distributions).
  **Price a cross-lane primitive by the output values it moves per instruction.**
- *(field, E12, E9)* On grouped GEMMs of the same family, every other option that paid cross-lane VALU or an LDS round
  trip for a wider store lost: CShuffle −21% on that dgrad NN kernel (−24% on its forward NT), DPP pairing −1.1%
  there and −1.9% on its TN weight-gradient kernel, a transposed-accumulator wide store −0.7 to −4.1% on a grouped
  MXFP4 NT GEMM with nearly the same output row stride. The gains were the free address permutation and the packed-row merge above.
- CShuffle is not only for `tile_n ≥ 128`: race it against the direct store on narrow tiles too. *(field, E13, E14, E15)* At 128–256-wide
  bf16 tiles it can decide the kernel: direct register stores lost 9–50% to it on two kernels (8 B stores where
  the width is recorded); on 96-wide tiles that store 72 columns (80-wide above M=96), direct 8–16 B stores lost
  2–14% to it at M ≥ 56 and were within 2% of it either way at M ≤ 48.
- **Feeding CShuffle, swap the MFMA operands:** a lane then holds 4 adjacent columns, so each fragment is one 8 B
  LDS write (64 `ds_write_b16` → 8 `ds_write2_b64` per wave), no cross-lane op. *(field, E16)* +0.3% to +3.2% per
  shape, all 17 faster (the same kept round also gave M=33–60 a single 64×32 row block).

**Instruction-level rules for cross-lane stores**
- A `permlane16_swap` result consumed only by a `buffer_store` (VMEM) needs no `s_nop`. That hazard covers a
  following VALU read (FP4:500-502).
- When a VALU consumes the result, use the `rocdl.permlane16_swap` / `permlane32_swap` intrinsics, not inline
  asm. The hazard recognizer cannot see asm, so asm forces an `s_nop` on both sides. *(field, E17)* Switching to the
  intrinsic cut `s_nop` from 412 to 174 in a fused bias + GELU epilogue (683 to 24 with FP6 row quantization fused
  as well).
- Issue all swaps first, then all stores, so the permlanes and the stores are far apart. *(field, E18)* Only the two
  changes together, dropping the hard-coded `s_nop` and the two-phase order, took the count from 80 to 0.
- **Never store the high half of an f32 accumulator as bf16.** That is truncation, not round-to-nearest-even.
  - *(field, E19)* It cost 5.8 dB SNR and made the store slower.
  - Convert with `v_cvt_pk_bf16_f32`.
- On gfx950, a multi-dword MUBUF VGPR tuple must be 64-bit aligned.

In FlyDSL: `../flydsl/gemm.md` § epilogue §3.

## 4. Overlap the store with the last MFMAs

At one wave per SIMD nothing overlaps the epilogue unless you put it inside MFMA shadows.
- **8-wave.** Store quadrants c00/c01 between the c10/c11 MFMAs of the last iteration (FP8:383-400).
  MXFP8 laps two quadrants into the last K iteration. "Past 2 the store temporaries outlive an MFMA pair
  and VGPRs spill" (`_MX_EPI_LAP`, MX8:91-93).
- **Whole loop.** Stop the asm loop one phase early and hand the last phase's MFMAs back as SSA values
  (`mainloop.md` §3). Issue them row-tile by row-tile and put each row-tile's store after the
  *next* row-tile's MFMAs. *(field, E20)* Distance 1 was best on a grouped GEMM; 2 and 4 were worse.
  - The hazard recognizer cannot see asm MFMAs. Put a `sched_barrier(0)` before each store group and
    trailing `s_nop`s after the last row-tile. Otherwise a VALU reads a stale accumulator, and only the last
    row-tile is wrong.
  - The ceiling is one K-block of MFMAs. *(field, E20)* That recovered an estimated 10% of the store cost on a
    grouped weight-gradient GEMM with 32 K-blocks per tile (+0.7–1.1%). *(field, E31)* On a grouped dgrad kernel
    whose last window without G2S held 8 MFMAs, it gained +0.02%, within noise.
- **Fold the store into a phase with no G2S.** gfx950 has a single in-order `vmcnt` for loads and stores,
  so a store in flight is serialized against operand DMA waits. FP4 therefore puts its fused C store in a
  peeled tail phase that issues no fills, and paces it at `_CRATE` lines per MFMA (FP4:776, 1095-1233,
  1745). Spent accumulators can be recycled as pack registers when the caller never reads them
  (`acc_dead`, FP4:679).

## 5. Cache policy on the store

- **C is write-once: race `nt` (aux=2) and `sc1` (aux=16) against the default.** Turbo stores it `nt` so it does
  not evict the A/B band the L2 tile order keeps (FP8:72-73). *(field, E21)* On a short-K bf16 GEMM, `sc1` won at
  every M (+3.1% to +20.3% per shape, all 17 faster) and `nt` lost at the largest M. *(field, E22)* On another
  (K=512, M ≈ 35k–340k), `nt` beat `sc1` by 0–10% and the default by 1–17%.
- **Give `nt` only to stores that write whole 128 B lines.** Sub-line stores (2 bytes per lane) rely on L2 to merge
  their 32 B pieces into 64 B writes, so `nt` and `sc0` both cost there. *(field, E32)* On a dense MXFP4 NT GEMM, on
  the 10 shapes whose A and B fitted the LLC and C did not, the default policy and `sc0` both lost to `nt` on every
  shape, by up to 6.4%. *(field, E34)* On another dense MXFP4 GEMM, whose A and B also fit the LLC and C did not
  (estimated from the shapes), all eight `sc0`/`nt`/`sc1` combinations of its folded C stores read within 0.56% and
  0.64% on two training shapes, inside the 2.0% spread of the default's own two reads: fitting A and B in the LLC does
  not by itself predict that `nt` pays.
- **Scope bits send even whole-line writes past the per-XCD L2 that merges them.** *(field, E33)* On a grouped
  MXFP4 GEMM whose C stores fill whole lines, `sc1` alone or added to `nt` cost 4–5% on all six weight-gradient
  cells. Turbo uses system scope `sc0|sc1|nt` (aux=19; agent scope is `sc1` alone) only for a beta=1 burst from
  one tile per workgroup that is too large for L2, switching at 26 MiB of C (`_cstore_aux`, FP8:74-77, 1885);
  its comment places that about where the burst stops fitting in L2 beside the operand band.
- **The sign is shape-dependent, so race it.** *(field, E23)* In the MXFP8 GEMM, `nt` cut the wide shapes by 4.6%
  and 3.0% and cost 0.9–1.9% on narrow shapes whose slabs were already L2-resident. It is raced at a 1%
  margin, not pinned (MX8:968-974).
- **Never mark byte-granular scale stores (E8M0) non-temporal.** They rely on L2 merging the write masks.
  - *(field, E24)* In the MXFP8 cast kernels that cost 15–40%.
- Keep split-K partials cached only if the same XCD reduces them; reduced on other XCDs, `sc1` stores (aux=16) were
  *(field, E25)* 2–4% faster. Do not mark partials `nt`: aux 18/19 cost *(field, E26)* 3–6% against aux 17
  (`sc0|sc1`).

In FlyDSL: `../flydsl/gemm.md` § epilogue §5.

## 6. Bounds without per-column masks

- Rebase the C descriptor per row band in i64 and pin base and `num_records` with `readfirstlane`
  (`make_row_band_resource`, GH:2010). A wide store has no per-column mask, so a whole column span is
  dropped by giving its band 0 records. Store classes that span several fragments require
  `N % tile_n == 0` (`col_safe`).
- Apply the predicate to the part of the offset that does not depend on the sub-tile, then add the sub-tile row
  delta. *(field, E27)* Otherwise `v_cndmask_b32` went from 14 to 64 in a grouped GEMM's fused epilogue.
- One descriptor per tensor per tile is enough when sub-tiles differ only by a compile-time row offset.
  Carry the row delta in the offset instead of building a descriptor per sub-tile.

In FlyDSL: `../flydsl/gemm.md` § epilogue §6.

## 7. beta = 1 (accumulate into C)

- Pipeline the read-back: load quadrant i+1 of C while quadrant i's store is in flight (`_store_quadrants`, GH:1528).
  Add in f32 so the result is rounded once.
- **Compile and race against a scratch output.** Turbo races at beta=0 and rebuilds the
  winning configuration with the accumulate epilogue (FP8:3008-3027).
- A persistent kernel that repeats its last tile for leftover workgroup ids is fine for overwrite stores
  but double-accumulates at beta=1. Turbo forces one tile per workgroup there.

In FlyDSL: `../flydsl/gemm.md` § epilogue §7.

## 8. Fused epilogues (activation, GLU, quantization)

- **Price the fusion before tuning inside it.**
  - **Tax** = fused kernel minus the same-shape bare GEMM, from one kernel trace. The two kernels often
    share a symbol name; group them by `vgpr_count`, `lds_size` and grid.
  - Then compare the tax with the unfused chain.
  - *(field, E28)* An MXFP4 dGELU + dual-quant epilogue cost 101.9 µs inside the GEMM. The standalone quantizer
    took 100.8 µs, so the fusion broke even. The epilogue inherited the GEMM's one workgroup per CU, which
    dropped its achieved bandwidth from an estimated 4.6 to 3.0 TB/s (435 MiB in 98.4 µs standalone, 294 MiB in
    101.9 µs fused).
- **A static VALU instruction count is not a bottleneck diagnosis.** Use it to confirm a mechanism landed,
  and test causality by deleting instructions and watching wall time. *(field, E29)* In the case above, deleting
  192 VALU (2.7% of the kernel) changed nothing.
- **Measure how much VALU the main loop can absorb before moving work into it.** Insert N dependent FMAs in
  the K-loop gaps with a side-effect sink. *(field, E30)* At occupancy 1, the first 8 FMAs were 65% absorbed and the
  next 8 only 25%. The free pool was about 38% of the epilogue, about 4.8× less than a static estimate.
- **Two co-resident workgroups (occupancy 2) need `LDS ≤ 80 KiB` per workgroup and
  `arch + AGPR ≤ 256` registers per wave, for 4-wave workgroups.** Check both before planning to hide the epilogue behind a second workgroup.
- **Make fused operands meet in one lane.** For GLU (B = gate‖up, `[2I, K]`), point the second B pool at the
  *up* band, `I` weight rows further in, instead of at the next columns. `gate[m, j]` and `up[m, j]` then land
  in the same lane at the same fragment index, and the epilogue pairs them without a shuffle (FP4:2281-2283,
  `gemm_mxfp4_glu_quant_flydsl_kernel`).
- Turbo's fused GLU/dGLU + MXFP4 quantization epilogues are bit-exact against the standalone quantizer
  (`primus_turbo/flydsl/utils/gemm_epilogue_helper.py`).

## Related
`mainloop.md` (peeled phases, register plan) · `datapath.md` (row-permuted
fills, i64 descriptors) · `schedule_and_locality.md` (L2 order the `nt` policy protects) ·
`autotune.md` (racing the store policy) · `correctness.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The Kimi-K3 forge runs installed `flydsl==0.3.2`. The month is the run's where the record dates it; where it
does not, the line says which date it is. "Turbo" is Primus-Turbo. A secondary reference without a directory sits in the
directory of the line's `local:` path.

- E1 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 counters per arm (`TCP_TCC_WRITE_REQ`, `TA_BUFFER_WRITE_WAVEFRONTS`); the dwordx2 arm's wall ruler is not stated] scalar bf16 C stores: 2.359e7 write requests against 755 MB ÷ 32 B = 2.36e7; paired dword stores: 1.206e7 against a predicted 1.206e7; widened to dwordx2 (`nt_cpack=4`): `TA_BUFFER_WRITE_WAVEFRONTS` 3.015e6 → 1.573e6, `TCP_TCC_WRITE_REQ` still 1.206e7, and +2.25% against the pair's +3.10% (−0.85 pp) (gpt-oss-20b down-projection forward NT, grouped fp8 tensorwise GEMM, G=32, M=131072, N=2880, K padded to 2944; campaign 20260817_065611 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/goal.md:204; memory.md:232)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=5, own harness on AKLR bench_cudagraph, arm order alternated, 5 rounds; timing-only arms whose output was wrong by design] storing 64 of the 72 columns at a 64-column (128 B) pitch read 1.0248 / 1.0378 / 1.0398 at m=24/64/96 against 0.9962 / 1.0115 / 1.0173 at the original 72-column (144 B) pitch (ratios incumbent ÷ variant) (gemm_n18432_k1536 forge run, iter 5, bf16, 72-column N tiles; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n18432_k1536/forge_experiments/lessons/iter_005.md:22)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode] the kept round that gave each workgroup a line-aligned 64-column block plus a slice of a leftover-column strip, also dropped an epilogue barrier and set the C store's aux per M bucket (0 at m ≤ 64, 17 above; the parent stored with 17 in its 32–96 buckets): +1.31% [+1.18, +1.45] over its parent, −0.42% (m=8) to +3.09% (m=48), 11 of 13 cases up; in the round's own harness the barrier drop alone read 0.9994–1.0029 (gemm_n18432_k1536, kept change c6b6e42 (iter 6); local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n18432_k1536/forge_experiments/ablation.json:241; lessons/iter_006.md:25)
- E4 [2026-09, gfx950, FlyDSL 0.2.2, N=?, same-process paired readings in both orders, 16 launches per window, with an inert control arm reading ±0.06–0.31%] with a 32×32×64 atom on a transposed 4(M)×2(N) wave grid, the build with pairing read −24.4% against the shipped kernel and the same build without it −25.7%: pairing was worth +1.3% there; that wave grid is the only one on which this atom keeps the paired store (gpt-oss-20b grouped fp8 tensorwise GEMM at B=24, NT forward; campaign 20260904_151102 rounds 13 and 17; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/rounds/round-13/agent.txt:15; memory.md:396)
- E5 [2026-09, gfx950, FlyDSL 0.2.2, N=?, the ruler of E4] switching the 16-wide pairing off (`nt_pairc=False`) on the shipped kernel cost +2.80 / +3.03 / +3.28 / +3.53% on the four NT forward cells (round 12); later rounds on the same kernel read +4.3% (round 13, `pairc0`) and +4.98 / +4.49% (round 17, both arms with pairing off) (the kernel of E4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/memory.md:346; rounds/round-13/agent.txt:15; memory.md:396)
- E6 [2026-09, gfx950, FlyDSL 0.2.2, N=?, same-process wall probe with drop and alias arms; the board at its 1400 W cap, full-load sclk 1723–1743 MHz] deleting the whole epilogue moved wall by 0% ± 1 pp on four cells (bal/B8 +1.53%, B24 −0.36%, B32 +0.36%); the same round's row-merge store change raised the benchmark's dgrad legs from 0.9739–0.9984 to 0.9985–1.0336 (gpt-oss-20b grouped fp8 tensorwise GEMM, non-persistent; campaign 20260904_151102 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/rounds/round-06/agent.txt:67)
- E7 [2026-08, gfx950, FlyDSL ?, N=?, per-tile time from a 6-point K-sweep fit t_tile = F + n·P and the probe's TF/s at K=2944, all arms in one run] deleting the C store: F 7.943 → 2.598 µs, 4.87 µs = 20.3% of the tile (probe 3721.7 → 4672.2 TF/s); same bytes and rows with write segments 4 → 1: +0.8%; 254 `v_accvgpr_read` + 254 `cvt_pk` removed with all 256 stores kept: −0.5%; a segment probe that folded four rows into one read −2.58 µs but wrote ~6× fewer bytes; a later round's counter run deleting the whole store read 579.2 → 476.5 µs (−17.7%) (gpt-oss MXFP4 grouped GEMM, NT forward, G=32; campaign 20260804_140240, round 8 after its replan and round 15; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_140240/rounds/round-08/agent.txt:5; memory.md:342)
- E8 [2026-06, gfx950, FlyDSL ?, N=?, NO_AUTOTUNE, time FlyDSL ÷ AITER, lower is faster] at 8192²×4096 the permlane16_swap + dwordx4 store (TACCW) read 0.998 against 1.039 for the default store and 1.070 for a ds_bpermute dwordx2 store; at 16384³ the three read 1.054–1.055 (Turbo MXFP4 4-wave dense GEMM; record dated 2026-06-29 in the file; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_epilogue_store.md:421)
- E9 [2026-08, gfx950, FlyDSL ?, N=?, ruler not stated (the campaign scores gm_ratio against the MXFP8 grouped GEMM)] the TACCW wide store was slower by 1.34% (fwd down), 0.70% (fwd gate_up) and 1.07% (dgrad gate_up); in the next campaign on the same kernel the tree's `store_tacc_wide` read +4.1% slower in the NT probe of E7, +3.5% with its software pipeline cut to depth 4 (gpt-oss MXFP4 grouped GEMM, NT, G=32, M=131072, N=2944/5760, 5888 B output rows; campaign 20260803_155024 round 1 and campaign 20260804_140240 round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/rounds/round-01/agent.txt:67; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_140240/rounds/round-08/agent.txt:6)
- E10 [2026-08, gfx950, FlyDSL ?, N=3, kernel-level palindrome A/B, 40 rounds per token distribution, 3 distributions, shared ref arm; ISA counts from the build] row-permuted B fill with paired dword C stores: +2.77 / +2.65 / +2.44% over the previous kept kernel, every round positive; `buffer_store_short` 160 → 32, `buffer_store_dword` 0 → 64, `v_cvt_pk` 160 → 96; `v_mfma` 920, `s_barrier` 304, `.vgpr_count` 256, spill 0, LDS 131072 unchanged (the kernel of E1; campaign 20260817_065611 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/memory.md:230; goal.md:200)
- E11 [2026-08, gfx950, FlyDSL ?, N=3, the ruler of E10 for the kept arm; the f32 arm's repeat count is not stated] row merge swapping f32 values one by one (64 permlanes): +0.13%; swapping packed row pairs (32 permlanes, `v_cvt_pk` 160 → 96): +0.83 / +0.43 / +0.62% (dgrad NN of the grouped fp8 GEMM of E1; campaign 20260817_065611 round 12; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/memory.md:244)
- E12 [2026-08, gfx950, FlyDSL ?, N=?, isolated screening runs (round 2) and the round's paired readings (round 7)] `store_cshuffle` (LDS-staged wide store): dgrad NN −21%, forward NT −24%; a DPP `quad_perm` + `v_perm` paired dword store: NN dgrad −1.09%, TN weight gradient −1.9% (the grouped fp8 GEMM family of E1, which the campaign's goal groups with the MXFP4 kernel of E9 as one family of about 5760–5888 B rows; campaign 20260817_065611 rounds 2 and 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/rounds/round-02/agent.txt:90; memory.md:178; goal.md:289)
- E13 [2026-10, gfx950, FlyDSL 0.3.2, N=?, own CUDA-graph harness (512 MB flush, alternating order, median of 5–7) in iter 7; scratch paired harness on AKLR's graph timer (3–5 interleaved rounds per arm, median delta) in iter 13] C stored straight from registers against the LDS-staged store: 8 B stores in the non-persistent kernel 195.0 / 121.2 against 165.0 / 102.5 µs at m=16384 / 9397; on the large rows +31.57 / +20.39 / +16.63 / +5.39 µs at m=16384 / 9397 / 2732 / 893, against the run's final per-case times 157.4 / 104.0 / 33.5 / 14.7 µs (about +18% to +50%) (gemm_n7168_k768 forge run, bf16; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k768/forge_experiments/lessons/iter_007.md:46; lessons/iter_013.md:12; ../run_state.json:85)
- E14 [2026-10, gfx950, FlyDSL 0.3.2, N=3, rocprofv3 kernel trace, median of 100 calls with an L2 flush before each, 3 repeats (iter 11); in-process harness on bench_cudagraph, median of 5–9 (iter 13)] C stored straight from registers: m=893 28.90 vs 26.56 µs, m=2732 60.86 vs 55.18 (iter 11); m=16384 299.5 vs 269.2, m=893 29.45 vs 26.10, m=2732 61.54 vs 55.50 (iter 13), +9% to +13%; the direct arm's store width is not recorded (gemm_n7168_k1536 forge run, bf16; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k1536/forge_experiments/lessons/iter_011.md:58; lessons/iter_013.md:58)
- E15 [2026-10, gfx950, FlyDSL 0.3.2, N=6, own harness on cudagraph with an L2 flush, alternating order, 6 rounds (two 6-round runs for the 8 B arm); incumbent against itself 0.9979–1.0013] ratios LDS-staged incumbent ÷ variant: 8 B per-fragment register store 0.8810–0.9414 at m=56–112, 0.9839–0.9953 at m=24–48, 0.9993–1.0055 at m=8–16; 16 B direct store 0.9446–0.9803 at m=56–112, 0.9977–1.0048 at m ≤ 48 (gemm_n18432_k1536 forge run, bf16, 96-wide tiles storing 72 columns, 80-wide above m=96; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n18432_k1536/forge_experiments/lessons/iter_010.md:7)
- E16 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode; LDS write counts from the distill step's probe compile of both arms (128×128 tile, m=2732)] the kept round that swapped the MFMA operands for 8 B LDS writes, and also gave m=33–60 a single 64×32 row block: +1.33% [+1.17, +1.49], all 17 cases faster, +0.26% (m=64) to +3.20% (m=16384); 64 `ds_write_b16` → 8 `ds_write2_b64` per wave (gemm_n7168_k1536, kept change 9d37aa3 (iter 6); local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k1536/forge_experiments/ablation.json:174; local:/workspace/code/gpt_oss_docker/work/sikl/kb_evidence/sources/k3_distill/gemm_n7168_k1536.distill.md:14)
- E17 [2026-10, gfx950, FlyDSL ?, N=1, ISA of one build per arm] permlane swaps through `rocdl.permlane16_swap` / `permlane32_swap` instead of inline asm: `s_nop` 412 → 174 in the scored arm, 683 → 24 in the fused row-pack arm, `v_*` unchanged (Flux A6W4 forward GEMM: the scored arm with a fused bias + GELU epilogue, the row-pack arm also quantizing rows to FP6 (E2M3) in the epilogue; campaign 20261012_143244 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_143244/memory.md:225; rounds/round-04/agent.txt:7; goal.md:32; goal.md:60; memory.md:199)
- E18 [2026-06, gfx950, FlyDSL ?, N=1, ISA of one build per arm] TACCW epilogue with the hard-coded `s_nop 1` removed and a two-phase order (all cvt + permlane first, then all stores): `s_nop` 80 → 0; removing the `s_nop` alone left 19, the two-phase order alone 64 (Turbo MXFP4 4-wave dense GEMM; record dated 2026-06-29 in the file; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_epilogue_store.md:438)
- E19 [2026-09, gfx950, FlyDSL 0.2.4, N=1, one campaign score for the truncating arm (gemm18: geomean over 18 cells against a frozen baseline), three for the comparison arm] BLOCK_N=192 with the in-loop folded store falling back to `buffer_store_short_d16_hi` of the accumulator: SNR 55.6 → 49.77 dB, `buffer_store` 104 → 200, `gptoss_out_proj_wgrad` 5022.5 TF/s and gemm18 1.0067, against 5108.7 / 5164.9 / 5120.9 TF/s and 1.0080 / 1.0075 / 1.0087 with the C store handed back to the epilogue's round-to-nearest-even path (Turbo MXFP4 dense GEMM; campaign 20260925_144843 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/memory.md:149)
- E20 [2026-08, gfx950, FlyDSL ?, N=?, drift-immune same-process probe; bench.sh with 3 + 3 alternating reps; store cost from a no-store control with the cache off] the last K-block peeled out of the whole-loop asm, each row-tile's store after the next row-tile's MFMAs: distance 1 down +1.03 / +1.12 / +0.92%, gate_up +0.95 / +0.72%; distance 2 +0.07 / +0.75 / +0.92%; distance 4 +0.62%; bench best +0.90%, median +0.95%; the store cost 8.31% (gate_up) and 10.16% (down) of the kernel at the deployed shape (gpt-oss-20b grouped fp8 tensorwise weight-gradient GEMM, G=32, per-expert M=4096, 32 K-blocks per tile; campaign 20260810_104512 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_104512/memory.md:164; memory.md:166)
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop ablation ladder row marked recorded (the campaign's paired KEEP timing), driver --bench-mode; per-shape µs from the round's AKLR graph timer, best of 2 medians of 3] C stored with `sc1` (aux=16) in every M bucket: +4.62% when kept, +5.54% recorded, bounds +4.40 … +6.68%, all 17 cases faster, +3.13% to +20.27% per case; at m=16384 the default read 15.19 µs and `nt` 15.37 (gemm_n1536_k128 forge run, bf16, K=128; kept change b118aca (iter 3); local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n1536_k128/forge_experiments/ablation.json:132; lessons/iter_002.md:68)
- E22 [2026-10, gfx950, FlyDSL 0.3.2, N=?, own harness: a CUDA graph of 10 launches, median of 10 replays, noted as noisy; GROUP_M=1] C store default / `sc1` / `nt`: m=35107 0.1286 / 0.1259 / 0.1252–0.1275 ms, m=104093 0.3785 / 0.3510 / 0.3136–0.3201, m=342073 1.1854 / 1.1084 / 1.0774–1.0784 (gemm_n3072_k512 forge run, bf16, K=512, 256×256 tile; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3072_k512/forge_experiments/lessons/iter_001.md:24)
- E23 [2026-09, gfx950, FlyDSL ?, N=?, kernel-only timing, repeat count not stated; the month is the commit's] `nt` on the MXFP8 GEMM's C store as an autotuned axis: linear1 −4.6%, linear2 −3.0% kernel-only, 0.9–1.9% slower on the narrow shapes whose slabs are already L2-resident; raced at a 1% margin (Turbo MXFP8 dense NT GEMM, the training step's five production linears; Primus-Turbo d522ee0a (#505))
- E24 [2026-09, gfx950, FlyDSL ?, N=?, ruler not stated; the month is the commit's] a non-temporal hint on the byte-granular E8M0 scale stores cost −15 … −40% across the thirteen production casts (Turbo MXFP8 dual-cast quantization kernels, `mxfp8_quant_flydsl.py`; Primus-Turbo d522ee0a (#505))
- E25 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode] split-K workspace partials stored with `sc1` (aux=16) for ma ≥ 3, with the K splits spread across XCDs, in a kept round that also used a 3-wide N tile at m=7: +2.12% [+2.02, +2.22] over its parent; m=35–98 +2.0% to +3.6%; m=14–28, which do not take that path, −0.06% to −0.21%; m=7 +4.5% (gemm_n768_k7168 forge run, kept change 23cee34 (iter 12); AKLR 2a4328229; local:/workspace/code/gpt_oss_docker/work/sikl/kb_evidence/sources/distill_12h/ablation.json:228)
- E26 [2026-10, gfx950, FlyDSL 0.3.2, N=4, own A/B races on AKLR bench_cudagraph (rep=20) over AKLR's 8 timing input sets, 4–8 alternating rounds, median; HEAD against HEAD 0.996–1.006] split-K plane stores with aux 18 / 19: 0.947–0.967 of the incumbent's aux 17 at m=8/40/64/112/893/2732; aux 0 0.982–0.998 at m=24/40/64/112 (gemm_n144_k7168 forge run, iter 10, `_PLANE_AUX=17` held fixed; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n144_k7168/forge_experiments/lessons/iter_010.md:50)
- E27 [2026-09, gfx950, FlyDSL 0.2.4, N=1, ISA of one build per arm] buffer resources hoisted from per sub-tile to per tile with the column predicate left on the load mask: `v_cndmask_b32_e64` +50 (14 → 64); with the predicate applied to the sub-tile-independent offset (`_drop_col`), the fc2 epilogue lost 126 / 112 / 111 instructions (MoE FC2 dgrad, a grouped MXFP4 GEMM with a fused dGLU + dual MXFP4 quantization epilogue, `StoreCdSwiGLUQuadQuant`; campaign 20260923_120547; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_120547/memory.md:442)
- E28 [2026-10, gfx950, FlyDSL ?, N=?, four kernels timed in one run (`_dgelu_ledger.py`); bandwidths are bytes ÷ time] pure GEMM 132.5, standalone quantizer 100.8, unfused pair 231.1, fused 234.4 µs, so the epilogue costs 101.9 µs inside the GEMM while traffic falls 627 → 294 MiB; the standalone chain up to dGELU took 98.4 µs for 435 MiB, which the record calls 4.3 TB/s, and the record puts the fused epilogue at 2.9 TB/s (Flux MXFP4 dgrad GEMM fc2_dgrad 8192×12288×3072 with a fused dGELU + dual MXFP4 quantization epilogue; record dated 2026-10-13; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flux_fuse_gemm_act_quant.md:117; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/07-epilogue-addressing-transpose.md:358)
- E29 [2026-10, gfx950, FlyDSL ?, N=?, ruler not stated] fp4 pk8 packing removed 192 instructions, 2.7% of the kernel: fc2 changed 0 (the kernel of E28; campaign 20261012_105732 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_105732/rounds/round-04/agent.txt:8)
- E30 [2026-10, gfx950, FlyDSL ?, N=?, campaign score `_bench_gelu_epi.py` (training-time-weighted TF/s of two shapes), ABBA order with a 20 s sleep between arms, runtime cache off] 0, 8 and 16 dependent FMAs in the K-loop gaps with a side-effect sink: 2844.97 TF/s (fc1_double 217.6 µs), 2693.07 (−5.34%, 227.9 µs), 2515.32 (−11.6%, 247.2 µs); about 1.53 cycles per instruction for the first 8 (65% absorbed) and 3.23 for the next 8 (25%); a free pool of about 5.0e4 cycles ≈ 25–28 µs, 38% of the 68 µs epilogue; the goal's static estimate was about 240k idle VALU issue cycles (A6W4 forward GEMM, 256×384 tile, one wave per SIMD, fused GELU epilogue; campaign 20261012_073126 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_073126/memory.md:90; goal.md:78)
- E31 [2026-08, gfx950, FlyDSL ?, N=?, ruler not stated for this arm; the tail window counted in the round's ISA gate] the last K-block peeled out of the whole-loop asm, each row-tile's store after the next row-tile's MFMAs (distance 1): +0.02%, within noise, VGPR 96 → 128; distance 2 −6.26% with a spill; the structural tail window without G2S held 8 MFMAs (dgrad NN of the grouped fp8 GEMM of E1; campaign 20260817_065611 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/memory.md:196; memory.md:198)
- E32 [2026-09, gfx950, FlyDSL ?, N=1, the 18-shape campaign score (gemm18), one run per arm; the 8 deep-K shapes, whose code did not change, as in-run control (+0.31% / +0.19%, max drift 1.59% / 0.49%)] C store policy on the 10 shallow-K shapes (K ≤ 6144) set to aux 1 (`sc0`, without `nt`) or aux 0 (default write-back) instead of `nt`: all 10 slower in both arms, worst QKV_dgrad −6.42% / −5.17%, out_proj_fwd −3.56% / −3.24%; A (67 MB) + B (8 MB) fit the 256 MB LLC, C (268 MB) does not (Turbo MXFP4 dense NT GEMM, gemm18 campaign round 10; note dated 2026-09-26 in the file; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:680)
- E33 [2026-08, gfx950, FlyDSL ?, N=3, interleaved A/B, 3 runs per arm (count from the myskill note), campaign score gm_ratio_pct against the large-M reference] on the folded C store whose 16 lanes fill a 128 B line (`buffer_store_dwordx2`) with `nt`: `nt sc1`, `sc1` alone and `sc0 sc1 nt` read 98.32 / 98.31 / 99.43 against 100.49–100.69, all six wgrad cells 4–5% lower; in round 3 `sc1`+`nt` read −0.5% to −1.8% on the deploy wgrad cells while the untouched reference cells ran 2.5% faster (gpt-oss-20b MXFP4 grouped GEMM, G=32, per-expert M=4096, skewed routing, wgrad; campaign 20260811_114814 rounds 13 and 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260811_114814/rounds/round-13/agent.txt:16; rounds/round-13/agent.txt:9; memory.md:150; memory.md:151; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:671)
- E34 [2026-09, gfx950, FlyDSL ?, N=2, `us_min` per arm, ABBA twice, FlyDSL disk cache cleared before each value, each arm's stores checked in the ISA dump] all eight `sc0`/`nt`/`sc1` combinations (aux 0–3, 16–19) of the folded C stores: linear1_fwd 0.4440–0.4465 ms, a 0.56% spread, against 2.0% between the aux=2 default's own two reads; linear2_dgrad 0.64% spread (Flux MXFP4 dense GEMM, 16384×21504×3072 and 16384×15360×3072, round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260927_035256/rounds/round-09/agent.txt:25; :56)
