---
title: "GEMM — the epilogue: what a C store costs on gfx950 and how to make it cheaper"
kind: topic
lever: gemm_epilogue
ops: [gemm, grouped_gemm, fused_moe]
gens: [gfx950]
dtypes: [fp8, mxfp4, mxfp8, bf16]
regimes: [large-M, "M 256-4096 (bf16)", beta-accumulate, fused-epilogue]
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

*(field, E43)* On four bf16 GEMMs at m 512–4096, deleting only the final global C store took **6.322–15.751% off
the kernel time** in paired cold-cache readings at every M timed: N=2048, K=2048: 6.604%/8.273% at m=2048/4096;
N=3584, K=512: 15.751%/9.903%/8.757% at m=1024/2048/4096; N=6144, K=1536: 6.322% at m=4096; N=7168, K=512:
11.138%/13.941% at m=512/1024 (m=256 not measured). Read it as roughly the most store work can return there, not
as a gain to expect: the deletion arm writes no C (its output is NaN by design), keeps the LDS staging and feeds the store data
to a side-effect sink. Its ISA kept the MFMA sequence and the LDS instructions and allocation; scheduling and
occupancy were not compared, so part of the difference may come from them.
*(field, E37)* Single probes during the campaigns, on earlier builds of the same four GEMMs, read 8–16%: 13–16% at
N=3584 (m 1024–4096), 8–12% at N=7168 (m=512, 1024), 10% at N=6144 (m=4096), and at N=2048 −1.655/−4.985 µs at
m=2048/4096 against baselines from other iterations (so the 8%/16% derived from them are not paired savings).
Price the store from paired readings of one build.

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
  *(field, E38)* Register-direct versus LDS-staged C stores, and wider versus narrower direct stores, were different
  comparisons on these bf16 kernels at m 256–4096. On grids whose B fill permuted rows so each lane held adjacent columns: at N=7168, K=512 (m ≤ 2048), with A
  first in the MFMA and 16 lanes writing a whole 128 B row, 8 B each, +0.87% over five m (m=256 +2.3%, m=512 +1.6%,
  m 1024–2048 +0.1–0.3%; the permuted 16 B form lost at every m); at N=2048, K=2048 paired-dword stores beat its 2 B
  direct stores by 0.44 and 0.64 µs at m=2048 and 4096, where an LDS-staged 16 B store lost 0.2–0.4 µs; and at N=3584
  on two tiles up to 128 wide (below). The permutation alone did not predict the sign (N=28672 below). It did not win
  on the two-round m=4096 of the N=7168 kernel (29.06–29.28 against 28.98–29.20 µs, overlapping), and lost at N=28672
  (+2 to +16% with the same permutation) and at N=6144, K=1536 (m 256–1024). At N=3584, K=512 it was mixed on tiles
  up to 128 wide: it won at m=256 on 32×112 tiles (5.01 against 5.08 µs; on the 64×64 tile that later replaced
  it, it lost 0.10 µs, and the kept 64×64 row stages C through LDS) and at m=1024 on 128×128 (−0.09 µs,
  kept), lost at m=512 on 64×112 (5.95–5.97 against 5.88–5.90 µs). At m 2048–4096 on 128–256 × 224 tiles,
  direct stores together with disabling overlap lost 3.1–7.7 µs; that comparison does not isolate the store path.
- **Feeding CShuffle, swap the MFMA operands:** a lane then holds 4 adjacent columns, so each fragment is one 8 B
  LDS write (64 `ds_write_b16` → 8 `ds_write2_b64` per wave), no cross-lane op. *(field, E16)* +0.3% to +3.2% per
  shape, all 17 faster (the same kept round also gave M=33–60 a single 64×32 row block).

**Instruction-level rules for cross-lane stores**
- A `permlane16_swap` result consumed only by a `buffer_store` (VMEM) needs no `s_nop`. That hazard covers a
  following VALU read (FP4:500-502).
- When a VALU consumes the result, use the `llvm.amdgcn.permlane16.swap` / `permlane32.swap` intrinsics, not inline
  asm. The hazard recognizer cannot see asm, so asm forces an `s_nop` on both sides. *(field, E17)* Switching to the
  intrinsic cut `s_nop` from 412 to 174 in a fused bias + GELU epilogue (683 to 24 with FP6 row quantization fused
  as well).
- Issue all swaps first, then all stores, so the permlanes and the stores are far apart. *(field, E18)* Only the two
  changes together, dropping the hard-coded `s_nop` and the two-phase order, took the count from 80 to 0.
- **Never store the high half of an f32 accumulator as bf16.** That is truncation, not round-to-nearest-even.
  - *(field, E19)* It cost 5.8 dB SNR and made the store slower.
  - Convert with `v_cvt_pk_bf16_f32`, and store the pair's high half with `buffer_store_short_d16_hi` where the
    toolchain can emit it; otherwise it costs one `v_lshrrev_b32`.
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
  - *(field, E39)* On bf16 GEMMs at m 1024–4096, overlapping the store with the last K slice paid; most of these
    readings also carried another change:
    at N=3584, K=512, staging C one row group at a time through the stage the last K slice freed,
    together with an exact-vmcnt correction, gained +2.5% (m=4096 +7.9%, m=2048 +4.4%).
    Double-buffering that staging together with removing write barriers gained +0.16% more; at N=7168, K=512 slicing the last K block so its stores
    overlap +1.5–1.6% at m=2048 and +0.9–1.0% at m=4096 in the probe (its ablation row, +1.4% over five m, also
    carried barrier-free per-wave C staging at m 256–1024, +3.0% at m=512); at N=6144, K=1536, storing each wave's first 32-row atom row in the last K step,
    removing a scheduling fence and changing SOFF gained +0.33% over five M Points; m=4096 gained +1.45%.
    This is the joint round's result, not overlap alone (two or three early rows lost).
- **Fold the store into a phase with no G2S.** gfx950 has a single in-order `vmcnt` for loads and stores,
  so a store in flight is serialized against operand DMA waits. FP4 therefore puts its fused C store in a
  peeled tail phase that issues no fills, and paces it at `_CRATE` lines per MFMA (FP4:776, 1095-1233,
  1745). Spent accumulators can be recycled as pack registers when the caller never reads them
  (`acc_dead`, FP4:679).
  - *(field, E40)* Moving a persistent bf16 kernel's C store into the next tile's K loop (N=28672, K=512) lost at
    m=4096 (118.7–121.4 against 111.8 µs) and m=1024 (36.8–37.2 against 35.8) and read 57.2–57.7 against 58.0 at
    m=2048 (single driver runs, not paired), so the store stays after the tile's own K loop.

## 5. Cache policy on the store

- **C is write-once: race `nt` (aux=2) and `sc1` (aux=16) against the default.** Turbo stores it `nt` so it does
  not evict the A/B band the L2 tile order keeps (FP8:72-73). *(field, E21)* On a bf16 GEMM with K=128 and N=1536,
  `sc1` won at every M from 8 to 16384 (+3.1% to +20.3% per shape, all 17 faster) and `nt` lost at the largest M.
  *(field, E22)* On another (K=512, M ≈ 35k–340k), `nt` beat `sc1` by 0–10% and the default by 1–17%. On the bf16
  GEMMs below the winner changed with N, with M and between builds of one kernel, and in paired sessions on two
  fixed inputs per M it sometimes changed sign between the inputs. Those inputs ran in separate time windows, so
  neither pool such a reversal away nor put it on the input alone: race the policy per M bucket on your own inputs,
  and again after the loop changes.
  - M 1–128, K=512, N=7168: *(field, E35)* `sc1` with paired 16 B stores gained 5.39% [+4.19, +6.60],
    all 8 M faster, and `sc1` alone was 0.10–0.20 µs faster at every M but m=64. *(field, E48)* On the campaign's
    final source, paired sessions with the C-store aux the only edit put `sc1` 2.115–5.470% ahead of the default at
    all 8 M on both inputs, m=64 included; the A/A intervals of those sessions spanned −0.227% to +0.240%.
  - M 1–128, K=512, N=28672: `sc1` lost. *(field, E36)* An earlier campaign build put `sc1` slower than the
    default at 7 of 8 M in single paired readings (+0.3% to +5.5%, four of them above 1%; m=2 −0.4%).
    *(field, E44)* On a later small-M campaign build (source `827136159797…`), with the C-store aux the only edit
    and the ISA otherwise identical, `sc1` minus default for the two inputs was: m=1 +2.935%/+10.183%;
    m=2 +3.037%/+8.472%; m=4 +3.882%/+12.581%; m=8 +13.363%/+12.190%; m=16 +0.465%/+0.415%;
    m=32 +0.433%/+0.577%; **m=64 −0.992%/+0.615%**; m=128 +2.195%/+1.916%. Every simultaneous 95% interval
    excluded zero (at m=64 [−1.034%, −0.951%] and [+0.561%, +0.669%]); every arm matched the reference and the
    aux arms were bitwise equal to the default.
  - M 256–4096, N=6144, K=1536: `sc1` over `sc0` and the default; `nt` slightly ahead of `sc1` at m 512–1024.
    *(field, E41)* In the campaign `sc1` won at m=256 (10.90 against 11.17 µs with `sc0`), was within 0.3% of `nt`
    at m=1024, and the default read 2.2% slower at m=4096. *(field, E45)* On the campaign's final source
    (which dispatches m=256/512/1024/4096 and has no m=2048 path) both inputs put `sc1` ahead of
    `sc0` at m=256 and of the default at m=4096; at m=512 `nt` read under 0.1% faster. At m=1024 `nt` was faster on
    both: 100×(time(nt)/time(sc1)−1) was −0.434645% [−0.467335, −0.401956] and −0.207454% [−0.244206, −0.170701],
    the second within ±0.3%.
  - M 256–4096, N=7168, K=512: `sc1` over the default everywhere; against `nt` it depends on m. *(field, E41)* An
    early race put `sc1` 2.4–2.9% ahead of `nt` at m=256, 2048 and 4096 and tied at m=1024; a later race on the
    kept m ≤ 2048 store put it 1–2% ahead at m 256–512 and tied at m 1024–2048; the default read 5–12% slower at
    m 256–2048 and 0.8% at m=4096. *(field, E46)* On the campaign's final source `sc1` beat the
    default at all five m on both inputs (3.203–9.924%) and `nt` at m=256/512. time(nt)/time(sc1)−1 was within
    ±0.3% at m=1024 (+0.111912% [+0.071354, +0.152469] and −0.163521% [−0.189426, −0.137616]) and changed sign
    beyond it at m=2048 (+2.036826% [+1.968119, +2.105533] and −0.383761% [−0.429174, −0.338349]); at m=4096
    `sc1` led `nt` by 4.672% and 4.112%.
  - M 512–4096, N=28672, K=512 (persistent, a head and a tail store): the head store's winner changed between
    builds. *(field, E41)* On early builds head `nt` was faster at m=1024 and 4096 (35.83 against 36.83, 112.2
    against 117.5 µs), m=2048 read nearly equal (59.53 against 59.46), and head `sc1` was faster at m=512
    (18.53–18.54 against 18.61–18.62 µs). *(field, E47)* On a later build (with `_STAGGER[2048]`
    3 and kernel arguments preloaded at m=2048; the refinalized kernel after it uses 2 and no preload there), with
    the tail kept at `nt`, head `sc1` was faster at each of m=512/1024/2048/4096 (at m=2048 by an interval that
    crossed ±0.3%); no default arm was timed. The persistent round count alone does not pick the policy.
  - M 512–4096, N=2048, K=2048: *(field, E41)* an early build read the default and `sc0` nearly equal at m=4096
    (`nt` +0.6%, `sc1` +1.9%), and `nt` and the default nearly equal at m=512–2048 (`sc1` not raced there).
    *(field, E47)* On the campaign's final source the default and `sc0` were within ±0.3% at
    m=4096 (`sc0` 0.027% faster) and `nt`/`sc1` slower by 0.688%/2.325%; `nt` was 0.496%/1.342% faster than the
    default at m=512/1024 and 1.256% slower at m=2048, each beyond ±0.3%.
  - N=3584, K=512: *(field, E41)* `sc1` paid on LDS-staged row stores (the default cost 1.39 µs at m=2048) and
    went both ways on register-direct stores: it lost at m=256 (+0.44 µs), at m=1024 on a 64×224 tile (+0.82) and
    at m=2048 (+1.9 against the default), and won on the kept m=1024 128×128 row-group store (the default +0.76,
    `nt` +0.24 µs against it).
  - Narrow N (64–256, K 4096–6144): *(field, E41)* no policy paid on the output except `sc1` at N=256, m=512 and
    1024 (0.3–0.8% faster before those buckets were split across workgroups, and kept on their split-K output
    store; it lost 1.7% at m=256).
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
  (`sc0|sc1`). *(field, E42)* At m=4096 (N=64, K=4096, split 4, the splits presumed on one XCD) aux 16 and 17 tied
  (10.22 µs); after each tile's counter got its own 256 B, aux 18/19 read 10.71–10.72 and the L2-cached aux 0, 1 and 3
  10.79–10.83 against 10.05 with aux 17 (single driver runs): the `sc0|sc1` partials won even with the splits presumed
  on one XCD.

In FlyDSL: `../flydsl/gemm.md` § epilogue §5.

## 6. Bounds without per-column masks

- Rebase the C descriptor per row band in i64 and pin base and `num_records` with `readfirstlane`
  (`make_row_band_resource`, GH:2010). A wide store has no per-column mask, so a whole column span is
  dropped by giving its band 0 records. Store classes that span several fragments require
  `N % tile_n == 0` (`col_safe`).
- When a store is bounded by selecting an out-of-range byte offset (one `v_cndmask` per offset), apply that select
  to the part of the offset that does not depend on the sub-tile, then add the sub-tile row delta.
  *(field, E27)* Otherwise `v_cndmask_b32` went from 14 to 64 in a grouped GEMM's fused epilogue.
- One descriptor per tensor per tile is enough when sub-tiles differ only by a compile-time row offset.
  Carry the row delta in the offset instead of building a descriptor per sub-tile.

In FlyDSL: `../flydsl/gemm.md` § epilogue §6.

## 7. beta = 1 (accumulate into C)

- Pipeline the read-back: load quadrant i+1 of C while quadrant i's store is in flight (`_store_quadrants`, GH:1528).
  Add in f32 so the result is rounded once.
- **Compile and race against a scratch output.** If building the kernel also runs it, a beta=1 build against the
  caller's C accumulates into it twice. Turbo races at beta=0 and rebuilds the winning configuration with the
  accumulate epilogue (FP8:3008-3027).
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
    dropped its achieved bandwidth from 4.3 to 2.9 TB/s by the record's account.
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
does not, the line says which date it is. "Turbo" is Primus-Turbo.

- E1 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 counters per arm (`TCP_TCC_WRITE_REQ`, `TA_BUFFER_WRITE_WAVEFRONTS`); the dwordx2 arm's wall ruler is not stated] scalar bf16 C stores: 2.359e7 write requests against 755 MB ÷ 32 B = 2.36e7; paired dword stores: 1.206e7 against a predicted 1.206e7; widened to dwordx2 (`nt_cpack=4`): `TA_BUFFER_WRITE_WAVEFRONTS` 3.015e6 → 1.573e6, `TCP_TCC_WRITE_REQ` still 1.206e7, and +2.25% against the pair's +3.10% (−0.85 pp) (gpt-oss-20b down-projection forward NT, grouped fp8 tensorwise GEMM, G=32, M=131072, N=2880, K padded to 2944)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=5, own harness on AKLR bench_cudagraph, arm order alternated, 5 rounds; timing-only arms whose output was wrong by design] storing 64 of the 72 columns at a 64-column (128 B) pitch read 1.0248 / 1.0378 / 1.0398 at m=24/64/96 against 0.9962 / 1.0115 / 1.0173 at the original 72-column (144 B) pitch (ratios incumbent ÷ variant) (gemm_n18432_k1536 forge run, bf16, 72-column N tiles)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode] the kept round that gave each workgroup a line-aligned 64-column block plus a slice of a leftover-column strip, also dropped an epilogue barrier and set the C store's aux per M bucket (0 at m ≤ 64, 17 above; the parent stored with 17 in its 32–96 buckets): +1.31% [+1.18, +1.45] over its parent, −0.42% (m=8) to +3.09% (m=48), 11 of 13 cases up; in the round's own harness the barrier drop alone read 0.9994–1.0029 (gemm_n18432_k1536)
- E4 [2026-09, gfx950, FlyDSL 0.2.2, N=?, same-process paired readings in both orders, 16 launches per window, with an inert control arm reading ±0.06–0.31%] with a 32×32×64 atom on a transposed 4(M)×2(N) wave grid, the build with pairing read −24.4% against the shipped kernel and the same build without it −25.7%: pairing was worth +1.3% there; that wave grid is the only one on which this atom keeps the paired store (gpt-oss-20b grouped fp8 tensorwise GEMM at B=24, NT forward)
- E5 [2026-09, gfx950, FlyDSL 0.2.2, N=?, the ruler of E4] switching the 16-wide pairing off (`nt_pairc=False`) on the shipped kernel cost +2.80 / +3.03 / +3.28 / +3.53% on the four NT forward cells; two later readings on the same kernel gave +4.3% (`pairc0`) and +4.98 / +4.49% (both arms with pairing off) (the kernel of E4)
- E6 [2026-09, gfx950, FlyDSL 0.2.2, N=?, same-process wall probe with drop and alias arms; the board at its 1400 W cap, full-load sclk 1723–1743 MHz] deleting the whole epilogue moved wall by 0% ± 1 pp on four cells (bal/B8 +1.53%, B24 −0.36%, B32 +0.36%); the same round's row-merge store change raised the benchmark's dgrad legs from 0.9739–0.9984 to 0.9985–1.0336 (gpt-oss-20b grouped fp8 tensorwise GEMM, non-persistent)
- E7 [2026-08, gfx950, FlyDSL ?, N=?, per-tile time from a 6-point K-sweep fit t_tile = F + n·P and the probe's TF/s at K=2944, all arms in one run] deleting the C store: F 7.943 → 2.598 µs, 4.87 µs = 20.3% of the tile (probe 3721.7 → 4672.2 TF/s); same bytes and rows with write segments 4 → 1: +0.8%; 254 `v_accvgpr_read` + 254 `cvt_pk` removed with all 256 stores kept: −0.5%; a segment probe that folded four rows into one read −2.58 µs but wrote ~6× fewer bytes; a later round's counter run deleting the whole store read 579.2 → 476.5 µs (−17.7%) (gpt-oss MXFP4 grouped GEMM, NT forward, G=32)
- E8 [2026-07, gfx950, FlyDSL ?, N=?, time FlyDSL ÷ AITER, lower is faster] at 8192²×4096 the permlane16_swap + dwordx4 store (TACCW) read 0.998 against 1.039 for the default store (Turbo MXFP4 4-wave dense GEMM, 16 KB output rows; unverified, first recorded 2026-07-05)
- E9 [2026-08, gfx950, FlyDSL ?, N=?, ruler not stated (the campaign scores gm_ratio against the MXFP8 grouped GEMM)] the TACCW wide store was slower by 1.34% (fwd down), 0.70% (fwd gate_up) and 1.07% (dgrad gate_up); in the next campaign on the same kernel the tree's `store_tacc_wide` read +4.1% slower in the NT probe of E7, +3.5% with its software pipeline cut to depth 4 (gpt-oss MXFP4 grouped GEMM, NT, G=32, M=131072, N=2944/5760, 5888 B output rows)
- E10 [2026-08, gfx950, FlyDSL ?, N=3, kernel-level palindrome A/B, 40 rounds per token distribution, 3 distributions, shared ref arm; ISA counts from the build] row-permuted B fill with paired dword C stores: +2.77 / +2.65 / +2.44% over the previous kept kernel, every round positive; `buffer_store_short` 160 → 32, `buffer_store_dword` 0 → 64, `v_cvt_pk` 160 → 96; `v_mfma` 920, `s_barrier` 304, `.vgpr_count` 256, spill 0, LDS 131072 unchanged (the kernel of E1)
- E11 [2026-08, gfx950, FlyDSL ?, N=3, the ruler of E10 for the kept arm; the f32 arm's repeat count is not stated] row merge swapping f32 values one by one (64 permlanes): +0.13%; swapping packed row pairs (32 permlanes, `v_cvt_pk` 160 → 96): +0.83 / +0.43 / +0.62% (dgrad NN of the grouped fp8 GEMM of E1)
- E12 [2026-08, gfx950, FlyDSL ?, N=?, isolated screening runs (the LDS-staged store) and a later candidate's paired readings (the DPP store)] `store_cshuffle` (LDS-staged wide store): dgrad NN −21%, forward NT −24%; a DPP `quad_perm` + `v_perm` paired dword store: NN dgrad −1.09%, TN weight gradient −1.9% (the grouped fp8 GEMM family of E1, grouped with the MXFP4 kernel of E9 as one family of about 5760–5888 B rows)
- E13 [2026-10, gfx950, FlyDSL 0.3.2, N=?, own CUDA-graph harness (512 MB flush, alternating order, median of 5–7); paired harness on AKLR's graph timer (3–5 interleaved rounds per arm, median delta)] C stored straight from registers against the LDS-staged store: 8 B stores in the non-persistent kernel 195.0 / 121.2 against 165.0 / 102.5 µs at m=16384 / 9397; on the large rows +31.57 / +20.39 / +16.63 / +5.39 µs at m=16384 / 9397 / 2732 / 893, against the run's final per-case times 157.4 / 104.0 / 33.5 / 14.7 µs (about +18% to +50%) (gemm_n7168_k768 forge run, bf16)
- E14 [2026-10, gfx950, FlyDSL 0.3.2, N=3, rocprofv3 kernel trace, median of 100 calls with an L2 flush before each, 3 repeats; in-process harness on bench_cudagraph, median of 5–9] C stored straight from registers: m=893 28.90 vs 26.56 µs, m=2732 60.86 vs 55.18 (rocprofv3); m=16384 299.5 vs 269.2, m=893 29.45 vs 26.10, m=2732 61.54 vs 55.50 (in-process), +9% to +13%; the direct arm's store width is not recorded (gemm_n7168_k1536 forge run, bf16)
- E15 [2026-10, gfx950, FlyDSL 0.3.2, N=6, own harness on cudagraph with an L2 flush, alternating order, 6 rounds (two 6-round runs for the 8 B arm); incumbent against itself 0.9979–1.0013] ratios LDS-staged incumbent ÷ variant: 8 B per-fragment register store 0.8810–0.9414 at m=56–112, 0.9839–0.9953 at m=24–48, 0.9993–1.0055 at m=8–16; 16 B direct store 0.9446–0.9803 at m=56–112, 0.9977–1.0048 at m ≤ 48 (gemm_n18432_k1536 forge run, bf16, 96-wide tiles storing 72 columns, 80-wide above m=96)
- E16 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode; LDS write counts from the distill step's probe compile of both arms (128×128 tile, m=2732)] the kept round that swapped the MFMA operands for 8 B LDS writes, and also gave m=33–60 a single 64×32 row block: +1.33% [+1.17, +1.49], all 17 cases faster, +0.26% (m=64) to +3.20% (m=16384); 64 `ds_write_b16` → 8 `ds_write2_b64` per wave (gemm_n7168_k1536)
- E17 [2026-10, gfx950, FlyDSL ?, N=1, ISA of one build per arm] permlane swaps through `rocdl.permlane16_swap` / `permlane32_swap` instead of inline asm: `s_nop` 412 → 174 in the scored arm, 683 → 24 in the fused row-pack arm, `v_*` unchanged (Flux A6W4 forward GEMM: the scored arm with a fused bias + GELU epilogue, the row-pack arm also quantizing rows to FP6 (E2M3) in the epilogue)
- E18 [2026-07, gfx950, FlyDSL ?, N=1, ISA of one build per arm] TACCW epilogue with the hard-coded `s_nop 1` removed and a two-phase order (all cvt + permlane first, then all stores): `s_nop` 80 → 0; removing the `s_nop` alone left 19, the two-phase order alone 64 (Turbo MXFP4 4-wave dense GEMM; ISA counts reported, unverified, first recorded 2026-07-05; the change is in Primus-Turbo 9f525dba, primus_turbo/flydsl/gemm/mxfp4_gemm_kernel.py:248; :302)
- E19 [2026-09, gfx950, FlyDSL 0.2.4, N=1, one campaign score for the truncating arm (a GEMM campaign's geomean over 18 cells against a frozen baseline), three for the comparison arm] BLOCK_N=192 with the in-loop folded store falling back to `buffer_store_short_d16_hi` of the accumulator: SNR 55.6 → 49.77 dB, `buffer_store` 104 → 200, `gptoss_out_proj_wgrad` 5022.5 TF/s and campaign score 1.0067, against 5108.7 / 5164.9 / 5120.9 TF/s and 1.0080 / 1.0075 / 1.0087 with the C store handed back to the epilogue's round-to-nearest-even path (Turbo MXFP4 dense GEMM)
- E20 [2026-08, gfx950, FlyDSL ?, N=?, drift-immune same-process probe; bench.sh with 3 + 3 alternating reps; store cost from a no-store control with the cache off] the last K-block peeled out of the whole-loop asm, each row-tile's store after the next row-tile's MFMAs: distance 1 down +1.03 / +1.12 / +0.92%, gate_up +0.95 / +0.72%; distance 2 +0.07 / +0.75 / +0.92%; distance 4 +0.62%; bench best +0.90%, median +0.95%; the store cost 8.31% (gate_up) and 10.16% (down) of the kernel at the deployed shape (gpt-oss-20b grouped fp8 tensorwise weight-gradient GEMM, G=32, per-expert M=4096, 32 K-blocks per tile)
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop ablation ladder row marked recorded (the campaign's paired KEEP timing), driver --bench-mode; per-shape µs from the round's AKLR graph timer, best of 2 medians of 3] C stored with `sc1` (aux=16) in every M bucket: +4.62% when kept, +5.54% recorded, bounds +4.40 … +6.68%, all 17 cases faster, +3.13% to +20.27% per case; at m=16384 the default read 15.19 µs and `nt` 15.37 (gemm_n1536_k128 forge run, bf16, K=128)
- E22 [2026-10, gfx950, FlyDSL 0.3.2, N=?, own harness: a CUDA graph of 10 launches, median of 10 replays, noted as noisy; GROUP_M=1] C store default / `sc1` / `nt`: m=35107 0.1286 / 0.1259 / 0.1252–0.1275 ms, m=104093 0.3785 / 0.3510 / 0.3136–0.3201, m=342073 1.1854 / 1.1084 / 1.0774–1.0784 (gemm_n3072_k512 forge run, bf16, K=512, 256×256 tile)
- E23 [2026-09, gfx950, FlyDSL ?, N=?, kernel-only timing, repeat count not stated; the month is the commit's] `nt` on the MXFP8 GEMM's C store as an autotuned axis: linear1 −4.6%, linear2 −3.0% kernel-only, 0.9–1.9% slower on the narrow shapes whose slabs are already L2-resident; raced at a 1% margin (Turbo MXFP8 dense NT GEMM, the training step's five production linears; Primus-Turbo d522ee0a (Primus-Turbo#505))
- E24 [2026-09, gfx950, FlyDSL ?, N=?, ruler not stated; the month is the commit's] a non-temporal hint on the byte-granular E8M0 scale stores cost −15 … −40% across the thirteen production casts (Turbo MXFP8 dual-cast quantization kernels, `mxfp8_quant_flydsl.py`; Primus-Turbo d522ee0a (Primus-Turbo#505))
- E25 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode] split-K workspace partials stored with `sc1` (aux=16) for ma ≥ 3, with the K splits spread across XCDs, in a kept round that also used a 3-wide N tile at m=7: +2.12% [+2.02, +2.22] over its parent; m=35–98 +2.0% to +3.6%; m=14–28, which do not take that path, −0.06% to −0.21%; m=7 +4.5% (gemm_n768_k7168 forge run)
- E26 [2026-10, gfx950, FlyDSL 0.3.2, N=4, own A/B races on AKLR bench_cudagraph (rep=20) over AKLR's 8 timing input sets, 4–8 alternating rounds, median; HEAD against HEAD 0.996–1.006] split-K plane stores with aux 18 / 19: 0.947–0.967 of the incumbent's aux 17 at m=8/40/64/112/893/2732; aux 0 0.982–0.998 at m=24/40/64/112 (gemm_n144_k7168 forge run, `_PLANE_AUX=17` held fixed)
- E27 [2026-09, gfx950, FlyDSL 0.2.4, N=1, ISA of one build per arm] buffer resources hoisted from per sub-tile to per tile with the column predicate left on the load mask: `v_cndmask_b32_e64` +50 (14 → 64); with the predicate applied to the sub-tile-independent offset (`_drop_col`), the fc2 epilogue lost 126 / 112 / 111 instructions (MoE FC2 dgrad, a grouped MXFP4 GEMM with a fused dGLU + dual MXFP4 quantization epilogue, `StoreCdSwiGLUQuadQuant`)
- E28 [2026-10, gfx950, FlyDSL ?, N=?, four kernels timed in one run] pure GEMM 132.5, standalone quantizer 100.8, unfused pair 231.1, fused 234.4 µs, so the epilogue costs 101.9 µs inside the GEMM while traffic falls 627 → 294 MiB; the standalone chain up to dGELU took 98.4 µs; the record puts the epilogue's memory efficiency at 4.3 TB/s standalone and 2.9 TB/s inside the GEMM (Flux MXFP4 dgrad GEMM fc2_dgrad 8192×12288×3072 with a fused dGELU + dual MXFP4 quantization epilogue)
- E29 [2026-10, gfx950, FlyDSL ?, N=?, ruler not stated] fp4 pk8 packing removed 192 instructions, 2.7% of the kernel: fc2 changed 0 (the kernel of E28)
- E30 [2026-10, gfx950, FlyDSL ?, N=?, campaign score (training-time-weighted TF/s of two shapes), ABBA order with a 20 s sleep between arms, runtime cache off] 0, 8 and 16 dependent FMAs in the K-loop gaps with a side-effect sink: 2844.97 TF/s (fc1_double 217.6 µs), 2693.07 (−5.34%, 227.9 µs), 2515.32 (−11.6%, 247.2 µs); about 1.53 cycles per instruction for the first 8 (65% absorbed) and 3.23 for the next 8 (25%); a free pool of about 5.0e4 cycles ≈ 25–28 µs, 38% of the 68 µs epilogue; a static estimate gave about 240k idle VALU issue cycles (A6W4 forward GEMM, 256×384 tile, one wave per SIMD, fused GELU epilogue)
- E31 [2026-08, gfx950, FlyDSL ?, N=?, ruler not stated for this arm; the tail window counted in the round's ISA gate] the last K-block peeled out of the whole-loop asm, each row-tile's store after the next row-tile's MFMAs (distance 1): +0.02%, within noise, VGPR 96 → 128; distance 2 −6.26% with a spill; the structural tail window without G2S held 8 MFMAs (dgrad NN of the grouped fp8 GEMM of E1)
- E32 [2026-09, gfx950, FlyDSL ?, N=1, the 18-shape campaign score, one run per arm; the 8 deep-K shapes, whose code did not change, as in-run control (+0.31% / +0.19%, max drift 1.59% / 0.49%)] C store policy on the 10 shallow-K shapes (K ≤ 6144) set to aux 1 (`sc0`, without `nt`) or aux 0 (default write-back) instead of `nt`: all 10 slower in both arms, worst QKV_dgrad −6.42% / −5.17%, out_proj_fwd −3.56% / −3.24%; A (67 MB) + B (8 MB) fit the 256 MB LLC, C (268 MB) does not (Turbo MXFP4 dense NT GEMM, a GEMM campaign over 18 shapes; reported 2026-09-26)
- E33 [2026-08, gfx950, FlyDSL ?, N=3 as reported, interleaved A/B, 3 runs per arm, campaign score gm_ratio_pct against the large-M reference] on the folded C store whose 16 lanes fill a 128 B line (`buffer_store_dwordx2`) with `nt`: `nt sc1`, `sc1` alone and `sc0 sc1 nt` read 98.32 / 98.31 / 99.43 against 100.49–100.69, all six wgrad cells 4–5% lower; in round 3 `sc1`+`nt` read −0.5% to −1.8% on the deploy wgrad cells while the untouched reference cells ran 2.5% faster (gpt-oss-20b MXFP4 grouped GEMM, G=32, per-expert M=4096, skewed routing, wgrad)
- E34 [2026-09, gfx950, FlyDSL ?, N=2, `us_min` per arm, ABBA twice, FlyDSL disk cache cleared before each value, each arm's stores checked in the ISA dump] all eight `sc0`/`nt`/`sc1` combinations (aux 0–3, 16–19) of the folded C stores: linear1_fwd 0.4440–0.4465 ms, a 0.56% spread, against 2.0% between the aux=2 default's own two reads; linear2_dgrad 0.64% spread (Flux MXFP4 dense GEMM, 16384×21504×3072 and 16384×15360×3072)
- E35 [2026-11, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder (remeasured), 3 pairs, AKLR graph timer, one-sided 95% bounds; the aux race: agent's harness running AKLR `bench_cudagraph` in one process, A-B-B-A over 4 rounds, medians, outputs `torch.equal`] C stores with `sc1` (aux=16) plus paired 16 B stores (B rows permuted in the fill so a lane's two N fragments are 8 adjacent C columns, one `buffer_store_dwordx4`): +5.39% [+4.19, +6.60], 8 / 0 cases, m=1 +4.09% … m=128 +7.42%; aux 0 / 16 / 17 / 18 / 2 at m=128 4.985 / 4.868 / 4.863 / 4.857 / 4.982 µs, at m=16 4.382 / 4.269 / 4.253 / 4.271 / 4.366; across the 8 m, aux=16 0.10–0.20 µs faster except m=64 (4.422 vs 4.411) (bf16 GEMM N=7168, K=512, m 1–128, whole-K LDS-DMA tile)
- E36 [2026-11, gfx950, FlyDSL 0.3.2, N=1, driver graph timer (warmup 3, 20 iterations), each arm paired with canonical in the same window] C store aux on all kernels at m=1 / 2 / 4 / 8 / 16 / 32 / 64 / 128: `sc1` (16) 6.653 / 6.444 / 6.380 / 6.602 / 6.618 / 7.199 / 8.095 / 10.888 µs, default (0) 6.306 / 6.471 / 6.345 / 6.433 / 6.584 / 7.177 / 7.939 / 10.694, `nt` (2) 6.443 / 6.425 / 6.363 / 6.520 / 6.615 / 7.216 / 8.054 / 10.615, `sc0` (1) 6.301 / 6.412 / 6.346 / 6.422 / 6.576 / 7.184 / 7.935 / 10.675 (bf16 GEMM N=28672, K=512, m 1–128, its skinny, band and row-owner kernels)
- E37 [2026-11, gfx950, FlyDSL 0.3.2, N=3–5, agents' interleaved probes on AKLR's graph timer (store removed or pushed out of bounds: wrong output by design)] N=3584 K=512 against 7.93–8.00 / 10.51–10.56 / 16.6–16.8 µs at m=1024 / 2048 / 4096: C store removed 6.78 / 9.12 / 14.03; N=7168 K=512: m=512 6.459 against 7.303, m=1024 9.826 against 10.733; N=2048 K=2048: −1.655 µs at m=2048, −4.985 at m=4096 (19.8/30.3 µs baselines came from different iterations and are not paired denominators); N=6144 K=1536 m=4096: 58.27 against 64.68 with the store
- E38 [2026-11, gfx950, FlyDSL 0.3.2, N=1–7, driver --bench-mode and agents' interleaved probes on AKLR's graph timer; ablations 3 pairs] N=7168 K=512, C stored from registers as whole rows (A first in the MFMA, B rows permuted, each lane a bf16x4) at m ≤ 2048: +0.87% [+0.69, +1.04], m=256 +2.33%, m=512 +1.57%, m=1024 +0.09%, m=2048 +0.26%; at m=4096 29.06–29.28 against 28.98–29.20; N=2048 K=2048 paired-dword C stores from a B-row permutation: m=2048 19.865 against 20.299, m=4096 30.423 against 31.063, ablation +0.96%; operands swapped and C staged through padded LDS with 16 B stores: 31.32 against 31.10 (m=4096), 20.71 against 20.31 (m=2048); N=28672 K=512 register-direct with B fill rows permuted: +8.33 / +2.09 / +15.70 / +12.92 / +15.17% at m=256…4096; N=6144 K=1536 direct stores 11.57 / 16.89 / 24.26 against 11.15–11.19 / 15.99–16.01 / 22.71–22.72 (m=256 / 512 / 1024); N=3584 K=512 register-direct at m=256 (32×112) 5.012 / 5.013 against 5.079, on the later 64×64 m=256 row +0.098 over 3 pairs, at m=512 (64×112) 5.945 / 5.973 against 5.900 / 5.882, at m=1024 (128×128, the kept row) −0.093, with direct stores and overlap disabled together, at m=4096 +7.388 / +7.709 and m=2048 +3.293 / +3.123 µs
- E39 [2026-11, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladders, 3 pairs, driver --bench-mode; probes on AKLR's graph timer] N=3584 K=512 store overlapped with the last K slice one row group at a time together with the exact-vmcnt correction: +2.53% [+2.50, +2.57], m=4096 +7.87%, m=2048 +4.41%; double-buffered drain staging together with removing write barriers at m=2048 / 4096 +0.16% [+0.12, +0.20]; N=7168 K=512 last K block sliced under the C stores (m=2048 / 4096) with per-wave staging (m=256–1024) in one ablation row: +1.37% [+1.30, +1.45], m=512 +2.96%; the slicing alone at m=2048 +1.59 / +1.49% and m=4096 +0.86 / +0.96% in the probe; N=6144 K=1536 first 32-row atom row stored in the last K step, scheduling fence removed and SOFF changed in one round: five-Point aggregate +0.33% [+0.19, +0.47], m=4096 +1.45%; two and three rows 63.81 and 63.66 against 63.38 µs
- E40 [2026-11, gfx950, FlyDSL 0.3.2, N=1–2, driver --bench-mode] N=28672 K=512 epilogue pass 3 deferred into the next tile's K loop: m=4096 118.70 / 118.72 (one store after K-blocks 0–3) and 119.87–121.4 (store after the mid-block barrier) against canonical 111.78; m=1024 36.82–37.24 against 35.81; m=2048 57.22–57.65 against 57.97; 254 VGPRs, no spills
- E41 [2026-11, gfx950, FlyDSL 0.3.2, N=1–5, driver --bench-mode and agents' interleaved probes on AKLR's graph timer] C store aux: N=6144 K=1536 aux 1 / 2 / 3 / 16 at m=256 11.17 / 11.04 / 11.05 / 10.90, m=512 16.01 / 15.51 / 15.53 / 15.40, m=1024 22.75 / 21.83 / 21.84 / 21.89; m=4096 aux 0 64.45 against 63.04; N=7168 K=512 aux 2 against 16: m=256 6.11 against 5.94–5.95, m=2048 17.76 against 17.34, m=4096 32.51 against 31.66–31.70; on the kept m ≤ 2048 store aux 2 / 16 at m=256 / 512 / 1024 / 2048 / 4096 5.53 / 7.40 / 10.54 / 16.26 / 29.72 and 5.42 / 7.33 / 10.58 / 16.30 / 29.20; aux 0 at m=4096 31.77 against 31.52; aux 0 against the incumbent: m=2048 17.462 / 16.428, m=1024 12.287 / 10.946, m=512 7.899 / 7.293, m=256 5.835 / 5.536; aux 16 at m=1024 10.957 against 10.946; N=28672 K=512 aux 2 / 16 at m=4096 112.2 / 117.5, at m=2048 59.53 / 59.46, at m=1024 35.83 / 36.83; m=512 `sc1` 0.01853–0.01854 against 0.01861–0.01862 ms; N=2048 K=2048 m=4096 against aux 0: 1 −0.012, 2 +0.195, 16 +0.571, 18 +1.244 µs; `nt` against the default at m=512 / 1024 / 2048 10.639 / 15.389 / 21.009 against 10.659 / 15.389 / 21.009; N=3584 K=512 aux 0 on the canonical m=2048 row +1.391 µs, `sc1` on register-direct stores +0.440 (m=256) and 8.621 → 9.437 (m=1024, 64×224); on the kept m=1024 128×128 register-direct row against its `sc1`: aux 0 +0.758, `nt` +0.236; register-direct at m=2048 against the LDS-staged canonical: aux 0 +1.774, `sc1` +3.694; N=128 K=6144 m=2048 nt / `sc0 sc1` / `sc1` 13.091 / 13.210 / 13.209 against 13.019–13.025; N=256 K=6144 output `sc1` before the split-K rounds: m=512 8.784–8.815 against 8.850–8.882, m=1024 11.697–11.709 against 11.737–11.748, m=256 6.441 against 6.332, later kept only on the split-K output store; N=64 K=4096 output aux 16 / 2 / 17 at m=4096 9.947 / 10.094 / 9.941 against HEAD 9.79–9.94, the other cases tied
- E42 [2026-11, gfx950, FlyDSL 0.3.2, N=1–2, single driver runs] bf16 GEMM N=64, K=4096, m=4096, split 4 (a tile's splits inferred to share an XCD): plane accesses aux 16 10.218 against aux 17 10.220 µs; with 256 B counter spacing (10.05 µs) aux 0 10.830, aux 1 10.823, aux 3 10.793, aux 18 10.720, aux 19 10.714
- E43 [2026-11, gfx950 MI355X, FlyDSL 0.3.2, N=24 fresh-input pairs per M, 5 graph-timer readings averaged per arm/pair, cyclic intact/remove/AA order, AKLR cold-cache bench_cudagraph as of 2026-11-06 (rep=20 ms; flush excluded)] terminal global C-store deletion only, data consumed by an empty side-effect sink, existing LDS staging retained; intact finite/reference-allclose (rtol=.02, atol=.002), deletion output all NaN by design, corrupt-output negative control and unchanged-input checks. Saving = mean of per-pair 100*(1-remove/intact), not ratio of means. N=2048 K=2048 m=2048/4096: 6.604%/8.273%; N=3584 K=512 m=1024/2048/4096: 15.751%/9.903%/8.757%; N=6144 K=1536 m=4096: 6.322% (the run's final source, not an intermediate experimental build); N=7168 K=512 m=512/1024: 11.138%/13.941%. Archived pairs recomputed on the CPU: all 16 Bonferroni Student-t (df=23) intervals of the savings positive, all AA intervals inside ±0.3%. ISA summaries: MFMA opcode sequence, LDS opcode histograms and allocation unchanged, global stores positive → zero; full ISA, scheduling and occupancy not compared. Per-reading samples and output tensors not archived.
- E44 [2026-11, gfx950-targeted compilation, FlyDSL 0.3.2, N=20 timing sessions per seed, small-M campaign build of N=28672 K=512, its decode dispatch at m=1/2/4/8/16/32/64/128, aux0/2/16 C-store-only edits, two fixed seeds 731/1731, 5 readings/arm/session, cyclic order, AKLR cold-cache graph timer as of 2026-11-06, rep=20 ms] primary difference = mean of per-session 100*(mean(aux16)/mean(aux0)-1). Simultaneous 95% Bonferroni Student-t intervals (df=19, family=32: 8 M × 2 seeds × primary+AA), all primary signs resolved and all AA intervals inside ±0.15%; aux2 comparisons exploratory, outside the primary family. Per-seed percentages in §5; m=64 seed731 −0.992476% [−1.033694, −0.951258], seed1731 +0.614804% [+0.560733, +0.668875]. Independently compiled full-ISA hashes matched the archived compile hashes and normalized ISA differed only in C-store flags. Official GEMM reference/DotProductBound, finite-output, aux/default bitwise, alternate-current-stream and corrupt-output rejection checks passed for all eight M × two seeds × three aux, before timing; raw keeps input hashes, PASS markers and timer samples, not output tensors. The two seeds ran in sequential time windows; no GPU idle snapshot or loaded code-object hash recorded.
- E45 [2026-11, gfx950-targeted compilation, FlyDSL 0.3.2, N=2 fixed inputs per M, 20 timing sessions/input, 5 readings/arm/session; timer not recorded in the verdict] campaign final source SHA256 540ffda92fac89cb364cbab2b52b7f725d275015ad9b7f1924883d657fad13aa (N=6144 K=1536), its m=256/512/1024/4096 dispatch (no m=2048 path), C-store cache-only variants. Effect = mean per-session 100*(time(b)/time(a)-1); a=sc1, b=sc0 at m=256, nt at m=512/1024, default at m=4096. Primary simultaneous 95% Bonferroni Student-t intervals (df=19, family=16, four M × two inputs × primary+AA); every AA interval inside ±0.15%, equivalence = the whole interval inside ±0.3%. m=1024 both inputs nt faster, seed731 beyond ±0.3% and seed1731 within it (effects in §5); m=512 (in the family of 16, marked observation-only): −0.034% [−0.060, −0.007] and −0.078% [−0.119, −0.036]. Reference/allclose, finite sentinel, aux/default bitwise, current-stream and corrupt-output checks passed; source/variant/case hashes matched. Compile summaries only (MFMA counts, LDS allocation, load-cache and store-width histograms); full ISA and output tensors not archived; inputs in sequential windows.
- E46 [2026-11, gfx950-targeted compilation, FlyDSL 0.3.2, N=2 fixed inputs per M, 20 timing sessions/input, 5 readings/arm/session; timer not recorded in the verdict] campaign final source SHA256 d5171067a306b2f9da3db2cabc811c396956fad77bdb684aaee88d6500b19a3b (N=7168 K=512), its m=256/512/1024/2048/4096 dispatch, C-store cache-only aux0/2/16 variants. Effect = mean per-session 100*(time(b)/time(sc1)-1), b=default or nt; primary simultaneous 95% Bonferroni Student-t intervals (df=19, family=30, five M × two inputs × two primary+AA), every AA interval inside ±0.15%. Both inputs: sc1 over default at all five M and over nt at m=256/512; sc1/nt signs differ between the inputs at m=1024 (both within ±0.3%) and m=2048 (both beyond it), effects in §5; m=4096 sc1/nt (in the family of 30, marked observation-only): +4.672% [+4.491, +4.853] and +4.112% [+4.038, +4.187]. Correctness checks, hash matches and compile summaries as in E45; inputs in sequential windows.
- E47 [2026-11, gfx950, FlyDSL 0.3.2, N=20 sessions per M (inputs drawn anew each session, seed 731 + session, one process), 5 readings per arm per session, arm order rotated; AKLR cold-cache bench_cudagraph (rep=20) as of 2026-11-06; simultaneous 95% Student-t intervals, df=19, Bonferroni families of 8 (N=28672: 4 contrasts + 4 AA) and 10 (N=2048: 6 contrasts + 4 AA); every AA interval inside ±0.15%] effect = mean per-session 100*(time(b)/time(a)-1). N=28672 K=512 campaign build (`_STAGGER[2048]` 3, kernel arguments preloaded at m=2048; the refinalized source has 2 and no preload at m=2048), its own dispatch, head C-store aux 2 or 16 with the tail fixed at aux 2, no default arm: time(nt)/time(sc1)-1 m=512 +0.445% [+0.285, +0.606]; time(sc1)/time(nt)-1 m=1024 −2.961% [−3.164, −2.758], m=2048 −0.307% [−0.510, −0.104], m=4096 −0.847% [−0.927, −0.766]. N=2048 K=2048 campaign final source, C-store aux 0/1/2/16 only: time(nt)/time(default)-1 m=512 −0.496% [−0.561, −0.432], m=1024 −1.342% [−1.428, −1.256], m=2048 +1.256% [+1.118, +1.394]; m=4096 against the default: sc0 −0.027% [−0.050, −0.004], nt +0.688% [+0.362, +1.015], sc1 +2.325% [+2.128, +2.521]. Before timing, every aux output finite, bitwise equal to the baseline arm and within rtol=0.02/atol=0.002 of an fp32 matmul, a corrupted output rejected; the 24 archived ISAs carry the intended store bits, MFMA counts and LDS sizes equal per M.
- E48 [2026-11, gfx950-targeted compilation, FlyDSL 0.3.2, N=2 fixed inputs per M, 20 timing sessions/input, 5 readings/arm/session; timer not recorded in the verdict; simultaneous 95% Bonferroni Student-t intervals, df=19, family=32] campaign final source SHA256 1aa1cde59b9b7a65aa90f499ce171812407860ccf587b117b8e05c7e2c03a4cc (bf16 GEMM N=7168 K=512, decode m=1–128), C-store aux 0/16 only: mean per-session 100*(time(default)/time(sc1)-1) 2.115–5.470% over 8 M × 2 inputs (m=64 +5.425% [+5.264, +5.586] and +5.470% [+5.341, +5.598]), every interval's lower end at or above +1.956%; AA intervals between −0.227% and +0.240%, outside ±0.15% on at least one input at m=1, 2, 4, 8, 32, 64 and 128.
