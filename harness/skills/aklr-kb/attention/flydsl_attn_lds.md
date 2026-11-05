---
title: "FlyDSL attention — LDS layout and transposed reads on gfx950"
kind: topic
lever: flydsl_attn_lds
ops: [attention]
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [prefill, backward, decode]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:17
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/cdna4.py:31
  - ROCm/FlyDSL@dd8373349:lib/Dialect/FlyROCDL/CDNA4/CopyAtom.cpp:146
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:92
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:2725
  - ROCm/FlyDSL@dd8373349:kernels/attention/swa_gfx950.py:233
---

# LDS layout and transposed reads in a FlyDSL attention kernel

> P·V contracts over kv, and so do dV = Pᵀ·dO and dK = dSᵀ·Q in the backward pass. With K/V/Q/dO stored
> `[row, d]` in LDS, one operand of each of these GEMMs is strided along the contraction and must be read
> transposed. Generic LDS facts — 64 banks, read phases, swizzle vs padding, write-to-read distance, DMA
> waits — are in `../gemm/datapath.md` §2–§4 and §9 and `../flydsl/guides.md` § lds-optimization.
> This card adds what attention adds.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- You are deciding how V (forward) or dO, Q, K, dS (backward) reach the MFMA transposed.
- A bank-conflict counter or `SQ_WAIT_INST_LDS` is high on an attention kernel.
- You are changing an LDS row stride, padding or swizzle phase.
- You are considering a register transpose, a write-side pre-transpose or a "wider" transposed read.

## 1. The gfx950 transposed reads
- The complete set: `ds_read_b64_tr_b16` (16-bit), `ds_read_b64_tr_b8` (fp8), `ds_read_b64_tr_b4` and
  `ds_read_b96_tr_b6`. `ds_read_b128_tr_b16`/`_b8`, `ds_read_b64_tr_b32`, `ds_load_tr16_b128` and the other
  `ds_load_tr*` ops are not gfx950 instructions.
- The `offset:` immediate is 16-bit unsigned. EXEC must be all ones and the LDS address aligned to the
  data size; `ds_read_b96_tr_b6` does not need an even VGPR pair.
- An 8-element-per-lane bf16 MFMA operand takes two `tr_b16` reads; each lane receives 4 consecutive
  values per read.
- A transposed bf16 operand moves 8 B per lane per instruction against 16 B for a K-contiguous
  `ds_read_b128`. When waves split the q rows, every wave needs the whole K/V tile, so bytes per wave are
  fixed by the tile and the transposed-read count is a structural constant. Budget the layout around it;
  no instruction halves it.

## 2. FlyDSL v0.3.2 API
`flydsl.expr.rocdl` re-exports the ROCDL ops, so three forms exist (`buffer_ops` is the repo's
`kernels/common/buffer_ops.py`, not part of the wheel):
```python
ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_off), address_space=3)    # !llvm.ptr<3>
v4 = rocdl.ds_read_tr16_b64(v4f16_ty, ptr).result      # op class: take .result
v4 = rocdl.ds_read_tr16_b64_(v4f16_ty, ptr)            # function form
atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), bf16)   # copy atom
```
- The other widths: `ds_read_tr8_b64`, `ds_read_tr4_b64`, `ds_read_tr6_b96`; atoms `LDSReadTrans8_64b`,
  `LDSReadTrans4_64b`, `LDSReadTrans6_96b`, or `rocdl.cdna4.LDSReadTrans(granularity, bits)` with only
  (16, 64), (8, 64), (4, 64), (6, 96) accepted.
- With a base pointer plus a compile-time constant, the op folds the constant into `offset:`; the ISA is
  identical to inline asm with an immediate.
- **Use the op or the atom, not inline asm,** unless the reads sit in a hand-scheduled asm region that supplies its
  own waits (`../gemm/datapath.md` §4). The compiler does not track an asm read's destination, so
  every consumer needs a hand-written `lgkmcnt` drain, pinned with `sched_barrier(0)`, which cuts the
  P·V cluster into fragments (`../gemm/datapath.md` §4). *(field, E1)* Swapping asm for the op
  emitted the same 96 reads, let the backend place graded waits, cut the slowdown from split K/V reads
  at 3 waves/SIMD from 3.26% to 1.95%, dropped spills at a 128-register target from 14–105 to 2, and measured
  occupancy rose from 2.985 to 3.978 waves/SIMD; at the power cap the wall gained only 0.50% (sclk −3.3%).
- FlyDSL's dual-wave forward reads V with inline asm and immediate offsets
  (`_ds_read_tr16_b64_imm`); its generic and SWA paths use the op and the atom.

## 3. Lane maps and k order
- **tr_b16**, measured with a position-encoded LDS image *(field, E2)*: each 16-lane group is independent;
  lane L supplies the 4 consecutive 16-bit values `r[L][0..3]` at its address, and output lane m, slot j
  is `r[4j + m//4][m % 4]`. If lane L addresses row `L//4`, columns `4·(L%4)…+3` of a 4×16 block, lane m
  receives column m's 4 rows: `A[m = column][k = row]`.
- **tr_b8** *(field, E3)*: lane j of a group supplies 8 consecutive bytes. With lane j addressing
  `(j//2)·W + (j%2)·8` (W = row bytes), the group transposes an 8-row × 16-byte block and lane j receives
  column j's 8 rows. Four reads give the 32 B/lane operand of a 16x16x128 fp8 MFMA.
- The lane→column map is fixed; the addresses choose only which rows and which 16-column chunk.
- **k order is free, but A and B must use the same one.** `C = Σ_k A[k][m]·B[k][n]`, so any row
  permutation the addresses produce is correct when both operands share it. *(field, E4)* Switching only dS
  to `tr_b8` while K kept its old k order dropped SNR to −1.8 dB. Two `tr_b16` reads 16 rows apart give a
  lane kv `{b..b+3} ∪ {b+16..b+19}`; one `tr_b8` gives 8 consecutive rows. The sets differ, so no byte
  shuffle inside a lane reconciles them: change the write-side layout.
- Probe a read with an LDS image holding its own index and an operand constant along k. All-ones inputs
  cannot see a wrong lane map.

## 4. Transposed reads need whole 16-element blocks
- A swizzle phase in element bits 4–5 (`col ^ ((row & 7) << 4)` on a 128-element row) keeps the
  transposed-read family conflict-free by construction. A non-transposed read that walks 16 rows of one
  column on the same image gets only 4 phases: 4-way.
- **Never move a phase bit into element bit 3** (blocks smaller than 16 elements).
  - *(field, E5)* `(row&7)<<3`, and a split phase built to leave the transposed reads' bank coverage unchanged,
    both read: `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE` 16.4% → 46.5%, `SQ_LDS_IDX_ACTIVE` +56%, wave cycles
    +9.6% / +10.4%, probe time +4.9% / +6.7%, same `ds_read` count. The transposed reads were split into more passes; bank coverage was not the cost.
- To serve a second read family with a different phase, give it its own copy of the tile when LDS allows,
  instead of retuning the shared phase.
- Where a swizzle mask is combined with another address term by `+`, check the two bit fields stay
  disjoint under the new phase. *(field, E6)* An overlap gave finite, silently wrong dK/dV (norms off 0.06% and
  0.48%). Gate layout arms on bitwise-equal output norms (`flydsl_attn_correctness.md`).

## 5. Bank-conflict counters are usually a red herring
- Compare `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE` with a floor computed by hand from the access widths
  (`../gemm/datapath.md` §3), then prove LDS bounds the kernel with a probe before touching
  the layout (`flydsl_attn_attribution.md`).
- *(field, E7)* On MFMA-latency-bound backward kernels, conflicts 19% → 0% (`LDSBankConflict`: `SQ_LDS_BANK_CONFLICT`
  per `GRBM_GUI_ACTIVE` cycle and CU, not the ratio above) and 21.7% → 4.4% bought no speed.
- With bf16 `head_dim = 64` (128-byte rows) the row index adds almost no bank phase and a 64-element row
  leaves little room for an XOR phase: *(field, E8)* every XOR and padding variant stayed near 19%. A
  128-element physical row with `(row&7)<<4` reached 0%; packing two logical rows into each 128-element
  row kept LDS size and global bytes unchanged.
- Separate a second workgroup's LDS pressure from layout conflicts with a forced-occupancy-1 probe (pad
  LDS with a dead buffer). *(field, E9)* On a sparse-MLA backward at occupancy 2, 7.6 points of a 22.8%
  `LDSBankConflict` came from the second workgroup's port and timing pressure; no swizzle moves that part.
- When one tile is read both naturally (QK) and transposed (P·V), one pad cannot satisfy both read
  families; decide which family is on the critical path first.

## 6. Row stride: keep 16-byte alignment
- *(field, E10)* A K/V padding sweep: strides whose rows were only 8-byte aligned split every `ds_read_b128`,
  3.97 ms against 1.84 ms (2.2× the time), with `SQ_LDS_BANK_CONFLICT` at 0 throughout. All 16-byte-aligned pads
  were healthy but not equal (−0.1% to −6.2% around the deployed value).
- Sweep the pad once on a new kernel, with an ISA gate on the `ds_read_b128` count; do not re-sweep a
  tuned one.
- With 16-byte alignment the dword stride is a multiple of 4, so padding alone cannot push a wide read
  below its conflict floor; only an XOR phase rotation with a matching writer can.

## 7. What does not cut transposed-read cost
- **A wider transposed read** does not exist (§1).
- **Write-side pre-transpose through LDS DMA.** One `buffer_load_dwordx4 … lds` chunk is 8 consecutive d
  of one kv; a kv-contiguous image needs 8 consecutive kv of one d, a 2-byte scatter. The free per-lane
  global offset only chooses which row a chunk lands in.
- **A register transpose** (`ds_bpermute` gathers, about 8 permutes and 12 selects per operand, replacing
  one `tr_b16` read) pays only when one transpose is reused across many tiles. *(field, E11)* A sparse-MLA
  backward kernel whose waves each own a d-tile and transpose Q once for all rank tiles: backward +11.8% on the
  one shape it is enabled for (the kernel itself 1.21×); with fewer rank tiles it ran 0.73–0.95×.
  Per-tile transposes: *(field, E12)* a sparse-MLA forward's P·V −77%; *(field, E13)* a dense backward 3.5× slower
  while its `LDSBankConflict` fell to 3%.
- **Fewer LDS instructions is not the currency.** *(field, E14)* Collapsing transposed-read pairs into plain
  `ds_read_b128` where the layout allowed (512 reads removed, −124 instructions) ran 1.65–1.87% slower:
  `lgkmcnt(0)` waits 53 → 118–119. Count graded waits, not reads.
- **One asm block for a run of reads, drained once,** helps only where the compiler drains each read
  separately. *(field, E15, E16)* On sparse-MLA backward kernels, +0.6% there; 0.86–0.93× where the compiler
  already overlapped the reads.

## 8. Addresses
- **A runtime LDS address term loses the DS immediate offset**, and every access recomputes its address.
  *(field, E17)* On a sparse-MLA backward, a per-access runtime XOR swizzle: −23%. *(field, E18)* On a sparse-MLA
  forward, a block index made runtime turned every `tr_b16` offset runtime: 825 → 551 TF/s; returning it as a
  compile-time int under `const_expr` restored 827.
- Keep swizzles as compile-time layout functions. Put per-slot or per-wave variation in the base pointer.
- Keep slot and tile offsets in bits the swizzle does not XOR. If the mask XORs the bits that also carry
  the d-tile index, address differences become lane-dependent and each transposed read needs its own
  address VGPR. *(field, E19)* A second LDS slot raised the read-base registers from 59 to 96 and spilled 197
  dwords.

## 9. Keep LDS write bursts out of transposed-read bursts
A legal restructuring that moved the next head's `ds_write` burst into the middle of a
transposed-read burst cost about 2.4%, estimated after crediting the barriers it removed. Dense
segments are sensitive to LDS port contention; schedule write bursts where the read family is idle.

## Verify
- The ISA shows only `ds_read_b64_tr_*` / `ds_read_b96_tr_b6` for transposed operands, issued by op or
  atom, with immediate offsets.
- Both operands of every MFMA use the same k order, checked with a position-encoded probe.
- Swizzle phases keep 16-element blocks for transposed reads; rows are 16-byte aligned.
- Any conflict-reduction claim comes with a hand-computed floor and a wall-time probe.

## Related
`flydsl_attn_mfma_layout.md` · `flydsl_attn_schedule_sync.md` · `flydsl_attn_registers_occupancy.md` ·
`flydsl_attn_attribution.md` · `flydsl_attn_correctness.md` · `../gemm/datapath.md` ·
`../flydsl/guides.md` § lds-optimization

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. Lines that cite only a `.claude/memory` note or a myskill card have no run record on this machine; the month
is the note's own date or file time. "Meta hd64" is the deterministic D=64 attention backward of the Meta-shape work
(B=4, Hq=128, Hkv=16, D=64, causal).

- E1 [2026-07, gfx950, FlyDSL 0.2.2, N=?, campaign scorer `_bench_campaign_fwd16384.py` (conv TF/s), same-session interleaved A/B; `.vgpr_count` and spills from the ISA dump; occupancy from rocprofv3 PMC] split K/V LDS reads with the V transposed read as inline asm against `rocdl.ds_read_tr16_b64`: at 3 waves/SIMD −3.26% against −1.95%; the same 96 `ds_read_b64_tr_b16` in the ISA; at `waves_per_eu=4` 128 registers with 14–105 spilled dwords against 128 with 2; occupancy 2.985 → 3.978 waves/SIMD; the kept build read +0.50% wall at sclk 1710 → 1654 MHz (−3.3%) (Primus-Turbo hd64 flash-attention forward, B=1, Sq=Skv=16384, causal; round 14; the 96 reads and the 14–105 spills are the round's own KB write-back, /workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:698 and :706; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/memory.md:249)
- E2 [2026-07, gfx950, FlyDSL ?, N=?, position-encoded probe: LDS[i] = i, lane L addressing base + 4L, results read back from global] `ds_read_tr16_b64`: each 16-lane group independent; lane L reads the 4 consecutive i16 r[L][0..3]; output lane m, slot j = r[4j + m//4][m%4]; a group reading 64 contiguous elements gives lane m column m (probe `_probe_tr.py`; note dated by its "2026-07-10" section and file time; local:/workspace/code/gpt_oss_docker/.claude/memory/ds_read_tr16_b64_gfx950.md:10)
- E3 [2026-09, gfx950, FlyDSL ?, N=?, position-encoded probe: every LDS byte holds its own linear index] `ds_read_b64_tr_b8`: lane j of a 16-lane group reads 8 consecutive bytes at (j//2)·W + (j%2)·8 and receives column j's 8 rows (W=16: lane0 → 0, 16 … 112; lane16 → 128, 144 … 240); four reads = 32 B/lane = one 16x16x128 operand (probe `work/fp8bwd/_p_tr8.py`, 2026-09-03; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_ds_read_tr8_b64_lane_map.md:15)
- E4 [2026-09, gfx950, FlyDSL ?, N=?, output SNR against the reference] GEMM3 with A (dS) read by `tr_b8` and B (K) still assembled by the old cvt route: SNR −1.8; two different row maps applied to both operands both matched an fp32 reference (rel 2e-5) (gpt-oss D=64 attention backward in fp8, GEMM3; note of 09-04; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_ds_read_tr8_b64_lane_map.md:45)
- E5 [2026-09, gfx950, FlyDSL ?, N=1, rocprofv3 --pmc, one profile per arm; arms with dK/dV bitwise equal to the control and the ISA unchanged] LDS row phase `row&7 << 3` and the split phase (bits 4–5 `row&3`, bit 3 the row's bit 2): `SQ_LDS_BANK_CONFLICT/SQ_LDS_IDX_ACTIVE` 16.4% → 46.5% for both, `SQ_LDS_IDX_ACTIVE` +56%, wave cycles +9.6% and +10.4%, `ds_read` count unchanged; in-process probe min-of-16 0.47949 and 0.48741 against 0.45697 ms (+4.9% and +6.7%; the campaign memory gives +4.9% for both) (llama-8b D=128 attention backward body, a16 path, occupancy 1, bf16, causal; round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260919_155238/memory.md:117; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260919_155238/rounds/round-07/deep_raw.jsonl:15974)
- E6 [2026-09, gfx950, FlyDSL ?, N=1, dK/dV norms against the control, same seed, same process] first build of `row&7 << 3`, whose pinned transposed-read base `_tr_base` combined mask and lane column with `+`: all outputs finite, dK norm off 0.06%, dV 0.48%; with `^` bitwise equal (same kernel and round; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260919_155238/memory.md:117)
- E7 [2026-07, gfx950, FlyDSL ?, N=?, `LDSBankConflict` = 100·`SQ_LDS_BANK_CONFLICT`/`GRBM_GUI_ACTIVE`/CU count from rocprofv3 PMC; kernel ms] PACK-128 layout at 0% conflicts against the baseline's 19%: same speed; myskill adds "coop 21.7%→4.4%" at the same speed without naming the kernel; PACK-128 plus `s_setprio` 7.09/23.67/31.02 against 7.23/23.99/31.41 ms (−1.2 to −1.9%); on the same kernel the ratio `SQ_LDS_BANK_CONFLICT/SQ_LDS_IDX_ACTIVE` read 50% where `LDSBankConflict` read 19% (Meta hd64 attention backward, dK/dV kernel, Sq 2k/8k/16k; KB notes, no run record found; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/05-lds-swizzle-prefetch-sched.md:48; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_meta_ck_replacement.md:76; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_meta_ck_replacement.md:52)
- E8 [2026-07, gfx950, FlyDSL ?, N=?, `LDSBankConflict` from rocprofv3 PMC] 64-element rows: XOR swizzles (`row&3<<4`, `row&7<<3`) and paddings (stride +2 … +32, with and without swizzle) all left the conflicts at 19% (19.2% for the two swizzles); a 128-element row with `(row&7)<<4` 0.00%; PACK-128 (two logical rows per 128-element block) 0% with LDS still BLOCK_Q×64 and 1× HBM reads (same Meta hd64 dK/dV kernel; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_meta_ck_replacement.md:60)
- E9 [2026-07, gfx950, FlyDSL ?, N=?, `LDSBankConflict` from rocprofv3 PMC; occupancy 1 forced by a dummy LDS buffer] conflicts 22.8 at occupancy 2 against 15.2 at forced occupancy 1; a base-pointer offset between the two workgroups left 22.8 unchanged, so the 7.6 points are port and timing contention (DeepSeek-V4 sparse-MLA backward, dQ kernel; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_bwd_dq_occ_truth.md:95)
- E10 [2026-08, gfx950, FlyDSL ?, N=2, campaign scorer `d64_gptoss_b4` forward ms in palindromic order, two readings per arm against a same-batch control read three times; every arm ISA-gated] K/V LDS line-stride sweep (stride in dwords 256 + pad/2): `KPAD=4` 3.9699 / 3.9671 and `KPAD=12` 3.9718 / 3.9776 ms against control 1.8379 / 1.8295 / 1.8448 (−54%), `SQ_LDS_BANK_CONFLICT` 0 throughout; 16-byte-aligned pads from `VPAD=16` −0.12% and `KPAD=24` −0.2% to `VPAD=8` −6.2%; deployed `KPAD=8`/`VPAD=32` the maximum (gpt-oss D=64 attention forward, B=4, Hq=64, Hkv=8, S=8192, causal; round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/rounds/round-11/NOTES.md:48)
- E11 [2026-07, gfx950, FlyDSL ?, N=?, end-to-end backward throughput of the scored shape] register-transpose interm kernel (`ds_bpermute` 16×16 transpose, waves owning d-tiles, dS/P prefetch): interm 1.918 → 1.583 ms (1.212×), pro cr4 backward 402.5 → 450 (+11.8%); enabled only for num_heads > 64 and topk ≥ 512 (pro cr4); with small R (fewer rank tiles) the transpose ran 0.949× at pro cr128 and 0.730× at pro cr0 (DeepSeek-V4 sparse-MLA backward, pro H=128, cr4, seq 4096; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_interm_register_transpose.md:10)
- E12 [2026-07, gfx950, FlyDSL ?, N=?, ?] register-transpose P·V rewrite of the forward (per-wave gather plus `ds_bpermute` transpose, about 2560 bpermute per token): 97.7 TF/s, −77% (DeepSeek-V4 sparse-MLA forward; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_fwd_opt.md:12)
- E13 [2026-07, gfx950, FlyDSL ?, N=?, dK/dV kernel ms; `LDSBankConflict` from rocprofv3 PMC] plain v4 reads plus a `ds_bpermute` transpose in place of `ds_read_tr16_b64`: conflicts 19% → 3.09%, dK/dV bitwise equal, 7.10 → 25.48 ms and 32 → 109 ms (3.5×), MfmaUtil 38% → 12% (Meta hd64 attention backward, dK/dV kernel; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_meta_ck_replacement.md:57)
- E14 [2026-09, gfx950, FlyDSL ?, N=?, campaign scorer `_bench_attn_b24.py` per-cell ms; ISA counts] transposed-read pairs collapsed into plain `ds_read_b128`: tr reads 512 → 0, b128 192 → 448, instructions −124, vgpr 467 → 461; in-loop `lgkmcnt(0)` 53 → 119, b24_bwd +1.65%, b32_bwd +1.87%; through `Vec.load` `lgkmcnt(0)` 53 → 118, +1.6–1.9% (gpt-oss D=64 attention backward, fused dK/dV body, B=24 and B=32, Hq=64, Hkv=8, S=8192, causal; round 26; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260905_095125/memory.md:407)
- E15 [2026-07, gfx950, FlyDSL ?, N=15, win-count over 15 trials alternating in one process] the 16 `ds_read_b64_tr_b16` of each rank tile issued as one inline-asm block drained by one `lgkmcnt(0)`: R1152 +0.6% (15/15), R640 +0.6% (15/15), R192 +0.3% (14/15) (DeepSeek-V4 sparse-MLA backward, interm kernel; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_interm_register_transpose.md:52)
- E16 [2026-07, gfx950, FlyDSL ?, N=?, ?] the same asm block over the dQ kernel's 32 P·V `ds_read_b64_tr_b16` (chunks of 8): 0.86–0.93× on all 6 shapes, output correct (99 dB) (DeepSeek-V4 sparse-MLA backward, dQ kernel; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_interm_register_transpose.md:60)
- E17 [2026-07, gfx950, FlyDSL ?, N=?, speed ratio against the reference kernel] per-access runtime XOR swizzle (dslice/hg key): 0.63, −23% (DeepSeek-V4 sparse-MLA backward, dQ kernel at occupancy 2; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_bwd_dq_occ_truth.md:96)
- E18 [2026-07, gfx950, FlyDSL ?, N=?, TF/s] `dt_base = block_idx.z` as a runtime value made the `tr16` offsets runtime: default baseline 825 → 551; `_dtidx` returning a compile-time int under `const_expr` restored 827 (DeepSeek-V4 sparse-MLA forward, pro cr4; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_fwd_dualwave_port.md:74)
- E19 [2026-07, gfx950, FlyDSL ?, N=?, ISA dump (`vspill`, `ds_read` base registers) and same-process interleaved A/B] two heads resident in LDS (`heads_per_dma=2`): 1004 (−15%), vspill 4 → 197, still 197 with the head barrier kept; both slots on one LDS buffer: vspill 0, vgpr 253; `ds_read` base registers 59 → 96 (Meta hd64 attention backward, dK/dV kernel, Sq=16384; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dkdv_clamp_hazard_truth.md:52)
