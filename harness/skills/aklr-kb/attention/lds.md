---
title: "Attention — LDS layout and transposed reads on gfx950"
kind: topic
lever: attn_lds
ops: [attention, mla]
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [many query rows, dQ dK dV, transposed operand, sparse]
flydsl: 0.3.2
---

# LDS layout and transposed reads in an attention kernel

> P·V contracts over kv, and so do dV = Pᵀ·dO and dK = dSᵀ·Q in the gradient kernels. With K/V/Q/dO stored
> `[row, d]` in LDS, one operand of each of these GEMMs is strided along the contraction and must be read
> transposed. Generic LDS facts — 64 banks, read phases, swizzle vs padding, write-to-read distance, DMA
> waits — are in `../gemm/datapath.md` §2–§4 and §9 and `../flydsl/guides.md` § lds-optimization.
> This card adds what attention adds.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- You are deciding how V (for P·V) or dO, Q, K, dS (for dQ, dK, dV) reach the MFMA transposed.
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

## 2. Issue transposed reads through the intrinsic, not inline asm
- The compiler intrinsics are `llvm.amdgcn.ds.read.tr16.b64`, `…tr8.b64`, `…tr4.b64` and `…tr6.b96`. With a
  base pointer plus a compile-time constant, the intrinsic folds the constant into `offset:`; the ISA is
  identical to inline asm with an immediate.
- **Use the intrinsic, not inline asm,** unless the reads sit in a hand-scheduled asm region that supplies its
  own waits (`../gemm/datapath.md` §4). The compiler does not track an asm read's destination, so
  every consumer needs a hand-written `lgkmcnt` drain, pinned with `sched_barrier(0)`, which cuts the
  P·V cluster into fragments (`../gemm/datapath.md` §4). *(field, E1)* Swapping asm for the intrinsic
  emitted the same 96 reads, let the backend place graded waits, cut the slowdown from split K/V reads
  at 3 waves/SIMD from 3.26% to 1.95%, dropped spills at a 128-register target from 14–105 to 2, and measured
  occupancy rose from 2.985 to 3.978 waves/SIMD; at the power cap the wall gained only 0.50% (sclk −3.3%).

In FlyDSL: `../flydsl/attention.md` § lds §2.

## 3. Lane maps and k order
- **tr_b16**, measured with a position-encoded LDS image *(field, E2)*: each 16-lane group is independent;
  lane L supplies the 4 consecutive 16-bit values `r[L][0..3]` at its address, and output lane m, slot j
  is `r[4j + m//4][m % 4]`. If lane L addresses row `L//4`, columns `4·(L%4)…+3` of a 4×16 block, lane m
  receives column m's 4 rows: `A[m = column][k = row]`.
- **tr_b8** (S1): lane j of a group supplies 8 consecutive bytes. With lane j addressing
  `(j//2)·W + (j%2)·8` (W = row bytes), the group transposes an 8-row × 16-byte block and lane j receives
  column j's 8 rows. Four reads give the 32 B/lane operand of a 16x16x128 fp8 MFMA.
- The lane→column map is fixed; the addresses choose only which rows and which 16-column chunk.
- **k order is free, but A and B must use the same one.** `C = Σ_k A[k][m]·B[k][n]`, so any row
  permutation the addresses produce is correct when both operands share it. Two `tr_b16` reads 16 rows apart give a
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
  0.48%). Gate layout arms on bitwise-equal output norms (`correctness.md`).

## 5. Bank-conflict counters are usually a red herring
- Compare `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE` with a floor computed by hand from the access widths
  (`../gemm/datapath.md` §3), then prove LDS bounds the kernel with a probe before touching
  the layout (`attribution.md`).
- *(field, E7)* On MFMA-latency-bound backward kernels, conflicts 19% → 0% (`LDSBankConflict`: `SQ_LDS_BANK_CONFLICT`
  per `GRBM_GUI_ACTIVE` cycle and CU, not the ratio above) and 21.7% → 4.4% bought no speed.
- With bf16 `head_dim = 64` (128-byte rows) the row index adds almost no bank phase and a 64-element row
  leaves little room for an XOR phase: *(field, E8)* every XOR and padding variant stayed near 19%. A
  128-element physical row with `(row&7)<<4` reached 0%; packing two logical rows into each 128-element
  row kept LDS size and global bytes unchanged.
- Separate a second workgroup's LDS pressure from layout conflicts with a forced-occupancy-1 probe (pad
  LDS with a dead buffer).
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
  one shape it was first enabled for (the kernel itself 1.21×); with fewer rank tiles it lost to the tuned kernel
  until the transpose burst overlapped the HBM loads, then won on every shape of that head count in isolation
  (1.04–1.35×).
  Per-tile transposes: *(field, E12)* a sparse-MLA forward's P·V −77%.
- **Fewer LDS instructions is not the currency.** *(field, E14)* Collapsing transposed-read pairs into plain
  `ds_read_b128` where the layout allowed (512 reads removed, −124 instructions) ran 1.65–1.87% slower:
  `lgkmcnt(0)` waits 53 → 118–119. Count graded waits, not reads.
- **One asm block for a run of reads, drained once,** helps where the compiler drains each read
  separately. *(field, E15)* On a sparse-MLA backward kernel, +0.6% there.

## 8. Addresses
- **A runtime LDS address term loses the DS immediate offset**, and every access recomputes its address.
- Keep swizzles as compile-time layout functions. Put per-slot or per-wave variation in the base pointer.
- Keep slot and tile offsets in bits the swizzle does not XOR. If the mask XORs the bits that also carry
  the d-tile index, address differences become lane-dependent and each transposed read needs its own
  address VGPR.

In FlyDSL: `../flydsl/attention.md` § lds §8.

## 9. Keep LDS write bursts out of transposed-read bursts
A legal restructuring that moved the next head's `ds_write` burst into the middle of a
transposed-read burst cost about 2.4%, estimated after crediting the barriers it removed. Dense
segments are sensitive to LDS port contention; schedule write bursts where the read family is idle.

## Verify
- The ISA shows only `ds_read_b64_tr_*` / `ds_read_b96_tr_b6` for transposed operands, issued by the
  intrinsic, with immediate offsets.
- Both operands of every MFMA use the same k order, checked with a position-encoded probe.
- Swizzle phases keep 16-element blocks for transposed reads; rows are 16-byte aligned.
- Any conflict-reduction claim comes with a hand-computed floor and a wall-time probe.

## Related
`mfma_layout.md` · `schedule_sync.md` · `registers_occupancy.md` ·
`attribution.md` · `correctness.md` · `../gemm/datapath.md` ·
`../flydsl/guides.md` § lds-optimization

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`.
"Primus-Turbo" commits are those of the Primus-Turbo repository;
their messages carry the numbers.
"Meta hd64" is the deterministic D=64 attention backward of the Meta-shape work
(B=4, Hq=128, Hkv=16, D=64, causal).

- E1 [2026-07, gfx950, FlyDSL 0.2.2, N=?, campaign scorer (conv TF/s), same-session interleaved A/B; `.vgpr_count` and spills from the ISA dump; occupancy from rocprofv3 PMC] split K/V LDS reads with the V transposed read as inline asm against `rocdl.ds_read_tr16_b64`: at 3 waves/SIMD −3.26% against −1.95%; the same 96 `ds_read_b64_tr_b16` in the ISA; at `waves_per_eu=4` 128 registers with 14–105 spilled dwords against 128 with 2; occupancy 2.985 → 3.978 waves/SIMD; the kept build read +0.50% wall at sclk 1710 → 1654 MHz (−3.3%) (Primus-Turbo hd64 flash-attention forward, B=1, Sq=Skv=16384, causal; the 96 reads and the 14–105 spills: unverified)
- E2 [2026-07, gfx950, FlyDSL ?, N=?, position-encoded probe: LDS[i] = i, lane L addressing base + 4L, results read back from global] `ds_read_tr16_b64`: each 16-lane group independent; lane L reads the 4 consecutive i16 r[L][0..3]; output lane m, slot j = r[4j + m//4][m%4]; a group reading 64 contiguous elements gives lane m column m (a `ds_bpermute` replica taking slot j of lane m from lane 4j + m//4, element m%4, matched it element-wise; Primus-Turbo 330ad461)
- E5 [2026-09, gfx950, FlyDSL ?, N=1, rocprofv3 --pmc, one profile per arm; arms with dK/dV bitwise equal to the control and the ISA unchanged] LDS row phase `row&7 << 3` and the split phase (bits 4–5 `row&3`, bit 3 the row's bit 2): `SQ_LDS_BANK_CONFLICT/SQ_LDS_IDX_ACTIVE` 16.4% → 46.5% for both, `SQ_LDS_IDX_ACTIVE` +56%, wave cycles +9.6% and +10.4%, `ds_read` count unchanged; in-process probe min-of-16 0.47949 and 0.48741 against 0.45697 ms (+4.9% and +6.7%; also reported as +4.9% for both) (llama-8b D=128 attention backward body, a16 path, occupancy 1, bf16, causal)
- E6 [2026-09, gfx950, FlyDSL ?, N=1, dK/dV norms against the control, same seed, same process] first build of `row&7 << 3`, whose pinned transposed-read base `_tr_base` combined mask and lane column with `+`: all outputs finite, dK norm off 0.06%, dV 0.48%; with `^` bitwise equal (same kernel and round)
- E7 [2026-07, gfx950, FlyDSL ?, N=?, `LDSBankConflict` = 100·`SQ_LDS_BANK_CONFLICT`/`GRBM_GUI_ACTIVE`/CU count from rocprofv3 PMC; kernel ms] PACK-128 layout at 0% conflicts against the baseline's 19%: same speed; also reported: "coop 21.7%→4.4%" at the same speed, kernel not named; PACK-128 plus `s_setprio`: dK/dV −1.2 to −1.9% against the baseline on every Sq (Meta hd64 attention backward, dK/dV kernel; unverified)
- E8 [2026-07, gfx950, FlyDSL ?, N=?, `LDSBankConflict` from rocprofv3 PMC] 64-element rows: XOR swizzles (`row&3<<4`, `row&7<<3`) and paddings all ineffective, an XOR swizzle capped at 4-way (~19%); a 128-element row with `(row&7)<<4` 0.00%; PACK-128 (two logical rows per 128-element block) 0% with LDS at the 64-element size and 1× HBM reads (same Meta hd64 dK/dV kernel)
- E10 [2026-08, gfx950, FlyDSL ?, N=2, campaign scorer `d64_gptoss_b4` forward ms in palindromic order, two readings per arm against a same-batch control read three times; every arm ISA-gated] K/V LDS line-stride sweep (stride in dwords 256 + pad/2): `KPAD=4` 3.9699 / 3.9671 and `KPAD=12` 3.9718 / 3.9776 ms against control 1.8379 / 1.8295 / 1.8448 (−54%), `SQ_LDS_BANK_CONFLICT` 0 throughout; 16-byte-aligned pads from `VPAD=16` −0.12% and `KPAD=24` −0.2% to `VPAD=8` −6.2%; deployed `KPAD=8`/`VPAD=32` the maximum (gpt-oss D=64 attention forward, B=4, Hq=64, Hkv=8, S=8192, causal)
- E11 [2026-07, gfx950, FlyDSL ?, N=?, end-to-end backward throughput of the scored shape; isolated kernel time for the later commit] register-transpose interm kernel (`ds_bpermute` 16×16 transpose, waves owning d-tiles, dS/P prefetch): interm 1.918 → 1.583 ms (1.212×), pro cr4 backward 402.5 → 450 (+11.8%); enabled only for num_heads > 64 and topk ≥ 512 (pro cr4), flash and small R (fewer rank tiles) keeping the tuned wide-BD kernel; after a later commit overlapped the transpose burst with the HBM loads, the isolated kernel beat that one on every pro shape (cr0 0.519 against 0.538 ms, cr128 0.591 against 0.799, cr4 1.422 against 1.909), end to end unmeasured (DeepSeek-V4 sparse-MLA backward, pro H=128, seq 4096; Primus-Turbo d0ff7050; Primus-Turbo 21effd1a; Primus-Turbo 415f63da)
- E12 [2026-07, gfx950, FlyDSL ?, N=?, ?] register-transpose P·V rewrite of the forward (per-wave gather plus `ds_bpermute` transpose, about 2560 bpermute per token): 97.7 TF/s, −77% (DeepSeek-V4 sparse-MLA forward; unverified)
- E14 [2026-09, gfx950, FlyDSL ?, N=?, campaign scorer per-cell ms; ISA counts] transposed-read pairs collapsed into plain `ds_read_b128`: tr reads 512 → 0, b128 192 → 448, instructions −124, vgpr 467 → 461; in-loop `lgkmcnt(0)` 53 → 119, b24_bwd +1.65%, b32_bwd +1.87%; through `Vec.load` `lgkmcnt(0)` 53 → 118, +1.6–1.9% (gpt-oss D=64 attention backward, fused dK/dV body, B=24 and B=32, Hq=64, Hkv=8, S=8192, causal)
- E15 [2026-07, gfx950, FlyDSL ?, N=15, win-count over 15 trials alternating in one process] the 16 `ds_read_b64_tr_b16` of each rank tile issued as one inline-asm block drained by one `lgkmcnt(0)`: R1152 +0.6% (15/15), R640 +0.6% (15/15), R192 +0.3% (14/15) (DeepSeek-V4 sparse-MLA backward, interm kernel; Primus-Turbo@dce83e02:primus_turbo/flydsl/attention/kernels/sparse_mla_v2/_bwd_interm_kernel.py:184)
- S1 Primus-Turbo@c527e317:primus_turbo/flydsl/utils/gemm_helper.py:1825 `S2RLoaderTr`, the fp8 GEMM's `ds_read_b64_tr_b8` operand loader, puts lane j of a 16-lane group on row j//2 and adds (j%2)·8 bytes (:1843), and issues 4 such reads per operand tile (:1847)
