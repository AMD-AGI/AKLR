---
title: CDNA4 ISA — the matrix cores (MFMA) for a GEMM main loop
kind: topic
ops: [gemm, moe, attention]
dtypes: [bf16, fp16, fp8, mxfp8, mxfp4]
regimes: [any]
gens: [gfx950]
flydsl: 0.3.2
---

# Matrix cores (MFMA) in the CDNA4 ISA

> What AMD's CDNA4 ISA guide says about MFMA, for someone writing or tuning a gfx950 GEMM main loop. Every
> statement cites the guide. Values marked *(derived)* are arithmetic on cited numbers, not text from the guide.
> Short names: **ch07** = `cdna4/ch07_matrix_arithmetic_instructions.md`, **vop3p** =
> `cdna4/ch12_10_vop3p_instructions.md`, **ch13** = `cdna4/ch13_microcode_formats.md`, **ch03** =
> `cdna4/ch03_kernel_state.md`. General instruction lookup is in `lookup.md`; memory instructions, `s_waitcnt`,
> LDS and barriers are in `memory_and_sync.md`.

## Route here when
- You are choosing an MFMA atom and need its shape, opcode, register count or cycle count.
- You are building operand fragments or reading accumulators and need the lane-to-element map.
- You are writing inline asm around MFMAs and must count the wait states yourself.
- A GEMM card makes a claim about MFMA hazards, register files or rates and you want the spec text.

## 1. MFMA families on gfx950

**Naming.** "V_MFMA_[output type]\_[M]X[N]X[K][\_[B]B]\_[input type] where B (which is 1 if not specified) is the
number of matrices (or blocks)" (ch07 §7.1). Each instruction is a `#### V_MFMA_…` or `#### V_SMFMAC_…` heading
in vop3p, followed by `Opcode: N`; from the skill folder,
`rg -n '^#### V_S?MFMA' isa/cdna4/ch12_10_vop3p_instructions.md` lists them and ch13 §13.3.6.1 Table 90 maps the
numbers. Chapter 7 gives the dense list with cycle counts (§7.1.2 Table 28), the F8F6F4 forms (§7.1.5) and the
sparse list (§7.5 Table 33).

The atoms a dense GEMM uses (C and D are F32, I32 for I8):

| Instruction (opcode) | A, B | Registers A / B / C,D per lane | Cycles (Table 28) | Passes |
|---|---|---|---|---|
| `V_MFMA_F32_16X16X32_BF16` (53), `…_F16` (84) | bf16, f16 | 4 / 4 / 4 *(derived)* | 16 | 4 |
| `V_MFMA_F32_32X32X16_BF16` (55), `…_F16` (85) | bf16, f16 | 4 / 4 / 16 *(derived)* | 32 | 8 |
| `V_MFMA_I32_16X16X64_I8` (54) | i8 | 4 / 4 / 4 *(derived)* | 16 | 4 |
| `V_MFMA_I32_32X32X32_I8` (56) | i8 | 4 / 4 / 16 *(derived)* | 32 | 8 |
| `V_MFMA_F32_16X16X32_{FP8,BF8}_{FP8,BF8}` (112–115) | fp8 or bf8 per operand | 2 / 2 / 4 *(derived)* | 16 | 4 |
| `V_MFMA_F32_32X32X16_{FP8,BF8}_{FP8,BF8}` (116–119) | fp8 or bf8 per operand | 2 / 2 / 16 *(derived)* | 32 | 8 |
| `V_MFMA_F32_16X16X128_F8F6F4` (45) | FP8, BF8, FP6, BF6 or FP4 per operand | F8 8, F6 6, F4 4 / same / 4 | 32 if either is F8, else 16 | 8 or 4 |
| `V_MFMA_F32_32X32X64_F8F6F4` (46) | as above | as above / 16 | 64 if either is F8, else 32 | 16 or 8 |
| `V_MFMA_SCALE_F32_16X16X128_F8F6F4`, `…_32X32X64_…` | as above, plus E8M0 scales | as above, plus one scale source each for A and B | as unscaled | — |

- **Passes.** vop3p states "This instruction performs N passes" for 84, 85 and 112–119. It states none for
  45, 46 and 53–56; for those the column is Table 28's cycles ÷ 4, from "PASS | 4 clock cycles" (ch07 §7.6).
- **Register counts** *(derived)* follow ch07 §7.1.4: `K_L = K / (64 / (M * B))` items per lane, two per register
  for 16-bit, four for 8-bit, eight for 4-bit; 6-bit inputs need "six contiguous registers" for 32 values. C and D
  hold M·N·B/64 32-bit items. The F8F6F4 counts are stated: F8 in both ch07 §7.1.5 and the vop3p entries'
  "Matrix size requirements", F6 and F4 only in ch07 §7.1.5.
- **XF32:** no XF32 instruction appears in any chapter file.
- **Other dense families** (Table 28): F32 inputs (`32x32x1_2B`, `16x16x1_4B`, `4x4x1_16B`, `32x32x2`,
  `16x16x4`); the older F16/BF16 (`32x32x8`, `16x16x16`) and I8 (`32x32x16`, `16x16x32`) forms with half the K;
  the multi-block `_2B`/`_4B`/`_16B` forms; F64 (`16x16x4`, `4x4x4_4B`).
- **Sparse SMFMAC** (ch07 §7.5): "a 4:2 structurally-sparse matrix A and dense matrices B, C, and D"; "Only the
  A-matrix may be sparse". Table 33 has F16/BF16, I8 and FP8/BF8 forms in 16x16 and 32x32, e.g.
  `V_SMFMAC_F32_16X16X64_BF16` (16 cycles). "the C and D matrices are identical, referenced by the instruction's
  VDST field"; SRC2 holds the sparsity index and "can only be VGPR", whatever ACC_CD says.

## 2. Operands and layout

### Register files and operand fields
VOP3P-MAI fields (ch13 §13.3.6.1 Table 89): VDST [7:0], CBSZ [10:8], ABID [14:11], ACC_CD [15], OP [22:16],
SRC0 [40:32], SRC1 [49:41], SRC2 [58:50], ACC [60:59], BLGP [63:61].

- **A and B choose their file independently; C and D share one.** "ACC[0] : 0 = read SRC-A from Arch VGPR; 1 =
  read SRC-A from Acc VGPR", ACC[1] the same for SRC-B; ACC_CD "Indicates that SRC-C and VDST use ACC VGPRs"
  (Table 89; same in ch07 §7.1).
- **Sources.** Table 89 has no SGPR codes ("0 - 107 | Reserved"): 128 is 0, 129–208 are small integers,
  240–248 float constants, 256–511 VGPRs. ch07 §7.1.2 rules: "Src0/1 can be only VGPR, SRC2 can be
  inline/constant"; F8F6F4 (§7.1.5 rules): "Src0/1 can be only VGPR/ACC_VGPR. SRC2 can be
  VGPR/ACC_VGPR/Constant". A zero accumulator can therefore be the inline constant 0.
- **Alignment.** ch07 §7.1: the registers "must be contiguous, and … the first register must be aligned to the
  number of registers required" (A in four registers: v4–v7 legal, v5–v8 not). The §7.1.2 and §7.1.5 rules
  tables only say "need to be even aligned". Meeting the first rule meets both.
- **Capacity.** ch03 §3.1 Table 2: V0–V255 and AV0–AV255. ch03 §3.6.4: "A wave may have up to 512 total VGPRs,
  256 of each type"; below 512 the split "is flexible"; "VGPRs are allocated in groups of eight Dwords".
- **Moves.** `V_ACCVGPR_READ` (vop3p, opcode 88), `V_ACCVGPR_WRITE` (89), `V_ACCVGPR_MOV_B32`
  (`cdna4/ch12_08_vop1_instructions.md`, opcode 82). VOP3P instructions are VALU instructions (ch06 §6.3).
  AccVGPRs "can also be loaded directly from memory" (ch03 §3.6.4): the DS, MUBUF/MTBUF and FLAT encodings carry
  an ACC bit (ch13 §13.4.1 bit 25, §13.5–§13.6 bit 55); see `memory_and_sync.md`.
- **EXEC, rounding, denormals.** "ignores exec mask from MODE and forces it to 1 for all threads"; "forces it to
  RNE"; input and output denormals are kept (ch07 §7.1.2 and §7.1.5 rules). §7.4 agrees for inputs narrower
  than 32 bits but says the F32-input MFMAs "honor the denormal-handling flags in MODE", which the §7.1.2 rules
  table contradicts (§6 below). Correct FP8/BF8 results need
  "SH_MEM_CONFIG, bit[8] … set to 1" (ch07 §7.3).
- **Negate.** The vop3p notes of 45, 46 and 53–56 say "NEG[1:0] and ABS[1:0] must be zero. NEG[2] and ABS[2]
  may be used to control matrix C". Table 89 has no NEG or ABS field. Only the F64 MFMAs reuse BLGP as negate
  bits for A, B and C (ch07 §7.1.6.4).

### Lane layout (ch07 §7.1.4)
- **Inputs.** "A[b,i,k] is placed in the item `k % K_L` of lane `i + M * (b + B * (k / K_L))`"; B[b,k,j] the
  same with j and N (§7.1.4.1). Items fill each 32-bit register little-endian from bit 0 (§7.1.4).
- **Outputs.** H = 4 for every non-F64 MFMA, `B_I = ceil(64 / (N * M / H))`, `M_I = (64 / B_I) / N`,
  `G = M / (H * M_I)`; D[b,i,j] is item `(i % H) + H * (i/(H * M_I) + G * (b / B_I))` of lane
  `j + N * ((i / H) % M_I + M_I * (b % B_I))` (§7.1.4.2).

Worked out for the GEMM atoms *(derived; they agree with the BF16 and IU8 tables in §7.1.5.1 and the
32x32 example in §7.1.3.1)*:

| Atom | Lane l holds of A (B: column instead of row) | Register r of lane l holds D[i][j] with |
|---|---|---|
| 16x16x32 bf16/f16 | row l%16, k = 8·(l/16) … +7 | i = 4·(l/16) + r, j = l%16, r = 0..3 |
| 32x32x16 bf16/f16 | row l%32, k = 8·(l/32) … +7 | i = 8·(r/4) + 4·(l/32) + r%4, j = l%32, r = 0..15 |
| 16x16x64 i8 | row l%16, k = 16·(l/16) … +15 | as 16x16 |
| 16x16x32 fp8/bf8 | row l%16, k = 8·(l/16) … +7 | as 16x16 |

- **F8F6F4** (§7.1.5.1 tables). FP4 and FP6 follow the formula: in 16x16x128, lane group l/16 holds
  k = 32·(l/16) … +31. The FP8 tables do not: in 16x16x128, lanes 0–15 hold "k=0-15" in v0–v3 and "k=64-79" in
  v4–v7; in 32x32x64, lanes 0–31 hold k = 0–15 and 32–47. Check PDF pages 59–60 before you build FP8 fragments
  from either source.
- The LDS transpose reads that produce MFMA operand layouts (`DS_READ_B64_TR_B16/_B8/_B4`,
  `DS_READ_B96_TR_B6`) are in ch11 §11.4; see `memory_and_sync.md`.

### CBSZ, ABID, BLGP (ch07 §7.1.6)
- **CBSZ and ABID** broadcast one block of A's lanes: `S = 64 / (1 << CBSZ)`, `p_a(l_a) = (l_a % S) + (S * ABID)`.
  "The largest legal value of CBSZ is 4"; (1 << CBSZ) above the block count is "undefined"; ABID ≥ (1 << CBSZ)
  "is not legal" (§7.1.6.1).
- **BLGP** permutes B's lanes (Table 29): 0 none; 1 and 2 broadcast the first or second 32 lanes; 3 "Rotate 16
  lanes left", `(l_b + 16) % 64`; 4–7 broadcast one 16-lane group.
- The documented output layout needs "CBSZ, ABID, and BLGP … all set to 0" (§7.1.3.1).
- **F8F6F4 repurposes them.** "CBSZ[2:0] defines the matrix A format, BLGP[2:0] defines the matrix B format":
  000 E4M3 (FP8), 001 E5M2 (BF8), 010 E2M3 (FP6), 011 E3M2 (BF6), 100 E2M1 (FP4); any mix is allowed and no
  broadcast happens (§7.1.5 rules, §7.1.6.2, §7.1.6.5). ABID[0] becomes the scale enable.

### Scaled MFMA (ch07 §7.2, §7.2.1, §7.1.5 rules)
- **Format.** The scale block is 32 along K, for F4, F6 and F8. The scale is "an 8-bit exponent (bias 127) with
  valid values in: -127, 127 (0xFF is NaN)", applied "after the normal dot product prior to output/accumulation"
  (§7.2).
- **Encoding.** A 4-dword instruction "that looks like two back-to-back VOP3P's": a Load-Scale pair, then the
  MFMA (§7.2.1). SRC0 is the A scale and SRC1 the B scale (9-bit `SCALE_SRC0/1` in figure
  `cdna4/images/x280.png`). `{OP_SEL_HI[n], OP_SEL[n]}` selects one byte of every lane: 00 → [7:0], 01 → [15:8],
  10 → [23:16], 11 → [31:24]. "Scale values (SRC0 and SRC1) can be either VGPRs or Inline constants (floats,
  using only the exponent portion)."
- **Lane map.** 64 scales per operand per instruction, "one-quarter of one VGPR across 64 lanes" (§7.2.1). Lane l
  carries row l%16, K block l/16 for 16x16x128, and row l%32, K block l/32 for 32x32x64 (§7.2.1 scale tables).
  One VGPR therefore holds four instructions' scales *(derived)*.
- **Enable.** "ABID[0] = 1'b1 : Must be set for V_MFMA_SCALE_…"; "ABID[0] = 1'b0 : forces all scales into the
  ALU as 1.0f" (§7.1.5 rules). "The scale value is used just for one instruction and does not carry forward into
  non-"scale" MFMA ops" (§7.2.1).
- **Exponent rule as printed:** "d_exp = (a0_exp+b0_exp) + (a1_exp+b1_exp) + … + c_exp + scale_a + scale_b"
  (§7.1.5 rules); PDF page 59 prints the same, with every term added.

## 3. Dependencies and wait states

Why they exist: "their partially-written results may be observable, a certain amount of independent instructions
must sometimes be present between the issuance of a matrix core instruction and accesses to its results or
modification of the registers that hold its inputs" (ch07 introduction). The rules are ch07 §7.6 Table 38
("VOP3P-Matrix Opcodes Required NOPs"); ch04 §4.5 Table 11 has no MFMA row. The program supplies them: "insert
NOPs (or independent VALU instructions)" (§7.6). `S_NOP n` covers n+1 ("s_nop 0 // Wait 1 cycle", "s_nop 0xf //
Wait 16 cycles", `cdna4/ch12_05_sopp_instructions.md` S_NOP), so a wait of 18 or 20 takes two.

**Terms** (§7.6): "XDLOP | Matrix math on {I8, F16, BF16}", "DGEMM | V_MFMA…F64", "PASS | 4 clock cycles".
"SGEMM" is used but not defined. FP8/BF8, F8F6F4 and the SCALE forms are not named; applying the XDL rows to
them is a reading, not text.

**Reading caveat.** Table 38 prints four values in each multi-value row, with no labels. The order 2, 4, 8, 16
passes of the first (writing) MFMA is inferred: the SGEMM block names its four rows `V_MFMA_F32_4X4X4F16`,
`…16X16X16F16`, `…32X32X8F16` and `…32X32X4F16`, which Table 28 gives 8, 16, 32 and 64 cycles. The table runs over
PDF pages 75–77; the Markdown places all of it under the page-74 marker.

**XDL or V_SMFMA\* writes a VGPR, then** (Table 38; the pass-count column labels are inferred):

| Second instruction | 2 pass | 4 pass | 8 pass | 16 pass | ISA comment |
|---|---|---|---|---|---|
| XDL reads SrcC exactly the same as the 1st vDst | 2 | 0 | 0 | 0 | "the two V_MFMA must be the same number passes and vDst and vSrc start from the same offset and same VGPR size" |
| XDL reads SrcC overlapped with the 1st vDst | 4 | 6 | 10 | 18 | "overlapped with XDL" |
| S/DGEMM reads SrcC | 3 | 6 | 10 | 18 | "Overlapped with S/DGEMM" |
| V_MFMA reads SrcA/SrcB (V_SMFMA\* also its index SrcC) | 5 | 8 | 12 | 20 | "No internal forwarding path, need to wait previous V_MFMA/V_SMFMA* commit result to VGPR" |
| VM, LDS, FLAT, Export read overlapped with the 1st vDst; VALU read/write (RAW + WAW) | 5 | 8 | 12 | 20 | |

Other rows:

| First instruction | Second instruction | Waits | ISA comment |
|---|---|---|---|
| Non-DLops VALU writes a VGPR | V_MFMA\* or V_SMFMA\* reads it | 2 | "No internal 4 & 8 cycle forwarding path." |
| XDL/SMFMA reads SrcC | VALU writes a VGPR overlapping the 1st SrcC (WAR) | 1 / 3 / 7 / 15 | "XDL read at S11, VALU write at S11, so needs 1 wait for this case" |
| V_CMPX\* writes EXEC | V_MFMA\* | 4 | "Doesn't support execution mask forwarding with XDL/DGEMM" |
| SGEMM writes a VGPR | SrcC exact · S/DGEMM SrcC overlapped · XDL SrcC overlapped · SrcA/B · VALU/VM/LDS | 2,0,0,0 · 2,4,8,16 · 0 · 4,6,10,18 · 4,6,10,18 | |

The DLops and F64 rows are in the same table; they do not bear on a dense GEMM loop.

**Applied to a GEMM loop** *(derived, under the reading above)*:
- **K chain on one accumulator.** The same atom accumulating into the same registers needs 0 for every atom in §1
  (all have at least 4 passes). If the pass count (an F8F6F4 operand switching between F8 and F6/F4), the start
  register or the size differs, the overlapped row applies: 6 / 10 / 18.
- **MFMAs on disjoint registers** match no row.
- **Reading accumulators out** (`V_ACCVGPR_READ`, any VALU, `ds_write`, `buffer_store`) or a VALU overwriting
  them: 8 after a 4-pass atom (16x16x32 bf16; 16x16x128 with FP6/FP4), 12 after an 8-pass atom (32x32x16 bf16;
  16x16x128 with F8), 20 after a 16-pass atom (32x32x64 with F8). Table 38 says "VGPR" throughout and does not
  separate AccVGPRs from arch VGPRs.
- **Zero-initialised accumulator.** A VALU zero-fill needs 2 before the MFMA reads it; an inline-constant 0 in
  SRC2 needs none.
- **An MFMA result used as A or B** of a later MFMA (not a plain GEMM): 8 / 12 / 20.
- **Overwriting A/B fragments** while an MFMA that read them is in flight: Table 38 has no row for a VALU, LDS or
  VMEM write to an MFMA's SrcA/SrcB registers. The guide is silent; probe the compiler (`llc`) instead.

## 4. Throughput

What the guide states:
- "The core operation implemented inside the matrix core is the 4 × 1 times 1 × 4 outer matrix product, yielding
  16 output values", combined "both in parallel and in series" (ch07 introduction).
- Cycle counts (Tables 28 and 33), "PASS | 4 clock cycles" (§7.6), pass counts in vop3p, and for F8F6F4 "Larger
  cycle count if either matrix A or B is F8" (Table 28).
- Co-execution: Table 38's WAR row is headed "Co-Execution Anti-Dependency" and says "XDL and VALU could access
  arch VGPR". With §7.6's "insert NOPs (or independent VALU instructions)" and the ch07 introduction's "a certain
  amount of independent instructions", it implies that other instructions execute while an MFMA is in flight.
  "XDL 2x opcodes read SRCC Faster" (SGEMM block) is not explained.
- Not stated in any chapter file: the MFMA issue rate per SIMD or CU, whether the cycle count blocks the next
  MFMA, whether LDS or VMEM instructions issue during an MFMA, the core clock frequency, or peak FLOP/s.

MACs per cycle *(derived: M·N·K·B ÷ Table 28 cycles)*:

| Atoms | MAC / cycle |
|---|---|
| f16/bf16 `32x32x8`, `16x16x16` | 256 |
| f16/bf16 `16x16x32`, `32x32x16`; fp8/bf8 `16x16x32`, `32x32x16`; i8 `32x32x16`, `16x16x32` | 512 |
| i8 `16x16x64`, `32x32x32`; F8F6F4 with an F8 operand | 1024 |
| F8F6F4 with FP6/BF6/FP4 on both operands | 2048 |

- fp8 through F8F6F4 runs at twice the per-cycle rate of the fp8 `16x16x32` form.
- A 16x16 atom and its 32x32 sibling tie per cycle. They differ in C/D registers per instruction (4 vs 16) and in
  the wait before the accumulators can be read (8 vs 12 for bf16).

## 5. Where this meets the GEMM cards

| Card section | Here |
|---|---|
| `../gemm/flydsl_gemm_mainloop.md` §5 MFMA atom choice | §1 atom table, §4 rates |
| mainloop §4 Register plan and inline-asm constraints | §2 register files and sources; §3 "VALU writes → MFMA reads: 2" (its constraint rule 1) |
| mainloop §2 8-wave pipeline, §3 4-wave whole loop | §3: K chains need no padding; the peeled first phase's `src2=0` is the inline constant; an epilogue scheduled between the closing MFMAs must still cover 8 / 12 / 20 |
| mainloop §7 Occupancy and allocation knobs | ch03 §3.6.4: granule 8, 256 + 256 per wave. The card's occupancy tiers and round-to-4 rule are not in the guide |
| `../gemm/flydsl_gemm_correctness.md` §6 Bounds (MFMA ignores EXEC) | §2 "forces it to 1 for all threads" |
| correctness §7 Register classes and inline asm | §3: the card's 8 / 8 / 12 / 20 match the VALU row at 4 / 4 / 8 / 16 passes |

## 6. Gaps and inconsistencies in the chapter text
- **Table 38:** four values per row with no pass-count labels (§3 caveat).
- **Load-Scale prefix:** the prose and figure x280 give 0xD3AC; the operand table gives "0xCC35 in bits [31:16]".
- **SCALE opcodes:** Table 90 and vop3p give no opcode number for `V_MFMA_SCALE_*`.
- **Clamp:** the ch07 rules tables say "Supported. uses the FP16_OVFL bit"; vop3p notes for 45, 46 and 53–56 say
  "CLAMP is not supported".
- **NEG[2]/ABS[2]** are named in vop3p notes but have no field in Table 89.
- **Alignment:** count-aligned (§7.1) versus even-aligned (rules tables).
- **FP8 F8F6F4 layout** tables disagree with the §7.1.4 formula (§2).
- **F64 `4X4X4_4B`:** 32 cycles in Table 28, "4 passes" in vop3p.
- **Denormals of F32-input MFMAs:** §7.4 says they honor MODE; the §7.1.2 rules table says the MFMAs ignore
  MODE's denorm control.

## Related
`INDEX.md` · `lookup.md` · `memory_and_sync.md` · `../gemm/flydsl_gemm_mainloop.md` ·
`../gemm/flydsl_gemm_correctness.md` · `../gemm/flydsl_gemm_schedule_and_locality.md`
