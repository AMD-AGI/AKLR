---
title: CDNA4 ISA — finding an instruction, encoding, operand rule, limit or data format
kind: method
ops: [any]
gens: [gfx950]
flydsl: 0.3.2
---

# Finding and reading things in the CDNA4 ISA

> Paths in the text are relative to `isa/`; the shell commands run from the skill folder. § and table
> numbers and quoted strings are the guide's own. The guide calls its target "CDNA4"; it never names gfx950
> or a product.

## Route here when
- A dump or a FlyDSL kernel shows an instruction and you need what it does, its opcode or its encoding.
- You need an operand rule, a register limit, a hardware-register field or a number-format bit layout.
- Not here: MFMA operand layout and dependencies → `matrix_core.md`; memory instructions, `s_waitcnt`, LDS,
  barriers and wait states → `memory_and_sync.md`.

## 1. Finding an instruction

Every instruction except the two `V_MFMA_SCALE_*` is a heading `#### NAME` in a chapter-12 file, followed by
`Opcode: N`, a one-sentence
description, usually a pseudocode block and sometimes a **Notes** paragraph. From an LLVM/FlyDSL mnemonic: drop
an encoding suffix (`_e32`, `_e64`, `_sdwa`, `_dpp`; no heading has one), upper-case, match the whole line.

```sh
n=$(echo v_add_f32_e64 | sed -E 's/_(e32|e64|sdwa|dpp)$//' | tr a-z A-Z)
grep -rn "^#### $n\$" isa/cdna4/                    # or: rg -n "^#### $n\$" isa/cdna4/
awk -v h="#### $n" '/^#/{p=($0==h); if(p) print "== " FILENAME} p' isa/cdna4/ch12_*.md    # full entries
awk -v h="#### $n" '/<!-- page/{pg=$3} $0==h{print FILENAME": PDF page "pg}' isa/cdna4/ch12_*.md
grep -rn '^#### V_CVT_SCALEF32_.*FP4' isa/cdna4/     # a family
awk '/^#### /{n=$2} /^Opcode: 53$/{print n}' isa/cdna4/ch12_10_vop3p_instructions.md     # opcode -> name
grep -rni 'v_mfma_f32_16x16x128_f8f6f4' isa/cdna4/   # every mention, prose and tables included
```

| File | § | Encoding | Headings |
|---|---|---|---|
| `cdna4/ch12_01_sop2_instructions.md` … `ch12_05_sopp_instructions.md` | §12.1–12.5 (§12.5.1 Send Message) | SOP2, SOPK, SOP1, SOPC, SOPP | 52, 21, 54, 20, 30 |
| `cdna4/ch12_06_smem_instructions.md` | §12.6 | SMEM (`S_LOAD_*`, …) | 82 |
| `cdna4/ch12_07_vop2_instructions.md` | §12.7 | VOP2 | 62 |
| `cdna4/ch12_08_vop1_instructions.md` | §12.8 | VOP1 | 85 |
| `cdna4/ch12_09_vopc_instructions.md` | §12.9 | VOPC (`V_CMP*`) | 198 |
| `cdna4/ch12_10_vop3p_instructions.md` | §12.10 | VOP3P, VOP3P-MAI: `V_PK_*`, `V_DOT*`, `V_MAD_MIX*`, `V_MFMA_*`, `V_SMFMAC_*`, `V_ACCVGPR_READ/WRITE` | 104 |
| `cdna4/ch12_11_vop3a_vop3b_instructions.md` | §12.11 | VOP3A, VOP3B | 507 |
| `cdna4/ch12_12_lds_instructions.md` | §12.12 | DS (`DS_*`) | 126 |
| `cdna4/ch12_13_mubuf_instructions.md` | §12.13 | MUBUF (`BUFFER_*`) | 74 |
| `cdna4/ch12_14_mtbuf_instructions.md` | §12.14 | MTBUF (`TBUFFER_*`) | 16 |
| `cdna4/ch12_15_flat_scratch_and_global_instructions.md` | §12.15.1–3 | FLAT, SCRATCH, GLOBAL | 54, 27, 61 |
| `cdna4/ch12_16_instruction_limitations.md` | §12.16 | instructions that cannot use DPP or SDWA | — |

`S_` names are spread over six files, so search all of `cdna4/`.

**Reading an entry** (e.g. `V_PK_ADD_F32`, ch12_10, PDF page 287):
- `Opcode: N` is the OP-field value in the encoding of the file you are in (§13). The same name in another
  file has another number.
- Pseudocode: `S0`/`S1`/`S2` are sources, `D0` the destination, `.f32`/`.u32`/`.b64` how the bits are read;
  `laneId`, `VCC.u64[laneId]` and `OPSEL` are per-lane state and instruction fields. Helpers such as
  `CalcBufferAddr`, `CalcDsAddr` and `f32_to_fp4_scale` are used but defined nowhere in the guide; take their
  meaning from the description sentence and the chapter text.
- Type suffixes: "If an instruction has two suffixes (for example, _I32_F32), the first suffix indicates the
  destination type, the second the source type" (`cdna4/ch12_00_instructions.md` §12). The §13 intro says the
  opposite; the instructions follow §12 (`V_CVT_F32_FP8` converts FP8 to F32).
- **Notes** give precision ("0.5ULP precision, denormals are supported"), which modifiers apply and, for MFMA,
  "This instruction performs N passes".
- Entries are in opcode order, not in the "alphabetic order" the preface promises. ch12_11 runs VOP1-origin
  entries first, then VOP2-origin and VOP3-only ones, then VOPC-origin ones.

**One instruction, several encodings.** All 85 VOP1 and 198 VOPC instructions, and 58 of the 62 VOP2 ones,
appear again in ch12_11 with their VOP3 opcode and the same description and pseudocode. Only `V_FMAMK_F32`,
`V_FMAAK_F32`, `V_MADMK_F16` and `V_MADAK_F16` have no VOP3 entry. The VOP3 form adds ABS/NEG/OMOD/CLMP and an
SGPR-pair carry or compare destination, and gives up the literal constant (§6.1, §6.2.2, §3.9, §12.7.1).
- VOP2 → VOP3: "VOP2 opcode + 0x100" (§12.7.1). `V_ADD_F32` 1 → 257.
- VOPC → VOP3A: "+ 0x000" (§12.9.1). `V_CMP_EQ_F32` is 66 in both.
- VOP1 → VOP3: §12.8.1 says "+ 0x140", but the `Opcode:` printed for VOP1-origin entries in ch12_11 and in
  Table 85 (§13.3.4) is VOP1 + 384 (`V_NOP` 0 → 384, `V_CVT_F32_FP8` 84 → 468). Those numbers collide with
  VOP3-only opcodes: 468 is printed for both `V_CVT_F32_FP8` and `V_MAX3_I32`. Use +0x140 (320) and confirm
  with `llvm-mc -arch=amdgcn -mcpu=gfx950 -show-encoding`, which encodes `v_nop_e64` with OP = 0x140.
- VOP3B is used only by the ten instructions listed in §12.11 and §13.3.5 (`V_ADD_CO_U32`, `V_DIV_SCALE_F32`,
  `V_MAD_U64_U32`, …); all other VOP3 entries are VOP3A.
- FLAT, SCRATCH and GLOBAL are separate names that share opcodes and differ only in the SEG field (§13.6):
  `FLAT_`, `SCRATCH_` and `GLOBAL_LOAD_DWORDX4` are all `Opcode: 23`.

**Names that differ from the heading.**
- `v_accvgpr_read_b32` / `v_accvgpr_write_b32` → `#### V_ACCVGPR_READ` / `#### V_ACCVGPR_WRITE` (ch12_10,
  opcodes 88 and 89).
- `V_MFMA_SCALE_F32_16X16X128_F8F6F4` and `…_32X32X64_F8F6F4` have no heading; they are in
  `cdna4/ch07_matrix_arithmetic_instructions.md` §7.1.2 Table 28 and §7.2.1 (`matrix_core.md`). The unscaled
  `V_MFMA_F32_16X16X128_F8F6F4` has a heading in ch12_10.
- The conversion tables (§6.7.1; §7.3 Tables 31–32) drop `V_` and write `CVT_SCALE_PK_FP4_F32`,
  `CVT_SR_FP16_F32`; the headings are `V_CVT_SCALEF32_PK_FP4_F32` and `V_CVT_SR_F16_F32`. Table 32's 6-bit names
  also leave out `PK32` / `2XPK16` (`CVT_SCALE_PK_FP6_F32` is `V_CVT_SCALEF32_2XPK16_FP6_F32`), so search a short
  fragment such as `FP6_F32`.
- ch07 writes some names with a lower-case `x` (`V_MFMA_F32_16x16x128_F8F6F4`); grep prose with `-i`.

## 2. Encodings (chapter 13)

Most subsections of `cdna4/ch13_microcode_formats.md` have a bit-map figure (an image), a **Fields** table
(every field with its bits; source fields carry the full operand-code list) and an **Opcodes** table
(number → name); §13.3.7–13.3.9 (SDWA, SDWAB, DPP) have no Opcodes table, and §13.6.2–13.6.3 have neither a
figure nor a Fields table. Table 64 gives the widths: SOP2, SOPK, SOP1, SOPC, SOPP, VOP1, VOP2, VOPC, DPP and
SDWA are 32 bits; SMEM, VOP3A/B, VOP3P, VOP3P-MAI, DS, MTBUF, MUBUF and FLAT/GLOBAL/SCRATCH are 64 bits.

| Format | ENCODING ("Must be") | OP | § | Instructions |
|---|---|---|---|---|
| SOP2 | [31:30] `10` | [29:23] | §13.1.1 | ch12_01 |
| SOPK | [31:28] `1011` | [27:23] | §13.1.2 | ch12_02 |
| SOP1 | [31:23] `10_1111101` | [15:8] | §13.1.3 | ch12_03 |
| SOPC | [31:23] `10_1111110` | [22:16] | §13.1.4 | ch12_04 |
| SOPP | [31:23] `10_1111111` | [22:16] | §13.1.5 | ch12_05 |
| SMEM | [31:26] `110000` | [25:18] | §13.2.1 | ch12_06 |
| VOP2 | [31] `0` | [30:25] | §13.3.1 | ch12_07 |
| VOP1 | [31:25] `0_111111` | [16:9] | §13.3.2 | ch12_08 |
| VOPC | [31:25] `0_111110` | [24:17] | §13.3.3 | ch12_09 |
| VOP3A, VOP3B | [31:26] `110100` | [25:16] | §13.3.4–13.3.5 | ch12_11 |
| VOP3P, VOP3P-MAI | `110100111` (see below) | [22:16] | §13.3.6, §13.3.6.1 | ch12_10 |
| DS | [31:26] `110110` | [24:17] | §13.4.1 | ch12_12 |
| MTBUF | [31:26] `111010` | [18:15] | §13.5.1 | ch12_14 |
| MUBUF | [31:26] `111000` | [24:18] | §13.5.2 | ch12_13 |
| FLAT, GLOBAL, SCRATCH | [31:26] `110111`; SEG [15:14] = 0, 2, 1 | [24:18] | §13.6 | ch12_15 |

- **Instruction → format:** the §12 file that holds its heading names the format; read that format's §13
  Fields table.
- **Instruction word → instruction:** match ENCODING, longest value first (VOP3P before VOP3A/B; VOP1 and VOPC
  before VOP2; SOP1, SOPC and SOPP before SOPK and SOP2). Extract OP and look it up in the §13 Opcodes table or as `Opcode: N` in
  the §12 file. `0xD1010000`: [31:26] = `110100` (VOP3A), OP = 0x101 = 257 → `V_ADD_F32` in ch12_11.
  `0xD3B24000`: [31:23] = `110100111` (VOP3P), OP = 0x32 = 50 → `V_PK_ADD_F32`.
- Tables 88–89 print the VOP3P ENCODING field as `[31:24]`, but the value has nine bits. OP is [22:16] and
  ch12_10 has opcodes up to 127, so the encoding occupies [31:23].
- **Extra dwords:** a 32-bit literal follows when a source field is 255 (§5.2, §6.2.1). An SDWA or SDWAB
  dword (§13.3.7–13.3.8) or a DPP dword (§13.3.9) follows a VOP1/VOP2/VOPC word whose SRC0 is 249 or 250. The
  scaled MFMAs are "4-dword instructions" (§7.2.1).
- **VALU modifier fields:** VOP3A ABS [10:8], OPSEL [14:11], CLMP [15], OMOD [60:59], NEG [63:61]; VOP3B
  SDST [14:8]; VOP3P NEG_HI, OPSEL, OPSEL_HI (Table 88). MFMA fields (CBSZ, ABID, ACC_CD, ACC, BLGP) →
  `matrix_core.md`. DS/MUBUF/FLAT fields (offsets, SC0/SC1/NT, LDS) → `memory_and_sync.md`.

## 3. Operands

The operand codes are in Table 23 (`cdna4/ch06_vector_alu_operations.md` §6.2.2) for VALU and in Table 13
(`cdna4/ch05_scalar_alu_operations.md` §5.2) for SALU; each §13 Fields table repeats them under SRC0/SSRC0.

| Code | Meaning | Code | Meaning |
|---|---|---|---|
| 0–101 | SGPR0–SGPR101 | 128 | 0 |
| 102, 103 | FLAT_SCRATCH_LO, _HI | 129–192 | integers 1..64 |
| 104, 105 | XNACK_MASK_LO, _HI | 193–208 | integers −1..−16 |
| 106, 107 | VCC_LO, VCC_HI | 235–238 | SHARED_BASE, SHARED_LIMIT, PRIVATE_BASE, PRIVATE_LIMIT |
| 108–123 | TTMP0–TTMP15 (privileged) | 240–247 | 0.5, −0.5, 1.0, −1.0, 2.0, −2.0, 4.0, −4.0 |
| 124 | M0 | 248 | 1/(2·π) |
| 126, 127 | EXEC_LO, EXEC_HI | 249, 250 | SDWA, DPP ("only valid as Source-0") |
| 251–253 | VCCZ, EXECZ, SCC as `{ zeros, X }` | 255 | 32-bit literal from the instruction stream |
| 256–511 | VGPR0–VGPR255 | 125, 209–234, 239, 254 | reserved |

- The float codes are "Single, double, or half-precision inline floats": 240 is 0.5 in the operand's own type.
  The exact bits of 248 are "half: 0x3118 single: 0x3e22f983 double: 0x3fc45f306dc9c882" (Table 23).
- Field width decides the reach: "all codes can be used when the vector source is nine bits; codes 0 to 255 …
  eight bits; codes 0 to 127 … seven bits" (§6.2.2). VOP2's VSRC1 [16:9] and VDST [24:17] hold plain VGPR
  numbers; the SALU destination is 7 bits (Table 13).
- MFMA sources (VOP3P-MAI, Table 89): codes 0–107 and 249–255 are reserved, so there is no SGPR and no
  literal, and the inline floats are float32.
- Limits (§6.2.1): "At most one SGPR can be read per instruction, but the value can be used for more than one
  operand"; "At most one literal constant can be used, and only when an SGPR or M0 is not used as a source".
  ADDC, SUBB and CNDMASK already read VCC. "Any of the 32-bit microcode formats may use a 32-bit literal
  constant, but not VOP3" (§6.1); SOPP and SOPK cannot (§5.2). Expansion of a literal used as a 64-bit
  source: §6.2.1.1.
- Alignment: 64-bit SGPR data in an even pair, quad alignment when a scalar load returns four or more dwords
  (§3.6.3, §5.2); 64-bit VGPR data in even VGPRs (§3.6.4); packed 32-bit ops (`V_PK_*_F32`) on even VGPRs (§6.7).
- AGPRs are "AccVGPR", `AV0-AV255` in Table 2 (§3.1). No operand code reaches them; an ACC bit selects them:
  VOP3P-MAI ACC [60:59] and ACC_CD [15], DS ACC [25], MUBUF/MTBUF/FLAT ACC [55] (§13). Moves:
  `V_ACCVGPR_READ`, `V_ACCVGPR_WRITE`, `V_ACCVGPR_MOV_B32`.
- Special registers: Table 2 (§3.1) lists all wave state with sizes; EXEC §3.3; M0 §3.7 (LDS-DMA offset, GPR
  index); SCC §3.8, §5.3; VCC and VCCZ §3.9 ("VCC physically resides in the highest two user SGPRs"). VGPR
  indexing through M0: §6.6. Out of range: a source reads SGPR0 or VGPR0 and a destination is not written
  (§3.6.1, §6.2.3).
- Modifiers (VOP3 only): ABS/NEG on float inputs, OMOD (×0.5, 1, 2, 4) and CLAMP on outputs. OMOD is ignored
  when output denormals are enabled or the IEEE mode bit is set (§6.2.2).

## 4. Kernel state and limits (chapter 3)

| Item | The ISA says | Where |
|---|---|---|
| Wave | "groups 64 work-items into a wavefront" | ch02 intro, ch01 §1.1 |
| Workgroup | "Up to 16 wavefronts (1024 work-items)" | ch04 §4.3 |
| VGPRs | "A wave may have up to 512 total VGPRs, 256 of each type"; the split between the two need not be equal | ch03 §3.6.4 |
| VGPR granule | "VGPRs are allocated in groups of eight Dwords." | ch03 §3.6.4 |
| SGPRs | "16 to 102 SGPRs, in units of 16 GPRs (Dwords)", "logically viewed as SGPRs 0-101"; VCC is SGPR 106–107; a trap handler reserves 16 more | ch03 §3.6.2 |
| LDS | "160 kB", "64 banks, each with 640 entries of 4 bytes"; "One work-group can request up to 160 kB" | ch02 §2.2.1, ch11 §11.1 |
| LDS granule | "contiguous blocks of 1280 bytes on 1280-byte alignment"; accesses clamp to the smaller of the allocation and M0 | ch03 §3.6.5 |
| Counters | VMCNT 6 bits, LGKMCNT 4 bits, EXPCNT 3 bits "Unused" | ch03 Table 2 → `memory_and_sync.md` |
| PC | 48-bit byte address; S_GET_PC, S_SET_PC, S_SWAP_PC use "an even-aligned SGPR pair" | ch03 §3.2 |

- Readbacks use other units: GPR_ALLOC's VGPR_SIZE is "as [7:2]. 0=4 VGPRs, 1=8 VGPRs", ACCV_OFF is "in units
  of 4 VGPRs", LDS_ALLOC counts "in units of 64 Dwords" (§5.8 Tables 21–22), and ch09 §9.1.9.1 speaks of "units
  of 32 Dwords". The allocation granules are the §3.6.4 and §3.6.5 statements (8 VGPRs, 1280 bytes).
- Table 2 lists SGPRs as "S0-S103", against §3.6.2 and the operand codes (SGPRs 0–101, 102–107 special).
- Not in the guide: waves per SIMD or per CU, occupancy as a function of VGPRs or LDS, CU count, clocks.

**Wave state and hardware registers.**
- STATUS (read-only: SCC, EXECZ, VCCZ, IN_BARRIER, HALT, …): §3.4 Table 3. MODE (FP_ROUND [3:0], FP_DENORM
  [7:4], DX10_CLAMP, IEEE, FP16_OVFL [23], GPR_IDX_EN, VSKIP, …): §3.5 Table 4 and §6.4 Table 26. TRAPSTS:
  §3.10.1 Table 5; trap registers and TTMPs: §3.10. HW_ID, XCC_ID: §3.12 Tables 6–7. Memory violations: §3.11.
- Access: `S_GETREG_B32` / `S_SETREG_B32` (ch12_02). The hwRegId codes are §5.8 Table 19 (1 MODE, 2 STATUS,
  3 TRAPSTS, 4 HW_ID, 5 GPR_ALLOC, 6 LDS_ALLOC, 7 IB_STS, 20 XCC_ID, …). "SIMM16 = {size[4:0], offset[4:0],
  hwRegId[5:0]}"; the pseudocode reads the size as `SIMM16[15:11] + 1`.
- Launch state (§3.13): VGPR0 holds "{ 2’b00, Z[9:0], Y[9:0], X[9:0] }", the index within the workgroup;
  Table 8 gives the SGPR load order (user data, workgroup IDs, then the wave-in-group word).
- Field by name: `grep -n 'FP16_OVFL' isa/cdna4/ch0[3-7]*.md`.

## 5. Data formats and conversions

- **Bit layouts:** ch07 §7.3 Table 30 ("Small Float Data Formats", PDF page 65) gives sign-exponent-mantissa,
  bias, zero/INF/NaN encodings, max and min for FP16; FP8 E4M3 (bias 7, no INF, NaN 0x7F/0xFF, max 448);
  BF8 E5M2 (bias 15, INF 0x7C/0xFC, max 57344); FP6 E2M3 (max 7.5), BF6 E3M2 (max 28.0) and FP4 E2M1
  (max 6.0), which have no INF or NaN. BF16, F32 and F64 get only the §12 suffix definitions ("Bfloat16
  format", "IEEE 754 …"); there is no BF16 bit table.
- Three Table 30 cells are wrong in the source (PDF page 65 prints the same); recompute from the field widths
  and the bias. BF6's min denorm `S.000.11 = +/- 0.0675` has E3M2 widths, but E3M2 with bias 3 gives `S.000.01` =
  0.0625 (and `S.000.11` is 0.1875). FP4's `S01.0` only lacks a dot; its value 1.0 is right. FP4's `S.0.01` has
  E1M2 widths; E2M1's min denorm is `S.00.1` = 0.5.
- **Block scaling (MX):** the guide never says "MX" or "OCP". §7.2: block size 32 along K; the scale is "an
  8-bit exponent (bias 127)", "0xFF is NaN"; §6.7.1 names it "E8M0, bias of 127". How the scaled MFMA reads its
  scales: §7.2.1 → `matrix_core.md`. The F8F6F4 format codes (0 = E4M3 … 4 = E2M1, selected by CBSZ for A and
  BLGP for B) are in the ch12_10 entry `V_MFMA_F32_16X16X128_F8F6F4`.
- **Conversions**, in reading order:
  1. §6.3 Table 24 lists every VALU instruction by encoding, `V_CVT_*` included (VOP3P is listed in §6.7).
  2. §7.3 Table 31 covers the unscaled FP8/BF8 and stochastic-rounding converts (encoding, OP_SEL, rounding);
     Table 32 and §6.7.1 the scaled 4/6/8-bit converts. The rules after Table 32 cover SDWA byte selection for
     VOP1 8-bit sources, E8M0 scales from a VGPR or inline constant, FP4/FP6 saturation, OP_SEL[3:2] byte
     steering, the FP16_OVFL table for F32 → FP8/BF8, and "SH_MEM_CONFIG, bit[8] must be set to 1 to produce
     the correct results for BF8 and FP8 operations".
  3. The entries: `grep -rn '^#### V_CVT_' isa/cdna4/`. The scaled converts (`V_CVT_SCALEF32_*`, including the
     `PK32` and `2XPK16` forms that Table 32 omits) exist only in ch12_11. `V_CVT_F32_FP8`, `V_CVT_PK_F32_FP8`
     and their BF8 twins are VOP1 entries with VOP3 copies.
  4. Rounding and denormals: MODE (§3.5) and §6.4 Table 26 (V_DOT2 flushes denormals); MFMA §7.4; float
     atomics §9.2.
- **Buffer data formats** (typed buffers): the codes are text in ch09 §9.1.2 Table 41 (DFMT 0–15, NFMT 0–7) and
  ch13 Table 97, which skips DFMT 7. The figure `cdna4/images/x393.png` (§9.1.11, PDF page 95) adds which
  DATA_FORMAT / NUM_FORMAT pairs are legal and NUM_FORMAT 9 SRGB, which a 3-bit NFMT field cannot hold; open it
  for the pairs.

## 6. Reading caveats

- **Page markers.** `<!-- page N -->` marks where a PDF page starts. A table that crosses pages sits whole under
  its first page's marker and the following markers are empty: ch07 Table 38 spans PDF pages 75–77 under the
  page-74 marker, and ch13 has several such runs. Rows can also land under the previous page's marker (the M0,
  EXEC→DPP, v_cmpx and permlane rows of ch04 Table 11 are on PDF page 29). To find the marker above a line:
  `awk -v L=415 '/<!-- page/{pg=$3} FNR==L{print "marker: page " pg; exit}' isa/cdna4/ch12_08_vop1_instructions.md`;
  confirm the page against the PDF before citing it.
- **Tables.** Converted tables can be misaligned: the rows of ch11 Table 57 (§11.3.1) are shifted by a column
  (`memory_and_sync.md` §4.1). Check a table cell, and any sign you rely on in a formula, constant, exponent or
  bit range, against the PDF page.
- **Text only in images.** grep cannot find what these figures carry; open them: `cdna4/images/x276.png`
  (block-scale formula, ch07 §7.2, page 64), `x280.png` (encoding of the scaled MFMA, ch07 §7.2.1, page 64; no
  §13 Fields table restates it), `x346.png` (buffer address equation, ch09 §9.1.1, page 83), `x359.png`
  (Figure 4, buffer addressing, ch09 §9.1.5, page 87), `x377.png` (LDS and memory address of a buffer load to
  LDS, ch09 §9.1.9, page 92) and `x393.png` (legal data-format pairs, ch09 §9.1.11, page 95). Address terms →
  `memory_and_sync.md`.
- **Empty alt text.** The other figures (encoding bit maps, …) appear as `![](images/xNNN.png)`; find them by the
  paragraph above. For most encodings the §13 Fields tables give the same fields as text.
- **The PDF's own cross-references** ("See the S_SETPRIO instruction (page 12-49)", Table 3) use printed
  numbering, not `<!-- page N -->`; search by name.
- **The guide contradicts itself** in places; these are in the source, not the conversion: the VOP1 → VOP3
  opcode (§1 above), the VOP3P ENCODING width (§2 above), the type-suffix order (§1 above), the SGPR count (§4
  above), Table 30's denorm cells (§5 above), and MODE.EXCP_EN: Table 4 lists bits up to [20] for a field declared [18:12], and §3.10
  calls it `MODE . EXCP_EN[8:0]`.

## Related
`INDEX.md` (file map) · `matrix_core.md` · `memory_and_sync.md` · `cdna4/01_contents.md` · `../SKILL.md`
