---
title: CDNA3 (gfx942) ISA — the reference guide
kind: index
gens: [gfx942]
---

# CDNA3 ISA

`cdna3/` holds AMD's "AMD Instinct MI300 Instruction Set Architecture Reference Guide" (5-August-2025), the ISA of
gfx942 (MI300X, MI308), converted from the PDF to Markdown and split by chapter (chapter 12 by section). The
conversion was checked against the PDF page by page; the split files differ from the converted guide only in the
links of `cdna3/01_contents.md`, which point at the file that holds each heading, and in the blank line that ended
each split. The guide's Specification Agreement and notices are in `cdna3/00_front_matter.md`. Figures are in
`cdna3/images/`.

Use it to settle what an instruction does on gfx942, which encoding and operands it takes, what a counter or wait
state requires, and what the hardware limits are. There are no guides to looking things up here yet; the chapters
have the same layout as the CDNA4 guide's, so `../gfx950/lookup.md`, `../gfx950/matrix_core.md` and
`../gfx950/memory_and_sync.md` show where to look, with gfx942's own chapter in place of CDNA4's. What gfx942
changes for a GEMM is in `../../gemm/flydsl_gemm_gfx942.md`.

## File map

| File | Covers |
|---|---|
| `cdna3/00_front_matter.md` | title page, Specification Agreement, disclaimer |
| `cdna3/01_contents.md` | table of contents, linked to the chapter files |
| `cdna3/02_preface.md` | scope, audience, conventions |
| `cdna3/ch01_introduction.md` | terminology |
| `cdna3/ch02_program_organization.md` | compute shaders, data sharing, device memory |
| `cdna3/ch03_kernel_state.md` | PC, EXEC, status and mode registers, GPRs and LDS, M0, SCC, VCC, traps, hardware IDs |
| `cdna3/ch04_program_flow_control.md` | program control, branching, workgroups, data dependency resolution, manually inserted wait states |
| `cdna3/ch05_scalar_alu_operations.md` | SALU formats, operands and instruction groups |
| `cdna3/ch06_vector_alu_operations.md` | VALU encodings, operands, denormals and rounding, clamp, VGPR indexing, packed math |
| `cdna3/ch07_matrix_arithmetic_instructions.md` | MFMA, BF8 and FP8 formats and conversions, floating-point handling, sparse matrices, required independent instructions |
| `cdna3/ch08_scalar_memory_operations.md` | SMEM encoding, operations, dependency checking, alignment and bounds |
| `cdna3/ch09_vector_memory_operations.md` | buffer instructions and resources, float memory atomics |
| `cdna3/ch10_flat_memory_instructions.md` | flat, global and scratch addressing |
| `cdna3/ch11_data_share_operations.md` | LDS: dataflow, access; GWS programming restriction |
| `cdna3/ch12_00_instructions.md` … `ch12_16_instruction_limitations.md` | every instruction, one file per encoding: SOP2, SOPK, SOP1, SOPC, SOPP, SMEM, VOP2, VOP1, VOPC, VOP3P, VOP3A/VOP3B, LDS and GWS, MUBUF, MTBUF, FLAT/scratch/global; instruction limitations |
| `cdna3/ch13_microcode_formats.md` | the bit layout of every encoding |

## Conventions in the text

- `<!-- page N -->` marks the PDF page the following text comes from.
- Every instruction is a heading `#### NAME` in upper case, followed by `Opcode: N`, its description and usually
  its pseudocode (1,480 of them). From the skill folder, `rg -n '^#### V_MFMA' hardware/gfx942/cdna3/` lists the
  MFMA instructions.
- The text is a conversion of the PDF. A table that crosses pages sits under its first page's marker, and a few
  converted tables keep the PDF's own misalignment. Check page numbers, table cells and signs that matter against
  the PDF before citing them.
