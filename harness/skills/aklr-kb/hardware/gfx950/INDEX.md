---
title: CDNA4 (gfx950) ISA — the reference guide and how to use it
kind: index
gens: [gfx950]
---

# CDNA4 ISA

`cdna4/` holds AMD's "CDNA4 Instruction Set Architecture Reference Guide", the ISA of gfx950 (MI350/MI355X),
converted from the PDF to Markdown and split by chapter (chapter 12 by section). The text is not hand-edited:
the split files differ from the converted guide only in the links of `cdna4/01_contents.md`, which point at the
file that holds each heading, and in the blank line that ended each split (see also the caveats below). The
guide's Specification Agreement and notices are in `cdna4/00_front_matter.md`. Figures are in `cdna4/images/`.

Use it to settle what an instruction does, which encoding and operands it takes, what a counter or wait state
requires, and what the hardware limits are. The cards elsewhere in this skill cite measured behaviour; the ISA
is the specification to check them against.

## Start here

| You want to… | Read |
|---|---|
| find an instruction, an encoding, an operand rule, a register limit or a data format | `lookup.md` |
| use MFMA in a GEMM: families, operand layout, dependencies and wait states | `matrix_core.md` |
| move data: buffer/global loads, cache bits, LDS DMA, LDS reads, `s_waitcnt`, barriers, hazards | `memory_and_sync.md` |
| read a chapter in full | the file map below, or `cdna4/01_contents.md` |

## File map

| File | Covers |
|---|---|
| `cdna4/00_front_matter.md` | title page, Specification Agreement, disclaimer |
| `cdna4/01_contents.md` | table of contents, linked to the chapter files |
| `cdna4/02_preface.md` | scope, audience, conventions |
| `cdna4/ch01_introduction.md` | terminology |
| `cdna4/ch02_program_organization.md` | compute shaders, data sharing, device memory |
| `cdna4/ch03_kernel_state.md` | PC, EXEC, status and mode registers, GPRs and LDS, M0, SCC, VCC, traps, hardware IDs, GPR initialization |
| `cdna4/ch04_program_flow_control.md` | program control, branching, workgroups, data dependency resolution, manually inserted wait states |
| `cdna4/ch05_scalar_alu_operations.md` | SALU formats, operands and instruction groups |
| `cdna4/ch06_vector_alu_operations.md` | VALU encodings, operands, denormals and rounding, clamp, VGPR indexing, packed math |
| `cdna4/ch07_matrix_arithmetic_instructions.md` | MFMA, block-scaled matrices, FP8/BF8 and smaller formats, sparse matrices, required independent instructions |
| `cdna4/ch08_scalar_memory_operations.md` | SMEM encoding, operations, dependency checking, alignment and bounds |
| `cdna4/ch09_vector_memory_operations.md` | buffer instructions and resources, float memory atomics |
| `cdna4/ch10_flat_memory_instructions.md` | flat, global and scratch addressing |
| `cdna4/ch11_data_share_operations.md` | LDS: dataflow, access, MFMA transpose load from LDS |
| `cdna4/ch12_00_instructions.md` … `ch12_16_instruction_limitations.md` | every instruction, one file per encoding: SOP2, SOPK, SOP1, SOPC, SOPP, SMEM, VOP2, VOP1, VOPC, VOP3P, VOP3A/VOP3B, LDS, MUBUF, MTBUF, FLAT/scratch/global, limitations |
| `cdna4/ch13_microcode_formats.md` | the bit layout of every encoding |

## Conventions in the text

- `<!-- page N -->` marks the PDF page the following text comes from; cite it once you have checked it (below).
- Every instruction except the two `V_MFMA_SCALE_*` is a heading `#### NAME` in upper case, followed by
  `Opcode: N`, its description and usually its pseudocode. From the skill folder, `rg -n '^#### V_MFMA' hardware/gfx950/cdna4/`
  lists the MFMA instructions.
- The text is a conversion of the PDF. A table that crosses pages sits under its first page's marker, some
  converted tables are misaligned, and six figures carry text that exists only in the image (`lookup.md` §6).
  Check page numbers, table cells and signs that matter against the PDF before citing them.

## Related
`../../SKILL.md` (this skill's map) · `../../gemm/INDEX.md` (dense GEMM package) ·
`../../gemm/mainloop.md` · `../../gemm/datapath.md` · `../../gemm/correctness.md`
