---
title: <one line: the technique and what it buys>
kind: technique
lever: <snake_case id, unique in this skill>
ops: [gemm]
dtypes: [bf16]
regimes: [<M range or regime: decode, small-M, large-M, split-k ...>]
gens: [gfx950]
flydsl: 0.3.2
basis: <source-semantics | measured N=<k> | single-observation>
sources:
  - <file the technique is read from, repository-relative or FlyDSL path>
---

# <Technique>

## What it does

<!-- The mechanism in two to five sentences: which cost it removes and why that cost exists on gfx950. -->

## How to write it

<!-- The FlyDSL v0.3.2 code that does it: a short snippet, or file:line of a kernel that does it. Long code goes to
     reference/ and is linked from here. -->

## When it applies

<!-- Preconditions: M range, dtype, tile, LDS and VGPR budget, the determinism contract (bitwise or bounded). -->

## Counter-signals

<!-- When not to use it, and the signal that tells you so. A failed attempt elsewhere belongs here, stated as the
     condition under which it fails, not as a story. -->

## How to confirm it took effect

<!-- What final_isa.s, the kernel descriptor (.amdhsa_*), a counter or a trace shows when the technique is in the
     binary. Wall time alone does not confirm it. -->

## Cost and risk

<!-- Registers, LDS, compile time, code size, correctness risk (for example: breaks bitwise reproducibility). -->

## Evidence

<!-- One line per measured claim, numbered, in this form:
- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=3, graph+cold-L2] <claim with the number> (<workloads>; <commit, PR, CI run or local:<absolute path>>)
     N=? when the count is unknown. A claim from source semantics cites file:line instead. Body text marks a measured number *(field, E1)*. -->
