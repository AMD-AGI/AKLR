---
title: "<area>: <what this card decides or prices>"
kind: topic
ops: [gemm]
dtypes: [<dtypes it covers>]
regimes: [<M ranges or regimes>]
gens: [gfx950]
flydsl: 0.3.2
sources:
  - <files the levers are read from, repository-relative or FlyDSL path>
---

# <Card>

<!-- One paragraph: what this card covers and what it leaves to other cards. -->

## Route here when

<!-- One bullet per symptom or decision that brings an agent here, naming the section that answers it. -->

## 1. <First lever or decision>

<!-- One lever or decision per numbered section: the mechanism, how to write it in FlyDSL v0.3.2, when it applies,
     when it does not, and what the ISA or a counter shows once it is in. Other pages cite sections as
     `<card>.md §<n>`: keep the numbers when you edit. A measured number is marked *(field, E1)*. -->

## Related

## Evidence

<!-- One line per measured claim, numbered, in this form:
- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=3, graph+cold-L2] <claim with the number> (<workloads>; <commit, PR, CI run or local:<absolute path>>)
     N=? when the count is unknown. A claim read from source: `- S1 <file>:<line> <what it shows>`. -->
