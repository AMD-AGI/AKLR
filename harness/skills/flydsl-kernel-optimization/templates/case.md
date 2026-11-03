---
title: <op and shape family> — <what made it fast>
kind: case
ops: [<op>]
dtypes: [<dtype>]
regimes: [<regimes it covers>]
contract: <bitwise | bounded>
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=<k>
measured:
  - {metric: <speedup over the faster of AKLR's served kernel and aiter, or time>, value: <number>, gpu: MI355X, workload: <schema and Points>, run_ref: <commit, run id or log path>, date: <YYYY-MM-DD>}
sources:
  - <the delivered kernel: aklr/ops/<op>/impls/... >
---

# <Case>

## Applies when

<!-- The shapes, dtype, regime and contract this solution is for. -->

## Counter-signals

<!-- Where the same approach did not pay or broke a gate, as conditions. -->

## Mechanism

<!-- Why it is fast: the techniques used (link their cards) and how they combine. -->

## Result

<!-- Per workload or regime, against the faster of AKLR's served kernel and aiter; matches `measured` above. -->

## Reproduce

<!-- The exact commands, for example `python -m aklr validate --base origin/main` on the schema. -->

## Rollback

<!-- How to stop serving it and what regression would require that. -->

## Evidence

<!-- Needed only when the page has measured numbers: one line each, in the format of templates/technique.md. -->
