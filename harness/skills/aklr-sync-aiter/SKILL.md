---
name: aklr-sync-aiter
description: Evaluate new human-written AITER commits and selectively port only compatible changes with a reproducible performance gain into AKLR. Use for ongoing AITER-to-AKLR synchronization, not for generated-kernel intake.
---

# Selectively Sync AITER

Keep AKLR compatible with useful AITER evolution without automatically copying
every upstream commit or replacing a better AKLR implementation.

## Required Context

Read the harness [rules](../../rules.md), [gates](../../gates.md), current
[gate configuration](../../gate-config.md), and
[references/workflow.md](references/workflow.md).

## Responsibilities

- Identify the exact AITER commit and the operator, interface, architecture,
  and shapes it affects.
- Determine whether AKLR already contains an equivalent or faster change.
- Reproduce the AITER improvement against the current AKLR incumbent in the
  same pinned environment.
- Port the smallest compatible change only when the gain is measurable on
  relevant supported workloads.
- Preserve AKLR's generated kernels and choose dispatch per supported regime
  rather than overwriting a stronger implementation.
- Run equivalent correctness and performance verification after the port.
- Record evaluated candidates that are rejected, including why no port was
  created.

## Acceptance Rule

No relevant, reproducible performance gain means no sync through this skill.
Correctness or security fixes without a performance gain should be reported for
a separate human decision rather than relabeled as performance syncs.

## Completion

Return a `PORT`, `REJECT`, or `ESCALATE` decision with source revision,
benchmark evidence, compatibility impact, changed paths, and limitations. Use
[references/report-template.md](references/report-template.md).
