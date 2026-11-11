---
name: review-kernel-pr
description: Review AKLR kernel pull requests for integration quality, correctness, edge cases, sanitizers, performance, benchmark fairness, and reward hacking. Use when deciding whether a generated or ported kernel PR is ready to merge.
---

# Review a Kernel Pull Request

Review the kernel as part of AKLR's public execution path, not only as a fast
standalone implementation.

## Required Context

Read the harness [rules](../../rules.md), [gates](../../gates.md), current
[gate configuration](../../gate-config.md), and
[references/workflow.md](references/workflow.md). Check the change against
[bad-cases](../bad-cases/SKILL.md).

## Responsibilities

- Inspect provenance, naming, placement, public reachability, dispatch, and
  fallback behavior.
- Make the code preview understandable: separate host integration from kernel
  bodies and flag opaque numeric names, generated clutter, or hidden behavior.
- Run or verify unit, reference-correctness, edge-case, sanitizer, and relevant
  end-to-end tests.
- Compare performance with the actual AKLR incumbent in the same environment.
- Audit warm-up, synchronization, input selection, timeouts, cache state,
  fallback paths, failures, and aggregation for unfair benchmarking.
- Look for reward hacking, including benchmark-only reachability, shape
  overfitting presented as general support, skipped failures, and comparisons
  against a deliberately weak baseline.
- Distinguish verified facts from missing evidence and reviewer inference.

## Review Boundary

Review is read-only unless the user also asks for fixes. Do not approve based on
the PR description alone, and do not claim that an unavailable GPU or
downstream test passed.

## Completion

Return a verdict of `PASS`, `CHANGES REQUIRED`, or `BLOCKED`, with findings
ordered by severity and a concise evidence table. Use
[references/report-template.md](references/report-template.md).
