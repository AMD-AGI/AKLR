---
name: aklr-port-generated-kernels
description: Port agent-generated GPU kernels into AKLR with their provenance, dispatch, fallbacks, and review evidence. Use for kernels produced in generator repositories.
---

# Port Generated Kernels

Integrate generated kernels as maintainable AKLR implementations, not as an
isolated collection of benchmark artifacts.

## Required Context

Read the harness [rules](../../rules.md), [gates](../../gates.md), and current
[gate configuration](../../gate-config.md). Read
[references/workflow.md](references/workflow.md) before changing code,
[references/porting-notes.md](references/porting-notes.md) while porting, and
[references/new-op.md](references/new-op.md) when AKLR has no contract for the op.
`aklr/ops/README.md` is the authority on the schema format and the audit.

## Responsibilities

- Pin and record the exact source revision, generator context, and provenance.
- Map the implementation to a AKLR contract and schema name, or add the op with
  its `<contract>_baseline`.
- Deliver the kernel as one schema JSON whose `kernel.sources` are
  self-contained and pass the audit, with `kernel.author` naming the generator.
- Replace numeric or opaque artifact names with meaningful operator, variant,
  dtype, shape, or architecture names without obscuring source traceability.
- Reach the kernel only through the contract's `@api` function; dispatch
  serves a Point once its measured speedup clears `AKLR_MIN_SPEEDUP`. Avoid a
  parallel public API for the generated kernel.
- Preserve unsupported cases by falling back safely rather than returning a
  silently incorrect result.
- Produce correctness, coverage, and fair incumbent-comparison evidence per
  Point, with `python -m aklr validate --base origin/main`.

## Decision Boundary

Do not redesign public interfaces, publish a branch, or open a pull request
without authority from the invoking user or workflow. Do not describe a kernel
as integrated when it is only copied into the repository or reachable solely
through a private benchmark.

## Completion

Return the changed paths, supported scope, fallback behavior, tests run,
benchmark comparison, known limitations, and source revision. Use
[references/report-template.md](references/report-template.md) for the handoff.
