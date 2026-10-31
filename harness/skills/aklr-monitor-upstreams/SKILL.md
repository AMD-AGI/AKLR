---
name: aklr-monitor-upstreams
description: Monitor configured upstream repositories for new kernels and optimizations, deduplicate candidates, and route them to the appropriate AKLR porting workflow. Use for scheduled or on-demand upstream change discovery.
---

# Monitor Upstream Changes

Turn upstream activity into a small, evidence-based candidate queue rather than
automatically copying every kernel-related commit.

## Required Context

Read the harness [rules](../../rules.md), [gates](../../gates.md), current
[gate configuration](../../gate-config.md), and
[references/workflow.md](references/workflow.md).

## Responsibilities

- Read the configured upstream repositories and last successfully examined
  revisions; do not invent monitoring state when configuration is absent.
- Identify new kernels, dispatch changes, optimizations, and relevant interface
  changes since the last checkpoint.
- Record source revision, affected operator, claimed gain, supported GPU scope,
  provenance, and available tests or benchmark evidence.
- Check AKLR and the candidate queue for equivalent or duplicate work.
- Route generated-kernel candidates to `aklr-port-generated-kernels` and
  human-written AITER candidates to `aklr-sync-aiter`.
- Rank candidates by relevance and evidence quality without treating upstream
  benchmark claims as verified AKLR results.

## Monitoring Boundary

Monitoring is read-only. A scheduled run may prepare a candidate report, but it
does not gain authority to modify repositories, create pull requests, or update
external state unless the automation explicitly grants that authority.

## Completion

Return the scanned revision range, candidate list, duplicates, routing decision,
and any missing configuration. Use
[references/report-template.md](references/report-template.md).
