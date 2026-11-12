---
name: qualify-downstream
description: Qualify an exact AKLR build with supported SGLang, vLLM, and ATOM revisions for API compatibility, correctness, and performance. Use before release or when integration changes may affect downstream engines.
---

# Qualify Downstream Engines

Verify that AKLR's interfaces (`aklr.ops` and the SGLang plugin) remain usable
by real engine paths and that kernel improvements survive end-to-end integration.

## Required Context

Read the harness [rules](../../rules.md), [gates](../../gates.md), current
[gate configuration](../../gate-config.md), and
[references/workflow.md](references/workflow.md).

## Responsibilities

- Pin the AKLR runtime image and each downstream repository revision.
- Confirm installation and API compatibility before running workloads.
- Exercise representative SGLang, vLLM, and ATOM paths on declared hardware.
- Compare end-to-end correctness with a trusted baseline and report tolerances.
- Measure relevant latency or throughput using the same workload and
  environment for baseline and candidate.
- Confirm that the intended AKLR kernel is actually selected; distinguish a
  passing fallback from successful kernel integration.
- Report each downstream independently as pass, fail, skipped, or blocked.

## Qualification Boundary

Do not patch a downstream project unless the user asks for that change. Never
generalize an untested engine, GPU, model, or workload from another passing
result.

## Completion

Return a support matrix with exact revisions, environment, workloads,
correctness, performance, selected kernel path, failures, and logs. Use
[references/report-template.md](references/report-template.md).
