---
name: kernel-automation-plan
description: Choose a single bounded AKLR automation action or candidate from observed facts, enforcing configured eligibility and admission rules before execution.
---

# Plan — select one bounded action

Use current Monitor observations, config and the ledger from [the entrypoint](../kernel-automation/SKILL.md).
This skill proposes an action; it does not allocate, launch, publish or write state itself.

## Prioritize existing work

Choose in order: an authorized stop or expired deadline; cleanup; reconciliation of an unresolved operation;
one eligible next phase or delivery of an existing campaign; authorized PR context changes; new admission.
An unknown resource/spending state blocks new admission globally. It does not prevent inspection or stopping
an exactly identified owned resource. If an operation is still running, return `wait` with its existing handle
and next observation time. Human-only branch updates and revalidation requests are reported, not scheduled
as an automatic repair campaign. `observe` returns proposals only; `drain` cannot admit new work.

## Choose one new campaign, only when eligible

Do this only in `run`, with no active GPU campaign, unresolved operation or cleanup, enough uncommitted
budget, and fewer than `max_open_prs` outstanding automation PRs plus candidates awaiting delivery. Evidence requiring human attention blocks
that task; any unknown resource/spending state blocks new admission globally.

At the fixed main SHA, apply configured strategies in order: explicit schema paths for `manual`; JSON files
under `aklr/ops/*/impls/*/*/` with empty `kernel.sources` for `kernel-less`. Parse JSON, do not infer emptiness
from text search. Deduplicate by repository/path/input hash, retain source strategy, sort within a discovery
strategy by relative path, and inspect up to `batch_size` eligible candidates. Missing manual inputs are
reported, never silently substituted. A main SHA change alone does not make a delivered input new.

Reject/skip with a reason: already served or delivered, competing PR, cooldown, unsupported op/backend,
missing baseline/environment, or schema requiring an unqualified transformation. First pilot supports only
the configured GEMM backend and schemas whose workloads fit one tuning regime. Read
`harness/skills/forge-loop-campaign/SKILL.md` for its small/large-M analysis; use that analysis for selection
without launching the campaign. Mixed-regime schemas go to the owner for splitting, not silent contract edits.
Do not assume support merely because the selector found a schema. If none qualify, report idle.

Immediately before admission, recheck main and the chosen input. Prepare a campaign ID, isolated
run/checkout paths, unique allocation label, branch name and fixed config snapshot. Compute the full campaign
model/node upper bounds against uncommitted cumulative budget, with no refunds. Verify the scheduler and
model hard limits cover the whole campaign. `--max-hours` bounds loop time, not all model cost or finalization.
Hand these to Knowledge to persist phase `admitted`, the budget commitment and allocation intent in one
snapshot **before** Execute calls remote-gpu. Do not treat a proposed plan as a saved admission.

## Result

Return one of `admit`, `advance`, `stop`, `clean`, `deliver`, `update_pr_context`, `reconcile`, `wait`, `idle`
or `blocked`, with campaign/input identity, expected main/head and state revision, reason, prerequisites,
operation kind, deadline, evidence references and any budget commitment. Candidate rejections retain their
reasons and discovery strategy. Knowledge saves the decision and Execute checks it immediately before acting.
An observed input or version change invalidates the plan; return to observation instead of using stale approval.
