---
name: kernel-automation
description: Orchestrate the four M1 AKLR kernel automation skills for one Cursor or manual round; also route initial setup and configuration changes.
---

# AKLR kernel automation — entrypoint

Use this single entrypoint for the configured Cursor Automation or a manual invocation. Read
the locally provisioned, Git-ignored `.cursor/kernel-optimizer/configuration.yaml` from the configured repository. For installation or a
configuration change, follow [setup](references/setup.md) first; setup does not start a paid campaign.
Never assume this conversation, Memories or shell variables contain the latest task facts.

## Four responsibilities

Read each skill when its responsibility is reached. These are steps in the same invocation, not four
Automations or instructions to spawn agents. Pass the current revision and explicit results between steps.

| M1 responsibility | Skill | Input → output |
|---|---|---|
| Monitor / Analyze | [kernel-automation-monitor](../kernel-automation-monitor/SKILL.md) | Config + saved state + trigger → timestamped repository/resource observations, proposed state changes and blockers. |
| Plan | [kernel-automation-plan](../kernel-automation-plan/SKILL.md) | Observations + config + ledger → one bounded action, target identity, prerequisites and budget commitment, or idle/blocked. |
| Execute | [kernel-automation-execute](../kernel-automation-execute/SKILL.md) | Admitted action + persisted intent → confirmed result, external handle, evidence or unknown. |
| Knowledge | [kernel-automation-knowledge](../kernel-automation-knowledge/SKILL.md) | Observations, decisions and execution results → versioned state, evidence index and handoff report. |

## One round

1. Use Knowledge to acquire the round lock and load state. Missing/corrupt state blocks mutation;
   only first-time setup may initialize it. If busy, report and exit without taking over the lock.
2. Use Monitor / Analyze to reconcile existing work before considering admission. Use Knowledge to save
   confirmed observations; in `observe`, retain proposed changes only in the report.
3. Use Plan to choose at most one bounded action. Stop/deadline and cleanup take priority. An existing
   remote action still running needs a later observation, not a second launch.
4. In `run` / `drain`, for an eligible external action, use Knowledge to save any budget commitment and operation intent **before**
   Execute makes the external call. For `reconcile`, route back to Monitor; for `wait` / `idle` / `blocked`,
   proceed to handoff. Execute rechecks action prerequisites and hands each result back to
   Knowledge immediately. A save failure prevents the side effect; a lost result requires reconciliation.
5. Use Knowledge to save the handoff, report, and release the round lock in a final cleanup step,
   including on failure. Releasing the round lock never frees campaign occupancy or stops remote jobs.

Knowledge supports the entire round; it is not only a final summary step. Pass config, lock token,
revision, campaign/operation identity and evidence references within this invocation; keep tokens out of
logs. Calling a responsibility skill directly does not bypass this entrypoint's mode, lock or admission rules.

## Shared operating boundaries

Validate repository, schema version, strategy types, paths and the single-campaign limit. For `run` / `drain`,
check the configured readiness record and budget authorization before paid or publishing actions. Missing
prerequisites block those actions, while read-only inspection and stopping/cleaning recorded resources remain
possible. Preserve configured model and reasoning effort through optimization and finalization; do not
silently substitute settings or ask again for authorization already recorded.

- `observe`: query and report only; no GPU/model/PR mutation or campaign-ledger writes. Acquiring/releasing
  the local round lock and appending the observation report are allowed.
- `run`: reconcile first, then admit or advance at most one GPU campaign within configured limits.
- `drain`: reconcile, finish or clean existing work; never admit a new campaign.

This pilot uses existing CLIs. The state helper serializes cooperating invocations and saves snapshots;
it does not enforce budgets, authorization or trustworthy validation. Use the configured dedicated execution
environment and verified external hard limits. The trusted MCP service in the M1 design is not implemented.

A running long action returns its handle and next observation time. The configured Automation schedule
provides the next wakeup; do not create per-round `/loop` timers. A manual round without a saved schedule
must report that no future wakeup is guaranteed. Respect host-provided PR watch/handoff rules when present.
