---
name: kernel-automation-knowledge
description: Load and persist AKLR automation facts, operation intents, budget commitments and version-bound evidence; produce the round handoff without treating Memories as execution facts.
---

# Knowledge — state, evidence and handoff

Use this skill throughout [the entrypoint](../kernel-automation/SKILL.md)'s round, including before and after
external calls. It owns durable recording, not candidate selection, GPU execution or kernel acceptance.
Read [state.md](references/state.md) for the state contract, helper CLI and manual crash recovery procedure.

## Load and own one round

Run the helper on the configured control host against the same persistent state directory. Acquire the
round lock before reading the snapshot used for decisions. If busy, exit; never reclaim a lock by age.
Missing/corrupt state blocks mutation; only initial setup can initialize a genuinely new ledger. Carry the
opaque token and current revision through the invocation and use the returned revision after each save.
Release the lock in a final cleanup step, even if blocked; this never frees remote campaign occupancy.

## Record decisions and results

- Save confirmed Monitor observations with source, identity and time. Preserve unknowns and the last
  successful observation; reconcile current resource fields with cleanup evidence so stale `RUNNING`
  snapshots do not contradict confirmed cancellation. Never infer cancellation from a closed PR.
- Before Execute, save Plan's admitted campaign, full budget commitment and initial operation intent in
  one snapshot. For subsequent side effects, save operation identity, unique external name and deadline
  before the call. Stop if persistence fails; never launch based only on a plan or in-memory intent.
- After each external result, save handles, confirmed outcome or unknown and its observation time.
  Keep intents whose response was lost for reconciliation; never overwrite them with a new attempt.
- Preserve cumulative commitments without refunds and retain completed campaign identities. Mark merged
  input identities delivered; confirmed failed/no-gain/closed tasks receive configured cooldown. Record
  the cause and timestamp, not just a terminal phase.
- Record cleanup only from observations of model processes, containers and the exact allocation. Unknown
  cleanup keeps occupancy. A stopped/closed campaign can still own resources until cleanup is confirmed.

The helper provides cooperating-writer exclusion, revision checks and atomic snapshots only. It does not
validate these domain facts, enforce a model hard cap, or protect the ledger from the optimization worker.
Do not describe a client-side ledger or observed under-budget spend as a verified hard spending limit.

## Evidence and reusable knowledge

Keep input/config snapshots, candidate digest, base/head, validator revision, image digest, per-Point verdicts,
logs and reports under the campaign's persistent artifact paths. Link evidence to its exact versions; a
main/head/candidate change invalidates matching evidence until revalidated. A summary never substitutes for
the underlying reports. Execute and the validation skill supply verdicts; Knowledge preserves them.

Memories may retain preferences, lessons and cross-run summaries, but cannot decide spending, cleanup or
acceptance. Preserve the reviewed knowledge patch returned by the campaign skill with provenance; method
changes follow that skill's separate-PR rule and are handed to the owner. Do not automatically promote raw
worker notes or generated files into shared skills.

## Report and finish

Save after each external result and at handoff, then release the lock. In `observe`, leave the campaign ledger
unchanged and put proposed updates only in the report; lock operations and report append remain allowed.
Use `harness/skills/report-progress/SKILL.md` and append to the configured state directory's `reports.md`.
Include changes, current campaign/phase, versioned measurements, unverified checks, job/PR handles,
committed budget, blockers and next action. Follow the entrypoint's scheduling rules; do not create another
timer. Read-only report helpers receive no publishing/spending authority. No Slack, email or owner ping is
implied. Use the short form when nothing changed. Never report paid automation as enabled from a manual run.
