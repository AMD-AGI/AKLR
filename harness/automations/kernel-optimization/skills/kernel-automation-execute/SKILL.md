---
name: kernel-automation-execute
description: Execute one admitted AKLR campaign action, validate and deliver a draft PR, or stop and clean owned resources using existing operational skills.
---

# Execute — campaign execution and delivery

Use an action selected by Plan through [the entrypoint](../kernel-automation/SKILL.md), with the current
config, campaign identity, revision and persisted operation intent. A direct invocation must enter the same
lock/observation/admission flow; this skill is not independent authority to spend or publish. Recheck the
mode, authorization, deadline and expected input/base/head before an external action. If they changed, return
to Monitor/Plan. Respect `observe` and `drain`; do not admit a campaign while advancing an existing one.

## Operational skills

Read the file when reaching its phase and follow its linked command reference; do not invent a second CLI.

| Action | Repository skill |
|---|---|
| Prepare, optimize, ablate, finalize, stress, publish candidate, stop | `harness/skills/forge-loop-campaign/SKILL.md` |
| Allocate, inspect, clean up exact owned resources | `harness/skills/remote-gpu/SKILL.md` |
| Interpret per-Point validation and fallback | `harness/skills/aklr-read-validation/SKILL.md` |
| Create draft PR with real head-bound evidence | `harness/skills/aklr-open-pr/SKILL.md` |

For kernel reasoning read `harness/skills/aklr-kb/SKILL.md` and the chosen backend's skill. Use the configured
model, reasoning effort and previously authorized bounds, including finalization. Entrypoint scheduling
rules replace per-round timers in operational skills. Report through Knowledge after returning results.

## Advance one bounded action

Use the campaign skill and its README for the actual commands. Drive these phases from recorded evidence:

`admitted → preparing → optimizing → finalizing → validating → cleaning → ready_to_deliver → pr_open`

A phase transition needs the preceding operation's confirmed result. Before every external side effect,
use Knowledge to save an operation ID, intended action, deadline and unique task name; afterwards pass its
returned handle and result to Knowledge for immediate persistence.
A running action returns immediately with the next observation time. Long setup/finalization/validation runs
also need remote background handles, not just forge-loop. Do not keep one Cursor invocation alive for hours.

Use distinct AKLR workspaces per campaign and preserve the schema/config/base snapshots. Budget remaining
must include ablate, finalize, stress and final validation. Enforce the absolute campaign deadline across
resumes; never reset the clock or add node/model quota automatically. A crashed loop may resume only after
proving the previous model processes stopped, within remaining time/quota and `max_auto_resumes`. If it crashes
again before a successful round, leave it stopped and request human help, as the campaign skill requires.

Optimization success requires evidence, not a log phrase alone. Collect the final candidate and reviewed
knowledge patch. In the trusted checkout, allow only the new implementation, hash-matched kernel-less schema
replacement and reviewed knowledge changes. Validate the exact intended commit against the fixed base with
`aklr export check`, targeted `aklr validate --base ... --stages audit,bench ...`, and the campaign's stress gate.
Use the validation skill to inspect every target Point; record measurement coverage, correctness, fallback,
baseline/incumbent comparisons, and any missing checks. No green summary can replace these records.

Preserve reports with candidate hash, base/head, validator revision, image digest and measured Points. A failed
check means no PR delivery; clean up and report failure or blocked. Do not start an automatic repair campaign.
Review the skill patch as the campaign skill requires; kernel and method changes follow its separate-PR rule.
Method changes are handed to the owner rather than silently broadening this campaign's publication scope.

After collecting required artifacts, stop remaining processes and release the exact allocation. Save verified
cleanup evidence, then free the GPU occupancy. Stop requests/deadlines take priority over other phases. A
blocked or failed run still needs cleanup; unknown cleanup keeps the slot occupied and the automation stopped.

## Deliver once and apply authorized PR context changes

Only deliver when `allow_draft_pr` is authorized, all required evidence is current, the intended commit is
frozen, and cleanup is confirmed. Otherwise retain the candidate and report the missing authority or evidence.
Re-fetch main before pushing; if it moved, mark evidence stale and hand the candidate to the owner. Follow
aklr-open-pr using the configured branch prefix, `--draft`, the approved diff and real head-bound reports.

Use Knowledge to persist push and PR-create intents separately. Use a stable campaign marker in the PR body and search for an
existing PR by exact repository/head branch before creation, including closed PRs. Response loss requires
reconciliation, never a second PR. Recheck base/head after creation and mark any raced evidence stale. Add
`ci:gpu` only when `allow_ci_gpu_label` is authorized; otherwise retain draft and tell the owner CI is pending.
Do not repeatedly toggle labels. Comments require `allow_pr_comments`; deduplicate by saved content hash.
When Plan selects an authorized PR context update, apply it only to the recorded PR: for stale evidence,
keep/return it to draft where authorized and report the required human revalidation. Save the update intent
and result through Knowledge. A context update grants no permission to edit the code or push its branch.

Monitor inspects CI/reviews and other merges on subsequent rounds. No code edits, automatic branch updates, approval or
merge of an existing PR. Knowledge stores the confirmed merge commit/input identity or close reason/cooldown returned by Monitor.
If tools cannot inspect the current state, report unknown rather than claim completion.

Return confirmed results and timestamps, external handles, candidate/evidence versions, cleanup observations
and the next action. A timeout or missing reply returns `unknown`, never permission to repeat the operation.
Long actions return without waiting for the whole campaign; only verified results permit a phase transition.
