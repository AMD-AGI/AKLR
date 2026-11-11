---
name: setup-shepherd
description: Configure shepherd and prepare its scheduled task. Use when installing shepherd or changing its repository, environment, model, budget, publication or schedule settings.
disable-model-invocation: true
---

# Set up shepherd

The human enters setup by pointing the agent at the pack's `FOR_AGENTS.md`, which sends it here. This
file and the round skill are read by repository path; they are not slash skills. Setup starts no paid
campaign, and creates or updates no scheduled task until the user explicitly asks.

## 1. Check the pack

The pack lives at `harness/automations/shepherd/` in the AKLR checkout the task will run in. Confirm that
checkout contains `FOR_AGENTS.md`, this file, `skills/shepherd/` (skill, `references/state.md`,
`scripts/state.py`) and the templates. Confirm the shared skills the round uses exist under
`harness/skills/`: forge-loop-campaign, remote-gpu, aklr-read-validation, aklr-open-pr, report-progress,
aklr-kb and the configured backend's knowledge.

A scheduler that checks out afresh needs all of these committed on its revision. One that uses an
existing checkout needs its revision and local edits recorded. Do not commit unless the user asks.

## 2. Adapt the configuration

Copy `templates/configuration.example.yaml` to `.shepherd/configuration.yaml` at the repository root
(ignored by Git). If that file exists, keep its values and the user's edits; never edit the example.
Write paths with environment variables such as `$USER`: no host names, usernames or personal absolute
paths, even locally. Secrets stay in the execution environment's credential store.

## 3. Fill the required choices

Ask for or confirm:

- Repository and base branch (infer from the checkout).
- Mode: `run`, `observe` or `drain`.
- State directory, cluster, image digest and runtime revision.
- Strategies (`manual` paths, `kernel-less`), `batch_size`, `allowed_ops`, backend, `max_open_prs`,
  `cooldown_hours`.
- Campaign model and reasoning effort, loop hours, node hours, model USD, `max_auto_resumes`.
- Cumulative budget, its recorded authorization and the provider's hard limit.
- Publication: draft PRs, PR comments, `ci:gpu` label, branch prefix.

Resolve the model to an identifier KernelForge's provider supports and verify the reasoning effort; do
not guess a model, invent a flag or drop the effort. Node hours cover setup, finalize, stress and
validation as well as the loop. The model budget covers every inner agent call; outer automation and
GPU CI need their own authorization. Never estimate hard caps from logs. Do not guess any paid-run
choice: a missing one blocks the actions that need it, not read-only inspection.

## 4. Check the execution environment

Run these checks where every scheduled round will run, not only in this session:

- It reads this pack and `.shepherd/configuration.yaml` by repository path.
- It can read main, PRs, CI and reviews of the repository; with publication on, it can push a branch
  and open a draft PR. The `ci:gpu` label is a separate authority.
- It runs on one host, as one identity, and reaches the same state directory every time: local Linux
  disk with working `flock` and atomic rename. Not temporary storage, a fresh agent checkout, a copy,
  or an unverified network filesystem.
- `remote-gpu` reads the configured cluster; `forge_task` and KernelForge help commands resolve in
  the pinned runtime. No node is needed for this.
- A separate trusted validation checkout exists: the optimization worker cannot edit its validator,
  overwrite final evidence or reach GitHub and state credentials. Image and tool versions are pinned.

Record the passing checks and the budget authorization in `readiness.md` in the state directory. If a
check has not run, leave paid mode off.

## 5. Initialize state

Read [state.md](../shepherd/references/state.md). Initialize a new state directory once with the helper.
An existing directory, or one missing after use, needs investigation, not a new `init`.

## 6. Prepare the scheduled task

Ask whether this is a first-time creation or an existing task.

**First time.** Use the owner's chosen scheduler, for example GitHub Actions on a self-hosted runner or
host cron calling the agent CLI. Give it `templates/automation-prompt.md` as the prompt; keep workflow
rules in the round skill, not in the prompt. Bind the task to the checkout and environment checked in
step 4, set the interval (15 minutes by default), and record the task identity, the outer model and its
reasoning effort. Repository events may trigger the same task; polling stays the fallback. Do not rely
on bot comments as triggers or invent a scheduler API.

**Existing task.** Update it in place; never create a second one. When moving to another scheduler,
disable the old trigger before enabling the new one. Keep the state directory and budgets.

If the scheduler cannot be configured from here, hand over the prompt, configuration path, readiness
results and remaining settings, and say the task is not saved.

## 7. Verify before paid operation

1. Run one `observe` round: it reads GitHub and existing runs, lists candidates, writes a report and
   changes nothing.
2. Run the state helper's CPU tests (`python -m pytest harness/tests/shepherd`). On the
   configured host, show two concurrent rounds cannot both acquire the lock, and a second fresh round
   reads what the first saved.
3. Enable the task in `run` only within the owner's authorization.
4. Accept it after one real campaign reaches verified cleanup and a draft PR, and a later round
   observes that PR. CPU tests or a prompt preview do not count.

Report each state separately: prepared, saved, enabled, runtime-verified.
