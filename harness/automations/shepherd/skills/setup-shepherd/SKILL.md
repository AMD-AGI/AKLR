---
name: setup-shepherd
description: Configure shepherd and prepare its Cursor automation. Use when installing shepherd or changing its repository, environment, model, budget, publication or schedule settings.
disable-model-invocation: true
---

# Set up shepherd

shepherd is a dormant automation pack in the AKLR repository. This file and the round skill are not slash skills.

The human enters setup by pointing Cursor at `FOR_AGENTS.md`, which sends you here in the checkout that will run rounds.

shepherd needs one configuration file, one state directory and one live Cursor automation.

Do not create or update an automation until the user explicitly asks. Setup starts no paid campaign. Never put a secret in plugin files, prompts or committed configuration.

## 1. Check the checkout

Do this before asking for configuration and before invoking `/automate`.

The round checkout is the AKLR checkout on the round host, on the base branch. It already holds this pack. Never copy pack files into it.

Verify that it contains `FOR_AGENTS.md`, this file, the round skill with its `references/` and `scripts/`, and the templates. If the copy that sent you here is newer (`git log -1 -- harness/automations/shepherd`), stop and ask the user to update the checkout.

The automation needs no plugin; it only starts the wrapper. Do not edit `.cursor/settings.json`.

The round agent must load the shared skills from the checkout. An installed plugin is a cached revision that can lag the pack. For Claude CLI, put `--plugin-dir harness` in `execution.round_agent`.

Once the round agent is chosen (step 3), disable any user-scoped `aklr` plugin (Claude CLI: `--settings '{"enabledPlugins":{"aklr@amd-aklr":false}}'`) and verify in the checkout that it resolves:

- `forge-loop-campaign`
- `remote-gpu`
- `aklr-read-validation`
- `aklr-open-pr`
- `report-progress`
- `aklr-kb`

If any does not resolve, stop and explain the failure. Do not add `harness/automations/shepherd/skills/` to a plugin manifest.

Tell the user that the pack must be committed on the base branch before the automation is enabled. Do not commit it unless the user asks.

## 2. Adapt the configuration

Copy `../../templates/configuration.example.yaml` to `.shepherd/configuration.yaml` in the repository root. Git ignores it. If it exists, keep its values. Do not edit the example.

Write paths with environment variables such as `$USER`. No host names, usernames or personal absolute paths, even locally. Secrets stay in the credential store.

The configuration is not committed. The live prompt names its path and never paraphrases its values, so the automation must run in the checkout that holds it (step 4).

## 3. Fill the required choices

Ask for or confirm, in one pass:

- Repository and base branch. Infer them from the checkout.
- Mode: `run`, `observe` or `drain`.
- State directory, cluster, image digest and runtime revision. Offer found values: the image is `AKLR_CI_IMAGE` in `.github/workflows/ci.yaml`, resolved to its digest. The runtime is the Hyperloom `AKLR` branch commit (KernelForge) plus the flydsl version `harness/scripts/forge_task/README.md` pins.
- `execution.round_agent`: the non-interactive agent command (CLI, model, reasoning effort, permission mode). It must load the shared skills from the checkout (step 1).
- Strategies (`manual` paths, `kernel-less`), `batch_size`, `allowed_ops`, `backends`, `max_parallel_campaigns`, `max_open_prs`, `execution.pool_hours`, `cooldown_hours`.
- Campaign model, reasoning effort, loop hours, node hours, model USD, `max_auto_resumes`, and `model_usd_per_hour`: the highest hourly spend of the model with KernelForge, from provider billing.
- Cumulative budget, its recorded authorization, and the provider's hard limit: the spend cap on the API key or gateway the campaign calls (`claude auth status`, `ANTHROPIC_BASE_URL`). If none exists, the owner must set one.
- Publication: draft PRs, PR comments, `ci:gpu` label, branch prefix.
- Model slug and schedule of the Cursor automation, 15 minutes by default.

Use only model slugs the user's Cursor model picker shows. Do not guess a slug or carry over a private default.

Resolve the campaign model to an identifier KernelForge's provider supports. Check the reasoning effort against the pinned KernelForge source (`--agent-reasoning-effort`). Do not guess a model, invent a flag or drop the effort.

Node hours cover setup, finalize, stress and validation as well as the loop. The model budget covers every inner agent call. The outer automation and GPU CI need their own authorization. Never estimate hard caps from logs.

Do not guess any paid-run choice. A missing one blocks the actions that need it, not read-only inspection. With `mode: run`, list each value still `null` as `prepared (blocked: ...)`, never as ready.

## 4. Check the host

Run these checks where every scheduled round will run, not only in this session.

The automation needs:

- A self-hosted Cursor worker, not a cloud agent. A cloud agent has no persistent state directory, no `/proc` of earlier rounds and no cluster access.
- A My Machines or repo-backed pool worker in the checkout that holds `.shepherd/configuration.yaml`. An any-repo pool worker with `--clone-git-repos` clones a fresh checkout without it.
- A run limit of at least the round's 2-hour `timeout`. Cursor does not document one, so measure it on this worker with a harmless long command. A shorter limit cuts every long round; the next trigger reconciles it.
- No GitHub or other integration. The automation only runs the wrapper.

The round agent needs:

- The host's own GitHub credentials (`gh auth status`): read access to main, PRs, CI and reviews; with publication on, branch push and draft PR creation. The `ci:gpu` label is a separate authority.
- `remote-gpu` reading the configured cluster, and `forge_task` and KernelForge help commands resolving in the pinned runtime. No node is needed for this.

The host needs:

- One host and one identity for the worker, wrapper, agent, ledger and GitHub credentials.
- A state directory on local Linux disk with working `flock` and atomic rename. Not temporary storage, a fresh checkout, a copy, or an unverified network filesystem.
- One PID namespace for every round, with `/proc` readable: the host itself or one long-lived container. The wrapper reads `/proc` to see whether a round is still running; a round in another namespace looks undecidable and blocks every later one.

Probe worker isolation on an allocated node. Start a container with the worker's mounts (`remote-gpu exec --image <image> --mount <probe>/worker --mount <probe>/secrets:ro`). Record its `docker inspect` mounts, that `gh auth status` fails, that no token-like variable is set, and that the validate directory and the cluster `$HOME` are not visible. The cluster `$HOME` holds no GitHub credentials. Image and tool versions are pinned.

Do not use undocumented integration endpoints.

## 5. Initialize state

Read [state.md](../optimize-and-deliver-kernels/references/state.md). Initialize a new state directory once with the helper; it creates missing private parents.

Setup cannot `save`; only a running round can. The configuration keeps the repository, authorization and caps. An existing directory, or one missing after use, needs investigation, not a new `init`.

Record the passing checks of step 4 and the budget authorization in `readiness.md` in the state directory. If a check has not run, leave paid mode off.

## 6. Prepare the live automation

Ask whether this is first-time creation or an existing automation.

Read `../../FOR_AGENTS.md` as the primary intent source for either path.

### First-time creation

1. Read `../../templates/automation-prompt.md` as secondary source material.
2. Turn `FOR_AGENTS.md`, the finished configuration and the template into a complete natural-language request.
3. Point the live prompt at the committed round skill by its repository-relative path. Do not copy its contents or use a plugin cache path.
4. Read and follow the built-in `automate` skill.
5. Let `automate` discover the repository and confirm that the pack is committed there.
6. Tell `automate` that `.shepherd/configuration.yaml` is uncommitted on purpose: the self-hosted worker uses the checkout that holds it.
7. Let `automate` show its draft table, obtain approval, ask readiness, and open the Automations editor.

Give `automate` this intent, filled from configuration:

- Name `shepherd`.
- Read and follow `harness/automations/shepherd/skills/optimize-and-deliver-kernels/SKILL.md` with `.shepherd/configuration.yaml` for every run.
- Trigger only on the configured schedule. No repository-event triggers.
- Run on the self-hosted worker checked in step 4, in the configured repository and base branch.
- Add no integrations.
- Run one round through the state helper's `round` wrapper, relay its result and report, then end.
- Never edit, rebase, approve or merge an existing pull request, manage schedules, or reinitialize state.

Do not duplicate `automate`'s own discovery, review, approval, readiness or handoff work.

### Existing automation

`automate` is creation-only. Do not use it to find, inspect or update an automation.

Finish configuration, host and state validation. Then ask the user to update in the Automations editor:

- Name and description
- Instruction to read `harness/automations/shepherd/skills/optimize-and-deliver-kernels/SKILL.md` with `.shepherd/configuration.yaml`
- Schedule trigger and interval; remove any repository-event trigger
- Self-hosted worker, repository and base branch
- No integrations
- Paraphrased one-round, wrapper, report and no-edit instructions

Do not create a replacement or duplicate. When moving from another scheduler, disable the old trigger first, and keep the state directory and budgets.

### Creation boundary

Never call an automation backend service or tool directly. Never use a browser URL that carries draft fields. Never build or open a Cursor deep link. The only finish path for a new automation is `automate`'s Automations editor handoff.

Record the automation's name, model and schedule in `readiness.md`. If the automation cannot be configured from here, hand over the prompt intent, configuration path, readiness results and remaining settings, and say it is not saved.

Enable the automation only with `mode: observe`. Keep it there until step 7 passes. Observe rounds change nothing, so the trigger tests can run on the live automation.

## 7. Test before paid operation

Confirm that the pack is committed on the automation's branch and that the live prompt points at the committed round skill. If not, stop and tell the user the automation cannot be enabled yet.

Verify with `mode: observe`:

1. One round through the wrapper (`round --now -- <round_agent> <prompt>`) reads GitHub and existing runs, lists candidates, writes a report and changes nothing.
2. `python -m pytest harness/tests/shepherd` passes with `TMPDIR` on a directory only you and root own, such as `/tmp`. The helper refuses a temp root under another user's directory.
3. Through the automation's own trigger: two concurrent triggers start one round; a second round reads what the first saved; after `kill -9` of a running agent, the next trigger takes over with `AKLR_RECOVERED_FROM` set.

Then switch to `mode: run` on probation, within the owner's authorization:

- `budget.total_*`: one campaign's and one pool's bounds, twice that if the owner allows a retry.
- `max_parallel_campaigns: 1`, `max_open_prs: 1`, `allow_draft_pr: true`.
- Reports say `enabled (probation)`.

Accept it after one real campaign reaches confirmed cleanup and a draft PR, and a later automation-started round observes that PR. CPU tests or a prompt preview do not count. If the probation budget runs out first, report `blocked`. After acceptance the owner records a new authorization and raises the budget.

Report each state separately: prepared, saved, enabled (observe), runtime-verified, enabled (probation).
