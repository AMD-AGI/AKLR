---
name: setup-shepherd
description: Configure shepherd and prepare its Cursor automation. Use when installing shepherd or changing its repository, environment, model, budget, publication or schedule settings.
disable-model-invocation: true
---

# Set up shepherd

shepherd ships as a dormant automation pack inside amd-aklr. The plugin manifest exposes only amd-aklr's normal skill root; this file and the round skill are not slash skills.

The human enters setup by pointing Cursor at the pack's `FOR_AGENTS.md`. The bootstrap flow copies the whole pack into the target repository, then reads this file directly at `harness/automations/shepherd/skills/setup-shepherd/SKILL.md`.

shepherd needs external configuration, a state directory and one live Cursor automation.

Do not create or update an automation until the user explicitly asks. Setup starts no paid campaign. Never put a secret value in plugin files, prompts, or committed configuration.

## 1. Copy the pack and enable shared amd-aklr skills

Do this before asking for shepherd configuration and before invoking the built-in `/automate` skill.

Ask which repository will run the automation; it must be an AKLR checkout. The source pack is the directory containing `FOR_AGENTS.md`. The destination is `<target-repository>/harness/automations/shepherd/`.

Merge the entire source pack into the destination:

1. Create the destination when it is absent.
2. Copy every source file to the same relative path.
3. Preserve destination-only files. Never delete unrelated files during install or refresh.
4. Keep user-owned configuration outside the destination. Never overwrite it.
5. When an existing source-managed file differs, inspect the diff and merge without discarding local edits. If ownership is ambiguous, stop and ask before replacing it.
6. Verify that the destination contains `FOR_AGENTS.md`, this setup file, the round skill with its `references/` and `scripts/`, and the templates.

If this file is already being read from the target destination, treat the copy as complete and run the same verification before continuing.

Add the `amd-aklr` plugin to the target repository's `.cursor/settings.json`. If the file or `.cursor` directory does not exist, create it.

Merge this entry into the existing JSON or JSONC:

```json
{
	"plugins": {
		"amd-aklr": { "enabled": true }
	}
}
```

Preserve every unrelated top-level setting and every other plugin entry. If `plugins.amd-aklr` already exists, change only its `enabled` value. Preserve comments and valid JSONC syntax when the file uses JSONC. Validate the file after editing it.

Reload the target project or start a fresh agent rooted there. Verify that these shared amd-aklr skills resolve from project scope:

- `forge-loop-campaign`
- `remote-gpu`
- `aklr-read-validation`
- `aklr-open-pr`
- `report-progress`
- `aklr-kb`

Do not count a skill loaded from the current session or a user-scoped plugin. The check must show that a fresh agent in the target repository receives amd-aklr through project settings.

If project-scoped plugin installation is unavailable or any shared dependency does not resolve, stop and explain the failure.

The shepherd files are read directly from `harness/automations/shepherd/`. Do not add that directory to a plugin manifest or expect its `SKILL.md` files to appear in the slash-skill list.

Tell the user that `.cursor/settings.json` and `harness/automations/shepherd/` must be committed before the automation is enabled. Do not commit them unless the user asks.

Once this check passes, the live automation prompt may read the committed round skill by its stable repository-relative path. It must not embed a plugin cache path or copy the file contents.

## 2. Adapt the configuration

Open the copied example `../../templates/configuration.example.yaml`.

Create the user-owned copy outside `harness/automations/shepherd/`, at `.shepherd/configuration.yaml` in the repository root (ignored by Git). It is a configuration file, not a pack file. If it exists, keep its values and the user's edits.

Do not edit the copied example. Pack refreshes may update source-managed files after conflict review, but they must never touch the user-owned copy.

Write paths with environment variables such as `$USER`: no host names, usernames or personal absolute paths, even locally. Secrets stay in the execution environment's credential store.

The configuration is not committed, so the live prompt names its path and never paraphrases its values; the automation must run in the checkout that holds it (step 4). Use stable repository-relative paths for committed pack files. Never reference the plugin source directory or a plugin cache path from a live automation.

## 3. Fill the required choices

Ask for or confirm:

- Repository and base branch (infer from the checkout).
- Mode: `run`, `observe` or `drain`.
- State directory, cluster, image digest and runtime revision. Offer found values for confirmation: the
  image from AKLR CI (`AKLR_CI_IMAGE` in `.github/workflows/ci.yaml`) resolved to its registry digest; the
  runtime as the Hyperloom `AKLR` branch commit (KernelForge) plus the flydsl version
  `harness/scripts/forge_task/README.md` pins.
- `execution.round_agent`: the non-interactive agent command (CLI, outer model, reasoning effort,
  permission mode) the wrapper runs.
- Strategies (`manual` paths, `kernel-less`), `batch_size`, `allowed_ops`, `backends`,
  `max_parallel_campaigns`, `max_open_prs`, `execution.pool_hours`,
  `cooldown_hours`.
- Campaign model and reasoning effort, loop hours, node hours, model USD, `model_usd_per_hour` (the
  highest hourly spend of the model with KernelForge, from the provider's billing, not a guess),
  `max_auto_resumes`.
- Cumulative budget, its recorded authorization and the provider's hard limit: the spend cap on the API
  key or gateway the campaign's model calls use (`claude auth status`, `ANTHROPIC_BASE_URL`). If none
  exists, say the owner must set one; it cannot be inferred from here.
- Publication: draft PRs, PR comments, `ci:gpu` label, branch prefix.
- Model slug and schedule interval of the Cursor automation (15 minutes by default).

Use only model slugs shown as available in the user's Cursor model picker or supported model list. Do not guess a slug and do not carry over a private default.

Resolve the campaign model to an identifier KernelForge's provider supports and verify the reasoning
effort against the pinned KernelForge source (`--agent-reasoning-effort`); do not guess a model, invent a
flag or drop the effort. Node hours cover setup, finalize, stress and validation as well as the loop. The
model budget covers every inner agent call; outer automation and GPU CI need their own authorization.
Never estimate hard caps from logs. Do not guess any paid-run choice: a missing one blocks the actions that
need it, not read-only inspection. Ask every item above in the same pass; with `mode: run`, end setup by
listing each one still `null` as `prepared (blocked: ...)`, never as ready.

## 4. Check integration capabilities

Run these checks where every scheduled round will run, not only in this session.

The automation needs:

- A self-hosted Cursor worker on the host below, not a Cursor cloud agent: a cloud agent has no
  persistent state directory, no `/proc` of earlier rounds and no cluster access. Use a My Machines or
  repo-backed pool worker in the checkout that holds `.shepherd/configuration.yaml`, on the base branch:
  these use the checkout they have, while an any-repo pool worker started with `--clone-git-repos` clones
  a fresh one without the Git-ignored configuration.
- Read access to this pack and `.shepherd/configuration.yaml` by repository path.
- Read access to main, PRs, CI and reviews of the repository; with publication on, branch push and a
  draft pull request action. The `ci:gpu` label is a separate authority.
- One host and one identity for the control plane (Cursor worker, wrapper, agent, ledger, GitHub
  credentials), reaching the same state directory every time: local Linux disk with working `flock` and
  atomic rename. Not temporary storage, a fresh agent checkout, a copy, or an unverified network
  filesystem.
- One PID namespace for every round, with `/proc` readable: the host itself or one long-lived container.
  The round wrapper decides whether a previous round is still running from `/proc`; a round in another
  namespace looks undecidable and blocks every later one.
- `remote-gpu` reading the configured cluster, and `forge_task` and KernelForge help commands resolving
  in the pinned runtime. No node is needed for this.
- Worker isolation, probed on an allocated node: start a container with the worker's mounts
  (`remote-gpu exec --image <image> --mount <probe>/worker --mount <probe>/secrets:ro`) and record its
  `docker inspect` mounts, that `gh auth status` fails, that no token-like variable is set, and that the
  validate directory and the cluster `$HOME` are not visible. The cluster `$HOME` holds no GitHub
  credentials. Image and tool versions are pinned.

Do not use undocumented integration endpoints.

## 5. Initialize state

Read [state.md](../optimize-and-deliver-kernels/references/state.md). Initialize a new state directory once with the helper;
it creates missing private parents. Setup cannot `save` (only a running round can); the configuration keeps
the repository, authorization and caps. An existing directory, or one missing after use, needs
investigation, not a new `init`.

Record the passing checks of step 4 and the budget authorization in `readiness.md` in the state directory.
If a check has not run, leave paid mode off.

## 6. Prepare the live automation

Ask whether this is first-time creation or configuration of an existing automation.

Read `../../FOR_AGENTS.md` from the copied pack as the primary user-intent source for either path. Use it to understand the trigger, tools, instructions, outcome, and shared rules.

### First-time creation

1. Read the matching copied prompt template as secondary internal source material.
2. Turn `FOR_AGENTS.md`, the finished shepherd configuration, and the template intent into a complete natural-language request.
3. Tell the live prompt to read and follow its exact committed operational file under `harness/automations/shepherd/`.
4. Use the stable repository-relative path, not a plugin source or cache path. Do not copy the operational file contents into the live prompt.
5. Read and follow the built-in `automate` skill.
6. Let `automate` discover the repository and connected integrations.
7. Let `automate` confirm that the copied pack is committed in the same repository where the automation will run.
8. Let `automate` show its draft table, obtain approval, ask readiness, and open the Automations editor.

Give `automate` this complete intent, filled from configuration:

- Name `shepherd`.
- Read and follow `harness/automations/shepherd/skills/optimize-and-deliver-kernels/SKILL.md` with `.shepherd/configuration.yaml` for every run.
- Trigger on the configured schedule, every 15 minutes by default; repository events may also trigger it. Never treat a bot comment as a trigger.
- Run on the self-hosted Cursor worker checked in step 4, in the configured repository and base branch.
- Include GitHub pull request, CI and review access, and draft pull request creation when publication allows it.
- Run one round per trigger through the state helper's `round` wrapper, relay its result and report, then end. The schedule supplies the next round.
- Never edit, rebase, approve or merge an existing pull request, manage schedules, or reinitialize state.

Do not duplicate `automate`'s repository, integration, completeness, authentication, draft-review, approval, readiness, or editor-handoff work.

### Existing automation

The built-in `automate` skill is creation-only. Do not use it to search for, inspect, or update existing automations.

Finish configuration, environment and state validation. Then give the user this concise editor checklist.

For the existing shepherd automation, update:

- Name and description
- Direct instruction to read `harness/automations/shepherd/skills/optimize-and-deliver-kernels/SKILL.md` with `.shepherd/configuration.yaml`
- Schedule trigger and interval, and any repository-event trigger
- Self-hosted Cursor worker, repository and base branch
- GitHub and draft pull request capabilities
- Paraphrased one-round, wrapper, report and no-edit instructions

Ask the user to update the automation directly in its Automations editor. Do not create a replacement or duplicate. When moving from another scheduler, disable the old trigger before enabling this one, and keep the state directory and budgets.

### Creation boundary

Never call a direct automation backend service or backend automation tool. Never use a browser URL that carries draft fields. Never build or open a Cursor protocol deep link. For a new automation, the only finish path is the built-in `automate` skill's reviewed Automations editor handoff.

Record the automation's name, model and schedule in `readiness.md`. If the automation cannot be configured from here, hand over the prompt intent, configuration path, readiness results and remaining settings, and say the automation is not saved.

Do not enable the automation until the test in step 7 passes after the editor save.

## 7. Test before paid operation

Before testing, confirm that the target repository's `.cursor/settings.json` and `harness/automations/shepherd/` are committed on the branch used by the automation checkout, and that the live prompt points at the exact committed round skill. If any check fails, stop. Tell the user that the automation cannot be enabled yet.

Verify:

1. One `observe` round through the wrapper (`round --now -- <round_agent> <prompt>`) reads GitHub and
   existing runs, lists candidates, writes a report and changes nothing.
2. The CPU tests pass (`python -m pytest harness/tests/shepherd`) with `TMPDIR` on a directory only you
   and root own, such as `/tmp`: the helper rightly refuses a temp root under another user's directory.
3. On the configured host, through the automation's own trigger, two concurrent triggers start one round,
   a second fresh round reads what the first saved, and after `kill -9` of a running agent the next
   trigger takes over with `AKLR_RECOVERED_FROM` set.

Then enable `run` on probation, within the owner's authorization: `budget.total_*` set to one campaign's
and one pool's bounds (twice that if the owner allows a retry), `max_parallel_campaigns: 1`,
`max_open_prs: 1`, `allow_draft_pr: true`. Reports say `enabled (probation)`.

Accept it after one real campaign reaches confirmed cleanup and a draft PR, and a later automation-started
round observes that PR. CPU tests or a prompt preview do not count. If the probation budget runs out first,
report `blocked`, not accepted. After acceptance the owner records a new authorization and raises the
budget.

Report each state separately: prepared, saved, enabled, runtime-verified.
