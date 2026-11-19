---
name: setup-shepherd
description: Configure shepherd and prepare its Cursor automation. Use when installing shepherd or changing its repository, environment, model, budget, publication or schedule settings.
disable-model-invocation: true
---

# Set up shepherd

shepherd is a dormant automation pack in the AKLR repository. This file and `optimize-and-deliver-kernels` are not slash skills.

The human enters setup by pointing an agent at this file, in the checkout that will run rounds.

shepherd needs one configuration file, one state directory and one live Cursor automation.

The owner creates and starts the Cursor automation themselves. Setup hands over its fields (step 6), asks nothing about Cursor and waits for nothing there. Setup starts no paid campaign. Never put a secret in plugin files, prompts or committed configuration.

## 1. Check the checkout

Do this before asking for configuration.

The round checkout is the AKLR checkout on the round host, on the base branch. It already holds this pack. Never copy pack files into it.

Verify that it contains this file, `optimize-and-deliver-kernels` with its `references/` and `scripts/`, and the templates. If you were sent here from another copy of the pack and it is newer (`git log -1 -- harness/automations/shepherd`), stop and ask the user to update the checkout.

The automation runs each round itself and needs no plugin. Do not edit `.cursor/settings.json`.

Rounds read the shared skills by path from the checkout, never from an installed plugin or a discovered slash skill, whose cached revision can lag the pack. Verify that the checkout holds `harness/skills/<name>/SKILL.md` for:

- `forge-loop-campaign`
- `remote-gpu`
- `read-validation`
- `open-pr`
- `report-progress`
- `kb`

If any is missing, stop and explain the failure. Do not add `harness/automations/shepherd/skills/` to a plugin manifest.

Tell the user that the pack must be committed on the base branch before the automation is enabled. Do not commit it unless the user asks.

## 2. Adapt the configuration

Copy `../../templates/configuration.example.yaml` to `.shepherd/configuration.yaml` in the repository root. Git ignores it. If it exists, keep its values. Do not edit the example.

Write paths with environment variables such as `$USER`. No host names, usernames or personal absolute paths, even locally. Secrets stay in the credential store.

The configuration is not committed. The live prompt names its path and never paraphrases its values, so the automation must run in the checkout that holds it (step 4).

## 3. Fill the required choices

Ask for or confirm, in one pass:

- Repository and base branch. Infer them from the checkout.
- Mode: `run`, `observe` or `drain`. The owner's choice: `run` needs no `observe` phase first.
- State directory, cluster, image digest and runtime revision. Offer found values: the image is `AKLR_CI_IMAGE` in `.github/workflows/ci.yaml`, resolved to its digest. The runtime is the Hyperloom `AKLR` branch commit (KernelForge) plus the flydsl version `harness/scripts/forge_task/README.md` pins.
- `execution.round_hours`: the round's lease, no longer than the automation's run limit, which the owner knows (step 4).
- Strategies (`manual` paths, `kernel-less`, `roofline` and its `min_gap_us`), `batch_size`, `allowed_ops`, `target_models`, `backends`, `max_parallel_campaigns`, `max_open_prs`, `execution.pool_hours`, `cooldown_hours`.
- Campaign model, reasoning effort, loop hours, node hours, model USD, `max_auto_resumes`, and `model_usd_per_hour`: the highest hourly spend of the model with KernelForge, from provider billing.
- Cumulative budget, its recorded authorization, and the provider's hard limit: the spend cap on the API key or gateway the campaign calls (`claude auth status`, `ANTHROPIC_BASE_URL`). If none exists, the owner must set one.
- Publication: draft PRs, PR comments, `ci:gpu` label, branch prefix, and the issue that is told when a round needs the owner (`attention_issue`; its subscribers are notified by GitHub).

Resolve the campaign model to an identifier KernelForge's provider supports. Check the reasoning effort against the pinned KernelForge source (`--agent-reasoning-effort`). Do not guess a model, invent a flag or drop the effort.

Node hours cover setup, finalize, stress and validation as well as the loop. The model budget covers every inner agent call. The outer automation and GPU CI need their own authorization. Never estimate hard caps from logs.

Do not guess any paid-run choice. A missing one blocks the actions that need it, not read-only inspection. With `mode: run`, list each value still `null` as `prepared (blocked: ...)`, never as ready.

## 4. Check the host

Run these checks where every scheduled round will run, not only in this session.

The automation is the owner's to set up. Tell them what it needs, and check none of it:

- A self-hosted Cursor worker, not a cloud agent. A cloud agent has no persistent state directory, no `/proc` of earlier rounds and no cluster access.
- A My Machines or repo-backed pool worker in the checkout that holds `.shepherd/configuration.yaml`. An any-repo pool worker with `--clone-git-repos` clones a fresh checkout without it.
- `execution.round_hours` at or below the worker's run limit: a round the runner cuts off is then taken over as soon as its lease expires. Cursor documents no limit; a harmless long command measures it.
- No GitHub or other integration. The round uses the host's own credentials.

The round needs:

- The host's own GitHub credentials (`gh auth status`): read access to main, PRs, CI and reviews; with publication on, branch push and draft PR creation. The `ci:gpu` label is a separate authority.
- `remote-gpu` reading the configured cluster, and `forge_task` and KernelForge help commands resolving in the pinned runtime. No node is needed for this.

The host needs:

- One host and one identity for the worker, ledger and GitHub credentials.
- A state directory on local Linux disk with working `flock` and atomic rename. Not temporary storage, a fresh checkout, a copy, or an unverified network filesystem.
- PyYAML for the system `python3` (the state helper reads the configuration).

Probe worker isolation on a node of the configured cluster. Allocate it yourself, without asking: read `harness/skills/remote-gpu/SKILL.md` by path and follow it (as step 1 says, never the slash skill), and release the node when the probes end. Start a container with the worker's mounts (`remote-gpu exec --image <image> --mount <probe>/worker --mount <probe>/secrets:ro`). Record its `docker inspect` mounts, that `gh auth status` fails, that no token-like variable is set, and that the validate directory and the cluster `$HOME` are not visible. The cluster `$HOME` holds no GitHub credentials. Image and tool versions are pinned.

Do not use undocumented integration endpoints.

## 5. Initialize state

Read [state.md](../optimize-and-deliver-kernels/references/state.md). Most setups are the first on their host: when the state directory is missing or holds no ledger, initialize it with the helper and go on, without asking. It creates missing private parents and takes a directory made ahead.

The helper refuses a directory that holds a ledger or its remains, and names what it found. With a valid ledger (`show` prints it), keep it and skip `init`. Otherwise stop and ask the user: only they can say whether a ledger was lost.

Setup cannot `save`; only a running round can. The configuration keeps the repository, authorization and caps.

Record the passing checks of step 4 and the budget authorization in `readiness.md` in the state directory. List each check that has not run, with the reason.

## 6. Prepare the live automation

The owner creates or updates the automation and starts it; the pack does not depend on where or how. Finish configuration, host and state validation, then hand over these fields:

- Name `shepherd`, and a description.
- Prompt: [`automation-prompt.example.md`](../../templates/automation-prompt.example.md), as is or adapted. It must point at the committed `optimize-and-deliver-kernels` skill by its repository-relative path and at `.shepherd/configuration.yaml`. It keeps the template's two `begin` commands, so a skipped trigger reads nothing more, and otherwise never copies the skill's contents, paraphrases configuration values or uses a plugin cache path.
- Trigger: only a schedule (every 15 minutes is cron `*/15 * * * *`). No repository-event triggers.
- A self-hosted worker as step 4 describes, the configured repository and base branch.
- No integrations.
- A strong model: it runs every round.

For an existing automation, update these fields in place. Do not create a replacement or duplicate. When moving from another scheduler, disable the old trigger first, and keep the state directory and budgets.

Never call an automation backend service directly, and never build a link that carries draft fields.

Hand over the configuration path and the readiness results with them. Setup does not see the automation: never report it saved or enabled.

## 7. Test

Confirm that the pack is committed on the automation's branch. If not, stop and tell the user the automation cannot be enabled yet.

`python -m pytest harness/tests/shepherd` passes with `TMPDIR` on a directory only you and root own, such as `/tmp`. The helper refuses a temp root under another user's directory.

With `mode: observe`, also run one round in this session (`begin --now`, the round, `end`): it reads GitHub and existing runs, lists candidates, writes a report and changes nothing. With `mode: run`, skip it: a round here could start a paid campaign, and the automation's first round is the test.

Hand these to the owner as options, not conditions:

- Trigger behaviour on the live automation: two concurrent triggers start one round; a second round begins once the first ended; after stopping a running round in Cursor, the first trigger after its lease expires takes over with `recovered_from` set. `mode: observe` makes these free: the helper refuses an observe round's saves (the first `begin` migrates an older ledger).
- Probation for a first `run`: `budget.total_*` at one campaign's and one pool's bounds (twice that with a retry), `max_parallel_campaigns: 1`, `max_open_prs: 1`, `allow_draft_pr: true`. It ends when one real campaign reaches confirmed cleanup and a draft PR, and a later automation-started round observes that PR; the owner then records a new authorization and raises the budget.

Report each of these separately: configuration, state, every check of step 4 (passed, failed or not run, with the reason), the tests above.
