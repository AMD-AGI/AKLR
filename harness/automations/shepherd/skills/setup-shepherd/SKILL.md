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
`scripts/state.py`, `scripts/request_board.py`) and the templates. Confirm the shared skills the round uses exist under
`harness/skills/`: forge-loop-campaign, remote-gpu, aklr-read-validation, aklr-open-pr, report-progress,
aklr-kb and the configured backends' knowledge.

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
  exists, the owner sets one or records `unlimited`; it cannot be inferred from here.
- Publication: draft PRs, PR comments, `ci:gpu` label, branch prefix.
- Issue requests: `requests.team` (a team of the repository's organization; start with the owner alone)
  and `requests.bot` (the shepherd App's login). Keep `requests.label` at `shepherd`, which
  `.github/workflows/shepherd.yaml` also names. Leave the `requests` strategy out until both are set.

Resolve the model to an identifier KernelForge's provider supports and verify the reasoning effort; do
not guess a model, invent a flag or drop the effort. Node hours cover setup, finalize, stress and
validation as well as the loop. The model budget covers every inner agent call; outer automation and
GPU CI need their own authorization. Never estimate hard caps from logs. Do not guess any paid-run
choice: a missing one blocks the actions that need it, not read-only inspection. Ask every item above in
the same pass; with `mode: run`, end setup by listing each one still `null` as `prepared (blocked: ...)`,
never as ready. Verify the effort against the pinned KernelForge source (`--agent-reasoning-effort`).

## 4. Initialize state

Read [state.md](../shepherd/references/state.md). Initialize a new state directory once with the helper;
it creates missing private parents. Setup cannot `save` (only a running round can); the configuration keeps
the repository, authorization and caps. An existing directory, or one missing after use, needs
investigation, not a new `init`.

## 5. Check the execution environment

Run these checks where every scheduled round will run, not only in this session:

- It reads this pack and `.shepherd/configuration.yaml` by repository path.
- It can read main, PRs, CI and reviews of the repository; with publication on, it can push a branch
  and open a draft PR. The `ci:gpu` label is a separate authority.
- The control plane (scheduler, wrapper, agent, ledger, GitHub credentials) runs on one host, as one
  identity, and reaches the same state directory every time: local Linux
  disk with working `flock` and atomic rename. Not temporary storage, a fresh agent checkout, a copy,
  or an unverified network filesystem.
- Every round runs in the same PID namespace with `/proc` readable: the host itself or one long-lived
  container. The round wrapper decides whether a previous round is still running from `/proc`; a round in
  another namespace looks undecidable and blocks every later one.
- `remote-gpu` reads the configured cluster; `forge_task` and KernelForge help commands resolve in
  the pinned runtime. No node is needed for this.
- Worker isolation, probed on an allocated node: start a container with the worker's mounts
  (`remote-gpu exec --image <image> --mount <probe>/worker --mount <probe>/secrets:ro`) and record its
  `docker inspect` mounts, that `gh auth status` fails, that no token-like variable is set, and that the
  validate directory and the cluster `$HOME` are not visible. The cluster `$HOME` holds no GitHub
  credentials. Image and tool versions are pinned.

Record the passing checks and the budget authorization in `readiness.md` in the state directory. If a
check has not run, leave paid mode off.

## 6. Prepare the scheduled task

Ask whether this is a first-time creation or an existing task.

**First time.** Use the owner's chosen scheduler, for example GitHub Actions on a self-hosted runner or
host cron. It runs the state helper's `round` wrapper with the agent CLI after `--` (see
[state.md](../shepherd/references/state.md)), never the agent directly, and sends SIGTERM to the wrapper on
timeout. A scheduler that can only send a prompt to an agent (for example a chat app's scheduled task)
may send `templates/automation-prompt.md` directly: the round skill then starts the wrapper with
`execution.round_agent` itself. Give the agent `templates/automation-prompt.md` as the prompt; keep workflow rules in the round
skill, not in the prompt. Bind the task to the checkout and environment checked in
step 5, set the interval (15 minutes by default), and record the task identity, the outer model and its
reasoning effort. Repository events may trigger the same task; polling stays the fallback. Do not rely
on bot comments as triggers or invent a scheduler API.

**Existing task.** Update it in place; never create a second one. When moving to another scheduler,
disable the old trigger before enabling the new one. Keep the state directory and budgets.

**Issue requests (systemd and the runner).** The owner's choice for request-driven rounds is the three
`systemd --user` units in `templates/`, on the host that holds the state directory, which is also where
the `aklr-claude` runner runs as the same user:
1. Create `<state_dir>/events` (mode 700). Set the repository variables `SHEPHERD_EVENTS_DIR` (its
   absolute path) and `SHEPHERD_CHECKOUT` (this checkout), and the secrets `SHEPHERD_APP_ID` and
   `SHEPHERD_APP_PRIVATE_KEY`. `.github/workflows/shepherd.yaml` labels new requests with the App's token
   and writes one file per issue event there, only when a team member or the App caused it.
2. Copy `shepherd-round.service`, `.path` and `.timer` to `~/.config/systemd/user/` and replace
   `@CHECKOUT@` (this checkout), `@STATE_DIR@` (the expanded state directory), `@ROUND_AGENT@`
   (`execution.round_agent`, space separated) and `@PROMPT@` (`templates/automation-prompt.md` without
   its comment line, on one line). Changing `round_agent` or the prompt means repeating this.
3. `systemctl --user daemon-reload`, then enable and start `shepherd-round.path` and
   `shepherd-round.timer` only when the user asks. The timer replaces any other scheduled task.
4. The shepherd App needs the organization's Members read permission (team membership) and the
   repository's Issues write permission (label, progress comments); the round's `gh` must act as it.
   Record in `readiness.md` that a non-member's issue gets no label, and a non-member's label or edit
   starts no round.

If the scheduler cannot be configured from here, hand over the prompt, configuration path, readiness
results and remaining settings, and say the task is not saved.

## 7. Verify before paid operation

1. Run one `observe` round through the wrapper (`round --now -- <round_agent> <prompt>`): it reads GitHub
   and existing runs, lists candidates, writes a report and changes nothing.
2. Run the CPU tests (`python -m pytest harness/tests/shepherd`) with `TMPDIR` on a directory only you
   and root own, such as `/tmp`: the helper rightly refuses a temp root under another user's directory. On the configured host, through the
   scheduler's own command, show that two concurrent triggers start one agent, a second fresh round reads
   what the first saved, and after `kill -9` of a running agent the next trigger takes over with
   `AKLR_RECOVERED_FROM` set.
3. Enable `run` within the owner's authorization.

Report each state separately: prepared, saved, enabled, runtime-verified.
