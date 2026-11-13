# Optimize automation prompt

> Source material for the setup workflow. Paraphrase this intent into a built-in `automate` draft after `automate` confirms that the pack is committed in the repository where the automation will run.

Read and follow `harness/automations/shepherd/skills/optimize-and-deliver-kernels/SKILL.md` for this run.

Configuration source. The file is ignored by Git and exists only on the host that runs rounds. Never use a plugin source or cache path:

```text
.shepherd/configuration.yaml
```

Trigger: one schedule, every 15 minutes by default (cron `*/15 * * * *`). No repository-event triggers: each round reads GitHub itself, and the wrapper skips triggers before the ledger's `wake_at`. The run uses no trigger data.

The creation intent should describe this as one scheduled round on the self-hosted worker that holds the state directory. It should include the configured repository and base branch, and no GitHub or other integration: the round agent uses the host's own credentials.

`optimize-and-deliver-kernels` starts the state helper's `round` wrapper, which runs `execution.round_agent` with the round prompt below. Relay the wrapper's result and report, then end. The schedule supplies the next round.

Round prompt, passed to `execution.round_agent`:

```text
Run one shepherd round in this repository: read and follow `harness/automations/shepherd/skills/optimize-and-deliver-kernels/SKILL.md` with `.shepherd/configuration.yaml`. This is not setup: do not manage schedules or reinitialize state. Write the round report in concise Chinese and end after one round; the scheduler supplies the next one.
```

Never edit, rebase, approve or merge an existing pull request.
