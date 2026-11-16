# Optimize automation prompt

> Source material for the setup workflow. Paraphrase this intent into a built-in `automate` draft after `automate` confirms that the pack is committed in the repository where the automation will run.

Read and follow `harness/automations/shepherd/skills/optimize-and-deliver-kernels/SKILL.md` for this run.

Configuration source. The file is ignored by Git and exists only on the host that runs rounds. Never use a plugin source or cache path:

```text
.shepherd/configuration.yaml
```

Trigger: one schedule, every 15 minutes by default (cron `*/15 * * * *`). No repository-event triggers: each round reads GitHub itself, and the state helper's `begin` skips triggers before the ledger's `wake_at`. The run uses no trigger data.

The creation intent should describe this as one scheduled round on the self-hosted worker that holds the state directory. It should include the configured repository and base branch, and no GitHub or other integration: the round uses the host's own credentials.

The automation runs the round itself: it calls the state helper's `begin`, does the round in this run so every step is visible, calls `end`, and replies with the round report in concise Chinese. The schedule supplies the next round; it manages no schedules and never reinitializes state.

Never edit, rebase, approve or merge an existing pull request.
