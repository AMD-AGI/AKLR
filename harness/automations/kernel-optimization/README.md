# AKLR kernel optimizer

This pack runs one Cursor Automation that selects schemas, optimizes kernels with
KernelForge, validates the results, opens draft PRs, and follows their status.
It has one entry skill and four skills for the M1 responsibilities.

## Start here

The workflow entry is [kernel-automation](skills/kernel-automation/SKILL.md).
It loads the configuration and calls the four skills for each round.

The other files help you set up that entry in Cursor:

- [FOR_AGENTS.md](FOR_AGENTS.md) tells an agent how to set up or edit this pack.
  Ask the agent to read it when configuring the automation.
- [automation-prompt.md](templates/automation-prompt.md) is the text to paste into
  the Cursor Automation prompt field. Each trigger runs this prompt, which tells
  Cursor to read the entry skill. It keeps the workflow rules in the skills.
- [configuration.example.yaml](templates/configuration.example.yaml) is a starting
  configuration. Copy it to `.cursor/kernel-optimizer/configuration.yaml` and fill
  in the environment, model, budgets, and permissions. Keep existing configuration
  if that file is already present. This user configuration is ignored by Git; keep
  machine names, deployment paths and account details there. The template itself is not used at runtime.

Saving these files does not create or enable an Automation. Follow the
[setup guide](skills/kernel-automation/references/setup.md) to configure it in Cursor.

## Skills

| Skill | Role |
|---|---|
| [kernel-automation](skills/kernel-automation/SKILL.md) | Entry: load configuration, manage the round, and call the four skills. |
| [Monitor / Analyze](skills/kernel-automation-monitor/SKILL.md) | Read repository, GPU job, and PR facts; identify changes and blockers. |
| [Plan](skills/kernel-automation-plan/SKILL.md) | Select one action after checking candidates, budgets, and existing work. |
| [Execute](skills/kernel-automation-execute/SKILL.md) | Run an admitted action, validate a kernel, clean up resources, or deliver a draft PR. |
| [Knowledge](skills/kernel-automation-knowledge/SKILL.md) | Save state, operation records, evidence, and the report for the next round. |

Knowledge works throughout the round: load state first, save intent before an
external call, save its result, then report and release the round lock. Long jobs
return a handle so a later trigger can check and continue the same work.

These skills belong to one Automation and share one state directory. They reuse
the optimization, GPU, validation, and PR skills in `harness/skills/`.

## Files

```text
kernel-optimization/
├── FOR_AGENTS.md
├── README.md
├── templates/
│   ├── automation-prompt.md
│   └── configuration.example.yaml
└── skills/
    ├── kernel-automation/
    │   ├── SKILL.md
    │   └── references/setup.md
    ├── kernel-automation-monitor/SKILL.md
    ├── kernel-automation-plan/SKILL.md
    ├── kernel-automation-execute/SKILL.md
    └── kernel-automation-knowledge/
        ├── SKILL.md
        ├── references/state.md
        └── scripts/state.py
```

The Automation reads skills by their committed repository paths. These pack skills
do not need to be installed in the shared skill directory. User configuration stays
outside this pack, at `.cursor/kernel-optimizer/configuration.yaml`.

## Setup

1. Ask Cursor to read `FOR_AGENTS.md` in this directory.
2. Create or update the user configuration and follow the setup guide. Check the
   environment, initialize state only on first use, and test an `observe` round.
3. Commit the pack, shared skills, and generic template on the branch the Automation
   checks out. Provision the ignored user configuration in its execution environment
   so every invocation can read it. Keep deployment details out of commits.
4. Paste `templates/automation-prompt.md` into one Cursor Automation. The default
   schedule is every 15 minutes; supported GitHub events can use the same entry.
5. After the checks pass and spending is authorized, set `mode: run` and enable it.

`observe` only reads and reports. It starts no GPU jobs or inner model calls and
makes no GitHub changes. `run` can admit new work. `drain` finishes existing work
without admitting another campaign.

## Current limits

The pack needs a Linux control host, a persistent state directory shared across
rounds, and access to GitHub and the GPU cluster. State must survive agent checkouts.

The state helper provides a round lock and atomic saves. Budget enforcement and
validation isolation still depend on the configured execution environment. The
trusted tool service is planned work.

Only one GPU campaign runs at a time. The automation tracks existing PRs;
maintainers update their code, review them, and merge them.
