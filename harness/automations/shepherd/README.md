# shepherd

shepherd is a scheduled automation that optimizes AKLR kernels unattended. Each round it observes GitHub and
the GPU jobs, then advances one KernelForge campaign or admits a new kernel-less schema, validates the
kernel, opens a draft PR and follows it until a maintainer merges or closes it.

The files here are read by repository path; they do not appear as slash skills. The round reuses the
optimization, GPU, validation, PR and reporting skills in `harness/skills/`.

## Set it up

1. Point the agent at [`FOR_AGENTS.md`](./FOR_AGENTS.md); it follows
   [`setup-shepherd`](./skills/setup-shepherd/SKILL.md).
2. Copy [`configuration.example.yaml`](./templates/configuration.example.yaml) to
   `.shepherd/configuration.yaml` at the repository root (ignored by Git) and fill it in.
3. Pass the environment checks, initialize the state directory once, and run one `observe` round.
4. Ask setup to create or update one scheduled task with
   [`automation-prompt.md`](./templates/automation-prompt.md), then enable `run` within your budget
   authorization.

## Files

```text
shepherd/
├── FOR_AGENTS.md                  intent and routing
├── templates/
│   ├── automation-prompt.md       scheduler prompt
│   └── configuration.example.yaml
└── skills/
    ├── setup-shepherd/SKILL.md       setup and configuration changes
    └── shepherd/
        ├── SKILL.md               one round
        ├── references/state.md    ledger, helper CLI, crash recovery
        └── scripts/state.py       round lock and atomic snapshots
```

## Limits

One Linux host with a persistent state directory, GitHub access and the GPU cluster. One GPU campaign
at a time. Budget enforcement and validation isolation rely on the execution environment; a trusted
tool service is planned.
