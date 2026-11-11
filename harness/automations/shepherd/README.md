# shepherd

shepherd is a scheduled automation that optimizes AKLR kernels unattended. Each round it observes GitHub and
the GPU jobs, then advances each KernelForge campaign as far as confirmed results allow and admits new
kernel-less schemas into free GPUs, validates the kernel, opens a draft PR and follows it until a maintainer merges or
closes it.

The files here are read by repository path; they do not appear as slash skills. The round reuses the
optimization, GPU, validation, PR and reporting skills in `harness/skills/`.

## Set it up

1. Point the agent at [`FOR_AGENTS.md`](./FOR_AGENTS.md); it follows
   [`setup-shepherd`](./skills/setup-shepherd/SKILL.md).
2. Copy [`configuration.example.yaml`](./templates/configuration.example.yaml) to
   `.shepherd/configuration.yaml` at the repository root (ignored by Git) and fill it in.
3. Initialize the state directory once, pass the environment checks, and run one `observe` round.
4. Ask setup to create or update one scheduled task with
   [`automation-prompt.md`](./templates/automation-prompt.md), then enable `run` on probation (one
   campaign's budget, `max_parallel_campaigns: 1`, one PR) until a real campaign reaches a draft PR, and
   raise the budget and parallelism after.

## Files

```text
shepherd/
├── FOR_AGENTS.md                  intent and routing
├── templates/
│   ├── automation-prompt.md       scheduler prompt
│   ├── configuration.example.yaml
│   └── shepherd-round.{service,path,timer}  systemd units: round on issue events or schedule
└── skills/
    ├── setup-shepherd/SKILL.md       setup and configuration changes
    └── shepherd/
        ├── SKILL.md               one round
        ├── references/state.md    ledger, helper CLI, crash recovery
        └── scripts/
            ├── state.py           round wrapper, takeover and atomic snapshots
            ├── request_board.py   issue requests: approval, parsing, progress comment
            └── freshness.py       which main changes make evidence stale
```

## Limits

One Linux host with a persistent state directory, GitHub access and the GPU cluster. Up to
`max_parallel_campaigns` campaigns at once, one per GPU of shared pool nodes (a node's GPUs are not
isolated from each other's containers, so a campaign's timings share the node with its neighbours; the PR's
GPU CI measures alone). The ledger's budget is bookkeeping, not a hard limit: the provider's hard limit and Slurm time
limits are.

Isolation is a mount boundary under one identity. Workers run only on allocated nodes, in containers that
mount their own directory (and read-only model credentials for model-calling steps), never the validator,
earlier evidence, the cluster `$HOME` or GitHub and state credentials. This stops a worker from editing
what judges it; it does not stop a deliberate container escape, since containers run as root. Independent
review is the PR's GPU CI (`ci:gpu`).
