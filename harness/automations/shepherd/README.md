# shepherd

shepherd gives you one automation that optimizes AKLR kernels unattended. Each round observes GitHub
and the GPU jobs, advances each KernelForge campaign as far as confirmed results allow, admits new
kernel-less schemas into free GPUs, validates the kernel, opens a draft PR and follows it until a maintainer
merges or closes it.

The files in this directory are dormant setup and automation sources. They do not appear as slash skills.
The round reuses the optimization, GPU, validation, PR and reporting skills in `harness/skills/`.

## Set it up

1. Point the agent at [`FOR_AGENTS.md`](./FOR_AGENTS.md) and name the AKLR checkout, on the base branch,
   that will run rounds. It already holds this pack; nothing is copied.
2. The Cursor automation runs each round itself, so its run shows every step. It reads the shared
   skills by path from that checkout, not from an installed plugin whose cached revision can lag.
3. Keep user-owned configuration outside the pack, in `.shepherd/configuration.yaml` (ignored by
   Git). Adapt [`configuration.example.yaml`](./templates/configuration.example.yaml).
4. Review the new automation draft or update the existing automation in its editor. Enable it with
   `mode: observe`, run the setup tests, then switch to `run` on probation (one campaign's budget,
   `max_parallel_campaigns: 1`, one PR) until a real campaign reaches a draft PR, and raise the budget
   and parallelism after.
