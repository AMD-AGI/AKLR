# shepherd

shepherd gives you one automation that optimizes AKLR kernels unattended. Each round observes GitHub
and the GPU jobs, advances each KernelForge campaign as far as confirmed results allow, admits new
kernel-less schemas into free GPUs, validates the kernel, opens a draft PR and follows it until a maintainer
merges or closes it.

The files in this directory are dormant setup and automation sources. They do not appear as slash skills.
The round reuses the optimization, GPU, validation, PR and reporting skills in `harness/skills/`.

## Set it up

1. Point the agent at [`FOR_AGENTS.md`](./FOR_AGENTS.md) and name the target repository, an AKLR checkout.
2. Let setup merge this whole directory into the target at `harness/automations/shepherd/`. It must preserve
   destination-only files and review conflicts instead of overwriting local edits.
3. Let setup enable amd-aklr in the target repository's `.cursor/settings.json` for shared dependencies:

```json
{
	"plugins": {
		"amd-aklr": { "enabled": true }
	}
}
```

4. Keep user-owned configuration outside the copied pack, in `.shepherd/configuration.yaml` (ignored by
   Git). Adapt [`configuration.example.yaml`](./templates/configuration.example.yaml).
5. Commit `.cursor/settings.json` and `harness/automations/shepherd/` before enabling the automation.
6. Review the new automation draft or update the existing automation in its editor. Then run one `observe`
   round and enable `run` on probation (one campaign's budget, `max_parallel_campaigns: 1`, one PR) until a
   real campaign reaches a draft PR, and raise the budget and parallelism after.

## Limits

One Linux host with a persistent state directory, GitHub access and the GPU cluster; the automation runs
there. Up to `max_parallel_campaigns` campaigns at once, one per GPU of shared
pool nodes (a node's GPUs are not isolated from each other's containers, so a campaign's timings share the
node with its neighbours; the PR's GPU CI measures alone). The ledger's budget is bookkeeping, not a hard
limit: the provider's hard limit and Slurm time limits are.

Isolation is a mount boundary under one identity. Workers run only on allocated nodes, in containers that
mount their own directory (and read-only model credentials for model-calling steps), never the validator,
earlier evidence, the cluster `$HOME` or GitHub and state credentials. This stops a worker from editing
what judges it; it does not stop a deliberate container escape, since containers run as root. Independent
review is the PR's GPU CI (`ci:gpu`).
