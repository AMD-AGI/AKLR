---
name: remote-gpu-setup
description: Add, re-probe, update or remove the clusters remote-gpu allocates on (~/.remote-gpu/clusters/), and verify them. Use for /remote-gpu-setup, when remote-gpu has no cluster or none that fits the task, or when the user asks to add, check or remove a cluster.
argument-hint: "[add|update|remove] [NAME]"
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/../remote-gpu/scripts/remote-gpu *) Bash(${CLAUDE_SKILL_DIR}/../remote-gpu/scripts/probe-cluster *)
---

# remote-gpu-setup — configure remote-gpu's clusters

Each cluster is `~/.remote-gpu/clusters/<name>/`: a manifest `cluster.env` and an adapter
`cluster.sh`. How to write and check one is in `references/add-cluster.md`; read it before step 3.

```bash
RG=${CLAUDE_SKILL_DIR}/../remote-gpu/scripts/remote-gpu
```

Where `${CLAUDE_SKILL_DIR}` is not expanded, the scripts are in `harness/skills/remote-gpu/scripts/`
from the AKLR root.

`$ARGUMENTS`: `add|update|remove [NAME]` skips step 2; empty runs every step.

## 1. Show the clusters

```bash
grep -H -E '^(VERIFIED|DESCRIPTION|HOST|GPU_ARCH|GPUS_PER_NODE|ACCOUNTS|MAX_TIME)=' ~/.remote-gpu/clusters/*/cluster.env
ls ~/.remote-gpu/clusters/*/
```

One row per cluster: name, description, login, arch, accounts, `VERIFIED`. Mark it stale if
`VERIFIED` is over 90 days old or either file is missing. No clusters: go to step 3 with `add`.

## 2. Choose

One AskUserQuestion: add, update one (stale first), remove one, or done.

## 3. Act

- **add**: `add-cluster.md` steps 1–4. A name already listed is an `update`.
- **update**: copy the directory to `~/.remote-gpu/clusters/.<name>.bak/`, run `probe-cluster` on
  its `HOST`, and show each key the probe disagrees with as `KEY: old -> new`. Ask which to take;
  keep the user's own choices (`PARTITION`, `ACCOUNTS`, `BURST_QOS`, `PRIORITY`, `DOCKER_FLAGS`)
  unless they say otherwise. Rewrite `cluster.sh` only if the scheduler or its version changed.
- **remove**: if `$RG --cluster <name> ls` shows jobs, stop (or release them on the user's word);
  confirm, then `rm -r ~/.remote-gpu/clusters/<name>`.

Never edit `~/.ssh/config` or keys. If `ssh -o BatchMode=yes <host> true` fails, tell the user what
to set up and stop.

## 4. Verify

Run `add-cluster.md` step 5's read-only checks. Offer its allocation smoke test once; it holds a
node for a few minutes, so run it only on a yes, and set `VERIFIED` to today only if it passed.
Delete the `.bak` once the checks pass.

## 5. Report

Show the step 1 table with what changed, what passed and what was not tried, then return to step 2
unless the user is done.
