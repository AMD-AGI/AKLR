---
name: remote-gpu-setup
description: Add, re-probe, update or remove the clusters remote-gpu allocates on (~/.remote-gpu/clusters/), Slurm-family clusters and Primus-SaFE workspaces, and verify them. Use for /remote-gpu-setup, when remote-gpu has no cluster or none that fits the task, or when the user asks to add, check or remove a cluster.
argument-hint: "[add|update|remove] [NAME]"
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/../remote-gpu/scripts/remote-gpu *) Bash(${CLAUDE_SKILL_DIR}/../remote-gpu/scripts/probe-cluster *)
---

# remote-gpu-setup — configure remote-gpu's clusters

Each cluster is `~/.remote-gpu/clusters/<name>/`: a manifest `cluster.env` whose `SCHEDULER`
(slurm, spur, or safe for Primus-SaFE) picks the adapter in remote-gpu's `schedulers/`, and
optionally `cluster.sh`, the hooks that site does differently. How to write and check one is in
`references/add-cluster.md`; read it before step 3.

```bash
RG=${CLAUDE_SKILL_DIR}/../remote-gpu/scripts/remote-gpu
PROBE=${CLAUDE_SKILL_DIR}/../remote-gpu/scripts/probe-cluster
```

Where `${CLAUDE_SKILL_DIR}` is not expanded, the scripts are in `harness/skills/remote-gpu/scripts/`
from the AKLR root.

`$ARGUMENTS`: `add|update|remove [NAME]` skips step 2; empty runs every step.

## 1. Show the clusters

```bash
grep -H -E '^(VERIFIED|DESCRIPTION|SCHEDULER|HOST|GPU_ARCH|GPUS_PER_NODE|ACCOUNTS|MAX_TIME)=' ~/.remote-gpu/clusters/*/cluster.env
ls ~/.remote-gpu/clusters/*/
```

One row per cluster: name, description, type, login, arch, accounts, `VERIFIED`. Mark it stale if
`VERIFIED` is over 90 days old or not a date, or its `cluster.sh` is an older one (see "Older
cluster directories" in `add-cluster.md`). No clusters: go to step 3 with `add`.

## 2. Choose

One AskUserQuestion: add, update one (stale first), remove one, or done.

## 3. Act

- **add**: `add-cluster.md` steps 1–4. A name already listed is an `update`.
- **update**: copy the directory to `~/.remote-gpu/clusters/.<name>.bak/`, run `add-cluster.md`
  step 1's probe with the manifest's `HOST` and `SCHEDULER` (safe: its `SAFE_URL` and the name
  too), and show each key the probe disagrees with as `KEY: old -> new`. Ask which to take; keep
  the user's own choices (`PARTITION`, `ACCOUNTS`, `BURST_QOS`, `PRIORITY`, `DOCKER_FLAGS`) unless
  they say otherwise. Replace the manifest by `mv` of a finished file, never in place: other
  sessions read it. An older `cluster.sh`: "Older cluster directories" says what to do.
- **remove**: if `$RG --cluster <name> ls` shows jobs, stop (or release them on the user's word);
  confirm, then `rm -r ~/.remote-gpu/clusters/<name>`.

Never edit `~/.ssh/config` or keys. If `ssh -o BatchMode=yes <host> true` fails, tell the user what
to set up and stop.

## 4. Verify

Run `add-cluster.md` step 5's read-only checks. Offer its allocation smoke test once; it holds a
node for a few minutes, so run it only on a yes, and set `VERIFIED` to today only if it passed.
Once the checks pass delete the `.bak`; if they cannot be made to pass, move it back.

## 5. Report

Show the step 1 table with what changed, what passed and what was not tried, then return to step 2
unless the user is done.
