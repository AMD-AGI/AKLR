---
name: remote-gpu
description: "Allocate whole remote GPU nodes on the clusters under ~/.remote-gpu/clusters/ and use them through one `remote-gpu` CLI: get nodes, run commands or docker images, start long background jobs, sync files, release. Use whenever a task needs a real GPU / ROCm machine that isn't local: run/benchmark/profile/test a kernel or model on MI355X/gfx950 or another AMD GPU, multi-node torchrun, \"get a node\", Slurm, spur, Primus-SaFE, salloc/sbatch/srun, aggressive allocation, or a temporary/preemptible machine."
argument-hint: "[--cluster NAME] <alloc|ready|exec|bg|wait|logs|release|extend|free|ls|info|push|pull> [args...]"
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/scripts/remote-gpu *)
---

# remote-gpu — whole GPU nodes on remote clusters

Use this CLI rather than raw `sbatch`/`srun`/`spur`: it handles each scheduler's traps (no GPUs
unless asked for, steps that eat stdin or kill background processes). It works locally (over ssh)
or on a login node:

```bash
RG=${CLAUDE_SKILL_DIR}/scripts/remote-gpu
```

Bash calls don't share variables: write the full path or start each command with `RG=...;`.
Where `${CLAUDE_SKILL_DIR}` is not expanded, it is `harness/skills/remote-gpu` from the AKLR root.

`$RG help` is the reference for every command, flag, default and exit code: read it before your
first command. This file holds what it does not say.

## Arguments

`$ARGUMENTS`:

- Empty or plain language: follow the workflow below.
- A command or `--cluster ...`: run `$RG $ARGUMENTS` once, exactly as given, in the background
  (see **Timeouts**), and once it exits report the `OK`/`ERR` lines and the job handle.

## Clusters

Each cluster is a manifest `~/.remote-gpu/clusters/<name>/cluster.env`, shared by every clone on
this machine. Its `SCHEDULER` is the cluster's type: `slurm`, `spur`, or `safe` (a Primus-SaFE
workspace). Choose from the manifests:

```bash
grep -H -E '^(DESCRIPTION|SCHEDULER|GPU_ARCH|GPUS_PER_NODE|MAX_TIME|PREEMPTION)=' ~/.remote-gpu/clusters/*/cluster.env
```

The same commands drive every type; one a type lacks is refused at once with
`ERR code=UNSUPPORTED` (rc 200). On `safe` those are `alloc -N 2`+, `--image`, `shell` and
`extend`, and every command runs as root in the manifest's `SAFE_IMAGE`. `free=0` there can still
start: the workspace may add a node for a pending job, and `reason=` says so.

If there is no cluster, the user names one not listed, or none has the `GPU_ARCH` the task needs,
run the `remote-gpu-setup` skill: only it writes `~/.remote-gpu/clusters/`.

## The workflow

```bash
$RG alloc [-N n] [-t H:MM:SS]        # -> OK cluster=mycluster job=mycluster:202642 nodes=... workdir=... arch=gfx950
$RG exec -j mycluster:202642 -C mycluster/src/myrepo -- bash -lc 'pip install -e . && pytest -x tests/'
$RG release -j mycluster:202642      # ALWAYS, as soon as you are done
```

- **Job handle.** Keep the `job=CLUSTER:ID` that `alloc` prints and pass it to every `-j`; it names
  the cluster too. Other agents of the same user hold jobs as well, so never pick a job from `ls`
  and never `release --all`. Without `-j`, the CLI uses this session's one job when
  `REMOTE_GPU_SESSION` is set (it prints `using job=...`); otherwise show `ls` and ask.
- **Which cluster.** Without `--cluster`, `alloc` races every cluster and keeps the first to get
  nodes (`RACE kept=...`), releasing the rest. Pass `--cluster NAME` when the user names one, when
  only some clusters have the needed `GPU_ARCH`, or when the code is on a path that cluster mounts.
- **Account race.** Where `ALLOC_AGGRESSIVE=1`, or with `--aggressive`, `alloc` also races the
  cluster's `ACCOUNTS` pairs. A job a race could not cancel is printed as `CANCEL_FAILED job=...`
  and is still yours: release it.
- **Preemptible** (`--preemptible`): only when the user asks for it or for a temporary or
  short-lived machine; never for a run that must not stop, such as a campaign. Team accounts still
  win the race; when `alloc` or `ready` warns that the job can be preempted, tell the user.
- **Queued jobs.** `alloc` blocks until a job starts or `--wait` runs out (rc 202, cancelled).
  `alloc --no-wait` queues on the cluster with the most free nodes and returns `state=PENDING`;
  `ready -j JOB` then waits for it (rc 205 = still pending). After the first 205, tell the user
  `reason=` and `start_estimate=` and ask whether to keep polling, leave it queued, or release. A
  reason starting `QOSMax` or `AssocMax` is the user's own limit: show `ls`.
- **Timeouts.** Run `alloc`, `ready`, `exec` and `wait` in the background (Bash
  `run_in_background`, `timeout` 7200000), there with `ready --wait 6900` and
  `wait --timeout 6900`; you are woken when they finish. Without a background mode, use a
  `timeout` of 600000. For `-N` ≥ 2 on a busy cluster use `--no-wait` + `ready`. If `alloc` was
  killed anyway, its stderr has `SUBMITTED job=...`; check `ls` before allocating again.
- **Release** when done; clusters are shared. A job also releases itself at its time limit (`-t`,
  default 4h, max `MAX_TIME`) and after `--idle` minutes (default 60) with no running `exec` or
  task, unless `alloc` warned that an idle job keeps its node.

## Running work

- **The command.** Everything after `--` is passed through exactly (spaces, quotes, `$`, `!`). For
  pipes, `&&` or redirection use `bash -lc '...'`, which runs on the node, so `$VAR` expands
  there. `-e K=V` values are visible on the node: pass no secrets that way.
- **Under ~8 min**: `exec`. An `exec` that outlives the Bash tool's timeout is killed with its
  remote command, and the output is lost.
- **Longer** (big builds, benchmarks, training, large image pulls): `bg` + `wait`:
  ```bash
  $RG bg -j mycluster:202642 --name bench -C mycluster/src/myrepo -- python bench.py --iters 1000
  $RG wait -j mycluster:202642 bench --timeout 6900   # in the background; repeat while it exits 205
  ```
- **Paths.** `workdir=` is the job's state dir and the default `-C`. A relative `-C`, `push` or
  `pull` path is under the cluster's work root; `push` prints where it put the code. A `-C` that
  doesn't exist exits 126.
- **Docker.** `--image IMG` runs CMD as root in `docker run --rm`, with the mounts `$RG help`
  lists plus the manifest's `DOCKER_FLAGS`. Add `--mount DIR[:ro]` when the command must not see
  or write anything but those directories; it is refused when the manifest sets `DOCKER_FLAGS`.
- **Multi-node.** `-N 2`+ with `exec --all` or `bg --all` runs one copy per node, each with the
  `REMOTE_GPU_*` and `MASTER_*` variables `$RG help` lists, e.g.
  `bash -lc 'torchrun --nnodes $REMOTE_GPU_NNODES --node-rank $REMOTE_GPU_NODE_RANK --master-addr $MASTER_ADDR ...'`.
  stderr is merged into the `[node]`-prefixed output; the exit code is the first non-zero one.
- **Busy GPUs.** `WARN node=... pre-existing docker containers` means someone's leftovers may hold
  GPU memory. Check with `exec -- timeout 60 amd-smi metric --mem-usage`; if busy, release and
  re-alloc with `--cluster X ... -x <node>`.
- **NFS.** A file just written on one host can look missing on another for ~60 s; check it on
  the host that wrote it, or poll.

## When things fail

Errors print `ERR code=<NAME> ...` on stderr. Since `exec` and `wait` return the remote command's
rc, remote-gpu's own codes are 200–210 and 255. What to do, beyond reading `msg` and `hint`:

| rc | code | what to do |
|---|---|---|
| 200 | USAGE / UNSUPPORTED | Fix the command, or the cluster's type lacks it (**Clusters**); the same call fails again. |
| 201 | QUOTA_FULL | A per-user job limit (`ls`, release what you don't need), or every team cap is full: wait and retry, or ask about `--preemptible`. |
| 202 | PENDING_TIMEOUT | Didn't start within `--wait`; cancelled. Use `--no-wait` + `ready`. |
| 203 | JOB_GONE | Time limit, idle release, preemption, or released. `alloc` again and re-run. |
| 204 | BAD_NODE | A node lacked GPUs or couldn't see the state dir; cancelled. Re-alloc with `--cluster X ... -x <node>` (from `ready`: tell the user the queued job was lost). |
| 205 | still running / pending | `wait`/`ready`: call again. `JOB_PENDING` from `exec`/`bg`: use `ready`. |
| 206 | SUBMIT_FAILED / UNSCHEDULABLE / NODES_UNKNOWN / INSTALL_FAILED | The request is wrong or can never start (fix it), the controller was busy (retry), or the shared filesystem is full (tell the user). |
| 207 | NO_TASK / TASK_START_FAILED | Wrong task name, or the task didn't start on some node (`started_on=` lists where it did). |
| 210 | NO_FREE_NODES | From `alloc --no-wait` without `--cluster`; nothing submitted. Tell the user the counts (`/burst=N` = burst nodes) and ask: queue on one (`--cluster NAME alloc --no-wait`, then `ready`; may take hours) or `--preemptible`. |
| 255 | NO_CLUSTER | No cluster yet: run the `remote-gpu-setup` skill. |
| 255 | ssh failure / PROBE_FAILED | Check `ssh <HOST> true`; `HOST=` in the manifest names the variable that pins another login node. Don't blindly re-run a non-idempotent command; check `ls` / `logs` first. |
