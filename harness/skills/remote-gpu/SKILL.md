---
name: remote-gpu
description: "Allocate whole remote GPU nodes on the Slurm-family clusters under ~/.remote-gpu/clusters/ and use them through one `remote-gpu` CLI: get nodes, run commands or docker images, start long background jobs, sync files, release. Use whenever a task needs a real GPU / ROCm machine that isn't local: run/benchmark/profile/test a kernel or model on MI355X/gfx950 or another AMD GPU, multi-node torchrun, \"get a node\", Slurm, spur, salloc/sbatch/srun, aggressive allocation, or a temporary/preemptible machine."
argument-hint: "[--cluster NAME] <alloc|ready|exec|bg|wait|logs|release|extend|free|ls|info|push|pull> [args...]"
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/scripts/remote-gpu *)
---

# remote-gpu — whole GPU nodes on any Slurm-family cluster

Compute nodes can't be reached by ssh, and each scheduler has traps (no GPUs unless asked for,
steps that eat stdin or kill background processes). The CLI handles them, so use it rather than
raw `sbatch`/`srun`/`spur`. It works locally (over ssh) or on a login node:

```bash
RG=${CLAUDE_SKILL_DIR}/scripts/remote-gpu
```

Bash calls don't share variables: write the full path or start each command with `RG=...;`.
Where `${CLAUDE_SKILL_DIR}` is not expanded, it is `harness/skills/remote-gpu` from the AKLR root.

## Arguments

`$ARGUMENTS`:

- Empty or plain language ("2 nodes on mycluster for bench.py"): follow the workflow below.
- A command or `--cluster ...`: run `$RG $ARGUMENTS` once, exactly as given, in the background
  (see **Timeouts**), and once it exits report the `OK`/`ERR` lines and the job handle.

## Clusters

Each cluster is `~/.remote-gpu/clusters/<name>/`, shared by every clone on this machine: `cluster.env`
(login, scheduler, accounts, and what a node is: `GPU_MODEL`, `GPU_ARCH`, `GPUS_PER_NODE`,
`MAX_TIME`, `EXTEND`, `PREEMPTION`) and `cluster.sh` (the scheduler adapter). `$RG help` lists
them. Choose by the manifests, not by guessing:

```bash
grep -H -E '^(DESCRIPTION|GPU_ARCH|GPUS_PER_NODE|MAX_TIME|PREEMPTION)=' ~/.remote-gpu/clusters/*/cluster.env
```

If there is none, the user names one not listed, or none has the `GPU_ARCH` the task needs, run
the `remote-gpu-setup` skill. Never write `~/.remote-gpu/clusters/` yourself.

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
- **Account race.** On clusters with `ALLOC_AGGRESSIVE=1`, `alloc` submits one job per `ACCOUNTS`
  pair and keeps the first to start. `-A`, `-q` or `--no-wait` use one account instead.
- **Preemptible** (`--preemptible`): only when the user asks for it or for a temporary or
  short-lived machine; never for a run that must not stop, such as a campaign. It adds the
  cluster's `BURST_QOS` to the race (team jobs still win). If the OK line's `qos=` is the burst
  QOS, tell the user the job can be preempted at any time.
- **Queued jobs.** `alloc` blocks until a job starts or `--wait` runs out (rc 202, cancelled).
  `alloc --no-wait` queues on the cluster with the most free nodes and returns `state=PENDING`;
  `ready -j JOB` then waits up to 420 s (rc 205 = still pending). After the first 205, tell the
  user `reason=` and `start_estimate=` and ask whether to keep polling, leave it queued, or
  release. A reason starting `QOSMax` or `AssocMax` is the user's own limit: show `ls`.
- **Timeouts.** Run `alloc`, `ready`, `exec` and `wait` in the background (Bash
  `run_in_background`, `timeout` 7200000); you are woken when they finish. Without a background
  mode, use a `timeout` of 600000. For `-N` ≥ 2 on a busy cluster use `--no-wait` + `ready`. If
  `alloc` was killed anyway, its stderr has `SUBMITTED job=...`; check `ls` before allocating again.
- **Release** when done; clusters are shared. A job also releases itself after `--idle` minutes
  (default 60) with no running `exec` or task, and at its time limit (`-t`, default 4h, max `MAX_TIME`).

## Commands

| command | what it does |
|---|---|
| `alloc [-N n] [-t H:MM:SS] [-J label] [--idle MIN] [--wait SEC] [--no-wait] [-x node] [--aggressive] [--preemptible]` | Whole nodes with all their GPUs, health-checked. `--aggressive` races accounts where it is not the default. `-p/-A/-q/-w/-x` need `--cluster`. |
| `ready -j JOB [--wait 420]` | Waits for a queued job, health-checks it, prints the OK line. In the background, pass `--wait 6900`. |
| `exec -j JOB [-w node \| --all] [-C dir] [--image IMG] [-e K=V] -- CMD ARGS...` | Runs and waits; exits with CMD's code. `-e` values are visible on the node: no secrets. |
| `bg -j JOB [exec options] [--name TASK] -- CMD...` | Starts a detached task; each name once per job. |
| `wait -j JOB TASK [--timeout 540]` | Waits for the task, prints log tails, exits with its rc (205 = still running). In the background, pass `--timeout 6900`. |
| `logs -j JOB [TASK] [-n 50]` | Lists tasks, or tails one's log. |
| `release -j JOB` | Stops the job's tasks and containers, gives the nodes back. |
| `extend -j JOB -t NEW_TOTAL` | New total time limit, where `EXTEND=1`. |
| `free` / `ls` / `info` | Free nodes / my jobs / accounts, QOS limits, node states; every cluster unless `--cluster`. |
| `push LOCAL [REMOTE]` / `pull REMOTE [LOCAL]` | rsync via the login node; needs `--cluster`. A git repo going to a new path is cloned there from its origin first, so only local changes are copied. A path the cluster mounts at the same place is left alone. |
| `shell -j JOB` | Interactive; humans only. |

Everything after `--` is passed through exactly (spaces, quotes, `$`, `!`). For pipes, `&&` or
redirection use `bash -lc '...'`, which runs on the node, so `$VAR` expands there.

## Running work

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
- **Docker.** `--image IMG` runs CMD in `docker run --rm` as root, with the GPUs, host network and
  IPC, the work root and `$HOME` mounted, plus the manifest's `DOCKER_FLAGS`. `release` removes it.
- **Multi-node.** `-N 2`+ with `exec --all` or `bg --all` runs one copy per node, each with
  `REMOTE_GPU_NODE_RANK`, `REMOTE_GPU_NNODES`, `REMOTE_GPU_NODES`, `MASTER_ADDR` and `MASTER_PORT`,
  e.g. `bash -lc 'torchrun --nnodes $REMOTE_GPU_NNODES --node-rank $REMOTE_GPU_NODE_RANK --master-addr $MASTER_ADDR ...'`.
  Output lines are prefixed `[node]`, stderr merged; the exit code is the first non-zero one.
- **Busy GPUs.** `WARN node=... pre-existing docker containers` means someone's leftovers may hold
  GPU memory. Check with `exec -- timeout 60 amd-smi metric --mem-usage`; if busy, release and
  re-alloc with `--cluster X ... -x <node>`.
- **NFS.** A file just written on one host can look missing on another for ~60 s; check it on
  the host that wrote it, or poll.

## When things fail

Errors print `ERR code=<NAME> ...` on stderr. Since `exec` and `wait` return the remote command's
rc, remote-gpu's own codes are 200–210 and 255:

| rc | code | what to do |
|---|---|---|
| 200 | USAGE / UNSUPPORTED | Bad flags, a bare job ID without `--cluster`, `-w` not in the job, a reused task name, or `extend` where `EXTEND=0`. Read `msg`. |
| 201 | QUOTA_FULL | A per-user job limit (`ls`, release what you don't need), or every team cap is full: wait and retry, or ask about `--preemptible`. |
| 202 | PENDING_TIMEOUT | Didn't start within `--wait`; cancelled. Use `--no-wait` + `ready`. |
| 203 | JOB_GONE | Time limit, idle release, preemption, or released. `alloc` again and re-run. |
| 204 | BAD_NODE | A node lacked GPUs or couldn't see the state dir; cancelled. Re-alloc with `--cluster X ... -x <node>` (from `ready`: tell the user the queued job was lost). |
| 205 | still running / pending | `wait`/`ready`: call again. `JOB_PENDING` from `exec`/`bg`: use `ready`. |
| 206 | SUBMIT_FAILED / UNSCHEDULABLE / NODES_UNKNOWN / INSTALL_FAILED | The request is wrong or can never start (fix it), the controller was busy (retry), or the shared filesystem is full (tell the user). |
| 207 | NO_TASK / TASK_START_FAILED | Wrong task name, or the task didn't start on some node (`started_on=` lists where it did). |
| 208 / 209 | EXTEND_FAILED / RSYNC_FAILED | Read the message. |
| 210 | NO_FREE_NODES | From `alloc --no-wait` without `--cluster`; nothing submitted. Tell the user the counts (`/burst=N` = burst nodes) and ask: queue on one (`--cluster NAME alloc --no-wait`, then `ready`; may take hours) or `--preemptible`. |
| 255 | NO_CLUSTER | No cluster yet: run the `remote-gpu-setup` skill. |
| 255 | ssh failure / PROBE_FAILED | Check `ssh <HOST> true`; `HOST=` in the manifest names the variable that pins another login node. Don't blindly re-run a non-idempotent command; check `ls` / `logs` first. |
