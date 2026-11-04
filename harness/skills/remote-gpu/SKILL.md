---
name: remote-gpu
description: "Allocate and use whole remote GPU nodes on the Slurm-family clusters listed under ~/.remote-gpu/clusters/ through one `remote-gpu` CLI, and add a cluster it does not know yet: ask the user for its login, probe it, write its manifest and adapter from the template. Get nodes (races every cluster unless one is named), run commands or docker images on them, start long background jobs, sync files, release nodes. Each cluster's cluster.env says what its nodes are (GPU model, gfx arch, GPUs, time limits) for other skills to choose by. Use whenever a task needs a real GPU / ROCm machine that isn't local: run/benchmark/profile/test a kernel or model on MI355X/gfx950 or another AMD GPU, multi-node torchrun, \"get a node\", \"申请机器/申请节点/远程机器/上集群跑\", \"add a cluster/接入新集群\", or mentions Slurm, spur, salloc/sbatch/srun, aggressive/激进模式/抢机器, or a temporary/preemptible machine (临时机器)."
argument-hint: "[--cluster NAME] <alloc|ready|exec|bg|wait|logs|release|extend|free|ls|info|push|pull> [args...]"
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/scripts/remote-gpu *) Bash(${CLAUDE_SKILL_DIR}/scripts/probe-cluster *)
---

# remote-gpu — whole GPU nodes on any Slurm-family cluster

One CLI for every cluster under `~/.remote-gpu/clusters/`. Compute nodes can't be reached by ssh, and each
scheduler has traps (no GPUs unless asked for, steps that eat stdin or kill background processes).
The script handles all of that, so use it rather than raw `sbatch`/`srun`/`spur`:

```bash
RG=${CLAUDE_SKILL_DIR}/scripts/remote-gpu   # works locally (via ssh) or on a login node
```

The Bash tool does not keep shell variables between calls, so write the full path every time
(or start each command with `RG=...;`). Where `${CLAUDE_SKILL_DIR}` is not expanded, use this
SKILL.md's directory; from the AKLR clone's root it is `harness/skills/remote-gpu`.

## Invoked with arguments

Arguments: `$ARGUMENTS`

- Empty: you were triggered by the user's request; follow the workflow below.
- Starts with a command or `--cluster`: run `${CLAUDE_SKILL_DIR}/scripts/remote-gpu $ARGUMENTS` once, exactly as given, with a Bash `timeout` of 600000, and report the `OK`/`ERR` lines and the job handle.
- `-j` missing where a command needs it: the CLI uses the job this session allocated if there is exactly one and `REMOTE_GPU_SESSION` is set (it prints `using job=...`); otherwise it fails, so show `ls` and ask. Never pick another agent's job.
- Plain language (e.g. "2 nodes on mycluster for bench.py"): translate it into commands and follow the workflow.
- A cluster that is not in `$RG help`'s list: add it first, as `references/add-cluster.md` says.

## Clusters

Each cluster is a directory `~/.remote-gpu/clusters/<name>/`, outside the repo, so every clone and
worktree on this machine sees the same clusters:

| file | what it is |
|---|---|
| `cluster.env` | The manifest: login, scheduler, partition, accounts, and what a node is (`GPU_MODEL`, `GPU_ARCH`, `GPUS_PER_NODE`, CPUs, memory), `MAX_TIME`, `EXTEND`, `PREEMPTION`. Plain `KEY=value`, one fact per line. |
| `cluster.sh` | Scheduler hooks and site facts, made from `templates/slurm.sh`; its header defines the adapter contract. |

To choose a cluster for a job, read the manifests rather than guessing:

```bash
grep -H -E '^(DESCRIPTION|GPU_ARCH|GPUS_PER_NODE|MAX_TIME|PREEMPTION)=' ~/.remote-gpu/clusters/*/cluster.env
```

A task that needs a GPU architecture (a gfx950 kernel, say) takes only the clusters whose
`GPU_ARCH` matches, and names one with `--cluster` when only some of them fit. `alloc` prints
`arch=` in its OK line.

**A new cluster.** When the user names a cluster not in the list, or none of the listed clusters
fits, follow `references/add-cluster.md`: ask the user for what only they know, run
`scripts/probe-cluster` for the rest, and write `~/.remote-gpu/clusters/<name>/` from `templates/`. Nothing in
a cluster's directory holds an absolute path: paths are `$HOME/...` or come from `WORK_ROOT=auto`,
resolved on the cluster.

## The workflow

```bash
$RG alloc [-N n] [-t H:MM:SS]        # -> OK cluster=mycluster job=mycluster:202642 nodes=... workdir=... arch=gfx950
$RG exec -j mycluster:202642 -C mycluster/src/myrepo -- bash -lc 'pip install -e . && pytest -x tests/'
$RG release -j mycluster:202642         # ALWAYS, as soon as you are done
```

- **Job handle.** `alloc` prints `job=CLUSTER:ID`; the job's name in `ls` starts with the cluster's name (`mycluster-<tag>-...`), so `ls` shows which jobs remote-gpu started. Keep the handle and pass it to every later `-j`; it also names the cluster, so later commands need no `--cluster`. Don't guess a job from `ls`: other agents of the same user hold jobs too, and you might run on (or release) theirs. For the same reason, never use `release --all`.
- **Picking a cluster.** Without `--cluster`, `alloc` races every cluster at once: each runs its own `alloc` (racing its accounts where it can), the first to get nodes is kept (`RACE kept=<cluster>`), and the others are stopped and whatever they submitted is released. With `--no-wait` it instead queues on the cluster with the most free nodes (`free`; ties go to the lowest `PRIORITY`). Use `--cluster NAME` when the user names a cluster, when the task needs a `GPU_ARCH` only some clusters have, or when the code must stay where it is (a path the cluster mounts too needs no `push`).
- **Aggressive mode is the default** on clusters with `ALLOC_AGGRESSIVE=1`: `alloc` submits one job per `ACCOUNTS` pair at once, keeps the first that starts and cancels the rest (`RACE kept=...`). It blocks until a job starts or `--wait` runs out (`PENDING_TIMEOUT`, all cancelled). `-A`, `-q` or `--no-wait` turn the race off and use one account.
- **Preemptible nodes** (`--preemptible`): only when the user adds it, or asks for a temporary or short-lived machine ("临时机器", "短时间用一下"). It adds the cluster's `BURST_QOS` to the race (a team job still wins; a burst winner waits 10 s for one), or with `--no-wait` queues on burst alone. If the OK line's `qos=` is the burst QOS, tell the user the job can be preempted at any time. Never for a long run that must not stop, such as a campaign.
- **No free nodes** (`NO_FREE_NODES`, rc 210; only from `alloc --no-wait` without `--cluster`): nothing was submitted. Tell the user the counts (`/burst=N` is how many nodes a cluster's `BURST_QOS` has) and ask: queue on one (`--cluster NAME alloc --no-wait`, then `ready`; it can take hours), or `--preemptible` if they accept preemption.
- **Queued jobs.** `alloc --no-wait` returns `OK ... state=PENDING`. `ready -j JOB` waits up to `--wait 420` s, then health-checks and prints the full OK line; rc 205 means still pending. After a first 205, tell the user the `reason=` and `start_estimate=` and ask whether to keep polling, leave it queued, or release it. A reason starting with `QOSMax` or `AssocMax` is the user's own limit, not the queue: list their jobs with `ls` instead of waiting.
- **Tool timeouts.** Run `alloc`, `ready`, `exec` and `wait` with a Bash `timeout` of 600000 ms. For `-N` ≥ 2 on a busy cluster, use `--no-wait` + `ready` (the health check adds up to 2 min). If `alloc` was killed anyway, the job is in its stderr line `SUBMITTED job=...`; check `ls` before allocating again.
- **Release** nodes when you finish; clusters are shared. As a safety net a job releases itself after `--idle` minutes (default 60) without activity (a running `exec` or bg task counts), and at its time limit (`-t`, default 4h, max the cluster's `MAX_TIME`).

## Commands

| command | what it does |
|---|---|
| `alloc [-N n] [-t H:MM:SS] [-J label] [--idle MIN] [--wait SEC] [--no-wait] [-x node] [--aggressive] [--preemptible]` | Gets whole nodes with all their GPUs and health-checks them (`GPUS_PER_NODE` GPUs, and the node sees the state dir). Cluster-specific `-p/-A/-q/-w/-x` need `--cluster`; the manifest's `PARTITION` is the default, and the account is the one `free` reports the most room for (`account=`), else the first `ACCOUNTS` pair. |
| `ready -j JOB [--wait 420]` | Waits for a queued job to start, health-checks it, prints the OK line. rc 205 = still pending. |
| `exec -j JOB [-w node \| --all] [-C dir] [--image IMG] [-e K=V] -- CMD ARGS...` | Runs a command and waits. The exit code is CMD's own. `-e` values show in the node's process list and the job's state dir, so never pass a secret that way. |
| `bg -j JOB [same as exec] [--name TASK] -- CMD...` | Starts a detached task that survives disconnects. |
| `wait -j JOB TASK [--timeout 540]` | Polls until the task ends, prints log tails, exits with its rc (205 = still running). |
| `logs -j JOB [TASK] [-n 50]` | Lists tasks with their states, or tails a task's log. |
| `release -j JOB` | Stops the job's bg tasks and containers, gives the nodes back. |
| `extend -j JOB -t NEW_TOTAL` | New total time limit, on clusters with `EXTEND=1`. |
| `free` / `ls` / `info` | Free nodes per cluster / my jobs (state, time left, start estimate, idle minutes) / accounts, QOS limits, node states. Every cluster unless `--cluster` is given. |
| `push LOCAL [REMOTE]` / `pull REMOTE [LOCAL]` | rsync through the login node; needs `--cluster` (the filesystems differ). A git repo going to a path that does not exist yet is first `git clone`d there from its origin at the same commit, so only uncommitted/unpushed changes are copied, and files deleted from the source are deleted there too; when the clone fails, a fresh directory receives `.git` only if the source repo holds no URL credentials or HTTP extraheader (or the command runs on the login node). Existing destinations keep their own `.git`, including with `--no-exclude`. A path the cluster mounts too, at the same place, is left as it is. |
| `shell -j JOB` | Interactive shell, for humans only. Agents use `exec`. |

Commands are arguments after `--`, passed through exactly: spaces, quotes, `$` and `!` all survive.
For pipes, `&&` or redirection, wrap them in `bash -lc '...'`; that runs on the node, so `$VAR` expands there.

## Running work

- **Short commands (under ~8 min)** use `exec`. It streams output and returns the command's exit code.
- **Anything longer** (large builds, benchmarks, training, pulls of big images) uses `bg` + `wait`:
  ```bash
  $RG bg -j mycluster:202642 --name bench -C mycluster/src/myrepo -- python bench.py --iters 1000
  $RG wait -j mycluster:202642 bench    # repeat while it exits 205 (state=RUNNING)
  ```
  An `exec` that outlives the Bash tool's 10 minutes is killed, the remote command with it, and the output is lost.
- **Paths.** The OK line's `workdir=` is the job's state dir and default `-C`. A relative `-C`, `push` or `pull` path is under the cluster's work root (`WORK_ROOT`: `$HOME`, or the shared filesystem `auto` found); `push` prints where it put the code.
- **Docker.** `--image IMG` wraps CMD in `docker run --rm` with the GPUs, `--network host --ipc host`, the work root's filesystem and `$HOME` mounted, plus the manifest's `DOCKER_FLAGS`. The container runs as root. Containers are labelled with the job, and `release` removes them.
- **Multi-node.** Allocate with `-N 2` or more and use `exec --all` or `bg --all`: one copy per node, in parallel. Each copy gets `REMOTE_GPU_NODE_RANK`, `REMOTE_GPU_NNODES`, `REMOTE_GPU_NODES`, `REMOTE_GPU_JOB`, `MASTER_ADDR` (rank 0) and `MASTER_PORT` (29500), e.g. `torchrun --nnodes $REMOTE_GPU_NNODES --node-rank $REMOTE_GPU_NODE_RANK --master-addr $MASTER_ADDR ...` inside `bash -lc`. With several nodes, lines are prefixed `[node]`, stderr is merged into stdout, and the exit code is the first non-zero one.
- `-C` to a directory that doesn't exist exits 126.

A `WARN node=... pre-existing docker containers` line means another user's leftovers may hold
GPU memory. Check with `exec -- timeout 60 amd-smi metric --mem-usage` (`rocm-smi` can hang;
wrap it in `timeout`), and if the GPUs are busy, release and re-alloc with `--cluster X ... -x <node>`.

NFS caching: a file just written on one host can show up as missing on another for up to ~60 s.
Check fresh files on the host that wrote them, or poll.

## When things fail

All failures print `ERR code=<NAME> ...` on stderr. `exec` and `wait` return the remote
command's own exit code, so remote-gpu keeps 200–210 for its own errors:

| rc | code | what to do |
|---|---|---|
| 200 | USAGE / UNSUPPORTED | Bad flags, a bare job ID without `--cluster`, `-w` names a node not in the job, `--name` reuses a task name of the job (each name is used once per job), or `extend` where `EXTEND=0`. Read `msg`. |
| 201 | QUOTA_FULL | A per-user job limit (check `ls`, release what you don't need), or every team cap of `ACCOUNTS` is full: wait and retry, or ask about burst. |
| 202 | PENDING_TIMEOUT | Plain `alloc` didn't start within `--wait`; the job was cancelled. Use `--no-wait` + `ready`. |
| 203 | JOB_GONE | The job ended: time limit, idle release, preemption, or someone released it. `alloc` again, then re-run. |
| 204 | BAD_NODE | A node showed fewer than `GPUS_PER_NODE` usable GPUs, or could not see the state dir; the job was cancelled. Re-alloc with `--cluster X ... -x <node>` (from `ready`: tell the user the queued job was lost). |
| 205 | still running / pending | From `wait`/`ready`: call again. `JOB_PENDING` from `exec`/`bg`: the job hasn't started; use `ready`. |
| 206 | SUBMIT_FAILED / UNSCHEDULABLE / NODES_UNKNOWN / INSTALL_FAILED | The scheduler rejected the request or it can never start as given (fix it); the controller was busy (retry); or the shared filesystem is full (tell the user). |
| 207 | NO_TASK / TASK_START_FAILED | Wrong task name, or the holder didn't start the task on some node. `started_on=` lists nodes where it runs anyway; `wait` on such a task exits 207 once those end. |
| 208 | EXTEND_FAILED | Read the message. |
| 209 | RSYNC_FAILED | Read rsync's message. |
| 210 | NO_FREE_NODES | See "No free nodes" above. |
| 255 | ssh failure / PROBE_FAILED | Check `ssh <HOST> true` with the cluster's `HOST`; set `<NAME>_HOST` (see `HOST=` in its manifest) to pin another login node. Never blindly re-run a non-idempotent command; check `ls` / `logs` first. |

For cluster facts, read its `cluster.env`. The adapter contract is in
[`templates/slurm.sh`](templates/slurm.sh), with migration guidance in
[`references/add-cluster.md`](references/add-cluster.md). Scheduler hooks and site facts live
in `cluster.sh`; shared allocation policy, holder cleanup/heartbeat, task files, and sync rules
live in `scripts/remote-gpu`. `ALLOC_AGGRESSIVE` only controls whether account racing is the
default; `--aggressive` works with every adapter. Allocation options are parsed once, and
account/QOS selections are complete pairs from `ACCOUNTS` (burst is an explicit opt-in).
