# Adding a cluster

remote-gpu knows a cluster once `~/.remote-gpu/clusters/<name>/` holds its manifest (`cluster.env`) and its
adapter (`cluster.sh`). You write both from `templates/`, with what the user tells you and what
`scripts/probe-cluster` reads from the login node. They live in your home, not in the repo: they name
login hosts and accounts.

## 1. Ask the user

Ask only what the cluster cannot tell you, in one AskUserQuestion:

- A short lowercase name (`[a-z][a-z0-9-]*`): it prefixes job handles (`<name>:ID`) and job names.
- The login node's ssh destination. It must work without a prompt: `ssh -o BatchMode=yes <host> true`.
  If it does not, stop and tell the user to set up their key or `~/.ssh/config` entry; don't edit
  either yourself.
- Which partition, account and QOS they are meant to use, if they know (the probe lists what they
  may use), and whether a preemptible QOS is acceptable.
- Anything they know the probe can't: the filesystem their work belongs on when there are several,
  a docker image the team uses, rules of the site.

## 2. Probe

```bash
${CLAUDE_SKILL_DIR}/scripts/probe-cluster <host>
```

Read-only; it submits nothing. It prints `KEY=value` facts and `== section` blocks: the scheduler
and its version, whether its commands need a login shell, partitions with their nodes, GRES, CPUs
and memory, one node's details, the user's associations, QOS limits, preemption settings, and the
network filesystems holding a writable directory named after the user.

## 3. Write ~/.remote-gpu/clusters/<name>/cluster.env

Copy `templates/cluster.env` and fill every key; each line's comment says what it means. From the
probe:

| key | from |
|---|---|
| `HOST`, `LOGIN_HOSTS` | the host the user gave; a glob of `LOGIN_HOSTNAME` that covers the other login nodes behind the same name |
| `LOGIN_PROFILE` | `LOGIN_PROFILE_NEEDED`; also 1 when the site's docs say the scheduler's environment (a controller address) comes from the login profile |
| `SCHEDULER`, `SCHEDULER_VERSION` | the same keys |
| `PARTITION`, `ACCOUNTS`, `BURST_QOS` | the user's answer, else `''` (the defaults); `ACCOUNTS` is `ACCOUNT:QOS` pairs from the associations, in the order to try |
| `MAX_TIME`, `EXTEND`, `PREEMPTION` | the QOS's MaxWall; whether users may run `scontrol update TimeLimit` (assume 0 on Slurm); PreemptMode, PreemptExemptTime and which QOS preempt the user's |
| `NODE_NAMES`, `NODE_COUNT`, `CPUS_PER_NODE`, `MEM_PER_NODE_GB` | the partition's node list and counts, `CPUTot`, `RealMemory` / 1024 |
| `GPU_MODEL`, `GPUS_PER_NODE` | `Gres=gpu:<model>:<n>` (spur lists one `gpu:<model>:1` per GPU: count them) |
| `GPU_ARCH` | from the model: MI355X and MI350X `gfx950`, MI325X and MI300X `gfx942`, MI250X `gfx90a`. Confirm it in step 5 |
| `WORK_ROOT` | `$HOME` when the home filesystem has room and is mounted on the nodes; else `auto`, which picks the network filesystem with the most free space holding a writable `$USER` dir (the probe's last section, top entry). If the one the user named is not what `auto` would pick, ask them to create `<mount>/$USER` there or free the other; never write its path into the manifest |
| `HOME_FULL` | 1 when the home filesystem is over ~90% used |
| `DOCKER_FLAGS` | `--privileged` only where the site allows and needs it |

No absolute path goes into a cluster's files: write `$HOME/...` or leave it to
`WORK_ROOT=auto`. Check with `grep -rnE '(^|[ =:"(])/[a-z]' ~/.remote-gpu/clusters/<name>/` (only `/dev`, `/sys`
and the shebang may show up).

## 4. Write ~/.remote-gpu/clusters/<name>/cluster.sh

- **Slurm** (`SCHEDULER=slurm`): copy `templates/slurm.sh`, put the name into its first line and
  `# Adapted: nothing` into the header, and change nothing else. If the probe shows a reason to adapt,
  each change goes into the header's `Adapted:` list.
- **spur, or a Slurm whose commands differ**: copy `templates/slurm.sh` and rewrite only the
  `# ADAPT:` blocks that differ, keeping each block's name and contract. List each block you changed, and why, in the
  header's `Adapted:` list.
- **Anything else** (PBS, Flux, Kubernetes): the template does not fit. Keep its structure and its
  contract below, write each block for that scheduler, and tell the user it is untested beyond step 5.

What `scripts/remote-gpu` needs from `cluster.sh`:

- `CLUSTER_RLIB`, run on the login node before every remote call, with `$C`, `$B` (state dir),
  `$REL` (work root) and the manifest set: `SQUEUE` and `SCANCEL` (the queue must take
  `-j ID -h -o FMT` and `-u USER -h -o FMT` with Slurm's format codes `%i %j %T %N %M %L %l %P %a %q
  %D %r %S %u`), `LATE_HINT`, `DOCKER_X`, `NODE_ENV`, and functions `jfinal JOB` (final state),
  `jstart JOB`, `nodes_of JOB` (rank order), `on_node JOB NODE CMD...` (rc and output of CMD there).
- `HOLDER`: the batch script, called with `IDLE SELF B C`. On every node of the job it must keep the
  allocation, start the bg tasks queued in `$B/jobs/<J>/spawn/` as its own children, end the job
  when `$B/jobs/<J>/heartbeat` is older than IDLE minutes, and remove the job's containers (label
  `<C>.job=<J>`) and tasks when the job ends.
- `R_FREE` (prints `OK cluster=... free=N`, plus `account=` for the account with the most room and
  `burst=` when there is a `BURST_QOS`), `R_RELEASE`, `R_INFO`; and `cmd_alloc` (prints
  `SUBMITTED job=<C>:<ID>` for every job it submits; `--preemptible`, and `--aggressive` when
  `ALLOC_AGGRESSIVE=1`), `cmd_ready`, `cmd_extend`, `shell_cmd`. `R_AWAIT`, `R_ALLOC` and `R_READY` are
  the template's own helpers for those commands.

## 5. Verify

```bash
RG=${CLAUDE_SKILL_DIR}/scripts/remote-gpu
$RG help | tail -5                        # the cluster is listed
$RG --cluster <name> free; $RG --cluster <name> info; $RG --cluster <name> ls
```

Then, with the user's go-ahead (it takes a node for a few minutes):

```bash
$RG --cluster <name> alloc -t 0:20:00 --idle 10              # OK ... gpus=8/8 arch=...
$RG exec -j <name>:ID -- bash -c 'grep -h gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u'
$RG bg -j <name>:ID --name smoke -- bash -c 'sleep 5; echo done'
$RG wait -j <name>:ID smoke
$RG exec -j <name>:ID --image <an image the nodes have> -- rocm-smi --showproductname
$RG release -j <name>:ID
```

`gfx_target_version 90500` is gfx950, `90402` gfx942, `90010` gfx90a: fix `GPU_ARCH` if it
disagrees. Then set `VERIFIED` to today and tell the user what passed and what was not tried.
