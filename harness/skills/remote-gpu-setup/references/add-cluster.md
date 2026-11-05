# Adding a cluster

A cluster is a directory `~/.remote-gpu/clusters/<name>/` with two files: the manifest `cluster.env`
and the adapter `cluster.sh`. Make both from remote-gpu's `templates/`, using the user's answers and
what `scripts/probe-cluster` reads from the login node. They stay in your home, never in the repo,
because they name login hosts and accounts.

## 1. Probe

The only input is the ssh destination of the login node. Ask for it only if the user hasn't given it.

```bash
${CLAUDE_SKILL_DIR}/../remote-gpu/scripts/probe-cluster <host>
```

It is read-only and submits nothing. It prints `KEY=value` lines and `== section` blocks: scheduler
and version, whether a login shell is needed, partitions and nodes, one node's details, the user's
associations, QOS limits, preemption settings, and the network filesystems that hold a writable
`$USER` directory owned by the user. If ssh fails or asks for a password, stop and tell the user to
set up their key or `~/.ssh/config`. Don't edit either yourself.

## 2. Decide, and ask only what the probe leaves open

Take every value from the probe:

- the name is `CLUSTER_NAME`, else the login host's short name, lowercased to `[a-z][a-z0-9-]*`.
  It prefixes job handles (`<name>:ID`) and job names.
- the partition is `DEFAULT_PARTITION`, or the only GPU partition the user's associations reach.
- the accounts and QOS come from the user's associations, with the non-preemptible ones first.

Ask the user, in one AskUserQuestion, only what the probe can't decide: several GPU partitions or
accounts with nothing to choose between them, only preemptible QOS available, a home that is too
full with no other filesystem, or a scheduler that isn't Slurm or spur. Don't ask about anything
else. Site rules and the team's docker image are used only if the user gives them.

## 3. Write cluster.env

Copy `templates/cluster.env` and fill every key; the comment on each line says what it means.
Where the values come from:

| key | from |
|---|---|
| `HOST`, `LOGIN_HOSTS` | the user's host; a glob of `LOGIN_HOSTNAME` that also matches the other login nodes behind that host |
| `LOGIN_PROFILE` | `LOGIN_PROFILE_NEEDED`; also 1 if the site's docs say the scheduler's environment (e.g. controller address) comes from the login profile |
| `SCHEDULER`, `SCHEDULER_VERSION` | the same keys |
| `PARTITION`, `ACCOUNTS`, `BURST_QOS` | step 2. `ACCOUNTS` lists `ACCOUNT:QOS` pairs from the associations, in the order to try; `BURST_QOS` is a preemptible QOS the user can reach, else `''` |
| `MAX_TIME` | the QOS's MaxWall |
| `EXTEND` | 1 only if users may run `scontrol update TimeLimit`; on Slurm assume 0 |
| `PREEMPTION` | PreemptMode, PreemptExemptTime, and which QOS can preempt the user's |
| `NODE_NAMES`, `NODE_COUNT`, `CPUS_PER_NODE`, `MEM_PER_NODE_GB` | the partition's node list and count, `CPUTot`, `RealMemory` / 1024 |
| `GPU_MODEL`, `GPUS_PER_NODE` | `Gres=gpu:<model>:<n>`. spur lists one `gpu:<model>:1` per GPU, so count them |
| `GPU_ARCH` | from the model: MI355X/MI350X `gfx950`, MI325X/MI300X `gfx942`, MI250X `gfx90a`. Step 5 checks it |
| `WORK_ROOT` | `$HOME` if home has room and the nodes mount it. Else `auto`, which picks the filesystem with the most free space in the probe's last section. If the user wants a different one, ask them to create `<mount>/$USER` there or free space on the other; never write the path itself |
| `HOME_FULL` | 1 if home is over ~90% used (`HOME_USED`) |
| `DOCKER_FLAGS` | `--privileged` only if the site needs and allows it |

No absolute paths in either file: use `$HOME/...` or `WORK_ROOT=auto`. Check with
`grep -rnE '(^|[ =:"(])/[a-z]' ~/.remote-gpu/clusters/<name>/`; only `/dev`, `/sys`, the
shebang and awk regexes such as `/gpu:/` may match.

## 4. Write cluster.sh

Copy `templates/slurm.sh` and put the cluster name in its first line. Then:

- **Slurm**: change nothing and write `# Adapted: nothing` in the header, unless the probe shows a
  reason to change something.
- **spur, or a Slurm whose commands differ**: rewrite only the hooks that differ, keeping each
  hook's name, arguments and output.
- **Anything else** (PBS, Flux, Kubernetes): rewrite every hook for that scheduler, and tell the
  user it is untested beyond step 5.

List every changed hook, and why, under `Adapted:` in the header. The [template header](../../remote-gpu/templates/slurm.sh) is the contract:
what each hook takes, prints and returns. Keep it in the copy. `remote-gpu` refuses to load an
adapter that is missing a required hook, or `scheduler_extend` when `EXTEND=1`. Two rules are easy
to break: `scheduler_query` and `scheduler_jobs` must not start a step inside a job, and an unknown
job or controller error must return nonzero, never look like a finished job.

Only scheduler commands and site facts belong in the adapter. Allocation policy, waiting, health
checks, release and the holder loop live in the `remote-gpu` CLI. If you update an old adapter
that still has `cmd_*`, `R_*`, `HOLDER`, `SQUEUE` or `SCANCEL`, back it up, move its scheduler
commands and site facts into the hooks, and delete the rest. Running jobs keep their old holder.

## 5. Verify

```bash
RG=${CLAUDE_SKILL_DIR}/../remote-gpu/scripts/remote-gpu
$RG help | grep <name>                    # the cluster is listed
$RG --cluster <name> free; $RG --cluster <name> info; $RG --cluster <name> ls
```

These only read. Allocation and running on a node are tested only by the following, which holds a
node for a few minutes, so ask the user first:

```bash
$RG --cluster <name> alloc -t 0:20:00 --idle 10              # OK ... gpus=8/8 arch=...
$RG exec -j <name>:ID -- bash -c 'grep -h gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u'
$RG bg -j <name>:ID --name smoke -- bash -c 'sleep 5; echo done'
$RG wait -j <name>:ID smoke
$RG exec -j <name>:ID --image <an image the nodes have> -- rocm-smi --showproductname
$RG release -j <name>:ID
```

`gfx_target_version` 90500 is gfx950, 90402 is gfx942, 90010 is gfx90a; fix `GPU_ARCH` if it
differs. If all of this passed, set `VERIFIED` to today. Tell the user what passed and what was not tried.
