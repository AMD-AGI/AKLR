# Adding a cluster

The manifest `cluster.env` is remote-gpu's `templates/cluster.env` filled with the user's inputs
and what `$PROBE` reads from the login node. It stays in your home, never in the repo: it names
login hosts and accounts. The steps are the same for every type; this table is what a type changes.

| | slurm | spur | safe |
|---|---|---|---|
| inputs | login node | login node | SaFE URL, the user's API key, and a login host that reaches the URL and mounts the workspace's volume at the same path as its pods (usually the site's Slurm or spur login node) |
| `PARTITION` | `DEFAULT_PARTITION`, or the only GPU partition the user's associations reach | same | the workspace ID |
| `MAX_TIME` | the QOS's MaxWall | same (in minutes) | `24:00:00` unless the user gives a limit |
| `PREEMPTION` | PreemptMode, PreemptExemptTime, and which QOS can preempt the user's | same | the workspace's preempt flag |
| node facts | `Gres=gpu:<model>:<n>`, `CPUTot`, `RealMemory` / 1024 | same; one `gpu:<model>:1` per GPU, so count them | the workspace's node count and its flavor |
| `WORK_ROOT` | `$HOME` if home has room and the nodes mount it. Else `auto`, which picks the filesystem with the most free space in the probe's last section. If the user wants a different one, ask them to create `<mount>/$USER` there or free space on the other | same | `<the volume's mountPath>/$USER`; not `auto`, which could pick a filesystem the pods do not mount |

## 1. Probe

The type is an input like the rest: it is the user's to name. Ask only for the inputs the user
hasn't given. For safe, fix the cluster's name first (step 2): the key file and the probe take
it. A secret (the SaFE API key) comes through a secret prompt, never chat, and goes to the login
host only:

```bash
ssh <host> 'umask 077; mkdir -p ~/.remote-gpu && cat > ~/.remote-gpu/<name>.key' < <key file>
```

Never print it, and never write it into `cluster.env`, `cluster.sh` or the repo. Then run the
probe, which only reads:

```bash
$PROBE <host> <type>                # slurm, spur, or a scheduler with no file in schedulers/
$PROBE <host> safe <URL> <name>
```

Done when its partition or workspace list has rows. An empty list on slurm, spur or safe means
the type, the URL or the key is wrong, or safe's login host lacks curl or jq (`HAVE_curl`,
`HAVE_jq`): tell the user and stop.

## 2. Decide, and ask only what the probe leaves open

Take every value from the probe:

- the name is `CLUSTER_NAME`, else the login host's short name, lowercased to `[a-z][a-z0-9-]*`;
  a SaFE workspace on a site that already has a cluster gets `<site>-safe`. It prefixes job
  handles (`<name>:ID`) and job names.
- slurm, spur: the accounts and QOS come from the user's associations, the non-preemptible first,
  equals in the probe's order.
- safe: the workspace must mount a volume (`hostpath:/path`) that the probe also lists as a network
  filesystem of the login host; if none does, tell the user and stop. Its scopes give `SAFE_KIND`:
  `Train` means `PyTorchJob`; without it, `Infer` means `Deployment`, where an idle job keeps its
  node until `release` or `-t`. Tell the user when it is `Deployment`. A manifest holds one
  workspace, so each workspace the user takes is a cluster of its own; `~/.remote-gpu/<name>.key`
  of each is a link to the first's.

Ask the user, in one AskUserQuestion, only what the probe can't decide: several partitions,
accounts or workspaces with nothing to choose between them, only preemptible QOS available, a home
that is too full with no other filesystem, the image SaFE jobs run in, or a scheduler with no
file in `schedulers/`. Don't ask about anything else. Site rules and the team's docker image are
used only if the user gives them.

## 3. Write cluster.env

Copy `templates/cluster.env`, keep the block of the cluster's type and delete the other, and fill
every key but `VERIFIED`, which a passed smoke test sets; the comment on each line says what it means. The
table above gives the keys that depend on the type. The rest:

| key | from |
|---|---|
| `HOST`, `LOGIN_HOSTS` | the user's host; a glob of `LOGIN_HOSTNAME` that also matches the other login nodes behind that host |
| `LOGIN_PROFILE` | `LOGIN_PROFILE_NEEDED`; also 1 if the site's docs say the scheduler's environment (e.g. controller address) comes from the login profile |
| `SCHEDULER`, `SCHEDULER_VERSION` | the same keys of the probe (`''` for safe's version) |
| `ACCOUNTS`, `BURST_QOS` | `ACCOUNT:QOS` pairs in step 2's order; a preemptible QOS the user can reach, else `''` |
| `EXTEND` | 1 only if users may lengthen a running job; on Slurm assume 0 |
| `GPU_ARCH` | from the model: MI355X/MI350X `gfx950`, MI325X/MI300X `gfx942`, MI250X `gfx90a`. Step 5 checks it |
| `HOME_FULL` | 1 if home is over ~90% used (`HOME_USED`) |
| `DOCKER`, `DOCKER_FLAGS` | 1, and 0 only if the smoke test's `--image` step shows the nodes have no docker; `--privileged` only if the site needs and allows it |
| `SAFE_RESOURCES` | what one pod asks for beside all GPUs: well under the flavor's CPU and memory, since the node keeps some. The template's value fits an 8-GPU MI355X node |

No absolute paths: use `$HOME/...` or `WORK_ROOT=auto`; safe's `WORK_ROOT` is the one exception.
`grep -nE '(^|[ =:"(])/[a-z]' ~/.remote-gpu/clusters/<name>/cluster.env` must print nothing else.

## 4. Site overrides (usually none)

A `cluster.sh` is written only when step 5 shows the site differs from its type. Then
`~/.remote-gpu/clusters/<name>/cluster.sh` sets `CLUSTER_RLIB` to a library that redefines only
the hooks that differ, with a comment saying why:

```bash
read -r -d '' CLUSTER_RLIB <<'EOF'
scheduler_capacity() { ...; }   # this site has no team caps: free = idle nodes
EOF
```

It is loaded after the type's adapter, so its functions win; each must keep its hook's contract
in [schedulers/README.md](../../remote-gpu/schedulers/README.md). For a scheduler with no file in
`schedulers/` (PBS, Flux), `cluster.sh` holds every hook; tell the user it is untested beyond
step 5.

## 5. Verify

```bash
$RG help | grep <name>                    # the cluster is listed
$RG --cluster <name> free; $RG --cluster <name> info; $RG --cluster <name> ls
```

These only read, and each must exit 0. Allocation and running on a node are tested only by the
smoke test, run only on the user's yes:

```bash
$RG --cluster <name> alloc -t 0:20:00 --idle 10              # OK ... gpus=N/N arch=...
$RG exec -j <name>:ID -- bash -c 'grep -h gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u'
$RG bg -j <name>:ID --name smoke -- bash -c 'sleep 5; echo done'
$RG wait -j <name>:ID smoke
$RG exec -j <name>:ID --image <an image the nodes have> -- rocm-smi --showproductname
$RG release -j <name>:ID                                     # also after a failure
```

It passes when every command exits 0; `ERR code=UNSUPPORTED` (rc 200) is the answer of a type
that lacks the command, not a failure. `gfx_target_version` 90500 is gfx950, 90402 is gfx942,
90010 is gfx90a; fix `GPU_ARCH` if it differs.

## Older cluster directories

A directory made before adapters moved into `schedulers/` has a `cluster.sh` that defines every
hook although `schedulers/<SCHEDULER>.sh` exists. It still works: it is loaded after the type's
adapter and redefines every hook, so fixes to `schedulers/` do not reach it. On `update`, compare
its library with `schedulers/<SCHEDULER>.sh`; when nothing site-specific differs, put this line
first in the file, so this CLI uses the repo's adapter while older checkouts of remote-gpu, which
need the whole file, keep working:

```bash
[[ -z ${SCHEDULERS:-} ]] || return 0   # this remote-gpu has schedulers/<type>.sh; older ones read on
```

Delete the file once no older checkout or plugin copy is in use. If it still has `cmd_*`, `R_*`,
`HOLDER`, `SQUEUE` or `SCANCEL`, back it up and delete it; running jobs keep their old holder.
