# Scheduler adapters

One adapter per scheduler type lives here as `<SCHEDULER>.sh`; the manifest's SCHEDULER
picks it. A file is sourced locally and adds to CLUSTER_RLIB, a Bash library evaluated remotely
after the manifest and shared library (C=cluster, REL=work root, B=state dir). A later library
redefines functions of an earlier one: spur.sh is slurm.sh plus the hooks spur does differently,
and a cluster's optional ~/.remote-gpu/clusters/<name>/cluster.sh redefines only what its site
does differently (or holds every hook, for a scheduler with no file here).
Before its library a file may set the capabilities its type lacks; the CLI refuses those
commands before connecting: EXTEND (extend), DOCKER (--image), MULTI_NODE (alloc -N 2+),
NODE_SHELL (shell), all 0/1, and IDLE_RELEASE=0 when an idle job keeps its nodes.

## The hook contract

```text
CLUSTER_RLIB contains these functions; stdout is data, stderr diagnostics.
All hooks return 0 on success, nonzero on failure, except where specified below.
  scheduler_submit N TIME NAME PARTITION ACCOUNT QOS HOLDER IDLE [OPTIONS...]
    Submit whole nodes/all GPUs, no requeue; print one numeric job ID. HOLDER is called with
    IDLE HOLDER B C. Return 201 for a per-user submit limit, 206 for another rejection.
  scheduler_query JOB
    Print one pipe-delimited record (empty unknown fields, no embedded pipes/newlines):
    STATE|NODES|TIMELEFT|REASON|START|PARTITION|ACCOUNT|QOS|TIMELIMIT|USER|NAME|ELAPSED|END|NNODES
    NODES is a comma-separated list in rank order; times are [D-]HH:MM:SS (or MM:SS),
    START is a start estimate, END is epoch seconds; NNODES includes pending requests. STATE uses Slurm's normalized states;
    terminal states must be explicit. Return nonzero for unknown jobs/controller failures.
  scheduler_jobs
    List this user's jobs, one JOB|<query record> per line; empty output is success.
    Neither it nor scheduler_query may start a step inside a job.
  scheduler_cancel JOB [pending]
    Cancel JOB; with pending, cancel atomically only if still pending (never a running job).
  scheduler_run JOB NODE MODE CMD...
    Run CMD on NODE, all GPUs, overlapping other steps, return CMD's rc and output.
    MODE=batch consumes no stdin; MODE=interactive uses the caller's tty. ONTO, when set,
    bounds runtime in seconds. NODE is always a member of the job, N0 is its rank-zero node.
  scheduler_capacity [info]
    No args: OK cluster=C free=N account=A qos=Q [burst=N burst_account=A] ...;
    account/qos identify an ACCOUNTS pair with most room. info: human-readable site details.
  scheduler_extend JOB TOTAL_TIME  (required only with EXTEND=1)
    Change the total time limit; no stdout required.
  scheduler_holder IDLE SELF B C
    Bootstrap exactly one holder per node; return there with J=job ID, H=rank's node name.
    A batch parent that launches steps must wait for them and exit, never run the loop too.
Site facts in CLUSTER_RLIB: DOCKER_X (docker argv), NODE_ENV (KEY=VALUE argv), LATE_HINT.
The shared CLI owns allocation policy, waiting/health, release, task and holder lifecycle.
```

## spur.sh: why its hooks differ from Slurm's

- slurm_ended: no `sacct -X`; an ended job's state comes from `spur history`.
- scheduler_run: `srun` steps don't work; rank 0 runs through `spur exec`, the others through
  pty-only `spur run` (CRLF, stderr merged). Both only ask the controller to start CMD, so
  killing them leaves CMD running: CMD's environment carries a tag, and when the caller is killed
  or ONTO runs out, every process on the node with that tag gets TERM.
- scheduler_holder: spur runs the batch script on every node, so there is no step to re-launch;
  the holder cancels the job when it exits (idle, time limit), as spur may not end it then.
- scheduler_capacity, info: team headroom from `scontrol show assoc_mgr` and `spur accounts
  show qos`; a site without GrpTRES team caps redefines scheduler_capacity in its cluster.sh.

## safe.sh: what stands for what on Primus-SaFE

```text
  login node   HOST: any host that mounts the volume the workspace mounts into its pods, at the
               same path, and reaches SAFE_URL. The state dir is on that volume.
  job          one SaFE workload of SAFE_KIND in the workspace PARTITION, one whole node. Its
               entry point is the shared holder, so the workload ends when the holder does. A
               Deployment would restart it, so there the entry point sleeps on after the holder
               and only release or the time limit ends the job (IDLE_RELEASE=0).
               The job ID is a number made here; $B/safe/ID.id holds the workload ID and name.
  node         the pod. Nothing can exec into it from the login node, so scheduler_run hands
               the command to the holder as a spawn request and reads its output and exit code
               back from files in the state dir.
```
