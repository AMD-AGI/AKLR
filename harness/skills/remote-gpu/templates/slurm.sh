# cluster.sh — remote-gpu backend for <NAME>, a Slurm cluster. Made from templates/slurm.sh.
# shellcheck shell=bash disable=SC2016,SC2034  # sourced by scripts/remote-gpu, which runs the single-quoted bodies remotely
# Sourced by scripts/remote-gpu after this cluster's cluster.env, whose values it reads. Every
# `# ADAPT:` comment marks a place a cluster may need changed; leave the rest as it is.
# Adapted: <none, or one line per ADAPT place changed and why>

read -r -d '' CLUSTER_RLIB <<'EOF'
# ADAPT: the queue and cancel commands, and what to do when a job nears its time limit.
SQUEUE=squeue SCANCEL=scancel
LATE_HINT="save results now; the job ends at its time limit"
# Docker gets the filesystem the work root is on, and $HOME unless it is on that one too.
MNT=$(df --output=target "$REL" 2>/dev/null | tail -1)
DOCKER_X=($DOCKER_FLAGS -v "$MNT:$MNT"); [[ $HOME == "$MNT"/* ]] || DOCKER_X+=(-v "$HOME:$HOME")
NODE_ENV=(); [[ $HOME_FULL == 1 ]] && NODE_ENV=(XDG_CACHE_HOME=$B/.cache)
jfinal() { sacct -j "$1" -X -n -o State%20 2>/dev/null | awk 'NF {print $1; exit}'; }
# ADAPT: the scheduler's start estimate for a pending job, '' if it has none.
jstart() { squeue --start -j "$1" -h -o '%S' 2>/dev/null | head -1 | tr -d ' '; }
# ADAPT: the node list, in rank order. Slurm's rank 0 is the first entry (the batch host).
nodes_of() { local nl; nl=$(jfield "$1" '%N'); [[ -n $nl ]] && scontrol show hostnames "$nl"; }
# ADAPT: on_node JOB NODE CMD...: run CMD on NODE of JOB as an overlapping step (all of the job's
# GPUs, no CPU pinning), streaming output and returning its rc. Set ONTO=<seconds> to bound it.
on_node() {
  local j=$1 n=$2 to=(); shift 2
  [[ -n ${ONTO:-} ]] && to=(timeout "$ONTO")
  "${to[@]}" srun --jobid "$j" --overlap -w "$n" -N1 -n1 --cpu-bind=none --quiet "$@" </dev/null \
    2> >(grep --line-buffered -vE '^srun: (error: .*: task [0-9]+: Exited with exit code|error: Timed out waiting for job step|warning)' >&2)
}
EOF

# ADAPT: the holder batch script. Slurm runs the batch script on the first node only, so it
# re-launches itself as one overlapping step per node. That step owns all of the job's GPUs
# (devices are cgroup-constrained per step), keeps the allocation alive, starts bg tasks, and
# cleans up this job's containers and bg tasks BEFORE the job ends: on `remote-gpu release`
# (release file), shortly before the time limit, and on idle release. A signal from scancel or
# preemption is only a best-effort fallback (srun turns it into SIGKILL for the step), so
# leftovers from a job that died that way are reaped by the next holder of this user on the node.
# Arguments: IDLE SELF B C.
read -r -d '' HOLDER <<'EOF'
#!/bin/bash
{ # braces: bash parses the whole script before running it, so it never re-reads this NFS file
J=$SLURM_JOB_ID; IDLE=${1:-60}; SELF=$2; B=$3; C=$4; D=$B/jobs/$J
if [ -z "${REMOTE_GPU_HOLDER_STEP:-}" ]; then   # batch step on the first node
  export REMOTE_GPU_HOLDER_STEP=1
  srun --overlap --ntasks-per-node=1 --cpu-bind=none --kill-on-bad-exit=0 bash "$SELF" "$IDLE" "$SELF" "$B" "$C"
  exit
fi
H=${SLURMD_NODENAME:-$(hostname -s)}   # must match the names in the job's node list
mkdir -p "$D/tasks" "$D/spawn"; touch "$D/heartbeat"
exec >>"$B/logs/$J.$H.log" 2>&1   # one log per node (the -o file is shared)
# cleanup WHY: stop this job's containers and bg tasks on this node, giving them ~20 s to exit
# cleanly (checkpoint), then mark the node clean for `remote-gpu release`.
cleanup() {
  local ids p n pid pg=() k a
  [ -e "$D/cleaned.$H" ] && return
  for p in "$D"/tasks/*.pid; do
    [ -e "$p" ] && [ ! -e "${p%.pid}.rc" ] || continue
    read -r n pid <"$p"; [ "$n" = "$H" ] && kill -TERM -- "-$pid" 2>/dev/null && pg+=("$pid")
  done
  ids=$(docker ps -aq --filter "label=$C.job=$J" 2>/dev/null); ds=
  [ -n "$ids" ] && { timeout 30 docker stop -t 20 $ids >/dev/null 2>&1 & ds=$!; }   # in parallel with the task wait
  for k in $(seq 20); do
    a=0; for pid in "${pg[@]}"; do kill -0 -- "-$pid" 2>/dev/null && a=1; done
    [ $a = 0 ] && break; sleep 1
  done
  for pid in "${pg[@]}"; do kill -KILL -- "-$pid" 2>/dev/null; done
  [ -n "$ds" ] && wait $ds
  [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  echo "$(date +%T) $H cleanup ($1): containers=[$(echo $ids)] tasks=${#pg[@]}"
  touch "$D/cleaned.$H"
}
# A reused pid only delays the reaping (until job cleanup), it never removes a live container.
# Containers whose `docker run` client died (caller interrupted, ssh dropped) keep running
# under dockerd with all GPUs. Every remote-gpu container carries its client's pid; reap orphans.
reap_orphans() {
  docker ps --filter "label=$C.job=$J" --format "{{.ID}} {{.Label \"$C.pid\"}}" 2>/dev/null |
    while read -r c p; do
      [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null && docker rm -f "$c" >/dev/null 2>&1 && echo "$(date +%T) reaped orphan container $c"
    done
}
# Containers of this user's earlier jobs that ended without cleanup (preemption, node failure).
reap_stale() {
  local c j s
  docker ps -a --filter "label=$C.user=$USER" --format "{{.ID}} {{.Label \"$C.job\"}}" 2>/dev/null |
    while read -r c j; do
      [ "$j" = "$J" ] && continue
      s=$(squeue -h -j "$j" -o %T 2>&1)   # unknown, or ambiguous (controller hiccup): keep it
      case $s in *"Invalid job id"* | COMPLETED | CANCELLED* | FAILED | TIMEOUT | PREEMPTED | NODE_FAIL | OUT_OF_MEMORY | BOOT_FAIL | DEADLINE)
        docker rm -f "$c" >/dev/null 2>&1 && echo "$(date +%T) removed stale container $c of job $j ($s)" ;;
      esac
    done
}
# endtime: the job's end as epoch seconds, or fail. Validate: `date -d ""` is today's midnight,
# and treating that as the end time would release the job at once.
endtime() {
  local s; s=$(squeue -h -j "$J" -o %e 2>/dev/null) || return 1
  [[ $s =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+$ ]] || return 1
  date -d "$s" +%s
}
# bg tasks are started here, as children of the holder step: processes launched through a
# login-node `srun` die when that step ends. Requests: $D/spawn/<host>.<task>.<rank> =
# NUL-separated "LOGBASE RUNNER CMD ARGS...".
spawn() {
  local r a F
  for r in "$D/spawn/$H".*; do
    [ -e "$r" ] || continue
    mapfile -d '' a <"$r"; rm -f "$r"; F=${a[0]}
    [ -f "${a[1]}" ] || { echo "$C: runner ${a[1]} missing" >"$F.log"; echo 127 >"$F.rc"; echo "$H 0" >"$F.pid"; continue; }
    setsid nohup bash "${a[1]}" "$D" "$F" "${a[@]:2}" >"$F.log" 2>&1 </dev/null &
    echo "$H $!" >"$F.pid.tmp"; mv -f "$F.pid.tmp" "$F.pid"
  done
}
trap 'cleanup signal; exit 0' TERM INT HUP USR1
echo "holder up on $H idle_release=${IDLE}m"
reap_stale; END=$(endtime) || END=${SLURM_JOB_END_TIME:-}; last=0
while :; do
  sleep 2 & wait $!
  spawn
  if [ -e "$D/release" ]; then cleanup release; exit 0; fi   # remote-gpu release cancels the job after this
  [ $SECONDS -ge $((last + 60)) ] || continue
  last=$SECONDS
  reap_orphans
  e=$(endtime) && [[ $e =~ ^[0-9]+$ ]] && END=$e   # the limit can change; keep the last good value on a hiccup
  if [[ $END =~ ^[0-9]+$ ]] && [ "$(date +%s)" -ge $((END - 90)) ]; then
    cleanup time-limit; exit 0   # every holder exits, so the step and then the job end
  fi
  [ "$IDLE" -gt 0 ] || continue
  m=$(stat -c %Y "$D/heartbeat" 2>/dev/null) || { touch "$D/heartbeat"; continue; }   # NFS hiccup != idle
  age=$(( $(date +%s) - m ))
  if [ "$age" -gt $((IDLE * 60)) ]; then
    echo "$(date) idle ${age}s > ${IDLE}m: releasing job $J"
    # No scancel: on a multi-node job it would SIGKILL holders still cleaning up. Every holder
    # sees the same stale heartbeat and exits, so the step and then the job end.
    cleanup idle; exit 0
  fi
done
exit
}
EOF

# Shared by alloc and ready: wait for a submitted job to start, then health-check its nodes.
read -r -d '' R_AWAIT <<'EOF'
# Pending reasons that never clear on their own: fail fast instead of waiting.
UNSCHEDULABLE='^(PartitionTimeLimit|PartitionNodeLimit|PartitionConfig|InvalidQOS|InvalidAccount|AccountNotAllowed|QOSNotAllowed|BadConstraints|QOSMaxWallDurationPerJobLimit|QOSMaxNodePerJobLimit|AssocMaxWallDurationPerJobLimit|AssocMaxNodesPerJobLimit)'
# Your own per-user limits (as opposed to QOSGrp*/AssocGrp*, which are shared caps).
PERUSER='^(QOSMax[A-Za-z]*PerUser|AssocMax)'
# await JOB WAIT ONTIMEOUT: returns 0 once RUNNING. ONTIMEOUT=cancel withdraws the job and fails;
# ONTIMEOUT=keep leaves it queued and exits 205.
await() {
  local J=$1 deadline=$((SECONDS + $2)) S reason st f
  while :; do
    IFS='|' read -r S reason < <(jfield "$J" '%T|%r')
    case $S in
      RUNNING) return 0 ;;
      CONFIGURING) ;;   # the prolog may run health checks on every node
      '') f=$(jfinal "$J"); [[ -n $f && $f != PENDING && $f != RUNNING ]] && err JOB_GONE "job=$C:$J state=$f" ;;   # else a controller hiccup
      PENDING)
        if [[ $reason =~ $UNSCHEDULABLE ]]; then
          $SCANCEL "$J" >/dev/null 2>&1
          err UNSCHEDULABLE "job=$C:$J(cancelled) reason=\"$(q "$reason")\" hint=\"the request can never run as given (time, node count, qos, -w); fix it and alloc again\""
        fi
        if ((SECONDS >= deadline)); then
          st=$(jstart "$J")
          if [[ $3 == keep ]]; then
            echo "PENDING job=$C:$J reason=\"$(q "$reason")\" start_estimate=${st:-unknown}"
            echo "next: remote-gpu ready -j $C:$J    |    give up: remote-gpu release -j $C:$J" >&2; exit 205
          fi
          withdraw "$J" && return 0
          [[ $reason =~ $PERUSER ]] &&
            err QUOTA_FULL "job=$C:$J(cancelled) reason=\"$(q "$reason")\" hint=\"you hit a per-user limit; release jobs you no longer need (remote-gpu ls)\""
          err PENDING_TIMEOUT "job=$C:$J(cancelled) reason=\"$(q "$reason")\" start_estimate=${st:-unknown} waited=$2s hint=\"cluster busy: use alloc --no-wait, then remote-gpu ready\""
        fi ;;
      *) $SCANCEL "$J" >/dev/null 2>&1; err JOB_GONE "job=$C:$J state=${S} reason=\"$(q "$reason")\" log=$B/logs/$J.out" ;;
    esac
    if ((SECONDS >= deadline + 120)); then   # stuck CONFIGURING, or the queue silent
      [[ $3 == keep ]] && { echo "PENDING job=$C:$J state=${S:-unknown}"; exit 205; }
      $SCANCEL "$J" >/dev/null 2>&1; err PENDING_TIMEOUT "job=$C:$J(cancelled) state=${S:-unknown}"
    fi
    sleep 5
  done
}
# GPU health check, run as a job step: that is what exec/bg see. GPUs = min(GPU agents in KFD
# sysfs, render nodes the step can open): a device cgroup blocks /dev/dri/* for a step without
# the GPUs while KFD sysfs still lists them all, and rocm-smi can hang. The node must also see
# the state dir. Sets NL (nodes) and N0 (rank 0) on success.
check_nodes() {
  local J=$1 n gpus foreign seen tmp
  NL=($(nodes_of "$J")); N0=${NL[0]:-}
  if ((${#NL[@]} == 0)); then
    [[ $(jfield "$J" '%T') == RUNNING ]] && err NODES_UNKNOWN "job=$C:$J msg=\"cannot resolve the job's nodes (controller busy?)\" hint=\"retry: remote-gpu ready -j $C:$J\""
    err JOB_GONE "job=$C:$J state=$(jfinal "$J")"
  fi
  local pids=(); tmp=$(mktemp -d)
  for n in "${NL[@]}"; do   # all nodes in parallel: at most ~2 min whatever -N is
    ONTO=120 on_node "$J" "$n" bash -c '
      r=0; for d in /dev/dri/renderD*; do { exec 3<>"$d"; } 2>/dev/null && r=$((r + 1)); exec 3>&-; done
      g=$(grep -l "^gfx_target_version [1-9]" /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l)
      echo $((g < r ? g : r)); timeout 20 docker ps -q --filter "label=$0.job=$1" 2>/dev/null | wc -l
      timeout 20 docker ps -q 2>/dev/null | wc -l; [ -d "$2" ] && echo 1 || echo 0' "$C" "$J" "$B" 2>/dev/null |
      tr '\n' ' ' | awk '{print $1, $3 - $2, $4}' >"$tmp/$n" & pids+=($!)
  done
  wait "${pids[@]}"   # not a bare `wait`: that would also wait for remote()'s stdin watchdog
  for n in "${NL[@]}"; do
    read -r gpus foreign seen <"$tmp/$n"
    if ((${gpus:-0} < GPUS_PER_NODE || ${seen:-0} == 0)); then
      rm -rf "$tmp"
      [[ $(jfield "$J" '%T') == RUNNING ]] || err JOB_GONE "job=$C:$J state=$(jfinal "$J") msg=\"ended during the health check\""
      $SCANCEL "$J" >/dev/null 2>&1
      err BAD_NODE "job=$C:$J(cancelled) node=$n gpus=${gpus:-0}/$GPUS_PER_NODE sees_workdir=${seen:-0} hint=\"retry: remote-gpu --cluster $C alloc ... -x $n\""
    fi
    ((${foreign:-0} > 0)) && warn "node=$n has $foreign pre-existing docker containers (not ours; GPUs may be busy)"
  done
  rm -rf "$tmp"
}
ok_line() {
  local J=$1 P A Q T
  IFS='|' read -r P A Q T < <(jfield "$J" '%P|%a|%q|%l')
  echo "OK cluster=$C job=$C:$J nodes=$(IFS=,; echo "${NL[*]}") partition=$P account=$A qos=$Q time=$T workdir=$B gpus=$GPUS_PER_NODE/$GPUS_PER_NODE arch=$GPU_ARCH"
  [[ -n $BURST_QOS && $Q == "$BURST_QOS" ]] && warn "job=$C:$J runs on $BURST_QOS and can be preempted (cancelled) at any time"
  echo "next: remote-gpu exec -j $C:$J -- hostname    |    release when done: remote-gpu release -j $C:$J" >&2
}
EOF

# ADAPT: the submit (R_ALLOC and cmd_alloc). Whole nodes with all their GPUs; without
# --gpus-per-node a job gets none.
read -r -d '' R_ALLOC <<'EOF'
N=$1 T=$2 IDLE=$3 WAIT=$4 NOWAIT=$5 NAME=$6; shift 6; X=("$@")
mkdir -p "$B/logs"
# --no-requeue: QOS preemption would otherwise requeue the holder and restart it hours later.
out=$(sbatch --parsable -J "$NAME" -N "$N" --gpus-per-node "$GPUS_PER_NODE" --exclusive --mem=0 --no-requeue -t "$T" \
      -o "$B/logs/%j.out" -e "$B/logs/%j.err" "${X[@]}" "$HOLDER_SH" "$IDLE" "$HOLDER_SH" "$B" "$C" 2>&1)
J=$(grep -Eo '^[0-9]+(;|$)' <<<"$out" | tail -1); J=${J%;}
if ! [[ $J =~ ^[0-9]+$ ]]; then
  [[ $out =~ (QOSMax|AssocMax)[A-Za-z]*(PerUser|Submit)|MaxSubmit ]] &&
    err QUOTA_FULL "msg=\"$(q "$out")\" hint=\"you hit a per-user job limit; release jobs you no longer need (remote-gpu ls)\""
  err SUBMIT_FAILED "msg=\"$(q "$out")\""
fi
mkdir -p "$B/jobs/$J/tasks"; touch "$B/jobs/$J/heartbeat"
echo "SUBMITTED job=$C:$J nodes=$N time=$T" >&2
((NOWAIT)) && { echo "OK cluster=$C job=$C:$J state=PENDING start_estimate=$(jstart "$J") name=$NAME workdir=$B"; echo "next: remote-gpu ready -j $C:$J" >&2; exit 0; }
await "$J" "$WAIT" cancel
check_nodes "$J"; ok_line "$J"
EOF

read -r -d '' R_READY <<'EOF'
J=$1
await "$J" "$2" keep
mkdir -p "$B/jobs/$J/tasks"; touch "$B/jobs/$J/heartbeat"
check_nodes "$J"; ok_line "$J"
EOF

# The manifest's PARTITION and first ACCOUNTS pair are the defaults; -p/-A/-q override them. remote-gpu
# passes -A itself when R_FREE prints account=.
cmd_alloc() {
  local n=1 t=4:00:00 name="$CLUSTER-${REMOTE_GPU_SESSION:-$(hostname -s | cut -c1-12)}-$RANDOM" idle=60 wait=300 nowait=0 extra=()
  local first=${ACCOUNTS%% *} p=$PARTITION a q qset=0 pr
  a=${first%%:*} q=${first#*:}
  while (($#)); do case $1 in
    -N | --nodes) need2 "$@"; n=$2; shift 2 ;;
    -t | --time) need2 "$@"; t=$2; shift 2 ;;
    -J | --name) need2 "$@"; [[ $2 =~ ^[A-Za-z0-9_.-]+$ ]] || die "alloc: -J label must match [A-Za-z0-9_.-]+"; name="$CLUSTER-${REMOTE_GPU_SESSION:+$REMOTE_GPU_SESSION-}$2"; shift 2 ;;
    --idle) need2 "$@"; idle=$2; shift 2 ;;
    --wait) need2 "$@"; wait=$2; shift 2 ;;
    --no-wait) nowait=1; shift ;;
    -p | --partition) need2 "$@"; p=$2; shift 2 ;;
    -A | --account) need2 "$@"; a=$2; shift 2 ;;
    -q | --qos) need2 "$@"; q=$2 qset=1; shift 2 ;;
    --preemptible) [[ -n $BURST_QOS ]] || die "alloc: $CLUSTER has no preemptible BURST_QOS"; q=$BURST_QOS qset=1; shift ;;
    -w | --nodelist | -x | --exclude | -C | --constraint | --reservation)
      need2 "$@"; extra+=("$1" "$2"); shift 2 ;;
    *) die "alloc: unknown option $1" ;;
  esac; done
  [[ $n =~ ^[0-9]+$ && $idle =~ ^[0-9]+$ && $wait =~ ^[0-9]+$ ]] || die "alloc: -N/--idle/--wait must be integers"
  if ((!qset)); then   # the QOS paired with the account in ACCOUNTS; none: the account's default
    q=''; for pr in $ACCOUNTS; do [[ ${pr%%:*} == "$a" ]] && q=${pr#*:}; done
  fi
  [[ -n $p ]] && extra+=(-p "$p"); [[ -n $a ]] && extra+=(-A "$a"); [[ -n $q ]] && extra+=(-q "$q")
  remote "$(with_script HOLDER_SH holder "$HOLDER")
$R_AWAIT
$R_ALLOC" "$n" "$t" "$idle" "$wait" "$nowait" "$name" "${extra[@]}"
}

cmd_ready() {
  local wait=420
  while (($#)); do case $1 in -j | --job) need2 "$@"; JOB=$2; shift 2 ;; --wait) need2 "$@"; wait=$2; shift 2 ;; *) die "ready: unknown option $1" ;; esac; done
  need_job; [[ $wait =~ ^[0-9]+$ ]] || die "ready: bad --wait"
  remote "$R_AWAIT
$R_READY" "$JOB" "$wait"
}

cmd_extend() {
  local t=''
  [[ $EXTEND == 1 ]] || err UNSUPPORTED "msg=\"users cannot lengthen a running job on $CLUSTER\" hint=\"ask for enough -t at alloc (max $MAX_TIME)\""
  while (($#)); do case $1 in -j | --job) need2 "$@"; JOB=$2; shift 2 ;; -t | --time) need2 "$@"; t=$2; shift 2 ;; *) die "extend: unknown option $1" ;; esac; done
  need_job; [[ $t =~ ^[0-9:-]+$ ]] || die "extend: -t NEW_TOTAL_LIMIT (e.g. 8:00:00) required"
  remote 'need_running "$1"; scontrol update JobId="$1" TimeLimit="$2" >/dev/null || err EXTEND_FAILED "job=$C:$1"
echo "OK job=$C:$1 timelimit=$(jfield "$1" %l) timeleft=$(jfield "$1" %L)"' "$JOB" "$t"
}

# ADAPT: free = whole idle nodes with all their GPUs in the partition alloc uses. `planned` nodes
# (held back by backfill for a pending job), mixed, reserved and drained nodes don't count, and
# nothing does once you are at the default QOS's running-jobs cap.
read -r -d '' R_FREE <<'EOF'
P=${PARTITION:-$(sinfo -h -o '%P' | sed -n 's/\*$//p' | head -1)}
idle=$(sinfo -h -p "$P" -N -o '%N|%T|%G' | sort -u | awk -F'|' -v g="$GPUS_PER_NODE" '$2 == "idle" && $3 ~ ":" g "(\\(|,|$)"' | wc -l)
Q=$(sacctmgr -nP show assoc user="$USER" format=defaultqos 2>/dev/null | grep -m1 .); Q=${Q:-normal}
max=$(sacctmgr -nP show qos "$Q" format=maxjobspu 2>/dev/null | head -1)
run=$(squeue -u "$USER" -h -t RUNNING -q "$Q" 2>/dev/null | wc -l)
free=$idle; [[ $max =~ ^[0-9]+$ ]] && ((run >= max)) && free=0
echo "OK cluster=$C free=$free idle=$idle partition=$P qos=$Q running=$run/${max:-none} arch=$GPU_ARCH"
EOF

# ADAPT: release, which follows the holder's cleanup. Running jobs get a release file first, and
# each node's holder stops this job's containers and tasks (with a ~20 s grace) and reports back;
# only then is the job cancelled. Cancelling first would SIGKILL the holders before they could
# clean up.
read -r -d '' R_RELEASE <<'EOF'
ALL=$1; shift
((ALL)) && set -- $($SQUEUE -u "$USER" -h -o '%i|%j' | sed 's/ //g' | awk -F'|' -v p="$C-" 'index($2, p) == 1 {print $1}')
(($#)) || { echo "nothing to release on $C"; exit 0; }
for j in "$@"; do
  # scancel exits 0 even for unknown or foreign job ids, so check ownership first.
  IFS='|' read -r s nn u < <(jfield "$j" '%T|%D|%u')
  if [[ -z $s ]]; then echo "OK job=$C:$j already ended state=$(jfinal "$j")"; continue; fi
  [[ $u == "$USER" ]] || { warn "job=$C:$j belongs to $u, not you; skipped"; continue; }
  if [[ $s == RUNNING ]]; then
    D=$B/jobs/$j; mkdir -p "$D"; touch "$D/release"
    for k in $(seq 90); do
      nfs_sync "$D"; c=$(ls "$D" 2>/dev/null | grep -c '^cleaned\.')
      ((c >= nn)) && break; sleep 1
    done
    ((c >= nn)) || warn "job=$C:$j only $c/$nn nodes confirmed cleanup within 90s; cancelling anyway (see $B/logs/$j.<node>.log)"
  fi
  # After a clean release the holders have exited, so the job may already be completing.
  if $SCANCEL "$j" >/dev/null 2>&1 || [[ $(jfield "$j" '%T') =~ ^(COMPLETING|COMPLETED|)$ ]]; then
    echo "OK released job=$C:$j"
  else warn "cancel failed job=$C:$j"; fi
done
EOF

# ADAPT: what `info` shows: the user's associations, the limits of the QOS they may use, their
# usage, and node states.
read -r -d '' R_INFO <<'EOF'
echo "== my associations (account | partition | allowed qos | default qos); default account: $(sacctmgr -nP show user "$USER" format=defaultaccount 2>/dev/null)"
assoc=$(sacctmgr -nP show assoc user="$USER" format=account,partition,qos,defaultqos 2>/dev/null)
sed 's/^/  /' <<<"$assoc"
echo "== qos limits (MaxTRESPU node= is the per-user node cap; Preempt = qos this one can preempt)"
mine=$(cut -d'|' -f3 <<<"$assoc" | tr ',' '\n' | sort -u | paste -sd'|')
sacctmgr -P show qos format=name,priority,maxwall,maxtrespu,maxjobspu,maxsubmitpu,preempt 2>/dev/null |
  awk -F'|' -v m="^($mine)$" 'NR == 1 || $1 ~ m' | column -t -s '|' | sed 's/^/  /'
echo "== my jobs per qos"
$SQUEUE -u "$USER" -h -o '%q|%T|%D' | sed 's/ //g' | awk -F'|' 'NF {s[$1 " " $2] += $3} END {for (q in s) print "  " q ": " s[q] " nodes"; if (!length(s)) print "  none"}'
echo "== partition node states"
sinfo -h -o '%P|%T|%D' 2>/dev/null | awk -F'|' '{s[$1] = s[$1] " " $2 "=" $3} END {for (p in s) print "  " p ":" s[p]}' | sort
EOF

# ADAPT: the interactive shell `remote-gpu shell` opens on a node.
shell_cmd() { echo "srun --jobid $1 --overlap ${2:+-w $2} -N1 -n1 --cpu-bind=none --pty bash -l"; }
