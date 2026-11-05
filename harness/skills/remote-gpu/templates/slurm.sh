# cluster.sh — remote-gpu scheduler adapter for <NAME>. Made from templates/slurm.sh.
# shellcheck shell=bash disable=SC2016,SC2034
# Adapted: <list the scheduler hooks changed, and why>
#
# Contract (also linked from remote-gpu-setup/references/add-cluster.md): cluster.sh defines CLUSTER_RLIB,
# a Bash library evaluated remotely after the manifest and shared library (C=cluster,
# REL=work root, B=state dir). It contains these functions; stdout is data, stderr diagnostics.
# All hooks return 0 on success, nonzero on failure, except where specified below.
#   scheduler_submit N TIME NAME PARTITION ACCOUNT QOS HOLDER IDLE [OPTIONS...]
#     Submit whole nodes/all GPUs, no requeue; print one numeric job ID. HOLDER is called with
#     IDLE HOLDER B C. Return 201 for a per-user submit limit, 206 for another rejection.
#   scheduler_query JOB
#     Print one pipe-delimited record (empty unknown fields, no embedded pipes/newlines):
#     STATE|NODES|TIMELEFT|REASON|START|PARTITION|ACCOUNT|QOS|TIMELIMIT|USER|NAME|ELAPSED|END|NNODES
#     NODES is a comma-separated list in rank order; times are [D-]HH:MM:SS (or MM:SS),
#     START is a start estimate, END is epoch seconds; NNODES includes pending requests. STATE uses Slurm's normalized states;
#     terminal states must be explicit. Return nonzero for unknown jobs/controller failures.
#   scheduler_jobs
#     List this user's jobs, one JOB|<query record> per line; empty output is success.
#   scheduler_cancel JOB [pending]
#     Cancel JOB; with pending, cancel atomically only if still pending (never a running job).
#   scheduler_run JOB NODE MODE CMD...
#     Run CMD on NODE, all GPUs, overlapping other steps, return CMD's rc and output.
#     MODE=batch consumes no stdin; MODE=interactive uses the caller's tty. ONTO, when set,
#     bounds runtime in seconds. NODE is always a member of the job, N0 is its rank-zero node.
#   scheduler_capacity [info]
#     No args: OK cluster=C free=N account=A qos=Q [burst=N burst_account=A] ...;
#     account/qos identify an ACCOUNTS pair with most room. info: human-readable site details.
#   scheduler_extend JOB TOTAL_TIME  (required only with EXTEND=1)
#     Change the total time limit; no stdout required.
#   scheduler_holder IDLE SELF B C
#     Bootstrap exactly one holder per node; return there with J=job ID, H=rank's node name.
#     A batch parent that launches steps must wait for them and exit, never run the loop too.
# Site facts in CLUSTER_RLIB: DOCKER_X (docker argv), NODE_ENV (KEY=VALUE argv), LATE_HINT.
# The shared CLI owns allocation policy, waiting/health, release, task and holder lifecycle.

read -r -d '' CLUSTER_RLIB <<'REMOTE_GPU_ADAPTER'
LATE_HINT="save results now; the job ends at its time limit"
# ADAPT: mounts and environment inherited by commands on a node.
mount_root=$(df --output=target "$REL" 2>/dev/null | tail -1)
DOCKER_X=($DOCKER_FLAGS -v "$mount_root:$mount_root")
if [[ $HOME != "$mount_root"/* ]]; then DOCKER_X+=(-v "$HOME:$HOME"); fi
NODE_ENV=()
if [[ $HOME_FULL == 1 ]]; then NODE_ENV=(XDG_CACHE_HOME=$B/.cache); fi

# ADAPT: the scheduler's row format and node expansion stay inside this adapter.
slurm_record() {
  local state nodes left reason start partition account qos limit user name elapsed end count
  IFS='|' read -r state nodes left reason start partition account qos limit user name elapsed end count <<<"$1"
  if [[ -n $nodes && $nodes != \(* ]]; then
    nodes=$(scontrol show hostnames "$nodes" 2>/dev/null | paste -sd,)
  else
    nodes=''
  fi
  if [[ $state == PENDING && -n ${2:-} ]]; then
    start=$(squeue --start -j "$2" -h -o '%S' 2>/dev/null | head -1 | tr -d ' ')
  fi
  if [[ $end =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+$ ]]; then
    end=$(date -d "$end" +%s 2>/dev/null)
  else
    end=''
  fi
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$state" "$nodes" "$left" "$reason" "$start" "$partition" "$account" "$qos" "$limit" "$user" "$name" "$elapsed" "$end" "$count"
}
scheduler_query() {
  local row state
  row=$(squeue -j "$1" -h -o '%T|%N|%L|%r|%S|%P|%a|%q|%l|%u|%j|%M|%e|%D' 2>/dev/null |
    head -1 | sed 's/^ *//;s/ *$//;s/ *| */|/g')
  if [[ -n $row ]]; then slurm_record "$row" "$1"; return; fi
  state=$(sacct -j "$1" -X -n -o State%30 2>/dev/null | awk 'NF {print $1; exit}')
  [[ -n $state ]] || return 1
  printf '%s|||||||||||||\n' "${state%%+*}"
}
scheduler_jobs() {
  local rows id row
  rows=$(squeue -u "$USER" -h -o '%i|%T|%N|%L|%r|%S|%P|%a|%q|%l|%u|%j|%M|%e|%D') || return
  while IFS='|' read -r id row; do
    [[ -n $id ]] || continue
    printf '%s|' "${id// /}"
    slurm_record "$(sed 's/^ *//;s/ *$//;s/ *| */|/g' <<<"$row")" "${id// /}"
  done <<<"$rows"
}
scheduler_cancel() {
  if [[ ${2:-} == pending ]]; then scancel -t PENDING "$1"; else scancel "$1"; fi
}
scheduler_run() {
  local job=$1 node=$2 mode=$3 timeout_args=(); shift 3
  if [[ -n ${ONTO:-} ]]; then timeout_args=(timeout "$ONTO"); fi
  if [[ $mode == interactive ]]; then
    "${timeout_args[@]}" srun --jobid "$job" --overlap -w "$node" -N1 -n1 --cpu-bind=none --pty "$@"
  else
    "${timeout_args[@]}" srun --jobid "$job" --overlap -w "$node" -N1 -n1 --cpu-bind=none --quiet "$@" </dev/null \
      2> >(grep --line-buffered -vE '^srun: (error: .*: task [0-9]+: Exited with exit code|error: Timed out waiting for job step|warning)' >&2)
  fi
}
scheduler_submit() {
  local nodes=$1 time=$2 name=$3 partition=$4 account=$5 qos=$6 holder=$7 idle=$8
  local options=() output job; shift 8
  if [[ -n $partition ]]; then options+=(-p "$partition"); fi
  if [[ -n $account ]]; then options+=(-A "$account"); fi
  if [[ -n $qos ]]; then options+=(-q "$qos"); fi
  output=$(sbatch --parsable -J "$name" -N "$nodes" --gpus-per-node "$GPUS_PER_NODE" --exclusive --mem=0 --no-requeue \
    -t "$time" -o "$B/logs/%j.out" -e "$B/logs/%j.err" "${options[@]}" "$@" "$holder" "$idle" "$holder" "$B" "$C" 2>&1)
  job=$(grep -Eo '^[0-9]+(;|$)' <<<"$output" | tail -1)
  job=${job%;}
  if [[ $job =~ ^[0-9]+$ ]]; then echo "$job"; return 0; fi
  echo "$output" >&2
  if [[ $output =~ (QOSMax|AssocMax)[A-Za-z]*(PerUser|Submit)|MaxSubmit ]]; then return 201; fi
  return 206
}
scheduler_holder() {
  J=$SLURM_JOB_ID
  if [[ -z ${REMOTE_GPU_HOLDER_STEP:-} ]]; then
    export REMOTE_GPU_HOLDER_STEP=1
    srun --overlap --ntasks-per-node=1 --cpu-bind=none --kill-on-bad-exit=0 bash "$2" "$1" "$2" "$3" "$4"
    exit $?
  fi
  H=${SLURMD_NODENAME:-$(hostname -s)}
}
scheduler_extend() { scontrol update JobId="$1" TimeLimit="$2"; }

# ADAPT: capacity must count whole usable nodes and the applicable account/QOS caps.
scheduler_capacity() {
  if [[ ${1:-} == info ]]; then
    slurm_info
    return
  fi
  local partition idle pair account qos max running free best=-1 best_account='' best_qos='' burst=0
  partition=${PARTITION:-$(sinfo -h -o '%P' | sed -n 's/\*$//p' | head -1)}
  idle=$(sinfo -h -p "$partition" -N -o '%N|%T|%G' | sort -u |
    awk -F'|' -v g="$GPUS_PER_NODE" '$2 == "idle" && $3 ~ ":" g "(\\(|,|$)"' | wc -l)
  local pairs=($ACCOUNTS)
  if ((${#pairs[@]} == 0)); then
    qos=$(sacctmgr -nP show assoc user="$USER" format=defaultqos 2>/dev/null | grep -m1 .)
    pairs=(":${qos:-normal}")
  fi
  if [[ -n $BURST_QOS ]]; then pairs+=("${pairs[0]%%:*}:$BURST_QOS"); fi
  for pair in "${pairs[@]}"; do
    account=${pair%%:*} qos=${pair#*:}
    max=$(sacctmgr -nP show qos "$qos" format=maxjobspu 2>/dev/null | head -1)
    running=$(squeue -u "$USER" -h -t RUNNING -q "$qos" 2>/dev/null | wc -l)
    free=$idle
    if [[ $max =~ ^[0-9]+$ ]] && ((running >= max)); then free=0; fi
    if [[ -n $BURST_QOS && $qos == "$BURST_QOS" ]]; then
      burst=$free
    elif ((free > best)); then
      best=$free best_account=$account best_qos=$qos
    fi
  done
  echo "OK cluster=$C free=$((best < 0 ? 0 : best)) account=$best_account qos=$best_qos idle=$idle partition=$partition burst=$burst arch=$GPU_ARCH"
}
slurm_info() {
  local associations mine
  echo "== my associations (account | partition | allowed qos | default qos); default account: $(sacctmgr -nP show user "$USER" format=defaultaccount 2>/dev/null)"
  associations=$(sacctmgr -nP show assoc user="$USER" format=account,partition,qos,defaultqos 2>/dev/null)
  sed 's/^/  /' <<<"$associations"
  echo "== qos limits (MaxTRESPU node= is the per-user node cap; Preempt = qos this one can preempt)"
  mine=$(cut -d'|' -f3 <<<"$associations" | tr ',' '\n' | sort -u | paste -sd'|')
  sacctmgr -P show qos format=name,priority,maxwall,maxtrespu,maxjobspu,maxsubmitpu,preempt 2>/dev/null |
    awk -F'|' -v m="^($mine)$" 'NR == 1 || $1 ~ m' | column -t -s '|' | sed 's/^/  /'
  echo "== my jobs per qos"
  squeue -u "$USER" -h -o '%q|%T|%D' | sed 's/ //g' |
    awk -F'|' 'NF {s[$1 " " $2] += $3} END {for (q in s) print "  " q ": " s[q] " nodes"; if (!length(s)) print "  none"}'
  echo "== partition node states"
  sinfo -h -o '%P|%T|%D' 2>/dev/null |
    awk -F'|' '{s[$1] = s[$1] " " $2 "=" $3} END {for (p in s) print "  " p ":" s[p]}' | sort
}
REMOTE_GPU_ADAPTER
