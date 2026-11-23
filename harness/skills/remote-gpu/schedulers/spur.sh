# shellcheck shell=bash disable=SC1091,SC2016,SC2034
source "${BASH_SOURCE[0]%/*}/slurm.sh"
read -r -d '' SPUR_RLIB <<'REMOTE_GPU_ADAPTER'
LATE_HINT="save results now, or extend the job: remote-gpu extend -j $C:ID -t NEW_TOTAL"
slurm_ended() { spur history -j "$1" -o State -n 2>/dev/null; }
slurm_sbatch() { spur submit "$@"; }
scheduler_run() {
  local job=$1 node=$2 mode=$3 tag rc; shift 3
  if [[ $mode == interactive ]]; then
    ${ONTO:+timeout "$ONTO"} spur run --jobid "$job" --overlap -w "$node" --pty "$@"
    return
  fi
  tag=$C-$job-$BASHPID-$RANDOM
  trap 'spur_kill_tagged "$job" "$node" "$tag"; exit 143' TERM INT HUP
  spur_step "$job" "$node" "${ONTO:-}" env "REMOTE_GPU_RUN_TAG=$tag" "$@"; rc=$?
  trap - TERM INT HUP
  if ((rc == 124)); then spur_kill_tagged "$job" "$node" "$tag"; fi
  return "$rc"
}
spur_step() {
  local job=$1 node=$2 bound=(); [[ -n $3 ]] && bound=(timeout "$3"); shift 3
  if [[ $node == "${N0:-}" ]]; then
    "${bound[@]}" spur exec "$job" "$@" </dev/null
  else
    "${bound[@]}" spur run --jobid "$job" --overlap -w "$node" "$@" </dev/null 2> >(grep -vE '^(srun|spur): warning' >&2) |
      sed -u 's/\r$//'
    return "${PIPESTATUS[0]}"
  fi
}
spur_kill_tagged() {
  spur_step "$1" "$2" 60 bash -c 'for f in /proc/[0-9]*/environ; do
    if grep -qzx "REMOTE_GPU_RUN_TAG=$1" "$f" 2>/dev/null; then p=${f%/environ}; kill -TERM "${p#/proc/}"; fi
  done' _ "$3" >/dev/null 2>&1
}
scheduler_holder() {
  J=${SPUR_JOB_ID:-$SLURM_JOB_ID} H=$(hostname -s)
  trap 'spur cancel "$J" >/dev/null 2>&1' EXIT
}
scheduler_capacity() {
  if [[ ${1:-} == info ]]; then
    spur_info
    return
  fi
  local idle manager caps pair account qos cap used headroom best=-1 best_account='' best_qos='' burst=0
  idle=$(spur nodes -p "$PARTITION" -h 2>/dev/null | awk '$5 == "idle" {s += $4} END {print s + 0}')
  manager=$(scontrol show assoc_mgr 2>/dev/null) || return
  caps=$(spur accounts show qos 2>/dev/null)
  spur_cap() { awk -v q="$1" '$1 == q && match($NF, /(^|,)node=[0-9]+/) {s = substr($NF, RSTART, RLENGTH); sub(/.*node=/, "", s); print s + 0}' <<<"$caps"; }
  spur_used() { local u; u=$(grep -A1 -E "^$1( |$)" <<<"$manager" | grep -o 'node=[0-9N]*([0-9]*)' | head -1); u=${u##*(}; echo "${u%)}"; }
  for pair in $ACCOUNTS; do
    account=${pair%%:*} qos=${pair#*:}
    cap=$(spur_cap "$qos") used=$(spur_used "QOS=$qos")
    [[ -n $used ]] || used=$(spur_used "Account=$account")
    headroom=$((${cap:-0} - ${used:-0})); ((headroom < 0)) && headroom=0
    ((headroom > best)) && best=$headroom best_account=$account best_qos=$qos
  done
  if [[ -n $BURST_QOS ]]; then
    cap=$(spur_cap "$BURST_QOS") used=$(spur_used "QOS=$BURST_QOS")
    headroom=$((${cap:-0} - ${used:-0})); ((headroom < 0)) && headroom=0
    burst=$((idle < headroom ? idle : headroom))
  fi
  best=$((best < 0 ? 0 : best))
  echo "OK cluster=$C free=$((idle < best ? idle : best)) account=$best_account qos=$best_qos idle=$idle partition=$PARTITION burst=$burst burst_account=${ACCOUNTS%%:*} arch=$GPU_ARCH"
}
spur_info() {
  local mine
  echo "== my associations (account -> allowed qos)"
  spur accounts show user 2>/dev/null | awk -v u="$USER" '$1 == u {for (i = 3; i <= NF; i++) if ($i ~ /qos/) {print "  " $2 " -> " $i; break}}'
  echo "== qos limits (GrpTRES node= is the team-wide node cap; MaxWall in minutes)"
  mine=$(for pair in $ACCOUNTS; do echo "${pair#*:}"; done; echo "$BURST_QOS")
  spur accounts show qos 2>/dev/null | awk -v m="$(echo $mine | tr ' ' '|')" 'NR <= 2 || $1 ~ "^(" m ")$"'
  echo "== my nodes per qos (running)"
  squeue -u "$USER" -t RUNNING -h -o '%q|%D' | sed 's/ //g' |
    awk -F'|' 'NF {s[$1] += $2} END {for (q in s) print "  " q ": " s[q]; if (!length(s)) print "  none"}'
  echo "== partition $PARTITION node states"
  spur nodes -p "$PARTITION" -h 2>/dev/null | awk '$4 ~ /^[0-9]+$/ {s[$5] += $4} END {for (k in s) printf "  %s=%d", k, s[k]; print ""}'
}
REMOTE_GPU_ADAPTER
CLUSTER_RLIB+=$'\n'$SPUR_RLIB
