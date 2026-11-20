# shellcheck shell=bash disable=SC2016,SC2034
: "${SAFE_URL:?} ${SAFE_KIND:?} ${SAFE_IMAGE:?} ${SAFE_RESOURCES:?}"
EXTEND=0 DOCKER=0 MULTI_NODE=0 NODE_SHELL=0
[[ $SAFE_KIND != Deployment ]] || IDLE_RELEASE=0

read -r -d '' CLUSTER_RLIB <<'REMOTE_GPU_ADAPTER'
LATE_HINT="save results now; SaFE stops the workload at its time limit"
DOCKER_X=() NODE_ENV=()

safe_api() {
  local key=$HOME/.remote-gpu/$C.key reply data=()
  if [[ ! -r $key ]]; then echo "$C: no SaFE API key in $key" >&2; return 1; fi
  if [[ -n ${3:-} ]]; then data=(--data-binary "$3"); fi
  reply=$(curl -sS --max-time 30 -X "$1" -H 'Content-Type: application/json' "${data[@]}" -w '\n%{http_code}' \
    -K <(printf 'header = "Authorization: Bearer %s"\n' "$(<"$key")") "$SAFE_URL/api/v1$2") || return 1
  printf '%s\n' "${reply%$'\n'*}"
  # A 404 means "no such workload" only in SaFE's own words; a gateway's is a controller failure.
  case ${reply##*$'\n'} in 2*) return 0 ;; 404) [[ $(jq 'has("errorCode")' <<<"${reply%$'\n'*}" 2>/dev/null) == true ]] && return 4 ;; esac
  return 1
}

scheduler_submit() {
  local nodes=$1 time=$2 name=$3 workspace=$4 holder=$7 idle=$8 excluded='' specified='' job entry body reply id
  shift 8
  while (($#)); do
    case $1 in
      -x | --exclude) excluded=$2 ;;
      -w | --nodelist) specified=$2 ;;
      *) echo "$C: SaFE has no $1" >&2; return 206 ;;
    esac
    shift 2
  done
  job=$(date +%s%3N)
  mkdir -p "$B/safe" && [[ -w $B/safe ]] || { echo "$C: cannot write $B/safe" >&2; return 206; }
  until (set -C; : >"$B/safe/$job.id") 2>/dev/null; do job=$((job + 1)); done
  mkdir -p "$B/jobs/$job/tasks" "$B/jobs/$job/spawn" "$B/jobs/$job/run"
  touch "$B/jobs/$job/heartbeat"; chmod 666 "$B/jobs/$job/heartbeat"
  printf -v entry 'export REMOTE_GPU_SAFE_JOB=%s\nbash %q %q %q %q %q\n' "$job" "$holder" "$idle" "$holder" "$B" "$C"
  if [[ $SAFE_KIND == Deployment ]]; then entry+=$'exec sleep infinity\n'; fi
  body=$(jq -n --arg name "$(tr 'A-Z_.' 'a-z--' <<<"$name" | tr -cd 'a-z0-9-' | cut -c1-40)" \
    --arg workspace "$workspace" --arg kind "$SAFE_KIND" --arg image "$SAFE_IMAGE" --arg entry "$(base64 -w0 <<<"$entry")" \
    --arg gpu "$GPUS_PER_NODE" --argjson resources "$SAFE_RESOURCES" --argjson timeout "$(tosec "$time")" \
    --arg excluded "$excluded" --arg specified "$specified" '
    {displayName: $name, workspaceId: $workspace, groupVersionKind: {kind: $kind, version: "v1"}, images: [$image],
     entryPoints: [$entry], resources: [$resources + {replica: 1, gpu: $gpu}], timeout: $timeout}
    + (if $excluded == "" then {} else {excludedNodes: ($excluded | split(","))} end)
    + (if $specified == "" then {} else {specifiedNodes: ($specified | split(","))} end)')
  if reply=$(safe_api POST /workloads "$body") && id=$(jq -er .workloadId <<<"$reply"); then
    printf '%s\n%s\n' "$id" "$name" >"$B/safe/$job.id"
    echo "$job"; return 0
  fi
  rm -rf "$B/safe/$job.id" "$B/jobs/$job"; echo "$reply" >&2
  return 206
}

scheduler_query() {
  local id name reply status record
  { read -r id && read -r name; } 2>/dev/null <"$B/safe/$1.id" || return 1
  reply=$(safe_api GET "/workloads/$id" 2>/dev/null); status=$?
  if ((status == 4)); then printf 'CANCELLED|||||||||%s|%s|||1\n' "${USER:-}" "$name"; return 0; fi
  ((status == 0)) || return 1
  record=$(jq -r --arg user "${USER:-}" --arg name "$name" --argjson now "$(date +%s)" '
    def two: tostring | if length < 2 then "0" + . else . end;
    def hms: floor | "\(. / 3600 | floor):\(. % 3600 / 60 | floor | two):\(. % 60 | two)";
    (.pods // []) as $pods | (.timeout // 0) as $limit
    | ((.startTime // "") | if . == "" then null else (sub("Z$"; "") + "Z" | fromdateiso8601) end) as $start
    | [(if .phase == "Succeeded" then "COMPLETED" elif .phase == "Failed" then "FAILED" elif .phase == "Stopped" then "CANCELLED"
        elif .phase == "Running" and ($pods | length) > 0 and all($pods[]; .phase == "Running" and (.adminNodeName // "") != "")
        then "RUNNING" else "PENDING" end),
       ([$pods[] | .adminNodeName // empty] | join(",")),
       (if $limit > 0 and (.secondsUntilTimeout // -1) >= 0 then .secondsUntilTimeout | hms else "" end),
       ((.message // "") + (if (.queuePosition // 0) > 0 then " queue position \(.queuePosition)" else "" end) | gsub("[|\n]"; " ") | .[0:160]),
       "", (.workspaceId // ""), "", "", (if $limit > 0 then $limit | hms else "" end), $user, $name,
       (if $start then $now - $start | hms else "" end),
       (if $start and $limit > 0 then $start + $limit | tostring else "" end), "1"] | join("|")' <<<"$reply") || return 1
  if [[ $record == RUNNING\|* && ! -s $B/safe/$1.node ]]; then cut -d'|' -f2 <<<"$record" >"$B/safe/$1.node"; fi
  printf '%s\n' "$record"
}

scheduler_jobs() {
  local file job record failed=0
  for file in "$B"/safe/*.id; do
    [[ -s $file ]] || continue
    job=${file##*/}; job=${job%.id}
    record=$(scheduler_query "$job") || { failed=1; continue; }
    if terminal_state "${record%%|*}"; then rm -f "$B/safe/$job".*; continue; fi
    printf '%s|%s\n' "$job" "$record"
  done
  return "$failed"
}

scheduler_cancel() {
  local id status
  read -r id 2>/dev/null <"$B/safe/$1.id" || return 1
  # shortcut: SaFE has no delete-if-pending, so a workload that starts between this check and the delete is cancelled too; use one when the API has it.
  if [[ ${2:-} == pending && $(scheduler_query "$1") != PENDING\|* ]]; then return 1; fi
  safe_api DELETE "/workloads/$id" >/dev/null 2>&1; status=$?
  ((status == 0 || status == 4))
}

read -r -d '' SAFE_RUN <<'SAFE_RUN_EOF'
#!/bin/bash
F=$2 T=$3; shift 3
if [[ -n $T ]]; then set -- timeout "$T" "$@"; fi
setsid "$@" >"$F.out" 2>"$F.err" </dev/null & pid=$!
while kill -0 "$pid" 2>/dev/null; do
  t=$(mktemp "${F%/*}/.sync.XXXXXX" 2>/dev/null) && rm -f "$t"
  if [[ -e $F.kill ]]; then kill -TERM -- "-$pid" 2>/dev/null; fi
  sleep 1
done
wait "$pid"; echo $? >"$F.rc.tmp"; mv -f "$F.rc.tmp" "$F.rc"
SAFE_RUN_EOF

scheduler_run() {
  local job=$1 node=$2 mode=$3 base request runner stream size finished=0 ticks=0 status=255 record traps
  local -A sent=([out]=0 [err]=0) descriptor=([out]=1 [err]=2)
  shift 3
  [[ $mode == batch ]] || return 200
  base=$B/jobs/$job/run/$(date +%s%N).$BASHPID
  request=$B/jobs/$job/spawn/.$node.${base##*/}
  runner=$B/saferun.$(cksum <<<"$SAFE_RUN" | cut -d' ' -f1).sh
  [[ -f $runner && -O $runner ]] || install_script "${runner##*/}" <<<"$SAFE_RUN"
  : >"$base.out"; : >"$base.err"
  traps=$(trap -p TERM INT HUP)
  trap 'touch "$base.kill"; exit 143' TERM INT HUP
  printf '%s\0' "$base" "$runner" "${ONTO:-}" "$@" >"$request" && mv -f "$request" "${request%/*}/${request##*/.}"
  while :; do
    [[ -e $base.rc ]] && finished=1
    for stream in out err; do
      size=$(wc -c <"$base.$stream")
      ((size > sent[$stream])) || continue
      tail -c +$((sent[$stream] + 1)) "$base.$stream" | head -c $((size - sent[$stream])) >&"${descriptor[$stream]}"
      sent[$stream]=$size
    done
    if ((finished)); then status=$(<"$base.rc"); break; fi
    sleep 1; nfs_sync "${base%/*}"; ticks=$((ticks + 1))
    if ((ticks == 120)) && [[ ! -e $base.pid ]]; then
      echo "$C: the holder of job $job did not start the command within 120 s (see $B/logs/$job.$node.log)" >&2
      rm -f "${request%/*}/${request##*/.}"; break
    fi
    if ((ticks % 30 == 0)) && record=$(scheduler_query "$job") && terminal_state "${record%%|*}"; then
      echo "$C: job $job ended (${record%%|*}) during the command" >&2; status=203; break
    fi
  done
  eval "${traps:-trap - TERM INT HUP}"
  rm -f "$base".*
  return "$status"
}

scheduler_holder() {
  local attempt
  J=$REMOTE_GPU_SAFE_JOB
  for attempt in $(seq 300); do
    nfs_sync "$B/safe"
    [[ -s $B/safe/$J.node ]] && break
    sleep 2
  done
  H=$(<"$B/safe/$J.node") || exit 1
}

scheduler_capacity() {
  local nodes free
  nodes=$(safe_api GET "/nodes?workspaceId=$PARTITION&limit=1000") || return 1
  if [[ ${1:-} == info ]]; then
    echo "== workspace $PARTITION (node | phase | schedulable | free GPUs)"
    jq -r '.items[]? | "  \(.nodeId // .nodeName) | \(.phase) | \(.available) | \(.availResources["amd.com/gpu"] // 0)"' <<<"$nodes"
    echo "== my jobs (job | state | node | time left)"
    scheduler_jobs | cut -d'|' -f1-4 | sed 's/^/  /'
    return
  fi
  free=$(jq --argjson gpus "$GPUS_PER_NODE" \
    '[.items[]? | select(.phase == "Ready" and .available and (.availResources["amd.com/gpu"] // 0 | tonumber) >= $gpus)] | length' <<<"$nodes")
  echo "OK cluster=$C free=$free account= qos= idle=$free partition=$PARTITION burst=0 arch=$GPU_ARCH"
}
REMOTE_GPU_ADAPTER
