#!/usr/bin/env bash
set -euo pipefail

# The runner and the GPU pod are in different clusters and share no storage, and the pod is not a
# runner, so no action runs in it: the pod fetches the commit from GitHub, and its log, exit code
# and reports come back through the SaFE pod-log API. ci.yaml sets the AKLR_SAFE_* variables.

: "${AKLR_SAFE_API_URL:?}" "${AKLR_SAFE_WORKSPACE:?}" "${AKLR_SAFE_WORKLOAD_KIND:?}" "${AKLR_SAFE_CA_URLS:?}"

read -r -a ca_urls <<< "$AKLR_SAFE_CA_URLS"
curl -fsSL --retry 3 "${ca_urls[@]}" > "$RUNNER_TEMP/amd-ca.pem"

api() {
  # The key goes in through a file descriptor: every process on the runner can read argv.
  curl -sS --fail-with-body --max-time 60 --cacert "$RUNNER_TEMP/amd-ca.pem" -X "$1" \
    -H @<(printf 'Authorization: Bearer %s\n' "$USER_APIKEY") -H 'Content-Type: application/json' \
    "$AKLR_SAFE_API_URL/api/v1/workloads$2" "${@:3}"
}

name=aklr-ci-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-${GITHUB_JOB%-test}

if [[ ${1:-} == delete ]]; then
  ids=$(api GET "?workspaceId=$AKLR_SAFE_WORKSPACE&kind=$AKLR_SAFE_WORKLOAD_KIND&workloadId=$name" |
    jq -r --arg name "$name" '.items[]?.workloadId | select(startswith($name))')
  for id in $ids; do
    echo "🧹 Deleting the workload $id"
    api DELETE "/$id" > /dev/null
  done
  exit
fi

script=${1:?usage: safe_job.sh <script> | delete}
: "${AKLR_SAFE_RESOURCES:?}" "${IMAGE:?}" "${TIMEOUT_MINUTES:?}"
# The runner unpacks the reports with it; fail before a GPU node is taken.
python3 -c 'import lzma'

# Every line the pod prints is "<number><tab><text>", so that a poll takes only the new ones.
# The numbering is a bash loop: the vllm image's awk (an old mawk) holds piped input back.
# The reports are xz: a full validation report is 14 MB as base64 gzip and 4 MB as base64 xz, and
# the pod's log is read from one file, which the kubelet rotates at 10 MiB by default.
# shortcut: a container that restarts in its pod runs the script again; detect it when a node proves unstable.
pod_command=$(
  cat << 'EOF'
{
  work=$(mktemp -d)
  export REPORTS=$work/reports
  mkdir "$work/src" "$REPORTS"
  cd "$work/src"
  # What the runner's actions/checkout checked out (GITHUB_SHA: the merge commit on a PR), with its
  # history and the tags, as with fetch-depth: 0.
  git init -q . &&
    git fetch -q --tags "$AKLR_REPO_URL" "$AKLR_COMMIT" &&
    git checkout -q "$AKLR_COMMIT" &&
    base64 -d <<< "$AKLR_SCRIPT" > "$work/script.sh" &&
    bash "$work/script.sh"
  code=$?
  # The newline ends a last line the script left open, which would swallow the marker.
  printf '\n::aklr-reports::\n'
  tar -C "$REPORTS" -cJ . | base64 -w 4000
  echo "::aklr-exit::$code"
} 2>&1 | {
  number=0
  while IFS= read -r line; do printf '%d\t%s\n' "$((++number))" "$line"; done
}
exec sleep infinity
EOF
)
workload=$(jq -n --arg name "$name" --arg command "$(base64 -w0 <<< "$pod_command")" --arg pass "${PASS_ENV:-}" \
  --arg script "$(base64 -w0 "$script")" '{
  displayName: $name,
  workspaceId: env.AKLR_SAFE_WORKSPACE,
  groupVersionKind: {kind: env.AKLR_SAFE_WORKLOAD_KIND, version: "v1"},
  images: [env.IMAGE],
  entryPoints: [$command],
  resources: [env.AKLR_SAFE_RESOURCES | fromjson],
  env: ({AKLR_REPO_URL: "\(env.GITHUB_SERVER_URL)/\(env.GITHUB_REPOSITORY)", AKLR_COMMIT: env.GITHUB_SHA, AKLR_SCRIPT: $script} +
    ($ENV | with_entries(select(.key | IN($pass | splits(" +")))))),
  priority: ({medium: 1, high: 2}[env.AKLR_GPU_PRIORITY] // error("AKLR_GPU_PRIORITY must be medium or high")),
  timeout: (env.TIMEOUT_MINUTES | tonumber * 60),
  useWorkspaceStorage: false
}' | api POST '' -d @- | jq -r '.workloadId // error(tostring)')
echo "🚀 Workload $workload in $IMAGE at $AKLR_GPU_PRIORITY priority"

# One poll reads the last $tail_lines lines; the largest run prints about 3500, reports included.
tail_lines=20000
log=$RUNNER_TEMP/gpu-log
last_phase='' last_message='' lost=0 printed=0 pod=''
: > "$log"
until grep -q $'^[0-9]*\t::aklr-exit::' "$log"; do
  sleep 15
  if ! state=$(api GET "/$workload"); then
    if ((++lost == 8)); then
      echo "::error::Workload $workload is gone: deleted, or stopped after $TIMEOUT_MINUTES minutes"
      exit 1
    fi
    continue
  fi
  lost=0
  phase=$(jq -r .phase <<< "$state") message=$(jq -r '.message // ""' <<< "$state")
  if [[ $phase != "$last_phase" || $message != "$last_message" ]]; then
    echo "⏳ Workload $workload: $phase${message:+ ($message)}"
  fi
  last_phase=$phase last_message=$message
  if [[ $phase == Failed || $phase == Stopped ]]; then
    echo "::error::Workload $workload $phase before the script finished"
    exit 1
  fi
  current=$(jq -r '.pods[0].podId // empty' <<< "$state")
  [[ -n $current ]] || continue
  # A new pod runs the script again from line 1.
  if [[ $current != "$pod" ]] && ((printed)); then
    echo "::error::The GPU pod $pod was replaced by $current during the script"
    exit 1
  fi
  pod=$current
  new=$(api GET "/$workload/pods/$pod/logs?tailLines=$tail_lines" | jq -r '.logs[]?' |
    awk -F'\t' -v printed="$printed" '$1 ~ /^[0-9]+$/ && $1 > printed') || continue
  [[ -n $new ]] || continue
  # Lost reports fail below, where they are unpacked.
  ((${new%%$'\t'*} == printed + 1)) ||
    echo "::warning::The pod's log lost lines $((printed + 1)) to $((${new%%$'\t'*} - 1)): more than $tail_lines new lines, or its file rotated"
  printf '%s\n' "$new" >> "$log"
  # The reports are base64, not for the job's log.
  awk -F'\t' -v printed="$printed" '{number = $1; sub(/^[0-9]+\t/, "")} $0 == "::aklr-reports::" {exit} number > printed' "$log"
  printed=$(awk -F'\t' 'END {print $1 + 0}' "$log")
done

mkdir -p "$RUNNER_TEMP/gpu-reports"
# python3's lzma: the runner image has no xz.
sed 's/^[0-9]*\t//' "$log" | sed -n '/^::aklr-reports::$/,/^::aklr-exit::/{//!p}' | base64 -d |
  python3 -c 'import lzma, sys; sys.stdout.buffer.write(lzma.decompress(sys.stdin.buffer.read()))' |
  tar -C "$RUNNER_TEMP/gpu-reports" -x
exit "$(sed -n 's/^[0-9]*\t::aklr-exit:://p' "$log")"
