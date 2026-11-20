#!/usr/bin/env bash
set -euo pipefail

# The runner (workspace global-cpu) and the GPU pod (workspace $WORKSPACE, another cluster) share
# no storage. The pod fetches the source from GitHub, and its log, exit code and reports come back
# through the SaFE pod-log API.

certs=https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs
curl -fsSL --retry 3 "$certs/amd-root-ca.crt" "$certs/amd-issuing-ca.crt" > "$RUNNER_TEMP/amd-ca.pem"

api() {
  curl -sS --fail-with-body --max-time 60 --cacert "$RUNNER_TEMP/amd-ca.pem" -X "$1" \
    -H "Authorization: Bearer $USER_APIKEY" -H 'Content-Type: application/json' \
    "https://global.primus-safe.amd.com/api/v1/workloads$2" "${@:3}"
}

name=aklr-ci-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-${GITHUB_JOB%-test}

if [[ ${1:-} == delete ]]; then
  ids=$(api GET "?workspaceId=$WORKSPACE&kind=Deployment&limit=100" |
    jq -r --arg name "$name" '.items[]?.workloadId | select(startswith($name))')
  for id in $ids; do
    echo "🧹 Deleting the workload $id"
    api DELETE "/$id" > /dev/null
  done
  exit
fi

script=${1:?usage: safe_job.sh <script> | delete}
# The checkout's commit: GITHUB_SHA on a push, the pull request's merge commit on a PR.
commit=$(git -C "$GITHUB_WORKSPACE" rev-parse HEAD)

# Every line the pod prints is "<number><tab><text>", so that a poll prints only the new ones.
# shortcut: a pod that restarts runs the script again; detect it when a node proves unstable.
pod_command=$(
  cat << 'EOF'
{
  work=$(mktemp -d)
  export REPORTS=$work/reports
  mkdir "$work/src" "$REPORTS"
  cd "$work/src"
  git init -q . &&
    git fetch -q --tags "https://github.com/$AKLR_REPOSITORY" '+refs/heads/*:refs/remotes/origin/*' "$AKLR_COMMIT" &&
    git checkout -q "$AKLR_COMMIT" &&
    base64 -d <<< "$AKLR_SCRIPT" > "$work/script.sh" &&
    bash "$work/script.sh"
  code=$?
  echo "::aklr-reports::"
  tar -C "$REPORTS" -cz . | base64 -w 4000
  echo "::aklr-exit::$code"
} 2>&1 | awk '{printf "%d\t%s\n", NR, $0; fflush()}'
exec sleep infinity
EOF
)
workload=$(jq -n --arg name "$name" --arg command "$(base64 -w0 <<< "$pod_command")" --arg pass "${PASS_ENV:-}" \
  --arg commit "$commit" --arg script "$(base64 -w0 "$script")" '{
  displayName: $name,
  workspaceId: env.WORKSPACE,
  groupVersionKind: {kind: "Deployment", version: "v1"},
  images: [env.IMAGE],
  entryPoints: [$command],
  resources: [{replica: 1, gpu: "8", cpu: "96", memory: "512Gi", sharedMemory: "64Gi", ephemeralStorage: "300Gi"}],
  env: ({AKLR_REPOSITORY: env.GITHUB_REPOSITORY, AKLR_COMMIT: $commit, AKLR_SCRIPT: $script} +
    ($ENV | with_entries(select(.key | IN($pass | splits(" +")))))),
  priority: ({medium: 1, high: 2}[env.AKLR_GPU_PRIORITY] // error("AKLR_GPU_PRIORITY must be medium or high")),
  timeout: (env.TIMEOUT_MINUTES | tonumber * 60),
  useWorkspaceStorage: false
}' | api POST '' -d @- | jq -r '.workloadId // error(tostring)')
echo "🚀 Workload $workload in $IMAGE at $AKLR_GPU_PRIORITY priority"

log=$RUNNER_TEMP/gpu-log
last_phase='' last_message='' lost=0 printed=0
: > "$log"
until grep -q $'\t::aklr-exit::' "$log"; do
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
  pod=$(jq -r '.pods[0].podId // empty' <<< "$state")
  [[ -n $pod ]] || continue
  # 20000 lines a poll is far more than a test run prints in 15 seconds, and holds the reports.
  api GET "/$workload/pods/$pod/logs?tailLines=20000" | jq -r '.logs[]?' |
    awk -F'\t' -v printed="$printed" '$1 ~ /^[0-9]+$/ && $1 > printed' >> "$log" || continue
  # The reports are base64, not for the job's log.
  awk -F'\t' -v printed="$printed" '{number = $1; sub(/^[0-9]+\t/, "")} $0 == "::aklr-reports::" {exit} number > printed' "$log"
  printed=$(awk -F'\t' 'END {print $1 + 0}' "$log")
done

mkdir -p "$RUNNER_TEMP/gpu-reports"
sed 's/^[0-9]*\t//' "$log" | sed -n '/^::aklr-reports::$/,/^::aklr-exit::/{//!p}' | base64 -d |
  tar -C "$RUNNER_TEMP/gpu-reports" -xz
exit "$(sed -n 's/^[0-9]*\t::aklr-exit:://p' "$log")"
