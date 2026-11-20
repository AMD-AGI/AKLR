#!/usr/bin/env bash
set -euo pipefail

certs=https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs
curl -fsSL --retry 3 "$certs/amd-root-ca.crt" "$certs/amd-issuing-ca.crt" > "$RUNNER_TEMP/amd-ca.pem"

api() {
  curl -sS --fail-with-body --max-time 60 --cacert "$RUNNER_TEMP/amd-ca.pem" -X "$1" \
    -H "Authorization: Bearer $USER_APIKEY" -H 'Content-Type: application/json' \
    "https://global.primus-safe.amd.com/api/v1/workloads$2" "${@:3}"
}

name=aklr-ci-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-${GITHUB_JOB%-test}
storage=$(awk '$2 ~ "^/shared_nfs/users/[^/]+$" {print $2; exit}' /proc/mounts)
export RUN_DIR=$storage/aklr-ci-runs/$name

if [[ ${1:-} == delete ]]; then
  rm -rf "$RUN_DIR"
  ids=$(api GET "?workspaceId=$WORKSPACE&kind=Deployment&limit=100" |
    jq -r --arg name "$name" '.items[]?.workloadId | select(startswith($name))')
  for id in $ids; do
    echo "🧹 Deleting the workload $id"
    api DELETE "/$id" > /dev/null
  done
  exit
fi

umask 007
mkdir -p "$RUN_DIR"
tar -C "$GITHUB_WORKSPACE" -cf "$RUN_DIR/src.tar" .
cp "${1:?usage: safe_job.sh <script> | delete}" "$RUN_DIR/script.sh"

pod_command=$(
  cat << 'EOF'
if [ ! -e "$RUN_DIR/started" ]; then
  touch "$RUN_DIR/started"
  {
    work=$(mktemp -d)
    export REPORTS=$work/reports
    mkdir "$work/src" "$REPORTS"
    cd "$work/src"
    tar -xf "$RUN_DIR/src.tar"
    bash "$RUN_DIR/script.sh"
    code=$?
    tar -C "$REPORTS" -cf "$RUN_DIR/reports.tar" .
  } >> "$RUN_DIR/log" 2>&1
  echo "$code" > "$RUN_DIR/exit_code"
elif [ ! -e "$RUN_DIR/exit_code" ]; then
  echo "::error::The GPU pod restarted during the script" >> "$RUN_DIR/log"
  echo 1 > "$RUN_DIR/exit_code"
fi
exec sleep infinity
EOF
)
workload=$(jq -n --arg name "$name" --arg command "$(base64 -w0 <<< "$pod_command")" --arg pass "${PASS_ENV:-}" '{
  displayName: $name,
  workspaceId: env.WORKSPACE,
  groupVersionKind: {kind: "Deployment", version: "v1"},
  images: [env.IMAGE],
  entryPoints: [$command],
  resources: [{replica: 1, gpu: "8", cpu: "96", memory: "512Gi", sharedMemory: "64Gi", ephemeralStorage: "300Gi"}],
  env: ({RUN_DIR: env.RUN_DIR} + ($ENV | with_entries(select(.key | IN($pass | splits(" +")))))),
  priority: ({medium: 1, high: 2}[env.AKLR_GPU_PRIORITY] // error("AKLR_GPU_PRIORITY must be medium or high")),
  timeout: (env.TIMEOUT_MINUTES | tonumber * 60),
  useWorkspaceStorage: true
}' | api POST '' -d @- | jq -r '.workloadId // error(tostring)')
echo "🚀 Workload $workload in $IMAGE at $AKLR_GPU_PRIORITY priority"

tail --pid=$$ -F "$RUN_DIR/log" 2> /dev/null &
last_phase='' lost=0
until [[ -s $RUN_DIR/exit_code ]]; do
  sleep 15
  if ! phase=$(api GET "/$workload" | jq -r .phase); then
    if ((++lost == 8)); then
      echo "::error::Workload $workload is gone: deleted, or stopped after $TIMEOUT_MINUTES minutes"
      exit 1
    fi
    continue
  fi
  lost=0
  [[ $phase == "$last_phase" ]] || echo "⏳ Workload $workload: $phase"
  last_phase=$phase
  if [[ $phase == Failed || $phase == Stopped ]]; then
    echo "::error::Workload $workload $phase before the script finished"
    exit 1
  fi
done
sleep 5
kill $!

mkdir -p "$RUNNER_TEMP/gpu-reports"
tar -C "$RUNNER_TEMP/gpu-reports" -xf "$RUN_DIR/reports.tar"
exit "$(< "$RUN_DIR/exit_code")"
