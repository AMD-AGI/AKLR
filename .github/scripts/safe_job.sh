#!/usr/bin/env bash
# Runs gpu_tests.sh on a Primus-SaFE GPU node, in $IMAGE, and waits for it: packs the checkout into
# a run directory on the workspace's shared storage, submits a workload that unpacks and runs it,
# streams its log, exits with its exit code and deletes the workload. The reports stay in $RUN_DIR
# for the upload steps.
#
# Runs on the aklr-safe runner, a Primus-SaFE GithubRunner pod with no GPU: the platform gives it
# USER_APIKEY, WORKSPACE and GITHUB_RUNNER_STATE_ROOT (shared storage the GPU workload also mounts).
set -euo pipefail
: "${SAFE_API:?}" "${IMAGE:?}" "${TESTS:?}" "${USER_APIKEY:?}" "${WORKSPACE:?}" "${GITHUB_RUNNER_STATE_ROOT:?}"

export JOB=$GITHUB_JOB
export RUN_DIR=$GITHUB_RUNNER_STATE_ROOT/ci-runs/$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-$JOB
echo "RUN_DIR=$RUN_DIR" >> "$GITHUB_ENV"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
# The workload may run as another user than the runner's.
chmod a+rwx "$RUN_DIR"
tar -C "$GITHUB_WORKSPACE" -cf "$RUN_DIR/src.tar" .
touch "$RUN_DIR/log"
chmod a+rw "$RUN_DIR/log"

# The API's certificate is signed by AMD's CA, which the runner image does not carry.
ca=$RUNNER_TEMP/amd-ca.pem
certs=https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs
curl --silent --show-error --fail --location "$certs/amd-root-ca.crt" "$certs/amd-issuing-ca.crt" > "$ca"
api() {
  curl --silent --show-error --fail-with-body --max-time 60 --cacert "$ca" \
    -H "Authorization: Bearer $USER_APIKEY" -H 'Content-Type: application/json' "$@"
}

# The workspace runs services (the Infer scope), not batch jobs: the workload is a one-replica
# Deployment that runs the tests once and then idles until this script deletes it. A pod that
# restarts must not run them again.
# shellcheck disable=SC2016 # expanded by the workload's shell
entry='if [ -e "$RUN_DIR/started" ]; then
  [ -e "$RUN_DIR/exit_code" ] || { echo "::error::The GPU pod restarted during the tests" >> "$RUN_DIR/log"; echo 1 > "$RUN_DIR/exit_code"; }
else
  touch "$RUN_DIR/started"
  mkdir -p /tmp/aklr-ci/src && cd /tmp/aklr-ci/src && tar -xf "$RUN_DIR/src.tar" &&
    bash .github/scripts/gpu_tests.sh >> "$RUN_DIR/log" 2>&1
  echo "$?" > "$RUN_DIR/exit_code"
fi
exec sleep infinity'
# One whole MI355X node.
payload=$(jq -n --arg image "$IMAGE" --arg entry "$(base64 -w0 <<< "$entry")" --arg workspace "$WORKSPACE" \
  --arg name "aklr-ci-$GITHUB_RUN_ID-${JOB%-test}" '{
    displayName: $name,
    description: "AKLR CI \(env.GITHUB_REPOSITORY) run \(env.GITHUB_RUN_ID) \(env.JOB)",
    workspaceId: $workspace,
    groupVersionKind: {kind: "Deployment", version: "v1"},
    images: [$image],
    entryPoints: [$entry],
    resources: [{replica: 1, gpu: "8", cpu: "96", memory: "512Gi", sharedMemory: "64Gi", ephemeralStorage: "300Gi"}],
    env: (env | {RUN_DIR, JOB, TESTS, VALIDATE, BASE, MEASURED_BY_BOT, AKLR_BASELINE_AITER, PLUGINS}
          | with_entries(select(.value != null and .value != ""))),
    # High (2 of 0-2): a CI run should not queue behind batch work.
    priority: 2,
    ttlSecondsAfterFinished: 60,
    useWorkspaceStorage: true
  }')
created=$(api -X POST "$SAFE_API/api/v1/workloads" -d "$payload") ||
  { echo "::error::Primus-SaFE refused the workload: $created"; exit 1; }
workload=$(jq -er .workloadId <<< "$created")
echo "🚀 Primus-SaFE workload $workload in $IMAGE"

tail -n +1 -F "$RUN_DIR/log" 2>/dev/null &
tail_pid=$!
# Always: the workload holds its GPU node until it is deleted, also after a cancelled or timed-out job.
trap 'kill "$tail_pid" 2>/dev/null || true
api -X DELETE "$SAFE_API/api/v1/workloads/$workload" > /dev/null || echo "::error::Could not delete workload $workload; delete it in Primus-SaFE"' EXIT

phase=
failures=0
until [[ -s $RUN_DIR/exit_code ]]; do
  sleep 15
  if detail=$(api "$SAFE_API/api/v1/workloads/$workload"); then
    failures=0
    new=$(jq -r .phase <<< "$detail")
    [[ $new == "$phase" ]] || echo "⏳ workload $workload: $new"
    phase=$new
    if [[ $phase =~ ^(Failed|Stopped)$ ]]; then
      echo "::error::Workload $workload ended $phase before the tests finished"
      jq -r '.conditions // [] | .[] | "\(.type // ""): \(.reason // "") \(.message // "")"' <<< "$detail"
      exit 1
    fi
  elif ((++failures >= 20)); then
    echo "::error::Lost the Primus-SaFE API for 5 minutes"
    exit 1
  fi
done
# The log's last lines reach the shared storage after the exit code does.
sleep 5
# The workload ran the PR's code: a link it left here would make the upload steps read the runner's files.
find "$RUN_DIR" -type l -delete
code=$(< "$RUN_DIR/exit_code")
[[ $code =~ ^[0-9]+$ ]] || { echo "::error::Unreadable exit code from workload $workload"; exit 1; }
exit "$code"
