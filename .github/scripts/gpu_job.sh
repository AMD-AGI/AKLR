#!/usr/bin/env bash
set -euo pipefail

# Use the same test body and report locations for both execution backends.
case ${AKLR_GPU_RUNNER:-aklr-safe} in
  aklr-safe) exec python3 .github/scripts/safe_gpu_job.py "$@" ;;
  aklr-gpu) ;;
  *) echo '::error::AKLR_GPU_RUNNER must be aklr-safe or aklr-gpu'; exit 1 ;;
esac

name=aklr-ci-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-$GITHUB_JOB

if [[ ${1:-} == delete ]]; then
  # An earlier checkout/setup failure may have left no container.
  if docker container inspect "$name" > /dev/null 2>&1; then
    docker rm --force "$name"
  fi
  exit
fi

script=${1:?usage: gpu_job.sh <script> | delete}
[[ -f $script ]]
: "${IMAGE:?}" "${TIMEOUT_MINUTES:?}"
[[ $TIMEOUT_MINUTES =~ ^[1-9][0-9]*$ ]]

# Pass only the variables the test body needs, as on SaFE. Do not inherit host credentials.
pass_env=()
read -r -a variables <<< "${PASS_ENV:-}"
for variable in "${variables[@]}"; do
  [[ $variable =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
  [[ ! -v $variable ]] || pass_env+=(--env "$variable=${!variable}")
done

docker create --name "$name" \
  --device=/dev/kfd --device=/dev/dri --ipc=host \
  --security-opt=seccomp=unconfined --ulimit memlock=-1:-1 \
  --workdir /tmp/aklr-src --env REPORTS=/tmp/aklr-reports \
  --env "TIMEOUT_MINUTES=$TIMEOUT_MINUTES" "${pass_env[@]}" \
  --entrypoint bash "$IMAGE" -ec \
  'mkdir -p "$REPORTS"; exec timeout --signal=TERM --kill-after=60s "${TIMEOUT_MINUTES}m" bash /tmp/aklr-job.sh'

# Copy into the container so editable installs, JIT and tests do not change the host checkout.
# The checkout includes its Git history, used to select the validation range.
docker cp "$GITHUB_WORKSPACE/." "$name:/tmp/aklr-src"
docker cp "$script" "$name:/tmp/aklr-job.sh"

# docker start's status is not the workload exit status; inspect it explicitly.
status=0
docker start --attach "$name" || status=$?
exit_code=$(docker inspect --format '{{.State.ExitCode}}' "$name")
mkdir -p "$RUNNER_TEMP/gpu-reports"
# Preserve reports from failing tests too; missing reports must fail a successful workload.
docker cp "$name:/tmp/aklr-reports/." "$RUNNER_TEMP/gpu-reports/" || status=1
if [[ $exit_code != 0 ]]; then
  exit "$exit_code"
fi
exit "$status"
