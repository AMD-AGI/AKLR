#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# GPU-node entry point: download, register and run the official GitHub runner.
# The Host supplies a clean environment and passes a token FILE, not a secret
# in argv. Spur owns cleanup of the node-local directory, processes and containers.
set +x
set -euo pipefail
umask 077

# Initialized by parse_arguments and shared by the bootstrap steps below.
repository=''
runner_name=''
registration_token_file=''
runner_version=''
runner_sha256=''
runner_directory=''

die() {
    echo "ERROR: $*" >&2
    exit 1
}

parse_arguments() {
    if (($# != 5)); then
        echo 'Usage: run_gpu_runner.sh REPOSITORY NAME TOKEN_FILE VERSION SHA256' >&2
        exit 2
    fi

    repository=$1
    runner_name=$2
    registration_token_file=$3
    runner_version=$4
    runner_sha256=$5

    if [[ ! $repository =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ||
          ! $runner_name =~ ^aklr-[a-f0-9]{8}-[1-9][0-9]*-[1-9][0-9]*$ ]]; then
        echo 'ERROR: Invalid repository or runner name' >&2
        exit 2
    fi
    if [[ ! $runner_version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ||
          ! $runner_sha256 =~ ^[a-f0-9]{64}$ ]]; then
        echo 'ERROR: Invalid runner version or SHA256' >&2
        exit 2
    fi
}

validate_runtime() {
    if [[ $registration_token_file != /* || ! -f $registration_token_file ||
          -L $registration_token_file ]]; then
        die 'Registration token must be an absolute path to a regular file'
    fi
    if [[ $(stat -c '%u:%a' "$registration_token_file") != "$(id -u):600" ]]; then
        die 'Registration token must be owned by this user with mode 0600'
    fi
    if ((EUID == 0)); then
        die 'Run the GPU runner as a non-root user'
    fi
}

create_workspace() {
    runner_directory=$(mktemp -d /tmp/aklr-github-runner.XXXXXXXX)
    export HOME="$runner_directory"
    cd "$runner_directory"
}

download_runner() {
    local archive_name download_url actual_version

    archive_name="actions-runner-linux-x64-$runner_version.tar.gz"
    download_url="https://github.com/actions/runner/releases/download/v$runner_version/$archive_name"
    curl --fail --location --proto '=https' --retry 2 --max-time 180 \
        --output runner.tar.gz "$download_url"

    # Verify the archive before extracting or executing anything from it.
    printf '%s  runner.tar.gz\n' "$runner_sha256" | sha256sum --check --status
    tar -xzf runner.tar.gz
    rm -- runner.tar.gz

    actual_version=$(./bin/Runner.Listener --version)
    if [[ $actual_version != "$runner_version" ]]; then
        die 'Runner version mismatch'
    fi
}

register_runner() {
    local registration_token

    registration_token=$(cat "$registration_token_file")
    rm -- "$registration_token_file"
    if [[ ! $registration_token =~ ^[A-Za-z0-9_-]{20,4096}$ ]]; then
        die 'Invalid registration token'
    fi

    # Only config.sh receives the token. It never appears in command arguments
    # or in the later run.sh environment.
    (
        export ACTIONS_RUNNER_INPUT_TOKEN="$registration_token"
        timeout -k 5s 180s ./config.sh --unattended \
            --url "https://github.com/$repository" \
            --name "$runner_name" \
            --no-default-labels --labels "$runner_name" \
            --work "$runner_directory/work"
    )
    unset registration_token
}

# The runner always mounts /var/run/docker.sock into job containers. Some nodes'
# docker-authz plugin denies that mount, and no CI step needs it: the runner finds
# docker on PATH, so a wrapper ahead of the real one drops that one -v pair.
filter_docker_sock() {
    local docker_path

    docker_path=$(command -v docker) || die 'docker not found'
    mkdir "$runner_directory/docker-sock-filter"
    cat > "$runner_directory/docker-sock-filter/docker" << EOF
#!/usr/bin/env bash
args=()
while ((\$#)); do
    if [[ \$1 == -v && \${2:-} == /var/run/docker.sock:/var/run/docker.sock ]]; then
        shift 2
        continue
    fi
    args+=("\$1")
    shift
done
exec $(printf '%q' "$docker_path") "\${args[@]}"
EOF
    chmod 700 "$runner_directory/docker-sock-filter/docker"
}

run_runner() {
    # Keep a session runner for ci.yaml's sequential GPU jobs, with official
    # updates enabled. Spur and run.sh provide the process lifecycle management.
    exec env -i \
        HOME="$runner_directory" \
        PATH="$runner_directory/docker-sock-filter:/usr/local/bin:/usr/bin:/bin" \
        LANG=C.UTF-8 \
        ACTIONS_RUNNER_REQUIRE_JOB_CONTAINER=1 \
        ./run.sh
}

main() {
    parse_arguments "$@"
    validate_runtime
    create_workspace
    download_runner
    register_runner
    filter_docker_sock
    run_runner
}

main "$@"
