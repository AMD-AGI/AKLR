#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Manage one workflow's GPU runner:
#   manage_spur_runner.sh prepare|release [allocation-id] [runner-name]
#
# Requires Bash, gh (repository runner administration), jq, timeout and Spur.
# gpu-node.yaml supplies the Spur settings and the official runner version/checksum.
# AKLR_SPUR_SHARED_DIR must be private and accessible at the same absolute path
# on the Host and GPU nodes. Bootstrap scripts/logs remain there for diagnosis;
# the short-lived registration token is removed after use.
#
# AKLR_SPUR_ACCOUNT and AKLR_SPUR_PARTITION each hold one or more names separated
# by commas and/or whitespace. An account may be written ACCOUNT:QOS to submit under
# that QOS; a bare account uses its default QOS. prepare submits one allocation per account and
# partition pair under the same job name; the first to start running is kept and
# every other one is cancelled. The registration token is written only once every
# other allocation has ended, so the kept allocation is the only one that can
# register the runner; its bootstrap downloads the runner meanwhile and waits for it.
#
# release takes the runner name prepare published, so a re-run attempt that reuses
# an earlier attempt's outputs releases that attempt's resources; the name must
# belong to this run. Without it, the name of the current attempt is used.
#
# Spur owns node/process/container cleanup. This script only submits or cancels
# the exact allocations and manages its GitHub runner registration.
set +x
set -euo pipefail
umask 077

readonly COMMAND_TIMEOUT_SECONDS=30
readonly RUNNER_ONLINE_TIMEOUT_SECONDS=1200
readonly CANCEL_TIMEOUT_SECONDS=120
readonly POLL_INTERVAL_SECONDS=2

# Shared by prepare, release and the prepare EXIT handler. No state DB is used.
runner_name=''
allocation_id=''
allocation_ids=()
declare -A allocation_logs=()
allocation_owner=''
spur_accounts=()
spur_partitions=()
handoff_dir=''
submission_started=false
runner_ready=false

# ---- Command setup ----------------------------------------------------------

die() {
    echo "ERROR: $*" >&2
    exit 1
}

github_api() {
    timeout -k 5s "${COMMAND_TIMEOUT_SECONDS}s" gh api --hostname github.com "$@"
}

run_spur() {
    timeout -k 5s "${COMMAND_TIMEOUT_SECONDS}s" spur "$@"
}

parse_arguments() {
    operation=${1:-}
    case "$operation" in
    prepare | release) ;;
    *) die 'Usage: manage_spur_runner.sh prepare|release [allocation-id] [runner-name]' ;;
    esac

    if (($# > 3)); then
        die 'Too many arguments'
    fi
    if [[ $operation == prepare && $# != 1 ]]; then
        die 'prepare takes no arguments'
    fi

    allocation_id=${2:-}
    if [[ -n $allocation_id && ! $allocation_id =~ ^[1-9][0-9]*$ ]]; then
        die 'Invalid allocation ID'
    fi
    runner_name=${3:-}
}

load_workflow_context() {
    local shared_dir repository_hash run_prefix

    [[ ${GITHUB_REPOSITORY:-} =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die 'Invalid GITHUB_REPOSITORY'
    [[ ${GITHUB_RUN_ID:-} =~ ^[1-9][0-9]{0,19}$ ]] || die 'Invalid GITHUB_RUN_ID'
    [[ ${GITHUB_RUN_ATTEMPT:-} =~ ^[1-9][0-9]{0,5}$ ]] || die 'Invalid GITHUB_RUN_ATTEMPT'

    # The same identity is used for the Spur job, GitHub runner and routing label.
    repository_hash=$(printf '%s' "$GITHUB_REPOSITORY" | sha256sum)
    run_prefix="aklr-${repository_hash:0:8}-${GITHUB_RUN_ID}"
    runner_name=${runner_name:-$run_prefix-$GITHUB_RUN_ATTEMPT}
    [[ $runner_name =~ ^$run_prefix-[1-9][0-9]{0,5}$ ]] || die 'Runner name does not belong to this run'
    allocation_owner=$(id -un)

    shared_dir=${AKLR_SPUR_SHARED_DIR:?Set AKLR_SPUR_SHARED_DIR to a private shared directory}
    if [[ $shared_dir != /* || ! -d $shared_dir || $(realpath -e "$shared_dir") != "$shared_dir" ]]; then
        die 'Shared directory must be canonical and already exist'
    fi
    if [[ $(stat -c '%u:%a' "$shared_dir") != "$(id -u):700" ]]; then
        die 'Shared directory must be owned by this user with mode 0700'
    fi

    handoff_dir="$shared_dir/$runner_name"
    [[ ! -L $handoff_dir ]] || die 'Refusing a symlink handoff'
}

load_github_credentials() {
    local token_file=${AKLR_GITHUB_TOKEN_FILE:-}

    # Otherwise gh uses the Host's existing authentication or GH_TOKEN.
    if [[ -z $token_file ]]; then
        return 0
    fi
    if [[ ! -f $token_file || -L $token_file || $(stat -c '%u:%a' "$token_file") != "$(id -u):600" ]]; then
        die 'GitHub token file must be owned/private (0600)'
    fi

    GH_TOKEN=$(cat "$token_file")
    export GH_TOKEN
}

# load_name_list SETTING LIST [QOS_ALLOWED]: with QOS_ALLOWED, a name may carry a :QOS suffix.
load_name_list() {
    local setting=$1 value=${!1:-} name
    local -n list=$2
    local valid='[A-Za-z0-9][A-Za-z0-9_.-]{0,63}' pattern

    pattern="^$valid\$"
    [[ -z ${3:-} ]] || pattern="^$valid(:$valid)?\$"
    read -ra list <<< "${value//,/ }"
    ((${#list[@]} > 0)) || die "Set $setting"
    for name in "${list[@]}"; do
        [[ $name =~ $pattern ]] || die "Invalid name in $setting: $name"
    done
}

load_prepare_settings() {
    load_name_list AKLR_SPUR_ACCOUNT spur_accounts qos
    load_name_list AKLR_SPUR_PARTITION spur_partitions

    cpus_per_task=${AKLR_SPUR_CPUS:-64}
    memory_per_node=${AKLR_SPUR_MEMORY:-512G}
    gpus_per_node=${AKLR_SPUR_GPUS_PER_NODE:-mi355x:8}
    allocation_time_limit=${AKLR_SPUR_TIME_LIMIT:-8:00:00}
    exclude_nodes=${AKLR_SPUR_EXCLUDE:-}

    [[ $cpus_per_task =~ ^[1-9][0-9]*$ ]] || die 'Invalid AKLR_SPUR_CPUS'
    [[ $memory_per_node =~ ^[1-9][0-9]*[KMGTkmgt]?$ ]] || die 'Invalid AKLR_SPUR_MEMORY'
    [[ $gpus_per_node =~ ^([A-Za-z0-9_-]+:)?[1-9][0-9]*$ ]] || die 'Invalid AKLR_SPUR_GPUS_PER_NODE'
    [[ $allocation_time_limit =~ ^[1-9][0-9]*:[0-5][0-9]:[0-5][0-9]$ ]] || die 'Time limit must be H:MM:SS'
    [[ -z $exclude_nodes || $exclude_nodes =~ ^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$ ]] || die 'Invalid AKLR_SPUR_EXCLUDE'

    [[ ${AKLR_RUNNER_VERSION:-} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die 'Set the official runner version'
    [[ ${AKLR_RUNNER_SHA256:-} =~ ^[a-f0-9]{64}$ ]] || die 'Set the official runner SHA256'
    [[ -f ${GITHUB_OUTPUT:-} && ! -L $GITHUB_OUTPUT ]] || die 'GITHUB_OUTPUT must be a regular file'
}

# ---- Exact resource lookup ---------------------------------------------------

# Print the queued allocation IDs, one per line. Lost submit responses are
# resolved by exact name and owner; unexpected rows never trigger a retry.
find_allocation_ids() {
    local queue_rows job_id job_name job_owner extra_fields

    queue_rows=$(run_spur queue --user "$allocation_owner" --name "$runner_name" \
        --noheader --format '%i|%j|%u') || return 1

    while IFS='|' read -r job_id job_name job_owner extra_fields; do
        [[ -n $job_id ]] || continue
        if [[ ! $job_id =~ ^[1-9][0-9]*$ || $job_name != "$runner_name" ||
              $job_owner != "$allocation_owner" || -n $extra_fields ]]; then
            echo 'Unexpected Spur queue row; inspect manually' >&2
            return 1
        fi
        printf '%s\n' "$job_id"
    done <<< "$queue_rows"
}

# Validate ownership before returning a status that could authorize cancellation.
read_allocation_status() {
    local target_id=$1
    local job_details field job_id='' job_name='' job_owner='' job_status=''

    job_details=$(run_spur show job "$target_id") || return 1
    for field in $job_details; do
        case "$field" in
        JobId=*) job_id=${field#*=} ;;
        JobName=*) job_name=${field#*=} ;;
        UserId=*) job_owner=${field#*=} ;;
        JobState=*) job_status=${field#*=} ;;
        esac
    done

    if [[ $job_id != "$target_id" || $job_name != "$runner_name" || -z $job_status ||
          ($job_owner != "$allocation_owner" && $job_owner != "$allocation_owner($(id -u))") ]]; then
        echo 'Allocation identity mismatch; refusing action' >&2
        return 1
    fi
    printf '%s\n' "${job_status%+}"
}

is_terminal_status() {
    case "$1" in
    COMPLETED | CANCELLED | FAILED | TIMEOUT | NODE_FAIL | PREEMPTED | OUT_OF_MEMORY | BOOT_FAIL | DEADLINE)
        return 0
        ;;
    *)
        return 1
        ;;
    esac
}

# Print one matching runner as JSON, or null. Require both its name and label.
find_github_runner() {
    local runner_pages

    runner_pages=$(github_api --paginate \
        "repos/$GITHUB_REPOSITORY/actions/runners?per_page=100") || return 1

    # jq collects page objects; this also works with gh versions lacking --slurp.
    printf '%s' "$runner_pages" | jq -scr --arg runner_name "$runner_name" '
        [.[].runners[] | select(.name == $runner_name)] |
        if length == 0 then
            null
        elif length == 1
            and (.[0].labels | any(.name == $runner_name))
            and (.[0].id | type == "number" and . > 0 and floor == .)
        then
            .[0]
        else
            error("Runner identity mismatch")
        end'
}

# ---- Release: let Spur clean the node, then remove GitHub registration --------

cancel_allocation() {
    local target_id=$1
    local allocation_status cancel_deadline

    allocation_status=$(read_allocation_status "$target_id") || return 1
    if ! is_terminal_status "$allocation_status"; then
        # Cancel PENDING jobs too: no node acknowledgement is required.
        if ! run_spur cancel "$target_id"; then
            echo 'Cancel returned an error; checking scheduler state' >&2
        fi

        cancel_deadline=$((SECONDS + CANCEL_TIMEOUT_SECONDS))
        while :; do
            allocation_status=$(read_allocation_status "$target_id") || return 1
            if is_terminal_status "$allocation_status"; then
                break
            fi
            if ((SECONDS >= cancel_deadline)); then
                echo "Cancellation unconfirmed: $target_id" >&2
                return 1
            fi
            sleep "$POLL_INTERVAL_SECONDS"
        done
    fi
    echo "Spur allocation $target_id ended ($allocation_status)."
}

# Print the known and queued allocation IDs except $1, without duplicates.
list_other_allocations() {
    local keep_id=$1 queued_ids id
    local -A seen=()

    [[ -z $keep_id ]] || seen[$keep_id]=1
    queued_ids=$(find_allocation_ids) || return 1
    for id in "${allocation_ids[@]}" $queued_ids; do
        [[ -z ${seen[$id]:-} ]] || continue
        seen[$id]=1
        printf '%s\n' "$id"
    done
}

remove_github_runner() {
    local runner_json runner_id

    runner_json=$(find_github_runner) || return 1
    if [[ $runner_json == null ]]; then
        return 0
    fi

    runner_id=$(jq -r '.id' <<< "$runner_json") || return 1
    if ! github_api --method DELETE "repos/$GITHUB_REPOSITORY/actions/runners/$runner_id"; then
        # A concurrent release may have deleted it already.
        runner_json=$(find_github_runner) || return 1
        [[ $runner_json == null ]] || return 1
    fi
}

release_runner() {
    # Explicit error propagation is required: the EXIT handler also calls this
    # function in an if condition, where Bash disables implicit errexit.
    local other_ids id cancel_failed=false
    local -a release_ids=()

    other_ids=$(list_other_allocations "$allocation_id") || return 1
    [[ -z $allocation_id ]] || release_ids+=("$allocation_id")
    for id in $other_ids; do
        release_ids+=("$id")
    done
    if ((${#release_ids[@]} == 0)) && [[ $submission_started == true ]]; then
        echo "Submission outcome unknown for $runner_name; inspect Spur manually, do not resubmit" >&2
        return 1
    fi

    if ((${#release_ids[@]} == 0)); then
        echo "No active Spur allocation named $runner_name."
    fi
    for id in "${release_ids[@]}"; do
        cancel_allocation "$id" || cancel_failed=true
    done
    [[ $cancel_failed == false ]] || return 1

    # End the allocation first, so bootstrap cannot register after our deletion.
    remove_github_runner || return 1
    rm -f -- "$handoff_dir/registration-token" "$handoff_dir/registration-token.partial" || return 1
    echo "Released $runner_name; node cleanup is handled by Spur."
}

# ---- Prepare: bootstrap input, one submission, queue and runner waits --------

on_prepare_exit() {
    local exit_status=$?
    trap - EXIT INT TERM

    # The token is written only after submission, and release removes it.
    if [[ $runner_ready != true && $submission_started == true ]] && ! release_runner; then
        echo "::error::Release incomplete; inspect Spur job $runner_name and GitHub runner $runner_name" >&2
    fi
    exit "$exit_status"
}

# Renamed into place, so a waiting bootstrap never reads a partial token.
write_registration_token() {
    local registration_token

    registration_token=$(github_api --method POST \
        "repos/$GITHUB_REPOSITORY/actions/runners/registration-token" --jq '.token')
    [[ $registration_token =~ ^[A-Za-z0-9_-]{20,4096}$ ]] || die 'Invalid GitHub registration token response'
    printf '%s' "$registration_token" > "$handoff_dir/registration-token.partial"
    unset registration_token
    mv -- "$handoff_dir/registration-token.partial" "$handoff_dir/registration-token"
}

submit_allocations() {
    local bootstrap_command submit_response entry account qos partition log
    local -a submit_args

    printf -v bootstrap_command 'exec env -i PATH=/usr/local/bin:/usr/bin:/bin /usr/bin/bash %q %q %q %q %q %q' \
        "$handoff_dir/run_gpu_runner.sh" "$GITHUB_REPOSITORY" "$runner_name" \
        "$handoff_dir/registration-token" "$AKLR_RUNNER_VERSION" "$AKLR_RUNNER_SHA256"

    # Submit each pair exactly once. Uncertain responses are never retried; the
    # EXIT handler and release resolve them by job name.
    for entry in "${spur_accounts[@]}"; do
        account=${entry%%:*} qos=''
        [[ $entry != *:* ]] || qos=${entry#*:}
        for partition in "${spur_partitions[@]}"; do
            log="$handoff_dir/allocation-${entry/:/-}-$partition.log"
            submit_args=(
                submit --parsable
                -A "$account" -p "$partition"
                -N 1 -c "$cpus_per_task" --mem "$memory_per_node"
                "--gpus-per-node=$gpus_per_node" -t "$allocation_time_limit"
                -J "$runner_name" --export=NONE
                --chdir "$handoff_dir" --output "$log"
                --wrap "$bootstrap_command"
            )
            [[ -z $qos ]] || submit_args+=(-q "$qos")
            [[ -z $exclude_nodes ]] || submit_args+=(--exclude "$exclude_nodes")

            submission_started=true
            if submit_response=$(run_spur "${submit_args[@]}") &&
                [[ $submit_response =~ ^[1-9][0-9]*$ ]]; then
                allocation_ids+=("$submit_response")
                allocation_logs[$submit_response]=$log
                echo "Submitted $submit_response (account $entry, partition $partition)."
            else
                echo "::warning::Spur submit for account $entry, partition $partition failed or was uncertain; not retried" >&2
            fi
        done
    done
    ((${#allocation_ids[@]} > 0)) || die 'No Spur submission succeeded; will not resubmit'
}

publish_workflow_outputs() {
    {
        printf 'allocation_id=%s\n' "$allocation_id"
        printf 'runner_label=%s\n' "$runner_name"
    } >> "$GITHUB_OUTPUT"
    echo "Using $allocation_id ($runner_name); bootstrap log: ${allocation_logs[$allocation_id]}"
}

# Keep the first allocation seen RUNNING. Ended ones are dropped from the race.
# No deadline: a queued allocation waits for as long as Spur keeps it queued.
wait_for_allocation() {
    local id allocation_status
    local -a waiting_ids

    while :; do
        waiting_ids=()
        for id in "${allocation_ids[@]}"; do
            allocation_status=$(read_allocation_status "$id")
            if [[ $allocation_status == RUNNING ]]; then
                allocation_id=$id
                return 0
            elif is_terminal_status "$allocation_status"; then
                echo "Allocation $id ended before bootstrap: $allocation_status" >&2
            else
                waiting_ids+=("$id")
            fi
        done
        allocation_ids=("${waiting_ids[@]}")
        ((${#allocation_ids[@]} > 0)) || die 'Every Spur allocation ended before bootstrap'
        sleep "$POLL_INTERVAL_SECONDS"
    done
}

# Every other allocation must have ended before the token is written: one still
# running could take it. Failing here lets the EXIT handler release them all.
cancel_other_allocations() {
    local other_ids id

    other_ids=$(list_other_allocations "$allocation_id") || die 'Could not list the other allocations'
    for id in $other_ids; do
        cancel_allocation "$id" || die "Could not cancel allocation $id"
    done
    allocation_ids=("$allocation_id")
}

wait_for_runner() {
    local online_deadline=$((SECONDS + RUNNER_ONLINE_TIMEOUT_SECONDS))
    local allocation_status runner_json

    while :; do
        allocation_status=$(read_allocation_status "$allocation_id")
        if [[ $allocation_status != RUNNING ]]; then
            die "Allocation left RUNNING before runner came online: $allocation_status"
        fi
        runner_json=$(find_github_runner)
        if ((SECONDS >= online_deadline)); then
            die 'Runner online deadline exceeded'
        fi
        if [[ $runner_json != null && $(jq -r '.status' <<< "$runner_json") == online ]]; then
            return 0
        fi
        sleep "$POLL_INTERVAL_SECONDS"
    done
}

prepare_runner() {
    local existing_allocations existing_runner

    load_prepare_settings
    existing_allocations=$(find_allocation_ids)
    [[ -z $existing_allocations ]] || die "Allocation ${existing_allocations//$'\n'/ } already exists; do not submit twice"
    existing_runner=$(find_github_runner)
    [[ $existing_runner == null ]] || die 'Runner already exists; use Re-run all jobs for a fresh attempt'

    # Exclusive creation prevents retries of uncertain submissions that may no
    # longer appear in the queue. Only our own preparation gets an EXIT handler.
    mkdir -m 700 -- "$handoff_dir"
    trap on_prepare_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    # Snapshot the trusted script before submitting: another Host checkout must
    # not alter a queued allocation's bootstrap. The Host API token stays local.
    cp -- "$(dirname -- "${BASH_SOURCE[0]}")/run_gpu_runner.sh" "$handoff_dir/run_gpu_runner.sh"
    submit_allocations
    wait_for_allocation
    cancel_other_allocations
    write_registration_token
    publish_workflow_outputs
    wait_for_runner

    runner_ready=true
    echo "Runner $runner_name is online. For retries use Re-run all jobs, not Re-run failed jobs."
}

main() {
    parse_arguments "$@"
    load_workflow_context
    load_github_credentials

    case "$operation" in
    prepare) prepare_runner ;;
    release) release_runner ;;
    esac
}

main "$@"
