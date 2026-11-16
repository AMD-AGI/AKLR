#!/usr/bin/env bash
set -euo pipefail

repo=AMD-AGI/AKLR
label=agentx-mi355x
version=2.338.2
dir=/opt/agentx-runner
name="$(hostname -s)-agentx"
hf_hub_dir=

usage() {
  echo "usage: RUNNER_TOKEN=<registration token> $0 [options]"
  echo "options: --repo $repo --label $label --version $version --dir $dir --name $name --hf-hub-dir <dir>"
}
die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --repo) repo=$2; shift 2 ;;
    --label) label=$2; shift 2 ;;
    --version) version=$2; shift 2 ;;
    --dir) dir=$2; shift 2 ;;
    --name) name=$2; shift 2 ;;
    --hf-hub-dir) hf_hub_dir=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option $1" ;;
  esac
done

[[ -n ${RUNNER_TOKEN:-} ]] || die "set RUNNER_TOKEN to a runner registration token"
[[ $EUID -ne 0 ]] || die "run as the runner user, not root"
docker info >/dev/null 2>&1 || die "docker is not usable by $(id -un) (missing, or not in the docker group)"
[[ -e /dev/kfd && -d /dev/dri ]] || die "no ROCm devices (/dev/kfd, /dev/dri)"
if [[ -n $hf_hub_dir ]]; then
  [[ -d $hf_hub_dir ]] || die "HF hub directory does not exist: $hf_hub_dir"
  echo "Prepare the pinned snapshot and tokenizer refs/main for every enabled model in agentx.yaml."
fi

sudo mkdir -p "$dir"
sudo chown "$(id -un):$(id -gn)" "$dir"
cd "$dir"
tarball="actions-runner-linux-x64-$version.tar.gz"
curl -fsSL -o "$tarball" "https://github.com/actions/runner/releases/download/v$version/$tarball"
tar xzf "$tarball"
./config.sh --unattended --replace --url "https://github.com/$repo" --token "$RUNNER_TOKEN" \
  --name "$name" --labels "$label" --work _work
sudo ./svc.sh install "$(id -un)"
sudo ./svc.sh start

echo "Registered $name with label $label."
echo "Next: set AGENTX_HF_HUB_DIR=${hf_hub_dir:-<hub dir>}, set the target's runner_label to $label" \
  "in benchmarks/agentx/agentx.yaml, then dispatch agentx-ci.yaml."
