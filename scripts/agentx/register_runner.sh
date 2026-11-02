#!/usr/bin/env bash
# Register this MI355X host as a self-hosted runner for agentx-weekly.yaml (demo).
#
#   RUNNER_TOKEN=<registration token> scripts/agentx/register_runner.sh --hf-hub-dir /data/hf/hub
#
# The token comes from Settings → Actions → Runners → New self-hosted runner and is read
# from the environment only. Run as the user the service should run as (in the docker group);
# sudo is used only for the install directory and the systemd service.
set -euo pipefail

repo=AMD-AGI/AKLR
label=agentx-mi355x
version=2.338.2
dir=/opt/agentx-runner
name="$(hostname -s)-agentx"
hf_hub_dir=

usage() {
  sed -n '2,8p' "$0"
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
mem_gb=$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)
# The TP4 arm pins 4 × 180 GB of HiCache host memory plus the indexer (design §7.2).
(( mem_gb >= 1000 )) || die "host has ${mem_gb} GB of memory; the TP4 HiCache arm needs about 1 TB"
if [[ -n $hf_hub_dir ]]; then
  [[ -d $hf_hub_dir/models--amd--GLM-5.2-MXFP4 ]] ||
    echo "warning: $hf_hub_dir has no amd/GLM-5.2-MXFP4 snapshot yet; prefill it before the first run" >&2
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
echo "Next: set the repository variables AGENTX_HF_HUB_DIR=${hf_hub_dir:-<hub dir>}" \
  "(and AGENTX_RUNNER_LABEL if not $label), then dispatch agentx-weekly.yaml."
