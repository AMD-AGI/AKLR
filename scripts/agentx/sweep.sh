#!/usr/bin/env bash
# Install and run the serving plan in its Docker image.
set -euo pipefail

serving='' hub='' out='' image='' cache='' work=''
die() { echo "error: $*" >&2; exit 1; }
usage() {
  echo "usage: $0 --serving <file> --hub <dir> --out <dir> [--image <image>] [--cache <dir>] [--work <dir>] [runtime options]"
}
while [[ $# -gt 0 ]]; do
  case $1 in
    --serving) serving=$2; shift 2 ;;
    --hub) hub=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --image) image=$2; shift 2 ;;
    --cache) cache=$2; shift 2 ;;
    --work) work=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) break ;;
  esac
done
[[ -n $out ]] || die "need --out <dir>"
mkdir -p "$out"
out=$(realpath "$out")
# Invalidate old results before setup can fail.
python3 - "$out/results.json" <<'PY'
import json, os, sys, uuid
path = sys.argv[1]
with open(path + '.tmp', 'w') as f:
    json.dump({'model': {}, 'run': {'completed': False, 'attempt_id': uuid.uuid4().hex}, 'points': []}, f)
os.replace(path + '.tmp', path)
PY
[[ -f $serving && -d $hub ]] || die "need --serving <file> --hub <dir>"
serving=$(realpath "$serving")
hub=$(realpath "$hub")
cache=${cache:-$out/cache}
mkdir -p "$cache"
cache=$(realpath "$cache")
planned_image=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["image"]["run"])' "$serving")
[[ -z $image || $image == "$planned_image" ]] || die "--image conflicts with the plan; regenerate it with --run-image"
image=$planned_image
work=${work:-$out/work}
mkdir -p "$work"
work=$(realpath "$work")
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
source_dirty=true
if source_status=$(git -C "$repo" status --porcelain --untracked-files=all) && [[ -z $source_status ]]; then
  source_dirty=false
fi

# Host networking shares localhost between the server and client.
docker run --rm --name "agentx-sweep-$(id -un)" --network host \
  --device=/dev/kfd --device=/dev/dri --ipc=host \
  --security-opt=seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$repo:/src:ro" -v "$hub:$hub:ro" -v "$serving:/agentx/serving.json:ro" -v "$out:/agentx/out" -v "$cache:/agentx/cache" -v "$work:/agentx/work" \
  -e HF_HUB_CACHE="$hub" -e HF_HUB_OFFLINE=1 -e GITHUB_SHA="$(git -C "$repo" rev-parse HEAD)" \
  -e AKLR_SOURCE_DIRTY="$source_dirty" \
  -e HOST_IDS="$(id -u):$(id -g)" \
  "$image" bash -c '
    set -euo pipefail
    trap "chown -R $HOST_IDS /agentx/out /agentx/cache /agentx/work" EXIT
    # Avoid copying NFS ACLs into the container.
    cp -r /src /aklr
    cd /aklr
    scripts/agentx/install_aklr.sh /agentx/cache 2>&1 | tee /agentx/out/install.log
    set -a; . /agentx/cache/aklr.env; set +a
    cd benchmarks/agentx
    python3 -m agentx_bench sweep --in-container --serving /agentx/serving.json \
      --work /agentx/work --out /agentx/out "$@"
  ' sweep "$@"
