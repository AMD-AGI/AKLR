#!/usr/bin/env bash
# Run the AgentX sweep once on this host, outside GitHub Actions, as the workflow's sweep job does.
#
#   scripts/agentx/run_local.sh --serving serving.json --hub /data/hf/hub --out results/ [sweep options]
#
# Copies this checkout into the AKLR CI image (serving.json's image.run), installs it there with
# install_aklr.sh and runs agentx_ci sweep; options after the known ones go to the sweep
# (--points tp8:1,tp4:4 --duration 600). The output directory is handed back to the caller's uid.
set -euo pipefail

serving= hub= out= image=
die() { echo "error: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case $1 in
    --serving) serving=$(realpath "$2"); shift 2 ;;
    --hub) hub=$(realpath "$2"); shift 2 ;;
    --out) out=$2; shift 2 ;;
    --image) image=$2; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
[[ -f $serving && -d $hub && -n $out ]] || die "need --serving <file> --hub <dir> --out <dir>"
mkdir -p "$out"
out=$(realpath "$out")
image=${image:-$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["image"]["run"])' "$serving")}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)

# Host networking: the client's uv/PyPI bootstrap crawls through the default bridge on some hosts.
# The server then listens on the host's 127.0.0.1, so the sweep's port check covers other users.
docker run --rm --name "agentx-sweep-$(id -un)" --network host \
  --device=/dev/kfd --device=/dev/dri --ipc=host \
  --security-opt=seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$repo:/src:ro" -v "$hub:$hub:ro" -v "$serving:/agentx/serving.json:ro" -v "$out:/agentx/out" \
  -e HF_HUB_CACHE="$hub" -e HF_HUB_OFFLINE=1 -e GITHUB_SHA="$(git -C "$repo" rev-parse HEAD)" \
  -e HOST_IDS="$(id -u):$(id -g)" \
  "$image" bash -c '
    set -euo pipefail
    trap "chown -R $HOST_IDS /agentx/out" EXIT
    cp -a /src /aklr && cd /aklr
    scripts/agentx/install_aklr.sh /agentx/out/cache 2>&1 | tee /agentx/out/install.log
    set -a; . /agentx/out/cache/aklr.env; set +a
    cd benchmarks/agentx
    python3 -m agentx_ci sweep --serving /agentx/serving.json \
      --work /agentx/out/work --out /agentx/out "$@"
  ' sweep "$@"
