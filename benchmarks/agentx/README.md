# Weekly AgentX end-to-end CI (demo)

A demo of the weekly GLM-5.2 MXFP4 · MI355X · sglang + AKLR run: the end-to-end flow, not the
full design. Each week it:

1. **resolves** InferenceX `main`'s `glm5.2-fp4-mi355x-sglang-agentic-mtp` entry and its recipe
   (`glm5.2/sglang/mi355x-fp4-mtp/agentic.yaml`) into one launch spec per point: 2 arms
   (`tp8`: c1, c2, c4; `tp4` + HiCache: c4 … c16), 9 points. The grid, the serving
   configuration and the client all come from that one InferenceX commit, through
   InferenceX's own matrix and binding code (`agentx_ci/ix_points.py`);
2. **sweeps** the points one at a time on an MI355X self-hosted runner, in the AKLR CI image
   (`AKLR_CI_IMAGE` of `ci.yaml`) with this checkout installed: start sglang, wait for
   `/health` and `/v1/models`, run InferenceX's AgentX client (`python3 -m infx.bench agentic`),
   stop the server;
3. **reports** per-arm and combined Pareto frontiers (throughput per GPU against P90 E2E
   interactivity), a week-over-week comparison with the previous published run, and what
   landed on AKLR main since (PRs and direct commits, from this checkout's git history), as a
   static page;
4. **publishes** the page to GitHub Pages.

```
.github/workflows/agentx-weekly.yaml   plan → sweep (or sweep-mock) → report → pages
benchmarks/agentx/
  agentx.yaml                          the InferenceX entry, arms, weights, host settings
  agentx_ci/recipe_resolve.py          fetch + resolve → serving.json
  agentx_ci/ix_points.py               runs inside the InferenceX checkout, its code only
  agentx_ci/sweep.py                   serving.json → results.json (live or --mock)
  agentx_ci/pareto.py                  frontiers, whole-arm week-over-week comparison
  agentx_ci/prs.py                     git log --first-parent between two AKLR commits
  agentx_ci/report.py                  index.html, results.json, report.json
  tests/                               unittest; data/ holds InferenceX 70a7601's own output
scripts/agentx/register_runner.sh      register an MI355X host as a runner
```

## Run it locally in mock mode

Needs Python ≥ 3.10, PyYAML, git, and network access to GitHub for the InferenceX fetch; no GPU.

```bash
cd benchmarks/agentx
python3 -m unittest discover -s tests -t .

# 1. Resolve InferenceX main (a sparse clone, about 5 s; add --ix-ref <sha> to pin it).
python3 -m agentx_ci resolve --work /tmp/agx/work --out /tmp/agx/serving.json

# 2. Two fake weeks; the second scales tp4's throughput by 6 %. A run records its AKLR commit
#    from $GITHUB_SHA (else HEAD): week 1 pretends to have run five commits ago.
GITHUB_SHA=$(git rev-parse HEAD~5) python3 -m agentx_ci sweep --mock --mock-seed 1 --run-id 101 \
  --serving /tmp/agx/serving.json --out /tmp/agx/week1
python3 -m agentx_ci report --results /tmp/agx/week1/results.json --out /tmp/agx/site1
python3 -m agentx_ci sweep --mock --mock-seed 2 --mock-scale tp4=1.06 --run-id 102 \
  --serving /tmp/agx/serving.json --out /tmp/agx/week2

# 3. Week 2 against week 1, with what landed between the two runs' AKLR commits.
python3 -m agentx_ci report --results /tmp/agx/week2/results.json \
  --baseline /tmp/agx/site1/results.json --out /tmp/agx/site2
xdg-open /tmp/agx/site2/index.html
```

Mock results are never compared with live ones; the page carries a `MOCK DATA` badge.

### What landed since the previous run

`git log --first-parent <previous run's AKLR sha>..<this run's sha>` on the checkout given by
`--git-dir` (default: this repository). A subject ending in `(#123)` (squash merge) or
`Merge pull request #123 from …` (merge commit, title from the body) is PR #123, linked under
`report.repo_url` of `agentx.yaml`; any other commit is listed as a direct commit, with
author and date. No GitHub API, no token. The first run has no previous run and lists
nothing. A shallow checkout is unshallowed, and a previous sha still missing is fetched from
`origin`; when that fails the page says so (the workflow checks out with `fetch-depth: 0`). A
previous sha that is not an ancestor of this run's (a rewritten branch) is an error too.

## Run it for real

1. On the MI355X host (≥ 1 TB memory, Docker, ROCm devices), with the pinned
   `amd/GLM-5.2-MXFP4` snapshot and the `semianalysisai/cc-traces-weka-062126` dataset in an
   HF hub cache: `RUNNER_TOKEN=… scripts/agentx/register_runner.sh --hf-hub-dir <hub>`.
2. Repository variables: `AGENTX_HF_HUB_DIR` (that hub directory), and optionally
   `AGENTX_RUNNER_LABEL` (default `agentx-mi355x`), `AGENTX_PAGES_URL` (where the previous
   `results.json` is read from), `AGENTX_SCHEDULE_ENABLED=true` to turn on the cron. Enable
   Pages with "GitHub Actions" as the source.
3. Dispatch `agentx-weekly.yaml`; `duration_s` shortens every point for a smoke run, `mock`
   runs the whole workflow on hosted runners.

The cadence is the `cron` line of the workflow. A full run takes about 15 h (9 points × ~90
min, each point restarts the server).

## Schemas

- `serving.json`: provenance (InferenceX repo, sha, commit, source blobs, deps), images (run and
  InferenceX's; they differ in ROCm, accepted), `arms` with `arm_serve_sha256`, `grid_sha256`,
  and per point `launch.command`, `launch.env`, `client_env`, `hashes.serve_spec_sha256`.
- `results.json`: `run` (id, mode, AKLR sha, host, images, InferenceX), `serving` (arm hashes,
  grid, warnings), and per point `status`, `tput_per_gpu`, `p90_e2e_norm_intvty`,
  `serve_spec_sha256`, `error`.

An arm is compared with the previous run only when its whole `arm_serve_sha256` matches;
otherwise it starts a new segment and the per-point hashes only say which points changed.

## Not in the demo (TODO)

- **Multi-host series**: `hosts.yaml`, a matrix over hosts, per-series baselines, the manual
  inbox upload path.
- **Last-good fallback and issue alerts**: today a failed resolve fails the run; the design
  falls back to the last good serving config (≤ 28 days, ≤ 3 runs) and then opens an issue.
- **Sandboxing**: InferenceX's resolution code runs as a plain child process with a minimal
  environment, not in a network-less, read-only container.
- **argcheck**: no check of the rendered flags and env against the run image's sglang.
- **Cookbook divergence table**: the sglang cookbook reference path is not included.
- **History store**: the baseline is only the previously published `results.json`; there is
  no `agentx-data` branch, trend, noise model (σ) or regression verdict.
- Shared `setup_aklr.sh` with `ci.yaml`, HF prefetch and verification, preflight GPU checks
  and fingerprints, per-arm JIT warm-up, job splitting under the 24 h limit, the D1 / D2
  diagnostics, and the monthly accuracy run.
