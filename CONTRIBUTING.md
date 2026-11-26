# Contributing to AKLR

## Code style and linting

We use [Ruff](https://docs.astral.sh/ruff/) for Python linting and formatting. All checks run
through [pre-commit](https://pre-commit.com/) and are enforced in CI.

Set up once after cloning:

```bash
pip install -r requirements.txt
pre-commit install
```

After this, the hooks run automatically on every `git commit`. To check all files manually
(e.g. before opening a PR):

```bash
pre-commit run --all-files
```

You can also run Ruff directly without pre-commit:

```bash
ruff check --fix .   # lint + auto-fix (import sorting, unused imports, ...)
ruff format .        # format code
```

Ruff rules live in `pyproject.toml`; tool versions are pinned in `.pre-commit-config.yaml` and
`requirements.txt`.

Style checks do not replace operator correctness tests or performance
qualification. Follow the workflows in [harness/README.md](harness/README.md)
for kernel lifecycle changes.

## The AITER baseline

[AITER](https://github.com/ROCm/aiter) is the baseline AKLR's kernels are timed against in
performance tests, at the commit `aiter` in `stack.json`, which CI installs over the image's
AITER (with the FlyDSL it pins). Install it the same way before validating locally:
`AITER_USE_SYSTEM_TRITON=1 pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@<that commit>"`.
To time an op against an AITER function, call it from the contract's baseline class in the op's
`baseline.py`. To move the baseline, change that commit: the next validation re-measures every
kernel against it.

## GPU CI runners

The `aiter-test`, `sglang-test` and `vllm-test` jobs use exactly one backend per workflow
run, after the lint/audit job passes:

| Trigger | Backend |
| --- | --- |
| PR with `ci:gpu` | `aklr-safe`: submit GPU workloads through the existing SaFE API |
| PR with `ci:gpu-direct` | `aklr-gpu`: execute tests in Docker on the runner, without SaFE |
| PR with both labels | `aklr-gpu` only; the direct label takes precedence |
| PR with neither label | Skip GPU tests |
| Push to `main` | `aklr-safe` (existing default) |
| Manual run | Choose `aklr-safe` or `aklr-gpu` in the `gpu_runner` dropdown; defaults to SaFE |

Selection is defined in `.github/workflows/ci.yaml`. Adding or removing a label restarts the PR
workflow and cancels its previous run.
To switch a direct PR run back to SaFE, remove `ci:gpu-direct` and keep `ci:gpu`.

SaFE workloads carry a priority; `aklr-gpu` has none:

| Trigger | SaFE priority |
| --- | --- |
| PR with `ci:gpu` | medium |
| PR with `ci:gpu` and `ci:priority-high` | high |
| Push to `main` | high: its run takes the published measurements |
| Manual run | Choose in the `gpu_priority` dropdown; defaults to medium |

`ci:priority-high` alone starts no GPU job. It stays on the PR, so every later push runs at high
too: remove it once the urgent run is done.

An `aklr-gpu` runner needs Linux, Docker access, ROCm GPU devices (`/dev/kfd` and `/dev/dri`),
network access to the images in `stack.json`, and enough local disk for image pulls, source
copies and builds. Docker must run on the runner host. Dedicate the visible GPUs to this
runner: each test container uses its visible devices, and multiple runner services on the
same GPU host would run competing tests. One runner service executes one job at a time.
The three test jobs may run in parallel when multiple runners are available.

The local container receives only the test environment variables and a copy of the checkout.
Its per-stage timeout is unchanged, failing tests still upload reports, and the final cleanup
step removes the job's container. This does not change the separate AgentX workflow.
There is no automatic fallback between backends when the selected runners are busy or offline.

For PRs changing only kernel schemas under `aklr/plugins/aiter` (optionally with Markdown case
notes under `harness/skills/kernel-optimization-book/cases/`), AITER's GPU baseline/reference comparison
runs every small point for the affected contract IDs. A replacement includes the IDs
from both versions. Unrecognized changes or a contract without a small point use the
full comparison. The other unit tests and per-image kernel validation keep their usual
coverage. Pushes and manual runs perform the full unit suite.
