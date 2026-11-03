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

AKLR neither ships nor patches [AITER](https://github.com/ROCm/aiter): AITER is only the
baseline AKLR's kernels are timed against, at the commit `AKLR_BASELINE_AITER` in
`.github/workflows/ci.yaml`, which CI installs over the image's AITER (with the FlyDSL it pins).
To time an op against an AITER function, call it from the op's `<contract>_baseline` in its
`baseline.py`. To move the baseline, change that commit: the next validation re-measures every
kernel against it.
