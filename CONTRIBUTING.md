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
performance tests, at the commit `AKLR_BASELINE_AITER` in
`.github/workflows/ci.yaml`, which CI installs over the image's AITER (with the FlyDSL it pins).

AKLR does not edit AITER. The dispatcher, which the SGLang and vLLM plugins start,
replaces each AITER entry point an `@adapter` in `aklr/seam/aiter.py` names with its
stand-in, under every name a loaded module binds it to; the adapter's baseline is what the
op is timed against. To route a new AITER function, declare its `@adapter`. To move the
baseline, change that commit and update the adapters to its signatures together: the next
validation re-measures every kernel against it.
