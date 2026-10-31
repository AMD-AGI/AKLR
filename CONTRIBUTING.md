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

## Changing AITER

AITER is the `3rdparty/aiter` submodule, pinned to the [ROCm/aiter](https://github.com/ROCm/aiter)
commit AKLR's routes describe. `pyproject.toml` packages it with `aklr` as one `amd-aiter`
distribution; `setup.py` reproduces AITER's default build (no kernel prebuild).

AKLR does not edit AITER. `aklr.ops.load_routes()`, which the SGLang and vLLM plugins
call at startup, replaces each AITER entry point an `@aklr_route` names with a proxy,
under every name a loaded module binds it to. To route a new AITER function, declare
its `@aklr_route`. To move to a newer AITER, update the routes to its signatures and the
submodule commit together.
