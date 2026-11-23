# Tests

1. **TDD loop.** Load the `aklr:tdd` skill before writing or changing a test here, and follow its loop.
2. **Test at seams.** A test imports and calls only seams. A seam is a name imported from a path
   with no `_` segment: defined in a public module, or exported by a public package's `__init__`
   (`from aklr.core.runtime.dispatch import DispatcherRegistry`, though it lives in `_registry.py`).
   A name reached only through a `_` module, and any `_` attribute, is not one. Code behind a seam is tested
   through the seam. These are the `aklr:tdd` skill's pre-agreed seams: write the test at one directly;
   confirm with the user only an interface still undecided.
3. **GPU tests run on a GPU.** Tests marked `gpu` or `gfx950` are skipped on a machine without that
   GPU (`conftest.py`), and a skip is neither red nor green: run them on a GPU, a remote node from the
   `remote-gpu` skill if none is local.
4. **Layout.** `tests/<path>/test_<what>.py` tests code in `aklr/<path>/`; every directory has an
   `__init__.py`. The exceptions: `tests/benchmarks/test_agentx_*.py` test `benchmarks/agentx/` and
   `tests/benchmarks/test_kernels_report.py` tests `benchmarks/kernels/`, both outside `aklr/`. Code
   shared by test modules that is not a fixture goes in a plain module beside
   them (`agentx_results.py`). No module imports a `test_*.py` or a `conftest.py`.
5. **Fixtures.** A fixture starts in the test module that uses it. When a second module needs it,
   move it to the `conftest.py` of their nearest common directory. A `conftest.py` holds fixtures
   and hooks only.
   - Keep the default function scope. Widen it only for a value that is costly to build and that no
     test changes.
   - Tear down after `yield`.
   - `autouse` is for isolation only, as `aklr_env` is.
   - Use `tmp_path`, `monkeypatch`, `capsys` and `caplog` instead of writing your own.
6. **Test functions.** Write plain `def test_<behaviour>` functions with bare `assert`, not
   `unittest.TestCase`. One behaviour over several inputs is one test under
   `pytest.mark.parametrize`, with `ids` when the values do not read as names. Give `pytest.raises`
   a `match`. A test passes alone and in any order: it leaves no state behind in a module, a class
   or the process.
7. **Markers.** Markers are registered in `conftest.py`. A GPU requirement is the `gpu` or `gfx950`
   marker, never a skip written in the test.
8. **Checks.** Rule 4 is checked in part by `python3 tests/check_layout.py`; pre-commit runs it. Run
   it after changing a test and fix what it reports in the test; do not add an ignore.
