# Adding an Op for a Ported Kernel

When no contract covers the kernel, add the op before its schema. The layout and the meaning of
each file are in `aklr/ops/README.md`; this is the order and the traps.

1. **`aklr/ops/<op>/contract.py`**: the `OpContract` constants and `CONTRACTS`. Name the axes,
   inputs and outputs after the math, not after the baseline's arguments. A contract with several outputs
   names none of them `out` (`outputs are 'out' alone, or names other than 'out'`); give each output
   its `error_bounds` and the contract its `determinism` (`bitwise` unless the op's kernels need
   atomics).
2. **`reference.py`**: one `<contract>_reference(point, row_major_args, decode)` per contract,
   in the dtype `decode` returns (float32 in validation), returning the outputs in contract order.
   Any rounding every correct kernel must do belongs here.
3. **`api.py`**: one `@api(<CONTRACT>)` stub per contract. Example: `aklr/ops/gemm/api.py`.
4. **`baseline.py`**: what kernels are timed against, one `<contract>_baseline` per contract that
   has one. It takes the api's params (called with the call's keyword arguments), imports AITER or
   SGLang inside the function and returns the outputs as the api does; a call it does not serve
   raises `NotImplementedError` (bench: `baseline_failed`). Example: `aklr/ops/gemm/baseline.py`.
   Without one the op is its own baseline: a kernel has nothing to beat.
   An op every rank of a group calls together (expert parallelism, collectives) sets the contract's
   `ranks` to the attr holding the group size; its reference takes `(point, group_args, rank,
   decode)`, and `group.py`'s `<contract>_group_init()` sets up what an engine would before the
   first call. See `aklr/ops/README.md`.
5. **`__init__.py`**: export the api functions.
6. **Tests**: add the contract's base Point, small, to `_SMALL_POINTS` in `tests/op_case.py` (contract
   order), so the contract and reference suites cover it, and regenerate the variant names with
   `AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py` (`tests/ops/names.txt`).
7. **Schemas**: `python -m aklr export` from a `AKLR_EXPORT` capture of the model, or narrow the
   contract to chosen Points with a script; say which in the PR. Then fill `kernel.sources`.

Such a PR changes code, so CI validates every committed schema, not only the new one
(`read-validation`, ci.md): its kernel waits for that run.
