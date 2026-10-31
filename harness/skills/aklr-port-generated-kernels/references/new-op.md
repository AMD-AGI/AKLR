# Adding an Op for a Ported Kernel

When no contract covers the kernel, add the op before its schema. The layout and the meaning of
each file are in `aklr/ops/README.md`; this is the order and the traps.

1. **`aklr/ops/<op>/contract.py`**: the `OpContract` constants and `CONTRACTS`. Name the axes,
   inputs and outputs after the math, not after AITER's arguments. A contract with several outputs
   names none of them `out` (`outputs are 'out' alone, or names other than 'out'`); give each output
   its `error_bounds` and the contract its `determinism` (`bitwise` unless the op's kernels need
   atomics).
2. **`reference.py`**: one `<contract>_reference(point, row_major_args, to_float64)` per contract,
   in float64, returning the outputs in contract order. Any rounding every correct kernel must do
   belongs here.
3. **`api.py`**: one `@api(<CONTRACT>)` stub and at least one route. A route is a function decorated
   `@aklr_route("<aiter module>.<function>")` with AITER's signature that calls the api; list the AITER
   parameters the contract does not model in `must_default(...)`, so a call with any of them changed
   stays on AITER. Example: `aklr/ops/gemm/api.py`; a route may also name an SGLang entry point
   (`aklr/ops/topk/api.py`). `load_routes()` proxies the AITER function; AITER itself is not edited.
4. **`__init__.py`**: export the api functions.
5. **Tests**: add the contract's base Point, small, to `_SMALL_POINTS` in `tests/op_case.py` (contract
   order), so the contract and reference suites cover it, and regenerate the variant names with
   `AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py` (`tests/ops/names.txt`).
6. **Schemas**: `python -m aklr export` from a `AKLR_EXPORT` capture of the model, or narrow the
   contract to chosen Points with a script; say which in the PR. Then fill `kernel.sources`.

Such a PR changes code, so CI validates every committed schema, not only the new one
(`aklr-read-validation`, ci.md): its kernel waits for that run.
