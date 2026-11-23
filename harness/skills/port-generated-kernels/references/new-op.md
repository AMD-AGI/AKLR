# Adding an Op for a Ported Kernel

When no contract covers the kernel, add the op before its schema. The layout and the meaning of
each file are in `aklr/plugins/aiter/README.md`; this is the order and the traps.

1. In **`aklr/plugins/aiter/<op>/plugin.py`**, create `family = Plugin("aiter")`, register each
   `OpContract` with `family.define(contract)` and call `family.build()`. Name the axes,
   inputs and outputs after the math, not after the baseline's arguments. A contract with several outputs
   names none of them `out` (`outputs are 'out' alone, or names other than 'out'`); give each output
   its `error_bounds` and the contract its `determinism` (`bitwise` unless the op's kernels need
   atomics).
2. **`reference.py`**: one `<contract>_reference(point, row_major_args)` per contract, in the op's
   dtypes or else float32 (for now float32, each input read through `semantics.to_float32`),
   returning the outputs in contract order. Only the roundings every correct kernel makes belong
   here ("Reference and baseline" in `aklr/plugins/aiter/README.md`).
3. In the same **`plugin.py`**, bind each named kernel signature with `@family.op`. Where AITER routes an equivalent public call, add its versioned `provide` declaration in `declare(host)`. See `aklr/plugins/aiter/rms_norm/plugin.py`.
4. **`baseline.py`**: what kernels are timed against, the engine's own call for the op, one
   `<contract>_baseline` per contract that has one. It takes the api's params (called with the
   call's keyword arguments), imports AITER or SGLang inside the function and returns the outputs
   as the api does; a call it does not serve raises `NotImplementedError` (bench:
   `baseline_failed`). Example: `aklr/plugins/aiter/gemm/baseline.py`. Without one the op is its own
   baseline: a kernel has nothing to beat.
   An op every rank of a group calls together (expert parallelism, collectives) sets the contract's
   `ranks` to the attr holding the group size; its reference takes `(point, group_args, rank)`,
   and `group.py`'s `<contract>_group_init()` sets up what an engine would before the first call.
   See `aklr/plugins/aiter/README.md`.
5. Include **`__init__.py`** for the family package.
6. **Tests**: add the contract's base Point, small, to `_SMALL_POINTS` in `tests/op_case.py` (contract
   order), so the contract and reference suites cover it, and regenerate the variant names with
   `AKLR_UPDATE_NAMES=1 pytest tests/plugins/aiter/test_contracts.py` (`tests/plugins/aiter/names.txt`).
7. **Schemas**: `python -m aklr export` from a `AKLR_EXPORT` capture of the model, or narrow the
   contract to chosen Points with a script; say which in the PR. Then fill `kernel.sources`.

CI chooses schemas from the change set. Non-documentation changes under `aklr/core/` select all committed schemas. See `aklr/core/validation/stages/validation_scope.py`.
