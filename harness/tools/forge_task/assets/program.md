# Program: implement @NAME@ in @LANGUAGE@

## Objective

Make `run` in `kernel.py` faster than AKLR's baseline on as many of the @N_WORKLOADS@ workloads below as
possible, keeping every one of them correct: a candidate that wins on time and loses a workload is rejected.

## Contract

- `@OP_TYPE@`: @DESCRIPTION@
- Attributes: `@ATTRS@`. Inputs: @INPUTS@. Outputs: @OUTPUTS@.
- Fixed axes: `@CONSTANT_AXES@`. Varying axes: `@VARIABLE_AXES@`.
- `run` takes AKLR's call keys as keyword arguments and writes each output into its preallocated argument;
  it returns nothing and allocates no output.
- `kernel.py` may import modules beside it relatively (`from .helpers import f`): those modules are part of the
  kernel AKLR publishes; a file `kernel.py` does not reach, such as a probe, is not.
- AKLR compares the outputs with `@COMPARE@`.
- Every shape, dtype, layout and error bound is in the schema: `@SCHEMA_JSON@`.
- What a kernel may and may not do (the audit, the ABI, determinism) is AKLR's rule:
  `@AKLR_ROOT@/aklr/ops/README.md`.

## Workloads

@WORKLOAD_TABLE@

## Where the numbers come from

`driver.py` judges correctness with `python -m aklr validate` and times the candidate with AKLR's graph
timer on AKLR's timing inputs. It is the measurement boundary and is not editable.

@ANCHOR@
