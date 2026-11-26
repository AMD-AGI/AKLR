# Validating, reproducing and rolling back a kernel

`python -m aklr validate` checks a definition file in two stages: audit (the file and its sources against the
rules, no GPU) and bench (each Point correct and timed, on a GPU). The commands and their options are in
`aklr/plugins/aiter/README.md` § Commands and `python -m aklr validate --help`. How to read the report is the
`read-validation` skill (`harness/skills/read-validation/SKILL.md`). This page is the procedures around it.

## Before you run

- **A git checkout.** Validate asks git for the repository root and takes only files inside it.
- **The file at its path, in the exact text.** After any edit to the kernel, `requires`, `author` or the Points,
  delete `accuracy`, `speedup` and `microseconds` from every workload and rename the file to the path
  `python -m aklr export check` names (README § Workflow, step 4; writing the file: `docs/ops_tutorial.md` § 2).
  Under `--base`, a measured field you kept or wrote is `perf_edited`.
- **The environment CI uses.** For `aklr/plugins/aiter`, the baseline aiter at the commit `aiter` in `stack.json`,
  installed as `CONTRIBUTING.md` § The AITER baseline says. For an op of the sglang or vLLM plugin, the engine image
  in `stack.json` as it ships.
  A toolchain outside the file's `kernel.requires` skips the file, as dispatch does; the report lists it.
- **One idle GPU, one validate.** `--devices 0` picks the GPU. Two validates on one GPU give wrong ratios
  (`../kernel-optimization-book/triton/tuning.md` §2). Give each run its own `--out`: the default
  `aklr_validation/` is overwritten.
- **A checkout of your own.** Do not validate in a clone that a forge_task run is using: its driver writes each
  candidate into `aklr/plugins/aiter/`.
- Run the Python snippets below from the repository root with `PYTHONPATH=.`, so that they import this checkout.

## Which command

`validate` below is `python -m aklr validate`. Always name the file: with no path and no base it takes every
committed file, which is hours.

| Call | Benched | Judged |
|---|---|---|
| `validate --stages audit <file>` | nothing; no GPU needed | the file's text, path and sources |
| `validate --base origin/main --devices 0 --out <dir> <file>`, file not on the base | the file | as a new kernel, as CI does: 1.05 against the baseline, and 0.99 against the file the base serves at each Point (the "incumbent"), timed in the same rounds |
| the same, file unchanged from the base | nothing ("no kernel to measure"), unless the branch changes `aklr/core/` code or `stack.json` | audit only |
| `validate --devices 0 --out <dir> <file>` (no base) | the file | as a kept kernel: 0.99 against the baseline at each Point, no incumbent; the file still needs one Point at 1.05 |
| `validate --base origin/main` (no path) | every definition file the branch adds, untracked ones included; every committed file if the branch changes `aklr/core/` code or `stack.json` | as CI does |
| add `--no-timing` | builds and checks, times nothing, writes nothing back | every finding fails the run |

The run writes `<dir>/report.md`, `<dir>/report.json` and `<dir>/schemas/` (each file whose measurements this run
changed, with them). It writes nothing into the checkout. Exit 0 passed, 1 failed, 2 usage, 3 the platform failed.

## What bench does at each Point

Source: `aklr/core/validation/stages/_bench/_evaluation/` (`_request.py` lists the steps).

| Step | What runs | It catches |
|---|---|---|
| baseline | The baseline once. Its output is compared with nothing | `baseline_failed`: the engine does not serve this workload |
| first call | The kernel once, outside any capture. It may compile here | `build_failed`: an import or compile error, or more than 16 GiB allocated |
| `first_capture/0`–`/2` | One capture, replayed twice more with new values copied into the same buffers, address and `_version` unchanged | `capture_unsafe`: a compile or a synchronization in the capture, or a replay on the new values that is wrong |
| `numeric/0`–`/2`, `numeric/3` | A fresh capture per input set. `/3` exists when the workload's scalars differ from the defaults, and is the only step that uses them | `numeric_mismatch` |
| `determinism/0` | `numeric/0` again, bytes compared; `bitwise` contracts only | `nondeterministic` |
| timing | The rounds of [timing.md](timing.md), on one input set at the scalars' defaults | `not_faster`, `regressed` |
| `post_timing/0`–`/3` | New values copied into the timed buffers | `post_timing_mismatch`: state kept across calls |

Before every captured call the outputs are filled with NaN, or `0xFF` bytes for integer and sub-byte dtypes. After
it, each read-only input is compared byte for byte (`input_mutation`) and each output's shape and dtype are checked.
The reference then runs on the same inputs, and its `check` judges the outputs. A batch gets 900 s to build plus
300 s per Point before `candidate_timeout`.

## What a finding usually means

Which findings fail the run and which only demote a Point: `read-validation` skill, step 2.

| Finding | Usual cause | Read |
|---|---|---|
| `schema_invalid` | not at its uuid path, not the exact text, measured on some workloads only | README § Schema fields |
| `source_rejected` | an audit rule; the detail names rule, file and line | README § The kernel section and ABI |
| `build_failed` | an import error, a compile error at the first call, a first call that fell back, memory over the limit | [languages.md](languages.md) |
| `capture_unsafe` | detail `compiled in the capture`: a compile key that is not a function of the Point. At `/1` or `/2`: device state written once (counters, sentinels, padding) | [languages.md](languages.md); `../kernel-optimization-book/triton/authoring.md` §8 |
| `input_mutation` | a read-only input written, a debug store included | — |
| `numeric_mismatch` | the math. Non-finite mismatches in a float output are often elements never written. An arrival-order sum past its bound also lands here | `../kernel-optimization-book/gemm/correctness.md`; `../kernel-optimization-book/triton/numerics.md` §10 |
| `nondeterministic` | two runs within the bounds but with different bytes: atomics, arrival order, a race | `../kernel-optimization-book/gemm/correctness.md` §3, §8 |
| `post_timing_mismatch` | state kept across calls | `../kernel-optimization-book/pitfalls/state_kept_across_calls.md` |
| `candidate_crash` | the child process died (`exited with code -11`: a segfault, which the compiler can raise too; `-6`: an abort), or an output has the wrong shape or dtype | `../kernel-optimization-book/triton/debugging.md` §6 |
| `candidate_timeout` | a hang: a workgroup waiting for another, a device loop on a value never written | § Stress below |
| `no_qualifying_workload` | no Point is both correct and at 1.05: the demoted list gives each Point's cause | `read-validation` |
| `platform_failure` | the tracer or the machine, not the kernel | [timing.md](timing.md) § Tracer faults |

A finding's detail carries the last line of the child's error. The child's other output is not kept: to read a
compiler diagnostic or a device print, run the launch outside validate (§ Reproduce one step outside validate).

## A/B one change

1. **Write B as a copy of A at its own uuid**, in the same directory, with the one change:

   ```python
   import json
   from pathlib import Path
   from aklr.core.contract.expr import json_file_text
   from aklr.core.contract.locate import schema_uuid
   from aklr.core.contract.op import OpContract

   a = Path("<A>.json")
   schema = json.loads(a.read_text())
   for w in schema["workloads"]:                       # only CI writes these
       for field in ("accuracy", "speedup", "microseconds"):
           w.pop(field, None)
   source = next(s for s in schema["kernel"]["sources"] if s["path"] == "kernel.py")
   assert source["content"].count(OLD) == 1
   source["content"] = source["content"].replace(OLD, NEW)
   uuid = schema_uuid(schema["name"], OpContract.authored_parts(schema))
   (a.parent / f"{uuid}.json").write_text(json_file_text(schema))
   ```

2. **If A is the committed file that serves the Points**, run `validate --base origin/main --devices 0 --out <dir>
   <B>.json`. Each of B's `workload_records` in `report.json` then has a second `timings` entry with
   `versus: "incumbent"`: A and B timed in the same rounds, in random order. `exp(log_ratio_mean)` of that entry is
   A's time over B's. The incumbent is the file with the best stored speedup at that Point on the base. A Point
   that no file on the base serves has no such entry.
3. **Otherwise alternate** A, B, A, B on one idle GPU, four runs each without a base, and compare the medians of
   `candidate_milliseconds`.
4. **The exit code is not the A/B.** A B slower than A is a `regressed` demotion, and the run still exits 0 while
   one Point beats the baseline. Read the timings, and read B's accuracy.
5. **Read the result against the noise**: `../kernel-optimization-book/triton/tuning.md` §1–§2.
6. **Clean up.** Delete the copies you do not ship: `validate --base` with no path benches every new file. Ship B
   in place of A, or as a second file that lists only the Points where it wins: dispatch serves each Point with
   the best measured file.

## Reproduce a stored number

1. Check out the commit whose stored values you want, install the baseline of its `stack.json`, and take one idle
   MI355X.
2. `python -m aklr validate --devices 0 --out <dir> <file>`, with no base: a base that has the file skips bench.
3. Compare per Point. If the run passed and `<dir>/schemas/` does not hold the file, every Point reproduced: the
   same accuracy, and a speedup within 2% of the stored one. Otherwise the copy there holds this run's values.
4. Compare `microseconds` as well as `speedup`: a ratio also moves with the baseline's build.
5. Say what you ran on. The report's first lines name the commit, GPU, ROCm, torch and FlyDSL; `report.json` adds
   Triton under `platform` and the ruler under `measurement_profile`. aiter is not recorded: print
   `importlib.metadata.version("amd-aiter")`.

Two CI runs of the same sources differ by a few percent ([timing.md](timing.md) § How far a number moves by itself).

## Reproduce one step outside validate

Build the step's call with AKLR's own trial code, then call the kernel and the reference yourself:

```python
from pathlib import Path
import torch
from aklr.core.contract import semantics
from aklr.core.contract.registry import ContractRegistry
from aklr.core.validation.stages._bench._evaluation._request import TrialRequest

path = "aklr/plugins/aiter/gemm/impls/flydsl/gemm_n6144_k4096/<uuid>.json"
registry = ContractRegistry.default()
schema = registry.parse_schema(Path(path).read_bytes(), path)
contract = registry.get(schema.contract_id)
workload = next(w for w in schema.workloads if w.axes == {"m": 16})
request = TrialRequest.of_workload(path, schema, contract, workload, thresholds={}, incumbent=None)
step = next(s for s in request.steps if s.name == "numeric/1")        # None in its place: the timed inputs
args = request.compose_inputs(step, "cuda")
call = {k: v for k, v in request.assemble_call(args, "cuda").items() if k in schema.kernel_call_keys}
for name in contract.allocated_outputs:
    call[name] = torch.empty(contract.storage_shape_of(name, request.point),
                             dtype=semantics.torch_dtypes(request.point.dtypes[name])[0], device="cuda")
expected = contract.reference.reference(**request.reference_call(args, "cuda"))
```

- Import the kernel as a package: write its sources into a directory (SKILL.md § Where the kernels are), put the
  directory's parent on `sys.path`, and `from <directory>.kernel import run`.
- `run(**call)` returns nothing. Take `outputs = {n: call[n] for n in contract.mutable_tensors}`;
  `contract.reference.check(outputs, expected, request.schema)` returns each failed tensor with the finding's detail.
- To time it with AKLR's timer: `bind_clock(contract.baseline).graph_reading(lambda: run(**call))`
  (`aklr/core/validation/stages/_measure.py`) returns milliseconds per call. `import aklr.core.utils.testing`
  before the process first touches the GPU, as bench's own child process does: the tracer registers at that
  import. Call `contract.baseline.setup()` before the first reading, and drop the first reading.
- A contract with `ranks` needs one process per rank. This recipe does not cover it.
- `harness/scripts/forge_task/assets/aklr_forge_driver.py` does all of this for a forge_task workspace (`_calls`,
  `_load_candidate`, `_measure_candidate`).

## Stress

Validate gives a kernel the GPU alone, so it passes a kernel whose workgroups wait for each other. The stress check
runs every listed Point on streams limited to 16 and to 64 CUs, and beside a second call of its own on another
stream. Each output must equal the output of the call alone: the same bytes under `bitwise`, within a few ulps
under `bounded`. A call not finished after 60 s counts as hung.

- In a forge_task workspace: `HIP_VISIBLE_DEVICES=0 python $RUN/workspace/driver.py --stress` must end with
  `stress: PASS` (`harness/scripts/forge_task/README.md`, step 5 and § When something goes wrong).
- For a kernel written elsewhere, run the same conditions: `_stress_conditions` and `_run_stress` in
  `harness/scripts/forge_task/assets/aklr_forge_driver.py`.
- A kernel that fails is not delivered. Make its fold wait-free or bound its wait
  (`../kernel-optimization-book/gemm/wait_free_fold.md`, `../kernel-optimization-book/attention/split_kv.md` §2).

## Roll back

A Point that CI measures again below `AKLR_MIN_SPEEDUP`, or as `accuracy: failed`, stops being served by itself.
Roll back for what CI's inputs do not reach: a wrong result on inputs outside the contract's distributions, a hang
or NaN under contention, a scalar value first seen inside a capture, a toolchain upgrade that changes the kernel.

| Scope | Do | Effect |
|---|---|---|
| One deployment, at once | `AKLR_DISABLE=<contract_id>[,...]`, or `AKLR_DISABLE=1` | Every file of those contracts falls back (`disabled`), not one file: `gemm@1.0` turns off every 16-bit GEMM kernel |
| One file | Delete the definition file, in a PR | Each of its Points goes to the next best file of the same `<name>` that reaches the threshold, in any language, else to the caller's own code |
| Some Points of a file | Remove those workloads, delete the measured fields, rename the file (`export check`) | The file is new: its other Points are not served until CI has measured it |
| A fix | Edit the source; then as the row above | The same |

Afterwards run the model with `AKLR_TRACE=op` and read which kernel, or which fallback reason, each Point gets. A
withdrawn kernel's PR says why: `harness/skills/open-pr/SKILL.md`.
