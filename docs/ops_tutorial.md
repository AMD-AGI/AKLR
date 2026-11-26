# Ops tutorial: adding a kernel to an AKLR op

This tutorial shows how to write a kernel for an op already in `aklr/plugins/aiter/`: generate its file, write the kernel, validate it and open a PR, up to inference using it. The example is the bf16 matrix multiply (`gemm@1.0`) at n=1024, k=4096. The last section covers creating an op when none fits.

The full rules are in [`aklr/plugins/aiter/README.md`](../aklr/plugins/aiter/README.md).

## 0. Concepts

An op has five parts:

| Part | Where | What it is |
|---|---|---|
| contract | `aklr/plugins/aiter/<op>/contract.py`, an `OpContract` | the op's definition: inputs, outputs, shapes, dtypes, how much error each output may have, and the reference and baseline below |
| reference | `reference.py`, an `OpReference` class | the right answer, computed in torch from the math |
| baseline | `baseline.py`, an `OpBaseline` class | the code the inference engine (aiter, sglang, vLLM) runs for the op today |
| api | `api.py`, an `@api` function | the entry the inference engine calls |
| schema | `impls/<language>/<scenario>/<uuid>.json` | one kernel: its source, and the shapes it serves |

A kernel has to pass two checks:

- **Correct**: its outputs differ from the reference's by no more than the contract allows (the error bound).
- **Fast**: it beats the baseline. Validation asks for a speedup of 1.05 by default; at inference the bar is the environment variable `AKLR_MIN_SPEEDUP`, also 1.05 by default.

Some words:

- **Axis**: an integer describing a shape, such as gemm's `m`, `n` and `k`. One that can change from call to call is a **var axis** (gemm's `m`, the tokens).
- **Scenario**: everything but the var axes fixed (n, k, the dtypes, whether there is a bias, ...). One schema file is one scenario.
- **Workload**: one value of the var axes, such as `m=256`. A schema lists every workload its kernel serves.

At inference the api takes a call and finds its scenario and workload: when a kernel there is correct and fast enough, it runs it; otherwise it returns a `FallbackReason` and the engine runs its own code.

## 1. Find the op and the scenario

Find the contract in `aklr/plugins/aiter/<op>/contract.py`. gemm's is in `aklr/plugins/aiter/gemm/contract.py`:

```python
GEMM = OpContract(
    contract_id="gemm@1.0",
    description="16-bit gemm: out = a @ b.T + bias (b is (k, n) unless trans_b).",
    axes={"m": ..., "n": ..., "k": ...},  # m is the var axis
    attrs={"trans_b": ..., "has_bias": ...},  # whether b is (n, k); whether there is a bias
    inputs={"a": ..., "b": ..., "bias": ...},
    outputs=...,  # out; a bf16 out's error bound: a signal-to-noise ratio (SQNR) of 45 dB or more
    ...
)
```

Then fix the scenario. This tutorial uses `n=1024` and `k=4096`; `a`, `b` and `out` in bf16; `b` of shape `(n, k)` (`trans_b=True`); no bias; and `m` at 1, 2, 4, ..., 4096.

In real work the scenario and the values of `m` come from a real model: run it once with `AKLR_EXPORT=1`, then `python -m aklr export --language <language>` writes every scenario the model called (the Workflow in `aklr/plugins/aiter/README.md`). This tutorial writes them out by hand.

When no op fits, see section 6 first.

## 2. Generate the schema file

A schema file is not written by hand: a script generates it from the contract. Save this script as `~/make_gemm_schema.py` (outside the repository):

```python
import sys
from pathlib import Path

from aklr.core.contract.locate import op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import PointSpec, SourceFile

contract = ContractRegistry.default().get("gemm@1.0")

# the scenario: everything but m fixed
point = PointSpec(
    contract_id="gemm@1.0",
    axes={"m": 1, "n": 1024, "k": 4096},  # m is the var axis; this 1 is a placeholder
    dtypes={"a": "bfloat16", "b": "bfloat16", "bias": None, "out": "bfloat16"},  # None: no such input
    attrs={"trans_b": True, "has_bias": False},
    layout_variant_name="row_major",  # the data layout: plain row-major
)

# the values of m to serve: one workload each
workloads = [
    contract.workload_for(point.model_copy(update={"axes": {**point.axes, "m": m}}), {})
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
]

# the kernel: given a directory, its .py files; without one, no sources
sources = []
if len(sys.argv) > 1:
    sources = [SourceFile(path=f.name, content=f.read_text(encoding="utf-8")) for f in sorted(Path(sys.argv[1]).glob("*.py"))]
kernel = {
    "language": "triton",
    "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
    "entry_point": "kernel.py::run",
    "requires": {"triton": ">=3.8"} if sources else {},  # none without a kernel
    "author": "<your name>",
    "sources": sources,
}
schema = contract.narrow(point, workloads=workloads, kernel=kernel)  # the contract's schema for this scenario
print("run's parameters:", ", ".join(schema.kernel_call_keys))
if not sources:
    sys.exit()  # no kernel yet: print the parameters only, write no file

path = op_dir(contract.op_package) / schema.relative_path()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(schema.schema_text(), encoding="utf-8")
print(path)
```

Run it from the repository root, **with `PYTHONPATH=.`**. Without it Python may use an aklr installed elsewhere, and write the file there.

```bash
PYTHONPATH=. python ~/make_gemm_schema.py
# run's parameters: a, b, trans_b, bias, out_dtype, out
```

The fields of `kernel`:

- `language`: `triton`, `flydsl`, `hip` or `hsaco`.
- `targets`: the GPUs the kernel runs on. No other GPU loads it.
- `entry_point`: the entry function, `<file>::<function>`, usually `kernel.py::run`.
- `requires`: required. The lowest version of the kernel's language (`triton`, `flydsl`, ...) it works with: the version you developed on. Add every library the kernel imports besides torch too. A machine whose versions fall short does not load the kernel.
- `author`: who wrote it; may be left out.

## 3. Write the kernel

Write `kernel.py` in `~/gemm_kernel/`. Here is a Triton kernel that works: it cuts `out` into 64×64 blocks, each computed by one parallel instance on the GPU, adding up 64 columns of `k` at a time.

```python
"""bf16 gemm out = a @ b.T, b as (n, k): one program per 64 x 64 tile of out, k in steps of 64."""

import triton
import triton.language as tl


@triton.jit
def _gemm(a, b, out, m, n, k, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, k, BLOCK_K):
        ks = start + tl.arange(0, BLOCK_K)
        x = tl.load(a + rows[:, None] * k + ks[None, :], mask=rows[:, None] < m, other=0.0)
        y = tl.load(b + cols[:, None] * k + ks[None, :], mask=cols[:, None] < n, other=0.0)
        acc += tl.dot(x, tl.trans(y))
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    tl.store(out + rows[:, None] * n + cols[None, :], acc.to(out.dtype.element_ty), mask=mask)


def run(*, a, b, trans_b, bias, out_dtype, out):
    m, k = a.shape
    n = b.shape[0]
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))
    _gemm[grid](a, b, out, m, n, k, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)
```

The rules `run` must keep (validation checks them; the full list is "The kernel section and ABI" in `aklr/plugins/aiter/README.md`):

1. **Parameters**: exactly the ones step 2 printed, all passed by name. An input the scenario does not have (`bias` here) is `None`.
2. **Results go into `out`**: `out` is already allocated; `run` writes into it and returns nothing.
3. **Write only outputs**: inputs are read only, except those the contract marks `mutable` (writable), such as a KV cache.
4. **Keep no state**: nothing computed from the arguments is kept for the next call, such as a repacked weight; no `global` or `nonlocal`. A compiled kernel may be cached per shape (in an `@functools.cache` function).
5. **Never wait for the GPU**: only submit work on the current torch stream, and call nothing that waits for the GPU to finish (`.item()`, `.cpu()`, `torch.cuda.synchronize()`, ...). The engine records calls into a CUDA graph and replays it, and such calls fail while it records.
6. **No side effects**: no `print`, no file reads or writes, no environment reads, no change to torch's global settings.
7. **Scratch memory** is allocated inside `run` (`torch.empty`, ...).
8. **Import nothing of aklr** but `aklr.core.jit`.
9. **The entry file opens with a docstring** saying how the kernel computes.

With the kernel written, run the script again, this time with the kernel's directory:

```bash
PYTHONPATH=. python ~/make_gemm_schema.py ~/gemm_kernel
# run's parameters: a, b, trans_b, bias, out_dtype, out
# aklr/plugins/aiter/gemm/impls/triton/gemm_n1024_k4096/<uuid>.json
```

The file name `<uuid>` comes from the file's content: after a change to the kernel, the workloads or the author, running the script again gives a new name, and the old file is yours to delete. The repository already has a FlyDSL kernel for this scenario (`aklr/plugins/aiter/gemm/impls/flydsl/gemm_n1024_k4096/`) to compare with: but for the `kernel` section and the measurements CI wrote into the committed file (`accuracy`, `speedup`, `microseconds`), the two files are the same.

## 4. Validate locally

Validation has two stages: audit checks the file's format, its agreement with the contract and the rules of step 3, without a GPU; bench checks on a GPU that the kernel is correct and fast. First the audit alone:

```bash
python -m aklr validate --stages audit aklr/plugins/aiter/gemm/impls/triton/gemm_n1024_k4096/<uuid>.json
```

Then audit and bench on an MI355X. Bench draws inputs for each workload, compares the kernel's outputs with the reference's, then runs the kernel and the baseline (aiter's `gemm_a16w16` here) in turns, timing both:

```bash
python -m aklr validate --base origin/main --stages audit,bench aklr/plugins/aiter/gemm/impls/triton/gemm_n1024_k4096/<uuid>.json
```

`--base origin/main` judges the file as a new kernel: it has to reach the bar over the baseline (1.05 by default), and not be slower than the kernel already in the same directory on main. Without it the bar is lower, and a file passing locally can still fail CI.

The results are in `aklr_validation/report.md`. For the kernel above, on an MI355X (excerpt):

```text
## AKLR validation: failed (bench)
| schema | axes | verdict | graph/baseline |
| triton/gemm_n1024_k4096/<uuid> | m=1    | not_faster (demoted) | 0.5000 |
| ...                              | ...    | ...                  | ...    |
| triton/gemm_n1024_k4096/<uuid> | m=4096 | not_faster (demoted) | 0.6375 |
- `no_qualifying_workload` triton/gemm_n1024_k4096/<uuid>: no workload both passes correctness and has a graph/baseline speedup >= 1.05
```

`graph/baseline` is the speedup over aiter, a cautious estimate from several rounds of timing; each round's ratio is held between 0.5 and 2, so 0.5 means half as fast as aiter, or slower. All 13 workloads of this kernel are correct, but none is fast enough (1.05 by default), so none can be used and the file fails.

There are three outcomes:

| Outcome | Meaning | What to do |
|---|---|---|
| passed | every workload is correct and fast enough | open a PR |
| some workloads marked `(demoted)` | those workloads are wrong (such as `numeric_mismatch`) or too slow (`not_faster`). Inference does not use the kernel for them; validation does not fail on them | improve the kernel, or remove those values of `m` from the script and generate the file again |
| `no_qualifying_workload` | no workload is both correct and fast; validation fails | improve the kernel |

How to read an accuracy failure, and whether a slow result is the machine's noise: `harness/skills/read-validation/SKILL.md`. How to make a kernel fast: `harness/skills/kernel-optimization-book/SKILL.md`.

## 5. Open a PR, and when it takes effect

- Write the body following `.github/pull_request_template.md`, with the results of step 4 and where the scenario and the values of `m` come from.
- Add the `ci:gpu` label: without it CI does not validate on a GPU.
- Do not write `accuracy`, `speedup` or `microseconds` into the file yourself: only CI writes them, and validation reports a hand-written one as `perf_edited`.

Once the PR merges, CI measures again on main, and a bot opens (or updates) a PR on the `aklr-perf/main` branch that writes the measurements into the file. **Once that PR merges**, inference uses your kernel.

At inference a call runs your kernel when all of these hold: the engine has the AKLR plugin on, so the call reaches the api (`aklr/plugins/README.md`); the machine matches `targets` and `requires`; the call's shape is one of the file's workloads; that workload's `accuracy` is not `failed`; and its `speedup` is at least `AKLR_MIN_SPEEDUP` (1.05 by default). Otherwise the api returns a `FallbackReason` and the engine runs its own code:

| `FallbackReason` | Why |
|---|---|
| `no_schema` | no schema is loaded |
| `disabled` | turned off by `AKLR_DISABLE` |
| `out_of_contract` | the call is outside what the contract defines |
| `no_workload` | the shape is not a workload of any schema |
| `not_measured` | no measurement yet |
| `inaccurate` | the measurement is `failed` |
| `not_qualified` | measured and correct, but its `speedup` falls short of `AKLR_MIN_SPEEDUP` |
| `not_warmed` | a CUDA graph is being recorded, and this shape never ran outside a recording before |

## 6. Create an op

When no op fits, create one first. An op is five files under `aklr/plugins/aiter/<op>/`. gemm is the example below; its full code is in `aklr/plugins/aiter/gemm/`.

1. **`contract.py`**: defines the `OpContract`, and lists it in the file's `CONTRACTS` (AKLR finds it there; there is nothing else to register).
   - `op_type` is the directory's name, `<op>`.
   - Name the axes, inputs and outputs after the math, not after the baseline's parameters. An op usually has one output, `out`; with several, none of them is called `out`.
   - Every output and every writable input has an error bound, `error_bounds`. Use the common ones in `aklr/core/contract/semantics/_error_bounds.py` where they fit (`tolerance(...)`, `GEMM_BF16_SNR`, ...).
   - `compare=CompareSpec(determinism=...)`: whether two runs on the same inputs must give the same bytes. `bitwise` (the same) is the default; a kernel using atomic adds, whose result changes with the order of execution, needs `bounded` (each run within the error bound).
   - Last, `reference=` and `baseline=`: instances of the two classes below.

2. **`reference.py`**: one `OpReference` class per contract.

   ```python
   class GemmReference(OpReference):
       def reference(self, *, a, b, trans_b, out_dtype, bias=None, **_):
           """``a @ b.T + bias``."""
           return {"out": _out(a.float() @ _weight(b, trans_b).mT, bias).to(out_dtype)}
   ```

   - Takes the kernel's parameters by name; `**_` takes the ones it does not use.
   - Returns a dict: one item per output (and per writable input it changes), **each in the dtype the contract declares**.
   - Written from the math, not copied from a kernel or the baseline. For now it computes in float32 throughout, with no rounding beyond what is needed.
   - A rule no error bound can state (such as top-k results in any order) goes in an override of `check`.

3. **`baseline.py`**: one `OpBaseline` class per contract: the engine's own call for the op.

   ```python
   class GemmBaseline(OpBaseline):
       def baseline(self, *, a, b, trans_b=True, bias=None, out_dtype=None, out=None):
           """aiter's tuned 16-bit gemm, b as (n, k) only."""
           from aiter.tuned_gemm import gemm_a16w16

           if not trans_b:
               raise NotImplementedError("aiter's gemm_a16w16 takes b as (n, k) only")
           return {"out": gemm_a16w16(a, b, bias=bias, otype=out_dtype)}
   ```

   - Calls the function the engine calls, with the arguments it passes: nothing reimplemented, nothing added.
   - Imports the engine inside the method, so a machine without the engine can still import the file.
   - Raises `NotImplementedError` for a call the engine does not serve.

4. **`api.py`**: one `@api` function per contract, whose body is only `...`:

   ```python
   @api(GEMM)
   def gemm(a, b, trans_b=True, bias=None, out_dtype=None, out=None): ...
   ```

5. **`__init__.py`**: exports the api functions.

6. **Tests**: add a small-shaped `PointSpec` (written like `point` in step 2's script) to `SMALL_POINTS` in `tests/plugins/aiter/cases.py`, in contract order, then run `AKLR_UPDATE_NAMES=1 pytest tests/plugins/aiter/test_contracts.py` to regenerate `tests/plugins/aiter/names.txt`, and commit it too.

An op several GPUs call together (expert parallelism, for example) sets the contract's `ranks`, and its reference and baseline are a `DistOpReference` and a `DistOpBaseline` (`aklr/plugins/aiter/README.md`).

Open the new op's code in a PR of its own, merged before the kernel's.

## Next

A kernel replaces one op. To fuse several steps of the engine into one computation (a whole MoE layer, for example), see the [SGLang plugin tutorial](sglang_plugin_tutorial.md) and the [vLLM plugin tutorial](vllm_plugin_tutorial.md).
