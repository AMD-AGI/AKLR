# `aklr/ops/` — one directory per op family

The entry point for agents writing kernels. Each section summarises rules that are written out once,
in the file it names; when the two disagree, the named file wins.

## What an op is

An op is one computation an engine hands to AKLR, such as a 16-bit gemm (`gemm@1.0`). It has five
parts: four in Python under `aklr/ops/<op>/`, and its kernels as JSON files.

| Part | Where | What it is | Used by |
|---|---|---|---|
| contract | `contract.py`, an `OpContract` | the op's definition: its axes, attrs, inputs and outputs, their shapes, dtypes and layouts, each written tensor's error bound, and the op's reference and baseline | everything below |
| reference | `reference.py`, an `OpReference` | the right answer, computed in torch from the math; its `check` tells whether a kernel's outputs are within their bounds | validation: is a kernel correct |
| baseline | `baseline.py`, an `OpBaseline` | what the engine runs for the op today (aiter, sglang or vLLM), called as the engine calls it | validation: is a kernel faster |
| api | `api.py`, an `@api(<contract>)` function | the function a plugin route or aiter's provider calls | runtime: runs a kernel, or returns a `FallbackReason` |
| schema | `impls/<language>/<name>/<uuid>.json` | the contract narrowed to one scenario, one kernel's sources, and the workloads it serves with their measurements | validation checks and times it; dispatch serves it |

- A **Point** (`PointSpec`) is one coordinate of the contract: a value for every axis, a dtype for every
  tensor (`None`: absent), every attr, and the layout variant. Every call has one Point.
- A **scenario** is a Point without its var axes, the ones that change from call to call (`m`, the
  tokens); a schema serves one scenario. A **workload** is one value of the var axes (and the scalars)
  in a schema: a Point its kernel serves.

```
an engine's call ─> plugin route or aiter provider ─> the api (api.py)
                                                        dispatch reads the call's Point:
                                                        ├─ a schema's workload, accuracy passed, speedup ≥ AKLR_MIN_SPEEDUP
                                                        │    ─> that kernel's run(**call)
                                                        └─ otherwise ─> a FallbackReason: the engine runs its own code

python -m aklr validate <schema>
  audit: the kernel's sources keep to the rules (SA0–SA5)
  bench, per workload, on inputs drawn from the contract:
    the kernel's outputs  against  reference.reference(...), judged by reference.check  ─> accuracy
    the kernel's time     against  baseline.baseline(...),   both timed by baseline.profile ─> speedup
```

- The contract holds one reference and one baseline, as instances: `OpContract(...,
  reference=GemmReference(), baseline=GemmBaseline())`. Neither holds the contract: the contract calls
  them and passes in what they need.
- The kernel, the reference and the baseline each get their own copy of the same inputs; they never
  share a tensor.
- **Correct**: every mutable tensor (the outputs, and the inputs the kernel may write) within its error
  bound against the reference's, a tolerance or an SQNR ([Schema fields](#schema-fields)). A rule no bound
  states (ids that may come in any order) is the op's own `check`
  ([Reference and baseline](#reference-and-baseline)).
- **Fast**: a workload's `speedup` is a lower bound of the baseline's time over the kernel's, both in a
  CUDA graph (`harness/skills/read-validation/references/ruler.md`). A file needs one workload at the entry
  threshold (`DEFAULT_MIN_SPEEDUP`, 1.05 by default) or more; dispatch serves only the workloads at
  `AKLR_MIN_SPEEDUP` (1.05 by default) or more.

`gemm@1.0`, shortened (`aklr/ops/gemm/`):

```python
# contract.py
GEMM = OpContract(
    contract_id="gemm@1.0",
    description="16-bit gemm: out = a @ b.T + bias (b is (k, n) unless trans_b).",
    op_type="gemm",
    axes={"m": _M, "n": _N_A16W16, "k": _K},  # m is var; n and k are const in a scenario
    attrs={"trans_b": _TRANS_B, "has_bias": _HAS_BIAS},
    inputs={"a": TensorSpec(...), "b": ..., "bias": ...},
    outputs=_out(...),  # out: its shape, dtype and error bound (GEMM_BF16_SNR, ...)
    layout_variants=_UNSHUFFLED_VARIANTS,
    reference=GemmReference(),
    baseline=GemmBaseline(),
)

# reference.py
class GemmReference(OpReference):
    def reference(self, *, a, b, trans_b, out_dtype, bias=None, **_):
        """``a @ b.T + bias``."""
        return {"out": _out(a.float() @ _weight(b, trans_b).mT, bias).to(out_dtype)}

# baseline.py
class GemmBaseline(OpBaseline):
    def baseline(self, *, a, b, trans_b=True, bias=None, out_dtype=None, out=None):
        """aiter's tuned 16-bit gemm, b as (n, k) only."""
        from aiter.tuned_gemm import gemm_a16w16

        if not trans_b:
            raise NotImplementedError("aiter's gemm_a16w16 takes b as (n, k) only")
        return {"out": gemm_a16w16(a, b, bias=bias, otype=out_dtype)}

# api.py
@api(GEMM)
def gemm(a, b, trans_b=True, bias=None, out_dtype=None, out=None): ...
```

A kernel of a schema is `run(*, a, b, trans_b, bias, out_dtype, out)`, writing `out`
([The kernel section and ABI](#the-kernel-section-and-abi)).

## Layout

```
aklr/ops/
  __init__.py                  re-exports op apis
  README.md                    this file
  <op>/                        attention, gemm, fused_moe, mhc, mla, rms_norm, topk
    __init__.py                re-exports api.py's apis
    contract.py                OpContract constants, each with its reference and baseline; CONTRACTS
    reference.py               one OpReference class per contract
    api.py                     one @api function per contract
    baseline.py                one OpBaseline class per contract
    impls/<language>/<name>/<uuid>.json        one kernel and the workloads it serves
```

- A contract with `ranks` (the attr holding the group size, e.g. `ep_size`) is called by every rank of its group
  at once. The bench runs a process a rank, one device each, in one `torch.distributed` group; each rank draws its
  own inputs. Its reference is a `DistOpReference`, whose `reference(calls, rank)` takes every rank's call in rank
  order, each on its rank's device (`cuda:<rank>`), and returns rank `rank`'s outputs. Its baseline is a
  `DistOpBaseline`, whose `profile` starts every rank's call together; its `setup()` runs once each rank's process
  is in the group.

- `<language>` equals the schema's `kernel.language` (`flydsl`, `hip`, `hsaco` or `triton`, `Language` in
  `aklr/core/contract/spec/_kernel.py`). `impls/` is not a Python package: it holds language directories, they hold
  `<name>/` directories, and those hold `<uuid>.json` only.
- Several files may share a `<name>` and its Points; dispatch serves each Point with the best measured one
  (`AKLR_MIN_SPEEDUP`).
- `<uuid>` is 16 hex digits of `sha256(sorted_key_json([name, authored_parts]))` (`OpContract.authored_parts`): it
  changes with the kernel, its targets, requires or author, the scalars or the Points, never with `speedup`, `accuracy`, `microseconds` or workload order.
  `export check` names the expected path of a file whose uuid is stale.

## Workflow

1. Run the model with `AKLR_EXPORT=1` and `AKLR_EXPORT_DIR` set to an absolute path outside the checkout (the
   default `./aklr_export` inside it is untracked data). Every dispatch
   is counted in `<export dir>/calls.pid<p>.<run>.jsonl`, served or fallen back alike. A count is the number of
   Python-level dispatches: a CUDA graph replay is never counted, so decode shows its capture count. The set of
   Points is complete; never read the counts as throughput weights.
2. `python -m aklr export [--from DIR] [--to DIR] [--language L]` writes under `--to`, per scenario name, one new
   `<op>/impls/<language>/<name>/<uuid>.json` holding every called Point no committed file serves, with
   `"sources": []`. A Point a committed file already serves is only listed with that file and its call count (and a
   note when the calls came from a target the file does not list); a Point outside its contract's domain is only
   counted.
   Export never rewrites a committed file and never writes into `aklr/`; `--to` belongs outside the repository. A
   rerun removes only the earlier export's files whose `sources` is empty; filled files stay.
3. Fill `kernel.sources` with one `{"path": "<file>.py", "content": "<the file's text>"}` per file; export writes
   `"entry_point": "kernel.py::run"`, so one file is `kernel.py` defining `run` (or edit `entry_point`); a hip
   kernel's are `.hip` and `.h` files, `kernel.hip::run` its entry ([hip kernels](#hip-kernels)); an hsaco
   kernel stores its code objects in `.py` files ([hsaco kernels](#hsaco-kernels)). To give a
   scenario a second kernel, copy its committed file, delete `accuracy`, `speedup` and `microseconds` from every workload (only CI
   writes them: `PERF_EDITED`), and fill the copy with the second kernel.
4. Copy the file to the same relative path under `aklr/ops/` and rename it to the `<uuid>.json` that
   `python -m aklr export check` names (the uuid hashes the kernel). To change a committed kernel, edit its
   `content` or `requires`, delete `accuracy`, `speedup` and `microseconds` from every workload (only CI writes them:
   `PERF_EDITED`), and rename the file the same way. Audit checks every committed kernel's sources, changed or not.
5. Self-test: `python -m aklr export check`, then `python -m aklr validate --base origin/main`. The result does
   not count; CI decides.
6. Open a PR.

## Schema fields

A schema file is the registered `OpContract` narrowed to one scenario's values (`OpContract.narrow`), exactly the
text `schema_text` writes: keys in the order below, two-space indent, a list of only numbers on one line, non-ASCII
characters unescaped, a final newline. Anything else is rejected;
`python -m aklr export` writes that text. Model: `OpContract` in `aklr/core/contract/op.py`.

| Key | Written by | Content |
|---|---|---|
| `schema_version` | export | `5` (`CONTRACT_VERSION`) |
| `name` | export | the scenario name, equal to the parent directory (see Names) |
| `contract_id` | export | `<contract>@<major>.<minor>` |
| `description` | export | what the op computes |
| `op_type` | export | the op family, the `<op>` of `aklr/ops/<op>/`; the api function is `aklr.ops.<op>.api.<contract>`. A plugin op's is its package's last name (`aklr/plugins/README.md`) |
| `axes` | export | the contract's axes, every one but the var ones pinned to this scenario's value; a derived axis's `expr` becomes its int value |
| `attrs` | export | every attr, keeping this scenario's value alone |
| `inputs` | export | every input of the contract, narrowed to this scenario |
| `outputs` | export | every output, narrowed to this scenario; always row-major |
| `workloads` | export, author, CI | one item per Point served (see Workloads) |
| `layout_variants` | export | this scenario's layout variant alone |
| `compare` | export | `determinism` |
| `kernel` | export (`language`, `targets`, `entry_point`, `"sources": []`), author (`sources`, `requires`, `author`) | the kernel section; empty `sources` in the repository make a definition: loaded and audited, never served or benched |

Every key written by export alone is derived from the Python contract and checked against it (`export check`). Editing one directly only produces an error. A committed file and its
export differ only in `kernel.sources`, `kernel.requires` and `kernel.author`.

The fields below belong to the Python contract (`TensorSpec` and `ScalarSpec` in `aklr/core/contract/spec/_args.py`,
read through the op's `contract.py`); a schema file's `inputs` and `outputs` items carry only `shape`, `dtype`,
`distribution` (tensor inputs that have one) and `description`.

- A tensor input has `shape` (axis names or ints, logical elements), `dtype`, `layout` (`{"kind": "row_major"}`
  unless this variant stores it otherwise), `storage_shape`, `can_squeeze` (only when `true`: the shape without its
  trailing 1 is accepted too), `lifetime` and `description`. `lifetime` `weight` means address and value stay
  fixed across calls in production, `per_call` that they change on every call. `mutable` (only when `true`, on a
  required input such as a KV cache) lets the kernel write it in place; it is not required to, and the input shares
  no byte with another input. The outputs and the mutable inputs are the op's mutable tensors
  (`OpContract.mutable_tensors`): each is checked against the reference after a call, under its `error_bounds`;
  every other input must be left unchanged.
- `storage_shape` is the shape of the bytes the kernel receives: const axes are ints, var axes stay names or
  expressions such as `ceil_div(m, 256) * 256`. A 4-bit dtype packs two values into one stored element along the
  last dimension. For integers, call `storage_shape_of(tensor, point)` on the op's contract; from the json alone,
  call `storage_shape` on the dtype storage shape with the layout parsed by the `aklr.core.contract.semantics` class
  whose `kind` is the layout's `kind`, e.g. `E8M0Scales.model_validate(layout).storage_shape(shape)`.
- A scalar input has `"shape": null`, `dtype`, `values` (an enumeration) or `interval`, an optional `default`, and
  `description`. A kernel must be correct for every value in the declared range.
- Every mutable tensor has `error_bounds` (in `contract.py`, not in the schema file; narrowing to the scenario
  picks one), and is compared with the reference, both in its dtype, under it: `tolerance` (`atol` and `rtol` per
  element, as torch's `assert_close`; both 0 for an exact one, such as an integer id) or `sqnr` (whole-output SQNR);
  each is an `ErrorBound` kind in `semantics/_error_bounds.py`. The common bounds are there too (`tolerance(*dtypes)`,
  `GEMM_BF16_SNR`, `MOE_MXFP4_SNR`, ...): a contract uses them, and writes its own numbers only where its op needs
  other ones.
- The reference and the baseline: [Reference and baseline](#reference-and-baseline).

## Reference and baseline

The reference says what the op computes; the baseline is what the engine runs for it today. A kernel is checked
against the first and timed against the second.

The reference, a class in `reference.py` subclassing `OpReference` (`aklr/core/contract/reference.py`), whose
instance is the contract's `reference`:

- `reference(self, *, <names>, **_)` takes the kernel's call by keyword: the inputs row-major, the outputs'
  buffers, the scalars and the attrs the api takes; an absent optional input is `None`. It returns a dict of the
  mutable tensors by name, each mutable input as the call leaves it, and may add what its `check` reads.
- Computes in the op's dtypes where torch has them (a bf16 gemm as torch's bf16 matmul), else in float32; never in
  float64. For now every reference computes in float32: `.float()` for a dtype torch casts,
  `semantics.to_float32(tensor, dtype)` for a packed one (fp4, e8m0).
- Rounds only where every correct kernel rounds (a router's logits to bf16, an fp8 linear's input), and lists those
  roundings in its docstring. No sum is computed exactly: engines sum in float32, each in its own order, and the
  difference is left to `compare`.
- Returns each tensor in its declared dtype, rounded there by the reference itself; validation compares it as
  returned.
- Is written from the math, independent of the kernels and the baseline. It runs on the inputs' device (the GPU in
  validation), reads each size from a tensor's shape, and never writes a tensor of the call.
- `check(outputs, expected, schema)` compares the kernel's outputs with `expected` under the error bounds of
  `schema`, the contract at one scenario, and returns the outputs past them, each with why.
  An op whose correct kernels can differ in a way no bound states overrides it:
  `qwen2_moe_router`'s reference also returns its bf16 logits, and its `check` lets an id differ from the
  reference's where the two experts' logits are equal or one bf16 value apart, as two engines' sums can round a
  logit to either; `topk_transform_paged`'s and `dsa_topk_transform`'s let a row holding no -1 list its entries in
  any order; the DSA indexer's references also return the slots they write, and their `check` restates a
  key stored at twice or half the reference's fp8 scale, its values half or twice, at the reference's scale. The
  bounds themselves never change in a `check`.

The baseline, a class in `baseline.py` subclassing `OpBaseline` (`aklr/core/contract/baseline.py`), whose instance
is the contract's `baseline`; every contract has exactly one:

- `baseline(self, *, <names>)` is what the engine (sglang, vLLM, aiter) calls for the op at the Point: the same
  function, the same arguments, in the engine's precision. Nothing is reimplemented, and nothing is added that the
  engine does not do (a copy of the output, a re-layout of a weight).
- Takes the kernel's call by keyword, imports the engine inside the method, returns the outputs by name (a mutable
  input it writes is read from the call), and raises `NotImplementedError` for a call the engine does not serve
  (bench: `baseline_failed`).
- `setup()` sets up what the calls take for granted beyond their arguments, as the engine does before its first
  (mega_moe's mori heap), and `teardown()` undoes it; nothing by default.
- `profile(fn)` times one call, the kernel's and the baseline's alike (a CUDA graph, cache flushes taken out);
  validation computes the speedup.
- Is for timing only: kernels are checked against the reference.

An output's `error_bounds` allow the difference two correct implementations make (another float32 summation order,
a rounding at a tie), measured at the op's shapes.

## The kernel section and ABI

`kernel` (model `KernelSpec`): `language`, `targets` (a list of `{arch, hardware}`; a missing field leaves that
dimension unrestricted (`DeviceTarget.matches_target`); only a load gate), `requires` (required with sources, see below), `author`
(optional, see below), `entry_point` (`"<path>::<symbol>"`), and `sources`, a list of `{path, content}`, `content`
being the file's whole text (utf-8 without a byte order mark or a non-utf-8 encoding declaration, no control
character but `\n`, one final newline and no blank line before it); a path is one `<identifier>.py` segment other
than `__init__.py`, sorted and unique (a hip kernel's: [hip kernels](#hip-kernels)). The module docstring of the entry
file describes the kernel; its prose carries no history, commit or speedup, and trailing `key: value` metadata lines
(such as `legacy_solution:`, `keep_commit:`, `device_edits:`) are allowed; audit rejects an empty docstring (SA4).

`requires` maps a toolchain or library to a PEP 440 specifier of the versions the kernel needs, such as
`{"triton": ">=3.8"}`; a dev or rc build counts as its release. A kernel with sources names at least its
language's toolchain (`TOOLCHAINS`): `flydsl`, `triton`, or `hip` for hip and hsaco kernels, which is the HIP
version torch was built with (`torch.version.hip`), not the ROCm release (ROCm 10.0 ships HIP 7.15), and every
library it imports beyond torch, such as `{"mori": ">=1.2.4"}` (`Library` in `aklr/core/contract/spec/_kernel.py`);
`export check` rejects one that misses either (`KernelSpec.requires_error`). It is omitted only in a definition. Where the installed toolchain misses it, the runtime skips the schema, logging why at INFO, and validation
skips it too, listing it in the report. Give the version the kernel was developed on as the lower bound, and an upper bound
only once a release breaks it. A kernel that fails where `requires` holds raises: nothing falls back on an error.

`author` names who wrote the kernel: a person, a team or an agent, such as `"GEAK"`. It is one line without
surrounding spaces, omitted when empty, and only recorded: dispatch and the build ignore it, and validation
compares kernels without it (`KernelSpec.same_kernel`), so a committed kernel whose `author` alone changes keeps
its audit exemption and the retention threshold. The uuid hashes it, so that edit still renames the file and
deletes `speedup`, `accuracy` and `microseconds` (step 4); dispatch serves the file again once CI writes them back.

```python
def run(*, <every name of OpContract.kernel_call_keys>) -> None: ...
```

- The entry is the module-level def `entry_point` names (`run` by convention). It takes exactly the
  `kernel_call_keys` names as keyword-only arguments (SA4); aklr calls `run(**call)`. These are the
  `api_params` but an `optional` attr at its default (an attr added after schemas were committed, left out of
  their files), its parameter and the inputs whose `present_when` reads it:

  | Keys | From | Value |
  |---|---|---|
  | tensors and scalars | the contract inputs; an optional one absent here is still a key | stored tensor or scalar; `None` when absent |
  | attr arguments (`trans_b`, `activation`) | attrs with a parameter name | the value in `attrs` |
  | layout arguments (`operand_format`, `"<a>:<b>"`) | `detect_facts` with `location` `arg` | that fact's `expected_value` |
  | `out_dtype` | only when the one output is `out` | its dtype, or `None` when the caller passes none (`out` holds the dtype) |
  | outputs | every name in `outputs` | the preallocated output tensor |

- `run` writes each output into the output argument and returns nothing; it never allocates or keeps an output. A
  required output (one carrying an axis no required input carries, such as `topk`) comes from the caller and is
  never allocated by aklr.
- Inputs other than the mutable ones are read-only: write only the mutable tensors and the kernel's own scratch
  space. Writes of an input include
  `a += x`, `torch.relu_(a)`, `out=a` and `inplace=True`. A method ending in `_` or named like a container
  mutator (`add`, `sort`, `update`, ...) writes its receiver even where torch's version is out of place: write
  `torch.add(a, b)` and `torch.sort(a)`, not `a.add(b)` or `a.sort()` (SA2).
- Change no process-wide torch state, also not in a compile function: no `torch.set_*`, `torch.cuda.set_*`,
  `torch.manual_seed`, `torch.use_deterministic_algorithms` or `torch.cuda.stream` (SA3).
- Submit work only on the current torch stream: no synchronization, `.item()`, `.cpu()`, `.tolist()`, I/O,
  `print`, clock or environment reads. The names `input`, `open`, `print`, `breakpoint`, `exit` and `quit` are
  rejected wherever they are read or assigned, device functions included, so a parameter of that name is rejected
  at its first use (SA3): call a pointer `inp` or `x`, not `input`.
- Scratch space is allocated inside the call (`torch.empty`, `torch.zeros`, `torch.*_like`, `.new_*`, ...) on an
  argument's device, or once per shape in a compile function and rewritten whole by every call before it is read.
  `torch.rand*`, `torch.tensor` and `torch.as_tensor` are rejected in launch code (SA3).
- Import nothing of `aklr` but `aklr.core.jit`, its api for kernels: a kernel is checked and timed against aklr's
  reference and baseline, and is served without them; a call the kernel does not run fast is one its schema does
  not list (SA5).
- A device function, a def whose single decorator is `@flyc.kernel`, `@flyc.jit`, `@triton.jit` or `@gluon.jit`,
  is compiled: its body skips the launch checks of SA2 and SA3, and every whole-file check still applies to
  it. With a second decorator (`@triton.autotune` or `@triton.heuristics` above
  `@triton.jit`) the def is host code whose body must pass SA2 and SA3; choose the config in a compile
  function instead.
- Compilation happens only on a Point's first call, and that call compiles every variant a scalar value (`beta`)
  selects under the Point: later calls of the Point may happen inside a CUDA graph capture.
- Compiled code is cached by shape in a compile function: a module-level def whose single decorator is
  `@functools.cache` or `@functools.lru_cache(...)`. Its own body runs outside the launch checks of SA2 and SA3 and
  may compile and allocate; a def or lambda nested in it (such as the returned `launch`) is launch code under every
  rule on `run`, so build constants in the compile function's own body. Its cache holds compiled code, constants and
  per-shape scratch space only, nothing derived from input values. For example:

  ```python
  @functools.cache
  def _launch_for(shape, device):
      ...  # choose a config, compile
      return launch, workspace_bytes


  def run(*, a, b, trans_b, bias, out_dtype, out):
      launch, workspace_bytes = _launch_for(tuple(a.shape), a.device)
      ws = (
          torch.empty(workspace_bytes, dtype=torch.uint8, device=a.device)
          if workspace_bytes
          else None
      )
      launch(a=a, b=b, trans_b=trans_b, bias=bias, out_dtype=out_dtype, out=out, ws=ws)
  ```

- Sources are imported as submodules of a package `_aklr_<hash16>`: do not depend on the package name, and import
  sibling files only relatively. Imports and reflection are not audited.
- What `run` does depends only on this call: keep nothing derived from an argument across calls, such as a packed
  or reshuffled copy of a weight, and no reference to an argument after returning; repack on every call. A weight's
  identity or `_version` does not tell whether it changed (`param.data.copy_` keeps the version, and a CUDA graph
  replay skips the check). Audit rejects `global` and `nonlocal` (SA1); review catches the rest.
- A `@triton.jit` or `@gluon.jit` body reads no module-level value: Triton accepts only `x = tl.constexpr(42)`
  globals and fails to compile a plain `BLOCK = 64` or an annotated `BLOCK: tl.constexpr = 64` read inside the
  kernel. Pass such a value as a `tl.constexpr` parameter (`_k[grid](..., BLOCK=BLOCK)`) or write it inside the
  device function.
- The authority is `aklr/core/validation/stages/_audit/_rules.py`, whose rules SA0–SA5 every source must pass
  before it reaches a GPU; a source too deeply nested to audit (a few thousand chained operators) is rejected
  (SA0).

### hip kernels

A `hip` kernel is C++: `.hip` translation units and `.h` headers whose paths may nest (`common/math.h`), so a
source tree keeps its includes; the entry file is a `.hip` beginning with a comment that describes the kernel.
On first load aklr builds the sources with `torch.utils.cpp_extension` against the running torch, for the GPU's
arch, under `<AKLR_KERNEL_CACHE_DIR>/hip/aklr_hip_<hash16>/` (`aklr/core/jit/hip_extension.py`): seconds per kernel, once
per machine, torch and ROCm, shared by the machine's processes. `aklr_hip_<hash16>` is the build's
`TORCH_EXTENSION_NAME` and the op namespace, and the entry is the op the library registers there:

```cpp
#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor a, Tensor b, bool trans_b, Tensor? bias, ScalarType? out_dtype, Tensor(a!) out) -> ()", &run);
}
```

- The op's arguments are exactly the `kernel_call_keys` names, in any order, each output and mutable input a
  `Tensor(a!)`, and it returns `()`: aklr calls it as `run(**call)`, and the rules of `run` hold (it writes the
  preallocated outputs and the mutable inputs, reads its other inputs only and keeps nothing across calls).
- Launch on `at::hip::getCurrentHIPStream()`; allocate scratch with `at::empty` on an argument's device.
- Audit applies SH0–SH5 (`aklr/core/validation/stages/_audit/_hip.py`) instead of SA0–SA5, on the text with
  comments and literals blanked: no unterminated comment or literal (SH0); the one registration above, in the entry
  file, and no other library macro (SH1: a second library of one namespace aborts the process that loads it);
  includes are sources, `hip/`, `ATen/`, `c10/`, `torch/library.h` or a listed standard header (SH2); no
  synchronization, allocation outside torch, IPC, device or process change, environment, I/O or CUDA name (SH3);
  no `static` or `__device__` variable that is not const (SH4: a kernel keeps nothing between calls); the entry
  file's leading comment (SH5).

### hsaco kernels

An `hsaco` kernel launches precompiled code objects. Sources must be text, so the code objects are stored
base64-encoded in a `.py` file, and the entry point launches them through `aklr/core/jit/hsaco.py`:

```python
import struct

from aklr.core.jit import hsaco

from .codeobjects import GEMM  # bytes

_GEMM = hsaco.Kernel(GEMM, "gemm")


def run(*, a, b, trans_b, bias, out_dtype, out):
    kernarg = struct.pack("<QQQii", a.data_ptr(), b.data_ptr(), out.data_ptr(), b.shape[0], a.shape[1])
    _GEMM.launch((a.shape[0],), (b.shape[0],), kernarg, device=a.device)
```

- `Kernel(code_object, symbol)` names a function in a code object built for the target's arch. The code object is
  loaded into the HIP runtime that torch uses, once per device, on the first launch on that device.
- `launch(grid, block, kernarg)` launches on the device's current stream. `kernarg` holds the kernel's explicit
  arguments, laid out as the code object expects; HIP fills in the hidden arguments.
- The launcher only supports regular launches through `hipModuleLaunchKernel`. Cooperative launches, cluster
  launches, and memory shared between ranks are not supported; kernels that need them require changes to
  `aklr/core/jit/hsaco.py`, or for shared memory, to the op's `OpBaseline.setup`.
- `requires` pins the HIP that loads the code objects, such as `{"hip": ">=7.15"}`.
- Sources are audited like other Python kernels (SA0–SA5), then by SC1–SC2
  (`aklr/core/validation/stages/_audit/_hsaco.py`), so that kernel code reaches the GPU only through the launcher.
  SC1 limits imports to sibling sources, `aklr.core.jit.hsaco`, `torch`, and a few standard modules with no side
  effects. SC2 rejects `__import__`, `eval`, `exec`, `compile`, `globals`, `vars`, and the torch APIs that reach
  native code (`torch._C`, `torch.ops.load_library`, `torch.utils.cpp_extension`, `torch.cuda.cudart`). The audit
  cannot see what the code objects do; bench checks their results.

A host capturing CUDA graphs calls every Point at least once outside a capture first (sglang does). A Point first
called inside a capture returns `not_warmed`: its fallback is recorded into the graph, and replays no longer pass
through dispatch.

## Inputs and workloads

Each present tensor input carries its `distribution` (a `kind` and its parameters, an expression where it reads a
var axis): the values a kernel may assume, and how bench draws them. It is the contract's at the Point, written
once per file; changing it is rejected, not a way to change the inputs.

`workloads` is a list (model `WorkloadSpec`): `{"axes": {<var axis>: int}, "scalars": {...}, "accuracy": "passed" | "failed", "speedup": float, "microseconds": float}`,
axes in contract order, sorted by their values, each Point once. A var axis the contract declares `any_size`
(the rows of a KV cache, which the kernel does not depend on) is served at every value of its domain: its workload
values are only the sizes bench runs. `scalars` holds every present scalar input, a
captured or authored value inside its declared range; it is omitted when the schema has none. `speedup` (over the contract's baseline) and
`accuracy` are written by CI on every workload or none (a `failed` one may lack `speedup`; it is never dispatched),
with one complete `kernel.targets` item. `microseconds` records the time behind that `speedup`: the kernel's median
time per call in the same rounds against the baseline (graph timer, 3 significant digits), absent on a workload that
was not timed. CI rewrites a file only when a workload's accuracy changes, a speedup or time comes or goes, or a
speedup moves by more than 2%, and then writes all three from one run, so `microseconds` always matches the
`speedup` beside it. It is not meant for comparing times across files measured in different runs.

No workload carries a uuid. A workload is located by its file and its `axes` (`m=1` in reports). Every tensor is
seeded from `workload_hash(name, workload)` (its axes and scalars), 16 hex digits of a sha256: the same Point with
the same scalars gets the same values in every file and every language on the same kind of device. Bench draws
them on its GPU, so they differ from a CPU draw (the tests') and between GPU models.

Correctness is always checked on the default inputs (every tensor drawn from its `distribution`, every scalar at
its declaration's default). Captured scalars other than the defaults add one more input set, named `workload`.

## Names

`<contract>`, then the name part of each variant symbol whose value is not its first declared one, then
`<axis><value>` for each const int axis, joined by `_`; var, derived and absent axes are not part of it.
A name is unique per variant, across every contract (`tests/ops/test_contracts.py`, as flashinfer-bench requires).

| symbol: attrs (in declaration order, whatever their type), then dtypes, then the layout variant | first declared value (no part) | name part otherwise |
|---|---|---|
| bool attr | `values[0]` | `<label>` when true, `no_<label>` when false |
| int attr | `values[0]` | `<label><value>` |
| int tuple attr (an entry point's int-pair argument, such as a window) | `values[0]` | `<label><v0>_<v1>...` |
| str attr | `values[0]` | the value, lowercased |
| dtype of a tensor holding its own (not another tensor's via `same_as_tensor` alone) | `any_of[0]`, or the `same_as_tensor` dtype when both are set | `<tensor>_<dtype>`; none when absent |
| layout variant | the first (`row_major`) | its name |

`<label>` is the attr's `abbrev`, else its name. Example: `gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096`.
An attr's or a dtype's `default` is its first declared value. Implementation: `OpContract.name_at` in
`aklr/core/contract/op.py`.

A name follows the declarations: reordering `values`/`any_of`/layout variants, inserting before the last, or renaming
an attr, tensor or variant renames committed schemas (`parse_schema` reports the moved name). Append new values.
A file is `impls/<language>/<name>/<uuid>.json`; see the directory section for `<uuid>`.

## Semantics

A name in a schema file means what `aklr/core/contract/semantics/` of the reading aklr says; a schema names no
semantics file. Each json kind is one class in `aklr/core/contract/semantics/` holding its fields and behaviour: the
distribution kind `log_uniform` is `LogUniformDistribution` in `semantics/_distributions.py`. Dtypes, determinism levels
and tensor lifetimes are plain enums read by the json functions.

Any input set name other than `workload` gives a default input set; the same `workload_hash` and name give the
same values bit for bit.

The existing first-capture and post-timing checks overwrite reused input buffers, including weights, through
`.data.copy_()`: addresses and tensor versions stay unchanged. This exposes stale weight caches keyed by address
or version without adding check steps or timing rounds. A mismatch fails the existing capture or post-timing check.

A schema file carries no semantics stamp. After a change to the schema format, rewrite every committed file as
`OpContract.schema_text` writes it (`ContractRegistry.parse_schema` names the first differing line).

## Commands

| Command | Does | Exit |
|---|---|---|
| `python -m aklr export [--from DIR] [--to DIR] [--language L]` | export (`--from` and `--to` default to `AKLR_EXPORT_DIR`, else `./aklr_export`; `--language` required); fails with `no calls.*.jsonl in DIR` when there are no calls | 0 done, 1 error, 2 usage |
| `python -m aklr export check` | checks every committed schema and `impls/`; no GPU, no environment | 0 clean, 1 problems on stderr, 2 usage |
| `python -m aklr validate [PATH ...] [--base REV] [--stages audit,bench] [--under DIR] [--exclude DIR] [--no-timing] ...` | audit and bench stages (default: both); writes `aklr_validation/report.json` and `report.md`. `--under` keeps the schemas under DIR, `--exclude` drops them; `--no-timing` builds and checks the kernels only, on the workloads not recorded `failed` (dispatch never serves those), timing and writing back nothing; any finding fails it. A schema whose `kernel.requires` the environment misses is skipped, as dispatch skips it | 0 passed, 1 failed, 2 usage, 3 aborted |

Options and stages of `validate`: `python -m aklr validate --help`.

## Environment variables

Read only by `RuntimeEnv()` in `aklr/core/platform/env.py`.

| Variable | Values | Effect |
|---|---|---|
| `AKLR_DISABLE` | `1`, `0`, unset, `<contract_id>,...` | emergency stop: every contract, or the listed ones, falls back |
| `AKLR_MIN_SPEEDUP` | a finite number | the least measured `speedup` over the baseline that a workload needs to be served, default `1.05`; `0` or less also serves unmeasured workloads, never a `failed` one |
| `AKLR_KERNEL_CACHE_DIR` | a path | where kernels are materialized; default `${XDG_CACHE_HOME:-$HOME/.cache}/aklr/kernels` |
| `AKLR_EXPORT` | `1`, `0`, unset | record calls; any other value is an error |
| `AKLR_EXPORT_DIR` | a path | the export directory, default `./aklr_export`; setting it alone records nothing |
| `AKLR_MEASURED_DIR` | a path | a directory laid out like the checkout whose schemas replace the committed ones they name |
| `AKLR_SGLANG_PLUGIN` | `1`, `0`, unset | `1`: the sglang plugin hooks sglang's layers to its ops (`aklr/plugins/README.md`); any other value is an error |
| `AKLR_AITER_PLUGIN` | `1`, `0`, unset | `1`: the sglang and vLLM plugins enable AKLR as a provider of aiter's kernel plugin interface (`aklr/plugins/README.md`), whatever `AKLR_SGLANG_PLUGIN` and `AKLR_VLLM_PLUGIN` say; any other value is an error |
| `AKLR_TRACE` | `plugin`, `op`, `all`, `0`, unset | log, as warnings, each distinct hooked call once per process: `plugin` the vLLM and sglang routes (`[aklr plugin]` lines, logger `aklr.trace.plugin`), `op` every op api dispatch, including the ones aiter makes (`[aklr op]` lines, logger `aklr.trace.op`: the Point, and the kernel serving it or the fallback reason), `all` both; CUDA graph replays run no Python and log nothing; any other value is an error |
