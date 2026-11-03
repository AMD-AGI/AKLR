# `aklr/ops/` — one directory per op family

The entry point for agents writing kernels. Each section summarises rules that are written out once,
in the file it names; when the two disagree, the named file wins.

```
aklr/ops/
  __init__.py                  re-exports op apis
  README.md                    this file
  <op>/                        biased_grouped_topk, gemm, fused_moe, mhc, mla, rms_norm, topk
    __init__.py                re-exports api.py's apis
    contract.py                OpContract constants, CONTRACTS
    reference.py               one <contract>_reference per contract
    api.py                     one @api function per contract
    baseline.py                one <contract>_baseline per contract that has a baseline
    impls/<language>/<name>/<uuid>.json        one kernel and the workloads it serves
```

- `<language>` equals the schema's `kernel.language` (`flydsl`, `hip` or `triton`, `Language` in
  `aklr/core/contract/spec/_kernel.py`). `impls/` is not a Python package: it holds language directories, they hold
  `<name>/` directories, and those hold `<uuid>.json` only.
- Several files may share a `<name>` and its Points; dispatch serves each Point with the best measured one
  (`AKLR_MIN_SPEEDUP`).
- `<uuid>` is 16 hex digits of `sha256(sorted_key_json([name, authored_parts]))` (`OpContract.authored_parts`): it
  changes with the kernel, its targets, requires or author, the scalars or the Points, never with `speedup`, `accuracy` or workload order.
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
   kernel's are `.hip` and `.h` files, `kernel.hip::run` its entry ([hip kernels](#hip-kernels)). To give a
   scenario a second kernel, copy its committed file, delete `accuracy` and `speedup` from every workload (only CI
   writes them: `PERF_EDITED`), and fill the copy with the second kernel.
4. Copy the file to the same relative path under `aklr/ops/` and rename it to the `<uuid>.json` that
   `python -m aklr export check` names (the uuid hashes the kernel). To change a committed kernel, edit its
   `content` or `requires`, delete `accuracy` and `speedup` from every workload (only CI writes them:
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
| `compare` | export | `max_violation_fraction`, `determinism`, `element_order` |
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
- A mutable tensor's `error_bounds` (in `contract.py`, not in the schema file; narrowing to the scenario
  picks one) is one of `dot_product`, `output_rounding`, `baseline_relative`, `tolerance`, `sqnr` (whole-output
  SQNR against the reference rounded to the output dtype); each formula is an `ErrorBound` kind in
  `semantics/_compare.py`. An integer one has none and must equal the reference exactly. `baseline_relative` needs
  the baseline's output: outside the repository, call the contract's `<contract>_baseline` yourself.
- `compare.element_order` is `fixed` (element by element) or `any_in_full_rows` (integer outputs only: a row
  holding no padding may list its entries in any order); each is an `ElementOrder` member.
- The reference is `<contract>_reference(point, row_major_args, decode) -> Tensor` (a tuple when there are
  several: the mutable tensors in order, each mutable input as the call leaves it), never rounded to the output
  dtype. It computes in the float dtype `decode(tensor, dtype)` returns the inputs in: float32 in validation
  (`semantics.to_float32`, on the GPU), float64 when a test passes `semantics.to_float64`; it upcasts to float64
  itself only where the op defines an exact value (an exact sum rounded once, an index past 2**24). It reads only
  `point.axes`, `point.attrs` and `point.dtypes`. A rounding every correct
  kernel must perform bit for bit is part of the op and is written in the reference (each docstring lists its
  rounding points); everything else is error left to `compare`.

## The kernel section and ABI

`kernel` (model `KernelSpec`): `language`, `targets` (a list of `{arch, hardware}`; a missing field leaves that
dimension unrestricted (`DeviceTarget.matches_target`); only a load gate), `requires` (optional, see below), `author`
(optional, see below), `entry_point` (`"<path>::<symbol>"`), and `sources`, a list of `{path, content}`, `content`
being the file's whole text (utf-8 without a byte order mark or a non-utf-8 encoding declaration, no control
character but `\n`, one final newline and no blank line before it); a path is one `<identifier>.py` segment other
than `__init__.py`, sorted and unique (a hip kernel's: [hip kernels](#hip-kernels)). The module docstring of the entry
file describes the kernel; its prose carries no history, commit or speedup, and trailing `key: value` metadata lines
(such as `legacy_solution:`, `keep_commit:`, `device_edits:`) are allowed; audit rejects an empty docstring (SA4).

`requires` maps a language to a PEP 440 specifier of the toolchain versions the kernel needs, such as
`{"triton": ">=3.8"}`; it is omitted when empty, and a dev or rc build counts as its release. Where the installed
toolchain misses it, the runtime skips the schema, logging why at INFO, and validation reports
`no_matching_toolchain`. Give a lower bound when the kernel uses a newer toolchain's feature, an upper bound only once
a release breaks it. A kernel that fails where `requires` holds raises: nothing falls back on an error.

`author` names who wrote the kernel: a person, a team or an agent, such as `"GEAK"`. It is one line without
surrounding spaces, omitted when empty, and only recorded: dispatch and the build ignore it, and validation
compares kernels without it (`KernelSpec.same_kernel`), so a committed kernel whose `author` alone changes keeps
its audit exemption and the retention threshold. The uuid hashes it, so that edit still renames the file and
deletes `speedup` and `accuracy` (step 4); dispatch serves the file again once CI writes them back.

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
  `torch.rand*`, `torch.tensor` and `torch.as_tensor` are rejected in launch code (SA3).- A device function, a def whose single decorator is `@flyc.kernel`, `@flyc.jit`, `@triton.jit` or `@gluon.jit`,
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
- The authority is `aklr/core/validation/stages/_audit/_rules.py`, whose rules SA0–SA4 every source must pass
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
- Audit applies SH0–SH5 (`aklr/core/validation/stages/_audit/_hip.py`) instead of SA0–SA4, on the text with
  comments and literals blanked: no unterminated comment or literal (SH0); the one registration above, in the entry
  file, and no other library macro (SH1: a second library of one namespace aborts the process that loads it);
  includes are sources, `hip/`, `ATen/`, `c10/`, `torch/library.h` or a listed standard header (SH2); no
  synchronization, allocation outside torch, IPC, device or process change, environment, I/O or CUDA name (SH3);
  no `static` or `__device__` variable that is not const (SH4: a kernel keeps nothing between calls); the entry
  file's leading comment (SH5).

A host capturing CUDA graphs calls every Point at least once outside a capture first (sglang does). A Point first
called inside a capture returns `not_warmed`: its fallback is recorded into the graph, and replays no longer pass
through dispatch.

## Inputs and workloads

Each present tensor input carries its `distribution` (a `kind` and its parameters, an expression where it reads a
var axis): the values a kernel may assume, and how bench draws them. It is the contract's at the Point, written
once per file; changing it is rejected, not a way to change the inputs.

`workloads` is a list (model `WorkloadSpec`): `{"axes": {<var axis>: int}, "scalars": {...}, "accuracy": "passed" | "failed", "speedup": float}`,
axes in contract order, sorted by their values, each Point once. A var axis the contract declares `any_size`
(the rows of a KV cache, which the kernel does not depend on) is served at every value of its domain: its workload
values are only the sizes bench runs. `scalars` holds every present scalar input, a
captured or authored value inside its declared range; it is omitted when the schema has none. `speedup` (over `<contract>_baseline`) and
`accuracy` are written by CI on every workload or none (a `failed` one may lack `speedup`; it is never dispatched),
with one complete `kernel.targets` item.

No workload carries a uuid. A workload is located by its file and its `axes` (`m=1` in reports). Every tensor is
seeded from `workload_hash(name, workload)` (its axes and scalars), 16 hex digits of a sha256: the same Point with
the same scalars gets the same values in every file and every language.

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
Only when an output has a `dot_product` bound, bench's `large_magnitude/0` inputs go through `magnified_inputs`, then
`stored_inputs`.

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
| `python -m aklr validate [PATH ...] [--base REV] [--stages audit,bench] ...` | audit and bench stages (default: both); writes `aklr_validation/report.json` and `report.md` | 0 passed, 1 failed, 2 usage, 3 aborted |

Options and stages of `validate`: `python -m aklr validate --help`.

## Environment variables

Read only by `RuntimeEnv()` in `aklr/core/platform/env.py`.

| Variable | Values | Effect |
|---|---|---|
| `AKLR_DISABLE` | `1`, `0`, unset, `<contract_id>,...` | emergency stop: every contract, or the listed ones, falls back |
| `AKLR_MIN_SPEEDUP` | a finite number | the least measured `speedup` over the baseline that a workload needs to be served, default `1.03`; `0` or less also serves unmeasured workloads, never a `failed` one |
| `AKLR_KERNEL_CACHE_DIR` | a path | where kernels are materialized; default `${XDG_CACHE_HOME:-$HOME/.cache}/aklr/kernels` |
| `AKLR_EXPORT` | `1`, `0`, unset | record calls; any other value is an error |
| `AKLR_EXPORT_DIR` | a path | the export directory, default `./aklr_export`; setting it alone records nothing |
| `AKLR_MEASURED_DIR` | a path | a directory laid out like the checkout whose schemas replace the committed ones they name |
| `AKLR_SGLANG_PLUGIN` | `1`, `0`, unset | `1`: the sglang plugin hooks sglang's layers to its ops (`aklr/plugins/README.md`); any other value is an error |
