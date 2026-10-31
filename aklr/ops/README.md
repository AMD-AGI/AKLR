# `aklr/ops/` — one directory per op family

The entry point for agents writing kernels. Each section summarises rules that are written out once,
in the file it names; when the two disagree, the named file wins.

```
aklr/ops/
  __init__.py                  imports every op's apis: registers their op schemas and routes
  README.md                    this file
  <op>/                        biased_grouped_topk, gemm, fused_moe, mhc, mla, topk
    __init__.py                re-exports api.py's apis
    schema.py                  OpSchema constants and any dispatch functions
    reference.py               one <op_schema>_reference per op schema
    api.py                     one @aklr_api function per op type, plus routes; a dispatch function may cover several op schemas
    impls/<language>/<name>/<uuid>.json        one kernel and the workloads it serves
```

- `<language>` equals the schema's `kernel.language` (`flydsl`, `hip` or `triton`, `Language` in
  `aklr/core/schema/comps/_kernel.py`). `impls/` is not a Python package: it holds language directories, they hold
  `<name>/` directories, and those hold `<uuid>.json` only.
- Several files may share a `<name>`; dtype, attrs and input presence distinguish their scenarios.
  Dispatch serves each workload's complete semantic key with the best measured implementation
  (`AKLR_MIN_SPEEDUP`).
- `<uuid>` is 16 hex digits of `sha256(sorted_key_json([name, authored_parts]))` (`OpSchema.authored_parts`): it
  changes with the kernel, its targets, requires or author, attrs, tensor dtypes, workload scalars or axes,
  never with `evaluations` or workload order.

`OpSchema.schema_data()` supplies the fields shared by schema serialization and UUID computation.
`OpSchema.relative_path()` computes the schema path and UUID without formatting and reparsing JSON.
`SchemaRegistry.schema_paths()` supplies
the shared file scan; its `SCHEMA_PATH_PATTERN` defines the path format used by validation and reports.

## Workflow

1. Run the model with `AKLR_ENABLE_TRACING=1`. At exit each traced call's shape is added as a workload to the
   kernel-less schema file of its scenario: under `AKLR_SCHEMA_PATH` (`schemas/<op_type>/<name>/<uuid>.json`; set it
   to an absolute path outside the checkout) or, unset, in `aklr/ops/<op>/impls/flydsl/<name>/` for a name `aklr/ops`
   already has; other names are not traced. A file that has a kernel is never changed.
2. Fill in its `kernel` section ([below](#the-kernel-section-and-abi)), `sources` holding one
   `{"path": "<file>.py", "content": "<the file's text>"}` per file; with `"entry_point": "kernel.py::run"`, one file
   is `kernel.py` defining `run`; a hip
   kernel's are `.hip` and `.h` files, `kernel.hip::run` its entry ([hip kernels](#hip-kernels)). To give a
   scenario a second kernel, copy its committed file, empty `evaluations` in every workload (only CI writes them:
   `PERF_EDITED`), and fill the copy with the second kernel.
3. Copy the file to the same relative path under `aklr/ops/` and rename it to its `<uuid>.json` (`Schema.uuid`: the
   schema without its workloads, so the kernel too). To change a committed kernel, edit its
   `content` or `requires`, empty `evaluations` in every workload (only CI writes them:
   `PERF_EDITED`), and rename the file the same way. Audit checks every committed kernel's sources, changed or not.
4. Self-test: `python -m aklr validate --base origin/main`. The result does
   not count; CI decides. A base schema of the same name that does not parse fails validation, one of another
   name is skipped.
5. Open a PR.

## Schema fields

A schema file is the registered `OpSchema` specialized to one scenario's values (`OpSchema.make_schema`), exactly the
text `schema_text` writes: keys in the order below, two-space indent, a list of only numbers on one line, non-ASCII
characters unescaped, a final newline. Anything else is rejected. Model: `OpSchema` in `aklr/core/schema/op.py`.

| Key | Written by | Content |
|---|---|---|
| `schema_version` | tracing | `1` (`SCHEMA_VERSION`) |
| `name` | tracing | a readable scenario label, equal to the parent directory, not its complete identity (see Names) |
| `description` | tracing | what the op computes |
| `op_type` | tracing | the op family: the name of its api function in `aklr/ops/<op>/api.py`, shared by the op schemas that function dispatches between (`gemm_fp8_nt_blockscaled` in `gemm/`) |
| `axes` | tracing | the op schema's axes, every one but the var ones pinned to this scenario's value; a derived axis's `expr` becomes its int value |
| `inputs` | tracing | every input of the op schema, narrowed to this scenario |
| `attrs` | tracing | one normalized value per `Scalar(is_attr=True)`, in input declaration order; an absent optional attr is `null` |
| `outputs` | tracing | every output, narrowed to this scenario; always row-major |
| `workloads` | tracing, author, CI | one item per Point served (see Workloads) |
| `compare` | tracing | the floating outputs' comparison, `{<method>: {<param>: <value>}}`; `{}` with no floating output |
| `kernel` | author | the kernel section; empty `sources` in the repository make a definition: loaded and audited, never served or benched |

Every key written by tracing alone is derived from the Python op schema; a committed file and the file tracing
writes differ only in `kernel`.

The fields below belong to the Python op schema (`Tensor` and `Scalar` in `aklr/core/schema/comps/_args.py`,
read through the op's `schema.py`); a schema file's `inputs` and `outputs` items carry `shape`, `strides` of a
padded input, `dtype`, `optional` when true, `distribution` for tensor inputs, and `description`.

- A tensor declaration has `dim_names`, optional fixed `dtype`, `optional` and `description`. Storage
  checks enforce contiguous tensors in their declared shape; an adapter brings a caller's tensors to it.
- The schema states the op's semantics only. A caller's convention (aiter's `is_shuffled`, a squeezed shape,
  a launch plan) is translated by its adapter; a limit of the baseline makes its baseline function return
  `None`; a limit of a kernel is a const axis of its file or handled in its `run`. A storage layout is an api
  parameter (`use_shuffled_weight`) its dispatch function picks the schema by; a value no schema describes
  makes the dispatch function return `None`, and the call falls back.
- An input a caller keeps padded declares its element `strides`, ints or expressions of const axes (sglang's
  DeepSeek-V4 kv pool: `ceil_div(page_size * 584, 576) * 576` per page); storage checks enforce those strides
  instead of contiguity, and validation allocates the input so, the padding zero.
- `dim_names` are the stored shape: an MN-major scale is declared `(scale_columns, m)`, and a route passing a
  transposed view writes `a_scale=x_scale.mT`.
- A 4-bit dtype packs two values into one stored element along the last dimension. Use
  `schema.storage_shape_of(tensor, workload)` for the actual stored shape.
- A scalar declaration has `dtype`, `optional`, `default`, `description` and `is_attr`. It remains an ordinary
  API/kernel argument whether fixed or dynamic. `is_attr=True` puts its normalized value in schema-level `attrs`
  and the dispatch key: transposition, quantization configuration, and MXFP4 GEMM's fixed `alpha=1.0`, `beta=0.0`
  use this path. Floats are supported, including float32 normalization and signed-zero canonicalization.
- Other scalar values remain in `workload.scalars` as test samples, not specialization keys. A kernel must support
  every value permitted by the op schema's `constraints`. Runtime dispatch validates scalar types, finite floats
  and every constraint; optional dynamic scalar presence is keyed separately. Schema attrs must never also appear
  in any workload's scalars.
- Distribution and `compare` branches must be decided by the fixed scenario, not dynamic scalar
  samples. Unresolved branches are rejected. Distribution parameters that remain expressions
  are evaluated per workload.
- Output dtype resolves from the call's `out_dtype` when given (`dtype_from_out_dtype=True`),
  otherwise `dtype_from` input, otherwise the declared `dtype` fallback. Constraints determine supported results. A supplied output buffer must match that
  resolved dtype; it does not override resolution. A fallback dtype does not restrict valid explicit overrides,
  while an input's fixed encoded dtype remains fixed. An output with `dtype_from` and no `dtype_from_out_dtype` has
  its input's dtype in every schema.
- The op schema's `compare` (narrowing to the scenario picks one branch) compares every floating output: elementwise
  `tolerance` (absolute and relative error), or whole-output `sqnr` or `cosine` against the
  reference rounded to the output dtype; each method is a `Comparison` class in `semantics/_compare.py`. An integer
  output must equal the reference exactly.
- An integer output with `any_order_in_full_rows` (in the op's `schema.py`, not in the schema file) may list the entries
  of a row in any order when that row of the reference holds no -1 padding; other rows compare element by element.
- The reference is `<op_schema>_reference(<ordinary tensor and scalar inputs>) -> Tensor` (a tuple in output order
  when there are several outputs). Validation supplies its reference inputs at the required precision; there is
  no Domain or Point argument. A rounding every correct
  kernel must perform bit for bit is part of the op and is written in the reference (each docstring lists its
  rounding points); everything else is error left to `compare`.

## The kernel section and ABI

`kernel` (model `Kernel`): `language`, `build.target_hardware` (a list of
`"<arch>"` or `"<arch>/<hardware>"`, such as `"gfx950"` for any gfx950 device or `"gfx950/MI355X"` for MI355X only
(`DeviceTarget.matches_target`); only a
load gate), `requires` (see below), `author` (see below), `entry_point` (`"<path>::<symbol>"`) and `sources`, a list of `{path, content}`, `content`
being the file's whole text (utf-8 without a byte order mark or a non-utf-8 encoding declaration, no control
character but `\n`, one final newline and no blank line before it); a path is one `<identifier>.py` segment other
than `__init__.py`, sorted and unique (a hip kernel's: [hip kernels](#hip-kernels)). The module docstring of the entry
file describes the kernel; its prose carries no history, commit or speedup, and trailing `key: value` metadata lines
(such as `legacy_solution:`, `keep_commit:`, `device_edits:`) are allowed; audit rejects an empty docstring (SA4).

`requires` maps a language to a PEP 440 specifier of the toolchain versions the kernel needs, such as
`{"triton": ">=3.8"}`; `{}` requires nothing, and a dev or rc build counts as its release. Where the installed
toolchain misses it, the runtime skips the schema, logging why at INFO, and validation reports
`no_matching_toolchain`. Give a lower bound when the kernel uses a newer toolchain's feature, an upper bound only once
a release breaks it. A kernel that fails where `requires` holds raises: nothing falls back on an error.

`author` names who wrote the kernel: a person, a team or an agent, such as `"GEAK"`. It is `""` when not
recorded, and only recorded: dispatch and the build ignore it, and validation
compares kernels without it (`Kernel.same_kernel`). The uuid hashes it, so that edit still renames the file and
empties `evaluations` (step 4); dispatch serves the file again once CI writes them back.

```python
def run(*, <every name of OpSchema.kernel_call_keys>) -> None: ...
```

- The entry is the module-level def `entry_point` names (`run` by convention). It takes exactly the
  `kernel_call_keys` as keyword-only arguments (SA4): the `api_params` names; aklr calls `run(**call)`:

  | Keys | From | Value |
  |---|---|---|
  | tensors and scalars | the op schema inputs; an optional one absent here is still a key | stored tensor or scalar; `None` when absent |
  | fixed scalar arguments (`trans_b`, `activation`, fixed `alpha`/`beta`) | scalar inputs marked `is_attr` | the normalized value in schema `attrs` |
  | `out_dtype` | only when the one output is `out` | its dtype |
  | outputs | every name in `outputs` | the preallocated output tensor |

- `run` writes each output into the output argument and returns nothing; it never allocates or keeps an output. A
  required output (one carrying an axis no required input carries, such as `topk`) comes from the caller and is
  never allocated by aklr.
- Inputs are read-only: write only the outputs and the kernel's own scratch space. Writes of an input include
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
arch, under `<AKLR_BUILD_CACHE_DIR>/hip/aklr_hip_<hash16>/` (`TorchBuilder` in `aklr/core/runtime/builder/builder.py`): seconds per kernel, once
per machine, torch and ROCm, shared by the machine's processes. `aklr_hip_<hash16>` is the build's
`TORCH_EXTENSION_NAME` and the op namespace, and the entry is the op the library registers there:

```cpp
#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor a, Tensor b, bool trans_b, Tensor? bias, ScalarType out_dtype, Tensor(a!) out) -> ()", &run);
}
```

- The op's arguments are exactly the `api_params` names, in any order, each output a `Tensor(a!)`, and it returns
  `()`: aklr calls it as `run(**call)`, and the rules of `run` hold (it writes the preallocated outputs, reads its
  inputs only and keeps nothing across calls).
- Launch on `at::hip::getCurrentHIPStream()`; allocate scratch with `at::empty` on an argument's device.
- Audit applies SH0–SH5 (`aklr/core/validation/stages/_audit/_hip.py`) instead of SA0–SA4, on the text with
  comments and literals blanked: no unterminated comment or literal (SH0); the one registration above, in the entry
  file, and no other library macro (SH1: a second library of one namespace aborts the process that loads it);
  includes are sources, `hip/`, `ATen/`, `c10/`, `torch/library.h` or a listed standard header (SH2); no
  synchronization, allocation outside torch, IPC, device or process change, environment, I/O or CUDA name (SH3);
  no `static` or `__device__` variable that is not const (SH4); the entry file's leading comment (SH5).

A host capturing CUDA graphs calls every Point at least once outside a capture first (sglang does). A Point first
called inside a capture returns `not_warmed`: its fallback is recorded into the graph, and replays no longer pass
through dispatch.

## Inputs and workloads

Each present tensor input carries its `distribution` (a `kind` and its parameters, an expression where it reads a
var axis): the values a kernel may assume, and how bench draws them. It is the op schema's at the Point, written
once per file; changing it is rejected, not a way to change the inputs.

`workloads` is a list (model `Workload`): `{"axes": {<var axis>: int}, "scalars": {...}, "evaluations": [...]}`,
axes in op schema order, sorted by their values, each axes combination once. `scalars` holds only present dynamic
scalar inputs, with captured or authored values satisfying the op schema; fixed scalar values live in schema-level
`attrs` and cannot differ between workloads. `scalars` is omitted when empty. `evaluations` holds one
`{"hardware": "<arch>/<hardware>", "accuracy": "passed" | "failed", "speedup": {<baseline>: {"commit", "value"}}}`
per device CI measured on, written on every workload or none; a run on another device keeps the other entries.
Dispatch reads only the current device's entry: a `failed` one is never served there, and a workload without one
is served there only at `AKLR_MIN_SPEEDUP` 0 or less.

No workload carries a uuid. A workload is located by its file and its `axes` (`m=1` in reports). Every tensor is
seeded from `workload_hash(name, workload)` (its axes and scalars), 16 hex digits of a sha256: the same Point with
the same scalars gets the same values in every file and every language.

Correctness is always checked on the default inputs (every tensor drawn from its `distribution`, schema attrs
unchanged, dynamic scalars at their declared defaults). If a dynamic scalar has no default, each workload uses
its own sample; schema construction never turns the first workload's sample into a default.
Captured dynamic scalars differing from declared defaults add one more input set, named `workload`.

## Names

`<name_prefix>`, then `<axis name><value>` for each present, non-derived const int axis, joined by `_`.
For example, `gemm_bf16_nt_n1024_k4096`. Variable and derived axes, dtypes, attrs and input presence do not appear in the name.
Storage variants that need separate op schemas have separate prefixes.

A name is a readable grouping label, not a unique definition. `OpSchema.scenario_key` identifies a scenario by
op schema prefix, fixed axes, every tensor dtype, typed scalar attrs, and present inputs. `workload_key` adds variable
axes. Existing coverage and benchmark incumbent selection use these keys, never name plus axes
alone. Dynamic scalar values are test samples within that identity.

Files with the same name but different semantics coexist at `impls/<language>/<name>/<uuid>.json`; the UUID
includes the authored attrs, tensor dtypes, kernel and workloads.

## Semantics

A name in a schema file means what `aklr/core/schema/semantics/` of the reading aklr says; a schema names no
semantics file. Each json kind is one class in `aklr/core/schema/semantics/` holding its fields and behaviour: the
distribution kind `log_uniform` is `LogUniformDistribution` in `semantics/_distributions.py`. Dtypes are a plain enum
read by the json functions.

Any input set name other than `workload` gives a default input set; the same `workload_hash` and name give the
same values bit for bit.

The existing first-capture and post-timing checks overwrite reused input buffers, including weights, through
`.data.copy_()`: addresses and tensor versions stay unchanged. This exposes stale weight caches keyed by address
or version without adding check steps or timing rounds. A mismatch fails the existing capture or post-timing check.

A schema file carries no semantics stamp. After a change to the schema format, rewrite every committed file as
`OpSchema.schema_text` writes it (`SchemaRegistry.parse_schema` names the first differing line).
The schema version is 1. Fixed scalar values live in top-level `attrs`, not workloads. Every read requires the
current canonical structure. Rewritten schemas must be measured again rather than carrying the previous UUID's
`evaluations`.

## Commands

| Command | Does | Exit |
|---|---|---|
| `python -m aklr validate [PATH ...] [--base REV] [--stages audit,bench,e2e] ...` | audit, bench and e2e stages (default: audit, bench); writes `aklr_validation/report.json` and `report.md` | 0 passed, 1 failed, 2 usage, 3 aborted |

Options and stages of `validate`: `python -m aklr validate --help`.

## Environment variables

Read only by `RuntimeEnv()` in `aklr/core/platform/env.py`.

| Variable | Values | Effect |
|---|---|---|
| `AKLR_DISABLE` | `1`, `0`, unset, `<op_type>,...` | emergency stop: every op, or the listed op types, falls back; an older AKLR's id `<op_type>@<version>` names its op type, an unknown op type fails the start |
| `AKLR_MIN_SPEEDUP` | a finite number | the least `speedup` over aiter, measured on the current device, that a workload needs to be served, default `1.03`; `0` or less also serves unmeasured workloads, never a `failed` one |
| `AKLR_BUILD_CACHE_DIR` | a path | where kernels are materialized; default `${XDG_CACHE_HOME:-$HOME/.cache}/aklr/kernels` |
| `AKLR_ENABLE_TRACING` | `1`, `0`, unset | record each call's shape as a workload of its scenario's kernel-less schema file, written at exit (see `AKLR_SCHEMA_PATH`); any other value is an error |
| `AKLR_SCHEMA_PATH` | a path | the dataset dispatch reads and tracing writes, laid out as `schemas/<op_type>/<name>/<uuid>.json` (uuid: the schema without its workloads); default `aklr/ops` in its own layout; the e2e stage points its child processes at a copy holding the bench's measurements |
