# `aklr/plugins/aiter/` — one directory per op family

The entry point for agents writing kernels. Each section summarises rules that are written out once,
in the file it names; when the two disagree, the named file wins.

```
aklr/plugins/aiter/
  __init__.py                  re-exports the op apis
  README.md                    this file
  <op>/                        attention, fused_moe, gemm, kda_conv, mega_moe, mhc, mla, mla_prefill, quant, rms_norm, topk
    __init__.py                re-exports api.py's apis
    schema.py                  the Schema templates, each with its Standard
    reference.py               the references the Standards check against
    baseline.py                the backend calls the Standards time against
    api.py                     one @aklr_api function per op type
    impls/<language>/<name>/<16 hex>.json      one schema file: at most one kernel and its workloads
```

- A **template** is a `Schema` constant in `schema.py` (model: `aklr/core/schema/_schema.py`): `op_type`, `axes`
  (`Var` or `Const`), `variant`, `inputs`, `outputs`, `constraints`, and a `Standard`. A **schema file** is a
  template at one set of `Const` axis values, as json. The json holds no `Standard`: `OpVariants.load` reads a
  file with its template's, any other read leaves it `None`.
- A `Standard` (`aklr/core/schema/_standard.py`) is what a kernel is held to: `init` draws the inputs, `reference`
  and `check` judge the outputs (default `default_check`, `_check.py`), `baseline` returns the backend's call ready
  to time, or `None` where the backend declines the inputs. Each is a plain `def` without free variables that uses
  only `torch`, `math` and functions of its own package, so `Standard.save` can write it as a standalone module.
- `@aklr_api(SCHEMA)` or `@aklr_api((VARIANT, ...))` (`aklr/core/runtime/dispatch/_ops.py`) declares the
  templates and makes the function the op's api. The function's body never runs: it only gives the signature.
- `<language>` is `kernel.build.language`: `flydsl`, `hip`, `hsaco` or `triton` (`Language` in
  `aklr/core/schema/_kernel.py`). `impls/` is not a Python package.
- Files of one `<name>` are identical except for `kernel` and `workloads` (`SchemaCollection.from_layout`). A file
  without a `kernel` is a definition: it lists workloads no kernel serves yet and is never served or benched.
- A group op (mega_moe) is called by every rank of its group at once. How many ranks, whether in fresh processes,
  and each rank's set-up are its `OpBench`, a subclass in
  `aklr/core/validation/stages/bench/_ops/<op_type>.py`, not part of the op package.

## Workflow

1. Run the model with `AKLR_ENABLE_TRACING=1` and `AKLR_SCHEMA_PATH` set to a directory outside the checkout.
   Every call that falls back to the backend is recorded once per Point, at exit, as a workload of the kernel-less
   file `<dir>/schemas/<op_type>/<name>/<uuid>.json`; its Standard is saved beside it as
   `<dir>/standards/<name>/{init,reference,check,baseline}.py`. With `AKLR_SCHEMA_PATH` set, dispatch reads its
   kernels from that directory too, so the committed kernels serve nothing and every call is recorded. Without
   it, workloads are added to the name's kernel-less file under `aklr/plugins/aiter/`, and a name that has no file there is
   not recorded.
2. Copy the file to `aklr/plugins/aiter/<op>/impls/<language>/<name>/` and add the `kernel` section (below). Leave every
   workload's `evaluations` empty: only CI writes them (`PERF_EDITED`).
3. Name the file `<Schema.uuid>.json`: 16 hex digits of the sha256 of the json without `workloads`, so it changes
   with the kernel and never with a measurement. Audit only requires 16 hex digits under the right `<language>`
   and `<name>`.
4. To change a committed kernel, write the changed schema as a new file with empty `evaluations` and delete the
   old one. A schema-only change benches only paths the base does not have, so a kernel edited in place is not
   measured again, and its stored `evaluations` may not be touched.
5. Self-test: `python -m aklr validate --base origin/main`. The result does not count; CI decides.
6. Open a PR.

## Schema fields

Keys in the order `save_json_file` writes them. Model: `Schema` in `aklr/core/schema/_schema.py`.

| Key | Content |
|---|---|
| `name` | the schema's name, equal to its parent directory (see [Names](#names)) |
| `op_type` | the op, the name of its api function |
| `axes` | `{"type": "var"}`, a size that differs per call, or `{"type": "const", "value": int}`, a size the kernel is specialized to |
| `variant` | `{}` or one `{<variant name>: <condition>}`: the op's calls that are this variant's, a Python expression over the api's arguments and `torch` |
| `workloads` | one `{"axes": {<var axis>: int}, "evaluations": [...]}` per Point served, each Point once |
| `inputs` | by name, in the order the kernel takes them: a tensor's `shape` (axis names), `dtype`, `description`; a scalar's `"shape": null`, `dtype`, `description` |
| `outputs` | as `inputs`; a schema with a kernel has tensor outputs only |
| `tags` | one `<model>/tp<n>[+dp<n>][+dcp<n>][+ep<n>]` per deployment that calls it, such as `glm-5.2/tp8+dp8` |
| `description` | what the op computes |
| `constraints` | Python expressions the inputs satisfy |
| `kernel` | `null`, or the kernel section |

Audit (`SCHEMA_INVALID`, `aklr/core/validation/stages/audit/_stage.py`) checks each file against its template: a
template of its `name` exists, `inputs` and `outputs` are the template's names in the template's order, its `axes`
are among the template's, and each workload's axes are `var` axes. A file with a kernel also needs at least one
workload and a template whose Standard has an `init`.

A template's fields the json leaves out (`_args.py`): a tensor's `param` and `tuple_idx` (the api argument it is,
or an item of), `optional`, an output's `dtype_from` and `dtype_from_scalar`; a `Const` axis's `abbrev` and
`extract`.

## Op variants

One api function can stand for several templates, its variants: the templates share the `op_type`
and differ in `variant` (and usually in `inputs`, reference or `init`).

```python
# schema.py
GEMM_BF16_NT = Schema(op_type="gemm_bf16_nt", variant={"default": "not use_shuffled_weight"}, ...)
GEMM_BF16_NT_SHUFFLED = replace(GEMM_BF16_NT, variant={"shuffled": "use_shuffled_weight"}, standard=...)
# in resolution order: the api takes the first whose condition matches the call
GEMM_BF16_NT_VARIANTS = (GEMM_BF16_NT, GEMM_BF16_NT_SHUFFLED)

# api.py
@aklr_api(GEMM_BF16_NT_VARIANTS)
def gemm_bf16_nt(a, b, bias=None, out=None, out_dtype=torch.bfloat16, use_shuffled_weight=False): ...
```

- `variant` is `{<name>: <condition>}` with one item. The name is lowercase words joined by `_`
  (`^[a-z0-9]+(_[a-z0-9]+)*$`). The condition is a Python expression evaluated on the call's bound arguments,
  defaults applied; it may read only the api's parameters, `torch` and builtins (checked when the api is
  declared).
- An empty `variant` field is the variant named `default`, of every call. A condition `"True"` gives a single variant
  a name of its own (`mega_moe.gate_up_interleaved`).
- A variant is named `<op_type>.<variant name>` (`Schema.op_name`): `gemm_bf16_nt.default`, `gemm_bf16_nt.shuffled`.
  The variant name is part of the schema name unless it is `default` (see [Names](#names)). Two variants of one
  api may not share a name, and two apis may not share an `op_type`.
- Resolution (`OpVariants.resolve`) takes the first variant, in the tuple's order, whose condition holds.
  The conditions are not checked for overlap: put the narrower one first. A call no variant takes goes to the
  caller's fallback, or raises when the api was called directly.
- Each variant is dispatched on its own: its kernels, workloads and `DispatchConfig` are kept per `op_name`, and
  a kernel of one variant never serves a call of another.
- Every api parameter is visible to the conditions, but a kernel receives only its variant's schema inputs:
  a flag such as `use_shuffled_weight` selects the variant and is not passed on.
- A schema file carries its template's `variant` item unchanged; audit finds its template by `op_name` and name
  (`OpVariants.variant_of`).

## The kernel section and ABI

`kernel` (model `Kernel` in `aklr/core/schema/_kernel.py`):

| Key | Content |
|---|---|
| `name` | unique among all schema files; by convention `<schema name>_<language>_<16 hex>` |
| `author` | who wrote it: a person, a team or an agent, such as `"GEAK"` |
| `build.language` | `flydsl`, `hip`, `hsaco` or `triton`, equal to the `<language>` directory |
| `build.target_hardware` | a list of `<arch>` or `<arch>/<hardware>`, such as `gfx950/MI355X`; the kernel is built and served only on a GPU that matches one |
| `build.entry_point` | `<path>::<symbol>`, the path one of `sources`; by convention `kernel.py::run`, `kernel.hip::run` |
| `build.requires` | `{<toolchain or library>: <PEP 440 specifier>}`, such as `{"triton": ">=3.8"}`; see below |
| `build.dependencies` | a list, usually empty |
| `build.destination_passing_style` | `true` (default): the outputs are the last arguments; `false`: the kernel returns them |
| `build.binding` | `null` |
| `sources` | a list of `{path, content}`, `content` the file's whole text; paths are relative, unique and have no `..` |
| `description` | optional |

`build.requires` names the versions the kernel needs; a dev or rc build counts as its release. A kernel names its
language's toolchain (`TOOLCHAINS` in `aklr/core/schema/_kernel.py`): `flydsl`, `triton`, or `hip` for hip and
hsaco kernels, which is the HIP torch was built with (`torch.version.hip`), not the ROCm release (ROCm 10.0 ships
HIP 7.15). It also names every library its sources import beyond torch, such as `{"mori": ">=1.2.4"}` (`LIBRARIES`
in `aklr/core/platform/toolchain.py`). Audit rejects a kernel that misses either (`SCHEMA_INVALID`). Where the
environment misses a requirement, the kernel does not run (`Build.runs_on`): dispatch does not serve it, and
bench skips its schema and lists it in the report. Give the version the kernel was developed on as the lower
bound, and an upper bound only once a release breaks it.

```python
def run(<every input name, in schema order>, <every output name, in schema order>): ...
```

- The entry is the module-level def `entry_point` names. Its parameters are exactly the schema's input names, then
  (destination-passing) its output names, in schema order, all plain positional parameters: no `*`, `/`, `*args` or
  `**kwargs` (SA4). aklr calls it positionally (`KernelFunction.__call__`).
- An input the call does not pass is `None`. An api argument that is not a schema input (`use_shuffled_weight`,
  `out_dtype`) is not passed: it selects the variant or the output dtype.
- Destination-passing: `run` writes each output into its argument and returns nothing. An output buffer the
  caller did not pass is allocated by aklr with the schema's shape and dtype, unless the output is `optional`.
  Value-returning: `run` returns the output, or a tuple of them in schema order.
- The entry file's module docstring describes the kernel; audit rejects a missing one (SA4).
- Inputs are read-only: write only the outputs and the kernel's own scratch space. Writes of an input include
  `a += x`, `torch.relu_(a)`, `out=a`, `inplace=True`, a subscript or attribute assignment, and any method ending
  in `_` or named like a container mutator (`add`, `sort`, `update`, ...) even where torch's version is out of
  place: write `torch.add(a, b)` and `torch.sort(a)`, not `a.add(b)` or `a.sort()` (SA2).
- Change no process-wide torch state, anywhere in the sources: no `torch.set_*`, `torch.cuda.set_*`,
  `torch.manual_seed`, `torch.seed`, `torch.use_deterministic_algorithms` or `torch.cuda.stream` (SA3).
- No I/O anywhere in the sources: `torch.save`, `torch.load`, and the names `open`, `print`, `input`,
  `breakpoint`, `exit` and `quit` wherever they are read or assigned, device functions included (SA3): call a
  pointer `inp` or `x`, not `input`.
- Launch code neither synchronizes nor times: no `.synchronize`, `.item`, `.cpu`, `.tolist`, `.numpy`, `Event`,
  `elapsed_time` or `perf_counter`. It allocates scratch space with `torch.empty`, `torch.zeros`, `.new_*` and the
  like; `torch.rand`, `torch.randn`, `torch.randint`, `torch.tensor` and `torch.as_tensor` are rejected (SA3).
- No `global` or `nonlocal` (SA1): keep nothing derived from an argument across calls, such as a packed copy of a
  weight. A weight's identity or `_version` does not tell whether it changed. Review catches what audit cannot.
- A device function, a def whose single decorator is `@flyc.kernel`, `@flyc.jit`, `@triton.jit` or `@gluon.jit`,
  is compiled: its body skips the launch checks of SA2 and SA3. With a second decorator (`@triton.autotune` above
  `@triton.jit`) the def is host code that must pass them; choose the config in a compile function instead.
- A compile function, a module-level def whose single decorator is `@functools.cache` or
  `@functools.lru_cache(...)`, caches compiled code by shape. Its own body skips the launch checks and may compile
  and allocate; a def or lambda nested in it (such as the returned `launch`) is launch code. Its cache holds
  compiled code, constants and per-shape scratch space only, nothing derived from input values:

  ```python
  @functools.cache
  def _launch_for(shape, device):
      ...  # choose a config, compile
      return launch


  def run(a, b, bias, out):
      _launch_for(tuple(a.shape), a.device)(a, b, bias, out)
  ```

- Sources are imported as a package named after the kernel under `<AKLR_BUILD_CACHE_DIR>/<language>/`
  (`PythonBuilder`): do not depend on the package name, and import sibling files only relatively.
- A `@triton.jit` or `@gluon.jit` body reads no module-level value: Triton fails to compile a plain `BLOCK = 64`
  read inside the kernel. Pass it as a `tl.constexpr` parameter or write it inside the device function.
- The authority is `aklr/core/validation/stages/audit/_rules.py` (SA0–SA4, reported as `SOURCE_REJECTED`); a
  source that does not parse or nests too deeply to audit is rejected (SA0).

### hip kernels

A `hip` kernel is C++: `.hip` translation units and headers whose paths may nest (`common/math.h`); the entry file
begins with a comment that describes the kernel. On first use aklr builds the sources with
`torch.utils.cpp_extension.load` for the GPU's arch under `<AKLR_BUILD_CACHE_DIR>/torch/` (`TorchBuilder` in
`aklr/core/runtime/builder/builder.py`). The package name is the build's `TORCH_EXTENSION_NAME` and the op
namespace, and the entry is the op the library registers there:

```cpp
#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor x, Tensor residual, Tensor weight, float epsilon, Tensor(a!) normalized, "
          "Tensor(a!) residual_out) -> ()", &run);
}
```

- The op's arguments are the schema's inputs then its outputs, in schema order (aklr calls it positionally), each
  output a `Tensor(a!)`, and it returns `()`. The rules of `run` hold.
- Launch on `at::hip::getCurrentHIPStream()`; allocate scratch with `at::empty` on an argument's device.
- Audit applies SH0–SH5 (`aklr/core/validation/stages/audit/_hip.py`) on the text with comments and literals
  blanked: no unterminated comment or literal (SH0); the one registration above, in the entry file, and no other
  library macro (SH1); includes are sources, `hip/`, `ATen/`, `c10/`, `torch/library.h` or a listed standard header
  (SH2); no synchronization, allocation outside torch, IPC, device or process change, environment, I/O or `cuda*`
  name (SH3); no `static`, `__device__`, `__managed__` or `thread_local` variable that is not const (SH4); the
  entry file's leading comment (SH5). Bench catches input writes and off-stream launches.

### hsaco kernels

An `hsaco` kernel launches precompiled code objects. Sources must be text, so the code objects are stored
base64-encoded in a `.py` file, and the entry point launches them through `aklr/core/jit/hsaco.py`:

```python
import struct

from aklr.core.jit import hsaco

from .codeobjects import GEMM  # bytes

_GEMM = hsaco.Kernel(GEMM, "gemm")


def run(a, b, bias, out):
    kernarg = struct.pack("<QQQii", a.data_ptr(), b.data_ptr(), out.data_ptr(), b.shape[0], a.shape[1])
    _GEMM.launch((a.shape[0],), (b.shape[0],), kernarg, device=a.device)
```

- `Kernel(code_object, symbol)` names a function in a code object built for the target's arch. The code object is
  loaded into the HIP runtime that torch uses, once per device, on the first launch on that device.
- `launch(grid, block, kernarg)` launches on the device's current stream. `kernarg` holds the kernel's explicit
  arguments, laid out as the code object expects; HIP fills in the hidden arguments.
- The launcher only supports regular launches through `hipModuleLaunchKernel`. Cooperative launches, cluster
  launches, and memory shared between ranks are not supported; kernels that need them require changes to
  `aklr/core/jit/hsaco.py`.
- `build.requires` pins the HIP that loads the code objects, such as `{"hip": ">=7.15"}`.
- Sources are audited like other Python kernels (SA0–SA4), then by SC1–SC2
  (`aklr/core/validation/stages/audit/_hsaco.py`), so that kernel code reaches the GPU only through the launcher.
  SC1 limits imports to sibling sources, `aklr.core.jit.hsaco`, `torch`, and a few standard modules with no side
  effects. SC2 rejects `__import__`, `eval`, `exec`, `compile`, `globals`, `vars`, and the torch APIs that reach
  native code (`torch._C`, `torch.ops.load_library`, `torch.utils.cpp_extension`, `torch.cuda.cudart`). The audit
  cannot see what the code objects do; bench checks their results.

## Workloads and dispatch

A workload is a Point: `{"axes": {<var axis>: int}, "evaluations": [...]}` (`aklr/core/schema/_workload.py`). CI
writes one evaluation per hardware:

```json
{"hardware": "gfx950/MI355X", "accuracy": "passed", "speedup": {"aiter": {"commit": "<sha>", "value": 1.31}}}
```

`speedup` is keyed by the baseline's name, the seam backend whose call the kernel replaces. A workload timed
against that baseline also gets `microseconds`, the time behind the speedup: the kernel's median time per call in
the same trials, to 3 significant digits. CI rewrites a file when a workload's accuracy changes, a speedup or a
time comes or goes, or a speedup moves by more than 2%, so a time always matches the speedup beside it; it is
not meant for comparing files measured in different runs.

Dispatch (`aklr/core/runtime/dispatch/`) resolves a call to the first variant whose condition matches, then looks
it up by its Const and Var axis values (`DispatchKey`). A kernel serves a workload on this GPU when
`build.target_hardware` matches and the workload has an evaluation for this hardware whose `accuracy` is `passed`
and whose speedup over the caller's baseline is at least `AKLR_MIN_SPEEDUP`, or was not measured against that
baseline (`Workload.qualifies`). Among several, the highest speedup wins. A Point no workload lists falls back to
the backend (`DispatchConfig.on_miss_policy`). A kernel that raises does not fall back.

## Names

`Schema.schema_name`: `<op_type>`, then `_<variant name>` unless the variant is `default`, then `<prefix><value>`
for each `Const` axis in declaration order, joined by `_`. `<prefix>` is the axis's `abbrev`, else its name; an
`abbrev` of `""` leaves the axis out. `Var` axes are not part of it. Examples: `add_rms_norm_hidden_size6144`,
`gemm_bf16_nt_shuffled_n4096_k4096`.

A variant is named `<op_type>.<variant name>` (`Schema.op_name`). Renaming an op type, a
variant or a `Const` axis, or changing an `abbrev`, renames committed schemas.

## Commands

| Command | Does | Exit |
|---|---|---|
| `python -m aklr validate [PATH ...] [--base REV] [--stages audit,bench] [--devices 0-7] [--under DIR] [--exclude DIR] [--no-timing] [--out DIR]` | audit, then bench; writes `report.json`, `report.md` and the measured schemas under `schemas/` in `--out` (default `aklr_validation/`). `--under` keeps the schemas under DIR, `--exclude` drops those under DIR. `--no-timing` builds and checks the kernels only, on the workloads not recorded failed: nothing is timed or written back | 0 passed, 1 failed, 2 usage, 3 aborted (the platform's fault) |

With `--base` and no `PATH`, a change that leaves `aklr/core/` alone validates only its changed schema files;
any other run validates every one. Other options: `python -m aklr validate --help`.

## Environment variables

Read only by `RuntimeEnv()` in `aklr/core/platform/env.py`; an empty value counts as unset.

| Variable | Values | Effect |
|---|---|---|
| `AKLR_DISABLE` | `1`, `0`, unset, `<op_type>,...` | `1`: emergency stop, no dispatcher starts and every call falls back. A list is parsed (`disabled_op_types`) but nothing reads it yet |
| `AKLR_MIN_SPEEDUP` | a finite number | the least measured speedup over the baseline that a workload needs to be served, default `1.05`; `0` or less also serves unmeasured workloads, never a `failed` one |
| `AKLR_BUILD_CACHE_DIR` | a path | where kernels are built; default `${XDG_CACHE_HOME:-$HOME/.cache}/aklr/kernels` |
| `AKLR_ENABLE_TRACING` | `1`, `0`, unset | record the calls that fall back as workloads; any other value is an error |
| `AKLR_SCHEMA_PATH` | a path | the directory tracing writes to and dispatch reads its kernels from, instead of `aklr/plugins/aiter/` |
| `AKLR_TRACE` | `plugin`, `op`, `all`, `0`, unset | log the plugins' routes, the op dispatches, or both; any other value is an error |
