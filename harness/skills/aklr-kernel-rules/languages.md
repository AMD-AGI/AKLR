# FlyDSL, Triton and HIP kernels in AKLR

What holds for a language only inside AKLR. The language itself is in the book's folders:
`../kernel-optimization-book/flydsl/SKILL.md`, `../kernel-optimization-book/triton/SKILL.md`,
`../kernel-optimization-book/hip/SKILL.md`. The rules for every language are in `aklr/plugins/aiter/README.md`
§ The kernel section and ABI, with § hip kernels and § hsaco kernels for those two.

## Every Python kernel (FlyDSL, Triton, Gluon)

- **The package.** AKLR writes a kernel's sources under `<AKLR_KERNEL_CACHE_DIR>/<language>/` and imports them as
  a package named `_aklr_<language>_` plus 16 hex digits of the kernel's content hash
  (`aklr/core/runtime/builder/builder.py`). A compiler diagnostic's path points there; its line is a line of the
  source in `kernel.sources`. A kernel with relative imports does not import as a top-level module: import it as a
  package ([validate.md](validate.md) § Reproduce one step outside validate).
- **Device code and launch code.** The audit treats a def whose single decorator is `@flyc.kernel`, `@flyc.jit`,
  `@triton.jit` or `@gluon.jit` as device code, and exempts a compile function's own body. Everything else that
  `run` reaches is launch code under SA2 and SA3: a def with a second decorator, a `@gluon.constexpr_function`,
  a def nested in a compile function. SA3's I/O names (`input`, `print`, `open`, ...) are rejected in device code
  too.
- **Passing the audit is not passing bench.** The audit reads names. A write to an input through a helper
  parameter with another name, a module-level dict that `run` fills, and a torch op whose output size depends on
  data (`torch.nonzero`, `x[mask]`) all pass it. Bench reports them as `input_mutation`, a mismatch and
  `capture_unsafe` (`../kernel-optimization-book/triton/authoring.md` §4, §8).
- **Every compile key is a function of the Point.** While bench captures a call it arms a tripwire on Triton's
  and FlyDSL's compile entries (`_CompileTripwire` in
  `aklr/core/validation/stages/_bench/_evaluation/_candidate.py`). A compile inside a capture is `capture_unsafe`
  with `compiled in the capture`. So take compile-time constants from the Point's shapes, dtypes and attrs. Keep
  at run time what can change while the Point stays the same: a scalar with a range (bench warms its default and
  the workload's value only), an `any_size` axis (dispatch does not key on it), the alignment of a pointer into a
  buffer you carve.
- **`requires`** names the language's toolchain and every library the kernel imports besides torch, each with the
  version you developed on as the lower bound (README). A kernel that imports aiter depends on the installed
  aiter, which AKLR does not ship: copying the code you reuse into the sources avoids that.
- **Search the definition files, not `.py` files.** Kernel sources are JSON strings:
  `grep -l kernarg-preload-count aklr/plugins/aiter/gemm/impls/flydsl/*/*.json` lists the kernels that use a construct.

## FlyDSL

- **The version is the baseline aiter's.** AKLR requires `flydsl>=0.3.2` (`pyproject.toml`). CI times
  `aklr/plugins/aiter` on the FlyDSL that the baseline aiter (`aiter` in `stack.json`) installs, and the report's
  first lines name it.
  The book's FlyDSL statements are for v0.3.2: check the version a number was taken on before you rely on a
  version-dependent statement (`../kernel-optimization-book/pitfalls/measured_a_different_build.md`).
- **The launch cache from FlyDSL 0.5.** From 0.5 FlyDSL traces a `JitFunction` again on every call. AKLR puts a
  cache in front of the JitFunctions defined in its kernel packages (`aklr/core/jit/flydsl_cache.py`): the first
  call of a specialization compiles, later calls launch the compiled function. The key is the Constexpr values,
  the type parameters, the compile hints and each runtime argument's signature (a tensor's dtype, alignment and
  layout). It follows that:
  - a module global that a traced function reads must not change after import: a cached launch does not read it
    again;
  - a call whose argument signature is new compiles, and inside a capture that is `capture_unsafe`;
  - a kernel imported any other way, in a probe or in forge_task's driver, is outside the cache. There every call
    traces again on FlyDSL 0.5 and later, and an eager timing measures the trace, not the kernel
    (`../kernel-optimization-book/pitfalls/eager_call_host_time.md`).
- **Host launch cost buys nothing on AKLR's ruler**, which reads kernel time in graph replays
  ([timing.md](timing.md)). It matters to an eager caller.
- **aiter's FlyDSL kernels are what you race**
  (`../kernel-optimization-book/flydsl/kernels/flydsl_kernel_library.md`). Price aiter through the contract's
  baseline class, the call bench times, not through a direct kernel call. Their counters and workspaces, kept in
  an `lru_cache` across calls, do not carry over to an AKLR kernel (SKILL.md, Stop table).

## Triton and Gluon

- `kernel.language` is `"triton"` for plain Triton and Gluon alike.
- **Triton's version is the image's.** CI runs the Triton of its image (`stack.json`) and fails if installing the
  baseline aiter replaces it. Locally, install aiter with `AITER_USE_SYSTEM_TRITON=1` and check
  `triton.__version__`; `report.json` records it under `platform`. Every committed Triton kernel requires
  `>=3.8`; what 3.7 lacks: `../kernel-optimization-book/triton/authoring.md` §3.
- **No module-level value in a jit body** (README): pass it as a constexpr parameter, or compute it in the device
  function. `@triton.autotune` or `@triton.heuristics` above `@triton.jit` makes the def launch code, which fails
  the audit: choose the configuration in a compile function.
- **The JIT key.** Triton specializes a runtime int on its class (1, a multiple of 16) and a pointer on its
  alignment. A value that changes class within a Point compiles again, possibly in a capture: list it in
  `do_not_specialize` (`../kernel-optimization-book/triton/authoring.md` §5). A warm `TRITON_CACHE_DIR` does not
  help: every in-process miss goes through the compile entry the tripwire watches.
- **Read what validate compiled.** `TRITON_CACHE_DIR=$PWD/tc python -m aklr validate --devices 0 --out <dir> <file>`
  leaves each compiled kernel's `.ttgir`, `.llir`, `.amdgcn` and options under `tc/`. AKLR's own flush kernel is
  there too. With `--base` the incumbent's kernels land in the same cache: the package name in each `.amdgcn`'s
  `.file` path tells them apart. Give each arm of an A/B its own cache directory.

## HIP

- ABI and audit (SH0–SH5): README § hip kernels. Timing: `harness/skills/read-validation/references/ruler.md`.
- AKLR builds a hip kernel with `torch.utils.cpp_extension` and `-O3` only (`aklr/core/jit/hip_extension.py`).
  What that means for the code: `../kernel-optimization-book/hip/SKILL.md` § Build and timing.
- `hip_extension.load_library(name, sources, root, arch)` keys its build directory on the toolchain and the arch
  only. A second package built under the same `root` finds the first one's marker and loads a library that does
  not exist. Give each variant its own `root` when you build variants yourself; AKLR's builder does.
- `requires` names `hip`: the HIP version torch was built with (`torch.version.hip`), not the ROCm release.
