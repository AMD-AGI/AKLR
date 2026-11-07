---
title: "Authoring a Triton or Gluon kernel for an AKLR schema"
kind: method
ops: [gemm, moe, attention, gdn]
gens: [gfx950]
triton: 3.8.0
basis: source-semantics
---

# Authoring a Triton or Gluon kernel for an AKLR schema

A procedure for authoring and validating a kernel at a schema's Points on gfx950 (MI355X) with Triton
3.8.0; following it does not guarantee that validation passes: language, files and entry, `run`, one binary per Point, scratch, launches, graph capture, fallbacks, a Gluon
skeleton, validation. AKLR's rules are linked, not restated (`aklr/ops/README.md` § The kernel section and ABI is the
authority). Source references are collected at the end; measured examples are limited to their stated shapes and toolchain.

## Route here when

- You write a new Triton or Gluon kernel for an AKLR schema (a core op under `aklr/ops/`, a plugin op under
  `aklr/plugins/sglang/`), or port one into a schema file, or your op has no contract yet (§3).
- You choose between plain `tl.*` and Gluon, or are about to add a launch.
- `python -m aklr validate` reports `schema_invalid` (`not at...`), `source_rejected` (SA0–SA4), `build_failed`,
  `capture_unsafe`, `input_mutation`, `no_matching_toolchain` or `no_qualifying_workload`, or demotes a Point.
- A kernel works eagerly and fails under graph capture or replay.

Elsewhere: layouts [gluon/layouts.md](gluon/layouts.md), data movement [gluon/memory.md](gluon/memory.md), the matrix
core [gluon/mfma.md](gluon/mfma.md), lane-level code [gluon/wave_ops.md](gluon/wave_ops.md), a working kernel's speed
[tuning.md](tuning.md), its IR, ISA and errors [debugging.md](debugging.md), accuracy [numerics.md](numerics.md), the
committed kernels by family [kernels/kernel_library.md](kernels/kernel_library.md).

## 1. The procedure

Start from the closest committed kernel and the branch its Point takes (§10); choose plain `tl.*` or Gluon (§2); write
the schema file, after the op's contract if it has none (§3), and a `run` that writes its outputs, keeps no state and
does not synchronize (§4); make fixed Point dimensions and compile-time configuration choices constexprs; keep ranged scalars and any_size axes runtime unless every selected variant is warmed (§5); allocate scratch in the call or cache per-shape scratch in a compile function, and initialize every region before its first read in each call (§6); consider a launch when a phase needs results from other workgroups or separate initialization/conversion; measure whether fusion is preferable, and keep every call capturable (§7, §8);
compile for gfx950 without a GPU (§10); after every edit give the file its new uuid and run
`python -m aklr export check` (§3); validate on one GPU, and fix or delete every Point it demotes (§11). The sections below explain the contract and implementation checks.

## 2. Plain Triton (`tl.*`) or Gluon

AKLR main has 154 Triton-language kernels: 4 plain `@triton.jit`, 150 `@gluon.jit`.

| Language | Families (kernel files) |
|---|---|
| plain `tl.*` (4) | `gemm@1.0` bf16 decode GEMMs n6144/k2048 and n2624/k6144 (2), RMSNorm and add-RMSNorm at hidden 6144 (2) |
| Gluon (150) | fp8 block-scaled GEMMs (24); MoE layer plugins: DeepSeek-V2/GLM MXFP4 (19), MiniMax-M2 FP8 (9) and MXFP4 (23), Qwen2-MoE router and experts (14); DSA sparse MLA decode (8); Qwen3.5 gated attention decode (9); GatedDeltaNet decode (18) and chunked prefill (26) |

The two share the launcher and the JIT cache (`@gluon.jit` is a `JITFunction` subclass, S1) and differ in the first
compiler stage: plain Triton runs `make_ttir` then `make_ttgir`, Gluon replaces both with `gluon_to_ttgir` (S2).

| | plain `tl.*` | Gluon |
|---|---|---|
| Layouts | assigned by the compiler: coalescing, the MFMA (`matrix_instr_nonkdim`; `kpack` is forced to 1 on gfx950, S4), conversion removal | written by you: `BlockedLayout`, `SliceLayout`, `DistributedLinearLayout`, `AMDMFMALayout`, `DotOperandLayout`, shared layouts; a `gl.arange`, `gl.zeros` or `gl.full` whose layout no other tensor fixes fails to compile ("Failed to infer return type") |
| Software pipelining | `num_stages` drives loop scheduling and LDS staging. In the n2624 16-row path, 3 stages use direct-to-LDS copies and about 99 KiB LDS; 1 stage uses register staging and 16 KiB LDS, and was about 2.2× slower at M=2/4/8. Recheck both resources and time when changing stages. | pipeline explicitly ([gluon/memory.md](gluon/memory.md) §12). In Triton 3.8.0, `num_stages`, `matrix_instr_nonkdim` and `schedule_hint` do not drive the Gluon pipeline; gfx950 normalizes `kpack` to 1. An ignored option can still split the JIT key. |
| Buffer instructions | `tl.load` / `tl.store` become buffer ops where a pass proves the range, from the `S` key (§5) or range analysis (S2, S3). | use `gl.amd.cdna4.buffer_load` / `buffer_store` explicitly; ordinary `gl.load` uses pointer loads ([gluon/memory.md](gluon/memory.md) §2, §3). Buffer and pointer versions can both be correct without having equivalent speed: the GDN two-token/four-slot recurrence had input-dependent differences under cold-device timing. |
| LDS | implicit | `gl.allocate_shared_memory`, `gl.amd.cdna4.async_copy.buffer_load_to_shared`, `commit_group` / `wait_group` (with two groups of two copies in flight, `wait_group(1)` becomes `vmcnt(2)`), swizzled and padded layouts |
| Matrix core | `tl.dot`, `tl.dot_scaled` | `gl.amd.cdna4.mfma` (`v_mfma_f32_16x16x32_bf16` in §10's skeleton), `mfma_scaled` with e8m0 scales (a 16 × 16 × 128 e4m3 tile: one `v_mfma_scale_f32_16x16x128_f8f6f4` with the scales in registers; `None` passes a register set to 0x7f, scale 1); `gl.dot_fma` is a vector-ALU dot product off the matrix core, on a `BlockedLayout` accumulator ([gluon/mfma.md](gluon/mfma.md) §1) |
| Lane-level code | `inline_asm_elementwise`, not knowing which lane holds which element | the same call, each element's lane fixed by the layout: `v_readlane_b32` of lane 5 returned x[5] with one element per lane, x[10] and x[11] with two (MI355X); shuffles and ballots: [gluon/wave_ops.md](gluon/wave_ops.md) |

- **`num_warps` is part of a Gluon kernel.** The module takes `ttg.num-warps` from the launch, and the verifier rejects
  a layout with another warp count ("Layout has 4 warps per CTA, but the context requires 8 warps per CTA") or with
  other than 64 lanes (S1). A launch without `num_warps` gets 4 (S4), which hard-coded 4-warp layouts rely on (S5).

The committed plain bf16 GEMMs and norms ([gemm/decode_gemm.md](gemm/decode_gemm.md)) are starting points,
not a language-choice rule. Their compiler-managed layouts and pipelining matter; reducing stages slowed the
measured decode GEMMs. Gluon is useful when you need e8m0 `mfma_scaled`, explicit LDS/async depth or lane-level
control. Compare complete implementations at the same Point; a faster Gluon implementation does not establish
that the language itself is faster, especially across different scale formats or fusion boundaries.

## 3. The schema and its files

- **Path.** `<package>/impls/triton/<name>/<uuid>.json`. A core op's package is `aklr/ops/<op>/`, a plugin op's
  `aklr/plugins/sglang/models/<pkg>/` or `aklr/plugins/sglang/layers/<pkg>/`; `<name>` is the scenario: the contract,
  the variants that differ from their first declared value and the const axes (`aklr/ops/README.md` § Names).
- **uuid.** The first 16 hex digits of the sha256 of `[name, authored parts]` as sorted-key JSON: the whole `kernel`
  section and each Point's axes and scalars, sorted (S6). Editing a source, `requires`, `author` or the Points renames
  the file; `speedup`, `accuracy` and the workload order do not. A file elsewhere fails audit with
  `<path>: not at <ops>/<op_type>/impls/triton/<name>/<uuid>.json`, which names the right uuid (S6); after every edit,
  `python -m aklr export check` names that path: exit 0 and no output on a clean repository, exit 1 and one `not at`
  line per misnamed file (`aklr/ops/README.md` § Workflow, steps 4–5).
- **Points.** `workloads` lists the Points the file serves: one value of each var axis, and the scalars. 116 of the 154
  files list one Point; per Point, dispatch serves the best measured file (`aklr/ops/README.md`, top). `kernel` holds
  `language` (`"triton"` for plain and Gluon alike), `targets` (gfx950, MI355X in all 154) and the four fields below.
- **`sources`.** One `{"path", "content"}` per file, each path one `<identifier>.py`, sorted (`aklr/ops/README.md` § The
  kernel section and ABI). 86 kernels are one `kernel.py`; 68 add 1–4 helpers (`attention.py`, `prologue.py`,
  `initial_state.py`,...) imported relatively (`from.adapter import validate_inputs`), the files being submodules of a
  package named after their content hash (S7). No schema imports another's files: each copies its helpers. Each
  `content` holds no control character but `\n`, as all 272 committed sources do, and ends in one newline with no blank
  line before it: the schema parser rejects a trailing blank line (`kernel.py:<line>: blank line at the end`), a tab and
  a carriage return (`kernel.py:<line>: forbidden character …`; same section).
- **`entry_point`.** `kernel.py::run` (139 files), or `main.py::run` with an `adapter.py` that checks the inputs (the
  15 N = 3072 fp8 GEMMs). The entry file needs a module docstring (its content: same section), else `source_rejected`
  (`SA4 kernel.py:0 no module docstring`); 135 files keep a `# Created by OpenAI` comment above it, which SA4 accepts.
- **`requires`.** `{"triton": ">=3.8"}` in 135 files. Triton 3.7.0 lacks `gl.clamp`, `gl.aggregate_replace`,
  `gl.expect_zero`, `gl.amd.slice`, `llvm_fn_attrs` and `gl.amd.cdna4`'s `compute_efficient_padded_shared_layout` and
  `scaled_upcast`; under it AKLR's toolchain check reports `>=3.8` unmet and an empty `requires` met, and none of
  the 11 n3072 fp8 GEMM files from m = 16 up compiles, m = 32 and 64 without calling those APIs. Declare it on every
  Gluon kernel you have not run with 3.7.0; the 19 files without it (the 15 n3072 fp8 GEMMs, the bf16 GEMMs, the
  norms) include 12 n3072 files that call 3.8-only API: add it when you start from one.
- **`author`.** One line naming a person, team or agent; the uuid hashes it, so editing it renames the file and drops
  the measurements until CI writes them again (same section).

A new scenario starts from `python -m aklr export` or the script of `docs/ops_tutorial.md` § 1. B; a second kernel
for a committed scenario from a copy of one of its files (`aklr/ops/README.md` § Workflow). Run from the repository
root, this writes the copy in the exact text the audit requires (S9) at its uuid path, and passes the schema-text/path checks when the source meets the `content` rule above. Passing source audit also requires the entry ABI and SA0–SA4 rules (§4):

```python
import json
from pathlib import Path
from aklr.core.contract.expr import json_file_text
from aklr.core.contract.locate import schema_uuid
from aklr.core.contract.op import OpContract

old = Path("aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json")   # a file of the scenario
schema = json.loads(old.read_text())
for w in schema["workloads"]:                      # only CI writes these
    w.pop("speedup", None)
    w.pop("accuracy", None)
schema["kernel"] = {"language": "triton", "targets": schema["kernel"]["targets"],
                    "requires": {"triton": ">=3.8"}, "author": "my-agent", "entry_point": "kernel.py::run",
                    "sources": [{"path": "kernel.py", "content": Path("/path/to/kernel.py").read_text()}]}
uuid = schema_uuid(schema["name"], OpContract.authored_parts(schema))
(old.parent / f"{uuid}.json").write_text(json_file_text(schema))
```

**Core op or plugin op.** A core op's kernel implements its contract (for example GEMM or RMSNorm) against the baseline defined by its package; current baselines include aiter and SGLang implementations. It is reached through its `aklr.ops.<op>` api; the current AKLR plugin route sources contain no direct call to those core apis. A plugin op's replaces a whole
SGLang function (an MoE layer's forward, a GatedDeltaNet layer, a decode attention step) through an `@sglang_route`,
against SGLang's own path, and may take framework state as mutable inputs (`aklr/plugins/README.md` § Architecture,
§ Rules for routes; `docs/plugin_tutorial.md`).

**No contract yet.** A kernel serves a contract: validate rejects a schema whose `contract_id` no op package registers
(`unknown contract_id`, exit 1). For an op without one, add the op package first, in its own pull request (its files:
`docs/ops_tutorial.md` § 1. A; for a plugin op, `docs/plugin_tutorial.md` steps 2–4). A layout variant that names an
output is rejected (`names unknown tensors`), and so is an output of `float4_e2m1`, AKLR's only 4-bit dtype (S45): an
output is row-major in the order of its declared shape (`per_group_quant_fp8@1.0` declares its scales
`[n_groups, n_tokens]` to store them group-major), and MXFP4 codes leave packed in `uint8`, an integer output that must
equal the reference ([numerics.md](numerics.md) §1). The nearest quantization op, `per_group_quant_mxfp8@1.0`, writes
row-major e4m3 codes and uint8 e8m0 bytes (S45). A standalone SwiGLU + MXFP4 requantization would be such an op. In the committed DeepSeek-V2 schema `7e9c13e0f2d58a3b`, the one-token `_finish_gate_up` launch writes packed uint8 activations and e8m0 scales into call-local `aq`/`aqs` scratch, then `_down_reduce` reads those buffers through `_tiny_projection`. Its activations are row-major; its expert weights use shuffled offsets. Other MoE scratch formats must be read from their own producer/consumer source chain.

## 4. `run`: what it receives, what it must not do

`run` takes exactly the schema's `kernel_call_keys`, keyword-only: its tensors and scalars (an absent optional key is `None`; inputs hidden by schema-default optional attrs are omitted),
attrs that have a parameter, layout arguments such as `operand_format`, `out_dtype` when the one output is `out`, and
every output, preallocated (`aklr/ops/README.md` § The kernel section and ABI); from the repository root,
`ContractRegistry.default().parse_schema(Path(p).read_bytes(), p).kernel_call_keys` (`aklr.core.contract.registry`)
lists them for the schema at `p`. The Point's logical sizes are derived from the tensors' stored shapes under the contract's layout and packing rules; a packed 4-bit last dimension holds two logical elements per stored element. It must write every output; leaving a row unwritten violates the contract. Mutable inputs, no return value, no output kept, and support for every value of a scalar's range are specified in the
same README, § Schema fields. MXFP4 and e8m0 tensors may arrive as `uint8` or
`torch.float4_e2m1fn_x2` / `torch.float8_e8m0fnu`; the latter are dtypes which
Triton 3.8.0 cannot type (`KeyError: 'float4_e2m1fn_x2'` at the launch, S20): pass `.view(torch.uint8)`, as the DeepSeek
MoE's entry does (S11).

Dispatch calls `run` only with tensors in the stored layout, not negated, aligned (`data_ptr % 16 == 0`), on the
registered device, and with no output byte overlapping an input (S10). Strides may therefore be folded from the
contract's shape. A scalar declared with one value can be hard-coded: DeepSeek MoE's
`routed_scaling_factor` is declared `(2.5,)` (S11). **Keep ranged scalars and `any_size` axes runtime.** A GDN slot
pool can change size within the same Point. Hard-coding RMSNorm `epsilon=1e-5` can pass validation's default inputs
and still be wrong for a legal `epsilon=1e-3`; validation is not exhaustive over a scalar range.

The audit's rules SA0The audit's rules SA0–SA4 are stated in `aklr/ops/README.md` § The kernel section and ABI (their code: S12); a
`source_rejected` finding names the rule, file and line. What they mean for a Triton or Gluon kernel:

- A def whose single decorator is `@triton.jit` or `@gluon.jit` is recognized as device code (S13).
  A `@gluon.constexpr_function` called from `run`, a plain helper, or a multiply decorated def is audited as host
  launch code. Use a compile function instead of first-call autotuning: multiple unpruned configurations may be
  timed on a cache miss, and different processes may select different binaries (S17).
- SA3's I/O names are rejected in device code too (S12): avoid a parameter named `input` and device-body `print`.
  Use `gl.device_print` during standalone debugging, then remove it.
- Passing audit is not proof of runtime safety. An input write hidden behind a renamed helper can evade SA2 but
  still violate the bytewise read-only input check (S14). Some host clocks/environment reads also pass audit;
  they do not become permitted state-dependent dispatch or capturable operations.
- **A `@triton.jit` / `@gluon.jit` body does not read an ordinary unwrapped module-level Python constant under the default compiler policy** (same section): pass it as a `tl.constexpr` /
  `gl.constexpr` argument or compute it in a `@gluon.constexpr_function` (`from triton.experimental import gluon`;
  3.8.0's `gl` has none). Triton also accepts globals made with `gl.constexpr(...)` (five AKLR files keep such globals,
  layouts in three, S15), modules, JIT functions and dtypes; a plain `BLOCK = 64` fails the default compile (S16). Reassigning a global read by a compiled kernel can raise `RuntimeError` on later launches (S17).
  Do not rely on mutation of globals to select per-call behavior.

## 5. One binary per Point

**Pass the token count as a constexpr.** 122 of the 150 Gluon kernels have a `@gluon.jit` parameter `M`, `ROWS`,
`TOKENS` or `BATCH` declared `gl.constexpr` (a name count, Qwen3-Next's runtime-count files included). Each Point then
specializes on first call; different Points may share a configuration. Constexpr dimensions can remove masks
when the tile is fully covered, but do not guarantee elimination of every branch or lower register use.
The first call must be outside capture and warm every variant that the Point's scalars can select
(`aklr/ops/README.md` § The kernel section and ABI). `gl.static_range` fully unrolls a constexpr count;
use it for short fixed sequences, not for a contract allowing thousands of sequences (S43).

**Choose the configuration once, where host and kernel both read it.** A `@gluon.constexpr_function` called outside
JIT returns its Python value and inside JIT a constexpr (S17), so one function gives the launch its grid and the kernel
its tile shapes: the Qwen2-MoE router's kernel reads `config: gl.constexpr = _shape_config(M)` and its host sizes the
grid from `_shape_config(rows)` (S18), as the skeleton of §10 does with `_config`.

**A `@functools.cache` compile function** is the pattern `aklr/ops/README.md` § The kernel section and ABI shows: a
module-level def of shapes whose cache holds compiled code, constants and per-shape scratch only. Allocation in the compile function is allowed for per-shape scratch, but passing the source audit does not
establish safety for every nested helper or cached object. Keep cached values limited to compiled code, constants
and scratch rewritten before every read.
Eleven of the 154 kernelsEleven of the 154 kernels have one: the two plain-Triton GEMMs cache a launch closure per shape
(`_launch_for`), n2624's with its workspace size (S19); the nine Qwen3.5 gated-attention files the fp32 unit KV scale
both launches read, `_unit_scale(device)` returning `torch.ones(1,...)` (S35): in `run`, `torch.tensor` is rejected
(SA3) and `torch.ones` allocates and fills a tensor per call. None of the 154 calls `warmup` or `triton.compile`: binaries live
in Triton's in-process JIT cache, which meets the first-call rule while the key below is a function of the Point.

**What enters Triton 3.8.0's JIT key** (S17): `kernel_key_cache` uses
`(tuple(specialization), str(options))`; the binary cache uses its mapped string key.

| Argument | Enters the key as | So |
|---|---|---|
| `tl.constexpr` / `gl.constexpr` | its value, which must be hashable (S17) | one binary per value; a list fails at the launch (§10) |
| int | `i32` / `i64` / `u64`; the value 1 becomes a constexpr; a multiple of 16 gets `D` (S20) | Points share a binary only within one class; `do_not_specialize=[...]` on `@gluon.jit` drops the value keys |
| runtime float, bool | `fp32`, `u1` (S20) | their tested signatures do not value-specialize |
| `None` | a constexpr (S20) | an absent optional input compiles its own variant |
| tensor (HIP) | dtype; `D` for 16-byte pointer alignment; `S` when buffer passes are enabled and whole storage is at most 2^31−1 bytes (S20, S21) | `D` becomes `tt.divisibility=16`, `S` becomes `tt.pointer_range=32`; a misaligned view or storage crossing 2 GiB can select another binary. The Python specialization callback runs per tensor even on cache hits. |
| launch options | `str(options)`: `num_warps`, `waves_per_eu`,... as passed, in keyword order (S17) | launch from one call site; an option Gluon ignores (`num_stages`) still splits the key |

**Keep runtime** what may change while the Point stays the same, since a key that changes after the Point's first call
compiles again, possibly inside a capture (§8): runtime floats (S22; tested fp32 signatures); scalar inputs, unless one value per process is guaranteed (the MiniMax EP files take `expert_offset` as
`OFFSET: gl.constexpr`, one value per process by their contract, S23); `any_size` axes (a pool's slot count), on which
dispatch does not key; and the alignment of pointers into a buffer you carve (§6). A runtime int is still keyed on its
class (1, a multiple of 16, i32, i64) and bench warms only the default and the workload's scalars (S14): put an int that
may change within a Point in `do_not_specialize` (dtype classes `i32`, `i64` and, for large non-negative values, `u64` still remain).

## 6. Scratch

- **Allocate inside the call**, on an argument's device, sized from the shapes: `torch.empty` (117 kernels, 686 call
  sites), `torch.empty_like`, `.new_empty`. One allocation can be carved into typed regions at aligned offsets (64 KiB
  or 256 bytes in AKLR's arenas, S24); an offset that is not a multiple of 16 bytes loses `D` and gets another binary
  (§5). A first call over bench's 16 GiB limit (S14) fails `build_failed`, as one allocating 17 GiB did.
- **`torch.empty` holds whatever was there**, in an eager call and in every replay. In every call, write before any read
  what a launch reads before it writes: atomic counters and cursors, `-1` job sentinels, padding rows, accumulators
  added into. AKLR's 154 Triton kernels never call `torch.zeros` or `fill_`: an earlier launch of the call writes the
  initial values, as program 0 of the MiniMax MXFP4 MoE's first launch stores 0 into the expert counts and the total and
  -1 into the job slots, on which the next launch takes atomic tickets (S25).
- **State that must start fresh** is cleared by a small launch reading its flag on the device, as GDN prefill's
  `initialize_fresh` zeroes the slot of each sequence without initial state (S5).
- **Nothing carries from one call to the next.** `run` keeps nothing derived from an argument across calls, a compile
  function only compiled code, constants (§5) and per-shape scratch rewritten whole before every read (same README
  section). The audit rejects a `global` statement (SA1), but does not detect every module-level container write.
  A dict retaining the first output can therefore reach bench and fail later fresh-input checks
  ([../pitfalls/state_kept_across_calls.md](../pitfalls/state_kept_across_calls.md)).
- **Reuse a region within a call in stream order**, as MoE files overlay short-lived scratch on a later buffer
  (S26): its last reader must be launched before its next writer, on the same stream. Never overlap live regions.

## 7. Launches: what one costs, where a boundary goes

What one more launch costs depends on the ruler. On a trivial Gluon load/add/store workload on MI355X:

| Where | Representative cost | What is included |
|---|---|---|
| Bare CUDA-graph replay | about 1.8 µs per added chained kernel | device graph execution; not a production SGLang graph |
| AKLR stored speedup | about 4 µs per trivial kernel in the tested process | summed Kineto kernel durations, flush excluded (S27); the floor is process-dependent |
| Eager Triton launch | about 7.3 µs with one tensor; about 1 µs more per tensor argument | Python/JIT specialization and enqueue; not intrinsic GPU kernel time |

An eager loop is host-limited only when kernels finish faster than Python issues them. Reduce launch and argument
count on an uncaptured prefill path; a graph replay does not rerun Python. Do not carry these costs between rulers
or treat them as fixed hardware constants ([debugging.md](debugging.md) §8).

**Where the committed kernels end a launch.** Typical source launch boundaries are: one kernel in fp8 GEMMs and norms; a GEMM plus fixed-order reduce in
n2624 at M64; several preparation, projection and finish kernels in MoE, attention and GDN.
The stages they separate:The stages they separate:
[moe.md](moe.md) §2, [attention.md](attention.md) §2, [gdn.md](gdn.md) §3 and §7.

**Consider a launch where a value crosses workgroups or a phase needs separate initialization/conversion.** `gl.barrier` synchronizes one workgroup, and gfx950 has no CTA
clusters to widen it (`num_ctas > 1 not supported on gfx950`, S4); no result may depend on a workgroup that may not be
running, since a launch's workgroups need not all be resident: no unbounded spin on a partner, and no giving up and
computing without its data. So in AKLR's Triton kernels a value one workgroup produces and another consumes — a token's
expert logits before its top-k, every split before a merge, every reader of a state before it is overwritten — ends the
launch, and stream order is the synchronization. An in-launch last-arriver fold is an alternative, not an automatic launch saving
([../gemm/wait_free_fold.md](../gemm/wait_free_fold.md)). Publish partials with the required memory ordering,
reset counters for **every** call/replay, and include initialization in the full-call comparison. A separate
zero node is one reset strategy; a correct last-arriver reset can avoid it. Tested Gluon folds were slower than
partial-plus-merge at the measured sizes; that is not a universal atomic-cost rule.

- Independent phases can share a program-id-role grid (DeepSeek router beside input MXFP4 quantization, S11;
  [moe.md](moe.md) §3). This removes a launch, but the fused grid can change occupancy and replay time.
- A small decision can be recomputed by consumers instead of published (S11; [moe.md](moe.md) §4). Compare
  repeated decision work against publication plus synchronization; top-k size, consumer count and split count
  can reverse the winner. Do not use a constant launch saving as a universal crossover threshold.
- A split reduction under a `bitwise` contract needs a repeatable accumulation order, as in the fixed-order
  reduce launch of n2624 (S19), not an arrival-order float-atomic sum.

A torch op in `run`A torch op in `run` is a launch too: the DSA files that cast `q` with `q.to(torch.bfloat16)` there add a kernel per call (S31).

## 8. CUDA-graph capture

SGLang captures decode, and AKLR's decode routes run inside its graphs; the GDN prefill routes return `out_of_contract`
while a capture runs, "a capture would replay this call's lengths" (S29). Bench captures every check step — a first
capture replayed twice with new values copied into the same buffers, fresh captures per input set, a repeat under
`bitwise`, the post-timing checks (S32) — and fills the outputs with NaN (0xFF bytes for integer and sub-byte dtypes)
before the call (S14). Leaving any output element unwritten violates the contract. Poisoning helps expose missing stores, but is not
proof that every omitted write will be caught in every invocation.

| Breaks | Why | Do |
|---|---|---|
| The Point's first call inside a capture | dispatch serves the fallback (`not_warmed`), and the graph keeps it in every replay (S33) | call every Point once eagerly first, as SGLang does (`aklr/ops/README.md` § The kernel section and ABI) |
| A JIT key that changes after warm-up (§5) | an in-process miss can compile during capture, even with a warm disk cache (S14, S17) | make keys depend only on the Point, or warm every legal variant; do not assume validation exhausts all changing keys |
| `.item`, `.cpu`, `.tolist`, `torch.cuda.synchronize`; data-dependent output sizing (`nonzero`, bool indexing, `argwhere`, `repeat_interleave` without `output_size`) | host/device synchronization is unsafe during capture | compute indices into a worst-case buffer on device; where applicable supply `output_size` |
| A host decision on device data: a grid or a branch from a count the GPU computed | it needs a synchronization, and a replay keeps the captured decision | size the grid from shapes for the worst case and exit early on a count read on the device (below) |
| State written once and assumed afterwards | a replay reruns only the kernels, never `run`'s Python | write counters, sentinels and padding in every call: "Padding is rewritten every invocation, including graph replay" (S34) |
| A Python value that changes per call | frozen into the graph at capture | read it on the GPU: "Live lengths, physical cache slots, and KV scales are always read on the GPU" (S35) |
| Allocation | capture reuses the captured allocation on replay | allocate per call (§6), initialize before read, and never depend on previous contents |

The worst-case grid with an early exit, as the MiniMax MXFP4 MoE sizes its expert GEMM (S36): the host launches
`capacity = cdiv(min(routes * block_m, routes + e * (block_m - 1)), block_m) * block_m` rows of tiles, and a program
returns at once when `tile * BM >= gl.load(Total)`, the row count the GPU computed.

## 9. Fallbacks: when the call is not yours

- **The kernel does not fall back.** Dispatch calls `run` only at the Points its file lists (and every value of an
  `any_size` axis); an exception from `run` is logged with the `AKLR_DISABLE` hint and raised (S33). Host `assert`s or
  `adapter.py`'s `ValueError`s document assumptions: "nothing falls back on an error" (`aklr/ops/README.md` § The
  kernel section and ABI).
- **Dispatch decides** when a call is not the kernel's: the api returns a `FallbackReason` and the caller takes its own
  path (S33) — `out_of_contract` when no Point describes the call's shapes, dtypes, layout, alignment or aliasing,
  `no_workload`, `not_measured`, `inaccurate`, `not_qualified`, `disabled`, `not_warmed` (`docs/ops_tutorial.md` § 5).
  To serve more or fewer Points, edit `workloads`, not `run`. A Point you add runs a branch CI never measured, and
  generated files keep branches that no longer fit the ABI: the Qwen3.5 GDN prefill file `0a07f9f9342c6cf1` takes its
  `m > 4096` path at none of its Points, and that path static-asserts a row stride the `normalized` output (2048)
  cannot meet (S42). Validate each added Point; schema validity and queue inclusion do not establish GPU execution,
  correctness or performance admission.
- **Routes decide on SGLang's configuration** (graph modes, features, parallelism) **and on how SGLang hands over its
  tensors** (the current routes have 5 composite decision blocks testing shapes, dtypes or lengths: DSA's `q_nope` / `q_rope` views of one buffer, the
  `out_proj` weight shape that selects SGLang's aiter GEMM, prefill lengths against rows) and return a `FallbackReason`.
  Shapes, dtypes, token counts and GPU models the op covers are the op's to judge; the route does not check them again
  (`aklr/plugins/README.md` § Rules for routes). The hook then runs SGLang's original function, as it always does under
  `torch.compile` tracing (S37). Where the op's inputs cost work, the route first asks `op.fallback_reason(...)` on
  stand-ins, as the Qwen3-Next prefill route does before its input projections (S29): the question launches no kernel
  and does not enter `run` (on `gemm@1.0`: `None` when served, `out_of_contract` for a misaligned `a`).

## 10. Starting a new kernel

1. **Start from the closest family** ([kernels/kernel_library.md](kernels/kernel_library.md) §3) and copy its file
   layout: entry, helpers, the host function that plans the launches. Read the branch your Point takes;
   keep its coupled constants together. A lone split or fold edit can select an undefined branch or crash LLVM.
2. **Or start from this skeleton**, a `gemm@1.0` kernel for a scenario with `trans_b` and no bias such as
   `gemm_n6144_k2048`: 16-byte loads in a `BlockedLayout`, masked `buffer_load` / `buffer_store`, one MFMA with an
   `AMDMFMALayout` accumulator and `DotOperandLayout` operands, the token count as a constexpr, one configuration
   function for kernel and launch. It is a compileable authoring example, not a competitive kernel: tested N6144/K2048 outputs were within
   the contract, but its listed decode Points were too slow for AKLR admission. Add and measure prefetch/grid
   changes before using it as a submission.

```python
"""Decode bf16 GEMM out = a @ b.T (gemm@1.0, trans_b, no bias) on gfx950 in Gluon: each program owns
BLOCK_N output columns for every row of a, walks all of K with fp32 MFMA accumulation and stores once."""

from typing import NamedTuple

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_Config = NamedTuple("_Config", [("block_m", int), ("block_n", int), ("block_k", int), ("warps", int)])

@gluon.constexpr_function
def _config(m):
    """The configuration of one Point, read by run and by the kernel."""
    block_m = 16 if m <= 16 else 32 if m <= 32 else 64
    return _Config(block_m, 64, 128, 4)

@gluon.jit
def _gemm(A, B, Out, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    cfg: gl.constexpr = _config(M)
    BM: gl.constexpr = cfg.block_m
    BN: gl.constexpr = cfg.block_n
    BK: gl.constexpr = cfg.block_k
    WARPS: gl.constexpr = cfg.warps
    gl.static_assert((M <= BM) & (N % BN == 0) & (K % BK == 0))
    # 8 bf16 (16 bytes) per lane along K, which is contiguous in both a (m, k) and b (n, k).
    a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [WARPS, 1], [1, 0])
    b_load: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, WARPS], [0, 1])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                              warps_per_cta=[1, WARPS])
    a_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
    b_op: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)

    col0 = gl.program_id(0) * BN
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, a_load))
    ka = gl.arange(0, BK, layout=gl.SliceLayout(0, a_load))
    kb = gl.arange(0, BK, layout=gl.SliceLayout(1, b_load))
    cols = col0 + gl.arange(0, BN, layout=gl.SliceLayout(0, b_load))
    acc = gl.zeros([BM, BN], gl.float32, layout=mma)
    for k0 in range(0, K, BK):
        a = gl.amd.cdna4.buffer_load(A, rows[:, None] * K + (k0 + ka)[None, :], mask=(rows < M)[:, None])
        b = gl.amd.cdna4.buffer_load(B, cols[None, :] * K + (k0 + kb)[:, None], cache=".cg")  # read once
        acc = gl.amd.cdna4.mfma(gl.convert_layout(a, a_op), gl.convert_layout(b, b_op), acc)

    out_rows = gl.arange(0, BM, layout=gl.SliceLayout(1, mma))
    out_cols = col0 + gl.arange(0, BN, layout=gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(acc.to(gl.bfloat16), Out, out_rows[:, None] * N + out_cols[None, :],
                              mask=(out_rows < M)[:, None])

def run(*, a, b, trans_b, bias, out_dtype, out):
    m, k = a.shape
    n = b.shape[0]
    cfg = _config(m)
    _gemm[(n // cfg.block_n,)](a, b, out, M=m, N=n, K=k, num_warps=cfg.warps, waves_per_eu=2)
```

   `version=4` is gfx950's MFMA (S38; [gluon/mfma.md](gluon/mfma.md) §1). The masks keep rows past M out: Triton's
   buffer descriptor covers 2^31 − 2 bytes from the base, not the tensor (`num_records` `0x7ffffffe` in every buffer op
   of 27 probe kernels), so an unmasked `buffer_load` past a tensor's end read the memory behind it, while masked lanes
   read 0 without `other`  ([gluon/memory.md](gluon/memory.md) §4). The two `convert_layout`s go through 8
   KiB of LDS (loading in the operand layout instead: [gluon/layouts.md](gluon/layouts.md) §6, §9). The launch is a
   start, not a tuned choice: N6144 gives 96 programs, fewer than 256 CUs. Prefetch improved this particular
   skeleton, but also changed register allocation and scheduling; bytes-in-flight causation is not isolated.
   `waves_per_eu=2` requests a register budget, not measured residency ([tuning.md](tuning.md) §6–§10).
   Decode GEMM designs: [gemm/decode_gemm.md](gemm/decode_gemm.md).
3. **Compile for gfx950 without a GPU.** `triton.compile` with an `ASTSource` (`GluonASTSource` for Gluon) and
   `GPUTarget("hip", "gfx950", 64)` builds the launch's instructions, given its constexprs and specialization (`"DS"`
   for 1 MiB tensors at aligned addresses and `"D"` for n = 1024 matched; a pointer 4 bytes off alignment in a 3 GiB
   range with n = 1000 gets neither). Signatures name the dtype the launch sees: `*bf16`, `*fp16`, `*fp32`, `*fp8e4nv`
   (`float8_e4m3fn`), `*fp8e5` (`float8_e5m2`), `*u8` (also the uint8 views of MXFP4 and e8m0), `*i32`, `*i64`; an int
   is `i32` or `i64` by signed range, or `u64` above the signed-64 range (a constexpr when it is 1); a float is `fp32`, a bool `u1` (S20). It needs Triton and the
   kernel's imports (torch in 152 of the 154 AKLR files), not a GPU. A kernel whose `kernel.py` imports a sibling
   relatively (DSA decode `1148d1c8e30397f7`) fails `from kernel import...` ("attempted relative import with no known
   parent package"): write its sources into `kpkg/`, put its parent on `sys.path` and `from kpkg.kernel import...`, a
   package as AKLR makes one (S7). Offline code is representative only when signature, specialization, constexprs, target and launch options
   match the real call. This compiles the skeleton with Triton 3.8.0 for gfx950:

```python
import triton
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from kernel import _gemm

def compile_gfx950(fn, signature, constexprs, attrs, options):
    sig = dict(signature, **{name: "constexpr" for name in constexprs})
    index = {name: (fn.arg_names.index(name),) for name in sig}
    source = GluonASTSource if fn.is_gluon() else ASTSource
    src = source(fn, sig, {index[n]: v for n, v in constexprs.items()},
                 {index[n]: HIPBackend.parse_attr(d) for n, d in attrs.items()})
    return triton.compile(src, target=GPUTarget("hip", "gfx950", 64), options=options)

k = compile_gfx950(_gemm, signature={"A": "*bf16", "B": "*bf16", "Out": "*bf16"},
                   constexprs={"M": 16, "N": 6144, "K": 2048}, attrs={"A": "DS", "B": "DS", "Out": "DS"},
                   options={"num_warps": 4, "waves_per_eu": 2})
print(k.metadata.shared, k.asm["amdgcn"].count("v_mfma_f32_16x16x32_bf16"))    # 8192 8
```

   LDS is `k.metadata.shared` (8192 here), not the static `.amdhsa_group_segment_fixed_size` (0).
   A Gluon `k.asm` holds frontend `source`, `ttgir`, `llir`, `amdgcn`, `hsaco`
   ([debugging.md](debugging.md) §1–§2).
4. **Know what only a launch catches.**
   - A constexpr must be hashable: `INSTR=[16, 16, 32]` at launch failed with `unhashable type: 'list'` where
     `triton.compile` worked , since the launcher hashes each constexpr into its key (S17): pass a tuple.
   - A misspelled launch option raises `KeyError: Keyword argument... was specified but unrecognised` when the launch
     compiles (S17); `triton.compile` silently drops options that are not `HIPOptions` fields (S4).
   - The launch derives `D` and `S` from the real tensors and ints (compile with the same attributes), and raises if a
     `gl.constexpr` global the kernel read has changed since compiling (§4, S17).
   - LDS over the 160 KiB (163840 bytes) a workgroup may take: `triton.compile` builds it (`metadata.shared` 262144 for
     gfx950) and the first launch raises `out of resource: shared memory` (S4; [debugging.md](debugging.md) §4).

## 11. Validating

```bash
python -m aklr validate --base origin/main --stages audit,bench --devices 0 <schema.json>
```

`--stages` is `audit` or `audit,bench`; `--devices` takes `0-7` or `0,2,5`; the report goes to
`aklr_validation/report.md` and `aklr_validation/report.json` unless `--out` says otherwise (S39). `--stages audit`
alone needs no GPU. Without `--base` the committed `083b493a` was timed against the baseline alone, at the retain
threshold; a copy with `--base origin/main` against the baseline at the entry threshold and the serving file
(`incumbent`) at the retain one. A Point that stores less than `AKLR_MIN_SPEEDUP` is not served (`not_qualified`): the
committed RMSNorm `cf637bff0784c454` stores 1.023–1.029 at its 7 Points of 3–18 tokens, none served at the default 1.03,
its Points storing 1.04 and more served. Tune past the dispatch threshold (`aklr/ops/README.md` § Environment variables;
`harness/skills/aklr-read-validation/references/ruler.md` § Judging; reading a report: its `SKILL.md`).

**Demoted Points.** Validate fails only on a file-wide finding or when no Point qualifies (`no_qualifying_workload`). A
finding at one Point — a speed miss, the correctness codes, `build_failed`, `candidate_crash` or `candidate_timeout`
there — demotes that Point (S41): a `numeric_mismatch` planted at one Point of three (1024 tokens) was listed under
"demoted:" in `aklr_validation/report.md` (`demotions` in `report.json`, with its `post_timing_mismatch` and
`regressed`), and the run printed `validation passed` and exited 0. A one-Point file that computes wrong fails as
`no_qualifying_workload`, the cause in the demoted list. Fix each demoted Point or delete it from `workloads` (the file
takes a new uuid, §3), then validate again (`docs/ops_tutorial.md` § 3).

What bites:

- **A git checkout**, with the file at its canonical path in the exact schema text (§3, S6, S9): validate asks git for
  the repository root (`git rev-parse --show-toplevel`) and takes only paths inside it (S39); a copy without `.git`
  and a file outside the repository each exit 2 before any check.
- **Measurements.** Delete `speedup` and `accuracy` from every workload of a new or edited file: with `--base`, measured
  fields that differ from the base are `perf_edited` (S13), and a file measured on some workloads only is invalid.
- **The image.** CI runs `lmsysorg/sglang:v0.5.22-rocm10-mi35x` with the baseline aiter (`AKLR_BASELINE_AITER`)
  installed over it, and refuses any Triton but 3.8.0 (S40). Validate in that image after installing the same aiter as
  `.github/README.md` § Install writes it, with `AITER_USE_SYSTEM_TRITON=1` to keep an installed Triton. Otherwise, aiter's setup.py attempts its Triton installer unless torch is below 2.9.1; installer failures may also be skipped (S40). A baseline that reaches aiter can move with its build; reproduce CI's installed baseline, not just its
  container tag ([tuning.md](tuning.md) §1). Check `triton.__version__` and the actual package build. Triton 3.7
  has different Gluon APIs and scheduler behavior; compile artifacts and tuning settings do not transfer merely
  because the hardware is unchanged.

Correctness: benchCorrectness: bench compares every output with the contract's reference under the schema's `compare` block —
`determinism` `bitwise` (the default: the plain-Triton GEMMs and norms, DSA, Qwen3.5, Qwen3-Next) or `bounded` (the
MoE plugins and the fp8 block-scaled GEMMs), and each output's error bound — at the steps of §8 (`aklr/ops/README.md`
§ Schema fields; [numerics.md](numerics.md) §1, §8). What a finding usually means here (at one Point, a demotion):

| Finding | Usually | § |
|---|---|---|
| `schema_invalid` | not at the uuid path, not the exact schema text, workloads half measured | 3, 11 |
| `source_rejected` | SA0–SA4 | 4 |
| `build_failed` | an import error, an entry wrapped so that it cannot take the call keys (a `run` missing a key is `source_rejected` already, SA4), a compile error at the first call, a first call over 16 GiB | 3, 6, 10 |
| `candidate_crash` | a memory fault or segfault, an output of the wrong shape or dtype | 6, 10 |
| `candidate_timeout` | a hang: a workgroup waiting inside a launch for another (§7), a device loop on a value never written (§6); reported after 900 s plus 300 s per Point of the batch (S41) | 6, 7 |
| `capture_unsafe` | compiled in the capture, synchronized, or a replay with new inputs gave another result | 8 |
| `input_mutation` | a read-only input written | 4 |
| `numeric_mismatch`, `nondeterministic`, `post_timing_mismatch` | the math ([numerics.md](numerics.md)), also an arrival-order sum moved past its bound (`numeric_mismatch`, not `nondeterministic`); `post_timing_mismatch`: state kept across calls | 6, 8 |
| `not_faster`, `regressed` | the speed ([tuning.md](tuning.md)) | — |
| `no_qualifying_workload` | no Point passed its checks and the entry threshold: the demoted list gives each Point's cause | 11 |
| `no_matching_toolchain`, `no_matching_device` | the installed Triton misses `requires`; the GPU is not in `targets` | 3 |

**A/B a variant** as a copy with the one change, at its own uuid path. Against the committed kernel of its Point,
validate the copy with `--base origin/main`: candidate, incumbent and baseline are timed in the same rounds.
For two uncommitted variants, alternate A, B, A, B on one idle GPU, with separate report/cache directories.
Read candidate time and paired uncertainty, not just the rounded console speedup; run A-vs-A controls for small
changes ([tuning.md](tuning.md) §2).

## Evidence

AKLR source paths refer to main `43e112e` unless another commit is stated.
Triton compiler paths refer to `triton-lang/triton@c01b677` (v3.8.0).
Measurements in the text are gfx950 examples, not universal performance guarantees.

- S1 triton-lang/triton@c01b677:python/triton/experimental/gluon/_runtime.py:45 `GluonJITFunction(JITFunction)` swaps only the AST source; the module takes `ttg.num-warps` from the options (:32) and `ttg.threads-per-warp` from the warp size (:34); the verifier requires a layout's warp count to equal the module's (lib/Dialect/TritonGPU/IR/Dialect.cpp:3877, lanes :3863)
- S2 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:585 `add_stages`: Triton `make_ttir` + `make_ttgir`, Gluon `gluon_to_ttgir` (:327–349, no pipeliner, no buffer-op pass); `make_ttgir` coalesces (:269), reads `matrix_instr_nonkdim` and `kpack` (:273) and `num_stages` (:288–289), runs the buffer-op passes under `use_buffer_ops` (:303); `schedule_hint` "right now no effect" (:104); `AMDGCN_USE_BUFFER_OPS` defaults to on (python/triton/knobs.py:517)
- S3 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/ConvertToBufferOps.cpp:135 `canUseBufferOps`: a splatted base plus 32- or 64-bit offsets, safe by `tt.pointer_range` 32 on the argument (:112–130) or by a byte offset proven under 2 GiB
- S4 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:78 `HIPOptions.num_warps = 4`; warp size 64 (:114); a power of two (:116); `kpack` other than 1 overwritten with 1 on gfx950, with a deprecation warning (:119–123); `parse_options` raises `num_ctas > 1 not supported on gfx950` (:162–163; `supportsMultiCTALaunch` is false there, third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:240) and keeps only `HIPOptions` fields (:181); `gl.barrier` synchronizes a CTA, or with `cluster=True` a cluster when `num_ctas > 1` (python/triton/experimental/gluon/language/_core.py:703); a kernel whose `metadata.shared` exceeds the device's `max_shared_mem` raises `OutOfResources` when its first launch loads it (python/triton/compiler/compiler.py:466)
- S5 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/010eab6660c4e415.json#initial_state.py:9 `_initialize_fresh` zeroes a slot when its `has_initial_state` flag is false, with a 4-warp `BlockedLayout` (:17), launched without `num_warps` (:24)
- S6 aklr/core/contract/locate.py:114 `schema_uuid`; the authored parts are `kernel` and each workload's axes and scalars, sorted (aklr/core/contract/op.py:884); `not at <ops>/...` when the path's tail differs (aklr/core/contract/op.py:959)
- S7 aklr/core/runtime/builder/builder.py:159 an in-memory package `_aklr_triton_<hash16>` (`TritonBuilder`, :193) whose `__path__` is the written sources (:160); the entry is imported as its submodule (:162) and must bind the call keys (:172)
- S9 aklr/core/contract/registry.py:101 a schema whose text is not its `schema_text` is rejected; the writer is `json_file_text` (aklr/core/contract/expr.py:108)
- S10 aklr/core/contract/fastkey/_fast_key_fn.py:284 the storage checks: stored layout, not negated, `data_ptr % 16 == 0` (:40), device; output overlap at :239
- S11 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/7e9c13e0f2d58a3b.json#kernel.py:695 the entry; the launch plan in the docstring (:11–14), the launches at :621–692; `routed_scaling_factor` declared `values=(2.5,)` (aklr/plugins/sglang/models/deepseek_v2/contract.py:158)
- S12 aklr/core/validation/stages/_audit/_rules.py:57 SA0 (:57), SA1 (:65), SA2 (:87), SA3 (:154, its lists at :136–152), SA4 (:192); SA3's I/O names (:142) are rejected wherever a name reads them, device code included (:167)
- S13 aklr/core/validation/stages/_audit/_stage.py:134 device decorators `triton.jit` and `triton.experimental.gluon.jit`, recognized only as a def's single decorator (:392); `perf_edited` against the base (:90)
- S14 aklr/core/validation/stages/_bench/_evaluation/_candidate.py:285 the compile tripwire on `JITFunction._do_compile`, "compiled in the capture" (:523); the first calls, before any capture, use the default inputs and the workload's own scalars (:379–381); inputs rewritten through `.data.copy_` (:502); `capture_error_mode="global"` (:517) and outputs poisoned in the capture (:520, :588–600); read-only inputs byte-compared, `input_mutation` (:548–553); `MAX_FIRST_CALL_BYTES = 16 << 30` (:36), `build_failed` (:229)
- S15 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/fbe6ddab5dac5958.json#kernel.py:95 `_SOFTMAX = gl.constexpr(gl.BlockedLayout(...))` at module level
- S16 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:387 the globals a JIT body may read; `NameError` at :411
- S17 triton-lang/triton@c01b677:python/triton/runtime/jit.py:593 the key `(tuple(specialization), str(options))`, a constexpr entering it as `("constexpr", value)` (:409); caches per device (:738); an in-process miss calls `_do_compile` (:757), which reads the disk cache inside (:877); unknown launch keyword `KeyError` (:713); changed globals `RuntimeError` (:761); `ConstexprFunction.__call__` returns the raw value outside JIT, a constexpr inside (:1153); `Autotuner.run` times every config at a key's first call (python/triton/runtime/autotuner.py:240)
- S18 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:27 `_shape_config`, read by the kernel (:65) and by the host (:291, :294)
- S19 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:96 `_launch_for` returns the launch and its workspace size (:127, :158); `run` (:161); `_reduce_kernel` adds the six split-K partials in a fixed order (:87); n6144's returns the launch alone (aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:116, :196)
- S20 triton-lang/triton@c01b677:python/src/specialize.cc:274 an int equal to 1 becomes a constexpr, otherwise `i32` or `i64` by range (:281), `u64` above (:297), `D` for a multiple of 16 (:283); each tensor's dtype goes through Python's `canonicalize_ptr_dtype` (:329), whose table has no `float4_e2m1fn_x2` or `float8_e8m0fnu` (python/triton/_utils.py:77) and raises `KeyError` (:120); a Python callback per tensor without native tensor specialization (:342–370); bool `u1` (:377), float `fp32` (:385), `None` a constexpr (:522)
- S21 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:236 `get_tensor_specialization`: `D` by alignment (python/triton/backends/compiler.py:89), `S` by `is_within_2gb` on the storage (:211); HIP opts out of native tensor specialization (:140); `parse_attr`: `S` → `tt.pointer_range` 32 (:229), `D` → `tt.divisibility` 16 (python/triton/backends/compiler.py:75)
- S22 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/409cb22b39b3fa00.json#kernel.py:311 "Scale and epsilon are runtime arguments, never specialization keys"
- S23 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/82e9371086a875bd.json#kernel.py:59 `OFFSET: gl.constexpr`; one value per process stated at aklr/plugins/sglang/models/minimax_m2/contract.py:94
- S24 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/a1c4a2129730d1a0.json#kernel.py:702 one arena cut into 64 KiB-aligned typed regions (:723); 256-byte alignment at aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b0651fe84db2f670.json#kernel.py:1619
- S25 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/0101652e6ad7286d.json#kernel.py:365 program 0 of `_prepare_3` initializes `Counts`, `Jobs`, `Total`; the next launch, `_select_3`, takes atomic tickets on them through `_publish_routes` (:449, :507)
- S26 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b94f5777fd5d2d88.json#kernel.py:1026 one `workspace`: router logits, terms and quantized inputs at its start, the bf16 `down` over the same bytes (:1045)
- S27 aklr/core/utils/testing.py:152 `bench_cudagraph`: calls captured behind a flush each and replayed under Kineto; the per-call time sums kernel durations without the flush (:111–149)
- S28 aklr/plugins/sglang/models/moe_blocks.py:16 the MoE routes fall back inside SGLang's piecewise and breakable CUDA graphs
- S29 aklr/plugins/sglang/models/qwen3_next/api.py:113 prefill returns `OUT_OF_CONTRACT` while capturing; `op.fallback_reason(...)` before the projections (:132); decode "Graph capture too" (:144)
- S30 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_decode_config.py:302 `LAUNCH_ATTRS` built once at import
- S31 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/0bb2f7cbe49d1be3.json#kernel.py:939 `q.to(torch.bfloat16)` on the fp8 `q` in `run`
- S32 aklr/core/validation/stages/_bench/_evaluation/_request.py:131 the check steps: first capture and two replays, numeric captures, workload scalars, large magnitude, determinism, post-timing; a scalar takes its default in every input set but the workload's (aklr/core/contract/semantics/_workloads.py:81)
- S33 aklr/core/runtime/dispatch/_dispatcher.py:51 `FallbackReason`; `not_warmed` for a Point not yet run outside a capture (:162); a hit runs `run` and re-raises its exception after logging the `AKLR_DISABLE` hint (:212)
- S34 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/7d7e54597a7e11bd.json#kernel.py:131 "Padding is rewritten every invocation, including graph replay"
- S35 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/f5ba496e04f8e989.json#attention.py:5 live lengths, slots and KV scales read on the GPU; "max_context is capacity, never a live loop bound" (:152); the same schema's kernel.py:19 `@functools.cache` `_unit_scale(device)` returns `torch.ones(1, dtype=torch.float32, device=device)` (:22), passed to both launches (:42, :54–55, :68–69)
- S36 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/84116a96e535f174.json#kernel.py:1373 the worst-case `capacity`; the exit on the device (:441–444)
- S37 aklr/plugins/sglang/hook.py:63 the route runs unless `torch.compiler.is_compiling`; a `FallbackReason` runs the original
- S38 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_layouts.py:34 `AMDMFMALayout` version 4 is gfx950
- S39 aklr/core/validation/command.py:31 `PATH`, `--base`, `--stages`, `--devices`, `--out`; validate asks git for the repository root and `HEAD` (aklr/core/validation/stages/validation_scope.py:40, :43) and takes only paths in it (:55)
- S40 .github/workflows/ci.yaml:296 "triton==3.8.0 is required"; the image at :5; the baseline aiter installed with `AITER_USE_SYSTEM_TRITON=1` (:292), whose setup.py otherwise replaces the installed Triton (ROCm/aiter@dbd8bf5:setup.py:124, :131–139)
- S41 aklr/core/validation/stages/_bench/_report.py:40 `_MEASURED_AS`: the correctness codes, `build_failed`, `candidate_crash`, `candidate_timeout`, `not_faster`, `regressed`; a finding located at a workload with one of them is a demotion (:129–137), and a schema with no qualified workload gets `no_qualifying_workload` (:138–151); bench locates every finding of a batch, a failed build included, at each of its workloads (aklr/core/validation/stages/_bench/_stage.py:343–348) and gives a candidate 900 s plus 300 s per workload (:49–50, :283); the run concludes `passed`, exit 0, unless a stage fails or aborts (aklr/core/validation/pipeline.py:155–164)
- S42 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/0a07f9f9342c6cf1.json#kernel.py:1262 `gl.static_assert(NORM_STRIDE == (8 + HEADS) * 128)` (and `CORE_STRIDE == NORM_STRIDE`, :1261) in `_norm_quant_large`, launched only when `m > 4096` (:1332, :1449) with `NORM_STRIDE=normalized.stride(0)` (:1469); the file's Points are 2049 and 4096 tokens; `normalized` is `[n_tokens, n_v_heads, value_dim]` (aklr/plugins/sglang/models/qwen3_5/contract.py:171)
- S43 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/3b99d19659746a36.json#kernel.py:72 `for candidate in gl.static_range(1, BATCH)`; `gl.static_range(1, BATCH)` in 10 more of the 15 Qwen3.5 GDN prefill files, `gl.static_range(BATCH)` in 2 others (aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len8192_n_v_heads16_key_dim128_value_dim128_conv_dim3072/5ead04831eb8048c.json#kernel.py:84); `n_sequences` 1–2047 (aklr/plugins/sglang/models/qwen3_5/contract.py:73)
- S45 aklr/core/contract/op.py:257 a layout variant may name only the contract's tensor inputs (:215), else "layout... names unknown tensors" (:258), and "an output is never a 4-bit dtype" (:214); a tensor a variant does not list is `RowMajor` (aklr/core/contract/spec/_layouts.py:33); `per_group_quant_mxfp8@1.0` (aklr/ops/quant/contract.py:55) outputs E4M3FN `quantized` `[n_tokens, hidden_size]` (:38, reused at :68) and uint8 `scales`, "E8M0 exponent bytes at [row, group]" (:70), in its one `row_major` variant (:75); `per_group_quant_fp8@1.0` declares its fp32 scales `[n_groups, n_tokens]` (:46)
- S40 .github/workflows/ci.yaml:296 "triton==3.8.0 is required"; the image at :5; baseline aiter installed with `AITER_USE_SYSTEM_TRITON=1` (:292); ROCm/aiter@dbd8bf5:setup.py:124, :131–139 controls its Triton installer.
