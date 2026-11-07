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
authority). The Evidence section lists related measurements and their recorded results; it is not a claim-by-claim certification of this page, and includes failures and unverified scope.

## Route here when

- You write a new Triton or Gluon kernel for an AKLR schema (a core op under `aklr/ops/`, a plugin op under
  `aklr/plugins/sglang/`), or port one into a schema file, or your op has no contract yet (§3).
- You choose between plain `tl.*` and Gluon, or are about to add a launch.
- `python -m aklr validate` reports `schema_invalid` (`not at ...`), `source_rejected` (SA0–SA4), `build_failed`,
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
`python -m aklr export check` (§3); validate on one GPU, and fix or delete every Point it demotes (§11). Each step is
checked in its section; no case runs the whole procedure end to end.

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
| Software pipelining | `num_stages`: loop scheduling, operands streamed into LDS by async copies. In the n2624 GEMM's 16-row path `num_stages=1` stages the tiles through registers into 16 KiB of LDS; the committed 3 uses 101312 B (about 99 KiB) fed by direct-to-LDS `buffer_load_dwordx4 … lds` (96 in the ISA at m = 2 and 8 with 4 warps, 48 at m = 1 and 16 with 2; the other m were not compiled); `num_stages=1` took 2.2–2.3× the time of 3 (18.33–18.61 against 7.97–8.36 µs at m = 2, 4, 8), while m = 64, whose launch did not change, stayed at 0.995–1.003× in speedup (3 runs; the LDS and ISA counts are from 1 compile) *(field, E3, E4)* | prefetch by hand ([gluon/memory.md](gluon/memory.md) §12); in two runs, 36 configurations across 8 Gluon families × 9 option variants had identical instructions. The four AMD Gluon stages' Python closure did not read `num_stages`, `matrix_instr_nonkdim`, `kpack`, `schedule_hint`, but option parsing normalized gfx950 `kpack=2` to 1. One launched copy kernel had 9 cache entries with identical code, not 9 different instruction sequences *(field, E11)* |
| Buffer instructions | `tl.load` / `tl.store` become `buffer_load` / `buffer_store` where a pass proves the range, from the `S` key (§5) or range analysis (S2, S3) | only where you call `gl.amd.cdna4.buffer_load` / `buffer_store`; `gl.load` stays `global_load` ([gluon/memory.md](gluon/memory.md) §2, §3). At GDN decode Point `299133886d479017` (2 tokens, 4 slots), fixed buffer `.cs` and pointer `.cs` arms passed the full contract and 18 paired repeat checks over 3 seeds, 3 slot patterns and 2 output poisons (36 calls, one run); a later cold-device timing run found only 4/9 seed/slot cases equivalent within ±1%, 3/9 pointer cases slower beyond 1%, and 2/9 unresolved at that boundary; this is not a universal same-speed result *(field, E6, E16)* For GDN 299133886d479017 at 2 tokens/4 slots, the original four-run-per-arm Triton 3.8 graph reports gave buffer .cs 8.79 µs and pointer .cs 8.81 µs, calculated as the mean of each run’s median call duration. This descriptive near-equality is not a statistical equivalence guarantee or a general buffer/pointer rule; the later cold-device matrix is separate evidence. |
| LDS | implicit | `gl.allocate_shared_memory`, `gl.amd.cdna4.async_copy.buffer_load_to_shared`, `commit_group` / `wait_group` (with two groups of two copies in flight, `wait_group(1)` becomes `vmcnt(2)`), swizzled and padded layouts |
| Matrix core | `tl.dot`, `tl.dot_scaled` | `gl.amd.cdna4.mfma` (`v_mfma_f32_16x16x32_bf16` in §10's skeleton), `mfma_scaled` with e8m0 scales (a 16 × 16 × 128 e4m3 tile: one `v_mfma_scale_f32_16x16x128_f8f6f4` with the scales in registers; `None` passes a register set to 0x7f, scale 1); `gl.dot_fma` is a vector-ALU dot product off the matrix core, on a `BlockedLayout` accumulator ([gluon/mfma.md](gluon/mfma.md) §1) |
| Lane-level code | `inline_asm_elementwise`, not knowing which lane holds which element | the same call, each element's lane fixed by the layout: `v_readlane_b32` of lane 5 returned x[5] with one element per lane, x[10] and x[11] with two (MI355X); shuffles and ballots: [gluon/wave_ops.md](gluon/wave_ops.md) |

- **`num_warps` is part of a Gluon kernel.** The module takes `ttg.num-warps` from the launch, and the verifier rejects
  a layout with another warp count ("Layout has 4 warps per CTA, but the context requires 8 warps per CTA") or with
  other than 64 lanes (S1). A launch without `num_warps` gets 4 (S4), which hard-coded 4-warp layouts rely on (S5).

The committed plain bf16 GEMMs and norms ([gemm/decode_gemm.md](gemm/decode_gemm.md)) are starting points, not a general language-choice rule. Reducing stages slowed n6144 at m=2/4/8/16 by 1.533–1.625× and RMSNorm at 1024/2048/4096 tokens by 1.047–1.120/1.092–1.095/1.051–1.052× (one run, 4 A/B rounds/arm; fixed order may bias the smaller effects) *(field, E12)*; add-RMSNorm was not tested. Plain layouts/LDS buffers are compiler-managed ([gemm.md](gemm.md) §9).
Gluon exposes e8m0 `mfma_scaled`, explicit LDS/async depth and lane-level code (table). Separately, unshuffled E4M3/FP32-block-scale AKLR kernels at m=2/8/16,N=3072,K=1536 took 0.777–0.781/0.755–0.764/0.759–0.768× aiter plain-Triton's flushed graph time (one run, 5 rounds, both within contract bounds) *(field, E12)*. This compares implementations, not language alone; preshuffled MXFP4/e8m0 performance, fused roles and chain-held-layout recommendations remain untested.

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
  `initial_state.py`, ...) imported relatively (`from .adapter import validate_inputs`), the files being submodules of a
  package named after their content hash (S7). No schema imports another's files: each copies its helpers. Each
  `content` holds no control character but `\n`, as all 272 committed sources do, and ends in one newline with no blank
  line before it: the schema parser rejects a trailing blank line (`kernel.py:<line>: blank line at the end`), a tab and
  a carriage return (`kernel.py:<line>: forbidden character …`; same section).
- **`entry_point`.** `kernel.py::run` (139 files), or `main.py::run` with an `adapter.py` that checks the inputs (the
  15 N = 3072 fp8 GEMMs). The entry file needs a module docstring (its content: same section), else `source_rejected`
  (`SA4 kernel.py:0 no module docstring`); 135 files keep a `# Created by OpenAI` comment above it, which SA4 accepts.
- **`requires`.** `{"triton": ">=3.8"}` in 135 files. Triton 3.7.0 lacks `gl.clamp`, `gl.aggregate_replace`,
  `gl.expect_zero`, `gl.amd.slice`, `llvm_fn_attrs` and `gl.amd.cdna4`'s `compute_efficient_padded_shared_layout` and
  `scaled_upcast` (S8); under it AKLR's toolchain check reports `>=3.8` unmet and an empty `requires` met, and none of
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
lists them for the schema at `p`. The Point's logical sizes are derived from the tensors' stored shapes under the contract's layout and packing rules; a packed 4-bit last dimension holds two logical elements per stored element. It must write every output; leaving a row unwritten violates the contract. The specific NaN-poisoned GEMM example in §8 failed, but that is not a guarantee that every validation invocation detects every omitted write; the rest (mutable inputs, no return value, no output kept, every value of a scalar's range) is in the same
README, § Schema fields. MXFP4 and e8m0 tensors may arrive as `uint8` or `torch.float4_e2m1fn_x2` / `torch.float8_e8m0fnu` (the MoE's four uint8 weight/scale calls were served, E10); the latter dtypes are ones which
Triton 3.8.0 cannot type (`KeyError: 'float4_e2m1fn_x2'` at the launch, S20): pass `.view(torch.uint8)`, as the DeepSeek
MoE's entry does (S11).

Dispatch calls `run` only with tensors in the stored layout, not negated, aligned (`data_ptr() % 16 == 0` in the
source), on the table's device, no output byte overlapping an input (S10): on `gemm@1.0` a non-contiguous `row_major`
input, a negated one, one 2 bytes off alignment, a CPU call against a `cuda:0` table and an output aliasing an input
each fell back `out_of_contract` before `run`. So strides can be constexprs folded from the shape, and a scalar whose
`ScalarSpec` in the op's `contract.py` declares one value can be hard-coded, as the DeepSeek MoE's entry hard-codes
`routed_scaling_factor`, declared `(2.5,)` (S11): 2.0/3.5 fell back before `run`, 2.5 was served. Six-contract probes refused changed attrs/non-any_size axes/layout facts before run; GDN any_size slots 4/8 both served, so do not hard-code them (E10). Pass a ranged
scalar at run time: an RMSNorm copy (`cf637bff`) with `epsilon` hard-coded to 1e-5 passed validate, yet called with
`epsilon` 1e-3 on inputs of scale 0.01 was off by up to 224%, the committed file exact *(field, E2)*.

The audit's rules SA0–SA4 are stated in `aklr/ops/README.md` § The kernel section and ABI (their code: S12); a
`source_rejected` finding names the rule, file and line. What they mean for a Triton or Gluon kernel:

- A def whose single decorator is `@triton.jit` is device code (S13 lists `@gluon.jit` beside it): `torch.tensor`,
  `a.add_` and `a.item()` pass the audit in its body and are rejected (SA3, SA2) in a plain def or under a second
  decorator. A `@gluon.constexpr_function` is not device code: called from `run`, it is audited as launch code. Nor is a
  def with `@triton.autotune` or `@triton.heuristics` above `@triton.jit`, where the same three calls are rejected; the
  README asks for a compile function instead. None of the 154 kernels uses either; an autotuner with multiple unpruned configurations normally times them on a key's first cache miss; a single remaining configuration or a disk-cache hit needs no timing (S17), so two processes can serve one Point with different binaries.
- SA3 rejects its I/O names in device code too (S12): a kernel parameter called `input` is rejected at its first use,
  and so is a `print` in a `@gluon.jit` body; `gl.device_print` passes.
- SA2 rejected `x.add_(1)` in `run` and `weight.add_(1)` in an input-named helper; a renamed `buf` helper passed, then bench reported `input_mutation` at `first_capture/0` (3 GPU runs including the old one-bit XOR plants). Bytewise/read-only comparison is source semantics (S14), not a signed-zero/NaN-payload measurement (E9).
- The tested env reads and host clocks passed audit: `os.environ[...]`/`.get`, `os.getenv`, `time.time`, `monotonic`, `monotonic_ns`, `process_time`, `perf_counter_ns`, `datetime.datetime.now`. SA3 rejected `time.perf_counter`, `torch.cuda.Event` and `elapsed_time`; passing audit is not permission or proof of later-stage safety (E9).
- **A `@triton.jit` / `@gluon.jit` body does not read an ordinary unwrapped module-level Python constant under the default compiler policy** (same section): pass it as a `tl.constexpr` /
  `gl.constexpr` argument or compute it in a `@gluon.constexpr_function` (`from triton.experimental import gluon`;
  3.8.0's `gl` has none). Triton also accepts globals made with `gl.constexpr(...)` (five AKLR files keep such globals,
  layouts in three, S15), modules, JIT functions and dtypes; a plain `BLOCK = 64` fails the default compile (S16). In two GPU runs, reassigning a read constexpr global 4→5 or an explicitly allowed plain global 3→4 raised `RuntimeError`; equal-value reassignment and changing an unread global did not. Object mutation/other global types were not tested *(field, E11)*.

## 5. One binary per Point

**Pass the token count as a constexpr.** 122 of the 150 Gluon kernels have a `@gluon.jit` parameter `M`, `ROWS`,
`TOKENS` or `BATCH` declared `gl.constexpr` (a name count, Qwen3-Next's runtime-count files included). Each Point then
specializes its compile configuration on first call; different Points may still share a kernel configuration. Tested 8/16/32/64-row tiles with M covering the tile had zero mask instructions (runtime 3–56), but uncovered masks remained. In one complete M=1–128 scan, `range(M)` had no branch at 1–22 and 1–2 at 23–128, VGPR 17–74 versus runtime 66; constexpr does not guarantee removal of every mask/loop/flag (E13). That
first call must be outside any CUDA-graph capture and must compile every variant the Point's scalars select
(`aklr/ops/README.md` § The kernel section and ABI). A `gl.static_range` over a constexpr count unrolls fully (a 4-load
body: 120 amdgcn instructions at 8 trips, 6513 at 512, 120–129 with `range`): 13 of the 15 Qwen3.5 GDN prefill files
walk their sequences that way (S43), fine at their Points' 1–8 sequences; for the 2047 the contract allows, use `range`.

**Choose the configuration once, where host and kernel both read it.** A `@gluon.constexpr_function` called outside
JIT returns its Python value and inside JIT a constexpr (S17), so one function gives the launch its grid and the kernel
its tile shapes: the Qwen2-MoE router's kernel reads `config: gl.constexpr = _shape_config(M)` and its host sizes the
grid from `_shape_config(rows)` (S18), as the skeleton of §10 does with `_config`.

**A `@functools.cache` compile function** is the pattern `aklr/ops/README.md` § The kernel section and ABI shows: a
module-level def of shapes whose cache holds compiled code, constants and per-shape scratch only. In three CPU
plants passed directly to `check_kernel`, allocation in its own body passed; nested `torch.tensor` and `.item()` were SA3-rejected (E7). The independent audit also passed `torch.empty`, `triton.compile` and `.warmup` in both bodies: none is on an SA deny-list. This observes coverage, not general safety (E9). Eleven of the 154 kernels have one: the two plain-Triton GEMMs cache a launch closure per shape
(`_launch_for`), n2624's with its workspace size (S19); the nine Qwen3.5 gated-attention files the fp32 unit KV scale
both launches read, `_unit_scale(device)` returning `torch.ones(1, ...)` (S35): in `run`, `torch.tensor` is rejected
(SA3) and `torch.ones` allocates and fills a tensor per call. None of the 154 calls `warmup()` or `triton.compile`: binaries live
in Triton's in-process JIT cache, which meets the first-call rule while the key below is a function of the Point.

**What enters Triton 3.8.0's JIT key** (S17): in two single-GPU runs each frontend had 12 specialization/option combinations and 12 binaries. `kernel_key_cache` uses `(tuple(specialization), str(options))`; the binary cache uses its mapped string key. Two-device isolation was not tested *(field, E11)*.

| Argument | Enters the key as | So |
|---|---|---|
| `tl.constexpr` / `gl.constexpr` | its value, which must be hashable (S17) | one binary per value; a list fails at the launch (§10) |
| int | `i32` / `i64` / `u64`; the value 1 becomes a constexpr; a multiple of 16 gets `D` (S20) | Points share a binary only within one class; `do_not_specialize=[...]` on `@gluon.jit` drops the value keys |
| runtime float, bool | `fp32`, `u1` in the tested signatures (S20) | 9 floats (including 0/1/-1/16/1e30) × True/False used one binary per frontend; 15 native float probes including inf/nan had no value specialization (one expanded GPU run), not a test of every float *(field, E11)* |
| `None` | a constexpr (S20) | an absent optional input compiles its own variant |
| tensor (HIP) | its dtype, `D` when `data_ptr() % 16 == 0` and `S` when AMD buffer passes are enabled and its whole storage is at most 2^31 − 1 bytes (S20, S21), from a Python callback run per tensor at every launch, cache hits included; on the argument `D` becomes `tt.divisibility = 16` and `S` `tt.pointer_range = 32`: offset 0 of a 4 KiB buffer got both, offset 8 only `S`, a 2 GiB + 16 KiB tensor only `D`, 3 binaries *(field, E2)*; in two compile runs, switching `S` changed 3 plain kernels only with AMD buffer passes on; 3 Gluon copy/gather kernels were unchanged with passes on/off, not a census of every C++ consumer *(field, E11)* | a misaligned view, or a storage that crosses 2 GiB, compiles another binary |
| launch options | `str(options)`: `num_warps`, `waves_per_eu`, ... as passed, in keyword order (S17) | launch from one call site; an option Gluon ignores (`num_stages`) still splits the key |

**Keep runtime** what may change while the Point stays the same, since a key that changes after the Point's first call
compiles again, possibly inside a capture (§8): runtime floats (S22; tested fp32 signatures, E11); scalar inputs, unless one value per process is guaranteed (the MiniMax EP files take `expert_offset` as
`OFFSET: gl.constexpr`, one value per process by their contract, S23); `any_size` axes (a pool's slot count), on which
dispatch does not key; and the alignment of pointers into a buffer you carve (§6). A runtime int is still keyed on its
class (1, a multiple of 16, i32, i64) and bench warms only the default and the workload's scalars (S14): put an int that
may change within a Point in `do_not_specialize` (dtype classes `i32`, `i64` and, for large non-negative values, `u64` still remain).

## 6. Scratch

- **Allocate inside the call**, on an argument's device, sized from the shapes: `torch.empty` (117 kernels, 686 call
  sites), `torch.empty_like`, `.new_empty`. One allocation can be carved into typed regions at aligned offsets (64 KiB
  or 256 bytes in AKLR's arenas, S24); an offset that is not a multiple of 16 bytes loses `D` and gets another binary
  (§5). A first call over bench's 16 GiB limit (S14) fails `build_failed`, as one allocating 17 GiB did *(field, E2)*.
- **`torch.empty` holds whatever was there**, in an eager call and in every replay. In every call, write before any read
  what a launch reads before it writes: atomic counters and cursors, `-1` job sentinels, padding rows, accumulators
  added into. AKLR's 154 Triton kernels never call `torch.zeros` or `fill_`: an earlier launch of the call writes the
  initial values, as program 0 of the MiniMax MXFP4 MoE's first launch stores 0 into the expert counts and the total and
  -1 into the job slots, on which the next launch takes atomic tickets (S25).
- **State that must start fresh** is cleared by a small launch reading its flag on the device, as GDN prefill's
  `initialize_fresh` zeroes the slot of each sequence without initial state (S5).
- **Nothing carries from one call to the next.** `run` keeps nothing derived from an argument across calls, a compile
  function only compiled code, constants (§5) and per-shape scratch rewritten whole before every read (same README
  section). The audit rejects a `global` statement (SA1), not a module-level dict written in `run`, which the recorded stale-output plant reached bench with: the r1 log reports that
  a `run` keeping its first output in one and copying it into later outputs failed `numeric_mismatch`
  (`first_capture/0`, `numeric/0`, `determinism/0`) and `post_timing_mismatch`
  ([../pitfalls/state_kept_across_calls.md](../pitfalls/state_kept_across_calls.md)).
- **Reuse a region within a call in stream order**, as MoE files overlay short-lived scratch on a later buffer (S26):
  launched on the same stream before the region's next writer, its last reader saw the values from before the
  overwrite in 20 of 20 eager calls and 20 of 20 graph replays over 16.8 M floats *(field, E4)*.

## 7. Launches: what one costs, where a boundary goes

What one more launch costs depends on who times it:

| Where | One more launch costs | Ruler |
|---|---|---|
| A replayed CUDA graph, as SGLang replays decode | 1.82 µs (1 program) to 1.84 µs (256 programs) of replay time per further chained trivial Gluon kernel (one 64-float load, add and store per program), in a bare CUDA graph, not SGLang's; the replay itself added 3.6–3.8 µs; with 8 launches per graph, 2.27 µs per launch with 1 tensor argument and 2.31 µs with 30 *(field, E1)* | CUDA events around graph replays, least squares over 1–64 kernels |
| AKLR's stored speedup | the call's kernel durations as Kineto reads them, flush excluded (S27): a copy and a sum read 10.84 µs in one call against 6.50 + 4.62 µs apart, a 6.5 µs call stayed 6.5 µs beside a 169.5 µs flush, 8 trivial kernels read 32.4 µs against 8 × 3.88 µs, and no dispatch read below about 4.2 µs *(field, E1)* ([../pitfalls/aklr_dispatch_floor.md](../pitfalls/aklr_dispatch_floor.md)) | Kineto in AKLR's bench |
| An eager call | 7.30 µs (1 program) to 7.44 µs (256 programs) of host time per `kernel[grid](*args)` of a one-tensor trivial Gluon kernel in Triton 3.8.0's launcher, the wall time within 0.2% of it, the GPU waiting on the host; 6.25 µs + 1.00 µs per tensor + 0.12 µs per int argument by least squares over 7 argument lists (7.32 µs with 1 tensor, 38.45 µs with 32, 17.12 µs with 8 tensors and 24 ints); HIP's Python callback for the `S` key (`get_tensor_specialization`, S20, S21) runs once per tensor per launch and took 0.58 µs per call timed alone, 57% of that slope ([debugging.md](debugging.md) §8); 8 launches of a 15-tensor kernel took 169 µs *(field, E1)* | host clock around 2000 launches |

Host time lengthens the call only while the GPU finishes kernels faster than Python issues them: those 8 launches took
172 µs with trivial kernels, their host time, and 3166 µs with 396 µs kernels, their sum *(field, E1)*. Only eager calls
pay it — plugin prefill (the GDN prefill routes refuse capture, the MoE routes decline SGLang's piecewise and breakable
graphs, S28, S29) and direct core-op calls — so there keep tensor arguments few; a 4-key option dict built per call cost
0.06 µs more per launch than one built once (Qwen3-Next decode builds its at import, S30) *(field, E1)*.

**Where the committed kernels end a launch.** The historical Kineto trace reported these per-call counts over 242 observed Points (not a current per-Point coverage guarantee): 1 in the fp8 GEMMs and the norms;
2 in the n2624 GEMM at m = 64, split-K `_gemm_kernel` then fixed-order `_reduce_kernel` ([gemm.md](gemm.md) §9); 2–8
in the MoE plugins, 2–3 in DSA, 3 in gated attention, 1–2 in GDN decode, 5–8 in GDN prefill. The stages they separate:
[moe.md](moe.md) §2, [attention.md](attention.md) §2, [gdn.md](gdn.md) §3 and §7.

**Consider a launch where a value crosses workgroups or a phase needs separate initialization/conversion.** `gl.barrier()` synchronizes one workgroup, and gfx950 has no CTA
clusters to widen it (`num_ctas > 1 not supported on gfx950`, S4); no result may depend on a workgroup that may not be
running, since a launch's workgroups need not all be resident: no unbounded spin on a partner, and no giving up and
computing without its data. So in AKLR's Triton kernels a value one workgroup produces and another consumes — a token's
expert logits before its top-k, every split before a merge, every reader of a state before it is overwritten — ends the
launch, and stream order is the synchronization. None tests an atomic's result for the last arrival or waits in a loop
on memory, so none uses an in-launch fold (each split counts in on an atomic counter, the last to arrive merges in a
fixed order: [../gemm/wait_free_fold.md](../gemm/wait_free_fold.md)) or a bounded wait that hands a timed-out share to
the last arrival, the bounded-wait scheme cited here was measured on FlyDSL kernels; the Gluon fold measurements below are separate. All 42 of AKLR's global atomic call sites are relaxed
([numerics.md](numerics.md) §8). The Gluon fold's counters must start at zero: without reset, 1999/1999 later replays were wrong at each of 9 sizes; a prior zero node yielded 2000/2000 bitwise-equal replays with `acq_rel` and `relaxed` (one run). A last-arriver reset worked in 1000/1000 replays in each of 3 older runs without that node. Merge took 0.03–0.65× acq_rel-fold replay time and 0.51–0.92× relaxed at 4/16/64 splits × 4096/16384/65536 outputs, 64 outputs per one-wave program; AKLR ratios 0.12–0.82 at splits≤16/outputs≤16384. Each measured fold included its extra zero node, so this is not an isolated atomic cost *(field, E14)*. Independent/recomputed phases can sometimes stay in one launch:

- independent phases can share a program-id-role grid (DeepSeek router beside input MXFP4 quantization, S11;
  [moe.md](moe.md) §3). Two 131072-element toy phases had equal outputs in N=4 runs: merged/two-launch replay time
  was 0.762–0.854× in 200/800/880, but 9.191/8.471 µs = 1.085× in 870; merging was not always a replay win.
  AKLR device-duration time remained 0.506–0.516× (merged 4.30–4.44 versus 8.49–8.61 µs) *(field, E2)*;
- a small decision may be recomputed instead of published (S11; [moe.md](moe.md) §4): the toy Triton MoE at 2 tokens, top-4 of 32 experts in 40 programs, took 0.843–0.889× replayed time and 0.742–0.772× AKLR time, same output (N=4 runs) *(field, E4)*. Counterexample: synthetic fp8 VALU S1/S2 top-8/256 MoEs at 1, 2, 8, 16 tokens were slower with all-consumer recomputation on both rulers, with bitwise-equal output at the baseline configuration (one r2 run, 16 rotated rounds/Point, not 16 independent runs) *(field, E8)*;
- a split reduction under a `bitwise` contract gets a fixed-order reduce launch, not float atomics (n2624 GEMM, S19).

Compare publishing with recomputation rather than assuming a fixed cost: removing a toy phase launch at 16 K–1 M elements saved 1.150–1.884 µs replayed in 200/800/880, but −0.102/1.587/0.325 µs at 16K/128K/1M in 870 (N=4 runs, all outputs equal). This is not a fixed positive launch saving or a directly tested decision rule/serving graph boundary. At 131072 elements AKLR device-duration saving was 4.17–4.24 µs *(field, E4)*. A separate synthetic sweep of 56 configurations favored top-8/256 recomputation at P=1024 but publishing at P=4096, for 4 or 16 split partials and 2 or 16 tokens. Median L-C was 3.27 µs replayed and 4.47 µs on AKLR; this constant predicted only 49/56 and 50/56 winners (one r2 run, 12 rotated rounds/config). Some identical-arm controls had nonzero CIs; these are configuration-specific observations, not universal costs or crossover thresholds *(field, E8)*.
A torch op in `run` is a launch too: the DSA files that cast `q` with `q.to(torch.bfloat16)` there add a kernel per call (S31).

## 8. CUDA-graph capture

SGLang captures decode, and AKLR's decode routes run inside its graphs; the GDN prefill routes return `out_of_contract`
while a capture runs, "a capture would replay this call's lengths" (S29). Bench captures every check step — a first
capture replayed twice with new values copied into the same buffers, fresh captures per input set, a repeat under
`bitwise`, the post-timing checks (S32) — and fills the outputs with NaN (0xFF bytes for integer and sub-byte dtypes)
before the call (S14). Leaving a float element unwritten violates the contract; in one recorded run, a copy of the fp8 GEMM `083b493a` leaving 8 of its 16 rows
unwritten failed at `first_capture/0`, 24576 = 8 × 3072 values non-finite *(field, E2)*.

| Breaks | Why | Do |
|---|---|---|
| The Point's first call inside a capture | dispatch serves the fallback (`not_warmed`), and the graph keeps it in every replay (S33) | call every Point once eagerly first, as SGLang does (`aklr/ops/README.md` § The kernel section and ABI) |
| A JIT key that changes after the Point's first call (§5) | Triton compiles at a call that may be captured, which AKLR rules out | make every key a function of the Point: a constexpr warmed at 16 and launched at 32 in a capture compiled there (one `_do_compile` call), and bench's tripwire on `JITFunction._do_compile` raises "compiled in the capture"; the changing-key RMSNorm validate produced no `capture_unsafe` finding, only `no_qualifying_workload`, so end-to-end rejection is unproved (S14, E15, §11); a warm `TRITON_CACHE_DIR` does not help, since every in-process miss calls `_do_compile`, which reads the disk cache (S17) |
| `.item()`, `.cpu()`, `.tolist()`, `torch.cuda.synchronize()`; a torch op whose output size depends on data: `torch.nonzero`, `x[mask]`, `torch.argwhere`, `repeat_interleave` without `output_size` | each failed a capture with `capture_error_mode="global"`, as bench captures (S14): no synchronization is allowed while a stream captures, and the torch ops synchronize to size their output (S44); with `output_size`, `repeat_interleave` captured | SA3 rejected the first four and passed all four data-dependent-size ops in the audited plants (3 GPU capture runs total, E9); the separate rt graph check audited only the first four and `nonzero`, not the other three (E15); capture failure is measured, synchronization mechanism is source semantics: compute indices in a kernel into a worst-case buffer (below) |
| A host decision on device data: a grid or a branch from a count the GPU computed | it needs a synchronization, and a replay keeps the captured decision | size the grid from shapes for the worst case and exit early on a count read on the device (below) |
| State written once and assumed afterwards | a replay reruns only the kernels, never `run`'s Python | write counters, sentinels and padding in every call: "Padding is rewritten every invocation, including graph replay" (S34) |
| A Python value that changes per call | frozen into the graph at capture | read it on the GPU: "Live lengths, physical cache slots, and KV scales are always read on the GPU" (S35) |
| Allocation | the same captured `torch.empty` tensor kept its address over 3 replays and held replay 3's value; a different live eager tensor's address does not prove a private pool (E15) | allocate per call as usual (§6) and never depend on what the buffer held |

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
  cannot meet (S42). Validate each added Point; the new check established only schema validity for a 37-row Point and 15 path-backed `_workloads_to_run` requests (one for that Point, `base=None`), not GPU execution, incumbent admission or `build_failed` demotion (E7).
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
   layout: entry, helpers, the host function that plans the launches. Read only the branch your Point takes, and keep
   that branch's constants together: changed alone, one of them broke a committed kernel (same page, §1), key splits 8 →
   4 in a 24-token DSA file failing to build, `gate_fold` 4 → 1 in a 1-token MiniMax file crashing, 4 of 4 runs each
   *(field, E3)*.
2. **Or start from this skeleton**, a `gemm@1.0` kernel for a scenario with `trans_b` and no bias such as
   `gemm_n6144_k2048`: 16-byte loads in a `BlockedLayout`, masked `buffer_load` / `buffer_store`, one MFMA with an
   `AMDMFMALayout` accumulator and `DotOperandLayout` operands, the token count as a constexpr, one configuration
   function for kernel and launch. It compiles with Triton 3.8.0 for gfx950 at M = 2, 16, 32 and 64 without spills,
   and passes SA0–SA4. In one historical GPU run of the check's embedded skeleton (the current block equals the available embedded copy, but the run-time source hash was not retained) at N6144/K2048/M=2,3,4,8,12,16,17,32,33,48,64, one positive and one signed input per M were bitwise stable over 20 eager calls, 20 graph replays and a second process; positive error 0–1 bf16 ulp, signed error 0.960–0.984 of 2^-8|sum|+2^-18 sum|ab|. Export succeeded, but all six validate Points (M2/4/8/16/32/64) were demoted in 3 rounds of that run, speedups 0.373–0.434 initially, no correctness/build/capture findings. Deleting all leaves no valid workload: performance repair is required *(field, E12)*.

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
   read 0 without `other` *(field, E1)* ([gluon/memory.md](gluon/memory.md) §4). The two `convert_layout`s go through 8
   KiB of LDS (loading in the operand layout instead: [gluon/layouts.md](gluon/layouts.md) §6, §9). The launch is a
   start, not a tuned choice: its 96 programs at N=6144 are fewer than 256 CUs. At m=2/4/8/16 it took 2.882–3.158× f806's time; one-step K prefetch improved it 1.695–1.729× (one run, 3 rounds/arm in fixed order; register allocation/scheduling changed too) *(field, E12)*. A separate stream rose from 876–886 to 3945–3995 GB/s with one/eight loads (3 runs, E4); bytes-in-flight limitation remains inference, not an isolated causal measurement ([tuning.md](tuning.md) §3, §4). Grid and prefetch: tuning.md §6, §10; `num_warps`:
   tuning.md §7; `waves_per_eu=2` becomes `"amdgpu-waves-per-eu"="2,2"` and `.amdhsa_next_free_vgpr` 169 for a skeleton
   using 70 VGPRs; the compiler reports `Occupancy: 2`, a resource-limit report rather than measured residency (tuning.md §8). Decode GEMM speed: [gemm/decode_gemm.md](gemm/decode_gemm.md).
3. **Compile for gfx950 without a GPU.** `triton.compile` with an `ASTSource` (`GluonASTSource` for Gluon) and
   `GPUTarget("hip", "gfx950", 64)` builds the launch's instructions, given its constexprs and specialization (`"DS"`
   for 1 MiB tensors at aligned addresses and `"D"` for n = 1024 matched; a pointer 4 bytes off alignment in a 3 GiB
   range with n = 1000 gets neither). Signatures name the dtype the launch sees: `*bf16`, `*fp16`, `*fp32`, `*fp8e4nv`
   (`float8_e4m3fn`), `*fp8e5` (`float8_e5m2`), `*u8` (also the uint8 views of MXFP4 and e8m0), `*i32`, `*i64`; an int
   is `i32` or `i64` by signed range, or `u64` above the signed-64 range (a constexpr when it is 1); a float is `fp32`, a bool `u1` (S20). It needs Triton and the
   kernel's imports (torch in 152 of the 154 AKLR files), not a GPU. A kernel whose `kernel.py` imports a sibling
   relatively (DSA decode `1148d1c8e30397f7`) fails `from kernel import ...` ("attempted relative import with no known
   parent package"): write its sources into `kpkg/`, put its parent on `sys.path` and `from kpkg.kernel import ...`, a
   package as AKLR makes one (S7). Matching launch signatures/options produced identical amdgcn in 12 independently forced offline recompiles (one run, E11); older tests cover BLOCK constexpr/no-attribute examples, not arbitrary kernels. This compiles the skeleton with Triton 3.8.0 for gfx950:

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

   LDS is `k.metadata.shared` (8192 here) while the amdgcn's `.amdhsa_group_segment_fixed_size` reads 0; actual packed launch metadata also supplied 8192 at all 11 tested M values (one run, E12). A Gluon
   kernel's `k.asm` holds `source` (the frontend's IR, before `gluon_to_ttgir`), `ttgir`, `llir`, `amdgcn`, `hsaco`
   ([debugging.md](debugging.md) §1–§2).
4. **Know what only a launch catches.**
   - A constexpr must be hashable: `INSTR=[16, 16, 32]` at launch failed with `unhashable type: 'list'` where
     `triton.compile` worked *(field, E2)*, since the launcher hashes each constexpr into its key (S17): pass a tuple.
   - A misspelled launch option raises `KeyError: Keyword argument ... was specified but unrecognised` when the launch
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
  `.github/README.md` § Install writes it, with `AITER_USE_SYSTEM_TRITON=1` to keep an installed Triton. Otherwise, aiter's setup.py attempts its Triton installer unless torch is below 2.9.1; installer failures may also be skipped (S40). A baseline that reaches aiter can move with the aiter build: the n6144 bf16 GEMM's baseline read
  7.02 µs on the image's aiter and 7.95 µs on the pinned one at m = 8, 4.5% apart the other way at m = 16, the candidate
  within 1.2% at both *(field, E3)* ([tuning.md](tuning.md) §1). A local container may hold another Triton (3.7.0 has
  been seen): check `triton.__version__`. On 3.7.0 the launch's keyword check rejects `llvm_fn_attrs` (`KeyError`), and
  `schedule_hint="memory-bound-attention"` (AKLR's n6144 GEMM passes it) adds `"amdgpu-sched-strategy"="iterative-ilp"`,
  which 3.8.0 ignores. Matched offline builds of one bf16 `tl.dot` probe (64×64×64 tile, 4 warps, 2 stages, that hint) produced 274/405 static instructions and 16384/33968 B LDS on 3.7.0/3.8.0; complete ISA, HSACO and hashes are saved in local:/tmp/kbmig/checks_study/kb_last_gap_stream-versions_1107.json. This does not authenticate the older 457/432 counts, whose original artifacts were not retained (S8, S2).

Correctness: bench compares every output with the contract's reference under the schema's `compare` block —
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
| `numeric_mismatch`, `nondeterministic`, `post_timing_mismatch` | the math ([numerics.md](numerics.md)), also an arrival-order sum moved past its bound (`numeric_mismatch`, not `nondeterministic`); `post_timing_mismatch`: state kept across calls; the r1 math/reduction-order/state-kept plants reported `numeric_mismatch` or `post_timing_mismatch`, not `nondeterministic` | 6, 8 |
| `not_faster`, `regressed` | the speed ([tuning.md](tuning.md)) | — |
| `no_qualifying_workload` | no Point passed its checks and the entry threshold: the demoted list gives each Point's cause | 11 |
| `no_matching_toolchain`, `no_matching_device` | the installed Triton misses `requires`; the GPU is not in `targets` | 3 |

**A/B a variant** as a copy with the one change, at its own uuid path. Against the committed kernel of its Point,
validate the copy with `--base origin/main`: the two are timed in the same rounds. Alternate two uncommitted variants on
one idle GPU (A, B, A, B, at least four runs per arm) and compare the candidate times in each `report.json`: a file and
its comment-only copy read mean times 0.987–1.005 of each other on four Points. The same file validated three times
moved its baseline speedup (exp of `log_ratio_mean`) by up to 2.8%: fp8 GEMM `083b493a` m = 16 2.4% (2.8% for its copy
under `--base`), GDN decode `29913388` 1.7%, MoE `d3bbbce5`, router `c3889614` 0.9% *(field, E5)*; [tuning.md](tuning.md) §2.

## Evidence

**Unverified scope (未验证):** `d_buffer_pointer_correctness_control.rate` was unmeasured; E16 now supplies the finite cold-device comparison, not other Points/configurations or AITER; `d_source_contract_and_point_census.09.build`: added-Point GPU execution/incumbent admission/demotion; `s_audit_rule_coverage`: universal cached-function safety, bytewise signed-zero/NaN-payload distinction, later-stage clock detection; `s_jit_key_and_buffer_range`: two-device isolation, arbitrary global mutation/float values/kernels; `s_plain_triton_vs_gluon_pipelining`: add-RMSNorm, universal language choice, preshuffled MXFP4/e8m0 performance, fused-role/chain-layout rules, isolated bytes-in-flight cause; `s_decode_moe_fusions_recompute.02`: universal crossover and production serving-gap decomposition. These untested scopes are not positive recommendations.

AKLR paths are at main `43e112e`; a committed kernel is cited as `<schema path>#<source file>:<line>`. Triton paths
are `triton-lang/triton@c01b677` (tag v3.8.0), aiter's `ROCm/aiter@dbd8bf5` (CI's baseline). `local:`: S8 the Triton
3.7.0 wheel, S44 the study host's torch. E1–E5 retain historical case-log counts; E6 onward count complete runs separately from calls/checks/rotated timing rounds.
- E15 [2026-11, gfx950, Triton 3.8.0, N=1 cleanup run, rt_graph_capture_compile tripwire/capture probes] all 22 checks executed, 21 PASS, .01 FAIL, 0 SKIP; three case logs are not three successful runs/check. 100 .11/.20/.21 had case exceptions; 150/870 are valid for those checks. .05: 16 processes each timed 2 configs then 0; both equal-cost DUMMY configs won (870: 8/16 picked 1), supporting only possible process disagreement. .12: both fresh processes call _do_compile once despite shared disk cache; 870 first launches 0.5631/0.001563 s. .17: same tensor address/replay write, not independent private-pool proof. .01: tripwire raises, but changing-key variant has zero capture_unsafe findings. Retain all prior s_/d_ verdicts and counts; no cleanup timing causality asserted (case triton/checks/rt_graph_capture_compile.py; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_graph_capture_compile.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_graph_capture_compile_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_rt_graph_capture_compile_cleanup.log; local:/tmp/kbmig/checks_study/verdicts/rt_graph_capture_compile.audit2.json).
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 new complete run; audit checks CPU, mutation/capture checks GPU] s_audit_rule_coverage .01–.06 confirms single device decorators skip launch checks; input-named writes fail SA2, renamed helpers pass audit then fail input_mutation on 14 RMSNorm workloads; nine host clock/env probes pass, perf_counter/Event/elapsed_time fail; own cached body passes all five calls, nested tensor/item fail but compile/warmup/empty pass; global/nonlocal fail while dict writes pass; 8 listed ops fail global capture, two controls capture. Earlier mutation/capture logs supply 2 additional GPU runs; audit coverage does not prove universal safety (case triton/checks/s_audit_rule_coverage.py; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_s_audit_rule_coverage.log:1–18; old cases/logs E2, S14, S44).
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 new complete run, dispatch stand-ins across 6 contracts] s_dispatch_contract_and_requires .01: 633 storage probes: tested violations refused before run, aligned controls served; 85 non-overlap controls served; .02: 87 dtype/attr/axis/layout/scalar probes, unlisted changes refused, but uint8 MoE w1/w2/scales served. GDN any_size slots 4/8 both served in the old dispatch check; do not hard-code any_size or assume a unique torch storage dtype. Cross-GPU device substitution not tested (case triton/checks/s_dispatch_contract_and_requires.py; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_s_dispatch_contract_and_requires.log:773–776; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_aklr_dispatch.log).
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 complete new runs, runtime JIT probes and forced offline compiles] s_jit_key_and_buffer_range .01–.06: D/S thresholds, float/bool signatures, read-global reassignment, 12 cache variants/frontend, 9 same-code option variants in a launched copy, 36 configurations/8 families × 9 variants independently compiled per run. Expanded r1 has 18 float/bool launches/frontend, 15 native float probes, 12 forced offline text-equality comparisons; 810 offline equality alone is not independent recompilation. Three plain S probes change only with buffer passes on; three Gluon probes unchanged. CASE FAIL due to another claim's .07, not failure of these checks (case triton/checks/s_jit_key_and_buffer_range.py; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/810_s_jit_key_and_buffer_range.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_jit_key_and_buffer_range_r1.log:1–18).
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 complete run, s_plain_triton_vs_gluon_pipelining .01–.06] stages tests 4 A/B rounds/arm; skeleton-prefetch tests 3 rounds/arm in fixed order; unshuffled E4M3/FP32-scale comparison 5 same-process flushed graph rounds. Skeleton 11 M values × one positive/one signed input, 20 eager/replay repeats and second-process equality; actual launch shared=8192/static field=0. Export succeeded, but validate demoted all 6 Points in each of 3 rounds, so CASE FAIL (.06) is a real performance counterexample, not successful delivery. These rounds/repeats are not independent full runs (case triton/checks/s_plain_triton_vs_gluon_pipelining.py; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/810_s_plain_triton_vs_gluon_pipelining.log:1–22).
- E13 [2026-11, CPU gfx950 compile target, Triton 3.8.0, N=1 complete corrected compile run, no GPU timing] s_constexpr_folding_and_load_width .01: 108 row builds, covered masks disappear but full M1–128 range scan retains branches at 23–128 and reaches 74 VGPR versus runtime 66; 11 twin Points do not all simplify. CASE FAIL (5/6 checks passed) is a counterexample, not a test defect (case triton/checks/s_constexpr_folding_and_load_width.py; local:/tmp/kbmig/checks_study/s_constexpr_folding_and_load_width.audit_fixed_local.log:1–7).
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 new complete run, Gluon fold .04] 4/16/64 splits × 4096/16384/65536 fp32 outputs, 64/one-wave program on 4–8 XCDs: zeroed acq_rel/relaxed both 2000/2000 right per size, never-reset 1999/1999 later replays wrong; merge faster with each fold including its zero node. Old num_atomics_semantics .03 self-reset counter 1000/1000 right in 3 runs disproves necessity of a separate zero node; E4's explicit-zero performance remains narrow. Overall new CASE FAIL from .01, not .04 (case triton/checks/s_inlaunch_order_vs_launch_end.py; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_s_inlaunch_order_vs_launch_end.log:1–8; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_num_atomics_semantics.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_num_atomics_semantics_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_num_atomics_semantics_r2.log).
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 complete r1 run, GDN decode 299133886d479017: n_tokens=2, n_slots=4, VH16/K128/V128, channels3072; fixed recurrence BK128/BV16/ROW_GROUP16, LARGE_STATE_POOL=False, nw2/waves1, norm BLOCK128/HISTORY_BLOCK128/nw1] seeds 39107/51749/82021 × slots (3,0)/(1,-1)/(-1,2) × buffer .cs/pointer .cs × NaN/17 poisons: 36 mod.run calls, 144 full output comparisons without contract violation, 18 bitwise repeat pairs and all guards passed. CPU DS compiles of four kernels: recurrence buffer/global load-store 4/4 + 1/0 versus 0/0 + 5/4, both 1024 B LDS; not runtime ISA. Rate SKIP, no latency or equivalence evidence (case triton/checks/d_buffer_pointer_correctness_control.py; checks .contract, .comparator_controls, .oracle_dtype_controls, .compile, .accuracy, .rate; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_d_buffer_pointer_correctness_control_r1.log:1–8).
- E16 [2026-11, gfx950, Triton 3.8.0, N=1 complete cold-device comparison, 3 seeds × 3 slot patterns] Original GDN 2991 buffer .cs versus pointer .cs, 2 tokens/4 slots; each call restores state then reads a 1 GiB flush, neither restore nor flush counted. Sum of the two target kernel device durations: 32 balanced A/B profiler rounds per input, 10 replays of 20 calls per measurement, 8 A/A pairs. Simultaneous intervals support ±1% equivalence in 4/9, pointer slowdown beyond 1% in 3/9 (B/A 1.02797, 1.01344, 1.01179), 2/9 cross the boundary; all A/A intervals contain 1. Post-measurement contract checks passed. These are not historical 8.79→8.81 µs or complete interface latency (local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/590_u_G05_device_ruler_r1.log; local:/tmp/kbmig/checks_study/study/G05_cold_device_judgment.json).
- E7 [2026-11, CPU source/AKLR checks only, N=1 complete run, no GPU measurement] .09.audit: own cached-plan allocation and nested-unlisted plants passed direct check_kernel; nested torch.tensor/item failed SA3. .09.schema/.09.queue: valid added n_tokens=37 Point among 15 workloads, 15 actual path-backed requests, one added request, base=None; .09.build SKIP. Overall 25 checks passed and 8 skipped, not 25 independent runs (case triton/checks/d_source_contract_and_point_census.py; local:/tmp/kbmig/checks_study/local_logs/d_source_contract_and_point_census.log:853–856,963).
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 complete r2 run, synthetic fp8 VALU MoEs and decision/consumer sweep, CUDA-graph replay and AKLR timer] .01: S1/S2 × tokens 1/2/8/16, 16 rotated paired rounds/Point, all-consumer recomputation slower on both rulers, baseline output bitwise equal. .02: 7 decisions × tokens 2/16 × P40/256/1024/4096 = 56 configs, 12 rotated rounds/config (requested rounds_used=[9] is not N), R/L bitwise equal; top-8/256 with 4/16 partials favored R at P1024 and L at P4096. Median L-C 3.2696/4.4679 µs, constant-cost predictions 49/56 and 50/56, below 90%; identical-arm CI excluded zero at 8 replay and 3 AKLR configs (uncorrected multiple comparisons), so near-zero effects and precise mechanisms are not established. .01/.02 FAIL; CASE FAIL (2/5 checks passed), not a general recommendation (case triton/checks/s_decode_moe_fusions_recompute.py; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_decode_moe_fusions_recompute_r2.log:4–20,23–80).

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, trivial Gluon kernels (one 64-float load, add and store per program): replays of a bare CUDA graph of 1–64 chained kernels timed with CUDA events (least-squares slope), eager launches (2000 × 5 rounds, host clock), AKLR's `bench_cudagraph` (Kineto); and probe kernels written for the case reading past a view with `gl.amd.cdna4.buffer_load`] replay 1.815 µs per further kernel at grid 1 and 1.844 µs at grid 256, intercept 3.643 and 3.754 µs; eager host 7.302 µs (grid 1) and 7.443 µs (grid 256), wall 7.311 and 7.452 µs; host time 6.249 µs + 1.004 µs per tensor + 0.1186 µs per int over 7.32 (1 tensor), 10.3, 14.23, 22.19, 38.45 µs (32 tensors), 15.24 and 17.12 µs (8 tensors with 8 or 24 ints); `get_tensor_specialization` 0.576 µs per call, 8 calls per 8-tensor launch; 8 launches × 15 tensors 168.8 µs of host time; 8 launches of 395.6 µs kernels 3166 µs wall against 3165 µs of kernels, of trivial kernels 171.9 µs wall against 171.7 µs host; graph replay 2.268 / 2.313 µs per launch with 1 / 30 tensors; option dict per call 7.744 against 7.686 µs; AKLR's bench: copy 6.498 + sum 4.62 µs apart, 10.84 µs together, flush 169.5 µs, 8 trivial kernels 32.42 µs against 3.879 µs each, median of 45 one-kernel readings 4.23 µs; buffer_load probes on a 64-element view of a larger allocation: unmasked offsets 64–79 returned 64.0–79.0, the next elements of the allocation; masked lanes without `other` returned 0; all 85 buffer ops of 27 compiled probe kernels take num_records `0x7ffffffe` (case triton/checks/bench_launch_cost.py (bench_launch_cost.03–.12); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_launch_cost.log; case triton/checks/rt_buffer_oob_masks.py (rt_buffer_oob_masks.05–.07); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_buffer_oob_masks.log)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, launches of small Gluon probe kernels (binaries counted in the JIT cache), and python -m aklr validate --stages audit,bench of planted copies on one GPU with direct calls against the contract's reference; and two toy Triton elementwise phases of 131072 elements as two launches and as one grid split by program id, timed as a CUDA graph replayed back to back (wall time) and by AKLR's `bench_cudagraph`, 3 runs, the check's bar (one grid at most 0.85× of two launches under both rulers) met in 2] `INSTR=[16, 16, 32]` as a constexpr failed at the launch with `unhashable type: 'list'` while `triton.compile` of the same kernel succeeded, a tuple launched; scales 0.5, 1e-6, 3.0, 2^-20 with flags True / False: 1 binary; a 1 KiB view at offset 0 of a 4 KiB buffer `tt.divisibility = 16` and `tt.pointer_range = 32`, at offset 8 only the range, a 2 GiB + 16 KiB tensor only the divisibility, 3 binaries, `is_within_2gb` called twice per launch of a two-tensor kernel, cache hits included; an RMSNorm copy of cf637bff0784c454 with `epsilon` hard-coded to 1e-5 passed audit and bench with no finding, and called with `epsilon` 1e-3 on 64 × 6144 inputs scaled by 0.01 had a max relative error of 2.235, the committed file 0; a copy of gemm 083b493a (m = 16) storing 8 of its 16 rows: `numeric_mismatch` at first_capture/0 with 24576 = 8 × 3072 non-finite mismatches, demoted, the committed file no finding, `_poison_outputs` filling uint8, int32 and fp4 outputs with 0xFF bytes and fp32, bf16 with NaN; `torch.empty(17 << 30)` in `run`: `build_failed` at the Point, "peak 18259902464 or resident 0 bytes: too many", file-level `no_qualifying_workload`; the two phases as one grid 6.33 / 4.30, 5.66 / 4.38, 5.66 / 4.33 µs (replayed / AKLR's timer) against two launches 7.42 / 8.49, 7.39 / 8.57, 7.43 / 8.50 µs, replay ratio 0.854, 0.766, 0.762, outputs equal in every run (case triton/checks/rt_jit_specialization_key.py (rt_jit_specialization_key.04, .07, .09); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_jit_specialization_key_r1.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_jit_specialization_key.log; case triton/checks/rt_aklr_validate_report.py (rt_aklr_validate_report.01); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_aklr_validate_report_r2.log; case triton/checks/rt_per_call_scratch_poison.py (rt_per_call_scratch_poison.03, .07); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_per_call_scratch_poison_r1.log; case triton/checks/rt_aklr_failure_findings.py (rt_aklr_failure_findings.02); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_aklr_failure_findings_r1.log; case triton/checks/bench_launch_design_rules.py (bench_launch_design_rules.06); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_launch_design_rules.log (0.854, FAIL against the bar), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_launch_design_rules_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_launch_design_rules_rep3.log) Cleanup addition (phases N=4; other cases unchanged): 870 .06 FAIL, outputs equal, merged/two replay 9.1913/8.4705 us=1.085, AKLR 4.4426/8.6096 us=0.516. Original three replay wins remain valid; 0.85 criterion passes only 800/880, not a semantic bug. (local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_bench_launch_design_rules_cleanup.log).
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, python -m aklr validate --stages audit,bench, one GPU, a committed file (A) and a copy with one change (B) alternated, 4 rounds (3 for the aiter builds), median candidate time; and the Triton compile of gemm_n2624_k6144 b88c8d5f's m ≤ 16 launch at m = 1, 2, 8, 16] the compile: `num_stages=1` 16384 B of LDS, the committed 3 101312 B, `buffer_load … lds` counted 48 / 96 / 96 / 48 at m = 1 / 2 / 8 / 16 (2 warps at m = 1 and 16, 4 at m = 2 and 8; the case's 96 bar failed at the 2-warp launches); Qwen3.5 GDN decode 29913388's historical buffer/pointer timing is not evidence of speed equivalence; the final check supplies only the fixed-Point accuracy observations in E6; gemm_n6144_k2048 f806ad2b on the image's aiter (e7d2453f2) against the pinned dbd8bf5: baseline 7.02 against 7.95 µs at m = 8 (13.3%), 7.50 against 7.16 µs at m = 16 (4.5%), 2.9%, 3.4%, 1.0%, 3.9% at m = 2, 4, 32, 64, candidate within 0.1–1.7%; DSA 2ee5 (24 tokens) splits 8 → 4: `build_failed` (CompilationError) 4/4, MiniMax MXFP4 6557 (1 token) `gate_fold` 4 → 1: `candidate_crash` 4/4, both committed files passed 4/4 (case triton/checks/lower_plain_triton_passes.py (lower_plain_triton_passes.09, .10); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_plain_triton_passes.log; case triton/checks/ab_gdn_state_access.py (ab_gdn_state_access.01); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_gdn_state_access.log; case triton/checks/rt_tooling_on_image.py (rt_tooling_on_image.02); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_tooling_on_image.log; case triton/checks/ab_coupled_constants_break.py (ab_coupled_constants_break.03); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/400_ab_coupled_constants_break.log)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, Gluon kernels in CUDA graphs: a split reduction of 16 splits × 16384 fp32 outputs, 200 replays per arm, compared bitwise with a partial + merge launch pair; one stream writing a 16,777,216-float region, reading it last, then overwriting it, 20 eager calls and 20 replays; python -m aklr validate --stages audit,bench on one GPU, a committed file (A) and a copy with one change (B) alternated 4 rounds per arm, median candidate time; a Gluon stream reading a 1 GiB bf16 tensor, timed with CUDA events; small Triton kernels written for the case, timed as a CUDA graph replayed back to back (wall time) and by AKLR's `bench_cudagraph`] fold with its counter zeroed earlier in the same graph: 200/200 equal with `acq_rel` and 200/200 with `relaxed`; without the zeroing node wrong from the second replay on, 199/199; replay time 76.73 µs (zero + fold) against 7.6 µs (partial + merge); the last reader saw the values from before the overwrite in 20/20 eager calls and 20/20 replays; gemm_n2624_k6144 b88c8d5f, `num_stages` 3 → 1 on the m ≤ 16 launch, A against B at m = 2 / 4 / 8: 7.97 / 8.11 / 8.35 against 18.34 / 18.43 / 18.49 µs, 8.03 / 8.13 / 8.36 against 18.45 / 18.53 / 18.61 µs, 8.00 / 8.08 / 8.34 against 18.33 / 18.43 / 18.49 µs, speedup B/A 0.433–0.453 (2.21–2.30× the time), control m = 64 1.003, 1.000, 0.9948, both arms accurate in every round; stream of 96 programs × 4 warps, one 16-byte load per lane in flight (0.375 MB) against eight (3 MB): 885.2 against 3995, 876 against 3945, 885.9 against 3985 GB/s; toy decode MoE (32 experts, top-4, hidden 256, intermediate 128, 2 tokens) recomputing top-k in its 40 consumer programs against a selection launch, same output (rel. L2 0): 9.67 / 9.69 against 11.41 / 12.76, 9.89 / 9.52 against 11.13 / 12.75, 9.67 / 9.78 against 11.47 / 12.68 µs (replayed / AKLR's timer); merging two elementwise phases saved 1.15 / 1.87 / 1.79, 1.61 / 1.88 / 1.80, 1.84 / 1.74 / 1.76 µs replayed at 16384 / 131072 / 1048576 elements, outputs equal, and 4.19, 4.19, 4.17 µs per removed launch on AKLR's timer at 131072 elements (.06's readings, E2) (case triton/checks/num_atomics_semantics.py (num_atomics_semantics.02); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_num_atomics_semantics_r2.log; case triton/checks/rt_per_call_scratch_poison.py (rt_per_call_scratch_poison.06); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_per_call_scratch_poison_r2.log; case triton/checks/ab_misc_rules.py (ab_misc_rules.03); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/400_ab_misc_rules.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_ab_misc_rules_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_ab_misc_rules_rep3.log; case triton/checks/bench_bytes_in_flight.py (bench_bytes_in_flight.04); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_bytes_in_flight.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_bytes_in_flight_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_bytes_in_flight_rep3.log; case triton/checks/bench_launch_design_rules.py (bench_launch_design_rules.07, .08); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_launch_design_rules.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_launch_design_rules_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_launch_design_rules_rep3.log) Cleanup addition (launch-design N=4; other cases unchanged): 870 .07 PASS, rel L2=0, recompute/selection replay 9.8745/11.6206=0.849747, AKLR 9.7931/13.1934=0.742275. .08 FAIL, equal outputs, saving at 16K/128K/1M −0.101600/1.587210/0.325200 us; .06 AKLR saving 4.166977 us. Three original .08 positive runs remain valid; no fixed positive replay launch cost. (local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_bench_launch_design_rules_cleanup.log).
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, python -m aklr validate --stages audit,bench, one idle GPU, each file alone: the committed file (A) alternated with a copy whose first line carries a comment (A', a new uuid), 4 runs per arm, and the first 3 runs of A; then three `validate --base HEAD` runs of A' of 083b493a; ratio = exp(log_ratio_mean) of the baseline entry, time = median candidate time] ratio spread (max - min) / min over the first 3 runs of A 2.385% (fp8 GEMM n3072 083b493a, m=16), 0.911% (Qwen2-MoE router c3889614, n_tokens=256), 1.731% (Qwen3.5 GDN decode 29913388, n_tokens=2: 2.4222, 2.4105, 2.4522), 0.942% (DeepSeek-V2 MXFP4 MoE d3bbbce5, n_tokens=2); 083b493a's A' under `--base` 1.4784, 1.4386, 1.4749 (2.77%); mean candidate time A' over A 0.987, 1.002, 1.005, 0.997 in the order of the four (case triton/checks/ab_noise_a_vs_a.py (ab_noise_a_vs_a.02, .04, .11; .03 FAIL: it expected at most 1.5% for the GEMM); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_noise_a_vs_a.log)
- S1 triton-lang/triton@c01b677:python/triton/experimental/gluon/_runtime.py:45 `GluonJITFunction(JITFunction)` swaps only the AST source; the module takes `ttg.num-warps` from the options (:32) and `ttg.threads-per-warp` from the warp size (:34); the verifier requires a layout's warp count to equal the module's (lib/Dialect/TritonGPU/IR/Dialect.cpp:3877, lanes :3863)
- S2 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:585 `add_stages`: Triton `make_ttir` + `make_ttgir`, Gluon `gluon_to_ttgir` (:327–349, no pipeliner, no buffer-op pass); `make_ttgir` coalesces (:269), reads `matrix_instr_nonkdim` and `kpack` (:273) and `num_stages` (:288–289), runs the buffer-op passes under `use_buffer_ops` (:303); `schedule_hint` "right now no effect" (:104); `AMDGCN_USE_BUFFER_OPS` defaults to on (python/triton/knobs.py:517)
- S3 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/ConvertToBufferOps.cpp:135 `canUseBufferOps`: a splatted base plus 32- or 64-bit offsets, safe by `tt.pointer_range` 32 on the argument (:112–130) or by a byte offset proven under 2 GiB
- S4 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:78 `HIPOptions.num_warps = 4`; warp size 64 (:114); a power of two (:116); `kpack` other than 1 overwritten with 1 on gfx950, with a deprecation warning (:119–123); `parse_options` raises `num_ctas > 1 not supported on gfx950` (:162–163; `supportsMultiCTALaunch` is false there, third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:240) and keeps only `HIPOptions` fields (:181); `gl.barrier` synchronizes a CTA, or with `cluster=True` a cluster when `num_ctas > 1` (python/triton/experimental/gluon/language/_core.py:703); a kernel whose `metadata.shared` exceeds the device's `max_shared_mem` raises `OutOfResources` when its first launch loads it (python/triton/compiler/compiler.py:466)
- S5 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/010eab6660c4e415.json#initial_state.py:9 `_initialize_fresh` zeroes a slot when its `has_initial_state` flag is false, with a 4-warp `BlockedLayout` (:17), launched without `num_warps` (:24)
- S6 aklr/core/contract/locate.py:114 `schema_uuid`; the authored parts are `kernel` and each workload's axes and scalars, sorted (aklr/core/contract/op.py:884); `not at <ops>/...` when the path's tail differs (aklr/core/contract/op.py:959)
- S7 aklr/core/runtime/builder/builder.py:159 an in-memory package `_aklr_triton_<hash16>` (`TritonBuilder`, :193) whose `__path__` is the written sources (:160); the entry is imported as its submodule (:162) and must bind the call keys (:172)
- S8 local:/opt/venv/lib/python3.12/site-packages/triton/runtime/jit.py:697 Triton 3.7.0 (3.7.0+gitb4e20bbe.rocm7.14.0a20260608): a launch keyword that is not an option raises `KeyError`; its `HIPOptions` has no `llvm_fn_attrs`; its Gluon has no `clamp`, `aggregate_replace`, `expect_zero`, `amd.slice`, `amd.cdna4.compute_efficient_padded_shared_layout` or `amd.cdna4.scaled_upcast`, all present in 3.8.0; `schedule_hint` `memory-bound-attention` sets `amdgpu-sched-strategy` `iterative-ilp` (backends/amd/compiler.py:393)
- S9 aklr/core/contract/registry.py:101 a schema whose text is not its `schema_text()` is rejected; the writer is `json_file_text` (aklr/core/contract/expr.py:108)
- S10 aklr/core/contract/fastkey/_fast_key_fn.py:284 the storage checks: stored layout, not negated, `data_ptr() % 16 == 0` (:40), device; output overlap at :239
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
- S37 aklr/plugins/sglang/hook.py:63 the route runs unless `torch.compiler.is_compiling()`; a `FallbackReason` runs the original
- S38 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_layouts.py:34 `AMDMFMALayout` version 4 is gfx950
- S39 aklr/core/validation/command.py:31 `PATH`, `--base`, `--stages`, `--devices`, `--out`; validate asks git for the repository root and `HEAD` (aklr/core/validation/stages/validation_scope.py:40, :43) and takes only paths in it (:55)
- S40 .github/workflows/ci.yaml:296 "triton==3.8.0 is required"; the image at :5; the baseline aiter installed with `AITER_USE_SYSTEM_TRITON=1` (:292), whose setup.py otherwise replaces the installed Triton (ROCm/aiter@dbd8bf5:setup.py:124, :131–139)
- S41 aklr/core/validation/stages/_bench/_report.py:40 `_MEASURED_AS`: the correctness codes, `build_failed`, `candidate_crash`, `candidate_timeout`, `not_faster`, `regressed`; a finding located at a workload with one of them is a demotion (:129–137), and a schema with no qualified workload gets `no_qualifying_workload` (:138–151); bench locates every finding of a batch, a failed build included, at each of its workloads (aklr/core/validation/stages/_bench/_stage.py:343–348) and gives a candidate 900 s plus 300 s per workload (:49–50, :283); the run concludes `passed`, exit 0, unless a stage fails or aborts (aklr/core/validation/pipeline.py:155–164)
- S42 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/0a07f9f9342c6cf1.json#kernel.py:1262 `gl.static_assert(NORM_STRIDE == (8 + HEADS) * 128)` (and `CORE_STRIDE == NORM_STRIDE`, :1261) in `_norm_quant_large`, launched only when `m > 4096` (:1332, :1449) with `NORM_STRIDE=normalized.stride(0)` (:1469); the file's Points are 2049 and 4096 tokens; `normalized` is `[n_tokens, n_v_heads, value_dim]` (aklr/plugins/sglang/models/qwen3_5/contract.py:171)
- S43 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/3b99d19659746a36.json#kernel.py:72 `for candidate in gl.static_range(1, BATCH)`; `gl.static_range(1, BATCH)` in 10 more of the 15 Qwen3.5 GDN prefill files, `gl.static_range(BATCH)` in 2 others (aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len8192_n_v_heads16_key_dim128_value_dim128_conv_dim3072/5ead04831eb8048c.json#kernel.py:84); `n_sequences` 1–2047 (aklr/plugins/sglang/models/qwen3_5/contract.py:73)
- S44 local:/opt/venv/lib/python3.12/site-packages/torch/_torch_docs.py:8242 `torch.nonzero` on a CUDA tensor causes host-device synchronization, `torch.argwhere` too (:6861); `repeat_interleave`'s `output_size` avoids it (:13248); a bool index goes through `nonzero` (torch/_meta_registrations.py:3634)
- S45 aklr/core/contract/op.py:257 a layout variant may name only the contract's tensor inputs (:215), else "layout ... names unknown tensors" (:258), and "an output is never a 4-bit dtype" (:214); a tensor a variant does not list is `RowMajor` (aklr/core/contract/spec/_layouts.py:33); `per_group_quant_mxfp8@1.0` (aklr/ops/quant/contract.py:55) outputs E4M3FN `quantized` `[n_tokens, hidden_size]` (:38, reused at :68) and uint8 `scales`, "E8M0 exponent bytes at [row, group]" (:70), in its one `row_major` variant (:75); `per_group_quant_fp8@1.0` declares its fp32 scales `[n_groups, n_tokens]` (:46)
