---
title: "Authoring a Triton or Gluon kernel for an AKLR schema"
kind: method
ops: [gemm, moe, attention, gdn]
gens: [gfx950]
triton: 3.8.0
basis: source-semantics
---

# Authoring a Triton or Gluon kernel for an AKLR schema

How to write a Triton or Gluon kernel for an AKLR schema and validate it at the schema's Points on gfx950 (MI355X) with
Triton 3.8.0: the language, the files and entry, `run`, one binary per Point, scratch, launches, graph capture,
fallbacks, a Gluon skeleton, validation. Following it does not guarantee that validation passes. AKLR's rules are
linked, not restated (`aklr/ops/README.md` § The kernel section and ABI is the authority).

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
does not synchronize (§4); make fixed Point dimensions and compile-time configuration choices constexprs, and keep
ranged scalars and `any_size` axes runtime unless every variant they select is warmed (§5); allocate scratch in the call
or cache per-shape scratch in a compile function, and initialize every region before its first read in each call (§6);
consider a launch where a phase needs results from other workgroups or a separate initialization or conversion, measure
whether fusing is faster, and keep every call capturable (§7, §8); compile for gfx950 without a GPU (§10); after every
edit give the file its new uuid and run `python -m aklr export check` (§3); validate on one GPU, and fix or delete every
Point it demotes (§11). Each step is checked in its section; the procedure as a whole was not run end to end.

## 2. Plain Triton (`tl.*`) or Gluon

At AKLR 43e112e there were 154 Triton-language kernels: 4 plain `@triton.jit`, 150 `@gluon.jit` (280 on main at 0e02a81:
4 plain, 276 Gluon). The table lists the families at 43e112e; the families added since (the DSA indexer, Qwen3-Next
gated attention, more DSA, MoE and GDN files) are in [kernels/kernel_library.md](kernels/kernel_library.md).

| Language | Families at 43e112e (kernel files) |
|---|---|
| plain `tl.*` (4) | `gemm@1.0` bf16 GEMMs at M ≤ 64, n6144/k2048 and n2624/k6144 (2), RMSNorm and add-RMSNorm at hidden 6144 (2) |
| Gluon (150) | fp8 block-scaled GEMMs (24); MoE layer plugins: DeepSeek-V2/GLM MXFP4 (19), MiniMax-M2 FP8 (9) and MXFP4 (23), Qwen2-MoE router and experts (14); one-query-token attention: DSA sparse MLA (8), Qwen3.5 gated attention (9); GatedDeltaNet one token per step (18) and chunked (26) |

The two share the launcher and the JIT cache (`@gluon.jit` is a `JITFunction` subclass, S1) and differ in the first
compiler stage: plain Triton runs `make_ttir` then `make_ttgir`, Gluon replaces both with `gluon_to_ttgir` (S2).

| | plain `tl.*` | Gluon |
|---|---|---|
| Layouts | assigned by the compiler: coalescing, the MFMA (`matrix_instr_nonkdim`; `kpack` is forced to 1 on gfx950, S4), conversion removal | written by you: `BlockedLayout`, `SliceLayout`, `DistributedLinearLayout`, `AMDMFMALayout`, `DotOperandLayout`, shared layouts; a `gl.arange`, `gl.zeros` or `gl.full` whose layout no other tensor fixes fails to compile ("Failed to infer return type") |
| Software pipelining | `num_stages`: loop scheduling, operands streamed into LDS by async copies. In the n2624 GEMM's 16-row path `num_stages=1` stages the tiles through registers into 16 KiB of LDS; the committed 3 uses 101312 B (about 99 KiB) fed by direct-to-LDS `buffer_load_dwordx4 … lds` (96 in the ISA at m = 2 and 8 with 4 warps, 48 at m = 1 and 16 with 2; the other m were not compiled); `num_stages=1` took 2.2–2.3× the time of 3 (18.33–18.61 against 7.97–8.36 µs at m = 2, 4, 8), while m = 64, whose launch did not change, stayed at 0.995–1.003× in speedup (3 runs; the LDS and ISA counts are from 1 compile) *(field, E1, E2)* | prefetch by hand ([gluon/memory.md](gluon/memory.md) §12): Gluon's AMD stages do not read `num_stages`, `matrix_instr_nonkdim`, `kpack` or `schedule_hint` (S2), and option parsing resets gfx950's `kpack=2` to 1 (S4); 36 kernel configurations from 8 Gluon families, each built with 9 such option variants, kept identical instructions across the variants (two runs), and a launched copy kernel got 9 cache entries with the same code *(field, E3)* |
| Buffer instructions | `tl.load` / `tl.store` become `buffer_load` / `buffer_store` where a pass proves the range, from the `S` key (§5) or range analysis (S2, S3) | only where you call `gl.amd.cdna4.buffer_load` / `buffer_store`; `gl.load` stays `global_load` ([gluon/memory.md](gluon/memory.md) §2, §3). In the GDN decode file `299133886d479017` at 2 tokens and 4 slots, the recurrence with buffer `.cs` and with pointer `.cs` accesses both met the contract in 36 calls (3 seeds × 3 slot patterns × 2 output poisons; 18 repeat pairs bitwise equal) *(field, E4)*. Validate read 8.79 µs for the buffer arm and 8.81 µs for the pointer arm, the mean of each run's median call time over 4 runs per arm *(field, E1)*; by device duration on a cold device, the two were within ±1% of each other for 4 of 9 seed and slot inputs, pointer slower by 1.2–2.8% for 3, and 2 straddled ±1% *(field, E5)*: no general buffer-or-pointer speed rule follows |
| LDS | implicit | `gl.allocate_shared_memory`, `gl.amd.cdna4.async_copy.buffer_load_to_shared`, `commit_group` / `wait_group` (with two groups of two copies in flight, `wait_group(1)` becomes `vmcnt(2)`), swizzled and padded layouts |
| Matrix core | `tl.dot`, `tl.dot_scaled` | `gl.amd.cdna4.mfma` (`v_mfma_f32_16x16x32_bf16` in §10's skeleton), `mfma_scaled` with e8m0 scales (a 16 × 16 × 128 e4m3 tile: one `v_mfma_scale_f32_16x16x128_f8f6f4` with the scales in registers; `None` passes a register set to 0x7f, scale 1); `gl.dot_fma` is a vector-ALU dot product off the matrix core, on a `BlockedLayout` accumulator ([gluon/mfma.md](gluon/mfma.md) §1) |
| Lane-level code | `inline_asm_elementwise`, not knowing which lane holds which element | the same call, each element's lane fixed by the layout (`v_readlane_b32` of lane 5 reads x[5] with one element per lane, x[10] and x[11] with two); shuffles and ballots: [gluon/wave_ops.md](gluon/wave_ops.md) |

- **`num_warps` is part of a Gluon kernel.** The module takes `ttg.num-warps` from the launch, and the verifier rejects
  a layout with another warp count ("Layout has 4 warps per CTA, but the context requires 8 warps per CTA") or with
  other than 64 lanes (S1). A launch without `num_warps` gets 4 (S4), which hard-coded 4-warp layouts rely on (S5).

The committed plain bf16 GEMMs and norms ([gemm/small_m_gemm.md](gemm/small_m_gemm.md)) are starting points for plain
`tl.*`, which leaves layouts and LDS buffers to the compiler ([gemm.md](gemm.md) §9), not a rule for choosing the
language. Their `num_stages` matters: fewer stages slowed the n6144 GEMM at m = 2, 4, 8, 16 by 1.533–1.625× and RMSNorm
at 1024, 2048, 4096 tokens by 1.047–1.120×, 1.092–1.095× and 1.051–1.052× (one run, 4 rounds per arm in a fixed order,
which may bias the smaller effects; add-RMSNorm not tested) *(field, E6)*. Gluon exposes e8m0 `mfma_scaled`, explicit
LDS and async-copy depth and lane-level code (table). AKLR's Gluon fp8 GEMMs at N 3072, K 1536 (unshuffled E4M3 with
fp32 block scales) took 0.777–0.781×, 0.755–0.764× and 0.759–0.768× the flushed graph time of aiter's plain-Triton
kernel at m = 2, 8, 16, both within the contract's bounds (one run, 5 rounds) *(field, E6)*: a comparison of two
implementations, not of the languages, and not of preshuffled MXFP4.

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
  files at 43e112e list one Point (224 of 280 on main); per Point, dispatch serves the best measured file
  (`aklr/ops/README.md`, top). `kernel` holds `language` (`"triton"` for plain and Gluon alike), `targets` (gfx950,
  MI355X in every file) and the four fields below.
- **`sources`.** One `{"path", "content"}` per file, each path one `<identifier>.py`, sorted (`aklr/ops/README.md` § The
  kernel section and ABI). At 43e112e, 86 kernels are one `kernel.py` and 68 add 1–4 helpers (`attention.py`,
  `prologue.py`, `initial_state.py`, ...) imported relatively (`from .adapter import validate_inputs`), the files being
  submodules of a package named after their content hash (S7). No schema imports another's files: each copies its
  helpers. Each `content` holds no control character but `\n`, as every committed source does (272 at 43e112e, 507
  on main), and ends in one newline with no blank line before it: the schema parser rejects a trailing blank line
  (`kernel.py:<line>: blank line at the end`), a tab and a carriage return (`kernel.py:<line>: forbidden character …`;
  same section).
- **`entry_point`.** `kernel.py::run` (139 files at 43e112e), or `main.py::run` with an `adapter.py` that checks the
  inputs (the 15 N = 3072 fp8 GEMMs). The entry file needs a module docstring (its content: same section), else
  `source_rejected` (`SA4 kernel.py:0 no module docstring`); 135 files at 43e112e keep a `# Created by OpenAI` comment
  above it, which SA4 accepts.
- **`requires`.** `{"triton": ">=3.8"}` in 135 files at 43e112e (261 of 280 on main). Triton 3.7.0 lacks `gl.clamp`,
  `gl.aggregate_replace`, `gl.expect_zero`, `gl.amd.slice`, `llvm_fn_attrs` and `gl.amd.cdna4`'s
  `compute_efficient_padded_shared_layout` and `scaled_upcast` (S8); under it AKLR's toolchain check reports `>=3.8`
  unmet and an empty `requires` met, and none of the 11 n3072 fp8 GEMM files from m = 16 up compiles, m = 32 and 64
  without calling those APIs *(field, E20)*. Declare it on every Gluon kernel you have not run with 3.7.0; the 19 files without it (the
  15 n3072 fp8 GEMMs, the bf16 GEMMs, the norms; 19 on main too) include 12 n3072 files that call 3.8-only API:
  add it when you start from one.
- **`author`.** One line naming a person, team or agent; the uuid hashes it, so editing it renames the file and drops
  the measurements until CI writes them again (same section).

A new scenario starts from `python -m aklr export` or the script of `docs/ops_tutorial.md` § 1. B; a second kernel for a
committed scenario from a copy of one of its files (`aklr/ops/README.md` § Workflow). Run from the repository root, this
writes the copy at its uuid path in the exact text the audit requires (S9), which passes the schema-text and path checks
when the source meets the `content` rule above; passing the source audit also needs the entry ABI and SA0–SA4 (§4):

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

**Core op or plugin op.** A core op's kernel implements its contract (for example GEMM or RMSNorm) against the baseline
its package defines (aiter and SGLang implementations among them), reached through its `aklr.ops.<op>` api; at 43e112e
and on main no AKLR plugin route calls those core apis directly. A plugin op replaces a whole SGLang function (an MoE
layer's forward, a GatedDeltaNet layer, an attention step for one query token per sequence) through an `@sglang_route`, against SGLang's own path,
and may take framework state as mutable inputs (`aklr/plugins/README.md` § Architecture, § Rules for routes;
`docs/plugin_tutorial.md`).

**No contract yet.** A kernel serves a contract: validate rejects a schema whose `contract_id` no op package registers (`unknown contract_id`, exit 1). For an op without one, add the op package first, in its own pull request (its files: `docs/ops_tutorial.md` § 1. A; for a plugin op, `docs/plugin_tutorial.md` steps 2–4). A layout variant that names an output is rejected (`names unknown tensors`), and so is an output of `float4_e2m1`, AKLR's only 4-bit dtype (S45): an output is row-major in the order of its declared shape (`per_group_quant_fp8@1.0` declares its scales `[n_groups, n_tokens]` to store them group-major), and MXFP4 codes leave packed in `uint8`, an integer output that must equal the reference ([numerics.md](numerics.md) §1). The nearest quantization op, `per_group_quant_mxfp8@1.0`, writes row-major e4m3 codes and uint8 e8m0 bytes (S45). A standalone SwiGLU + MXFP4 requantization would be such an op. In the committed DeepSeek-V2 schema `7e9c13e0f2d58a3b`, the one-token `_finish_gate_up` launch writes packed uint8 activations and e8m0 scales into call-local `aq` / `aqs` scratch, which `_down_reduce` reads through `_tiny_projection` (S11); its activations are row-major and its expert weights use shuffled offsets. Read any other MoE's scratch format from its own producer and consumer code.

## 4. `run`: what it receives, what it must not do

`run` takes exactly the schema's `kernel_call_keys`, keyword-only: its tensors and scalars (an absent optional key is `None`; inputs hidden by schema-default optional attrs are omitted), attrs that have a parameter, layout arguments such as `operand_format`, `out_dtype` when the one output is `out`, and every output, preallocated (`aklr/ops/README.md` § The kernel section and ABI); from the repository root, `ContractRegistry.default().parse_schema(Path(p).read_bytes(), p).kernel_call_keys` (`aklr.core.contract.registry`) lists them for the schema at `p`. The Point's logical sizes follow from the tensors' stored shapes under the contract's layout and packing rules: a packed 4-bit last dimension holds two logical elements per stored element. It must write every output: bench poisons the outputs and caught a GEMM copy that left rows unwritten (§8), which does not mean every omitted write is caught. The rest (mutable inputs, no return value, no output kept, every value of a scalar's range) is in the same README, § Schema fields. MXFP4 and e8m0 tensors may arrive as `uint8` (the MoE's uint8 weights and scales were served *(field, E7)*) or as `torch.float4_e2m1fn_x2` / `torch.float8_e8m0fnu`, which Triton 3.8.0 cannot type (`KeyError: 'float4_e2m1fn_x2'` at the launch, S20): pass `.view(torch.uint8)`, as the DeepSeek MoE's entry does (S11).

Dispatch calls `run` only with tensors in the stored layout, not negated, aligned (`data_ptr() % 16 == 0` in the source), on the table's device, no output byte overlapping an input (S10): on `gemm@1.0` a non-contiguous `row_major` input, a negated one, one 2 bytes off alignment, a CPU call against a `cuda:0` table and an output aliasing an input each fell back `out_of_contract` before `run` *(field, E7)*. So strides can be constexprs folded from the shape, and a scalar whose `ScalarSpec` in the op's `contract.py` declares one value can be hard-coded, as the DeepSeek MoE's entry hard-codes `routed_scaling_factor`, declared `(2.5,)` (S11): 2.0 and 3.5 fell back before `run`, 2.5 was served *(field, E7)*. Across six contracts, calls with a changed attr, an unlisted value of a non-`any_size` axis or another layout fell back before `run`, while GDN calls with 4 and with 8 slots, an `any_size` axis, were both served *(field, E7)*: do not hard-code an `any_size` axis. Pass a ranged scalar at run time: an RMSNorm copy (`cf637bff`) with `epsilon` hard-coded to 1e-5 passed validate, yet called with `epsilon` 1e-3 on inputs of scale 0.01 was off by up to 224%, the committed file exact *(field, E8)*.

The audit's rules SA0–SA4 are stated in `aklr/ops/README.md` § The kernel section and ABI (their code: S12); a
`source_rejected` finding names the rule, file and line. What they mean for a Triton or Gluon kernel:

- A def whose single decorator is `@triton.jit` is device code (S13 lists `@gluon.jit` beside it): `torch.tensor`,
  `a.add_` and `a.item()` pass the audit in its body and are rejected (SA3, SA2) in a plain def or under a second
  decorator. A `@gluon.constexpr_function` is not device code: called from `run`, it is audited as launch code. Nor is a
  def with `@triton.autotune` or `@triton.heuristics` above `@triton.jit`, where the same three calls are rejected; the
  README asks for a compile function instead. None of the 154 kernels at 43e112e uses either (none of the 280 on main).
  An autotuner times its unpruned configurations at a key's first in-process miss, unless one configuration remains or
  the disk cache holds the choice (S17), so two processes can serve one Point with different binaries: with two
  equal-cost configurations, each of 16 fresh processes timed both at its first call and none after, and 8 picked the
  first, 8 the second *(field, E9)*.
- SA3 rejects its I/O names in device code too (S12): a kernel parameter called `input` is rejected at its first use,
  and so is a `print` in a `@gluon.jit` body; `gl.device_print` passes.
- SA2 rejected `x.add_(1)` in `run` and `weight.add_(1)` in a helper whose parameter carries the input's name; the same
  write through a helper parameter renamed `buf` passed the audit, and bench reported `input_mutation` at
  `first_capture/0` *(field, E10)*, since it compares read-only inputs byte for byte (S14).
- Reading `os.environ[...]`, `os.environ.get`, `os.getenv`, `time.time`, `time.monotonic`, `time.monotonic_ns`,
  `time.process_time`, `time.perf_counter_ns` and `datetime.datetime.now` passed the audit; SA3 rejected
  `time.perf_counter`, `torch.cuda.Event` and `elapsed_time` *(field, E10)*. Passing the audit does not make a call safe
  at a later stage.
- **A `@triton.jit` / `@gluon.jit` body does not read a plain module-level Python constant under the default compiler
  policy** (same section): pass it as a `tl.constexpr` / `gl.constexpr` argument or compute it in a
  `@gluon.constexpr_function` (`from triton.experimental import gluon`; 3.8.0's `gl` has none). Triton also accepts
  globals made with `gl.constexpr(...)` (five AKLR files at 43e112e keep such globals, layouts in three, S15; 18 on
  main), modules, JIT functions and dtypes; a plain `BLOCK = 64` fails the default compile (S16). Reassigning a
  `gl.constexpr` global the kernel reads (4 → 5), or a plain global it was allowed to read (3 → 4), after the first
  compile raised `RuntimeError` at the next launch; reassigning the same value, or changing a global the kernel does not
  read, did not (two runs; mutating an object in place was not tested) *(field, E3)*.

## 5. One binary per Point

**Pass the token count as a constexpr.** 122 of the 150 Gluon kernels at 43e112e (210 of 276 on main) have a `@gluon.jit` parameter `M`, `ROWS`, `TOKENS` or `BATCH` declared `gl.constexpr` (counted by name, Qwen3-Next's runtime-count files included). Each Point then compiles its own binary at its first call, though different Points may share a configuration. Compiled for gfx950, 8-, 16-, 32- and 64-row tiles whose constexpr M covered the tile had no mask instructions (3–56 with a runtime M), while masks remained where M did not fill the tile; over M = 1–128, a `range(M)` loop compiled without a branch at M = 1–22 and with 1–2 at 23–128, using 17–74 VGPRs against 66 with a runtime M *(field, E11)*: a constexpr removes some masks, loops and flags, not all. That first call must be outside any CUDA-graph capture and must compile every variant the Point's scalars select (`aklr/ops/README.md` § The kernel section and ABI). A `gl.static_range` over a constexpr count unrolls fully (a 4-load body: 120 amdgcn instructions at 8 trips, 6513 at 512, 120–129 with `range` *(field, E21)*): 13 of the 15 Qwen3.5 GDN prefill files walk their sequences that way (S43), fine at their Points' 1–8 sequences; for the 2047 the contract allows, use `range`.

**Choose the configuration once, where host and kernel both read it.** A `@gluon.constexpr_function` called outside
JIT returns its Python value and inside JIT a constexpr (S17), so one function gives the launch its grid and the kernel
its tile shapes: the Qwen2-MoE router's kernel reads `config: gl.constexpr = _shape_config(M)` and its host sizes the
grid from `_shape_config(rows)` (S18), as the skeleton of §10 does with `_config`.

**A `@functools.cache` compile function** is the pattern `aklr/ops/README.md` § The kernel section and ABI shows: a
module-level def of shapes whose cache holds compiled code, constants and per-shape scratch only. Checked directly with
AKLR's audit (`check_kernel`), its own body passed `torch.empty`, `torch.tensor`, `.item()`, `triton.compile` and
`.warmup`; in a def nested inside it, `torch.tensor` and `.item()` were rejected (SA3) and the other three passed
*(field, E10, E12)*. Passing there means none is on an SA deny-list, not that it is safe. Eleven of the 154 kernels at
43e112e have one (42 of 280 on main, with the 31 Qwen3-Next gated-attention files): the two plain-Triton GEMMs cache a
launch closure per shape (`_launch_for`), n2624's with its workspace size (S19); the nine Qwen3.5 gated-attention files
the fp32 unit KV scale both launches read, `_unit_scale(device)` returning `torch.ones(1, ...)` (S35): in `run`,
`torch.tensor` is rejected (SA3) and `torch.ones` allocates and fills a tensor per call. None of the 154 (nor of the 280
on main) calls `warmup()` or `triton.compile`: binaries live in Triton's in-process JIT cache, which meets the
first-call rule while the key below is a function of the Point.

**What enters Triton 3.8.0's JIT key** (S17): the launcher looks `(tuple(specialization), str(options))` up in
`kernel_key_cache`, then the binary under the string key it maps to, in a cache per device; 12 specialization and option
combinations gave 12 binaries in each frontend (two runs on one GPU) *(field, E3)*.

| Argument | Enters the key as | So |
|---|---|---|
| `tl.constexpr` / `gl.constexpr` | its value, which must be hashable (S17) | one binary per value; a list fails at the launch (§10) |
| int | `i32` / `i64` / `u64`; the value 1 becomes a constexpr; a multiple of 16 gets `D` (S20) | Points share a binary only within one class; `do_not_specialize=[...]` on `@gluon.jit` drops the value keys |
| runtime float, bool | `fp32`, `u1` (S20) | 9 floats (0, 1, −1, 16 and 1e30 among them) with True and False used one binary per frontend, and 15 more floats, inf and NaN included, added none *(field, E3)* |
| `None` | a constexpr (S20) | an absent optional input compiles its own variant |
| tensor (HIP) | its dtype, `D` when `data_ptr() % 16 == 0` and `S` when AMD buffer passes are enabled and its whole storage is at most 2^31 − 1 bytes (S20, S21), from a Python callback run per tensor at every launch, cache hits included; on the argument `D` becomes `tt.divisibility = 16` and `S` `tt.pointer_range = 32`: offset 0 of a 4 KiB buffer got both, offset 8 only `S`, a 2 GiB + 16 KiB tensor only `D`, 3 binaries *(field, E8)*; `S` changed the code of 3 plain-Triton kernels, only with AMD buffer passes on, and of 3 Gluon copy and gather kernels neither way (two compile runs) *(field, E3)* | a misaligned view, or a storage that crosses 2 GiB, compiles another binary |
| launch options | `str(options)`: `num_warps`, `waves_per_eu`, ... as passed, in keyword order (S17) | launch from one call site; an option Gluon ignores (`num_stages`) still splits the key |

**Keep runtime** what may change while the Point stays the same, since a key that changes after the Point's first call
compiles again, possibly inside a capture (§8): runtime floats, which enter the key only as `fp32` *(field, E3)*, as
the Qwen3.5 GDN decode keeps its scale and epsilon (S22); scalar inputs, unless one value per process is guaranteed (the
MiniMax EP files take `expert_offset` as `OFFSET: gl.constexpr`, one value per process by their contract, S23);
`any_size` axes (a pool's slot count), on which dispatch does not key; and the alignment of pointers into a buffer you
carve (§6). A runtime int is still keyed on its class (1, a multiple of 16, i32, i64) and bench warms only the default
and the workload's scalars (S14): put an int that may change within a Point in `do_not_specialize` (the dtype classes
`i32`, `i64` and, for large non-negative values, `u64` remain).

## 6. Scratch

- **Allocate inside the call**, on an argument's device, sized from the shapes: `torch.empty` (117 kernels and 686 call
  sites at 43e112e), `torch.empty_like`, `.new_empty`. One allocation can be carved into typed regions at aligned
  offsets (64 KiB or 256 bytes in AKLR's arenas, S24); an offset that is not a multiple of 16 bytes loses `D` and gets
  another binary (§5). A first call over bench's 16 GiB limit (S14) fails `build_failed`, as one allocating 17 GiB did
  *(field, E8)*.
- **`torch.empty` holds whatever was there**, in an eager call and in every replay. In every call, write before any read
  what a launch reads before it writes: atomic counters and cursors, `-1` job sentinels, padding rows, accumulators
  added into. None of AKLR's Triton kernels calls `torch.zeros` or `fill_` (154 at 43e112e, 280 on main): an earlier
  launch of the call writes the initial values, as program 0 of the MiniMax MXFP4 MoE's first launch stores 0 into the
  expert counts and the total and -1 into the job slots, on which the next launch takes atomic tickets (S25).
- **State that must start fresh** is cleared by a small launch reading its flag on the device, as GDN prefill's
  `initialize_fresh` zeroes the slot of each sequence without initial state (S5).
- **Nothing carries from one call to the next.** `run` keeps nothing derived from an argument across calls, a compile
  function only compiled code, constants (§5) and per-shape scratch rewritten whole before every read (same README
  section). The audit rejects a `global` statement (SA1) but not a module-level dict written in `run` *(field, E10)*: a
  `run` that kept its first output in one and copied it into later outputs passed the audit and failed bench with
  `numeric_mismatch` (`first_capture/0`, `numeric/0`, `determinism/0`) and `post_timing_mismatch` *(field, E8)*
  ([../pitfalls/state_kept_across_calls.md](../pitfalls/state_kept_across_calls.md)).
- **Reuse a region within a call in stream order**, as MoE files overlay short-lived scratch on a later buffer (S26):
  launched on the same stream before the region's next writer, its last reader saw the values from before the
  overwrite in 20 of 20 eager calls and 20 of 20 graph replays over 16.8 M floats *(field, E2)*.

## 7. Launches: what one costs, where a boundary goes

What one more launch costs depends on who times it:

| Where | One more launch costs | Ruler |
|---|---|---|
| A replayed CUDA graph, as SGLang replays its fixed-shape one-token-per-sequence (decode) steps | 1.82 µs (1 program) to 1.84 µs (256 programs) of replay time per further chained trivial Gluon kernel (one 64-float load, add and store per program), in a bare CUDA graph, not SGLang's; the replay itself added 3.6–3.8 µs; with 8 launches per graph, 2.27 µs per launch with 1 tensor argument and 2.31 µs with 30 *(field, E13)* | CUDA events around graph replays, least squares over 1–64 kernels |
| AKLR's stored speedup | the sum of the call's per-kernel device durations, flush excluded, from AKLR's rocprofiler-sdk collector since #33 (S46), from Kineto before (S27). Measured before #33: a copy and a sum read 10.84 µs in one call against 6.50 + 4.62 µs apart, a 6.5 µs call stayed 6.5 µs beside a 169.5 µs flush, 8 trivial kernels read 32.4 µs against 8 × 3.88 µs, and no dispatch read below about 4.2 µs *(field, E13)* ([../pitfalls/aklr_dispatch_floor.md](../pitfalls/aklr_dispatch_floor.md)) | AKLR's `bench_cudagraph` |
| An eager call | 7.30 µs (1 program) to 7.44 µs (256 programs) of host time per `kernel[grid](*args)` of a one-tensor trivial Gluon kernel in Triton 3.8.0's launcher, the wall time within 0.2% of it, the GPU waiting on the host; 6.25 µs + 1.00 µs per tensor + 0.12 µs per int argument by least squares over 7 argument lists (7.32 µs with 1 tensor, 38.45 µs with 32, 17.12 µs with 8 tensors and 24 ints); HIP's Python callback for the `S` key (`get_tensor_specialization`, S20, S21) runs once per tensor per launch and took 0.58 µs per call timed alone, 57% of that slope ([debugging.md](debugging.md) §8); 8 launches of a 15-tensor kernel took 169 µs *(field, E13)* | host clock around 2000 launches |

Host time lengthens the call only while the GPU finishes kernels faster than Python issues them: those 8 launches took
172 µs with trivial kernels, their host time, and 3166 µs with 396 µs kernels, their sum *(field, E13)*. Only eager calls
pay it — uncaptured plugin calls such as many-token (prefill) batches (the GDN prefill routes refuse capture, the MoE routes decline SGLang's piecewise and breakable
graphs, S28, S29) and direct core-op calls — so there keep tensor arguments few; a 4-key option dict built per call cost
0.06 µs more per launch than one built once (Qwen3-Next decode builds its at import, S30) *(field, E13)*.

**Where the committed kernels end a launch.** A Kineto trace of the calls at each of the 242 Points at 43e112e counted
launches per call: 1 in the fp8 GEMMs and the norms; 2 in the n2624 GEMM at m = 64, split-K `_gemm_kernel` then
fixed-order `_reduce_kernel` ([gemm.md](gemm.md) §9); 2–8 in the MoE plugins, 2–3 in DSA, 3 in gated attention, 1–2 in
GDN decode, 5–8 in GDN prefill *(field, E14)*. The stages they separate: [moe.md](moe.md) §2,
[attention.md](attention.md) §2, [gdn.md](gdn.md) §3 and §7.

**Consider a launch where a value crosses workgroups or a phase needs a separate initialization or conversion.** `gl.barrier()` synchronizes one workgroup, and gfx950 has no CTA clusters to widen it (`num_ctas > 1 not supported on gfx950`, S4); no result may depend on a workgroup that may not be running, since a launch's workgroups need not all be resident: no unbounded spin on a partner, and no giving up and computing without its data. So in AKLR's Triton kernels a value one workgroup produces and another consumes — a token's expert logits before its top-k, every split before a merge, every reader of a state before it is overwritten — ends the launch, and stream order is the synchronization. At 43e112e none tests an atomic's result for the last arrival or waits in a loop on memory, so none uses an in-launch fold (each split counts in on an atomic counter, the last to arrive merges in a fixed order: [../gemm/wait_free_fold.md](../gemm/wait_free_fold.md), measured there on FlyDSL kernels) or a bounded wait that hands a timed-out share to the last arrival; all 42 of its global atomic call sites are relaxed (all 87 on main; [numerics.md](numerics.md) §8). A Gluon fold needs its counters at zero before each replay: without a reset, 1999 of 1999 later replays were wrong at each of 9 sizes; with the counters zeroed by an earlier node of the graph, 2000 of 2000 replays were bitwise equal with `acq_rel` and with `relaxed` (one run; 200 of 200 at 16 splits × 16384 outputs in three earlier runs); a counter the last arrival reset itself gave 1000 of 1000 correct replays in each of 3 runs. A separate merge launch took 0.03–0.65× the replay time of the `acq_rel` fold and 0.51–0.92× that of the `relaxed` one at 4, 16, 64 splits × 4096, 16384, 65536 fp32 outputs (64 outputs per one-wave program), and 0.12–0.82× on AKLR's timer at up to 16 splits and 16384 outputs; each fold's time includes its zeroing node *(field, E2, E15)*. Phases that are independent, or cheap to recompute, can share one launch:

- independent phases can share a grid that splits roles by program id (DeepSeek router beside input MXFP4 quantization,
  S11; [moe.md](moe.md) §3). Two toy elementwise phases of 131072 elements, outputs equal, took 0.762–0.854× the replay
  time of two launches in three of four runs and 1.085× (9.191 against 8.471 µs) in the fourth, and 0.506–0.516× on
  AKLR's timer (4.30–4.44 against 8.49–8.61 µs) in all four *(field, E8)*;
- a small decision may be recomputed instead of published (S11; [moe.md](moe.md) §4): a toy Triton MoE at 2 tokens,
  top-4 of 32 experts recomputed in its 40 consumer programs, took 0.843–0.889× the replay time and 0.742–0.772× the
  AKLR time of a separate selection launch, same output (4 runs) *(field, E2)*. In two synthetic fp8 VALU MoEs (top-8
  of 256) at 1, 2, 8, 16 tokens, recomputing in every consumer was slower on both rulers, the output bitwise equal (one
  run, 16 rotated rounds per Point) *(field, E16)*;
- a split reduction under a `bitwise` contract gets a fixed-order reduce launch, not float atomics (n2624 GEMM, S19).

Compare publishing with recomputation by measuring it, not by a fixed launch cost: removing one toy phase launch saved
1.150–1.884 µs of replay time at 16 K, 128 K and 1 M elements in three runs, but −0.102, 1.587 and 0.325 µs in a fourth
(outputs equal), and 4.17–4.19 µs on AKLR's timer at 131072 elements *(field, E2)*. In a synthetic sweep of 56
configurations (7 decisions × 2 or 16 tokens × 40, 256, 1024 or 4096 consumer programs), recomputing a top-8-of-256
decision won at 1024 programs and publishing it won at 4096, with 4 or 16 split partials at 2 or 16 tokens; the median
cost of the extra launch (3.27 µs replayed, 4.47 µs on AKLR's timer), taken as a constant, predicted the faster arm in
only 49 and 50 of the 56 (one run, 12 rotated rounds per configuration; some identical-arm controls differed from zero,
so effects that small are not resolved) *(field, E16)*. A torch op in `run` is a launch too: the DSA files that cast `q`
with `q.to(torch.bfloat16)` there add a kernel per call (S31).

## 8. CUDA-graph capture

SGLang captures its one-token-per-sequence (decode) batches, and AKLR's routes for them run inside its graphs; the GDN prefill routes return `out_of_contract`
while a capture runs, "a capture would replay this call's lengths" (S29). Bench captures every check step — a first
capture replayed twice with new values copied into the same buffers, fresh captures per input set, a repeat under
`bitwise`, the post-timing checks (S32) — and fills the outputs with NaN (0xFF bytes for integer and sub-byte dtypes)
before the call (S14). Leaving a float element unwritten violates the contract: in one run, a copy of the fp8 GEMM
`f3970dec` that left 8 of its 16 rows unwritten failed at `first_capture/0` with 24576 = 8 × 3072 non-finite values
*(field, E8)*.

| Breaks | Why | Do |
|---|---|---|
| The Point's first call inside a capture | dispatch serves the fallback (`not_warmed`), and the graph keeps it in every replay (S33) | call every Point once eagerly first, as SGLang does (`aklr/ops/README.md` § The kernel section and ABI) |
| A JIT key that changes after the Point's first call (§5) | Triton compiles at a call that may be captured, which AKLR rules out | make every key a function of the Point: a constexpr warmed at 16 and launched at 32 inside a capture compiled there (one `_do_compile` call), which bench's tripwire on `JITFunction._do_compile` reports as "compiled in the capture" (S14), yet an RMSNorm copy whose key changed per call got no `capture_unsafe` finding from validate, only `no_qualifying_workload` *(field, E9)*, so do not count on validate to name it (§11). A warm `TRITON_CACHE_DIR` does not help: every in-process miss calls `_do_compile`, which reads the disk cache (S17), as each of two fresh processes sharing a warm cache did once *(field, E9)* |
| `.item()`, `.cpu()`, `.tolist()`, `torch.cuda.synchronize()`; a torch op whose output size depends on data: `torch.nonzero`, `x[mask]`, `torch.argwhere`, `repeat_interleave` without `output_size` | each failed a capture with `capture_error_mode="global"`, as bench captures (S14), and `repeat_interleave` with `output_size` captured *(field, E10)*: no synchronization is allowed while a stream captures, and the torch ops synchronize to size their output (S44) | SA3 rejects the first four in `run` but passes the four data-dependent-size ops, which only the capture catches *(field, E10)*: compute indices in a kernel into a worst-case buffer (below) |
| A host decision on device data: a grid or a branch from a count the GPU computed | it needs a synchronization, and a replay keeps the captured decision | size the grid from shapes for the worst case and exit early on a count read on the device (below) |
| State written once and assumed afterwards | a replay reruns only the kernels, never `run`'s Python | write counters, sentinels and padding in every call: "Padding is rewritten every invocation, including graph replay" (S34) |
| A Python value that changes per call | frozen into the graph at capture | read it on the GPU: "Live lengths, physical cache slots, and KV scales are always read on the GPU" (S35) |
| Allocation | a `torch.empty` tensor allocated in a capture kept its address over 3 replays and held the third replay's values *(field, E9)* | allocate per call as usual (§6) and do not depend on what the buffer held |

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
  To serve more or fewer Points, edit `workloads`, not `run`. A Point you add runs a branch CI has not measured, and
  generated files keep branches that no longer fit the ABI: the Qwen3.5 GDN prefill file `0a07f9f9342c6cf1` takes its
  `m > 4096` path at none of its Points, and that path static-asserts a row stride the `normalized` output (2048) cannot
  meet (S42). Validate each added Point on a GPU: a 37-token Point added to a schema parsed as valid, and validate
  queued it with the file's other 14 Points (`base=None`) *(field, E12)*, but only the GPU stages show whether it builds,
  runs correctly and qualifies.
- **Routes decide on SGLang's configuration** (graph modes, features, parallelism) **and on how SGLang hands over its
  tensors** (at 43e112e the routes had 5 composite decision blocks testing shapes, dtypes or lengths: DSA's `q_nope` /
  `q_rope` views of one buffer, the `out_proj` weight shape that selects SGLang's aiter GEMM, prefill lengths against
  rows) and return a `FallbackReason`. Shapes, dtypes, token counts and GPU models the op covers are the op's to judge;
  the route does not check them again (`aklr/plugins/README.md` § Rules for routes). The hook then runs SGLang's
  original function, as it always does under `torch.compile` tracing (S37). Where the op's inputs cost work, the route
  first asks `op.fallback_reason(...)` on stand-ins, as the Qwen3-Next prefill route does before its input projections
  (S29): the question launches no kernel and does not enter `run` (on `gemm@1.0`: `None` when served, `out_of_contract`
  for a misaligned `a`).

## 10. Starting a new kernel

1. **Start from the closest family** ([kernels/kernel_library.md](kernels/kernel_library.md) §3) and copy its file
   layout: entry, helpers, the host function that plans the launches. Read only the branch your Point takes, and keep
   that branch's constants together: changed alone, one of them broke a committed kernel (same page, §1), key splits 8 →
   4 in a 24-token DSA file failing to build, `gate_fold` 4 → 1 in a 1-token MiniMax file crashing, 4 of 4 runs each
   *(field, E1)*.
2. **Or start from this skeleton**, a `gemm@1.0` kernel for a scenario with `trans_b` and no bias such as
   `gemm_n6144_k2048`: 16-byte loads in a `BlockedLayout`, masked `buffer_load` / `buffer_store`, one MFMA with an
   `AMDMFMALayout` accumulator and `DotOperandLayout` operands, the token count as a constexpr, one configuration
   function for kernel and launch. It compiles with Triton 3.8.0 for gfx950 at M = 2, 16, 32 and 64 without spills, and
   passes SA0–SA4. Run at N 6144, K 2048 and M = 2, 3, 4, 8, 12, 16, 17, 32, 33, 48, 64, it was bitwise stable for one
   positive and one signed input per M over 20 eager calls, 20 graph replays and a second process, within 0–1 bf16 ulp
   on the positive input and at 0.960–0.984 of the bound 2^-8 |Σab| + 2^-18 Σ|ab| on the signed one. Exported as a
   `gemm_n6144_k2048` file and validated, it had no correctness, build or capture finding, but all six Points (m = 2, 4,
   8, 16, 32, 64) were demoted for speed at 0.373–0.434 of the baseline (one run, 3 rounds) *(field, E6)*: it needs
   tuning before it can serve a Point.

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

   `version=4` is gfx950's MFMA (S38; [gluon/mfma.md](gluon/mfma.md) §1). The masks keep rows past M out: Triton's buffer descriptor covers 2^31 − 2 bytes from the base, not the tensor (`num_records` `0x7ffffffe` in every buffer op of 27 probe kernels), so an unmasked `buffer_load` past a tensor's end read the memory behind it, while masked lanes read 0 without `other` *(field, E13)* ([gluon/memory.md](gluon/memory.md) §4). The two `convert_layout`s go through 8 KiB of LDS (loading in the operand layout instead: [gluon/layouts.md](gluon/layouts.md) §6, §9). The launch is a start, not a tuned choice: its 96 programs at N 6144 are fewer than the 256 CUs. At m = 2, 4, 8, 16 it took 2.882–3.158× the time of the committed `f806ad2b086d757d`, and prefetching one K step ahead made it 1.695–1.729× faster (one run, 3 rounds per arm in a fixed order; register allocation and scheduling changed too) *(field, E6)*. A Gluon stream of 96 programs × 4 warps rose from 876–886 to 3945–3995 GB/s with eight 16-byte loads per lane in flight instead of one (3 runs) *(field, E2)*, so the skeleton is probably short of bytes in flight: an inference, not a measured cause ([tuning.md](tuning.md) §3, §4). Grid and prefetch: tuning.md §6, §10; `num_warps`: tuning.md §7; `waves_per_eu=2` becomes `"amdgpu-waves-per-eu"="2,2"` and `.amdhsa_next_free_vgpr` 169 for a skeleton using 70 VGPRs; the compiler reports `Occupancy: 2`, a resource limit, not a measured residency (tuning.md §8). Small-M GEMM speed: [gemm/small_m_gemm.md](gemm/small_m_gemm.md).
3. **Compile for gfx950 without a GPU.** `triton.compile` with an `ASTSource` (`GluonASTSource` for Gluon) and `GPUTarget("hip", "gfx950", 64)` builds the launch's instructions, given its constexprs and specialization (`"DS"` for 1 MiB tensors at aligned addresses and `"D"` for n = 1024 matched; a pointer 4 bytes off alignment in a 3 GiB range with n = 1000 gets neither). Signatures name the dtype the launch sees: `*bf16`, `*fp16`, `*fp32`, `*fp8e4nv` (`float8_e4m3fn`), `*fp8e5` (`float8_e5m2`), `*u8` (also the uint8 views of MXFP4 and e8m0), `*i32`, `*i64`; an int is `i32` or `i64` by signed range, or `u64` above the signed-64 range (a constexpr when it is 1); a float is `fp32`, a bool `u1` (S20). It needs Triton and the kernel's imports (torch in 152 of the 154 AKLR files at 43e112e, 278 of 280 on main), not a GPU. A kernel whose `kernel.py` imports a sibling relatively (DSA decode `1148d1c8e30397f7`) fails `from kernel import ...` ("attempted relative import with no known parent package"): write its sources into `kpkg/`, put its parent on `sys.path` and `from kpkg.kernel import ...`, a package as AKLR makes one (S7). Built with the launch's signature, attributes and options, offline compiles gave the launch's amdgcn in 12 of 12 comparisons (one run) *(field, E3)*. This compiles the skeleton with Triton 3.8.0 for gfx950:

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

   LDS is `k.metadata.shared` (8192 here) while the amdgcn's `.amdhsa_group_segment_fixed_size` reads 0; launches of the
   skeleton passed the same 8192 B at all 11 M values tested *(field, E6)*. A Gluon kernel's `k.asm` holds `source`
   (the frontend's IR, before `gluon_to_ttgir`), `ttgir`, `llir`, `amdgcn`, `hsaco` ([debugging.md](debugging.md)
   §1–§2).
4. **Know what only a launch catches.**
   - A constexpr must be hashable: `INSTR=[16, 16, 32]` at launch failed with `unhashable type: 'list'` where
     `triton.compile` worked *(field, E8)*, since the launcher hashes each constexpr into its key (S17): pass a tuple.
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
alone needs no GPU. Without `--base` the committed `f3970dec` was timed against the baseline alone, at the retain
threshold; a copy with `--base origin/main` against the baseline at the entry threshold and the serving file
(`incumbent`) at the retain one. A Point that stores less than `AKLR_MIN_SPEEDUP` is not served (`not_qualified`): the
committed RMSNorm `cf637bff0784c454` stores 1.023–1.029 at its 7 Points of 3–18 tokens, none served at the default 1.03,
and 1.04 or more at the others, all served *(field, E17)*. Tune past the dispatch threshold (`aklr/ops/README.md` §
Environment variables; `harness/skills/aklr-read-validation/references/ruler.md` § Judging; reading a report: its
`SKILL.md`).

**Demoted Points.** Validate fails only on a file-wide finding or when no Point qualifies (`no_qualifying_workload`); a
Point qualifies when it has no correctness finding and its ratio to the baseline beats the entry threshold (S47). A
finding at one Point — `not_faster`, `regressed`, the correctness codes, `build_failed`, `candidate_crash` or
`candidate_timeout` there — demotes that Point (S41). `not_faster` and `regressed` are recorded as measured, not failed
(S47), so a file whose only Point regressed against the incumbent still passes, exit 0, when that Point beats the
baseline. A `numeric_mismatch` planted at one Point of three (1024 tokens) was listed under "demoted:" in
`aklr_validation/report.md` (`demotions` in `report.json`, with its `post_timing_mismatch` and `regressed`), and the run
printed `validation passed` and exited 0; a one-Point file that computes wrong failed as `no_qualifying_workload`, exit
1, the cause in the demoted list *(field, E8)*. Fix each demoted Point or delete it from `workloads` (the file takes a
new uuid, §3), then validate again (`docs/ops_tutorial.md` § 3).

What bites:

- **A git checkout**, with the file at its canonical path in the exact schema text (§3, S6, S9): validate asks git for
  the repository root (`git rev-parse --show-toplevel`) and takes only paths inside it (S39); a copy without `.git`
  and a file outside the repository each exit 2 before any check.
- **Measurements.** Delete `speedup` and `accuracy` from every workload of a new or edited file: with `--base`, measured
  fields that differ from the base are `perf_edited` (S13), and a file measured on some workloads only is invalid.
- **The image.** CI runs `lmsysorg/sglang:v0.5.22-rocm10-mi35x` with the baseline aiter (`AKLR_BASELINE_AITER`) installed over it, and refuses any Triton but 3.8.0 (S40). Validate in that image after installing the same aiter as `.github/README.md` § Install writes it, with `AITER_USE_SYSTEM_TRITON=1` to keep an installed Triton: otherwise aiter's setup.py runs its Triton installer unless torch is below 2.9.1, and may skip an installer failure (S40). A baseline that reaches aiter can move with the aiter build: the n6144 bf16 GEMM's baseline read 7.02 µs on the image's aiter and 7.95 µs on the pinned one at m = 8, 4.5% apart the other way at m = 16, the candidate within 1.2% at both *(field, E1)* ([tuning.md](tuning.md) §1). A local container may hold another Triton (3.7.0 has been seen): check `triton.__version__`. On 3.7.0 the launch's keyword check rejects `llvm_fn_attrs` (`KeyError`), and `schedule_hint="memory-bound-attention"` (AKLR's n6144 GEMM passes it) adds `"amdgpu-sched-strategy"="iterative-ilp"`, which 3.8.0 ignores (S8, S2). Built offline for gfx950 with the same options (a bf16 `tl.dot` loop, 64 × 64 × 64 tile, 4 warps, 2 stages, that hint), one probe had 274 static instructions and 16384 B of LDS on 3.7.0 against 405 and 33968 B on 3.8.0 *(field, E18)*.

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
| `numeric_mismatch`, `nondeterministic`, `post_timing_mismatch` | the math ([numerics.md](numerics.md)), also an arrival-order sum moved past its bound (`numeric_mismatch`, not `nondeterministic`); `post_timing_mismatch`: state kept across calls. Planted math, reduction-order and kept-state errors were reported as `numeric_mismatch` or `post_timing_mismatch`, none as `nondeterministic` *(field, E8)* | 6, 8 |
| `not_faster`, `regressed` | the speed ([tuning.md](tuning.md)) | — |
| `no_qualifying_workload` | no Point passed its checks and the entry threshold: the demoted list gives each Point's cause | 11 |
| `no_matching_toolchain`, `no_matching_device` | the installed Triton misses `requires`; the GPU is not in `targets` | 3 |

**A/B a variant** as a copy with the one change, at its own uuid path. Against the committed kernel of its Point,
validate the copy with `--base origin/main`: the two are timed in the same rounds. Alternate two uncommitted variants on
one idle GPU (A, B, A, B, at least four runs per arm) and compare the candidate times in each `report.json`: a file and
its comment-only copy read mean times 0.987–1.005 of each other on four Points. The same file validated three times
moved its baseline speedup (exp of `log_ratio_mean`) by up to 2.8%: fp8 GEMM `f3970dec` m = 16 2.4% (2.8% for its copy
under `--base`), GDN decode `29913388` 1.7%, MoE `d3bbbce5`, router `c3889614` 0.9% *(field, E19)*;
[tuning.md](tuning.md) §2.

## Evidence

AKLR paths are on `0e02a81` (current main) unless a line names another commit; a committed kernel is cited as
`<schema path>#<source file>:<line>`, and #64 renamed the 15 n3072_k1536 GEMM schemas after 43e112e. Triton paths are
`triton-lang/triton@c01b677` (tag v3.8.0), aiter's `ROCm/aiter@dbd8bf5` (CI's baseline). Each E line names the commit
or CI run its numbers were measured at; those without a CI run were taken for #89 with AKLR's code as of 43e112e, whose
bench timed calls with Kineto (AKLR's rocprofiler-sdk collector since #33).

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, python -m aklr validate --stages audit,bench on one GPU, a committed file (A) and a copy with one change (B) alternated, 4 runs per arm (3 for the aiter builds), median candidate time; an offline compile for gfx950 of gemm_n2624_k6144 b88c8d5f's m ≤ 16 launch at m = 1, 2, 8, 16] the compile: `num_stages=1` 16384 B of LDS, the committed 3 101312 B, `buffer_load … lds` 48, 96, 96, 48 at m = 1, 2, 8, 16 (2 warps at m = 1 and 16, 4 at m = 2 and 8); Qwen3.5 GDN decode 299133886d479017 at 2 tokens and 4 slots, recurrence with buffer `.cs` (A) against pointer `.cs` (B): 8.79 against 8.81 µs as the mean of each run's median call time, 8.827 against 8.813 µs as the median (B/A 0.9985); gemm_n6144_k2048 f806ad2b on the image's aiter (e7d2453f2) against the pinned dbd8bf5: baseline 7.02 against 7.95 µs at m = 8 (13.3%), 7.50 against 7.16 µs at m = 16 (4.5%), 2.9%, 3.4%, 1.0%, 3.9% apart at m = 2, 4, 32, 64, the candidate within 0.1–1.7%; DSA 2ee5 (24 tokens) with KV splits 8 → 4: `build_failed` (CompilationError) in 4 of 4 runs; MiniMax MXFP4 6557 (1 token) with `gate_fold` 4 → 1: `candidate_crash` in 4 of 4; both committed files passed 4 of 4 (b88c8d5f28d86777, 299133886d479017, f806ad2b086d757d, 2ee56a6dbabc48a6, 655723f593921b08; #89)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (the toy MoE and phase launches N=4), Gluon kernels in CUDA graphs: a split reduction of 16 splits × 16384 fp32 outputs, 200 replays per arm, compared bitwise with a partial + merge launch pair; one stream writing a 16,777,216-float region, reading it last, then overwriting it, 20 eager calls and 20 replays; python -m aklr validate --stages audit,bench on one GPU, a committed file (A) and a copy with one change (B) alternated, 4 runs per arm, median candidate time; a Gluon stream reading a 1 GiB bf16 tensor, timed with CUDA events; small Triton kernels timed as a CUDA graph replayed back to back (wall time) and by AKLR's `bench_cudagraph` (Kineto, before #33)] fold with its counter zeroed earlier in the same graph: 200 of 200 replays equal with `acq_rel` and 200 of 200 with `relaxed`; without the zeroing node wrong from the second replay on, 199 of 199; replay time 76.73 µs (zero + fold) against 7.6 µs (partial + merge); the last reader saw the values from before the overwrite in 20 of 20 eager calls and 20 of 20 replays; gemm_n2624_k6144 b88c8d5f with `num_stages` 3 → 1 on the m ≤ 16 launch, A against B at m = 2, 4, 8: 7.97, 8.11, 8.35 against 18.34, 18.43, 18.49 µs; 8.03, 8.13, 8.36 against 18.45, 18.53, 18.61 µs; 8.00, 8.08, 8.34 against 18.33, 18.43, 18.49 µs; speedup B/A 0.433–0.453 (2.21–2.30× the time), the unchanged m = 64 launch 1.003, 1.000, 0.9948, both arms accurate in every run; the stream at 96 programs × 4 warps with one 16-byte load per lane in flight (0.375 MB) against eight (3 MB): 885.2 against 3995, 876 against 3945, 885.9 against 3985 GB/s; a toy decode MoE (32 experts, top-4, hidden 256, intermediate 128, 2 tokens) recomputing top-k in its 40 consumer programs against a selection launch, same output (relative L2 0): 9.67 / 9.69 against 11.41 / 12.76, 9.89 / 9.52 against 11.13 / 12.75, 9.67 / 9.78 against 11.47 / 12.68, 9.875 / 9.793 against 11.621 / 13.193 µs (replayed / AKLR's timer); removing one of two elementwise phase launches saved 1.15, 1.87, 1.79; 1.61, 1.88, 1.80; 1.84, 1.74, 1.76; −0.102, 1.587, 0.325 µs of replay time at 16384, 131072, 1048576 elements, outputs equal, and 4.19, 4.19, 4.17, 4.17 µs on AKLR's timer at 131072 elements (gemm_n2624_k6144 b88c8d5f28d86777 and probe kernels; #89)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, launches of small Gluon and plain-Triton probe kernels with the JIT cache inspected; offline compiles for gfx950 compared by text] 12 specialization and option combinations gave 12 binaries in each frontend; 9 floats (0, 1, −1, 16, 1e30 among them) with True and False used one binary per frontend, and 15 more floats, inf and NaN included, added none (one run); reassigning a `gl.constexpr` global the kernel reads (4 → 5) or an allowed plain global (3 → 4) raised `RuntimeError` at the next launch, the same value or an unread global did not; a launched copy kernel built with 9 option variants got 9 cache entries with the same code; 36 kernel configurations from 8 Gluon families, each built with the 9 variants, kept identical instructions across them; offline compiles matched the launch's amdgcn in 12 of 12 comparisons (one run); `S` changed the code of 3 plain-Triton probes only with AMD buffer passes on, and of 3 Gluon copy and gather probes neither way (probe kernels and committed Gluon kernels; #89)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, direct calls of `run` checked against the contract's reference, 3 seeds × 3 slot patterns × 2 output poisons; offline compiles for gfx950 with `DS` attributes] Qwen3.5 GDN decode 299133886d479017 at 2 tokens and 4 slots (16 V heads, K 128, V 128, 3072 channels; recurrence BK 128, BV 16, ROW_GROUP 16, LARGE_STATE_POOL False, 2 warps, `waves_per_eu` 1; norm BLOCK 128, HISTORY_BLOCK 128, 1 warp) with its recurrence on buffer `.cs` and on pointer `.cs` accesses: seeds 39107, 51749, 82021 × slots (3, 0), (1, −1), (−1, 2) × NaN or 17 in the outputs before the call: 36 calls, 144 output comparisons within the contract, 18 repeat pairs bitwise equal; the recurrence compiled to 4 buffer loads, 4 buffer stores, 1 global load and no global store on the buffer arm against 0, 0, 5 and 4 on the pointer arm, 1024 B of LDS on both (299133886d479017; #89)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the sum of the two target kernels' device durations per call, each call after restoring the state and a 1 GiB read flush, neither counted; 32 balanced A/B profiler rounds per input, each 10 replays of 20 calls; 8 A/A pairs] the recurrence with buffer `.cs` (A) against pointer `.cs` (B) at 3 seeds × 3 slot patterns: within ±1% for 4 of 9 inputs, B slower by more than 1% for 3 (B/A 1.02797, 1.01344, 1.01179), 2 straddling the boundary; every A/A interval contained 1; the contract checks after the timing passed (Qwen3.5 GDN decode 299133886d479017, 2 tokens, 4 slots; #89)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, python -m aklr validate --stages audit,bench on one GPU: `num_stages` A/B 4 runs per arm, the skeleton against f806ad2b086d757d and the skeleton with one-step K prefetch 3 runs per arm in a fixed order; AKLR's fp8 GEMMs against aiter's plain-Triton kernel in 5 flushed graph rounds in one process; the skeleton's direct calls against the contract's reference, 20 eager calls, 20 replays and a second process per input] fewer stages: gemm_n6144_k2048 at m = 2, 4, 8, 16 1.533–1.625× slower, RMSNorm at 1024, 2048, 4096 tokens 1.047–1.120×, 1.092–1.095×, 1.051–1.052× slower; AKLR's unshuffled E4M3 fp8 GEMMs with fp32 block scales (N 3072, K 1536) at m = 2, 8, 16 0.777–0.781×, 0.755–0.764×, 0.759–0.768× of aiter's time, both within bounds; the skeleton at N 6144, K 2048 and M = 2, 3, 4, 8, 12, 16, 17, 32, 33, 48, 64 with one positive and one signed input each: bitwise stable, 0–1 bf16 ulp on the positive input, 0.960–0.984 of the bound on the signed one, 8192 B of `metadata.shared` passed at every launch while the amdgcn's group segment read 0; validated at m = 2, 4, 8, 16, 32, 64: no correctness, build or capture finding, all six Points demoted in each of 3 runs at speedup 0.373–0.434; 2.882–3.158× the time of f806ad2b086d757d at m = 2, 4, 8, 16, and 1.695–1.729× faster with one-step K prefetch (f806ad2b086d757d, cf637bff0784c454, gemm_a8w8_blockwise_scaled_n3072_k1536; #89)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR dispatch called with stand-in tensors on 6 contracts, the served file or fallback reason recorded] 633 storage probes on `gemm@1.0` and others: every tested violation (a non-contiguous `row_major` input, a negated one, one 2 bytes off alignment, a CPU call against a `cuda:0` table, an output aliasing an input) fell back `out_of_contract` before `run`, aligned controls and 85 non-overlapping controls served; 87 dtype, attr, axis, layout and scalar probes: changes no Point lists fell back before `run` (DeepSeek-V2 `routed_scaling_factor` 2.0 and 3.5 fell back, 2.5 served), while uint8 MoE `w1`, `w2` and their scales were served; GDN decode called with 4 and with 8 slots (`any_size`) served both times; substituting another GPU was not tested (gemm@1.0, deepseek_v2_mxfp4_moe@1.0, gdn_decode@1.0 and three more contracts; #89)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the two-phase grid N=4), launches of small Gluon probe kernels with the binaries counted in the JIT cache; python -m aklr validate --stages audit,bench of planted copies on one GPU, and direct calls against the contract's reference; two toy Triton elementwise phases of 131072 elements as two launches and as one grid split by program id, timed as a CUDA graph replayed back to back (wall time) and by AKLR's `bench_cudagraph` (Kineto, before #33)] `INSTR=[16, 16, 32]` as a constexpr failed at the launch with `unhashable type: 'list'` while `triton.compile` of the same kernel succeeded, and a tuple launched; scales 0.5, 1e-6, 3.0, 2^-20 with flags True and False: 1 binary; a 1 KiB view at offset 0 of a 4 KiB buffer got `tt.divisibility = 16` and `tt.pointer_range = 32`, at offset 8 only the range, a 2 GiB + 16 KiB tensor only the divisibility, 3 binaries, `is_within_2gb` called twice per launch of a two-tensor kernel, cache hits included; an RMSNorm copy of cf637bff0784c454 with `epsilon` hard-coded to 1e-5 passed audit and bench with no finding, and called with `epsilon` 1e-3 on 64 × 6144 inputs scaled by 0.01 had a maximum relative error of 2.235, the committed file 0; a copy of the fp8 GEMM f3970dec (m = 16) storing 8 of its 16 rows: `numeric_mismatch` at `first_capture/0` with 24576 = 8 × 3072 non-finite mismatches, demoted, the committed file no finding (one run); `_poison_outputs` filled uint8, int32 and fp4 outputs with 0xFF bytes, fp32 and bf16 with NaN; `torch.empty(17 << 30)` in `run`: `build_failed` at the Point, "peak 18259902464 or resident 0 bytes: too many", file-level `no_qualifying_workload`; a `run` keeping its first output in a module-level dict passed the audit and got `numeric_mismatch` at `first_capture/0`, `numeric/0`, `determinism/0` and `post_timing_mismatch` at `post_timing/0`; a `numeric_mismatch` planted at the 1024-token Point of a three-Point RMSNorm copy: demoted with `post_timing_mismatch` and `regressed`, "validation passed", exit 0; the same error in a one-Point file: `no_qualifying_workload`, exit 1; a math error: `numeric_mismatch` and `post_timing_mismatch`; a changed reduction order: the same in one run, only `regressed` in the other; state kept across calls: `post_timing_mismatch` only; no `nondeterministic` in any; the two phases as one grid 6.33 / 4.30, 5.66 / 4.38, 5.66 / 4.33, 9.191 / 4.443 µs (replayed / AKLR's timer) against two launches 7.42 / 8.49, 7.39 / 8.57, 7.43 / 8.50, 8.471 / 8.610 µs, replay ratio 0.854, 0.766, 0.762, 1.085, outputs equal in every run (RMSNorm cf637bff0784c454 and fp8 GEMM f3970dec copies, probe kernels; #89)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, CUDA-graph captures and JIT probes on one GPU; python -m aklr validate --stages audit,bench of an RMSNorm copy whose JIT key changed per call; an autotuner probe in 16 fresh processes] a constexpr warmed at 16 and launched at 32 inside a capture compiled there with one `_do_compile` call, and bench's tripwire raised "compiled in the capture"; the RMSNorm copy got no `capture_unsafe` finding, only `no_qualifying_workload`; two fresh processes sharing a warm `TRITON_CACHE_DIR` each called `_do_compile` once (first launches 0.5631 s and 0.001563 s); an autotuner with two equal-cost configurations timed both at its first call and neither afterwards in each of 16 processes, 8 picking the first and 8 the second; a `torch.empty` tensor allocated in a capture kept its address over 3 replays and held the third replay's values; `.item()`, `.cpu()`, `.tolist()`, `torch.cuda.synchronize()` and `torch.nonzero` failed the capture (RMSNorm cf637bff0784c454 copy and probe kernels; #89)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (input mutation and capture N=3), AKLR's source audit (CPU) and python -m aklr validate --stages audit,bench on one GPU of planted copies; `torch.cuda.graph` captures with `capture_error_mode="global"`] a def whose single decorator is a device decorator skipped the launch-code checks; `x.add_(1)` in `run` and `weight.add_(1)` in a helper taking the input's name failed SA2, and the same write through a helper parameter named `buf` passed the audit and failed `input_mutation` at `first_capture/0` on all 14 RMSNorm workloads; `os.environ[...]`, `os.environ.get`, `os.getenv`, `time.time`, `time.monotonic`, `time.monotonic_ns`, `time.process_time`, `time.perf_counter_ns` and `datetime.datetime.now` passed, `time.perf_counter`, `torch.cuda.Event` and `elapsed_time` failed SA3; in a `@functools.cache` function's own body `torch.empty`, `torch.tensor`, `.item()`, `triton.compile` and `.warmup` passed, in a def nested in it `torch.tensor` and `.item()` failed and the other three passed; `global` and `nonlocal` failed SA1, a module-level dict written in `run` passed; `.item()`, `.cpu()`, `.tolist()`, `torch.cuda.synchronize()`, `torch.nonzero`, `x[mask]`, `torch.argwhere` and `repeat_interleave` without `output_size` each failed the capture, two controls (`repeat_interleave` with `output_size` among them) captured (RMSNorm cf637bff0784c454 copies; #89)
- E11 [2026-11, gfx950 (offline), Triton 3.8.0, N=1, `triton.compile` for gfx950 without a GPU] 108 builds of row tiles: with a constexpr M covering an 8-, 16-, 32- or 64-row tile no mask instructions (3–56 with a runtime M), masks kept where M did not fill the tile; over M = 1–128 a `range(M)` loop compiled without a branch at M = 1–22 and with 1–2 at 23–128, using 17–74 VGPRs against 66 with a runtime M; of 11 committed Points compiled with a constexpr and with a runtime token count, not all became simpler (Gluon row-tile probes; #89)
- E12 [2026-11, gfx950 (offline), Triton 3.8.0, N=1, AKLR's source audit and schema parser on CPU, no GPU] `check_kernel` on planted files: allocation in a `@functools.cache` compile function's own body passed, `torch.tensor` and `.item()` in a def nested in it failed SA3; a schema with an added 37-token Point parsed as valid among its 15 workloads, and validate's queue built 15 path-backed requests with `base=None`, the added Point's among them; the build on a GPU was not run (planted copies of committed schemas; #89)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, trivial Gluon kernels (one 64-float load, add and store per program): replays of a bare CUDA graph of 1–64 chained kernels timed with CUDA events, least-squares slope; eager launches, 2000 × 5 rounds, host clock; AKLR's `bench_cudagraph` (Kineto, before #33); probe kernels reading past a view with `gl.amd.cdna4.buffer_load`] replay 1.815 µs per further kernel at grid 1 and 1.844 µs at grid 256, intercept 3.643 and 3.754 µs; eager host time 7.302 µs (grid 1) and 7.443 µs (grid 256), wall 7.311 and 7.452 µs; host time 6.249 µs + 1.004 µs per tensor + 0.1186 µs per int over 7.32 (1 tensor), 10.3, 14.23, 22.19, 38.45 µs (32 tensors), 15.24 and 17.12 µs (8 tensors with 8 or 24 ints); `get_tensor_specialization` 0.576 µs per call, 8 calls per 8-tensor launch; 8 launches of a 15-tensor kernel 168.8 µs of host time; 8 launches of 395.6 µs kernels 3166 µs wall against 3165 µs of kernels, of trivial kernels 171.9 µs wall against 171.7 µs of host time; graph replay 2.268 and 2.313 µs per launch with 1 and 30 tensors; an option dict built per call 7.744 against 7.686 µs built once; AKLR's bench: copy 6.498 + sum 4.62 µs apart, 10.84 µs together, flush 169.5 µs, 8 trivial kernels 32.42 µs against 3.879 µs each, median of 45 one-kernel readings 4.23 µs; on a 64-element view of a larger allocation, unmasked `buffer_load` offsets 64–79 returned 64.0–79.0, the allocation's next elements, and masked lanes without `other` returned 0; all 85 buffer ops of 27 compiled probe kernels have `num_records` `0x7ffffffe` (probe kernels; #89)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=?, Kineto trace of the calls at each Point, before #33] kernel launches per call over the 242 Points of the 154 Triton schemas: 1 in the fp8 GEMMs and the norms; 2 in gemm_n2624_k6144 at m = 64; 2–8 in the MoE plugins; 2–3 in DSA; 3 in gated attention; 1–2 in GDN decode; 5–8 in GDN prefill (the 154 Triton schemas at 43e112e; #89)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the self-resetting counter N=3), Gluon fold and merge kernels in CUDA graphs over 4–8 XCDs, replay time and AKLR's `bench_cudagraph` (Kineto, before #33)] 4, 16, 64 splits × 4096, 16384, 65536 fp32 outputs, 64 outputs per one-wave program: counters zeroed by an earlier node, 2000 of 2000 replays right with `acq_rel` and with `relaxed` at each size; without a reset, 1999 of 1999 later replays wrong; the merge launch 0.03–0.65× the replay time of the `acq_rel` fold and 0.51–0.92× of the `relaxed` one, 0.12–0.82× on AKLR's timer at up to 16 splits and 16384 outputs, each fold's time including its zeroing node; a counter the last arrival reset itself: 1000 of 1000 replays right in each of 3 runs (Gluon fold probes; #89)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, synthetic fp8 VALU MoEs and a decision sweep, CUDA-graph replay time and AKLR's `bench_cudagraph` (Kineto, before #33), arms rotated per round] two MoE layouts × 1, 2, 8, 16 tokens, 16 paired rounds per Point: recomputing the top-8-of-256 decision in every consumer slower on both rulers, the output bitwise equal to the baseline configuration's; 7 decisions × 2 or 16 tokens × 40, 256, 1024 or 4096 consumer programs = 56 configurations, 12 rounds each, the recomputing and publishing arms bitwise equal: top-8 of 256 with 4 or 16 split partials faster recomputed at 1024 programs and published at 4096; median extra-launch cost 3.2696 µs replayed and 4.4679 µs on AKLR's timer, which as a constant predicted the faster arm in 49 and 50 of the 56; identical-arm intervals excluded zero at 8 configurations replayed and 3 on AKLR's timer, not corrected for multiple comparisons (synthetic probe kernels; #89)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench, speedup against the schema's baseline] RMSNorm cf637bff0784c454 stored 1.023–1.029 at its 7 Points of 3–18 tokens and 1.040–1.214 at 64–4096 tokens, unchanged on main at 0e02a81 (aklr/ops/rms_norm/impls/triton/rms_norm_hidden_size6144/cf637bff0784c454.json; AKLR@d908d01, CI run 42000899019)
- E18 [2026-11, gfx950 (offline), Triton 3.7.0 and 3.8.0, N=1, `triton.compile` for gfx950 without a GPU, static instruction count of the whole kernel] a plain bf16 `tl.dot` K loop (64 × 64 × 64 tile, 4 warps, 2 stages, `schedule_hint="memory-bound-attention"`, a runtime trip count with `D`): 274 instructions, 16384 B of LDS and 12 `v_mfma_f32_32x32x16_bf16` on 3.7.0 against 405, 33968 B and 16 on 3.8.0; the hint changed 3.7.0's build (278 → 274 instructions, `iterative-ilp` added) and left 3.8.0's identical (probe kernel; #89)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, python -m aklr validate --stages audit,bench on one idle GPU, each file alone: the committed file (A) alternated with a copy whose first line carries a comment (A', a new uuid), 4 runs per arm, the spread taken over the first 3 runs of A; then three `validate --base HEAD` runs of A' of f3970dec; ratio = exp(log_ratio_mean) of the baseline entry, time = median candidate time] ratio spread (max − min) / min over 3 runs of A: 2.385% (fp8 GEMM n3072 f3970dec, m = 16), 0.911% (Qwen2-MoE router c3889614, 256 tokens), 1.731% (Qwen3.5 GDN decode 29913388, 2 tokens: 2.4222, 2.4105, 2.4522), 0.942% (DeepSeek-V2 MXFP4 MoE d3bbbce5, 2 tokens); f3970dec's A' under `--base` 1.4784, 1.4386, 1.4749 (2.77%); mean candidate time of A' over A 0.987, 1.002, 1.005, 0.997 in the order of the four (f3970dec, c3889614, 29913388, d3bbbce5; #89)
- E20 [2026-11, gfx950 (offline), Triton 3.7.0 and 3.8.0, N=1, `triton.compile` for gfx950 without a GPU] the 11 n3072 fp8 GEMM files from m = 16 up (m = 16, 32, 64, 128, 256, 1024, 2048, 4096, 6144, 8192, 8193) all compiled under 3.8.0 and all failed under 3.7.0: m = 16 and 64 because `shared_memory_descriptor._reinterpret()` requires `dtype`, m = 32 on an `AssertionError` in `core.py` `__getitem__`, m = 128 on its offsets layout, the other seven on the missing `compute_efficient_padded_shared_layout` (n3072_k1536 schemas before #64; #89)
- E21 [2026-11, gfx950 (offline), Triton 3.8.0, N=1, `triton.compile` for gfx950 without a GPU, amdgcn instruction count] a test kernel with a 4-load loop body: `gl.static_range` gave 120 instructions at 8 trips and 6513 at 512, `range` 120 at 8 and 129 at 512 (test kernel, not an AKLR schema; #89)
- S1 triton-lang/triton@c01b677:python/triton/experimental/gluon/_runtime.py:45 `GluonJITFunction(JITFunction)` swaps only the AST source; the module takes `ttg.num-warps` from the options (:32) and `ttg.threads-per-warp` from the warp size (:34); the verifier requires a layout's warp count to equal the module's (lib/Dialect/TritonGPU/IR/Dialect.cpp:3877, lanes :3863)
- S2 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:585 `add_stages`: Triton `make_ttir` + `make_ttgir`, Gluon `gluon_to_ttgir` (:327–349, no pipeliner, no buffer-op pass); `make_ttgir` coalesces (:269), reads `matrix_instr_nonkdim` and `kpack` (:273) and `num_stages` (:288–289), runs the buffer-op passes under `use_buffer_ops` (:303); `schedule_hint` "right now no effect" (:104); `AMDGCN_USE_BUFFER_OPS` defaults to on (python/triton/knobs.py:517)
- S3 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/ConvertToBufferOps.cpp:135 `canUseBufferOps`: a splatted base plus 32- or 64-bit offsets, safe by `tt.pointer_range` 32 on the argument (:112–130) or by a byte offset proven under 2 GiB
- S4 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:78 `HIPOptions.num_warps = 4`; warp size 64 (:114); a power of two (:116); `kpack` other than 1 overwritten with 1 on gfx950, with a deprecation warning (:119–123); `parse_options` raises `num_ctas > 1 not supported on gfx950` (:162–163; `supportsMultiCTALaunch` is false there, third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:240) and keeps only `HIPOptions` fields (:181); `gl.barrier()` synchronizes a CTA, `gl.barrier(cluster=True)` a cluster when `num_ctas > 1` (python/triton/experimental/gluon/language/_core.py:703); a kernel whose `metadata.shared` exceeds the device's `max_shared_mem` raises `OutOfResources` when its first launch loads it (python/triton/compiler/compiler.py:466)
- S5 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/010eab6660c4e415.json#initial_state.py:9 `_initialize_fresh` zeroes a slot when its `has_initial_state` flag is false, with a 4-warp `BlockedLayout` (:17), launched without `num_warps` (:24)
- S6 aklr/core/contract/locate.py:114 `schema_uuid`; the authored parts are `kernel` and each workload's axes and scalars, sorted (aklr/core/contract/op.py:884); `not at <ops>/...` when the path's tail differs (aklr/core/contract/op.py:959)
- S7 aklr/core/runtime/builder/builder.py:159 an in-memory package `_aklr_triton_<hash16>` (`TritonBuilder`, :193) whose `__path__` is the written sources (:160); the entry is imported as its submodule (:162) and must bind the call keys (:172)
- S8 triton==3.7.0+gitb4e20bbe.rocm7.14.0a20260608:python/triton/runtime/jit.py:697 in Triton 3.7.0 (that ROCm wheel), a launch keyword that is not an option raises `KeyError`; its `HIPOptions` has no `llvm_fn_attrs`; its Gluon has no `clamp`, `aggregate_replace`, `expect_zero`, `amd.slice`, `amd.cdna4.compute_efficient_padded_shared_layout` or `amd.cdna4.scaled_upcast`, all present in 3.8.0; `schedule_hint` `memory-bound-attention` sets `amdgpu-sched-strategy` `iterative-ilp` (backends/amd/compiler.py:393)
- S9 aklr/core/contract/registry.py:101 a schema whose text is not its `schema_text()` is rejected; the writer is `json_file_text` (aklr/core/contract/expr.py:108)
- S10 aklr/core/contract/fastkey/_fast_key_fn.py:284 the storage checks: stored layout, not negated, `data_ptr() % 16 == 0` (:40), device; output overlap at :239
- S11 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/7e9c13e0f2d58a3b.json#kernel.py:695 the entry; the launch plan in the docstring (:11–14), the launches at :621–692; `routed_scaling_factor` declared `values=(2.5,)` (aklr/plugins/sglang/models/deepseek_v2/contract.py:158)
- S12 aklr/core/validation/stages/_audit/_rules.py:57 SA0 (:57), SA1 (:65), SA2 (:87), SA3 (:154, its lists at :136–152), SA4 (:192); SA3's I/O names (:142) are rejected wherever a name reads them, device code included (:167)
- S13 aklr/core/validation/stages/_audit/_stage.py:134 device decorators `triton.jit` and `triton.experimental.gluon.jit`, recognized only as a def's single decorator (:392); `perf_edited` against the base (:90)
- S14 aklr/core/validation/stages/_bench/_evaluation/_candidate.py:267 the compile tripwire on `JITFunction._do_compile`, "compiled in the capture" (:505); the first calls, before any capture, use the default inputs and the workload's own scalars (:361–363); inputs rewritten through `.data.copy_` (:484); `capture_error_mode="global"` (:499) and outputs poisoned in the capture (:502, :570–582); read-only inputs byte-compared, `input_mutation` (:530–535); `MAX_FIRST_CALL_BYTES = 16 << 30` (:36), `build_failed` (:211)
- S15 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/fbe6ddab5dac5958.json#kernel.py:95 `_SOFTMAX = gl.constexpr(gl.BlockedLayout(...))` at module level
- S16 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:387 the globals a JIT body may read; `NameError` at :411
- S17 triton-lang/triton@c01b677:python/triton/runtime/jit.py:593 the key `(tuple(specialization), str(options))`, a constexpr entering it as `("constexpr", value)` (:409); caches per device (:738); an in-process miss calls `_do_compile` (:757), which reads the disk cache inside (:877); unknown launch keyword `KeyError` (:715); changed globals `RuntimeError` (:761); `ConstexprFunction.__call__` returns the raw value outside JIT, a constexpr inside (:1153); `Autotuner.run` times every config at a key's first call (python/triton/runtime/autotuner.py:240)
- S18 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:27 `_shape_config`, read by the kernel (:65) and by the host (:291, :294)
- S19 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:96 `_launch_for` returns the launch and its workspace size (:127, :158); `run` (:161); `_reduce_kernel` adds the six split-K partials in a fixed order (:87); n6144's returns the launch alone (aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:116, :196)
- S20 triton-lang/triton@c01b677:python/src/specialize.cc:274 an int equal to 1 becomes a constexpr, otherwise `i32` or `i64` by range (:281), `u64` above (:297), `D` for a multiple of 16 (:283); each tensor's dtype goes through Python's `canonicalize_ptr_dtype` (:329), whose table has no `float4_e2m1fn_x2` or `float8_e8m0fnu` (python/triton/_utils.py:77) and raises `KeyError` (:120); a Python callback per tensor without native tensor specialization (:342–370); bool `u1` (:377), float `fp32` (:385), `None` a constexpr (:522)
- S21 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:236 `get_tensor_specialization`: `D` by alignment (python/triton/backends/compiler.py:89), `S` by `is_within_2gb` on the storage (:211); HIP opts out of native tensor specialization (:140); `parse_attr`: `S` → `tt.pointer_range` 32 (:229), `D` → `tt.divisibility` 16 (python/triton/backends/compiler.py:75)
- S22 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/409cb22b39b3fa00.json#kernel.py:311 "Scale and epsilon are runtime arguments, never specialization keys"
- S23 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/82e9371086a875bd.json#kernel.py:59 `OFFSET: gl.constexpr`; one value per process stated at aklr/plugins/sglang/models/minimax_m2/contract.py:94
- S24 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/a1c4a2129730d1a0.json#kernel.py:702 one arena cut into 64 KiB-aligned typed regions (:723); 256-byte alignment at aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b0651fe84db2f670.json#kernel.py:1619
- S25 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/0101652e6ad7286d.json#kernel.py:365 program 0 of `_prepare_3` initializes `Counts`, `Jobs`, `Total`; the next launch, `_select_3`, takes atomic tickets on them through `_publish_routes` (:449, :507)
- S26 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b94f5777fd5d2d88.json#kernel.py:1026 one `workspace`: router logits, terms and quantized inputs at its start, the bf16 `down` over the same bytes (:1045)
- S27 AMD-AGI/AKLR@43e112e:aklr/core/utils/testing.py:152 before #33, `bench_cudagraph` captured calls behind a flush each and replayed them under Kineto; the per-call time summed kernel durations without the flush (:111–149)
- S28 aklr/plugins/sglang/models/moe_blocks.py:16 the MoE routes fall back inside SGLang's piecewise and breakable CUDA graphs
- S29 aklr/plugins/sglang/models/qwen3_next/api.py:145 prefill returns `OUT_OF_CONTRACT` while capturing; `op.fallback_reason(...)` before the projections (:164); decode "Graph capture too" (:176)
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
- S44 torch==2.12.0+rocm7.14.0a20260608:torch/_torch_docs.py:8242 `torch.nonzero` on a CUDA tensor "causes host-device synchronization", `torch.argwhere` too (:6861); `repeat_interleave`'s `output_size` "will avoid stream synchronization" (:13248); a bool index goes through `nonzero` (torch/_meta_registrations.py:3634); the same notes in the public docs, https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html, torch.argwhere.html, torch.repeat_interleave.html
- S45 aklr/core/contract/op.py:257 a layout variant may name only the contract's tensor inputs (:215), else "layout ... names unknown tensors" (:258), and "an output is never a 4-bit dtype" (:214); a tensor a variant does not list is `RowMajor` (aklr/core/contract/spec/_layouts.py:33); `per_group_quant_mxfp8@1.0` (aklr/ops/quant/contract.py:55) outputs E4M3FN `quantized` `[n_tokens, hidden_size]` (:38, reused at :68) and uint8 `scales`, "E8M0 exponent bytes at [row, group]" (:70), in its one `row_major` variant (:75); `per_group_quant_fp8@1.0` declares its fp32 scales `[n_groups, n_tokens]` (:46)
- S46 AMD-AGI/AKLR@0e02a81:aklr/core/utils/testing.py:164 `bench_cudagraph` since #33: as many calls as fit `rep` = 20 ms, each behind `flush_l2_cache()` and inside its own profiler scope (:202–210), the graph replayed 10 times (:212–217), a call's time the sum of its scope's kernel durations over the replays divided by 10 (:222–224), the median over calls returned (:245); AKLR's rocprofiler-sdk collector (`kernel_tracer`, :42) records the durations; `graph_kernel_timing_environment` sets `ROCP_TOOL_ATTACH=1` and unsets `GPU_FORCE_QUEUE_PROFILING` (:25–31)
- S47 aklr/core/validation/stages/_bench/_report.py:51 `not_faster` and `regressed` are measured as `passed` (:50–52); a workload qualifies its schema when it has no other finding and `beats_baseline` (:119–126), its graph ratio lower bound against the baseline above the entry threshold (:73–82), 1.02 (aklr/core/validation/stages/_measure.py:44–47)
