---
name: aklr-kb/triton
description: >
  AKLR Triton/Gluon on gfx950 with Triton 3.8.0: authoring, validation,
  layouts, memory, MFMA, numerics, debugging and operator design.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Triton and Gluon kernels for AKLR on gfx950

Use these pages to write and optimize kernels on MI355X with Triton 3.8.0.
Choose a page by the problem you are solving; copy mechanisms, not unexplained constants.
`aklr/` paths are relative to the repository root.

## By task

| Task | Read |
|---|---|
| Choose a kernel family and starting implementation | [Kernel library](kernels/kernel_library.md) §3; [Authoring](authoring.md) §10 |
| Entry ABI, scratch allocation and specialization keys | [Authoring](authoring.md) §3–§6 |
| Graph capture and validation | [Authoring](authoring.md) §8, §11; [Debugging](debugging.md) §5–§6 |
| IR, ISA, registers, LDS and spills | [Debugging](debugging.md) §1–§3 |
| Compile errors and wrong outputs | [Debugging](debugging.md) §4–§6; [Numerics](numerics.md) |
| Fair timing and per-Point configuration | [Tuning](tuning.md) §1–§3, §14 |
| Grid, prefetch, launch options and cache policy | [Tuning](tuning.md) §4–§12 |
| Layouts and conversions | [Layouts](gluon/layouts.md) |
| Buffer addressing, masks, DMA and LDS reuse | [Memory](gluon/memory.md) |
| Matrix instructions, operand formats and accumulators | [MFMA](gluon/mfma.md) |
| Inline assembly, lane operations and hazards | [Wave operations](gluon/wave_ops.md) |
| Quantization, rounding and determinism | [Numerics](numerics.md) |
| fp8 block-scaled GEMM | [GEMM](gemm.md) |
| Small-M bf16 GEMM | [Decode GEMM](gemm/decode_gemm.md) |
| Routing, expert GEMMs and combine | [MoE](moe.md) |
| Decode attention and split KV | [Attention](attention.md) |
| GatedDeltaNet decode and prefill | [GDN](gdn.md) |
| Cross-language mechanisms | [Knowledge base](../SKILL.md) |

## Authoring rules

- Follow the contract and entry ABI in `aklr/ops/README.md`. Use the schema's exact keyword-only
  `kernel_call_keys`, write preallocated outputs, and return nothing.
- Choose configuration from shapes and dtypes before capture. Warm every reachable specialization eagerly;
  a new key can compile inside capture. Keep ranged scalars and `any_size` axes runtime unless the contract
  and warmed variants justify specialization.
- Allocate scratch per call or cache it per shape in the compile function. Initialize every consumed region,
  counter and sentinel before its first read in each call; never depend on a previous invocation's contents.
- Do not synchronize, inspect tensor values on the host, or retain input-derived state in launch code.
  Passing an audit is not proof that a call is capture-safe or numerically correct.
- Pass packed MXFP4 and E8M0 storage to Triton as `.view(torch.uint8)` where the launch type requires it.
  Declare the required Triton version in the schema.
- Under a bitwise contract, use reproducible reductions. A fixed-order scratch merge is one option;
  arrival-order floating-point atomics can produce different results across runs.

## Gluon design rules

- Layouts define ownership and access patterns. Derive row/column indices from slices of the tensor's layout,
  and keep its warp count consistent with the launch. Convert explicitly when values must move between lanes
  or waves.
- Gluon does not add the plain-Triton automatic `num_stages` software pipeline. Build prefetch and buffer
  lifetimes explicitly; its warp-pipeline pass also supports explicit stage annotations.
- A buffer descriptor is not a logical tensor bound. Mask invalid lanes, keep byte offsets within descriptor
  range, and rebase the scalar pointer in 64-bit arithmetic before an address product can overflow.
- `gl.barrier()` is not an async-copy completion wait. Commit DMA groups and use `wait_group` before consuming
  their data. Before refilling LDS, ensure all readers of that slot have finished.
- Inspect LDS through `metadata.shared`, not just the assembly's static group segment. Check register and
  spill counts together; a smaller VGPR count achieved through spilling is not automatically an improvement.
- Preserve the reference's required rounding boundaries. Promote explicitly when FP32 arithmetic is needed;
  BF16 multiplied by a Python float can remain BF16.
- For masked E8M0 scale loads that require a neutral scale, use `other=127`, representing one.
- Declare inline-assembly operands and clobbers correctly. Read assembler diagnostics and the instruction's
  ISA hazard requirements; an opaque multi-instruction string needs its own internal scheduling care.

## Optimization and validation

- Compare each Point with the faster of its served kernel and baseline. Report both call times and the
  definition of the ratio; stored speedup includes a confidence-bound adjustment.
- Match the deployed timer, cache state, preprocessing and output contract. Do not compare a hot isolated
  kernel with a flushed whole-call baseline.
- Calibrate with identical-code A/A controls. Use paired uncertainty for small effects; a sample range is not
  the uncertainty of a mean difference.
- Treat prefetch depth, split count, tile shape, cache hints and program order as candidates, not universal
  settings. Equal nominal bytes in flight do not imply equal bandwidth.
- Measure fusion over the whole producer/consumer chain. Removing a launch can add register pressure,
  duplicated work or synchronization elsewhere.
- Inspect final IR/ISA when testing a compiler mechanism, then validate full outputs. A successful compile,
  a process exit code or a faster time alone does not establish correctness.
- Check the actual Triton version and required baseline dependencies. Re-time on the CI image rather than
  assuming a configuration carries across compiler builds.

## References

- `aklr/ops/README.md`: kernel ABI, contract semantics and validation workflow.
- `harness/skills/aklr-read-validation/references/ruler.md`: AKLR measurement conventions.
- [Triton 3.8.0 source](https://github.com/triton-lang/triton/tree/v3.8.0): language and AMD backend implementation.
- [gfx950 ISA guide](../hardware/gfx950/INDEX.md): instruction semantics, resource limits and synchronization.
