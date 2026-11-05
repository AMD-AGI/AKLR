# DeepSeek-V4-Pro AITER kernel imports

These implementations originate in two GEAK `kernel_workflow` campaigns:

- TP8/EP8: `/shared_nfs/chuschen/agent-kernel-arena/agentx-dsv4pro/exp_default/exp1016`.
- TP8: `/shared_nfs/chuschen/agent-kernel-arena/agentx-dsv4pro/exp_docker1019/exp1024`.

They are library implementations reached through AKLR's contract APIs. The
schemas contain their complete HIP sources and declare MI355X/gfx950 targets.
They retain the upstream MIT attribution and name GEAK and the porting agent
in `kernel.author`. No capture tensors, prebuilt code objects, measurement
fields, or original module-replacement machinery are installed.

## Source selection

The TP8 sources come from `state/<task>/best/kernel_src`, checked against the
hashes in `final/val_final1026/<task>/ut_result_run1.json`. The mutable `work/`
copies differ from those snapshots. The TP8/EP8 prefill source comes from
`final/snap_final/opus_mla_v4_sparse_prefill/kernel_src`, matching the selected
E2E overlay documented in `e2e/APPLYBACK_final.md`.

| Implementation | Campaign source | SHA256 of the principal source |
|---|---|---|
| Group-128 FP8 quantization | exp1024 `quant_g128_fp32/kernels/quant_kernels.cu` | `1d5016c051e0c6e0c8815b964d91bc891c8a3206c67a4ab97591d89bec59caaa` |
| Group-32 MXFP8 quantization | exp1024 `quant_mx_g32_e8m0/include/mxfp8_coalesced.h` | `805d4972ba74af2a2b4e649beb68e17c9a490372e19993e655e00a463057cdda` |
| mHC decode | exp1024 `mhc_fused_post_pre/kernels/mhc_kernels.cu` | `b41cb4b126ab95fcd9c412d49408e8e3d5a43c5f04d503b3e4f34bf25367e251` |
| CSR prefill, row order and tail split | exp1024 `mla_v4_prefill_opus/include/mla_v4_prefill_opus.h` | `211172154c4550a5eadfd53d730a124dccb3ca9a7e30731afc6d010c59868bb5` |
| CSR prefill, producer/consumer pipeline | exp1016 `opus_mla_v4_sparse_prefill/mla_v4_prefill_opus.h` | `4c03dda30e3391e886e867d737af29e9860ec791934b9c3b4428ff51be12494b` |

The prefill ports vendor the OPUS helpers retained by the exp1024 mHC capture:
`opus.hpp` SHA256 `fd9b9ae50d7fb92ba2e96910b77fa24f95ad6d1350a8abd0d51edb7b3d271cf1`
and `dtypes.hpp` SHA256 `8a40bd239520d390e6a708dfaaf65099166d4395487d6fc7bac240131c75e7c5`.
Only reached attention families are retained. Comments and disabled profiling
branches are removed, local include paths use `.h`, and `std::bit_cast` is
spelled with its equivalent compiler intrinsic for AKLR's C++17 build.

## Interfaces and port changes

- `aklr.ops.quant.per_group_quant_fp8`: dense BF16 rows, groups of 128,
  E4M3FN output and float32 scales. Scales have logical shape `[groups, rows]`;
  AITER's transposed raw scale buffer is viewed in that order without copying.
  The port preserves row-band traversal, the packed BF16 maximum, and the
  corrected reciprocal calculation. FP32 scale and FP8 rounding are explicit
  parts of the reference operation.
- `aklr.ops.quant.per_group_quant_mxfp8`: dense BF16 rows, groups of 32,
  E4M3FN output and row-major E8M0 exponent bytes. It preserves two-lane groups
  and the streaming block-size choice. The port uses FP32-to-FP8 conversion
  after exact power-of-two scaling: the original packed BF16 conversion did
  not reproduce the required rounding in the selected ROCm runtime.
- `aklr.ops.mhc.mhc_fused_post_pre`: the existing contract, with norm weights,
  hidden size 7168, and decode rows 1, 2, 4, 7, 8 and 16. The two reached device
  functions retain their arithmetic and intrinsic calls. Ordered launches and
  per-call torch scratch replace the original thread-local producer hint.
- `aklr.ops.mla_prefill.sparse_prefill_mxfp8`: two CSR ranges over separate
  prefix and extend pools, 128 heads, split NoPE/RoPE, and a denominator-only
  sink. A NoPE row stores 448 FP8 codes, fourteen E8M0 bytes, and padding to
  512 bytes. `Mxfp8NopeRowsDistribution` represents that existing packed format;
  the reference independently decodes it and computes attention in float64.
  The exp1024 tail split allocates counters and partials per call through torch.
  Its final chunk always performs the merge, fixing rescaling and accumulation
  order regardless of completion order. The selected path has no more tail
  blocks than compute units. The contract retains bitwise determinism.
  The exp1016 producer/consumer port fixes its original default mode and keeps
  only Points whose pool offsets fit uint32.

All outputs are supplied by AKLR. Inputs are read-only, launches use the current
stream, and kernels retain no tensor-dependent state across calls. The new
operators describe full dense quantization rows and contiguous split-precision
CSR operands; dynamic row-count and other storage variants remain with the
caller's implementation. Other shapes or targets follow AKLR's normal
`FallbackReason` path. This change does not install an AITER monkeypatch or
change an inference framework's public API.

## Workloads and qualification

Quantization shapes come from the exp1024 task case lists. mHC uses the existing
AKLR scenario's decode Points and captured scalars (including post multiplier
2.0), plus the source campaign's seven-row MTP verification shape. Both prefill implementations are tested on the
same four exp1024 capture shapes and CSR entry counts; capture metadata was
read on the meta device without loading its tensor values:

| Queries | Prefix pool rows | Extend pool rows | Prefix entries | Extend entries |
|---:|---:|---:|---:|---:|
| 2048 | 162436 | 16384 | 524288 | 262144 |
| 2048 | 4927942 | 16384 | 2359296 | 262144 |
| 344 | 162436 | 2699 | 220160 | 44032 |
| 344 | 4927942 | 2699 | 396288 | 44032 |

These are authored narrowings of contracts using recorded dimensions. They are
not an `AKLR_EXPORT` run. Every Point uses AKLR-generated operands and an
independent reference; the old reports' speedups and E2E results are provenance,
not qualification of these ports. Some original large-pool captures lacked
complete tensor values, which are not substituted for reference evidence.

Install the AITER revision and runtime image pinned by the repository's CI,
then run `python -m aklr export check` and
`python -m aklr validate --base origin/main --stages audit,bench` with the new
schema paths. The PR body records actual head validation and unmeasured checks.
Only CI writes accuracy and speedup fields into committed schemas; dispatch
continues to enforce `AKLR_MIN_SPEEDUP` per Point.

The additions introduce contract names and one input-distribution kind without
changing the meaning or version of existing schemas. They change Python code,
so the PR requires `ci:gpu`, full CI schema validation, and code-owner review.
End-to-end serving qualification of the AKLR ports is a separate release gate.

## Remaining source tasks

The user deferred multi-GPU integration. EP combine, MegaMoE prepare/stage1/
stage2 and reduce-scatter therefore remain outside this import: their shared
memory, IPC and rank protocols need a distributed contract and validation.
The exp1016 final stage1 experiment also hangs in serving; its deployable
interim snapshot would be the source for a later port, not the hanging variant.

The exp1024 all-reduce and all-gather tasks have no promoted `state/<task>/best`
snapshot. The full fused-MoE path still needs a dedicated self-contained port
of its combined FlyDSL/HIP implementation. The ASM decode result changes the
launcher around an existing code object and needs an auditable source/build
route rather than importing the binary. The indexer and C128 compressor tasks
target SGLang rather than AITER and are outside the requested AITER scope.
