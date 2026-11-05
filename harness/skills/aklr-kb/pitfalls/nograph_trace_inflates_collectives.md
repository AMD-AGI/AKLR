---
title: "A collective's trace time includes each GPU's wait for the others; with CUDA graphs off the wait is expected to grow"
kind: pitfall
scope: [per-kernel tables from multi-GPU traces, aiter's custom allreduce kernels and the allreduce fused with RMSNorm]
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
sources:
  - "ROCm/aiter@dbd8bf5:csrc/include/custom_all_reduce.cuh:161"
  - "ROCm/aiter@dbd8bf5:csrc/include/custom_all_reduce.cuh:1945"
---

# A collective's trace time includes the wait for the other GPUs

## Symptom

A per-kernel table from a multi-GPU decode trace puts a large share of the step on the allreduce
(`cross_device_reduce_1stage`, `allreduce_fusion_kernel_1stage` and the like). Expect the kernel's per-call times to
have a long tail: most calls near what the kernel takes on its own, a few far longer. Concluding from the summed
time that communication dominates, or that the allreduce kernel is slow, is not supported by the sum alone.

A compute kernel's long tail in the same trace is a different case: contention with a collective that overlaps it
(`../moe/grouped_measurement.md` §3).

## Root cause

- aiter's one-stage allreduce kernels open with a barrier across the GPUs and close with another. On entry each
  workgroup writes its flag to the workgroup of the same index on every GPU, then spins until each of theirs has
  arrived (S1); the closing barrier spins the same way (S2). The allreduce fused with RMSNorm does both (S3).
  `cross_device_reduce_2stage` opens with the same barrier and waits again between its stages; its write-mode
  variant waits twice with the closing barrier's code (S4). Each of them waits for every GPU at least once.
- So the kernel's duration on one GPU includes the time until the latest GPU reached it, with graphs or without:
  a lateness elsewhere is billed to the allreduce on every GPU that arrived earlier.
- Without CUDA graphs, each rank's process launches every kernel one by one, and the ranks drift apart by their
  launch jitter: the wait grows (inferred, not measured here). Per-kernel tables are often taken that way, because
  the profiler can miss what a graph launches (`profiler_misses_kernels.md`). With graphs on, the host jitter is
  gone, but a skew on the device side (uneven expert loads under EP, a slower kernel or clock on one GPU) is billed
  the same way.

## How to detect

- Read the distribution of the kernel's per-call times, not their sum. A median close to the same kernel timed
  alone at the same size, under a far longer tail, is waiting.
- Group the calls by message size (the op's `Input Dims`) before taking a median: a speculative-decoding step runs
  the same allreduce at the draft's and at the verify's token counts.
- With a trace of every rank, take each call's minimum over the ranks: the GPU that arrived last waited least.

## Fix or workaround

- Report a collective's cost as median × calls (per message size), beside the sum. When one rank is late at every
  call, the median holds waiting too, and median × calls is an upper bound; the standalone timing below is the
  check.
- With graphs off, most of the difference between the sum and median × calls is expected to be host launch jitter
  (inferred), which a deployment with graphs does not pay. Drop it only after a step timed with graphs on, or each
  call's minimum over the ranks, shows the skew is on the host side: skew on the device side is paid in the
  deployment too. Do not move it to another kernel.
- Before working on the allreduce, time it alone at the deployment's size and GPU count
  (`op_tests/multigpu_tests/test_custom_allreduce.py`, `test_fused_ar_rms.py` in the baseline aiter). Work on it
  only when that time, not the trace's sum, is large.

## Scope

aiter's custom allreduce kernels that wait for the other GPUs inside the kernel (S1–S4), the allreduce fused with
RMSNorm among them. An EP dispatch or combine kernel that polls other GPUs' flags has the same exposure;
`../communication/ep_dispatch_combine.md` covers what EP dispatch costs. RCCL's kernels, which large messages use,
were not read here: check their median against a standalone run the same way.

## Evidence

- S1 ROCm/aiter@dbd8bf5:csrc/include/custom_all_reduce.cuh:179 `start_sync` spins until every rank's flag for this workgroup index has arrived; each workgroup wrote its own to every rank at :174
- S2 ROCm/aiter@dbd8bf5:csrc/include/custom_all_reduce.cuh:231 `end_sync` spins the same way; `cross_device_reduce_1stage` calls `start_sync` at :405 and ends with `end_sync<ngpus, true>` at :480
- S3 ROCm/aiter@dbd8bf5:csrc/include/custom_all_reduce.cuh:1979 `allreduce_fusion_kernel_1stage` calls `start_sync`, and `end_sync<ngpus, true>` at :2059
- S4 ROCm/aiter@dbd8bf5:csrc/include/custom_all_reduce.cuh:518 `cross_device_reduce_2stage` calls `start_sync`, then `end_sync` between its reduce-scatter and its allgather at :554; `cross_device_reduce_2stage_write_mode` (:569) has no `start_sync` and calls `end_sync` at :618 and :672
