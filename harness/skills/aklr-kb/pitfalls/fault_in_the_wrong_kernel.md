---
title: "A GPU memory access fault names the kernel that faulted, not the one that wrote out of bounds"
kind: pitfall
scope: [runs of many kernels (a server, an e2e step) with FlyDSL kernels whose buffer descriptors are wider than their buffers]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "ROCm/FlyDSL@dd8373349:kernels/common/buffer_ops.py:338"
  - "ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:287"
  - "local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_171610/rounds/round-08/agent.txt"
---

# A fault in the wrong kernel

## Symptom

`Memory access fault by GPU node-<n>` in a run of many kernels, with a pattern that does not fit one faulty
kernel. In the case recorded, a GLM-5.2 server with four tensor-parallel ranks:

- all four ranks faulted at once, at page-aligned addresses that differed between ranks;
- the fault came after tens of decode batches (decode/prefill counts at the crash between 93/31 and 136/43) *(field, E1)*;
- it did not reproduce in isolation (eight M values, 257 and 258 experts, 300 mixed decode/prefill rounds, little
  free memory), and 4096-element canaries on nine buffers caught nothing *(field, E2)*;
- guard bands on suspect buffers and initialising a suspect table did not change it;
- with a host sync after every launch in the suspected MoE chain, every kernel there returned cleanly: the fault
  was outside it.

## Root cause

A buffer access lands past the end of its buffer, inside another live allocation, because the descriptor's bound
is larger than the buffer. The writing kernel exits cleanly; a later kernel that reads the corrupted value, for
example as an index, faults, and the fault names that later kernel.

- FlyDSL's `make_buffer_tensor` bounds nothing by default (`num_records = 0xFFFFFFFF`,
  `../flydsl/language/limits.md` § By design; `../gemm/flydsl_gemm_correctness.md` §6).
- The raw helper `create_buffer_resource(t, max_size=False)` falls back to the same 4 GiB bound, without a
  warning, when the memref's shape is dynamic. It is not in the wheel: FlyDSL's repository and the baseline
  aiter each carry a copy, and both do this (S1, S3). The wheel's `make_buffer_tensor(t, max_size=False)` derives
  the bound from the layout at run time instead (S2).
- In the recorded case a FlyDSL MoE sorting kernel wrote its row-inverse table 5 int32 past its end at capacities
  48 and 84 when rows inside the token count carried repeated expert ids, which is what padding a batch to the
  captured graph size leaves behind *(field, E3)*.

## How to detect

- **Attribute with a host sync, not with the order of launches.** After each launch outside graph capture, call
  `torch.cuda.synchronize()` and print an `ok` line with the launch's name; the suspect is the last launch with no
  `ok` after it. A sync that returns shows that every kernel launched before it finished without faulting. Skip the
  sync while `torch.cuda.is_current_stream_capturing()` is true: a sync inside a capture is an error.
- **Check that `AMD_SERIALIZE_KERNEL=3` serializes before reading its last launch as the culprit.** In the recorded
  environment it moved the crash later (decode batch 135 against 93) without lengthening the run *(field, E4)*.
- **When every kernel of the suspected chain returns `ok`, look for the write.** Put poison words after each
  buffer and check them after the kernel. Print how many buffers the probe checked: a probe that registered fewer
  buffers than intended reports a false clean. A canary next to the buffer misses an index that jumps far.
- **Give the probe the inputs the deployment makes.** Here only repeated expert ids within a token triggered the
  write. Top-k ids drawn by `torch.topk` on random scores do not repeat within a token, and a probe built on them read
  clean on 280 cases while the write was there *(field, E6)*; ids from `torch.randint` repeat, which reaches the
  trigger, and in that session also set off canaries on buffers unrelated to the fault.
- Remove the probes before timing anything: their host-side cost lands on every call
  (`../gemm/flydsl_gemm_correctness.md` §6 discusses one such reading).

## Fix or workaround

- Bound every access whose index comes from data with the buffer's real size in the descriptor
  (`num_records_bytes=`, which may be a runtime value), as `../gemm/flydsl_gemm_correctness.md` §6 describes.
  With the helper of S1 and a dynamic shape, `max_size=False` is not enough.
- Prove the bound works before trusting it: feed an input known to go out of bounds and see the write land before
  the change and get dropped after it. A bound you have not seen stop a write looks the same as no bound. In the
  recorded case the canary probe went from out of bounds on every repeated-id case to clean on 350 of 350, and two
  server runs went 342 decode batches each without a fault *(field, E3, E5)*.

## Scope

- Recorded once, in 2026-09, on MI355X: aiter's FlyDSL MoE sorting kernel in a GLM-5.2 TP4/EP4 sglang server; the
  FlyDSL version of that tree is not recorded. S1 and S2 are read from FlyDSL `v0.3.2` source, S3 from the
  baseline aiter.
- Applies to any store or load through a descriptor wider than its buffer, in any kernel of the run.

## Evidence

E1, E2, E4, E6 and the host-sync result in the Symptom are written down only in KB notes from the debugging
session (local:/workspace/code/gpt_oss_docker/.claude/memory/project_glm52_moe_rowinv_oob.md:72); E3 and E5 are
in the round's record.

- E1 [2026-09, gfx950, FlyDSL ?, N=7, GLM-5.2 TP4/EP4 server runs with the MoE campaign's kernels] decode/prefill counts at the fault 93/31, 100/33, 107/35, 114/33, 114/36, 121/35, 136/43 (KB note; local:/workspace/code/gpt_oss_docker/.claude/memory/project_glm52_moe_rowinv_oob.md:54)
- E2 [2026-09, gfx950, FlyDSL ?, N=?, isolated MoE runs] M = 128 to 7168 in eight values, 257 and 258 experts, 300 mixed rounds, 1.5 GiB free: no fault; 4096-element canaries on nine buffers: 0 out of bounds (KB note; local:/workspace/code/gpt_oss_docker/.claude/memory/project_glm52_moe_rowinv_oob.md:88)
- E3 [2026-09, gfx950, FlyDSL ?, N=350, canary probe over 350 cases] row-inverse table written 5 int32 past its end at capacity 48 and 84 with repeated expert ids inside the token count; out of bounds on 4 of 4 such cases before the descriptor bound, 350 of 350 clean after (GLM-5.2 MoE sorting, FlyDSL; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_171610/rounds/round-08/agent.txt:11; :13; :26)
- E4 [2026-09, gfx950, FlyDSL ?, N=1, server run with AMD_SERIALIZE_KERNEL=3] the fault came at decode batch 135 against 93 without it, and wall time did not grow (KB note; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_gpu_fault_attribution.md:32)
- E5 [2026-09, gfx950, FlyDSL ?, N=2, e2e server runs P/B/B/P with the fix] 0 memory access faults over 342 decode batches in each run (GLM-5.2 TP4/EP4 decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_171610/rounds/round-08/agent.txt:15)
- E6 [2026-09, gfx950, FlyDSL ?, N=280, canary probe with top-k ids from `torch.topk` on random scores] 280 of 280 cases clean while the out-of-bounds write was present (KB note; local:/workspace/code/gpt_oss_docker/.claude/memory/project_glm52_moe_rowinv_oob.md:41)
- S1 kernels/common/buffer_ops.py:343 at ROCm/FlyDSL dd8373349: with `max_size=False` and no `num_records_bytes`, a memref shape with a dynamic dimension gives `None` (:301) and the descriptor falls back to `0xFFFFFFFF` (:347)
- S2 python/flydsl/expr/rocdl/universal.py:287 at ROCm/FlyDSL dd8373349: `make_buffer_tensor(max_size=False)` sets `num_records_bytes` from `cosize(layout)` times the element size
- S3 aiter/ops/flydsl/kernels/buffer_ops.py:373 at ROCm/aiter dbd8bf5: the same fallback to `0xFFFFFFFF` when the static size cannot be read (:376)
