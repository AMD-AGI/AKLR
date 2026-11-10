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
---

# A fault in the wrong kernel

## Symptom

`Memory access fault by GPU node-<n>` in a run of many kernels, with a pattern that does not fit one faulty
kernel. In the case recorded, a GLM-5.2 server with four tensor-parallel ranks:

- all four ranks faulted at once, at page-aligned addresses that differed between ranks;
- the fault came after 90–140 decode batches *(field, E1)*;
- it did not reproduce in isolation (eight shapes, 257 and 258 experts, 300 mixed rounds, memory pressure), and
  4096-element canaries caught nothing *(field, E2)*;
- guard bands on suspect buffers and initialising a suspect table did not change it;
- with a host sync after each launch, the launch that the order of launches had blamed returned cleanly: the fault
  was elsewhere.

## Root cause

A buffer access lands past the end of its buffer, inside another live allocation, because the descriptor's bound
is larger than the buffer. The writing kernel exits cleanly; a later kernel that reads the corrupted value, for
example as an index, faults, and the fault names that later kernel.

- FlyDSL's `make_buffer_tensor` bounds nothing by default (`num_records = 0xFFFFFFFF`,
  `../flydsl/language/limits.md` § By design; `../flydsl/gemm.md` § correctness §6).
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
  environment it moved the crash later without lengthening the run *(field, E4)*.
- **When every kernel of the suspected chain returns `ok`, look for the write.** Put poison words after each
  buffer and check them after the kernel. Print how many buffers the probe checked: a probe that registered fewer
  buffers than intended reports a false clean. A canary next to the buffer misses an index that jumps far.
- **Give the probe the inputs the deployment makes.** Here only repeated expert ids within a token triggered the
  write. Top-k ids drawn by `torch.topk` on random scores do not repeat within a token, so a probe built on them
  cannot reach that trigger; ids from `torch.randint` repeat, which reaches it.
- Remove the probes before timing anything: their host-side cost lands on every call
  (`../gemm/correctness.md` §6 discusses one such reading).

## Fix or workaround

- Bound every access whose index comes from data with the buffer's real size in the descriptor
  (`num_records_bytes=`, which may be a runtime value), as `../gemm/correctness.md` §6 describes; derive it from
  traced arguments (`../flydsl/gemm.md` § correctness §6).
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

E2, E4 and the rest of the Symptom are written down only in notes from debugging;
E1, E3 and E5 are
in the round's record.

- E1 [2026-09, gfx950, FlyDSL ?, N=?, GLM-5.2 TP4/EP4 server runs with the MoE campaign's kernels] before the fix the server crashed in a window of 90–140 decode batches (GLM-5.2 TP4/EP4 decode)
- E2 [2026-09, gfx950, FlyDSL ?, N=?, isolated MoE runs] eight shapes, 257 and 258 experts, 300 mixed rounds and memory pressure: no fault; 4096-element tail canaries could not catch the far index
- E3 [2026-09, gfx950, FlyDSL ?, N=350, canary probe over 350 cases] row-inverse table written 5 int32 past its end at capacity 48 and 84 with repeated expert ids inside the token count; out of bounds on 4 of 4 such cases before the descriptor bound, 350 of 350 clean after (GLM-5.2 MoE sorting, FlyDSL)
- E4 [2026-09, gfx950, FlyDSL ?, N=?, server run with AMD_SERIALIZE_KERNEL=3] the fault came later than without it, and wall time did not grow
- E5 [2026-09, gfx950, FlyDSL ?, N=2, e2e server runs P/B/B/P with the fix] 0 memory access faults over 342 decode batches in each run (GLM-5.2 TP4/EP4 decode)
- S1 kernels/common/buffer_ops.py:343 at ROCm/FlyDSL dd8373349: with `max_size=False` and no `num_records_bytes`, a memref shape with a dynamic dimension gives `None` (:301) and the descriptor falls back to `0xFFFFFFFF` (:347)
- S2 python/flydsl/expr/rocdl/universal.py:287 at ROCm/FlyDSL dd8373349: `make_buffer_tensor(max_size=False)` sets `num_records_bytes` from `cosize(layout)` times the element size
- S3 aiter/ops/flydsl/kernels/buffer_ops.py:373 at ROCm/aiter dbd8bf5: the same fallback to `0xFFFFFFFF` when the static size cannot be read (:376)
