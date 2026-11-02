---
title: "A split-K fold in which no workgroup waits for another: the last split to arrive folds"
kind: technique
lever: wait_free_split_k_fold
ops: [gemm]
dtypes: [bf16]
regimes: [small-M, split-k]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=4
sources:
  - "local:/shared_nfs/kyle/moe/k3six/runs/gemm_s_fixwork/workspace/kernel.py"
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k16384/6bf403875a978f7e.json
---

# Wait-free split-K fold

## What it does

A one-launch split-K GEMM has to add the splits' partial tiles. A fold in which an owner split polls flags, or
launch-tagged data, until its partners have published is right only while every workgroup of the launch is
resident at once: when fewer CUs are free (a stream limited to part of the GPU, another kernel holding CUs), the
waiting workgroups hold the slots their partners need, and the fold gives up or hangs. Here no workgroup ever waits.
Each split publishes its partial, counts itself in on an atomic counter for its tile, and the split that finds all
the others already counted does the whole fold; the others exit.

## How to write it

In the kernel above (line numbers of `kernel.py`):

1. Each split stores its packed partial into its own plane with `sc0 sc1` (`_SC01 = 17`, line 50), which bypasses
   the non-coherent caches so the data meets across XCDs (S1).
2. `rocdl.s_waitcnt(vmcnt=0)` and a barrier: the stores are acknowledged before the split counts in (line 453, S2).
3. Thread 0 adds 1 to the tile's counter, `llvm.atomicrmw(add, …, syncscope="agent")`, and passes the old count to
   its workgroup through LDS (line 457, S3).
4. The split that read `SPLIT - 1` folds: it loads every plane, its own included, with `sc0 sc1` and adds them in
   split order (line 467, S4). Its own partial goes through the same packing as the published ones, so the sum is
   bitwise the same whichever split folds.
5. No host reset: the counter's high word holds a per-launch tag, `llvm.amdgcn.dispatch.id` XOR
   `llvm.amdgcn.queue.ptr` (`_launch_tag`, line 139). A split that finds another launch's tag claims the counter
   with one `cmpxchg`, which fails only when a split of this launch already did (lines 435-441, S5). The counters
   and planes sit in the call's own workspace, so two calls running at once share nothing.

## When it applies

- Any FlyDSL kernel whose workgroups combine results through global memory: split-K GEMM folds, split-KV attention
  merges, cross-workgroup reductions.
- The result stays bitwise reproducible if the fold adds in a fixed order (split index), never in arrival order.

## Counter-signals

- A fold that waits is faster where its work spreads over all splits: the owner-per-slice spin fold beat this one
  by 15.6% at m=128 (8 splits, one workgroup now reads 7 planes) and by 22% at m=256 on a larger tile
  *(field, E1)*. That speed is not available safely; take it from fewer splits or a cheaper partial format instead.
- A pairwise tree in fixed order does not help: each level waits for its own stores to be acknowledged, and the
  splits of a tile finish close together, so the last split pays every level. It was no faster than the flat
  fold at m=128 and 1.7-2.9% slower at m=32, 64, 256 and 512 *(field, E2)*.

## How to confirm it took effect

- `final_isa.s` has no loop around the flag or plane loads, and no `s_sleep`.
- On streams limited to 16 and 64 CUs (`hipExtStreamCreateWithCUMask`), and beside a second call of its own on
  another stream, every workload's output is bit for bit its output alone; forge_task's `driver.py --stress` runs
  this. A fold that waits failed it with millions of NaN at m=256 and 512 *(field, E3)*.

## Cost and risk

- The fold's whole read goes through one workgroup: the cost grows with the split count (E1).
- The counter's tag is 32 bits of the dispatch id and queue address; two launches with the same tag in one
  workspace would merge counts, which per-call workspaces rule out.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=4, AKLR graph timer (driver --bench-mode), interleaved] owner-per-slice spin fold → wait-free flat fold: gemm_n6144_k16384 m=128 37.95 → 43.90 µs, m=64 32.54 → 33.19 µs, m≤32 within 0.5%; m=256 56.2 → 68.6 µs, m=512 87.1 → 94.3 µs (local:/shared_nfs/kyle/moe/k3six/fix/gemm_s_tree.md:15, local:/shared_nfs/kyle/moe/k3six/fix/gemm_l_tree.md:17)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=4, AKLR graph timer (driver --bench-mode), interleaved] fixed-order pairwise tree vs flat fold: m=128 43.79 vs 43.90 µs, m=64 33.94 vs 33.19, m=32 33.04 vs 32.10, m=256 70.2 vs 68.6, m=512 95.9 vs 94.3 (local:/shared_nfs/kyle/moe/k3six/fix/gemm_s_tree.md:15, local:/shared_nfs/kyle/moe/k3six/fix/gemm_l_tree.md:17)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=3, eager on a CU-masked stream, output against the call alone] spin fold at m=256: 3,151,872 NaN on 64 CUs, 2,340,864 on 128 CUs, 35.5 and 12.4 ms per call; m=512: 4,276,224 and 3,158,016 NaN; the wait-free fold: none, bit for bit equal (local:/shared_nfs/kyle/moe/k3six/tasks2/step53b.out:12, local:/shared_nfs/kyle/moe/k3six/tasks2/step73.out:9)
- S1 local:/shared_nfs/kyle/moe/k3six/runs/gemm_s_fixwork/workspace/kernel.py:50 `_SC01 = 17`, system-scope access past the non-coherent caches
- S2 local:/shared_nfs/kyle/moe/k3six/runs/gemm_s_fixwork/workspace/kernel.py:453 `s_waitcnt vmcnt(0)` before the arrival
- S3 local:/shared_nfs/kyle/moe/k3six/runs/gemm_s_fixwork/workspace/kernel.py:457 the agent-scope `atomicrmw add`
- S4 local:/shared_nfs/kyle/moe/k3six/runs/gemm_s_fixwork/workspace/kernel.py:467 the last split folds in split order
- S5 local:/shared_nfs/kyle/moe/k3six/runs/gemm_s_fixwork/workspace/kernel.py:139 `_launch_tag`, and 435-441 the claiming `cmpxchg`
