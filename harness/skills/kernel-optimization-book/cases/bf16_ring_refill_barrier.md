---
title: "bf16 GEMM, N 4096 × K 8192 at m 1–8 and its twin N 8192 × K 4096 at m 1–16 — whole-K column strips whose waves read one LDS ring that DMA refills; the ring barrier waits `lgkmcnt(0)` as well as `vmcnt`, or the refill overwrites a slot still being read"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-8 (N 4096 × K 8192)", "M 1-16 (N 8192 × K 4096)"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=2000
measured:
  - {metric: "calls with a wrong output, and calls whose two back-to-back runs of one input differ in bits, of 2000 beside a second process running matmuls on the same GPU", value: "m 2: 10 wrong and 26 differing with `vmcnt`-only waits, 0 and 0 with `lgkmcnt(0)` added; m 1 and 4: 0 and 0 in both builds; m 8: 0 and 0 with `lgkmcnt(0)`", gpu: MI355X, workload: "N 4096 × K 8192, m 1, 2, 4, 8, eager calls with a fresh `a` each call", date: "2026-11"}
---

# bf16 GEMM at small m, N 4096 × K 8192 and N 8192 × K 4096: the ring barrier waits for the LDS reads too

Two FlyDSL kernels for `out = a @ b.T` on a 64 MiB weight; their sources differ only in N, K and the per-m table.
"m" is M. At m ≤ 8 (N 4096 × K 8192) and m ≤ 16 (N 8192 × K 4096) a workgroup owns 8 or 16 output columns over the
whole K, and all its waves read one LDS ring that DMA refills one K stage at a time, behind a barrier. This page is
about the wait in front of that barrier. [gemm/datapath](../gemm/datapath.md) §4 states the rule and has the
measurement, `../gemm/correctness.md` §4 lists it among the rules for waits and barriers, and
`../flydsl/gemm.md` § datapath §4 says how the wait is written in FlyDSL.

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`, with
  (N, K) = (4096, 8192) or (8192, 4096).
- m is on the ring builder's rows of the kernel's per-m table: m ≤ 8 for N 4096 × K 8192, m ≤ 16 for
  N 8192 × K 4096. Of the sizes the kernels are written for, m 1, 4 and 8 are on those rows in the first and m 2, 8
  and 16 in the second; their other sizes, m 72 and up, run tile builders that this page does not describe.
- More widely: any loop in which the waves of a workgroup read a ring slot and a DMA issued after the next barrier
  refills that slot (`../gemm/datapath.md` §4).
- Both kernels are held to `bitwise` accuracy: two runs on the same inputs must give the same bytes. They use no
  atomics, and the waves' partial sums are added in wave order.
- The book gives no time and no speedup for this path; what it measured is whether the output is right (§ Result).

## Counter-signals

- **A check that runs the kernel by itself passes the broken wait.** With `vmcnt`-only waits, 300 calls each at
  m 1, 2, 4 and 8, run alone, were all right; the wrong outputs came beside a second process on the same GPU
  *(field, E1)*.
- **One row failed, and the book does not say why that one.** Beside the second process the `vmcnt`-only build was
  wrong at m 2 and gave 0 wrong and 0 differing outputs of 2000 at m 1 and at m 4; m 8 is reported beside it only
  with the wait added *(field, E1)*. In that kernel's table the m 2 row differs from the m 1 and
  m 4 rows, which did not fail, in several things at once: a 3-slot ring against 8, two waves against four, four
  32-K slices per wave and stage against two and one, and six DMAs per thread and stage against three and one. The
  m 8 row, which has the m 2 row's waves, slices and DMAs on 8 slots, is the one reported only with the wait added
  (§ Mechanism). The measurement separates none of these.
- **The measurement is on N 4096 × K 8192; the code shown is the twin's.** The loop under § Code is from
  N 8192 × K 4096, whose ring builder is the same text; the book reports no run of the twin beside a second process.
  m 2, the size that failed, has a row in the N 4096 × K 8192 table but is not one of the sizes that kernel is
  written for.
- **A deeper ring does not replace the wait here.** `../gemm/datapath.md` §4 gives a second way to be right: one
  spare ring slot between the slot read and the slot refilled. This loop refills the slot read one stage earlier at
  every ring depth it is built with (3, 4 or 8 slots), so it has no spare slot and rests on the wait.
- **What the wait costs is not measured on these kernels.** On another bf16 GEMM in FlyDSL (N 6144 × K 1536, LDS
  DMA rings), a bare `s_barrier` with one spare ring slot was 0.7% faster at m 128 and within 0.2% at m 16, 32 and 64
  than `gpu.barrier()`, which in that build's ISA waited on outstanding memory operations, `lgkmcnt(0)` among them,
  before each loop `s_barrier` *(field, E2, E3)*. In the same probe, at m 128, `lgkmcnt(0)` followed by a bare
  `s_barrier`, the arm nearest to this page's wait, read 8.096 µs against 7.935 for a bare barrier without the spare
  slot, which does not cover the hazard, one number each *(field, E2)*; the spare-slot build was not kept.

## Mechanism

The ring builder, the same text in both kernels:

- **Column strips over the whole K.** A workgroup owns `ROWS` rows of B (8 or 16), which are `ROWS` output columns,
  and all m rows of A: N / `ROWS` workgroups of two or four waves, with no K split across workgroups and no
  workspace.
- **A K stage fills one ring slot.** A stage is `BK` = 32 · `U` · waves elements of K (128 or 256). Its bytes, the
  workgroup's B rows and then m rounded up to 8 rows of A, go into one slot of the LDS ring as XOR-swizzled 16 B
  chunks, by `buffer_load_dwordx4 … lds` DMAs, B with `sc0 nt` and A at the default cache policy
  (`../gemm/datapath.md` §2, §5). Every wave issues all `PASSES` DMAs of a stage ("so vmcnt counts agree across
  waves", the code's comment); an A row past m is sent to an offset past the buffer descriptor.
- **`issue(st)`** sends the DMAs of stage `st` to slot `st % NS`, `NS` being the ring depth. The K it reads is stage
  `(rot + st) % STAGES`, where `rot` is the workgroup index times the row's K-start step, so that workgroups start
  their K walks at different stages (no step in the m 2 row of N 8192 × K 4096).
- **`frag(slot, chunk0, row, u)`** is a 16 B `ds_read` of a slot: 8 bf16 of one row (a B row at `chunk0` 0, an A row
  at `chunk0` `ROWS * KV`) from the wave's `u`-th 32-K slice of the stage. A wave takes `U` slices, disjoint from the
  other waves', so every wave of the workgroup reads every slot.
- **The loop** over the K stages is unrolled when the kernel is built (`range_constexpr`). Stages 0 to `NS − 2` are
  issued before it. Stage `st` waits until at most `NS − 2` stages of DMAs are in flight and the wave's own LDS reads
  have landed (`s_waitcnt vmcnt(vm) lgkmcnt(0)`), meets the other waves at `s_barrier`, issues stage `st + NS − 1`
  into the slot that stage `st − 1` read, then reads slot `st % NS` and runs `mfma_f32_16x16x32_bf16`, A first, into
  one fp32 accumulator per 16 rows of A.
- **Fold.** After the loop each wave writes its accumulators to an LDS scratch; after a `gpu.barrier()` the threads
  add them over the waves in wave order, round once to bf16 and store. Columns past the strip and rows past m go to
  an offset past the buffer descriptor, where the store is dropped.

The rows of the two tables; a row covers every m above the row before it, up to its own:

| Kernel | m | Waves | K per stage (32-K slices per wave) | K stages | Ring slots | Columns per workgroup (workgroups) | DMAs per thread and stage | K-start step |
|---|---|---|---|---|---|---|---|---|
| N 4096 × K 8192 | 1 | 4 | 256 (2) | 32 | 8 | 16 (256) | 3 | 1 |
| | 2 | 2 | 256 (4) | 32 | 3 | 16 (256) | 6 | 1 |
| | 4 | 4 | 128 (1) | 64 | 8 | 8 (512) | 1 | 3 |
| | 8 | 2 | 256 (4) | 32 | 8 | 16 (256) | 6 | 1 |
| N 8192 × K 4096 | 1 | 4 | 128 (1) | 32 | 8 | 16 (512) | 2 | 3 |
| | 2 | 4 | 256 (2) | 16 | 3 | 8 (1024) | 2 | 0 |
| | 4 | 4 | 128 (1) | 32 | 8 | 16 (512) | 2 | 3 |
| | 8 | 4 | 128 (1) | 32 | 4 | 16 (512) | 2 | 3 |
| | 16 | 4 | 128 (1) | 32 | 8 | 16 (512) | 2 | 1 |

The source has two more builders with a ring and a barrier each: the tile builder of the larger sizes, and one that
no row of these two tables selects. All three barriers wait `vmcnt(N) lgkmcnt(0)`.

## Code

### 1. Wait for the DMA and for the waves' LDS reads before the ring's barrier (N 8192 × K 4096, m ≤ 16)

The ring loop of the N 8192 × K 4096 kernel, which covers every m up to 16; the loop of the N 4096 × K 8192 kernel
is the same text. `issue(st)` sends the DMAs that fill ring slot `st % NS`, and `frag` is a `ds_read` of a slot;
`NS` is the ring depth, `STAGES` the number of K stages, `PASSES` the DMAs a thread issues per stage, `U` the 32-K
slices a wave reads per stage, `MF` the 16-row groups of A, `ROWS` the B rows a workgroup owns, `KV` the 16 B chunks
of one row in a stage (an eighth of the stage's K elements), and `rb_rd` and `ra_rd[i]` the lane's B row and A rows.
`vmcnt` counts the DMA, and the waves' own `ds_read`s of a slot count in `lgkmcnt`: after `s_waitcnt vmcnt(N)` and
`s_barrier` alone a wave can still have them in flight when the refill is issued, and the DMA overwrites what they
read. The wait therefore names both counters, in one inline-asm string with the barrier.

```python
for st in range_constexpr(NS - 1):
    issue(st)
for st in range_constexpr(STAGES):
    vm = min(NS - 2, STAGES - 1 - st) * PASSES
    llvm.InlineAsmOp(
        None, [], f"s_waitcnt vmcnt({vm}) lgkmcnt(0)\n\ts_barrier", "", has_side_effects=True
    )
    if const_expr(st + NS - 1 < STAGES):
        issue(st + NS - 1)  # refills the slot the waves read one stage earlier
    for u in range_constexpr(U):
        bf = frag(st % NS, 0, rb_rd, u)
        afs = [frag(st % NS, ROWS * KV, ra_rd[i], u) for i in range_constexpr(MF)]
        for i in range_constexpr(MF):
            acc[i] = rocdl.mfma_f32_16x16x32_bf16(
                T.vec(4, T.f32), [afs[i], bf, acc[i], 0, 0, 0]
            )
```

Not this: `f"s_waitcnt vmcnt({vm})\n\ts_barrier"`. The `ds_read`s of the stage before can still be in flight when
`issue` refills their slot: with that wait the N 4096 × K 8192 kernel gave wrong output at m 2 in 10 of 2000 calls
(relative error up to 8.7e-2) and different bits for the same input in 26, while another process ran matmuls on the
same GPU *(field, E1)*.

Explained in [gemm/datapath](../gemm/datapath.md) §4.

## Result

One measurement, on the N 4096 × K 8192 kernel on one MI355X with FlyDSL 0.3.2. The calls were eager, with a fresh
`a` each call, and every input was run twice back to back. An output counts as wrong when it is non-finite or its
relative max error against fp32 `a @ bᵀ` is above 2e-2, and as differing when the two runs of its input gave
different bits. The second process ran back-to-back bf16 8192 × 8192 matmuls on its own stream on the same GPU
*(field, E1)*.

| m | Waits before the barrier *(field, E1)* | Beside the second process: wrong, of 2000 | Differing, of 2000 | Alone: wrong, of 300 |
|---|---|---|---|---|
| 1 | `vmcnt` only | 0 | 0 | 0 |
| 2 | `vmcnt` only | 10 (worst relative error 8.7e-2) | 26 | 0 |
| 4 | `vmcnt` only | 0 | 0 | 0 |
| 8 | `vmcnt` only | not reported | not reported | 0 |
| 1, 2, 4, 8 | `vmcnt` and `lgkmcnt(0)` | 0 | 0 | not reported |

No page of the book gives a time for this path, with or without the wait.

## Evidence

- E1 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=2000 calls per m and build under contention, 300 per m alone; eager calls, a fresh A each call, each input run twice back to back and compared bit for bit, the output against fp32 `a @ bᵀ` (wrong: non-finite or relative max error > 2e-2); contention: a second process running back-to-back bf16 8192 × 8192 matmuls on its own stream on the same GPU] the N 4096 × K 8192 kernel under contention, m=2: with `vmcnt`-only waits 10 of 2000 wrong (worst relative error 8.7e-2) and 26 of 2000 with different bits; with `lgkmcnt(0)` added to the waits of both ring barriers, 0 and 0; m=1 and 4: 0 and 0 in both builds; m=8 with `lgkmcnt(0)`: 0 and 0; alone, with `vmcnt`-only waits, m 1, 2, 4, 8: 0 wrong of 300 each (bf16 GEMM n4096 k8192)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=3, a probe on the per-kernel CUDA-graph timer, the canonical kernel timed alongside in one process in alternating order over 4–6 rounds, medians; then the run's bench harness twice] loop barrier as a bare `s_barrier` with one spare ring slot per operand (rb1s1) against `gpu.barrier()`: m=128 7.945 / 7.908 / 7.909 against 8.007 / 7.955 / 7.962 µs, second bench 7.937 / 7.933 against 8.002 / 7.997; m=64 6.423 / 6.429 against 6.436 / 6.438; m=32 5.389 against 5.384; m=16 5.174 against 5.170; without the spare slot (hazard not covered) m=128 7.935; `lgkmcnt(0)` then a bare `s_barrier` 8.096; ISA of rb1s1 at m=128: at most 3 LDS ops in flight at each loop barrier, all from the most recent tile; not kept (it failed the run's gate) (bf16 GEMM N=6144, K=1536, 16 waves at m=128)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=1, compile-only probe (`COMPILE_ONLY=1 ARCH=gfx950`, `examples/01-vectorAdd.py` plus an LDS store, `fx.gpu.barrier()` and an LDS read), its IR and ISA dumps; and the ISA of a GEMM build in a tuning run] after convert-gpu-to-rocdl: `llvm.fence syncscope("workgroup") release`, `rocdl.s.barrier`, `llvm.fence syncscope("workgroup") acquire`; final ISA: `global_store_dwordx4` (line 89), then `ds_write_b32`, `s_waitcnt lgkmcnt(0)`, `s_barrier` (lines 96-98), with no `vmcnt` wait in between; the tuning run's record: "The ISA showed that `gpu.barrier()` emits waits on outstanding memory ops, including `lgkmcnt(0)`, before each loop `s_barrier`" (bf16 GEMM N=6144, K=1536)
