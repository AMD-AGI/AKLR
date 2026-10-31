# Examples ladders: FlyDSL v0.3.2 kernels on gfx950, one technique per step

Two ladders of small FlyDSL programs in the style of CuTe DSL's `fp16_gemm_0.py … fp16_gemm_6.py`: each file is
the previous one plus exactly one technique, with a docstring that says what the step adds and what it is
estimated to cost or save. Every file runs alone and checks itself against torch.

- `gemm/step0 … step9`: bf16 `C = A·Bᵀ`, A (M×K) and B (N×K) row-major, bf16 out, fp32 accumulate,
  M = N = K = 4096.
- `moe/step0 … step2`: bf16 fused MoE, top-k routing, gate/up + SiLU, then down and the weighted sum;
  T = 64 tokens, E = 8 experts, top-2, model dim D = 1024, inter dim I = 512 (a decode-sized batch).
- `common.py`: what the steps share and none of them teaches (run mode, tolerance, timer, inputs, torch
  references, the host-side B preshuffle and MoE routing sort).
- `run.sh`: runs every step on a GPU, or compiles every step for gfx950 without one.

The API is FlyDSL v0.3.2's (`dd8373349`) as used by its `examples/01-04` and `kernels/gemm/preshuffle_gemm.py`,
`kernels/gemm/gemm_a16w16_gfx950.py`, `kernels/moe/`; the files run from the `flydsl==0.3.2` wheel alone.

## Status

- **Compile-checked, gfx950 ISA**: all 13 files compile for gfx950 with FlyDSL 0.3.2 (wheel), on CPU tensors
  with no GPU (`./run.sh --compile-only`); 0 VGPR/SGPR spills and 0 scratch in every kernel. The ISA facts
  below come from those dumps.
- **Run on a GPU, all 13 pass against torch** (2026-10, one MI355X, FlyDSL 0.3.2 wheel, torch 2.11.0+rocm7.2,
  run once): the GEMM steps' max abs error is 0.999 at max |ref| 339.7 and the MoE steps' 0.0039 at max |ref|
  2.0, both one half bf16 step (2⁻⁸ relative) of an output near that size. Times, median of 20 calls with a
  hot L2 (3 for step 0), M = N = K = 4096:

  | GEMM step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | µs | 17633 | 1803 | 271.5 | 198.7 | 159.8 | 169.8 | 156.5 | 153.4 | 158.4 | 152.3 |
  | TFLOP/s | 7.8 | 76 | 506 | 692 | 860 | 809 | 878 | 896 | 868 | 902 |

  MoE steps 0, 1, 2: 334.8, 199.3, 89.2 µs. Two steps are slower than the one before: step 5 by 6% (2 waves per
  SIMD by registers, against step 4's 4) and step 8 by 3% (3 against step 7's 4); step 9 returns to 4 and is 4%
  faster than step 8, 0.7% faster than step 7. A single run: differences of a few percent are not established.
  Every cost/saving in the docstrings marked *estimated* is still an estimate; numbers marked *field* are
  measured values quoted from the cards.

## How to run

```bash
PYTHON=/path/to/python-with-flydsl-0.3.2 ./run.sh                  # GPU: check + median time per step
PYTHON=... ./run.sh gemm/step5 moe                                  # a subset (path substrings)
PYTHON=... ./run.sh --compile-only                                  # no GPU: compile for gfx950, dump, ISA diffs
python gemm/step3_swizzle.py                                        # one step
```

- GPU mode prints, per step, the max abs error against an fp32 torch reference and a median time (CUDA
  events, 20 calls, hot L2: for comparing steps, not a benchmark). Tolerance: `|out − ref| ≤ 1e-2 + 1e-2·|ref|`
  elementwise (bf16 output rounding alone is ≤ 2⁻⁸ relative). Exit status 1 if any step fails.
- `--compile-only` sets `COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950`, empties `*_VISIBLE_DEVICES`, keeps ROCm
  tools off `PATH`, and dumps every pass with `FLYDSL_DUMP_IR=1` under `$KBEX_DUMP` (default `/tmp/kbex/dump`):
  the ISA is `<step>/<kernel or jit name>/21_final_isa.s`. It then prints each step against the previous one
  with `../tools/isa_diff.py`. `FLYDSL_GPU_ARCH` is needed as well as `ARCH`: `s_waitcnt` and the buffer-descriptor
  flags ask the host for its arch (`../language/limits.md`).
- Step files put `examples/` on `sys.path` themselves, so `python -I` works. Kernel names are the same in every
  step of a ladder (`gemm_kernel_0`; `moe_stage1_kernel_0`, `moe_stage2_kernel_1`, `moe_reduce_kernel_2`), so
  `isa_diff.py` pairs them.

## The GEMM ladder

Tile from step 2 on: 128×128×64 per workgroup, 4 waves split along N (`make_layout((1, 4, 1), (0, 1, 0))`,
preshuffle_gemm's split), MFMA `v_mfma_f32_16x16x32_bf16`. "Loop" = the main K loop of the final ISA, per
iteration; "occ." = waves per SIMD by registers and the resulting dispatch rounds for the 1024 tiles of this
shape (derived with `../gemm/flydsl_gemm_grid_pricing.md` §1; here workgroups per CU = waves per SIMD).

| Step | Adds | Card | Compile-checked gfx950 ISA | VGPR / LDS | Occ. (derived) |
|---|---|---|---|---|---|
| 0 `step0_naive` | one output per thread, scalar loads, fp32 FMA | `../optimize/gemm-optimization.md` §1 | loop (8 k): 2 `global_load_dwordx4` (LLVM unrolled 8× and merged the scalar loads), 8 `v_mul_f32` + 8 `v_add_f32` (no contraction without `fastmath`); 1 `global_store_short` | 23 / 0 | 8; 65536 WGs, 32 rounds |
| 1 `step1_mfma` | one wave per 16×16 tile, `MFMA(16,16,32,bf16)`, operands straight from global | `../optimize/gemm-optimization.md` §6.2; `../gemm/flydsl_gemm_mainloop.md` §5 | `v_mfma_f32_16x16x32_bf16` appears; loop (4 k-steps, LLVM 4× unroll): 4 MFMA, 8 `buffer_load_dwordx4`; 4 `buffer_store_short` | 36 / 0 | 8 (32 one-wave WGs/CU); 8 rounds |
| 2 `step2_lds` | 128×128 tile on 4 waves; A and B global → VGPR → LDS, row-major | `../gemm/flydsl_gemm_datapath.md` §1, §7 | loop (1 K tile): 32 MFMA, 8 `buffer_load_dwordx4`, 8 `ds_write_b128`, 20 `ds_read_b128` (16 A + 4 B), 2 `s_barrier`, 12 `s_waitcnt`; 64 `buffer_store_short` | 114 / 32 KiB | 4; 1 round |
| 3 `step3_swizzle` | XOR swizzle `Swizzle<3,3,3>` composed into the LDS views | `../gemm/flydsl_gemm_datapath.md` §2–3; `../optimize/lds-optimization.md` | same memory instructions (`ds_read_b128` / `ds_write_b128` kept); +4 `v_bitop3_b32` (the XOR, outside the loop); waits 12 → 20 as LLVM now interleaves the reads with the MFMAs. Bank conflicts are not in the ISA: by hand, 4-way on every MFMA operand read before, none after | 112 / 32 KiB | 4; 1 |
| 4 `step4_dma` | global → LDS by DMA (`cdna4.BufferLoadAsyncLDS128b`, `asyncmark` / `wait_asyncmark`); swizzle moves to the global address | `../gemm/flydsl_gemm_datapath.md` §4–5 | 8 `buffer_load_dwordx4 … offen lds` replace 8 `buffer_load_dwordx4` + 8 `ds_write_b128`; one `s_mov_b32 m0` per DMA (+6 `s_nop`); loop waits 20 → 4; 1 `v_readfirstlane_b32` (wave base) | 120 / 32 KiB | 4; 1 |
| 5 `step5_pingpong` | 2 LDS stages: next tile's DMA overlaps this tile's MFMAs; one barrier per tile | `../gemm/flydsl_gemm_mainloop.md` (2-stage baseline); `../optimize/prefetch-data-load.md` | loop (2 tiles): 64 MFMA, 16 DMA, 40 `ds_read_b128`, 2 `s_barrier` (1 per tile, was 2); no vmcnt wait between a stage's DMA and the other stage's reads. Peak VGPR is in the peeled last tile (all its reads hoisted; the loop uses v0–v119) | **216** / 64 KiB | **2**; 2 rounds |
| 6 `step6_preshuffle_b` | B preshuffled on the host (`shuffle_weight`), loaded straight to MFMA registers, register double buffer | `../gemm/flydsl_gemm_datapath.md` §10; `../optimize/gemm-optimization.md` §12 | loop (2 tiles): 64 MFMA, 8 DMA (A), 8 `buffer_load_dwordx4` (B → VGPR), 32 `ds_read_b128` (was 40); B waits are partial (`vmcnt(4..7)`); LLVM sinks the B loads into the middle of the MFMA stream | **138** / 32 KiB | **3**; 2 rounds (2nd ⅓ full) |
| 7 `step7_sched_hints` | `sched_barrier(0)` after the loads and after the MFMAs; a `sched_group_barrier` table (4 reads ahead, then {4 MFMA, 2 reads} × 8) | `../gemm/flydsl_gemm_schedule_and_locality.md` §3; `../optimize/gemm-optimization.md` §5 | same instruction counts; all 8 loads now at the top of each K step; LDS reads one group ahead (waits `lgkmcnt(3)/(2)` instead of `(1)/(0)`); s_waitcnt 34 → 38, `s_nop` 2 → 7 in the loop | 128 / 32 KiB | 4; 1 |
| 8 `step8_cshuffle_epilogue` | C through LDS (`@fx.union` with the A stages), read back 16 B per lane along rows | `../gemm/flydsl_gemm_epilogue.md` §1–3 | loop unchanged; epilogue 64 `buffer_store_short` → 64 `ds_write_b16` + 8 `ds_read_b128` + 8 `buffer_store_dwordx4`, +2 `s_barrier`; LDS stays 32 KiB (union) | **130** / 32 KiB | **3**; 2 rounds |
| 9 `step9_occupancy` | `value_attrs={"rocdl.waves_per_eu": 4, "llvm.passthrough": [["amdgpu-agpr-alloc", "0"]]}` | `../gemm/flydsl_gemm_mainloop.md` §4, §7; `../gemm/flydsl_gemm_grid_pricing.md` §1, §4 | LLVM IR carries `"amdgpu-waves-per-eu"="4"`, `"amdgpu-agpr-alloc"="0"`; +1 `s_nop`, nothing else | 128 / 32 KiB | 4; 1 |

Relative to the suggested 0–6 ladder: "DMA with a swizzled layout" became steps 3 and 4 (the swizzle shows on
its own in the synchronous copy; the DMA then moves it to the global address), "scheduling hints / preshuffled B"
became steps 7 and 6, and step 9 was added because step 8 crossed a register tier.

Deltas worth a reviewer's attention (all compile-only):
- **Step 5's barrier.** `fx.gpu.barrier()` carries the workgroup fence: in step 5's ISA every `s_barrier` comes after
  the stage's `ds_read`s have been waited for (`lgkmcnt(0)`). A bare `rocdl.s_barrier()` is no fence, so a read of a
  stage may still be in flight while other waves' DMA overwrites it; FlyDSL's `gemm_a16w16` pins the MFMAs with
  `sched_barrier(0)` instead (what step 7 adds).
- **Hint tables can make the schedule worse.** In step 7, the table "1 read, 2 MFMA" made LLVM wait `lgkmcnt(0)`
  right after every read and put two MFMAs on one accumulator back to back; fences alone gave 120 VGPRs and the
  reads two at a time; the table that is in the file adds one group of read lookahead for 8 VGPRs. Under every
  table tried, the four A DMAs stayed in one burst ahead of the `asyncmark` that closes their group; only the B
  loads and the LDS reads moved.
- **Step 8 → 9.** The CShuffle epilogue raised VGPRs 128 → 130 through allocation in the main loop, one granule over
  the 128 tier: 4 → 3 workgroups per CU, 1 → 2 dispatch rounds at this shape. `rocdl.waves_per_eu: 4` alone made
  the allocator split 64 arch / 64 AGPR and spill 96 VGPRs (164 B scratch); with `amdgpu-agpr-alloc=0` it fits in
  128 with no spill.
- Steps 5, 6 and 8 sit in a lower occupancy tier than their neighbours: read their GPU timings with that in mind.
- `../tools/isa_diff.py` prints `agpr -1` / `-2` for kernels with no AGPRs whose VGPR count is not a multiple of 4
  (`accum_offset` is rounded up): a display artifact; `.agpr_count` in the metadata is 0 for every kernel here.

## The MoE ladder

All three steps compute the same thing: stage 1 `h = silu(W1g·x)·(W1u·x)` in bf16, stage 2 `out = Σⱼ wⱼ·W2·hⱼ`,
summed in a fixed order (no atomics: bitwise repeatable).

| Step | Adds | Card | Compile-checked gfx950 ISA |
|---|---|---|---|
| 0 `step0_gemv` | GEMVs: stage 1 one workgroup per (token, expert), each thread 2 outputs × D; stage 2 one workgroup per token walking its top-k experts in order | `../moe/flydsl_moe_decode.md`; `../grouped_gemm/flydsl_grouped_moe_inference.md` §2 | 2 kernels, no MFMA. Stage 1 inner loop per 4 k: 3 `global_load_dwordx2` (x, gate row, up row; LLVM merged 4 scalar loads), `v_pk_mul_f32`/`v_pk_add_f32`; VGPR 24. Stage 2: VGPR 24 |
| 1 `step1_sorted_mfma` | routed items sorted by expert into 16-row blocks (host sort, `common.moe_sort`); stage 1 and 2 become 16×64 MFMA tiles (4 waves along N), A rows gathered by token; stage 2 writes weighted fp32 partials; a third kernel sums them | `../grouped_gemm/flydsl_grouped_moe_inference.md` §2, §5, §6 | `v_mfma_f32_16x16x32_bf16` appears. Stage 1: expert id by `s_load_dword`, uniform `s_cbranch_scc1` exit, W1 descriptor base in SGPRs (`s_lshl_b32 s0, s10, 20` = e·2I·D); loop per 4 k-steps: 8 MFMA (gate + up), 12 `buffer_load_dwordx4` (4 gathered A + 8 W1); VGPR 48. Stage 2: K loop fully unrolled, 16 MFMA, 32 `buffer_load_dwordx4`; epilogue 4 `buffer_load_dword` (weights) + 4 `buffer_store_dword` on bounded descriptors (`num_records` 512 B and 512 KiB: sentinel rows read 0 / are dropped); VGPR 18. Reduce: 8 `global_load_dword`, 4 `global_store_short` |
| 2 `step2_fused_sort` | the routing sort folded into stage 1, in LDS; column block 0 publishes the block's ids and expert for stage 2 | `../moe/flydsl_moe_decode.md` §1–2; `../gemm/flydsl_gemm_datapath.md` §8 | Stage 1: LDS 1140 B; sort loop 16 iterations × 2 broadcast `ds_read_b128`; the 8 counts come back through `v_readfirstlane_b32` and the block → expert search runs on SALU; 3 `s_barrier`; publish = 2 `buffer_store_dword` whose masked lanes point past the bounded buffer (offsets 0x400 and 64 = `num_records`); expert branch uniform; SGPR 24 → 58, VGPR 48 unchanged; GEMM loop identical to step 1. Stage 2 and reduce identical to step 1. No host sort: 3 FlyDSL kernels per call |

Not written (each would be one more step): stage 2 accumulating into the output with atomics, which drops the
reduce kernel but gives bounded rather than bitwise determinism (`../grouped_gemm/flydsl_grouped_moe_inference.md`
§6); A staged once per block in LDS instead of re-read by each wave; a persistent stage 2.

## FlyDSL v0.3.2 behaviour these files ran into

- **A method call on a local inside a runtime `for` / `if` makes that local loop-carried.** The AST rewriter
  carries every name whose method the body calls (`x.retile(...)`, `tiled_mma.thr_slice(...)`), and a
  `ThrCopy` slice cannot be carried: `TypeError: ThrCopy.__init__() missing 1 required positional argument:
  'thr_idx'`. Hoist such calls out of the loop; put the body of a runtime `if` in a nested function and call it
  (`if e >= 0: expert_block()`), as the MoE steps do (`../language/compile_errors.md`).
- **Masks without branches.** A store that must not happen goes to an index past a bounded buffer
  (`make_buffer_tensor(..., max_size=False)`; dropped by the hardware) or to a spare LDS slot; a load past a
  bounded buffer reads 0. The MoE steps use this for sentinel rows, routing weights and the publish step.
- `fx.rocdl.readfirstlane` on a value read from LDS keeps a per-expert buffer descriptor in SGPRs
  (`../gemm/flydsl_gemm_datapath.md` §8).

## Layout in the skill

These files sit at `examples/` in the skill root; the card links above are relative to `examples/`, the
docstring links (`../../gemm/...`) to `examples/gemm/` and `examples/moe/`.
