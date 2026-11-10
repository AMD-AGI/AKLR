---
title: "fp8 block-scaled GEMM, shuffled B: N 4096 × K 256–8192, N 8192 / 32768 × K 1024, N 16384 / 65536 × K 1536, N 7168 × K 384 (m 64 and 128 only) — one structure per shape; skinny paths at large N and small m; at N 4096 with K 256, 512 and 1024, 8–11 of 13 Points per file are not served"
kind: case
ops: [gemm]
dtypes: [fp8]
regimes: ["M 1-64", "M 128-512", "M 1024-4096", skinny, large-N, short-K]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, CI, per Point", value: "1.010–1.857; 5 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k256, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.986–1.474; 5 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k512, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "d908d01 (#71, CI run 42000899019, measurement profile 18)", date: "2026-11-05"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.809–1.110; 2 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.044–1.302; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k2048, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "d908d01 (#71, CI run 42000899019, measurement profile 18)", date: "2026-11-05"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.151–2.139; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.092–2.590; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "d908d01 (#71, CI run 42000899019, measurement profile 18)", date: "2026-11-05"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.401–1.847; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n8192_k1024, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.061–2.152; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n32768_k1024, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, local validate lower bound (measurement profile 19) per Point; no CI measurement on main", value: "1.328–1.945; 5 of 5 Points ≥ 1.03; 0 served", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n16384_k1536, m 1, 2, 4, 16, 32", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/fp8/fp8_ef050f07296547ba/report.json", date: "2026-11-09"}
  - {metric: "speedup over aiter, local validate lower bound (measurement profile 19) per Point; no CI measurement on main", value: "1.026–1.092; 5 of 6 measured Points ≥ 1.03 (the file lists those 5); 0 served", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536, m 1, 2, 4, 8, 16, 32, measured on the kernel.py of 133d48070dcb6c3d's predecessor (E11)", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/fp8/fp8_e5f9fd89c5a994fa/report.json", date: "2026-11-09"}
  - {metric: "speedup over aiter, local validate lower bound (measurement profile 19) per Point; no CI measurement on main", value: "1.020–1.155; 2 of 8 measured Points ≥ 1.03 (the file lists those 2); 0 served", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n7168_k384, m 1, 2, 4, 8, 16, 32, 64, 128", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/fp8/fp8_2c481a926208c5cd/report.json", date: "2026-11-09"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k256/5fcd9f0ebdced65c.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k512/9a8319559bcb2f29.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024/e5d65066d6572075.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k2048/0610d6f957904254.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096/8c907984278eefc4.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192/10faf68f4a569ef8.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n8192_k1024/41f2a05011466520.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n32768_k1024/42ffbb4ee1697e03.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n16384_k1536/ef050f07296547ba.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536/133d48070dcb6c3d.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n7168_k384/7944f0214f92cf4d.json
---

# fp8 block-scaled GEMM at N 4096 across K, at large N with K 1024 and 1536, and at N 7168 with K 384

## Applies when

- Contract `gemm_a8w8_blockwise_scaled@1.0`, layout variant `b_shuffled` (`operand_format` `none:preshuffled`,
  `a_scale.is_row_major` false): a and b fp8 e4m3fn, b in aiter's 16×16 `shuffle_weight` order, one fp32 a-scale
  per 1×128 of a in the raw column-major payload (`S_a[i, kb] = a_scale_flat[kb·m + i]`), one fp32 b-scale per
  128×128 of b, `trans_b`, `scale_major_mode` K, no bias, bf16 out.
- N = 4096 with K = 256, 512, 1024, 2048, 4096 or 8192 (2 to 64 scale blocks along K), N = 8192 or 32768 with
  K = 1024, N = 16384 or 65536 with K = 1536 (12 scale blocks), and N = 7168 with K = 384 (3). Each (N, K) pair is
  its own schema. The six N 4096 files and the two K 1024 files list m = 1, 2, 4, …, 4096; n16384_k1536 lists
  m = 1, 2, 4, 16, 32 (all its scenario has), n65536_k1536 m = 1, 2, 4, 8, 16 and n7168_k384 m = 64, 128. The
  scenarios' other Points, n65536 m = 32 and n7168 m = 1–32 and 256, have no kernel and run aiter.
- Dispatch serves a listed Point only once CI has written a `speedup` that reaches `AKLR_MIN_SPEEDUP` into it
  (`aklr/core/contract/spec/_workloads.py:38`; `aklr/ops/README.md` § Environment variables): 77 of the 104 Points
  of the eight CI-measured files on main as of #102, and none of the three newer files' Points until CI measures them
  (Result).
- Compare: each schema's `compare` block is `determinism: bounded`, `max_violation_fraction: 0.0`, against the
  contract's tolerance (atol = rtol = 1e-2 for bf16 out; `aklr/ops/gemm/contract.py`). One file's split-K
  depends on it (Counter-signals).
- Read the file for your shape before reusing a design: the eleven files are eleven different kernels, not one
  kernel tuned eleven ways. n16384_k1536 and n65536_k1536 are built like n32768_k1024's skinny path (m ≤ 64), and
  n7168_k384 like n4096_k256's whole-K tile.

## Counter-signals

- **N 4096 with K 256, 512 or 1024 at small m is at or below aiter.** n4096_k256 reads 1.010–1.026 at m ≤ 64 and
  1.022 at m = 128 *(field, E1)*. n4096_k512 reads 0.986–1.000 at m ≤ 128 *(field, E2)*; it splits K nowhere, and
  at m ≤ 64 one 16×64 tile holds all of K (64 workgroups at m ≤ 16). n4096_k1024 clears 1.03 only at m = 2048, 4096
  (1.105, 1.110); its unsplit four-wave body reads 0.809–1.005 at m = 1, 2 and 32–1024, and its eight-wave
  in-workgroup K split at m = 4, 8, 16 reads 0.998–1.000 in #102 after 1.073–1.080 in #71 *(field, E3, E9)*.
- **N 7168 with K 384 at m ≤ 32 reads 1.020–1.028** *(field, E12)*. The kernel's time is flat in m, 4.23–4.31 µs
  from m = 1 to 128, and so is aiter's up to m = 32, 4.38–4.41 µs; the margin opens only where aiter's time grows,
  4.47 µs at m = 64 and 4.90 µs at m = 128 *(field, E12)*. At m ≤ 16 the kernel runs a 16×16 one-wave tile (448
  workgroups), at m = 32 a 16×64 four-wave tile (224 workgroups). The scenario's m = 256 was not measured.
- **Points within a few percent of 1.03 move across it between runs.** Across the six to ten CI measurements AKLR's main
  holds per file (#3 to #102; measurement profile 17 up to #38, 18 from #41 to #71, 19 in #102): n4096_k256 m ≤ 64
  0.986–1.043, at or above 1.03 only in the first run, and m = 128 1.002–1.038; n4096_k512 m = 256 0.997–1.044;
  n4096_k1024 m = 4, 8, 16 0.998–1.085, below 1.03 in four of ten runs, #102 among them, and m = 32 1.004–1.032;
  n4096_k2048 m = 2048 1.023–1.073; n4096_k8192 m = 4096 1.005–1.100; n32768_k1024 m = 64 1.016–1.026 in five runs, then
  1.061 in #102 *(field, E9)*. Which of these dispatch serves depends on the run that wrote the file. Between the two
  local runs of the three newer kernels the ruler changed: measurement profile 18 at a12866b, timed through Kineto, then
  profile 19 at 0e02a81, timed through AKLR's rocprofiler-sdk kernel tracer (#33). Every kernel and aiter time read
  0.2–3.2% higher in the second run, the ratios moved by −1.2% to +1.0%, and n7168_k384 m = 8 (1.031 → 1.028) and
  n65536_k1536 m = 32 (1.035 → 1.026) crossed 1.03 *(field, E13, E11, E12)*.
- **The skinny path's margin shrinks at the m where it widens to 128 columns per workgroup.** n32768_k1024 at m = 64
  (64-row slab, 8 waves, 256 workgroups) reads 1.061 in #102, against 1.319–1.500 at m ≤ 32 with 16- or 32-row slabs and
  64 columns (512 workgroups), and 1.016–1.026 in the five runs before *(field, E8, E9)*. n65536_k1536 at m = 32 (32-row
  slab, 8 waves, 512 workgroups, 4 K blocks of weights in flight instead of 8) reads 1.026 against 1.047–1.092 at m ≤
  16; its time rises from 16.66 µs at m = 16 to 18.60 µs, aiter's from 18.22 to 19.09 µs *(field, E11)*.
- **At N 65536 the margin is 5–9%, under a read-roof cap.** At m ≤ 16 the kernel reads its 100.7 MB of weights in
  16.50–16.66 µs, an estimated 6.0–6.1 TB/s, about 93% of the 6.5 TB/s read roof a read-only kernel measured on
  MI355X (`../patterns/memory_bound.md`); aiter, at 17.34–18.22 µs, reaches an estimated 5.5–5.8 TB/s
  *(field, E11)*. That roof caps any kernel at an estimated 1.12–1.18 over aiter at these Points;
  n16384_k1536's margins do not carry over.
- **n16384_k1536's margin follows aiter's time, not the kernel's.** The kernel takes 5.17–5.61 µs at all five
  Points; aiter takes 9.70–10.73 µs at m = 1, 4, 16 (1.868–1.945) but 6.91–7.92 µs at m = 2, 32 (1.328–1.408)
  *(field, E10)*. For this shape aiter 0.1.21.dev66+gdbd8bf5bd's tuned table names its Triton kernel at m = 1, 4, 16
  and a CK kernel at m = 2, 32 (S12). The a8w8 lookup, `get_CKGEMM_config`, tries M exactly and then padded, as
  `../gemm/small_m.md` §7 describes for the bf16 table (S13). Another aiter build or table moves these numbers:
  re-measure after an aiter upgrade.
- **n4096_k8192 at m = 64 and 128 reduces its split-K partials in arrival order.** Split-K 4 across workgroups: each
  split stores its fp32 partial, bumps an agent-scope counter, and the last to arrive adds its own partial first and
  then the others from split index (own + 1) mod 4 onwards. The association depends on which split arrives last, so
  the output bits can differ between runs. The contract is `bounded`, so this passes; under `bitwise` these Points
  need a fixed order (`../gemm/correctness.md` §8, `../gemm/small_m.md` §4). At m = 256
  (split 2) the two-term sum does not depend on the order.
- **The same split-K keeps its counters between calls**: the builder zeroes them once per shape (`torch.zeros`) and
  the last arriver subtracts the split count. No AKLR call may depend on what an earlier call left (`../SKILL.md`,
  Stop table), and two streams calling the same Point at once would share the counters.

## Mechanism

What the eleven kernels share:

- **One 16×16×128 fp8 MFMA per 128-wide K block**, into a zeroed fp32 partial, folded into the fp32 accumulator
  with S_a·S_b, by one FMA in most bodies (`../gemm/mainloop.md` §5). n4096_k256 and n7168_k384 keep their two or
  three K-block partials apart and combine them once at the end.
- **B straight to registers** from the 16×16 preshuffled layout, two 16-byte loads per lane per K block
  (`../gemm/datapath.md` §10), except where noted below (n4096_k4096 at m ≥ 64 and n32768_k1024 at
  m ≥ 128 DMA B into LDS, where several waves share it: `../gemm/datapath.md` §7).
- **A through XOR-swizzled LDS** for most Points (`../gemm/datapath.md` §2), written from registers
  or by DMA (`../gemm/datapath.md` §5). The skinny paths (n32768_k1024 at m ≤ 64, n16384_k1536, n65536_k1536)
  instead store the whole activation slab once with its row stride padded by 16 bytes (`../gemm/datapath.md` §3).
- **S_a·S_b staged in LDS once per workgroup**, in the prologue, at some or all Points of five files (k512, k2048,
  k8192, n8192, n32768), so the K loop reads scales with LDS loads only.
- **No K split at K ≤ 2048** except n4096_k1024 at m 4–16. K 4096 splits inside the workgroup at m ≤ 256; K 8192
  splits inside the workgroup at m ≤ 32 and 256–512, and across workgroups at m 64–256
  (`../gemm/grid_pricing.md` §2).
- **Configuration per m from a table**, compiled in a cached function (`lru_cache`, `functools.cache`) on a Point's
  first call (`../SKILL.md`, Stop table); XCD-aware tile order in five files (`../gemm/schedule_and_locality.md` §2).

Per file (functions are in the `kernel.py` each schema embeds):

**n4096_k256** (`_pick_tiles`) — the whole K = 256 in one tile.
- Tile 16×16 with one wave (m ≤ 16), 16×64 (32, 64), 32×64 (128, 256), 32×128 (512, 1024), 64×128 (2048, 4096)
  with four waves; 128 to 2048 workgroups.
- MFMA roles swapped (weights are the A operand): a lane holds 4 adjacent output columns.
- A straight into registers at m ≤ 4, else written from registers into XOR-swizzled LDS. With 128-column tiles,
  S_a is staged in LDS behind the A tile and S_b read by scalar loads.
- Stores: m ≤ 64 direct 8-byte stores of 4 columns; above, C transposed through XOR-swizzled LDS (its own LDS tile
  at m 512, 1024) and written as row-contiguous 16-byte stores; cache policy sc0|sc1 up to m = 1024, nt above
  (`../gemm/epilogue.md` §3, §5). From m = 2048 consecutive workgroups take adjacent column tiles.

**n4096_k512** (`_pick_tiles`, `_pick_g3_opts`) — no K split at any m.
- m ≤ 64: 16×64 tile holding all of K in one LDS stage (64, 128, 256 workgroups at m ≤ 16, 32, 64). m = 128:
  32×64, two 256-deep stages. m 256, 512: 32×128, every A K tile DMA'd (`BufferCopyLDS128b`) into its own LDS
  buffer in the prologue, B in 4 register stages. m 1024–4096: 64×128 the same way; at m 2048 and 4096 each
  workgroup walks 2 and 4 adjacent 128-column subtiles over its resident A and issues the next subtile's first MFMA
  before this subtile's epilogue. 512 workgroups from m = 512.
- S_a·S_b pre-multiplied into an LDS table in the prologue at m ≥ 4 (m 1, 2 load S_a per tile).
- Standard MFMA orientation. Stores: 2-byte on the register path (m ≤ 128); on the DMA path the bf16 subtile goes
  row-major through XOR-swizzled LDS and out as 16-byte stores with sc1.

**n4096_k1024** (`_select_tiles`; `_compile_wave_splitk_gemm` at m 4, 8, 16).
- m 4, 8, 16: a 16×16 tile per workgroup, 8 waves, one K block per wave, A and B straight into registers; wave 0
  sums the 8 partials from LDS in K-block order. 256 workgroups. m 4, 8 swap the MFMA operands for 8-byte stores;
  m = 16 stores 2 bytes per value.
- Other m: four waves along N, A written from registers into a 2-stage XOR-swizzled LDS ping-pong (async DMA,
  `BufferLoadAsyncLDS128b`, at m = 256), B into registers double-buffered, FlyDSL `preshuffle_gemm`-style
  MFMA/VMEM/DS interleave hints (`../gemm/schedule_and_locality.md` §3,
  `../flydsl/gemm.md` § mainloop §8). Tiles 16×64×256 (m ≤ 32), 16×128×256 (64, 128), 32×128×256 (256),
  64×128×256 (512), 64×256×128 (m ≥ 1024); 64 to 1024 workgroups; 2-byte stores.

**n4096_k2048** (`_choose_tiles`) — no K split.
- Tiles 16×64×512 (m ≤ 64), 32×64×512 (128, 256), 64×64×512 (512), 64×128×256 (1024), 128×128×128 (2048,
  4096); four waves along N; 64 to 1024 workgroups.
- A written from registers into a 2-stage XOR-swizzled LDS ping-pong, B into registers; S_a·S_b for all 16 K
  blocks in LDS; interleave hints at m ≤ 1024. At 32 ≤ m ≤ 1024 the N tiles run along grid x, so workgroups
  sharing a B panel land on one XCD.
- Epilogue: lane pairs swap bf16 halves with one DPP `quad_perm`, then each lane stores 4 bytes (2 adjacent
  columns) per row (`../gemm/epilogue.md` §3).

**n4096_k4096** (`_pick_config`, `_SK2_CFG_SMALL`, `_SK2_CFG`, `_pick_large_config`) — every path issues the MFMA
transposed and stores 8 bytes (4 adjacent columns).
- m ≤ 32: 16×16 (m ≤ 16) or 16×32 (32) workgroup tile, A and B straight into registers, K split over 8 waves
  (m 1, 2) or 4 (m 4–32); tile i is reduced by wave i mod KS from the others' LDS slots in a fixed order.
- m 64–256: two independent halves of K per workgroup, each with its own async-DMA LDS ring for A and B
  (`BufferLoadAsyncLDS128b`); the halves hand partials over through the then-idle B ring. Tiles 32×32, 32×64,
  64×64.
- m ≥ 512: 64×128 (512) or 128×128 tile, A and B DMA'd into a 5-, 3-, 2-, 2-stage LDS ring (m 512, 1024, 2048,
  4096), A XOR-swizzled on the global side so each lane's DMA slot stays linear; at m 2048 and 4096 each XCD gets a
  compact patch of tiles.
- 256 workgroups up to m = 1024, 512 at 2048, 1024 at 4096.

**n4096_k8192** (`_pick_tiles`, `_compile_skinny` at m ≤ 32).
- m ≤ 32: 16×16 or 32×16 tile, four waves each taking 16 contiguous K blocks; A by async DMA
  (`RawPtrBufferLoadAsyncLdsOp`) into a wave-private swizzled LDS ring at m ≥ 2, B into registers; the waves'
  partials summed from LDS in wave order. 256 workgroups.
- m 64, 128: 64×64 / 64×128 tiles with split-K 4 across workgroups (Counter-signals). m 256: split-K 2 with two
  4-wave groups per workgroup taking alternate K blocks of the workgroup's share, summed through LDS. m 512: the two groups
  without the cross-workgroup split. m 1024: 64×128 unsplit; m 2048, 4096: 64×256, its two 128-column halves
  sharing the A reads.
- Main body: A written from registers into 2-stage LDS, B in a register ring, S_a·S_b in LDS, and a block order
  that gives each XCD contiguous (N tile, split) panels so all K slices of a tile share one L2.

**n8192_k1024** (`_pick_tiles`) — no K split; four waves.
- Tiles 32×64 (m ≤ 64), 64×64 (128), 64×128 (256, 512), 128×128 (m ≥ 1024); 128 to 2048 workgroups.
- MFMA transposed at m ≤ 64 and m ≥ 256 (a lane's 4 adjacent columns share one combined scale); standard at
  m = 128. At m ≤ 16 only the first 16-row fragment is computed.
- A from registers into LDS at m ≤ 64; from m = 128 async DMA into a 2- or 3-slot LDS ring whose rows are permuted
  and XOR-swizzled; B into registers, two K blocks ahead below m = 1024, one from m = 1024.
- Combined S_a·S_b per (K block, row) staged in LDS in the prologue; XCD-aware block order. Stores: m ≥ 256
  through an XOR-swizzled LDS epilogue as 16-byte row stores with nt; m ≤ 64 4 adjacent bf16 per lane directly;
  m = 128 2 bytes per value.

**n32768_k1024** (`_skinny_cfg`, `_pick_config`, `_Runner`) — no K split.
- m ≤ 64: weights as the MFMA A operand loaded straight into registers; the activation slab (all of K for 16, 32 or
  64 rows) goes through LDS once with a row stride padded by 16 bytes, behind one barrier; each wave owns one
  16-column tile over all of K.
- m = 128: 8 waves, 128×128 tile, 3-stage LDS ring; the two wave groups alternate, one issuing loads and LDS reads
  while the other runs its MFMAs at raised priority (`s_setprio`). m = 256: 8 waves, 128×128 with the operands
  swapped; m ≥ 512: 4 waves, 256×128 swapped, two workgroups per CU.
- From m = 128 both operands are DMA'd into LDS (`BufferCopyLDS128b`): activations XOR-swizzled, weights in their
  preshuffled order. S_a·S_b for the workgroup tile is folded into an LDS table at kernel start. Swapped tiles pair
  adjacent tiles across lanes (`permlane16_swap`) into 16-byte stores of 8 columns
  (`../gemm/epilogue.md` §3).

**n16384_k1536** (`_skinny_cfg`, `compile_skinny_gemm`, `_Runner`) — n32768_k1024's skinny path, compiled per m.
- 256 workgroups of four waves, one dispatch round on 256 CUs (`../gemm/grid_pricing.md` §1); each wave owns one
  16-column tile over all 12 K blocks, the workgroup 64 columns. One M block covers all rows: 16 (m ≤ 16) or 32
  (m = 32). No K split, no reduction.
- Weights are the MFMA A operand, two 16-byte loads per lane per K block straight into registers: all 12 K blocks
  issued up front at m ≤ 4, 9 in flight at m 16 and 32. Cache policy `nt` (aux 2) on these loads at m ≤ 4 and
  m = 32, the default at m = 16 (`../gemm/small_m.md` §6).
- Activations: only the loads that cover the m real rows are issued (one 16-byte load per thread at m 1, 2, two at
  m = 4; the buffer range is m·K, so the rest reads zero), into LDS at a 1552-byte row stride. At m ≤ 4 the slab
  lands behind one barrier. At m 16 and 32 it comes in six 256-byte K slices whose loads are interleaved, in K
  order, with the weight loads of the first nine K blocks; each slice is written to LDS behind its own barrier just
  before the K loop reaches it. Lanes past m read row 0; their stores fall outside the output's buffer range and are
  dropped.
- Scales: the workgroup's 12 b-scales as invariant loads through the constant address space. At m = 1 the a-scales
  come the same way; at m 2–32 one 16-byte load per thread copies the whole raw a_scale into LDS. S_a·S_b per row
  and K block is formed in registers before the K loop.
- The compile requests a preload of the five pointer arguments into SGPRs (`amdgpu-kernarg-preload-count` 5);
  whether it took is read from the ISA (`../gemm/small_m.md` §6). `_Runner` compiles once per m and calls the
  compiled function directly.
- Stores: a lane holds 4 adjacent columns of one row; m ≤ 16 8-byte stores; m = 32 stages C through LDS (row stride
  72 bf16) and writes 16-byte row stores.

**n65536_k1536** (`_skinny_cfg`, `compile_skinny_gemm`) — the same skinny path with one configuration for every
listed m.
- m ≤ 16: 1024 workgroups of four waves, each wave one 16-column tile over all 12 K blocks; 16-row slab;
  `waves_per_eu` 4. (m = 32 would run 32 rows and 128 columns, Counter-signals.)
- Weights as the MFMA A operand, two 16-byte loads per lane per K block, 8 of the 12 K blocks in flight, every load
  `nt` (aux 2), which the source gives for a read-once weight stream.
- The slab goes from global to registers in six 16-byte loads per thread, each pass's offset in the lane's
  `voffset` so the buffer's range check returns zeros for slab rows past m (`../gemm/datapath.md` §8), then to
  LDS at a 1552-byte row stride, behind one barrier; the LDS read base passes through a `v_mov` so the per-read
  constants fold into the `ds_read` offsets.
- Scales: the workgroup's 12 b-scales in three 16-byte loads moved to SGPRs with `readfirstlane`, a lane's a-scales
  one 4-byte load per K block; S_a·S_b formed in registers before the K loop.
- Stores: 8-byte stores of 4 adjacent columns of one row. `_launch_for` compiles once per m, at the first call,
  and calls the compiled function directly.

**n7168_k384** (`_pick_config`, `_compile`) — n4096_k256's whole-K tile at K = 384, three K blocks.
- Weights are the MFMA A operand, loaded straight into registers for all three K blocks before the first MFMA; tokens
  are the B operand. No K split.
- m = 64: 16×64 tile, four waves of 16 columns, 448 workgroups; A copied 8 bytes per thread into XOR-swizzled LDS
  behind one barrier. m = 128: 32×64 tile, four waves, 448 workgroups; A copied 16 bytes per thread.
- Scales: a lane loads its row's a-scale and its 128-column block's b-scale per K block with raw buffer loads; the
  three K-block partials stay in separate accumulators and are scaled and summed once after the last MFMA.
- Stores with sc0|sc1 (aux 17): m = 64 8-byte stores of 4 adjacent columns straight from the accumulators, rows past
  m dropped by the output's buffer range; m = 128 transposes C through XOR-swizzled LDS (reusing the A buffer after a
  barrier) and writes 16-byte row stores (`../gemm/epilogue.md` §3, §5).

## Result

Each file is the only kernel at its Points, so the faster of AKLR's served kernel and aiter is aiter. The yardstick
is a one-sided lower bound of the aiter / kernel time ratio over CUDA-graph replays with L2 flushed before each call,
read at 16 rounds and, where a threshold is still undecided, again at 32, each look at α = 0.025
(`aklr/core/validation/stages/_measure.py:49`; `alpha_per_look` in a report). Dispatch compares CI's value with
`AKLR_MIN_SPEEDUP`, 1.03 by default (`aklr/ops/README.md` § Environment variables).

The eight N 4096 and K 1024 rows are main's CI measurements as of #102 (6d7cfec): n4096_k256, n4096_k1024,
n4096_k4096, n8192_k1024 and n32768_k1024 from run 42549655045 (#102, measurement profile 19); n4096_k512,
n4096_k2048 and n4096_k8192, which #102 did not re-measure, from run 42000899019 (#71, measurement profile 18).
Profile 19 times the graph replays through AKLR's rocprofiler-sdk kernel tracer (#33), profile 18 through Kineto.
CI installs the baseline aiter, ROCm/aiter@dbd8bf5, and the FlyDSL that aiter pins, 0.3.2 (read from the
workflow; `../pitfalls/measured_a_different_build.md`). The n16384_k1536, n65536_k1536 and n7168_k384 rows come from
`aklr validate` on one MI355X at main 0e02a81 with measurement profile 19, FlyDSL 0.3.2 and aiter
0.1.21.dev66+gdbd8bf5bd, after 16 rounds (32 at n7168_k384 m = 8); the n65536_k1536 row is of the kernel.py of
`133d48070dcb6c3d`'s predecessor (E11). No CI run has measured them, so none of their Points is served yet.

| File | m 1–64 | m 128–512 | m 1024–4096 | ≥ 1.03 | Served |
|---|---|---|---|---|---|
| n4096_k256 | 1.010–1.026 *(field, E1)* | 1.022 (128), 1.058–1.189 (256, 512) *(field, E1)* | 1.240–1.857 *(field, E1)* | 5 of 13 | 5 |
| n4096_k512 | 0.991–1.000 *(field, E2)* | 0.986 (128), 1.044–1.061 (256, 512) *(field, E2)* | 1.177–1.474 *(field, E2)* | 5 of 13 | 5 |
| n4096_k1024 | 0.991–1.004 (1–32), 0.956 (64) *(field, E3)* | 0.809–1.005 *(field, E3)* | 0.918 (1024), 1.105–1.110 (2048, 4096) *(field, E3)* | 2 of 13 | 2 |
| n4096_k2048 | 1.131–1.215 *(field, E4)* | 1.140–1.302 *(field, E4)* | 1.044–1.112 *(field, E4)* | 13 of 13 | 13 |
| n4096_k4096 | 1.447–2.139 *(field, E5)* | 1.193–1.407 *(field, E5)* | 1.151–1.405 *(field, E5)* | 13 of 13 | 13 |
| n4096_k8192 | 1.624–2.590 *(field, E6)* | 1.395–1.468 *(field, E6)* | 1.092–1.266 *(field, E6)* | 13 of 13 | 13 |
| n8192_k1024 | 1.416–1.626 *(field, E7)* | 1.401–1.486 *(field, E7)* | 1.785–1.847 *(field, E7)* | 13 of 13 | 13 |
| n32768_k1024 | 1.319–1.500 (1–32), 1.061 (64) *(field, E8)* | 1.520–1.876 *(field, E8)* | 1.814–2.152 *(field, E8)* | 13 of 13 | 13 |
| n16384_k1536 (local) | 1.328–1.945 (1, 2, 4, 16, 32) *(field, E10)* | — | — | 5 of 5 | 0 |
| n65536_k1536 (local) | 1.047–1.092 (1–16), 1.026 (32) *(field, E11)* | — | — | 5 of 6 | 0 |
| n7168_k384 (local) | 1.020–1.028 (1–32), 1.055 (64) *(field, E12)* | 1.155 (128) *(field, E12)* | — | 2 of 8 | 0 |

All 104 CI Points pass accuracy (`accuracy: passed` in each workload), and so do the 19 local ones (worst violation
fraction 0). The kernels of the local rows take 5.17–5.61 µs (n16384_k1536), 16.50–16.66 µs at m ≤ 16
(n65536_k1536) and 4.23–4.31 µs (n7168_k384) *(field, E10, E11, E12)*.

## Reproduce

On an MI355X, in a checkout of main at 0e02a81, with the three K 1536 and K 384 files added for those, one schema at
a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024/e5d65066d6572075.json
python -m aklr validate --base 0e02a81 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n16384_k1536/ef050f07296547ba.json
```

and the same for the other paths under `sources`. Each runs the audit and bench stages and writes
`aklr_validation/report.md` with the per-Point speedup (`aklr/ops/README.md` § Commands). Give `--base` a commit
that lacks the file, such as 0e02a81 for the three newer ones: with the base set, bench measures only schemas that
are not at the base, so with a base that has the file it skips the bench (`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`).
The local result does not count; CI's measurement is the one dispatch reads. The baseline is aiter's tuned path:
with an aiter other than dbd8bf5, compare the kernel times above before comparing speedups.

## Rollback

Delete the schema file (in a PR): its Points go unserved, since no other file serves them. `AKLR_DISABLE`
takes a whole contract id and would also drop every other `gemm_a8w8_blockwise_scaled@1.0` kernel. Points CI
measures below 1.03 are not served without any rollback (8 Points of n4096_k256, 8 of n4096_k512, 11 of n4096_k1024
on main as of #102). The three newer files serve nothing until CI writes their `speedup`; after that a listed
Point CI measures below 1.03 drops out the same way. The scenario Points they do not list (n65536 m = 32, n7168
m = 1–32 and 256) run aiter. Delete n4096_k8192's file, or give its m 64 and 128 Points a fixed-order reduction, if
the contract's `compare` becomes `bitwise`.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` source a schema embeds in `kernel.sources`.

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR CI, measurement profile 19] speedup over aiter 1.010–1.857 per Point, 5 of 13 ≥ 1.03; m 1–64 1.010–1.026, m 128 1.022, m 256 1.058, m 512 1.189, m 1024–4096 1.240–1.857 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k256, m 1–4096; 6d7cfec, #102, CI run 42549655045)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR CI, measurement profile 18] speedup over aiter 0.986–1.474 per Point, 5 of 13 ≥ 1.03; m 1–64 0.991–1.000, m 128 0.986, m 256 1.044, m 512 1.061, m 1024–4096 1.177–1.474 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k512, m 1–4096; d908d01, #71, CI run 42000899019)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR CI, measurement profile 19] speedup over aiter 0.809–1.110 per Point, 2 of 13 ≥ 1.03; m 1 0.991, 2 0.999, 4 0.998, 8 1.000, 16 1.000, 32 1.004, 64 0.956, 128 0.809, 256 0.862, 512 1.005, 1024 0.918, 2048 1.105, 4096 1.110 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024, m 1–4096; 6d7cfec, #102, CI run 42549655045)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR CI, measurement profile 18] speedup over aiter 1.044–1.302 per Point, 13 of 13 ≥ 1.03; m 1–64 1.131–1.215, m 128–512 1.140–1.302, m 1024–4096 1.044–1.112 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k2048, m 1–4096; d908d01, #71, CI run 42000899019)
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR CI, measurement profile 19] speedup over aiter 1.151–2.139 per Point, 13 of 13 ≥ 1.03; m 1–64 1.447–2.139, m 128–512 1.193–1.407, m 1024–4096 1.151–1.405 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096, m 1–4096; 6d7cfec, #102, CI run 42549655045)
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR CI, measurement profile 18] speedup over aiter 1.092–2.590 per Point, 13 of 13 ≥ 1.03; m 1–64 1.624–2.590, m 128–512 1.395–1.468, m 1024–4096 1.092–1.266 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192, m 1–4096; d908d01, #71, CI run 42000899019)
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR CI, measurement profile 19] speedup over aiter 1.401–1.847 per Point, 13 of 13 ≥ 1.03; m 1–64 1.416–1.626, m 128–512 1.401–1.486, m 1024–4096 1.785–1.847 (gemm_a8w8_blockwise_scaled_b_shuffled_n8192_k1024, m 1–4096; 6d7cfec, #102, CI run 42549655045)
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR CI, measurement profile 19] speedup over aiter 1.061–2.152 per Point, 13 of 13 ≥ 1.03; m 1–32 1.319–1.500, m 64 1.061, m 128–512 1.520–1.876, m 1024–4096 1.814–2.152 (gemm_a8w8_blockwise_scaled_b_shuffled_n32768_k1024, m 1–4096; 6d7cfec, #102, CI run 42549655045)
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=6–10, AKLR CI, every measurement of each file on main; measurement profile 17 in #3–#38, 18 in #41–#71, 19 in #102] near-threshold Points: n4096_k256 m 1–64 0.986–1.043 (≥ 1.03 only in #3), m 128 1.002–1.038 (ten runs: #3, #30, #38, #41, #51, #53, #62, #70, #71, #102); n4096_k512 m 256 0.997–1.044 (seven: #3, #10, #30, #38, #62, #70, #71); n4096_k1024 m 4, 8, 16 0.998–1.085 (below 1.03 in #30, #51, #62, #102; 1.073–1.080 in #71), m 32 1.004–1.032 (ten: #3, #10, #30, #38, #41, #51, #53, #62, #71, #102); n4096_k2048 m 2048 1.023–1.073 (eight); n4096_k8192 m 4096 1.005–1.100 (six); n32768_k1024 m 64 1.016–1.026 in #3, #30, #41, #51, #53 and 1.061 in #102 (those schemas' history on main, e.g. `git log 9a29616..6d7cfec -- <schema>`)
- S1 aklr/ops/gemm/contract.py:188 the contract notes that k8192's split-K order follows CTA arrival; compare `bounded` at :269
- S2 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192/10faf68f4a569ef8.json#kernel.py:962 `finish` sums its own partial, then peers (bid_z + d) & (split_k − 1) at :968; counter zeroed once at :1399, decremented by the last arriver at :155
- S3 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192/10faf68f4a569ef8.json#kernel.py:234 `_pick_tiles`: split 4 at m 64 and 128, split 2 with two groups at 256, two groups at 512
- S4 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024/e5d65066d6572075.json#kernel.py:638 m 4, 8, 16 take `_compile_wave_splitk_gemm`; wave 0 sums the K blocks in order at :596
- S5 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k512/9a8319559bcb2f29.json#kernel.py:771 `_pick_tiles`; the kernel has no split-K parameter; one LDS buffer per K tile on the DMA path at :143
- S6 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k256/5fcd9f0ebdced65c.json#kernel.py:101 `tile_k = K`
- S7 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096/8c907984278eefc4.json#kernel.py:304 owner-wave reduction; half exchange through the B ring at :1092
- S8 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n32768_k1024/42ffbb4ee1697e03.json#kernel.py:1139 skinny kernel; path choice in `_Runner` at :1415
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR validate, measurement profile 19, 16 rounds per Point] speedup over aiter m 1 1.871, 2 1.328, 4 1.868, 16 1.945, 32 1.408; median kernel / aiter time 5.17 / 9.70, 5.20 / 6.91, 5.23 / 9.78, 5.52 / 10.73, 5.61 / 7.92 µs; worst violation fraction 0 (gemm_a8w8_blockwise_scaled_b_shuffled_n16384_k1536 ef050f07296547ba, m 1, 2, 4, 16, 32; main 0e02a81, aiter 0.1.21.dev66+gdbd8bf5bd; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/fp8/fp8_ef050f07296547ba/report.json)
- E11 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR validate, measurement profile 19, 16 rounds per Point] speedup over aiter m 1 1.047, 2 1.056, 4 1.061, 8 1.065, 16 1.092, 32 1.026; median kernel / aiter time 16.52 / 17.34, 16.50 / 17.46, 16.52 / 17.54, 16.57 / 17.65, 16.66 / 18.22, 18.60 / 19.09 µs; worst violation fraction 0; measured as e5f9fd89c5a994fa, at AMD-AGI/AKLR@3877820 (branch kyle/fp8-blockscale-campaigns), which also lists m 32 and embeds the same kernel.py as 63974391515c66e5, the predecessor of 133d48070dcb6c3d; 133d48070dcb6c3d moves each slab pass's offset from `soffset` into the lane's offset, the same 295 instructions in a compile-only ISA dump at m 16 (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/fp8_soffset_isa/before_63974391515c66e5_m16.s:82-89 against after_133d48070dcb6c3d_m16.s:82-89; README.txt in that folder), and calls the compiled launcher directly (gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536, m 1–32; main 0e02a81, aiter 0.1.21.dev66+gdbd8bf5bd; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/fp8/fp8_e5f9fd89c5a994fa/report.json)
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR validate, measurement profile 19, 16 rounds per Point, 32 at m 8] speedup over aiter m 1 1.021, 2 1.020, 4 1.021, 8 1.028, 16 1.025, 32 1.027, 64 1.055, 128 1.155; median kernel time 4.31, 4.29, 4.29, 4.28, 4.27, 4.29, 4.23, 4.23 µs, aiter 4.40, 4.39, 4.38, 4.40, 4.38, 4.41, 4.47, 4.90 µs; worst violation fraction 0; measured as 2c481a926208c5cd, at AMD-AGI/AKLR@3877820 (branch kyle/fp8-blockscale-campaigns), which also lists m 1–32 and embeds the same kernel.py as 7944f0214f92cf4d (gemm_a8w8_blockwise_scaled_b_shuffled_n7168_k384, m 1–128; main 0e02a81, aiter 0.1.21.dev66+gdbd8bf5bd; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/fp8/fp8_2c481a926208c5cd/report.json)
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR validate, measurement profile 18 (Kineto), 16 rounds per Point, 32 at n7168_k384 m 1, 8, 16] the same three kernel.py files at a12866b: n65536_k1536 m 1–16 1.045–1.094, m 32 1.035; n7168_k384 m 1–32 1.020–1.031 (m 8 1.031), m 64 1.055, m 128 1.154; n16384_k1536 m 1, 2, 4, 16, 32 1.894, 1.333, 1.888, 1.966, 1.395; profile 19 (E10–E12) read every median kernel and aiter time 0.2–3.2% higher (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/fp8/profile18/fn65536_k1536.json; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/fp8/profile18/fn7168_k384.json; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/fp8/profile18/fn16384_k1536.json)
- S9 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n16384_k1536/ef050f07296547ba.json#kernel.py:343 `_skinny_cfg`; slab loads trimmed to m rows at :107; slab slices and weights issued in K order at :216, each slice stored before its K blocks at :274; kernarg preload requested at :385
- S10 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536/133d48070dcb6c3d.json#kernel.py:284 `_skinny_cfg`; `B_CPOL = 2` on the weight loads at :292; padded slab row stride at :93; slab loads with the pass offset in the slice at :165; one barrier at :216; `_launch_for` compiling at the first call at :296–308
- S11 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n7168_k384/7944f0214f92cf4d.json#kernel.py:389 `_pick_config`; whole K per tile at :80; K-block partials scaled after the last MFMA at :272; direct stores at :282
- S12 ROCm/aiter@dbd8bf5:aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv:671 N 16384 × K 1536 on 256 CUs: `triton` at m 1, 4, 8, 16 (:671, :673–675), CK kernel 7 at m 2, 32 (:672, :676); no other bpreshuffle table of that commit lists this (N, K)
- S13 ROCm/aiter@dbd8bf5:aiter/ops/gemm_op_a8w8.py:453 `get_CKGEMM_config` looks up (gfx, cu_num, M, N, K) with M as given, then `get_padded_m` at granularity 0 and 1 (:480); `gemm_a8w8_blockscale_bpreshuffle` (:913) uses it
