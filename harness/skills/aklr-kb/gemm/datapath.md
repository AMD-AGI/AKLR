---
title: "GEMM — LDS, global feed and prefetch on gfx950"
kind: topic
lever: gemm_datapath
ops: [gemm]
gens: [gfx950]
dtypes: [fp8, bf16, fp16, mxfp4]
regimes: [compute-bound, feed-bound]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py:283
  - ROCm/FlyDSL@dd8373349:kernels/common/mma/mfma_preshuffle_pipeline.py:30
  - ROCm/FlyDSL@dd8373349:kernels/gemm/gemm_a16w16_gfx950.py:489
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/gpu.py:157
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:59
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/cdna4.py:42
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://github.com/nod-ai/amd-shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/gemm_helper.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/prims.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
---

# LDS, global feed and prefetch in a GEMM

> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets. "PS" is FlyDSL's
> `kernels/gemm/preshuffle_gemm.py`; "GH", "PR" and "FP4" are Primus-Turbo's
> `utils/gemm_helper.py`, `utils/prims.py` and `gemm/gemm_mxfp4_kernel.py` under `primus_turbo/flydsl/` at
> main `7615a5a2`. "A16" is FlyDSL's `kernels/gemm/gemm_a16w16_gfx950.py`.

## Route here when
- An LDS, bank-conflict, swizzle or `lgkmcnt` question in a GEMM.
- Deciding how an operand reaches the MFMA: LDS, direct-to-VGPR, sync copy or DMA.
- Global loads or C stores look like the cost, or a prefetch did not help.

FlyDSL's GEMM guide shows the baseline data path on `preshuffle_gemm`: A through LDS by sync copy or
async DMA, B preshuffled straight to VGPRs, the XOR swizzle and its write/read pair
(`../flydsl/guides.md` § gemm-optimization (FlyDSL's §3, §4, §11, §12)). The general LDS and prefetch skills
(`../flydsl/guides.md` § lds-optimization, `../flydsl/guides.md` § prefetch-data-load) cover diagnosis from a trace,
swizzle vs padding, write-to-read distance and the prologue/loop/epilogue prefetch transformation, for any
kernel. gfx950 has 160 KiB of LDS per CU in 64 banks of 4 bytes.

## 1. Allocate LDS the v0.3.2 way

In FlyDSL: `../flydsl/gemm.md` § datapath §1.

## 2. Swizzle at 16-byte granularity, composed into the layout

- XOR whole 16-byte slots, never elements inside a vector: `swizzle_xor16` is
  `col ^ ((row % k_blocks16) * 16)` with `k_blocks16 = tile_k·elem_bytes / 16`
  (`mfma_preshuffle_pipeline.py:30`). XOR-ing at element granularity scrambles the lanes of a
  `ds_read_b64/b128`.
- The swizzle is per-lane VALU (or folded into the layout), not "one SALU": the row index lives in a
  VGPR.
- PS keeps a fixed `Swizzle<3,3,3>` on the bf16 sync path; its source comment gives 1011 against 417 TF/s
  for the derived swizzle at `tile_k=256` *(field, E1)* (PS:284-288). Re-measure a swizzle change; "derived
  from the tile width" is not automatically better.
- **One function computes the key for both sides.** Turbo's `lds_row_swizzle(row, chunks) =
  ((row % 16) // 2) & (chunks - 1)` (in 16 B chunks) is called by the G2S writer (`swizzle_128`) and by
  the transpose reader, "so the two cannot drift" (GH:220-231).
- **Transpose reads need a wave term too.** With a 1024 B per-wave chunk stride, every wave's
  transpose reads hit the same banks. Turbo XORs the chunk index with `W << 1` on both sides (`wswz`,
  GH:1809-1830, 1863). *(field, E2)* On a grouped GEMM with that stride, bank conflicts went from 14% to 0%.

In FlyDSL: `../flydsl/gemm.md` § datapath §2.

## 3. Bank conflicts: compute the floor, then measure

**First tell the three LDS failure modes apart; each has a different fix.**

| Mode | How it shows up | Fix |
|---|---|---|
| Capacity | LDS per workgroup caps occupancy | smaller tile or fewer stages |
| Layout | bank conflicts from the stride or the access pattern | swizzle or padding |
| Timing | a `ds_write` too close to the `ds_read` that depends on it | more write-to-read distance (`../flydsl/guides.md` § lds-optimization) |

Treating every LDS symptom as a swizzle problem wastes the most time: check the write-to-read distance before
the layout.

gfx950 LDS has 64 banks of 4 bytes; bank = `(byte_addr / 4) mod 64`. A read is served in phases and
conflicts count within a phase: `ds_read_b32` is one phase of all 64 lanes, `b64` two phases of 32
consecutive lanes, `b128` four phases of 16 lanes in a non-consecutive pattern ({T0–3, T12–15, T20–23, T24–27},
{T32–35, T44–47, T52–55, T56–59}, {T4–7, T8–11, T16–19, T28–31}, {T36–39, T40–43, T48–51, T60–63}). For a `ds_read_b32` whose lane stride is `s` dwords, the 64 lanes
touch `64 / gcd(s, 64)` distinct banks: a 128-byte row stride (`s = 32`) lands every lane on two banks, a
32-way conflict. For `b64`/`b128`, count the stride in 8- or 16-byte units: with one row per lane, an odd
stride keeps every phase on distinct banks.

- **Swizzle or pad.** XOR swizzle when the access pattern is regular and LDS headroom is tight; padding when
  a small stride bump removes the conflict and the swizzle arithmetic would clutter the address code.
  gfx950's 160 KiB LDS often leaves room for padding. Derive any swizzle or padding from the 64-bank rule
  above; a layout you did not derive that way is unverified.
- Padding must keep the 8- or 16-byte alignment a `ds_read_b64/b128` needs. `+1 element` on fp16 breaks
  it; pad by whole 16-byte slots, or swizzle.
- **Declare 16-byte alignment on a 16-byte LDS read** where LLVM cannot prove it; otherwise the read splits into
  `ds_read2_b32` or `ds_read2_b64` and conflicts more.
- Compare `SQ_LDS_BANK_CONFLICT` against the floor you computed by hand, not against zero.
- **Zero conflicts is not a speedup and LDS is not the bound until a probe says so.** *(field, E3)* An
  attention kernel whose `WAIT_INST_LDS` came to 81% of its CU-cycle budget gained nothing measurable
  (−0.25%, −0.09% and −0.09%, all noise) when each of three LDS read classes was halved in turn; the
  stall counter could not say whether LDS bounded it. Use a halve-the-reads probe
  (`attribution.md`).

In FlyDSL: `../flydsl/gemm.md` § datapath §3.

## 4. Waits around LDS

- **Same wave, write then read: no wait.** LDS operations of one wave complete in order;
  `llc -mcpu=gfx950` emits `ds_write_b32` directly followed by `ds_read_b32` of a neighbouring lane's
  slot.
- **Another wave reads what you wrote:** the writer waits `lgkmcnt(0)`, then everyone meets at
  `s_barrier` — which is what `fx.gpu.barrier()` lowers to.
- **A FLAT (generic-address) access counts in both `vmcnt` and `lgkmcnt`.** *(field, E4)* Waiting
  `lgkmcnt(0)` after a flat load from global took 96 ticks against 56 after a `global_load`, and both
  took about 250 to `vmcnt(0)` on a line read before (about 1000 on first touch). The lgkm count retires
  before the data arrives, so `lgkmcnt(0)` alone does not make a flat load's data ready. Keep hot-loop
  pointers in the buffer or global address space.
- `lgkmcnt` also counts scalar memory loads, which return out of order. A long stall on
  `s_waitcnt lgkmcnt(0)` may be an `s_load`, not LDS; keep scalar loads out of that window instead of issuing
  them there to overlap with the LDS wait.
- LDS data operands may be AGPRs on gfx950 (`ds_read_b32 a0, v1` and `ds_write_b128 v1, a[0:3]`
  assemble); only the address must be a VGPR.
- `lgkmcnt` is a 4-bit field (max 15) and `vmcnt` a 6-bit field (max 63).
- **After an intrinsic LDS DMA, the compiler drains everything before your `ds_read`.** It cannot prove
  which reads alias the DMA's destination, so it inserts `s_waitcnt vmcnt(0)`. Turbo issues those
  `ds_read`s in inline asm and supplies the wait itself (`vmcnt_hint`, `_packed_ds_read_tr_offsets`,
  GH:1783). *(field, E5)* At long K this drain removal gave +6.8% (2580 → 2768 TF/s on a dense fp8 TN GEMM,
  8192×8192×28672); `vmcnt(3)` was the deepest setting that stayed bitwise repeatable. An asm `ds_read`
  must also drain its own `lgkmcnt` before the consumer, because the compiler does not track it.
  Or keep the tracked DMA and emit the read as an `llvm.load` from `!llvm.ptr<3>` carrying an
  `alias_scopes` attribute: it got no compiler wait *(field, E29)*, so the counted wait is yours.

In FlyDSL: `../flydsl/gemm.md` § datapath §4.

## 5. Global → LDS DMA

| | gfx950 |
|---|---|
| `buffer_load_dword … lds` | yes |
| `buffer_load_dwordx4 … lds`, `global_load_lds_dwordx4` | yes (16 B/lane) |
| `ds_read_b64_tr_b16` and the other transpose reads | yes |

- An LDS DMA is a VMEM load that writes LDS: it counts in `vmcnt` like any other load.
- The MUBUF immediate `offset` is 12 bits. `llvm-mc` encodes `offset:4096` as `offset:0` without an
  error (`0x1000` lands outside the field), so a hand-written `offset:4096` silently reads the wrong
  place. *(field, E7, E8, E9)* On gfx950 the LDS-DMA immediate applies to the LDS destination as well as
  the memory address, which lets a wave-major fill share one M0 write across steps. Turning it off cost one
  MXFP4 dense GEMM 0.77% of an 18-cell geomean (an estimated ~0.9% on each cell that used it); on an MXFP8
  grouped GEMM and on an A6W4 GEMM the wave-major fill read within noise. Treat it as a way to free SALU
  issue slots and address registers, and price it per kernel.
- In a hand-written LDS DMA, cache bits go before `lds`: gfx950's assembler rejects `offen lds nt`
  ("invalid operand for instruction") and accepts `offen nt lds`.
- The M0 base of an LDS DMA must never go negative; see `correctness.md` §5.

**How Turbo issues the DMA** (`G2SLoader`, GH:305-386):
- The per-lane global offsets are Python ints computed at trace time, so the hot loop has no address
  VALU. The wave-uniform K offset rides the SGPR `soffset`.
- **M0.** In the whole-loop asm each fill is `s_add_u32 m0, <pool base>, <step·nwaves·1024>` followed by
  `buffer_load_dwordx4 … offen lds`. The MXFP4 kernel uses a **wave-major** layout instead: each wave's
  chunks are contiguous, so consecutive steps are 1 KiB apart and ride the 12-bit immediate. One M0 write
  then covers a 4 KiB window (`g2s_lds_imm`, GH:294; `_MXFP4_G2S_IMM`, FP4:583-586). The immediate applies
  to the memory address too, so pre-subtract it from the global voffset, which must stay ≥ 0.

In FlyDSL: `../flydsl/gemm.md` § datapath §5.

## 6. What a global access costs

- **The cost unit is unique 128-byte lines touched per instruction, not bytes or requests.**
  *(field, E10, E11)* On a grouped GEMM, narrowing a `buffer_load … lds` to a quarter of its width with the
  same lines touched bought 0.14%; cutting the lines one instruction touches from 8 to 1 at the same
  instruction count bought 20.2% in a timing-only probe whose output was wrong. Storing an fp6 A operand
  densely (96-byte rows) saved 25% of the bytes but scattered each wave's DMA over 64 lines and lost 26.6%;
  re-tiling it so a wave's 64 lanes read 1 KiB contiguously turned the same byte saving into +2.0%. Merging
  requests pays only when it also cuts unique lines.
- **Keep `tile_k` at a whole 128 B line per row; do not halve it to buy stages.** bf16 `tile_k` 32 fills 64 B,
  half a line. *(field, E12, E13, E14)* On three kernels, `tile_k` 64 → 32 with 3–6 stages lost 4–17% at
  M ≈ 900–16k and 31–43% at M = 80–112.
- **A contraction stride ≡ 64 mod 128 bytes straddles lines.** Padding K so the row pitch is a
  multiple of 128 B removed the straddle on a grouped fp8 GEMM, 2196 → 2554 TF/s *(field, E15)*.
- **Stores go out in 64-byte requests on gfx950.** Pairing two columns per lane halves the request
  count, 2.36e7 → 1.21e7 write requests on a grouped fp8 GEMM *(field, E16)*; `dwordx2` alone only cuts
  instruction count. Watch `TCP_TCC_WRITE_REQ` and `TA_BUFFER_WRITE_WAVEFRONTS` together.
- **Price the C store before optimizing it.** *(field, E17)* An instruction-class ablation of one fused
  epilogue put 6.3% on the stores at K = 4096; removing the VALU that feeds them read +0.43% and +0.52%,
  both within noise. Store cost is estimated to scale with 1/K (C bytes per FLOP), not with tile shape;
  only K = 4096 was measured.
- **Separate `nt` from whole-row coalescing before crediting either.** *(field, E18, E19)* On one shape,
  dropping `nt` from the C stores cost 10.8% with whole-row stores and 10.4% with half-row ones (0 of 61
  rounds faster): there the bit was the gain. A whole-row against half-row reading taken on another shape
  cannot split the two. After a tile change silently turned coalescing off, restoring it with `nt`
  unchanged saved 2.2%. Both are per-shape, per-tile choices. Choose `nt` per load by reuse class (scale
  streams that are re-read stay cached).
- **LDS immediates have windows.** A base at or above `0x10000` cannot use the 16-bit DS offset
  (`ds_read_b128 … offset:65536` does not assemble), and `ds_read2st64`'s 8-bit offset (256-byte units)
  breaks when tiles widen. *(field, E20)* On a cast kernel, doubling the row stride from 512 to 1024 B
  pushed offsets out of that window and the paired reads split: +38% `ds_read` instructions per unit of
  work (84 → 232 for twice the work).

## 7. Prefetch inside the K loop

- Carry everything the next iteration needs — scale values and addresses, not only the tiles.
- **Prefetch pays only when independent work can fill the latency it exposes**: next-tile loads, address
  arithmetic, MFMAs that do not depend on the load in flight, epilogue setup that touches no data in flight.
  The general transformation (prologue, loop-carried buffers, epilogue) is `../flydsl/guides.md` § prefetch-data-load.
- **Prefetch hides latency, not capacity.** *(field, E21)* When a grouped GEMM's working set thrashed L2,
  deeper prefetch bought nothing; fix the tile order (`schedule_and_locality.md` §2) or the footprint.
- **Bypassing LDS for an operand several waves share costs twice.** Each wave reloads it from global,
  and at one workgroup per CU nothing hides the longer L2 latency. *(field, E22, E23)* In an 8-wave kernel, B
  direct-to-VGPR reached 77.8% of the throughput of B through LDS; A direct-to-VGPR ran 2.5× slower on a
  long K. PS can load B straight to VGPR because B is pre-shuffled and each wave owns its own N slice.
  Private rows do not rescue a row-major A: it must then be loaded in the MFMA lane layout, which touches more lines
  per instruction (§6), or permuted in registers. *(field, E30)* With 8×1 waves, each owning its A rows, both lost
  8–32% to A through LDS on a bf16 GEMM, while contiguous loads of the same rows (wrong output by design) were as
  fast as it or up to 3% faster.
- A prefetch that pushes the kernel over a register tier costs more than it hides; re-read
  `.vgpr_count` after adding one.
- **Gate a prefetched load exactly as its consumer is gated**, in the prologue and in the loop body. A
  load behind a mode flag (e.g. a scale load that only one quantization mode needs) that is gated
  differently in the two places feeds the consumer a value from the other branch.

In FlyDSL: `../flydsl/gemm.md` § datapath §7.

## 8. Buffer descriptors: per-tile i64 rebase, pinned to SGPRs

The buffer descriptor's `num_records` and the per-lane `voffset` are 32-bit, so one descriptor addresses
at most 4 GiB.
- **Fold each tile's large base into the descriptor's 64-bit base** and keep only the in-tile offset in
  `voffset`/`soffset` (`make_fp8_buffer_tensor_rebased`, GH:118-147). Bound `num_records` to the rest of the
  tensor from that base: out-of-range M/N/K reads then return zero for free.
- **Pin base and `num_records` with `readfirstlane`.** A value the divergence analysis cannot prove
  uniform lands in VGPRs, and a descriptor built from VGPRs puts every buffer access behind a
  readfirstlane waterfall (`_readfirstlane_i32`, PR:338-344). *(field, E24)* On a grouped GEMM, leaving
  them in VGPRs cost 16% (gate_up) and 13% (down).
- When one operand's strided traversal spans 2³² bytes or more, rebase the descriptor per load (the
  `rebase` mode of `G2SLoader`). Clamp the shifted `num_records` at 0 for tiles past the end.
- Check SNR on both sides of 2³¹ elements, where 32-bit index arithmetic overflows. Equal SNR means there is no
  overflow break.

In FlyDSL: `../flydsl/gemm.md` § datapath §8.

## 9. K-row operands: hardware transpose reads

- `ds_read_b64_tr_b8` (fp8) and `ds_read_b64_tr_b16` (bf16) read a K-row tile (stored `[K, ·]`) from LDS directly into MFMA
  operand order (gfx950 only). Turbo runs NN and TN natively with them, with no transposed copy of either
  operand (`S2RLoaderTr`, GH:1831-1963).
- Pack the four reads of one fragment onto one base VGPR with immediate offsets. One address VGPR
  instead of four avoids an address-register spill (`_packed_ds_read_tr_offsets`, GH:1783-1805).
- The lane→column map of a transpose read is fixed by the hardware. Only which 16-column chunk a 16-lane
  group reads can be chosen through the address.
- *(field, E25)* Do not "halve the instruction count" by replacing four `ds_read_b64_tr_b8` with two
  `ds_read_b128` at the same addresses. On a grouped GEMM the probe took the wgrad speed score (higher is
  faster) from 0.971 to 0.566: transpose-read base addresses are laid out per transpose unit, and `b128` at
  those addresses collides on banks. The instruction and the LDS layout must change together.
- A transpose-read operand costs more LDS instructions per MAC than a K-contiguous one: 8 B per read
  against 16 B. An MXFP8 NT grouped GEMM's steady state ran an estimated 11–14% faster per K iteration
  than an fp8 TN grouped GEMM with the same 256×256 tile, which also differs in wave count and in
  accumulator registers.

## 10. Operand layout: preshuffle or permute in the fill; keep rows on cache lines

There are two ways to get an operand into MFMA order without a runtime shuffle:
- **Preshuffle it on the host.** FlyDSL's PS takes B as `[N/16, K/kpack, 4, 16, kpack_bytes]`, so each
  `buffer_load_dwordx4` lands in MFMA register layout and B skips LDS (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §12));
  the MXFP4 kernels take `shuffle_weight_w4` / `shuffle_scale_w4` layouts. The cost is a host-side pass, paid
  once for weights that do not change, as in inference.
- **Permute rows in the fill.** Changing only which global row each G2S load fetches reorders a lane's
  n-fragments with no runtime cost, and the LDS image, swizzle and reads stay the same
  (`compute_global_swizzle_pair`, GH:256-267; epilogue use in `epilogue.md` §3). Turbo's dense
  paths use this and need no offline step, which matters in training, where weights change every step.

- **Keep every operand row on a 128 B line.** Take the row stride from the tensor, not from K, so a
  caller can pad rows (`row_bytes`, FP4). For MXFP4, `K % 256 == 0` is what puts the row stride `K/2` on
  a 128 B line; it is not a kernel constraint. *(field, E26)* With the block count fixed, row phases of 64,
  32 and 96 B cost +8.4%, +11.6% and +12.0%. Pad K when memory allows.

## 11. MX scales

- **Pack four E8M0 bytes per dword.** The scaled MFMA picks a byte with `op_sel` (`scale_opsel(k, pack) = k % pack`,
  GH:114). The field is 2 bits, so 4 is a hardware limit. What packing buys depends on whether the scale loads are
  exposed. *(field, E28)* In a dense per-K loop that already prefetched its scales one step ahead, a probe that reused
  the k = 0 scales put the scale cost at about 0, so packing had nothing to recover there. *(field, E27)* Where the
  scale loads were exposed, on a grouped GEMM's variable-K wgrad, going from 1 to 4 bytes per dword cut VMEM
  instructions by 14% and gained 5–8%.
- **A wrong `op_sel` silently scales by another K block's exponent.** Probes with `randn`-derived scales
  cannot see it, because all exponents are similar. Use random exponents in the correctness probe.
- **Scales can skip LDS.** Turbo's MXFP4 loads them with `buffer_load_dwordx4` straight into pinned VGPRs
  (FP4:1393-1417). The LDS path delayed consumption by the refill-ahead distance, so a direct load
  must offset its K position by that lag.
- **Do the scale preshuffle in the same host call.** A caller that reuses weights can preshuffle their
  scales once (`preshuffle_mxfp4_scales`, FP4:4877).
- Never store E8M0 scales with a non-temporal hint (`epilogue.md` §5).

In FlyDSL: `../flydsl/gemm.md` § datapath §11.

## Related
`mainloop.md` · `schedule_and_locality.md` · `epilogue.md` · `correctness.md` ·
`attribution.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The Kimi-K3 forge runs installed `flydsl==0.3.2`. Where a record is not dated, the month is that of the
earliest record that has the number.

- E1 [2026-08, gfx950 (the source does not name the GPU), FlyDSL ?, N=?, ruler not stated] FlyDSL's own source comment: on the bf16 sync path at `tile_k=256`, 1011 TF/s with the fixed `Swizzle<3,3,3>` against 417 TF/s with the swizzle derived from `tile_k` (FlyDSL `preshuffle_gemm`, shape not stated; ROCm/FlyDSL #1008, commit 3a02102e, read at dd8373349:kernels/gemm/preshuffle_gemm.py:284-288)
- E2 [2026-07, gfx950, FlyDSL ?, N=?, rocprofv3 LDSBankConflict] XOR-ing `W << 1` into the chunk index on the G2S write and on the transpose read took LDSBankConflict from 14% to 0.00% at a 1024 B chunk stride (grouped fp8 TN wgrad, 4 waves, two LDS pools, MoE expert shapes at m=2048 and 4096; Primus-Turbo commit bd4bd3a8; local:/workspace/code/gpt_oss_docker/.claude/memory/project_wgrad_occ_feed_bound.md:1504)
- E3 [2026-08, gfx950, FlyDSL ?, N=7, same-process interleaved A/B, ISA counts checked] `SQ_WAIT_INST_LDS` came to ~81% of the CU-cycle budget; halving each of three LDS read classes in turn (`ds_read_b128` 384 → 320 and 384 → 256, `ds_read_b64_tr_b16` 1536 → 1280) read −0.25%, −0.09% and −0.09%, all noise (base won 3/7, 3/7, 2/7), recorded as 0.0% (gpt-oss fused attention backward, B=3 S=8192 Hq64 Hkv8 D64; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_003925/goal.md:37, :130-135)
- E4 [2026-10, gfx950, FlyDSL 0.3.2 with AMD LLVM 22.0.0git, N=40, s_memtime ticks, median per variant] one wave, 4 B per lane, an inline-asm load from global memory, then a wait: `flat_load_dword` + `lgkmcnt(0)` 96 ticks, `global_load_dword` + `lgkmcnt(0)` 56; `vmcnt(0)` 244 (flat) and 242 (global) on a line read before, 1032 and 984 on first touch (probe p8_flat_lgkm.py; local:/workspace/code/gpt_oss_docker/work/turbo_port/fe032/raw/out_p8_fd0.3.2.txt:115-119)
- E5 [2026-05, gfx950, FlyDSL ?, N=? for timing and 1000 runs for determinism, TF/s through the public entry] issuing both operands' `ds_read_b64_tr_b8` in inline asm, which removes the compiler's `vmcnt(0)` drain: 2580 → 2768 TF/s (+6.8%); `vmcnt_hint=3` stayed deterministic over 1000 runs, `vmcnt_hint` ≥ 4 raced (dense fp8 tensorwise TN, 8192×8192×28672; Primus-Turbo commit ba6d40f8)
- E7 [2026-08, gfx950, hand-written HIP probe, N=?, LDS placement check, then the campaign bench] `buffer_load_dwordx4 … lds offset:N` adds N to the LDS address (M0 + N + TID·16) and to the memory address (the data landed at LDS dwords 256/512 while lane 0 still read element 0); a wave-major fill sharing one M0 write then read −0.23% and −0.46% in two samples against a 0.55% noise band and was priced at 0 (MXFP8 grouped GEMM, gpt-oss, six balanced cells; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260813_160237/memory.md:197)
- E8 [2026-09, gfx950, FlyDSL 0.2.4, N=1 per arm, gemm18 scorer: geomean over 18 dense MXFP4 cells (6 gpt-oss, 12 Llama)] turning the LDS-DMA immediate off on all 18 cells read 0.9980 against a 1.0057 baseline (−0.77%); 16 of the 18 cells use it, recorded as −0.87% per affected cell; the hot loop grew from 381 to 404 instructions, all `s_add_u32 m0` (Primus-Turbo MXFP4 dense GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/memory.md:134)
- E9 [2026-10, gfx950, FlyDSL ?, N=3, six-shape A6W4 forward scorer] a wave-major A fill with the K-tile offset in `soffset` and each step in the 12-bit immediate cut the hot loop from 949 to 879 instructions; the scorer read 3964.08/3954.31/3951.09 against 3950.12/3952.65/3956.84, recorded as 0% (A6W4 forward GEMM, six training shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/memory.md:178)
- E10 [2026-08, gfx950, FlyDSL ?, N=1 per probe, gm_tf of `_bench_campaign_mxfp4.py` in the same session as baseline reads 4076.3/4065.3] `buffer_load_dword` in place of `dwordx4`, same lines touched: 4082.0 (+0.14%); every lane on one address, so one instruction touches 1 line instead of 8 (timing only, wrong output): 4901.6 (+20.2%) (gpt-oss MXFP4 grouped GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_140240/memory.md:396)
- E11 [2026-10, gfx950, FlyDSL ?, N=1 (dense rows) and N=5 (re-tiled) against 3 baseline reads, six-shape A6W4 forward scorer] fp6 A stored at 96 B per row: 2847.61 (−26.6%); stored densely with a 64-row preshuffle: 3959.89/3960.54/3953.53/3950.35/3960.10 against 3901.73/3881.19/3878.16, median +2.03% (A6W4 forward GEMM, six training shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/memory.md:164)
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=50 graph replays (median), scratch CUDA-graph timer] `tile_k` 32 with 4 stages against the current `tile_k` 64 rows: m=2732 220.40 against 190.35 µs, m=16384 1364.94 against 1204.59 µs (Kimi-K3 bf16 GEMM N=6144 K=7168, forge iteration 3; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6144_k7168/forge_experiments/lessons/iter_003.md:28-30)
- E13 [2026-10, gfx950, FlyDSL 0.3.2, N=? (iteration 2: driver --bench-mode, 3 warm-up and 10 timed calls; iteration 6: timer not stated for the item)] `tile_k` 32: m=2732 with 4 stages 220.6 µs against 198.4 for 2 stages of `tile_k` 64; m=16384, 256×256 with 4 or 3 stages, 1.258 / 1.286 ms against 1.110; m=893 with 5 or 6 stages 84.42 / 85.22 µs against 72.74 (Kimi-K3 bf16 GEMM N=6016 K=7168, forge iterations 2 and 6; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6016_k7168/forge_experiments/lessons/iter_002.md:15-28; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6016_k7168/forge_experiments/lessons/iter_006.md:29)
- E14 [2026-10, gfx950, FlyDSL 0.3.2, N=? (m=80/112: own script without L2 flush; m=2732: driver single-case bench; m=9397/16384: CUDA-graph script with a 512 MB flush, median of 4 interleaved repeats)] `tile_k` 32: m=112 / 80 with 5 stages 44.69 / 44.06 µs against 34.02 / 30.78; m=2732 with 4 / 3 stages 0.306 / 0.318 ms against 0.294; with 4 stages m=9397 903 → 1010 µs and m=16384 1615 → 1792 µs (Kimi-K3 bf16 GEMM N=8448 K=7168, forge iterations 2, 6 and 8; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n8448_k7168/forge_experiments/lessons/iter_002.md:36-37; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n8448_k7168/forge_experiments/lessons/iter_006.md:22-27; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n8448_k7168/forge_experiments/lessons/iter_008.md:7-9)
- E15 [2026-08, gfx950, FlyDSL ?, N=? (median), isolated grouped-GEMM bench] down forward with the contraction K = 2880 (fp8 row pitch ≡ 64 mod 128 B): 2196 TF/s; K padded to 2944: 2554 TF/s; only N padded: 2202 (grouped fp8 GEMM, gpt-oss down forward, M=131072, G=32 balanced; local:/workspace/code/gpt_oss_docker/.claude/memory/feedback_grouped_gemm_bench_pitfalls.md:37-38)
- E16 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 TCP_TCC_WRITE_REQ] pairing two output columns per lane so that one store covers 64 B: `TCP_TCC_WRITE_REQ` 2.359e7 → 1.206e7 (−48.9%), `TCP_TCC_READ_REQ` unchanged; widening further to `dwordx2` left it at 1.206e7 (gpt-oss-20b down projection, grouped tensorwise fp8 forward, G=32, M=131072, N=K=2880; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/goal.md:29; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/memory.md:220)
- E17 [2026-09, gfx950, FlyDSL ?, N=1 per arm, baseline read at both ends of a palindromic order (1559.27 / 1569.43 µs), 90 s warm-up] removing only the 64 `buffer_store_dwordx2`: 1471.60 µs (+6.30%); only the 128 `v_cvt_pk_bf16_f32`: +0.43%; only the `v_accvgpr_read_b32` layer: +0.52%, both within noise; all three: +7.75% (Primus-Turbo MXFP4 dense GEMM, Llama FC1_fwd row, K=4096; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_041317/memory.md:414-419)
- E18 [2026-10, gfx950, FlyDSL ?, N=61 paired rounds, in-process paired probe] on linear1 at a 256×256 tile, removing `nt` from the C store cost +10.4% with half-row stores (store_mode 0) and +10.8% with whole-row stores (store_mode 1), 0 of 61 rounds faster; on linear2, whole-row against half-row stores, both cached, read +0.07% (A6W4 forward GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/rounds/round-03/agent.txt:17)
- E19 [2026-10, gfx950, FlyDSL ?, N=1 per scorer read, linear1 drifting 522.4–525.5 µs in the session] a later round credited that 10.4% to whole-row coalescing by subtracting linear2's +0.07% arm; after a 256×384 tile had switched coalescing off, restoring it with `nt` unchanged took linear1 from 533.9 to 522.4 µs (−2.2%, spread 0.43%) (A6W4 forward GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/memory.md:209-210, :220, :225)
- E20 [2026-09, gfx950, FlyDSL 0.2.4, N=1, static ISA count (FLYDSL_DUMP_IR)] widening the tile from 128×256 to 128×512 (LDS row stride 512 → 1024 B) took `ds_read` from 84 to 232 for twice the work (×2.76, +38% per unit of work) as `ds_read2st64_b32` pairs stopped fusing (gpt-oss MXFP4 fused dual-cast kernel; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_085930/rounds/round-15/agent.txt:56)
- E21 [2026-07, gfx950, FlyDSL ?, N=?, ruler not stated] a 3-stage B ring with a longer prefetch distance measured +0.05% (neutral) where eight experts' weights thrashed the 4 MB per-XCD L2 (grouped GEMM forward, shapes not stated; recorded by 2026-07-05, myskill 6afa91f; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/03-lds-l2-datapath.md:28)
- E22 [2026-06, gfx950, FlyDSL ?, N=?, TF/s] B loaded straight to VGPRs, bit-exact, reached 77.8% of the throughput of B through LDS, recorded as ~22% slower (MXFP4 8-wave GEMM, 4096×4096×8192; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_8wave_opt.md:98)
- E23 [2026-07, gfx950, FlyDSL ?, N=? (min/median), TF/s in one session] A loaded straight to VGPRs: 1990/1982 TF/s (min/median) against 5033/5006 with A through LDS, 2.5× slower (MXFP4 8-wave GEMM, 8192×8192×28672; local:/workspace/code/gpt_oss_docker/.claude/memory/agpr_phase5_lds.md:48-54)
- E24 [2026-06, gfx950, FlyDSL ?, N=?, ruler not stated] with the rebased descriptor's base and num_records left in VGPRs, every K-loop `buffer_load` went through a readfirstlane waterfall: gate_up −16%, down −13% (grouped fp8 GEMM, MoE gate_up and down, shapes not stated; recorded 2026-06-17, myskill 2563ed1:flydsl-fp8-gemm-tuning/05-int64-addressing.md:30; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/07-epilogue-addressing-transpose.md:217)
- E25 [2026-07, gfx950, FlyDSL ?, N=?, wgrad geomean of `_bench_tw_vs_mx.py`] the A pools read with plain `ds_read_b128` (half the DS instructions, same bytes) instead of `ds_read_b64_tr_b8`: wgrad score 0.971 → 0.566, recorded as −80%; probe invalid (gpt-oss tensorwise fp8 grouped wgrad, whole-loop; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260731_073404/memory.md:194)
- E26 [2026-08, gfx950, FlyDSL ?, N=? (interleaved A/B), ms] 12 K blocks held fixed, fp4 row stride phase 0 / 64 / 32 / 96 B: 0.2753 / 0.2985 (+8.4%) / 0.3072 (+11.6%) / 0.3082 ms (+12.0%) (Primus-Turbo MXFP4 dense GEMM, M=32768, N=5120, K=3072/2944/2880/3008; Primus-Turbo #487, commit f6d5ab685, first committed as a5606e87)
- E27 [2026-07, gfx950, FlyDSL ?, N=?, geomean of tw/mx over 14 cells (`_bench_campaign_mxfp8.py`) and per-workgroup PMC] wgrad with four E8M0 scales per dword: `SQ_INSTS_VMEM` per workgroup 3947 → 3387 (−14.2%), MFMA and VALU unchanged, six configurations +5.4–8.2% (gpt-oss MXFP8 grouped GEMM, variable-K wgrad; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:109)
- E28 [2026-07, gfx950, FlyDSL ?, N=?, ruler not stated] in an MXFP8 dense per-K loop that already prefetched its scales one step ahead, a probe that loaded only the k=0 scales and reused them put the scale tax at ≈0% (all 4 shapes ≤ 1%, some slower) (MXFP8 dense GEMM, 4 shapes not named; recorded by 2026-07-05, myskill 9d3f961; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:205, :365)
- E29 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile and run, final ISA read] two `BufferCopyLDS128b` DMAs, an inline-asm `s_waitcnt vmcnt(1)` + `s_barrier`, then one `ds_read_b128`: the plain `llvm.load` got `s_waitcnt vmcnt(0)` before it, the same load with an `alias_scopes` attribute none; output correct in both (probe kernel; local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/knowledge_patch_v2/probe_alias/probe.py, ISA in dump0/ and dump1/probe_kernel_0/21_final_isa.s). On a fused-MoE stage 2 the same change removed the `vmcnt(4..12)` the compiler put before its A reads (local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/workspace/forge_experiments/lessons/iter_011.md:24; :27)
- E30 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, in-process probe, L2 flushed before each call, randn inputs, every arm against a same-process control; arms equal to the A-through-LDS build except the diagnostics] bf16, m=512, 256×192 tile, 8×1 waves (32 A rows per wave): A through LDS 96.6–97.8 µs; A in the MFMA lane layout 113–118 (2 A stages), 106.0 (3 stages); quad-contiguous loads + `ds_bpermute` 107.7–110.0; permlane/DPP register transpose 124.8 (7 spills) and 127.5 (248 VGPRs, none); contiguous-load diagnostics, wrong output, 92.7–96.1 against controls of 95.3–96.7 (bf16 GEMM n6144 k16384; local:/shared_nfs/kyle/moe/k3six/runs/gemm_l/workspace/forge_experiments/lessons/iter_012.md:3; :7)
