---
title: "Grouped GEMM — fp8 tensorwise, MXFP8 and MXFP4"
kind: topic
lever: grouped_lowprec
ops: [grouped_gemm]
gens: [gfx950]
dtypes: [fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, skew, fused-epilogue, quantization]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_glu_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp8_glu_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_glu_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/pytorch/ops/grouped_gemm_fp8.py
---

# Low precision in a grouped GEMM

> What fp8 with per-tensor scales, MXFP8 and MXFP4 add to a grouped GEMM: per-group scale tensors and their
> preshuffle, scale packing, scale delivery, quantization cost, padding, the C store, and GLU/dGLU fusion. The
> scale and MFMA mechanics that are the same as in a dense GEMM are in `../gemm/datapath.md` §11 and
> `../gemm/mainloop.md` §5–§6 and are not repeated. "FP8", "MX8", "MX4", "FP8G", "MX8G", "MX4G" are
> Primus-Turbo's `grouped_gemm_{fp8,mxfp8,mxfp4}{,_glu}_kernel.py` under `primus_turbo/flydsl/grouped_gemm/` and
> "OPS8" its `pytorch/ops/grouped_gemm_fp8.py`, at main `ca1fdd67`.
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end. Use
> them for direction and order of magnitude, not as targets.

## Route here when
- You add fp8 or MX formats to a grouped kernel, or a quantized grouped kernel is slower than its bf16 or
  tensorwise sibling.
- A scale preshuffle, quantizer or padding kernel shows up next to the GEMM in a trace.
- You are choosing the C store or cache policy for grouped outputs, or fusing a gated activation.

## 1. What the group dimension adds
- **Tensorwise fp8**: two scalars per GEMM, applied once in the epilogue (`a_scale * b_scale`); nothing per group.
- **MX**: E8M0 scales per 32 contraction elements and per group. The quantizer pads each group to the scale
  granule, so the kernel reads through a padded offset table and writes through a tight one: MX8 takes
  `group_offs` (padded read) and `group_offs_out` (tight write, MX8:1244-1247). Its preshuffle stages each group
  into its own 64-row slabs so a 32-aligned group base needs no 64-alignment from the quantizer (MX8:263-264).
- **MX dgrad runs as NT on a column-quantized weight** (OPS8:806): the preshuffled block scale of B is "only
  walkable by the NT addressing" (MX4G:21-23). Tensorwise dgrad stays NN with transpose reads.
- **MX wgrad takes column-quantized operands `[OUT, M_total]`**; K tails then need a run-time guard
  (`grouped_wgrad.md` §3).
- **Quantization runs every call for activations and gradients.** Weights quantize once per optimizer step and
  the backward reuses them; a benchmark that passes high-precision weights re-quantizes them per call. *(field, E1)*
  Pre-quantizing weights took one fused MoE MLP forward from 4.28 to 3.47 ms; the backward did not change.

## 2. The scale preshuffle is a per-call helper kernel
Scales arrive per call (activations, gradients), so the repack into the MFMA layout runs every call. Launch it
from the GEMM's host launcher, so the pair costs one launcher call and runs in stream order (MX8:1101-1120;
`../gemm/datapath.md` §11). Then find out what bounds it.
- **E8M0 repack (MXFP8) is dispatch- and latency-bound.** *(field, E2)* On the measured build it took 0.8–1.5% of the
  call's wall time at an effective 2.6–2.9 TB/s, a third of HBM bandwidth. *(field, E3)* Only cutting the workgroup count
  paid: one workgroup taking two 64-row slabs halved the A-side workgroups, +1.04% on the scored total, 0.45 points
  of it from a change of machine; a palindrome probe of the pairing alone read within its 1.3% same-arm spread. More
  threads per workgroup (256 → 512) was 24–28% slower (twice the waves meet at each LDS barrier; *(field, E32)* an
  MXFP4 repack with no LDS barrier was not slower at 512 threads); dword → dwordx4 reads were net zero. The
  binding budget is LDS per resident workgroup (160 KB / 8 = 20 KB at eight per CU), not the 160 KB total: pair
  slabs only while eight still fit. The staged row pitch moves the best k-tile, so
  re-sweep it after a pitch change.
- **MXFP4 repack can be VALU-bound.** *(field, E4)* A per-thread O(G) group scan to map 64-row slabs back to source
  rows ran 759 VALU per wave for 8 useful dwords per thread; the lane-resident group table of
  `grouped_group_mapping.md` §5 cut it to ~206 and the kernel from 35.3 to 16.3 µs, +1.1% geomean over 18
  cells. Rule: more than ~10 VALU per useful dword means index arithmetic owns the kernel; widening stores or
  staging in LDS will not help.
- **Its cost is not what removing it saves.** *(field, E5)* Turning the MXFP8 repack off made the call 6 µs slower: it
  had warmed the scale workspace in L2 for the GEMM. *(field, E6)* A trace overstates such kernels 2–3× against their
  in-stream cost (`grouped_host_autotune.md` §7).

In FlyDSL: `../flydsl/moe.md` § grouped_lowprec §2.

## 3. Pack four E8M0 per dword — per group on wgrad
- `op_sel` picks one of four bytes, so four is the hardware ceiling (`../gemm/datapath.md` §11).
- **NT (forward, dgrad).** The k-step's byte is `k % 4`, known at compile time. MX8 packs NT only
  (`_GG_SCALE_PACK = 4`, MX8:74-76). *(field, E7)* The repack took 32% less time (14.3 → 9.7 µs), the GEMM was
  unchanged and bit-exact, end to end an estimated ≈1%: on that build the repack was ~3.5% of the forward/dgrad time (§2
  has another build's share).
- **Variable-K wgrad.** A group's contraction starts at any 128-element boundary, so a global `k % 4` byte select
  is wrong for most groups; Turbo's MXFP8 wgrad stays unpacked for that reason. Pack each group from its own
  start instead: base `ks0 // 4 + g` for a group starting at 128-block `ks0`, row-group stride `K128 // 4 + G` (the
  `+g` / `+G` is the at most one partial dword per group; `floor(a/p) + ceil(n/p) ≤ floor((a+n)/p) + 1` keeps the
  regions disjoint). The alignment work moves into the repack kernel, which runs once per call; the GEMM adds one
  integer add.
  - *(field, E8)* MXFP8 variable-K wgrad, six cells: +5–8% (scale loads were exposed there; VMEM per workgroup
    −14.2%, the byte selects went to the separate scalar port).
  - *(field, E9)* Packing that assumed 512-aligned group starts read 4.09 and −2.44 dB SNR on unaligned groups while
    the determinism gate and a check that the repacked and raw-scale paths agree both passed. *(field, E10)* The per-group form read
    55.56–55.64 dB on 16 layouts (128- but not 512-aligned starts, empty first/middle/last groups, ragged G=32),
    and pack 1 and pack 4 gave byte-identical outputs. Gate packing with unaligned group starts.
- **Where packing buys nothing.** *(field, E11)* In a dense per-K loop whose scales were already prefetched one step
  ahead, a probe that reused the k=0 scales measured the scale tax at ≈0. Pack where the scale loads are exposed.

## 4. Scale delivery in NT: the floor for its scale-slab layout
- *(field, E12)* MXFP8 8-wave NT: scale loads to VGPRs and the operand DMA share `vmcnt`, and each K iteration's drain
  retires both, so a scale prefetch can reach at most one iteration ahead; prefetching a whole pack group, or
  spreading the scale loads evenly, measured +0.17% and +0.08%, inside the noise. The main loop's scale loads were
  already dwordx4; the narrow loads in the ISA histogram were the prologue's group-offset scan. Split ISA counts
  by region first.
- *(field, E13)* MXFP4 whole loop: removing the two scale `buffer_load_dwordx4` per phase was worth 4.69%; a probe that
  froze the scale addresses (same instructions) split it into 1.79% memory and 2.90% structure. Deeper scale
  prefetch was flat. One scale dword serves four MFMAs through `op_sel`; eight row blocks × two sub-steps need four
  dwords per operand per phase, and dwordx4 is the widest buffer load: two loads per phase is the floor for this
  slab layout. Going lower means re-laying the slab (two dwords per phase, each serving two MFMAs), which moves
  bytes into the preshuffle.
- The scaled MFMA itself costs nothing extra. *(field, E14)* Dense: scaled 3068 vs unscaled 3093 TF/s; grouped wgrad
  807 → 807 µs with the scale operand removed. Scale cost is load issue and registers, not the instruction.
- MX tile heights are fixed by the scale layout. In an MXFP4 grouped kernel whose A-scale repack groups
  64 rows and whose wave regions walk 128, `BLOCK_M` must be a multiple of 128: a 288-row tile puts a wave at row
  144 and selects the wrong byte (S1). Re-lay the scale slab before trying odd tile heights.

## 5. Quantization: a large share of the forward, a small lever
- *(field, E15)* Grouped MX quantization was 35–42% of the forward's absolute time on four shapes at 4096 tokens per
  group. A bit-exact fused dual-cast quantizer still left the backward unchanged and moved the forward little: 60%
  of the forward's gap and 90% of the backward's gap sat in the GEMMs. A share of time is not a share of the gap,
  and the difference between two speed ratios is neither.
- **Fuse the per-call metadata into one prologue kernel.** *(field, E16)* Each quantizer workgroup ran an O(G) group search over
  three offset tables; in a short memory-bound kernel that exposed ~42 µs. One prologue thread per M block does
  the search once and writes the block's row base, offset and end, launched back to back from the same host
  launcher as the quantizer. *(field, E16)* Kernel 155 µs, call 181 µs, faster than the HIP quantizer. Moving the
  search to host-side torch ops gave a 153 µs kernel and a 236 µs call: ~10 tiny launches added 80–120 µs.
- **Quantize all expert weights in one launch.** Batch the dense quantizer over `[G, N, K]` with a per-batch scale
  byte base, dword-aligned so neighbouring experts' packed scales do not share a dword. *(field, E31)* 1.3–1.54×
  faster than the HIP kernel; a Python loop over experts was 2.7–4.3× slower than HIP.

In FlyDSL: `../flydsl/moe.md` § grouped_lowprec §5.

## 6. Padding goes on the contraction dimension
- A row pitch ≡ 64 mod 128 B makes every other row straddle two lines (`../gemm/datapath.md` §6, §10).
  *(field, E17)* Tensorwise fp8 grouped forward, G=32 balanced, K and N both ≡ 64 mod 128: padding neither / K
  only / N only / both gave 2196 / 2554 / 2202 / 2554 TF/s. The whole +16% came from the contraction pad; the N
  pad matters for dgrad, where N becomes the contraction. Count FLOPs with the real dims.
- *(field, E18)* The same pad (to 128) for a fused SwiGLU MLP whose hidden and expert widths are both ≡ 64 mod 128:
  forward 8.2% and backward 2.6% less time.
- **When the operand cannot be padded:** FP8's NN dgrad reads each row's enclosing line-aligned window when
  `K % 128 == 64` (`nn_kshear`, FP8:218, 277-280; its comment reports +25.7% L1→L2 read requests without it).
  For MX, pad only the scales: zero E8M0 past K makes over-read terms exactly zero, so the operand keeps its real
  K (`k_real`, MX4:364-373; `../gemm/mainloop.md` §6).
- Never re-pad a weight per call: `grouped_host_autotune.md` §4. Free-dim padding for wgrad outputs:
  `grouped_wgrad.md` §12.

## 7. The C store for grouped outputs
The store classes and their measured prices are in `../gemm/epilogue.md` §2–§4, cache bits in §5 and
line-aligned column blocks in §1. "Folded" below names a store form, not a split-K fold.
What grouped adds:
- **The store is bytes, not VALU.** *(field, E19)* MXFP4 grouped NT: deleting the C-store epilogue saved 61% of the
  per-tile fixed cost (4.87 of ~7.9 µs per tile, fitted on that build; the F/P fit is in
  `grouped_tile_pricing.md` §6); deleting 254 `v_accvgpr_read` and 254 conversions while keeping every
  store saved 0.5–1.0%, hidden behind the store drain.
- **With narrow (2-byte) folded stores, do not issue consecutive stores to the same 128 B line.** *(field, E20)*
  MXFP4 grouped at one wave per SIMD: grouping folded stores by output row so four in a row fill one line:
  −0.76% and −0.52%; issuing by accumulator, four lines in turn, was faster.
- **Cache-policy bits on C (grouped MX kernels, beta=0 stores): only `nt`, only on whole-row stores, and by the
  number of slabs competing for L2.**
  - *(field, E21)* 2-byte scalar stores with `sc0` (aux = 1): 8 of 8 MXFP8 NT cells 0.2–1.7% slower.
  - *(field, E22)* Narrow folded 2-byte stores with `nt`: −4.5% on one build, no gain on another, where DRAM write
    requests rose 54%, because a store's 32 B pieces merge into 64 B writes only in L2.
  - *(field, E23)* Whole-row folded stores (16 lanes × 8 B fill a 128 B line) with `nt` at G=32 thin groups: wgrad
    +2.1–2.4%, NT +0.88 points; L2 hit 59.2 → 61.2%, DRAM reads −72 to −99 MB, write bytes unchanged. At G=4
    with large M the same bit was flat or slightly negative. The gain is the weight residency C stops evicting,
    so it scales with the expert slabs live in L2 at once; choose it from the runtime shape. MX4 keeps a
    separate thin-group table entry with `nt` on (`_GMXFP4_NT_CFG_THIN`, MX4:1100-1101; selector
    MX4:1164-1170).
  - *(field, E24)* `sc1` alone or added to `nt` cost 4–5% on six wgrad cells: it sends the writes past the per-XCD L2
    that merges them.
- E8M0 scale stores are never non-temporal (`../gemm/epilogue.md` §5).

## 8. Comparing MX with tensorwise
- **A fair comparison.** Both on the same GEMM backend; the tensorwise path in its native layouts (an option that
  forces NT in the backward adds transposes of both operands); quantization inside the timed region, fresh each
  step. MX's reference is a scaled MX GEMM, not an unscaled per-tensor one.
- **Find what the gap is made of before attacking it.** Compare memory instructions per MFMA, steady-state time per
  K iteration, synchronization instructions per MFMA and K iterations per tile. *(field, E25)* An MXFP8 8-wave NT kernel
  against the tensorwise 4-wave wgrad it had to beat led on the first two and lost on the last two
  (`grouped_tile_pricing.md` §2, §6); DRAM bytes, L1→L2 requests and power were level. Its extra
  synchronization is not free to delete: the 8-wave barriers keep the waves in phase (§2 there).
- **Instruction trimming in the steady state pays only when non-MFMA issue demand approaches the MFMA shadow.**
  *(field, E26)* Cutting the MXFP4 grouped loop from 442 to 419 (NT) and 410 to 387 (wgrad) instructions, all SALU,
  changed wall time by 0: non-MFMA issue demand was 0.25–0.28 of the MFMA shadow. Below ~0.5, do not invest.
- At the power cap a cycle saving converts to wall at a shape-dependent rate
  (`../gemm/attribution.md` §4).

## 9. Why MXFP4 grouped does not reach 2× MXFP8
- *(field, E27)* The fp4/fp8 grouped speed ratio was 1.32–1.69 (geomean 1.51 over 18 cells): lowest on shallow-K wide
  shapes, closest to 2 on deep K. The fp4 kernel ran 4-wave workgroups at one wave per SIMD (the fp8 kernel 8-wave
  workgroups at two) and its MFMA pipe starved for operands: MfmaUtil was 36.6% on the lowest-ratio cell and
  51.6% on the highest. Compare utilization counters across wave counts only after
  `../gemm/attribution.md` §9.
- Raising fp4 occupancy by splitting the 256×256 tile over eight waves lost with both pipeline forms
  (`grouped_tile_pricing.md` §2). Raise the per-wave feed at one wave per SIMD instead.

## 10. GLU and dGLU fused into the epilogue
- **Accept on fused total against unfused total**, not on the fused kernel's TF/s against a bare GEMM
  (`../gemm/epilogue.md` §8). *(field, E28)* Fused glu (GEMM + SwiGLU) ran ~14–15% and fused dglu ~28–29%
  slower than a bare GEMM, and both were net wins on the total. The gap is structural: glu writes a third stream
  (the activation) that one workgroup per CU cannot hide; dglu must stream the pre-activation in and its gradient
  out, about twice the HBM traffic (an estimated 24% roofline floor at 8 TB/s). *(field, E33)* `waves_per_eu` 2/3/4
  changed nothing on glu.
- **Gated activations with a padded half: compute at the pitch, store at the real width.** With gate and up halves
  of width `I` padded to `Ip`, the weight shape, loop bounds and the up half's column offset use `Ip`; the store
  width, column masks and the compact `[M, 2I]` output use `I`, and pad columns are written as zero. Plain zero
  propagation is right for forward and dgrad but leaves the weight gradient with pad rows between gate and up.
- **Port the plain kernel's store wins into the fused epilogue.** *(field, E29)* Paired-column stores in fused glu:
  +3.7%; it has three output streams to gain on.
- **Measure the kernel you changed, not an average of glu and dglu.** *(field, E30)* A +0.55% glu-only gain, clean over
  five readings, was invisible in the averaged score.
- **Turbo's forms.** fp8: the forward emits the pre-activation and the activation together, and the fc2 dgrad
  consumes the activation gradient in registers so it never reaches HBM, both through the NT/NN mainloops behind
  `glu`/`dglu` flags (FP8G:14-21). MXFP8: GEMM, then a separate GLU + dual-quantization kernel, because folding
  the epilogue "pushes the tile past 128 VGPRs and into scratch" (MX8G:59-61). MXFP4: both directions NT
  (MX4G:21-23). Operand pairing for gate/up in one lane: `../gemm/epilogue.md` §8.

## Verify
- Packed-scale kernels are gated with unaligned group starts and random E8M0 exponents, against a reference.
- Each helper kernel (repack, quantizer, padding) was priced in the stream and launches from the GEMM's host launcher.
- The padded dimension of every operand is its contraction dimension, checked per pass.
- `nt` on C is enabled only for whole-row stores and was measured at the deployed group count.
- Fused GLU/dGLU was accepted on total time, with pre-quantized weights.

## Related
`grouped_wgrad.md` · `grouped_group_mapping.md` · `grouped_tile_pricing.md` ·
`grouped_host_autotune.md` · `grouped_correctness.md` · `grouped_measurement.md` ·
`../gemm/datapath.md` · `../gemm/epilogue.md` · `../gemm/mainloop.md` ·
`../gemm/attribution.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. "Turbo" is Primus-Turbo, and `Turbo <rev>` the base or HEAD commit the record names. A line whose only
reference is a myskill or memory note gives the month of that note ("recorded by"); such notes rarely state N. Secondary
references in a line sit in the directory of its first `local:` path unless they carry their own.

- E1 [2026-08, gfx950, FlyDSL ?, N=?, same-process A/B on one GPU, pad-128 arm] fused MoE MLP forward 4.28 ms with bf16 weights quantized inside every call against 3.47 ms with pre-quantized weights; the backward reuses the forward's fp8 weights (gpt-oss-20b MoE MLP, fc1+SwiGLU and fc2 grouped fp8 GEMMs, H=I=2880, G=32; Primus-Turbo #483 branch; local:/workspace/code/gpt_oss_docker/.claude/memory/project_fused_mlp_padnk_campaign.md:116; /workspace/code/gpt_oss_docker/myskill/optimizer/methodology/18-operator-fusion.md:64)
- E2 [2026-08, gfx950, FlyDSL ?, Turbo 23bdcdc7 base, N=?, rocprofv3 kernel trace, preshuffle share of the main kernel within the same trace] MXFP8 E8M0 preshuffle (`kern_0`) 0.82–1.53% of the call's wall time, effective 2.6–2.9 TB/s against 8 TB/s HBM (33–36%) (gpt-oss MXFP8 grouped GEMM, G=4, M=131072, gate_up/down × fwd/dgrad/wgrad; campaign 20260805_150542 rounds 13 and 15; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/goal.md:378-379)
- E3 [2026-08, gfx950, FlyDSL ?, Turbo 23bdcdc7 base, N=?, campaign 18-cell bench (gm), kernel trace for `kern_0`, palindrome wall probe for the slab-pairing sweep] one preshuffle workgroup taking two 64-row A-scale slabs (A-side workgroups 2052 → 1026): round gain +1.04% on the scored gm, of which 0.45 points the campaign attributes to a change of machine; a later palindrome probe of AS ∈ {None, 1, 2, 4} read flat (same-arm spread 1.3%); `BLK` 256 → 512 threads +24–28% slower; dword → dwordx4 reads: `kern_0` net zero (same workload as E2; campaign 20260805_150542 rounds 5, 13, 14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/goal.md:4; memory.md:244; memory.md:298; memory.md:225; memory.md:148)
- E4 [2026-08, gfx950, FlyDSL ?, N=2, two bench runs of the kept change; kernel-trace median for `kern_0`; gm over 18 configs] MXFP4 grouped preshuffle with the lane-resident group table instead of a per-thread O(G=32) scan: VALU 759 → ~206 per wave, `kern_0` 35.34 → 16.32 µs (−54%), bench gm 1.48844 / 1.48601 against 1.4710 (+1.10%), min +1.98% (gpt-oss MXFP4 grouped GEMM, fwd/dgrad/wgrad; campaign 20260803_155024 round 6, chi2798 `/opt/venv-tw`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/rounds/round-06/agent.txt:7-8; :25)
- E5 [2026-08, gfx950, FlyDSL ?, N=?, the bench's per-op event timing (L2 cold, host enqueue inside the window)] MXFP8 grouped forward (down) 814.9 µs with the preshuffle against 820.9 µs with `preshuffle=False` (gpt-oss MXFP8 grouped GEMM, G=4; campaign 20260805_150542 round 15; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/memory.md:312)
- E6 [2026-09, gfx950, FlyDSL 0.2.4, N=?, in-process additive probe (0/1/2/4 extra launches of the same compiled kernel, slope fitted) against rocprofv3 kernel-trace Start→End] MXFP4 grouped helper kernels: preshuffle `kern_0` 20.32 µs traced vs ≈10.2 µs in stream (≈2×); an idle output-tail kernel 5.10 µs traced vs 1.579 µs in stream (3.2×) (gpt-oss-20b MXFP4 grouped MoE GEMMs; campaign 20260925_090609 round 10, `kyle_mxfp4_204` `/opt/venv-syncv4`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/goal.md:254; rounds/round-10/agent.txt:18)
- E7 [2026-07, gfx950, FlyDSL ?, N=?, ?] MXFP8 grouped NT scale preshuffle with four E8M0 per dword: 14.3 → 9.7 µs (−32%), GEMM unchanged, bit-exact on all shapes; the preshuffle ~3.5% of forward/dgrad time (Turbo MXFP8 grouped GEMM; shapes not named; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/11-mxfp8-dense.md:121)
- E8 [2026-07, gfx950, FlyDSL ?, N=?, campaign bench (`_bench_campaign_mxfp8.py`, tw/mx ratio over 14 configs) and per-workgroup PMC] variable-K wgrad with per-group packed E8M0 (`pack=4`): six wgrad configs +5–8% (round 4 record, gm 1.02127 → 1.04757); `SQ_INSTS_VMEM` per workgroup 3947 → 3387 (−14.2%), SALU +19%; myskill records +4.7–9.3% and geomean +2.7%, a range not found in the campaign's round records (the global `pack=4` of round 2 read +5.4–8.2%) (gpt-oss MXFP8 grouped GEMM, G=32, M=131072; campaign 20260729_093030 round 4, chi2798 `mlperf_gptoss` `/opt/venv-syncv3`; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:365-369; /workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:130; /workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/rounds/round-02/agent.txt:17)
- E9 [2026-07, gfx950, FlyDSL ?, N=?, SNR against the reference on unaligned group layouts; determinism gate; preshuffle=False byte comparison] global `pack=4` assuming 512-aligned group starts: SNR 4.09 dB / −2.44 dB on non-aligned starts while `det=True` and the bytewise preshuffle=False == full check both passed (same workload as E8; campaign 20260729_093030 round 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/rounds/round-02/agent.txt:17)
- E10 [2026-07, gfx950, FlyDSL ?, N=?, SNR against the reference over 16 layouts; byte comparison of pack 1 and pack 4] per-group packing: 55.56–55.64 dB on 16 layouts (128- but not 512-aligned starts, a single odd start, empty first/middle/last groups, G=32 ragged, two OUT_M variants); pack 1 and pack 4 outputs byte-identical (same workload as E8; recorded 2026-07-30, myskill 587b618; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:376-378)
- E11 [2026-07, gfx950, FlyDSL ?, N=?, ruler not stated] in an MXFP8 dense per-K loop that already prefetched its scales one step ahead, a probe that loaded only the k=0 scales and reused them put the scale tax at ≈0% (all 4 shapes ≤ 1%, some slower) (MXFP8 dense GEMM, 4 shapes not named; recorded by 2026-07-05, myskill 9d3f961; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:205; :365)
- E12 [2026-07, gfx950, FlyDSL ?, N=2 base readings, drift-immune 14-config bench (tw/mx in one process)] MXFP8 grouped NT scale delivery: prefetch distance 1 K-iteration → a whole pack group gm 1.08433 (NT eight configs +0.17%); the three scale dwords spread over the pack group 1.08483 (+0.08%); base 1.08492 / 1.08459 (gpt-oss MXFP8 grouped GEMM; campaign 20260729_093030 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/rounds/round-09/agent.txt:12-13; /workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:228-229)
- E13 [2026-08, gfx950, FlyDSL ?, N=1 per probe arm, same session, campaign bench gm_tf (18 configs)] MXFP4 grouped whole loop: removing the two scale `buffer_load_dwordx4` per phase 4071.3 → 4262.4 (+4.69%); freezing the scale addresses with the instructions kept 4144.1 (+1.79%), so 1.79% memory side + 2.90% structure; scale prefetch two phases deep 4070.8 / 4060.7 / 4057.5 against base 4066.4, flat (gpt-oss MXFP4 grouped GEMM; campaign 20260804_140240 rounds 16 and 17; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_140240/memory.md:354; :366)
- E14 [2026-07, gfx950, FlyDSL ?, N=?, probes NOSCV / NOSCALE_MMA / NOSCMMA] the scaled MFMA instruction itself: dense MXFP8 down projection scaled 3068 vs unscaled 3093 TF/s (0.8%); grouped MXFP8 wgrad 807 → 807 µs with the scaled MFMA removed (recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:158-160)
- E15 [2026-07, gfx950, FlyDSL ?, N=?, absolute time split quantize vs GEMM, M=4096] grouped MX quantization 39 / 35 / 38 / 42% of the forward's absolute time on 4096×7168 / 8192×4096 / 4096×4096 / 2880×2880; a bit-exact fused dual-cast quantizer moved the forward 1.2 → 1.07× and the backward almost not; 60% of the forward's and 90% of the backward's gap in the GEMMs (Turbo MXFP8 grouped GEMM; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:296-309)
- E16 [2026-07, gfx950, FlyDSL ?, N=?, kernel trace for the kernel, wrapper time for the call] grouped MXFP8 quantizer: per-workgroup O(G) search exposed ~42 µs; fused meta prologue kernel 155 µs, wrapper 181 µs (faster than the HIP quantizer); the search moved to ~10 host-side torch ops: kernel 153 µs, wrapper 236 µs (+80–120 µs of launches) (Turbo MXFP8 grouped quantizer; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/12-mxfp8-grouped.md:45-47; /workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/02-measurement-noise.md:430)
- E17 [2026-08, gfx950, FlyDSL ?, N=?, median TF/s] down forward, M=131072, G=32 balanced, K and N ≡ 64 mod 128: pad neither / K only / N only / both = 2196 / 2554 / 2202 / 2554 TF/s (gpt-oss-20b tensorwise fp8 grouped GEMM, syncv3; local:/workspace/code/gpt_oss_docker/.claude/memory/feedback_grouped_gemm_bench_pitfalls.md:37-38)
- E18 [2026-08, gfx950, FlyDSL ?, N=?, same-process A/B, pre-quantized weights] pad128 vs pad0 on a fused SwiGLU MoE MLP: forward 3.785 → 3.473 ms (−8.24%), backward 6.969 → 6.791 ms (−2.55%) (gpt-oss-20b, H=I=2880, G=32; local:/workspace/code/gpt_oss_docker/.claude/memory/project_fused_mlp_padnk_campaign.md:116; /workspace/code/gpt_oss_docker/myskill/optimizer/methodology/18-operator-fusion.md:58)
- E19 [2026-08, gfx950, FlyDSL ?, N=?, direct ablations on the campaign bench; K-sweep fit of F] MXFP4 grouped NT: deleting the C-store epilogue 4.87 µs per tile (20.3% of wall, 61% of F, F fitted at 7.91 µs); removing 254 `v_accvgpr_read` and 254 `v_cvt_pk_bf16_f32` with every store kept −0.5% to −1.0% (gpt-oss MXFP4 grouped GEMM; campaign 20260804_140240; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_140240/memory.md:269; :250; :148)
- E20 [2026-08, gfx950, FlyDSL ?, N=1 per arm, same session, campaign bench gm_tf] folded C stores issued by output row so four fill one 128 B line (`_CRATE=2` / `_CRATE=4`): 4040.5 / 4050.2 against 4071.3 (−0.76% / −0.52%) (gpt-oss MXFP4 grouped GEMM, one wave per SIMD; campaign 20260804_140240 round 16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_140240/memory.md:354)
- E21 [2026-07, gfx950, FlyDSL ?, N=?, drift-immune 14-config bench] plain 2-byte scalar C store with `cstore_aux=1` (`sc0`): 8 of 8 NT configs −0.2% to −1.7% (gm −0.46%), wgrad unchanged in the same run (gpt-oss MXFP8 grouped GEMM; campaign 20260729_093030 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:130; /workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:438-439)
- E22 [2026-08, gfx950, FlyDSL ?, N=2, md5-gated samples per arm, interleaved, campaign bench gm_tf] `nt` on the narrow folded 2-byte C store: 3885.2 / 3924.7 against 4103.0 / 4109.0 (−4.49%) in round 24; on the folded store of round 15, `nt` cut EA reads 10.030M → 8.384M and raised EA writes 12.059M → 18.589M (+54%), kernel time 583.1 vs 579.2 µs, no gain (one PMC run per arm) (gpt-oss MXFP4 grouped GEMM; campaign 20260804_140240 rounds 15 and 24; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_140240/memory.md:435; :342; :344)
- E23 [2026-08, gfx950, FlyDSL ?, Turbo a8079ca7 base, N=3, bench repeats (median), deploy/large-M ratio; 6+6 interleaved paired samples for NT; PMC on deploy gate_up wgrad] `nt` on the whole-line folded C store (16 lanes × 8 B) at G=32: wgrad +2.1–2.4% (gm 96.1 → 97.0), L2 hit 59.21/59.25/59.94 → 60.74/60.94/61.18%, DRAM reads −72 to −99 MB, write bytes and `TCP_TCC_READ_REQ` unchanged; NT +0.88 gm (101.42 vs 100.54); at G=4, large M, flat or slightly negative (gpt-oss-20b MXFP4 grouped GEMM; campaign 20260811_114814 rounds 2 and 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260811_114814/rounds/round-08/agent.txt:21; memory.md:142; /workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:645-651)
- E24 [2026-08, gfx950, FlyDSL ?, N=3, interleaved A/B, 3 runs per arm (count from the myskill note), campaign score gm_ratio_pct against the large-M reference] on the folded C store whose 16 lanes fill a 128 B line (`buffer_store_dwordx2`) with `nt`: `nt sc1`, `sc1` alone and `sc0 sc1 nt` read 98.32 / 98.31 / 99.43 against 100.49–100.69, all six wgrad cells 4–5% lower (gpt-oss-20b MXFP4 grouped GEMM, G=32, per-expert M=4096, skewed routing, wgrad; campaign 20260811_114814 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260811_114814/rounds/round-13/agent.txt:16; /workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:671)
- E25 [2026-08, gfx950, FlyDSL ?, N=?, one rocprofv3 trace plus `FLYDSL_DUMP_IR` ISA of both kernels; power from rocm-smi in the same session] MXFP8 8-wave NT vs the tensorwise 4-wave wgrad ruler: steady state per K-iteration 1.164 vs 1.31–1.35 µs (11–14% faster), memory instructions per MFMA 0.52–0.60×; sync + scheduling instructions per MFMA 0.933 vs 0.126, K-iterations per tile 23 vs 134; `TCP_TCC_READ_REQ` 1.02×, HBM 1.90 vs 1.84 GB, power 1399.5 vs 1374 W (gpt-oss MXFP8 grouped GEMM vs tensorwise fp8 grouped wgrad; campaign 20260805_150542 round 16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260805_150542/goal.md:124; :139-140; :145; :178)
- E26 [2026-09, gfx950, FlyDSL 0.2.4, N=3, bench gm_ratio_pct, campaign floor ±0.44%; ISA census of the steady-state loop] SALU trims on the MXFP4 grouped loop: NT 442 → 419, wgrad 410 → 387 instructions, MFMA/LDS/VMEM unchanged; gm 100.15 → 100.15 / 100.20 / 100.28; non-MFMA issue demand / MFMA shadow 0.28 (NT) and 0.25 (wgrad) (gpt-oss-20b plain MXFP4 grouped MoE GEMMs; campaign 20260925_090609 rounds 1–2, `kyle_mxfp4_204` `/opt/venv-syncv4`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/rounds/round-02/agent.txt:11; :53; rounds/round-01/agent.txt:20)
- E27 [2026-07, gfx950, FlyDSL ?, N=?, kernel-only, pre-quantized operands; rocprofv3 PMC] fp4/fp8 grouped speed ratio 1.320 (gpt-oss down, M=8192) to 1.686 (DeepSeek-V3 gate_up, M=8192), geomean 1.508 over 18 cells; fp4 MfmaUtil 36.6% and 51.6% on those two cells at occupancy 1.00 (fp8 55.8% / 65.1% at ~1.95) (3 MoE models × gate_up/down × M 2048/4096/8192, B=8; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_fp4vs8_occ.md:13-18)
- E28 [2026-08, gfx950, FlyDSL ?, N=40, campaign baseline: frozen plain GEMMs and the fused kernels interleaved under one clock, median of 40 reps] fused glu ~−14% (ratio ~0.86) and fused dglu ~−28% (ratio ~0.72) against the same GEMM with no epilogue work; myskill records −15% / −29% (gpt-oss-20b fused SwiGLU grouped fp8 GEMMs, G=32, 4096 tokens per expert, M_total=131072, H=I=2880 padded to 2944; campaign 20260828_073654; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_073654/memory.md:13-15; /workspace/code/gpt_oss_docker/myskill/optimizer/methodology/18-operator-fusion.md:81-82)
- E29 [2026-08, gfx950, FlyDSL ?, N=?, interleaved same-process A/B, median and min] paired-column (dword) stores in the fused glu forward epilogue: +3.7% (d(med) +3.2–4.4%, d(min) +3.2–4.3%), byte-identical (M=32768, G=8, I=2880, Kp=2944; commit:/workspace/code/gpt_oss_docker/sync/syncv3/Primus-Turbo@21c97a5f)
- E30 [2026-08, gfx950, FlyDSL ?, N=5, five readings, no overlap with the base] a glu-only change +0.55%, not visible in the score averaged over glu and dglu (gpt-oss-20b fused SwiGLU grouped GEMMs; recorded 2026-08-31, myskill 0314b7b; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/18-operator-fusion.md:25)
- E31 [2026-07, gfx950, FlyDSL ?, N=?, ?] batched MXFP8 expert-weight quantizer (one launch over `[B, N, K]`, per-batch dword-aligned scale byte base): 1.3–1.54× faster than the HIP kernel (275→207 / 344→239 / 152→116 / 94→61); the Python loop over experts it replaced (8 launches) was 2.7–4.3× slower than HIP (Turbo MXFP8 weight quantization; shapes not named; recorded by 2026-07-05, myskill 9d3f961 methodology/49; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/12-mxfp8-grouped.md:50; :52)
- E32 [2026-08, gfx950, FlyDSL ?, N=?, kernel-trace min and median of the preshuffle kernel; JIT cache off per arm] MXFP4 grouped NT preshuffle (pure gather/pack, no LDS barrier) with `_PRESHUF_BLK` 256 → 512: `kern_0` min 15.16 → 14.00 µs (−7.7%), median 15.58 → 15.38 µs (about flat); whole chain about −7 µs = 0.13%, below the ruler's resolution, not kept (gpt-oss-20b MXFP4 quantization + grouped GEMM; campaign 20260817_092541 round 9, smci355 n01-25 `kyle_dev` `/opt/venv-syncv4`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_092541/memory.md:156; :158)
- E33 [2026-08, gfx950, FlyDSL ?, N=?, single-process A/B per env setting (one process per arm), batched-event timing interleaved in one session] env overrides of the fused glu build's `waves_per_eu` (2/3/4), `nt_esplit` (4/0) and `nt_vmcnt`: all ~1957 µs / 2218–2222 TF/s, no change; the accumulator is in AGPRs and `waves_per_eu=2` is already the occupancy ceiling (gpt-oss-20b fused SwiGLU grouped fp8 GEMM, M=131072, H=I=2880, G=32; recorded 2026-08-28; local:/workspace/code/gpt_oss_docker/.claude/memory/project_fused_mlp_epilogue_5pct_ceiling.md:20; /workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_073654/memory.md:100-101)
- S1 AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:705 the A-scale slab row base `sa_b = a_pre_g*64 + bm*BLOCK_M + wave_m_off`; `_scsoff` groups rows by 64 (:578-582) and `_wia = sa_b // 128` (:736), so `wave_m_off` must be a multiple of 64 and of 128; a 288-row tile puts a wave at row 144
