---
title: "GatedDeltaNet stepped one token per sequence in Gluon: Qwen3.5 (16 value heads, conv dim 3072, fp8 group-quantized or bf16 output) and Qwen3-Next (8 value heads, conv dim 2048, fp8 group-quantized output), 1–256 rows — each head's 128 × 128 fp32 state split over CTAs at few rows and owned whole above; the conv history shifted in place by its one owner, behind a barrier, or in a second launch"
kind: case
ops: [gdn]
dtypes: [bf16, fp8, fp32]
regimes: ["rows 1-32 and 128-256 (Qwen3.5, fp8 output)", "rows 64 (Qwen3.5, bf16 output)", "rows 1-256 (Qwen3-Next)"]
accuracy: bitwise
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup over what SGLang runs for the layer (five calls), per size, one reference run of the ruler per size", value: "fp8 output 1.282 (256 rows) to 2.507 (2 rows); bf16 output 1.461 (64 rows)", gpu: MI355X, workload: "Qwen3.5, 16 value heads, key and value dim 128, conv dim 3072: fp8 output at 1, 2, 4, 8, 16, 32, 128 and 256 rows, bf16 output at 64 rows", date: "2026-11"}
  - {metric: "speedup over what SGLang runs for the layer, per size, one later run of the ruler", value: "best kernel per size 1.457 (256 rows) to 3.350 (32 rows); the 9 kernels 1.151 (replicated lanes, 256 rows) to 3.350", gpu: MI355X, workload: "Qwen3-Next, 8 value heads, key and value dim 128, conv dim 2048, at 1, 2, 4, 8, 16, 32, 64, 128 and 256 rows", date: "2026-11"}
---

# GatedDeltaNet in Gluon, one token per sequence step: Qwen3.5 and Qwen3-Next at 1–256 rows

Eighteen Gluon kernels that advance every sequence of a GatedDeltaNet layer by one token, a family of the Triton and
Gluon kernel library that [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps. That page says
which family to start from; [triton/gdn](../triton/gdn.md) §2–6 explains the techniques and names a kernel by the row
count it was built for: "the 2-row kernel" is Qwen3.5's unless Qwen3-Next is named, and "the template" is the
Qwen3-Next kernel written for all nine sizes, the table-driven kernel of this page. The fp8 group quantization the
kernels fuse is in [triton/numerics](../triton/numerics.md) §2, their inline assembly in
[triton/gluon/wave_ops](../triton/gluon/wave_ops.md). This page has the family's kernels by size, their code, and
what was measured. Sequences taken in chunks of many tokens are another family:
[gdn_chunked_gluon](gdn_chunked_gluon.md).

The op:

Per row: a width-4 causal convolution over the projected Q, K, V with the conv history shifted in place, the gates,
the gated delta-rule update of a 128 × 128 fp32 state per (slot, value head), a gated RMSNorm and e4m3 per 128 with
fp32 scales. Qwen3.5 (16 value heads): 8 kernels, 1–256 rows, plus one with bf16 output (64 rows); Qwen3-Next (8
value heads, grouped projections): 9 kernels over 9 sizes, one of them the template, table-driven and written for
all 9. After the library's growth these are 9, 12 and 14 kernels
([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §2). The baseline chains a conv update, the
recurrence, the norm and the quantization *(field, E1)*. No kernel uses MFMA *(field, E2)*.

## Applies when

- The op is the one-token step of Qwen3.5's or Qwen3-Next's GatedDeltaNet layer: one row per sequence, each row with
  its own state slot. A row whose slot index is −1 is padding: no state changes and its core output is zero. Both
  layers have 4 key heads and key and value width 128; Qwen3.5 has 16 value heads, Qwen3-Next 8, and value head h
  reads key head h // (value heads / key heads).
- Operands, bf16 unless stated. `projected_qkvz`: rows × 5120 in Qwen3.5 (every Q, every K, every V, then every Z),
  rows × 3072 in Qwen3-Next (grouped by key head: its q, its k, then the v and the z of its value heads).
  `projected_ba`: rows × 32 in Qwen3.5 (every B, then every A), rows × 16 in Qwen3-Next (a key head's B's, then its
  A's). `conv_state`, slots × 3072 or 2048 channels × 3, and the fp32 `delta_state`, slots × value heads × 128 × 128
  in [value, key] order, both updated in place. `cache_indices`: one int32 slot per row, distinct among the rows that
  are not padding. `conv_weight`, channels × 4, and in Qwen3-Next only `conv_bias`; fp32 `a_log` and bf16 `dt_bias`
  per value head; `norm_weight` (128); the fp32 scalars `scale` and `eps`. `conv_state`, `conv_weight` and `conv_bias`
  keep every Q, every K, every V in both models, so a Qwen3-Next kernel reads conv channel c from another projection
  column.
- Outputs. Qwen3.5 writes `values`, rows × 2048 e4m3, and `scales`, one fp32 per (row, value head), the fp8 group
  being a head's 128 values; its bf16-output op, the one for layers whose projections are unquantized, writes
  `normalized`, rows × 16 × 128 bf16. Qwen3-Next writes `normalized` (rows × 8 × 128), `values` (rows × 1024) and
  `scales` (rows × 8).
- The path replaced is SGLang's `Qwen3_5GatedDeltaNet.forward` or `Qwen3GatedDeltaNet.forward` where each sequence
  advances by one token; a speedup is against what SGLang runs for it. For Qwen3.5 that is five calls: SGLang's
  projection split, conv update, recurrent step and gated norm, then aiter's quantization per 1 × 128. The Qwen3-Next
  baseline ends with the same aiter quantization.
- A kernel is written for one row count or a short list of them (§ Mechanism). Qwen3.5 with the fp8 output: 1, 2, 4,
  8, 16, 32, 128 and 256 rows, a kernel each; with the bf16 output: 64 rows. Qwen3-Next: 1, 2, 4, 8, 16, 32, 64, 128
  and 256 rows, each with a per-size kernel and the table-driven one, 128 and 256 rows with two per-size designs.
  Among these 18 kernels another row count has no kernel, and the table-driven kernel raises on one.
- The family is held to `bitwise` accuracy: every run on the same inputs gives the same bytes as the first. No kernel
  uses an atomic, and five calls on fresh copies of the inputs were byte-identical at each kernel's first and last
  size; the sizes between a kernel's first and last were not run *(field, E3)*. Each output must also stay within
  its own error bound against the reference ([triton/numerics](../triton/numerics.md) §1): the Qwen3.5 tolerances on
  the fp8 output, atol 2 and rtol 0.125 on the codes and rtol 0.02 on the scales, admit a code one e4m3 step off and a
  scale one ulp off, and 65536 moved codes and 512 moved scales passed *(field, E4)*.

## Counter-signals

- **The ratio is lowest at the largest sizes.** Qwen3.5 reads 1.426 at 128 rows and 1.282 at 256 *(field, E5)*;
  Qwen3-Next's best kernel per size reads 1.782 and 1.457 there, and its replicated-lane kernel 1.151 at 256
  *(field, E6)*. For Qwen3.5 at 128–256 rows the state stream is most of the work, 256 MiB each way at 256 rows,
  where the kernel and its baseline both stream the state near a read stream's rate: ratio 1.30–1.31, 3 runs
  *(field, E7)*.
- **At few rows the ratio is mostly a count of launches.** Where the ratios are highest, up to 16 rows for Qwen3.5
  and 64 for Qwen3-Next, one or two launches replace a five-call baseline and the state stream is small. On the
  ruler's timer, read from a rocprofiler-sdk tracer in that run, launch floors were 96–97% of both sides' time for
  Qwen3.5 at 2 rows, where the ratio, 2.49, is the launch ratio 5/2, and 80% at 16 rows, 1 run *(field, E8)*. That
  timer charges a launch more than a graph replay pays: a chained trivial kernel added 4.04–4.08 µs to its reading
  under Kineto and 1.77–1.84 µs to a bare replay, 5 runs *(field, E9)*
  ([pitfalls/profiled_timer_floor](../pitfalls/profiled_timer_floor.md)).
- **The Qwen3-Next numbers are from one run.** Every Qwen3-Next speedup on this page is from one run of the ruler on
  one MI355X, in which all 44 sizes of the library's 20 Qwen3-Next GatedDeltaNet kernels, these 9 among them, passed
  the accuracy check *(field, E13)*. A repeat measurement of each of the 20 read its speedup within 4.7% of that run
  at the kernel's smallest size *(field, E10)*; no accuracy result of that repeat was kept.
- **A row count between the sizes has no kernel here, and the nearest kernel is not a rule.** Run at 24 rows, the
  log-nearest Qwen3.5 kernel, the 32-row one, took 16.02–16.23 µs against the 16-row kernel's 15.48–15.59 µs, and at
  48 rows 23.25–23.50 µs against the 128-row kernel's 21.53–21.60 µs, in two runs of two rounds per arm; at 3 and 6
  rows the nearest was 0.2–2.3% slower, at 12 and 96 rows the faster *(field, E14)*. The Qwen3.5 kernels for 64, 128
  and 256 rows map programs as `token = (pid // (VH * 8)) * 8 + pid % 8` and do not check that the row count is a
  multiple of 8: at 12 rows, 32 of the 192 (row, head) pairs would not be computed and 32 programs would read rows
  past the input (derived).
- **The Qwen3.5 16-row kernel compiles again for every pool size.** It passes the pool size as a `gl.constexpr` for a
  `gl.assume`. The pool size changes from call to call, so each new pool size compiles again, inside a graph capture
  too (four pool sizes, four binaries; with a runtime pool size, one *(field, E15)*): keep it a runtime argument
  ([triton/gdn](../triton/gdn.md) §2, [triton/authoring](../triton/authoring.md) §5).
- **Dropping the Qwen3.5 per-tap rounding failed the accuracy check.** Qwen3.5's reference rounds each conv tap
  product to bf16, Qwen3-Next's sums exact products ([triton/gdn](../triton/gdn.md) §1). A copy of the Qwen3.5 128-row
  kernel without the per-tap rounding came closer to an unrounded reference (values' relative L2 0.0155 → 0.0018) and
  failed its accuracy check on values and state, on one input set in two checked runs *(field, E16)*; which
  rounding a kernel may skip is decided by its outputs against the bound, not by copying every rounding
  ([triton/numerics](../triton/numerics.md) §1).
- **The Qwen3.5 fp8 kernels leave FP16_OVFL set.** In the 2-row kernel's second launch Triton's fp8 cast sets the
  mode bit before the first bf16 convert, and a finite bf16 overflow then saturates to ±0x7F7F *(field, E19)*. The
  kernels pass an accuracy check whose inputs are drawn finite. The Qwen3-Next table-driven kernel's bf16 results
  saturate too at 16, 64, 128 and 256 rows *(field, E19)*; entry 5 of § Code is what it does at the other sizes.
- **A kernel that stores only what changed times faster on replayed state than on new tokens.** A timer that replays
  the state in place, as a bench of graph replays does, starts each call from the state the last one left. The 64-row
  bf16-output kernel behind this page's 1.461 stores every state segment. A second 64-row bf16-output kernel, added
  after the library's growth, stores only the segments whose bits changed (entry 8 of § Code): on another timer than
  this page's ruler (each call timed by CUDA events after a 1 GiB write and a GPU spin, medians of 200 rounds in which
  the kernel alternates with what SGLang 0.5.22 runs for the step) it read 2.40× its baseline with the state replayed
  in place and 1.62× with it refreshed before every call *(field, E20)*; neither compares with the 1.461 above.
- **The kernels added since are not on this page's ruler.** This page describes the library's 18 kernels. After the
  library's growth the three ops have 9, 12 and 14 kernels, and none of the kernels added has a speedup measured on
  this page's ruler ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §2).

## Mechanism

The row count picks the design ([triton/gdn](../triton/gdn.md) §2, §3). With few rows each head's 128 value rows are
cut into tiles, one CTA per tile, and a second launch does what a tile CTA cannot: the gated RMSNorm and the fp8 group
each span one whole value head (a variant that coordinates or recomputes within one launch was not tested), and the
conv history may be overwritten only after every reader of the old history has read it. Where rows × value heads
reaches 128, Qwen3.5 at 8 rows and Qwen3-Next at 16, a CTA or workgroup owns whole heads and the epilogue is fused
into it. The update runs on the VALU: per (row, head) two 128 × 128 matrix-vector products and a rank-1 update on a
state no other row reads.

| Model, rows | Kernels | Launches | Decomposition |
|---|---|---|---|
| Qwen3.5, 1–4 | a kernel per count (1), (2), (4) | 2 | each head's 128 value rows split over CTAs, 256 at 1, 2 and 4 rows; then a launch of 40 programs per row; the conv history shifted in place |
| Qwen3.5, 8–256 | a kernel per count (8), (16), (32), (64, bf16 output), (128), (256) | 2 | convolution with exclusive channel ownership, history shifted in place; one CTA per (row, head) owns the state, 4 waves at 8–32 rows, 16 from 64, the norm (and quantization) in its epilogue |
| Qwen3-Next, 1–8 | three kernels (1), (2), (4, 8) | 2 | producers of one or two waves per (row, value head or head pair, block of value rows), then a finisher |
| Qwen3-Next, 16–256 | a kernel per count (16), (32), (64); a one-owner kernel (128, 256) and a kernel with replicated lanes behind a barrier (128, 256) | 1 | one workgroup per (row, key head) owns the conv history and both value heads' states |

- **Techniques.** In the Qwen3.5 kernels for 1, 2 and 4 rows, moving the history shift before the reader launch failed 20 of 20 trials each; shifting inside the reader failed 14, 5 and 17 of 20, while the kernels' own shift in the second launch passed 20 of 20 *(field, E21)* ([triton/gdn](../triton/gdn.md) §3): a race need not fail every trial. The one-owner kernels, Qwen3.5 at 8 rows and Qwen3-Next at 128, passed 40 of 40 each; the Qwen3-Next kernel with replicated lanes (128) passed 100 of 100 with its barrier and failed 100 of 100 without it. A 4-wave kernel with a delayed reader was exact 40 of 40 with one owner or a barrier and mismatched 40 of 40 with neither; a probe without the delay had shown no race *(field, E21)*. The table-driven kernel keeps one configuration row per row count and raises on an unlisted count. Inline asm supplies `v_med3_f32` (all 8 Qwen3.5 kernels, the ±448 clamp; `gl.clamp` emits the same instruction) and, in Qwen3-Next, `v_rcp_f32`, `v_exp_f32`, `v_log_f32`, `v_div_fixup_f32`, `v_pk_fma_f32`, `v_cvt_scalef32_pk_fp8_f32`, `s_setreg_imm32_b32` and DPP wave reductions (`v_max_f32` / `v_add_f32` with `row_shr`, then `v_readlane_b32`) ([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §3, §4, §6, §7).
- **Resources** *(field, E2)*: Qwen3.5 18 compiled kernels, `num_warps` 1, 2, 4 or 16, `waves_per_eu` 1, 6 or 8 in 5, VGPR
  16–146, LDS 0–2 KiB; Qwen3-Next 27 compiled kernels, 1–8 warps, VGPR 17–208, LDS 0–2.5 KiB. No AGPRs, no spills.

Per-size CTA shapes, grids and VGPR counts: [triton/gdn](../triton/gdn.md) §2; the state's loads and stores and their
cache hints per size: [triton/gdn](../triton/gdn.md) §5.

## Code

### 1. q and k read back from LDS in the state layout (Qwen3.5, 2 rows)

From the first launch of the Qwen3.5 kernel for 2 rows, whose CTAs each update a tile of 16 value rows of one head. A
CTA convolves and normalizes its key head's whole q and k, exchanges them between its waves through LDS (`qk`, 2 ×
`BK` fp32, 1 KiB; `BK` is the key width, 128) and reads them back in the state layout: summing a `[1, BK]` slice over
its unit axis yields the vector in `SliceLayout(0, state_layout)` with no data movement. Compiled with Triton 3.8.0
for gfx950 that is four `ds_read_b128` per vector and no lane exchange. The kernel reads q after the state store "to
shorten its register lifetime", but the late read compiles as an early one: with q read together with k the
instructions are identical, 61 VGPRs *(field, E19)*.

```python
# the 2-row kernel; the update and the state store sit between the read of k and the read of q
        # Hold Q in LDS until state writeback to shorten its register lifetime.
        qk_shared = gl.allocate_shared_memory(
            gl.float32, (2, BK), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), qk,
        )
        k = gl.sum(qk_shared.slice(1, 1, 0).load(state_layout), 0)
        q = gl.sum(qk_shared.slice(0, 1, 0).load(state_layout), 0)
        result = gl.sum(state * q[None, :], 1)
```

Not this: `qk_shared.index(1).load(gl.SliceLayout(0, state_layout))`, which Triton 3.8.0 rejects for gfx950 at the
descriptor's `index` step: "rank must be equal to or one less than the shape size. Got 2 and 1".

Explained in [triton/gdn](../triton/gdn.md) §4.

### 2. The fp8 group quantization that ends every head (Qwen3.5, fp8 output)

One helper ends every head in all eight Qwen3.5 kernels with the fp8 output, the same function text in each; `value`
is a head's 128 outputs of the gated RMSNorm. It rounds the head to bf16, takes the amax of its non-NaN values floored
at 1e-10, forms the scale as `amax · (1/448)`, multiplies by one reciprocal of the scale and clamps to ±448 with an
inline `v_med3_f32`, the instruction `gl.clamp` lowers to (S1); the helper's last lines convert `clipped` to e4m3
and store the codes and `group_scale`. The scale and the codes come from the bf16-rounded output, as the references
quantize it, not from its fp32 precursor. This is the one-reciprocal-per-group form of
[triton/numerics](../triton/numerics.md) §2: one rounding more than a division per value, so a code can move at a tie.
The convert in the helper's last lines is a Triton fp8 cast, which makes Triton set MODE.FP16_OVFL ahead of the
helper's bf16 rounding (in the 2-row kernel's second launch, before the first bf16 convert), so that rounding
saturates a finite overflow to ±0x7F7F *(field, E19)*; the kernels leave the bit set and pass a check whose inputs
are drawn finite (§ Counter-signals), and entry 5 shows two ways to keep IEEE overflow.

```python
value = value.to(gl.bfloat16, fp_downcast_rounding="rtne").to(gl.float32)  # quantize the BF16-rounded head
magnitude = gl.where(value == value, gl.abs(value), 0.0)  # a NaN does not enter the amax
maximum = gl.maximum(gl.max(magnitude, 0), 1.0e-10)
group_scale = maximum * (1.0 / 448.0)
inverse_scale = 1.0 / group_scale  # one reciprocal per head
scaled = value * inverse_scale
clipped = gl.inline_asm_elementwise(
    "v_med3_f32 $0, $1, $2, $3",
    constraints="=v,v,v,v",
    args=[scaled, -448.0, 448.0],
    dtype=gl.float32,
    is_pure=True,
    pack=1,
)
```

Not this: an amax that is not floored. In a probe kernel an all-zero group got scale 0 and code 0xFE without the
floor, the clamp turning the NaN into −448, and a finite scale and code 0 with it *(field, E22)*; a padding row's
zero core is such a group ([triton/gdn](../triton/gdn.md) §5).

Explained in [triton/gdn](../triton/gdn.md) §6.

### 3. A conv tile in which every history entry has one owner (Qwen3-Next one-owner kernel, 128 and 256 rows)

The exclusive-owner shift of the Qwen3-Next one-owner kernel for 128 and 256 rows, in which one workgroup per (row,
key head) owns the conv history and both value heads' states. Its convolution runs on one 1-D tile over the key
group's 512 conv channels (`conv_width`: the group's q, its k and the v of its two value heads) in `NW` waves, 4 at
128 rows and 8 at 256, and the kernel asserts `gl.static_assert(NW * 64 <= conv_width)`. No lane or wave then holds a
copy of a channel, so each thread shifts the history it read, with no barrier. The kernel passed 40 of 40 trials at
128 rows *(field, E21)*; proving single ownership in the layout: [triton/gdn](../triton/gdn.md) §3.

```python
conv_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
# This packed tile has no replicated lanes/waves. Every history read and
# write has one owner, even though Q/K are subsequently shared by two heads.
c = gl.arange(0, conv_width, layout=conv_layout)
```

Not this: the same in-conv shift where the tile has more thread slots than channels. At 32 rows the table-driven
kernel's tile has 2 × 64 × 8 = 1024 slots for 512 channels, two owners per channel; forcing the shift inside the conv
there left `delta_state` wrong on 10 of 10 inputs (max |d| 3.24) and was at most 1.5% faster, 3 runs *(field, E7)*.

Explained in [triton/gdn](../triton/gdn.md) §3.

### 4. Replicated lanes: a barrier before the history shift (Qwen3-Next replicated-lane kernel, 128 and 256 rows)

The other Qwen3-Next design for 128 and 256 rows keeps replicated lanes and orders the reads before the shift with
`gl.barrier()`: every wave has read the history before any wave shifts it. Each call of its convolution helper runs
on an 8-wave tile of 512 thread slots over 128 channels (q, then k) or 256 (v), so several threads read the same
entry; `live` is false for a padding row, and the excerpt is condensed from that helper (the last history load, the
last tap, the barrier, the shift). On gfx950 `gl.barrier()` lowers to a workgroup release fence, `s_barrier` and an
acquire fence, so it also orders the CTA's global accesses (S2). With its barrier the kernel passed 100 of 100 trials
at 128 rows *(field, E21)*. It measured the slowest of the three designs for 128 and 256 rows, 1.466 and 1.151
against the one-owner kernel's 1.782 and 1.425 *(field, E6)*; the designs differ in more than the shift
([triton/gdn](../triton/gdn.md) §2).

```python
h2 = gl.load(history + base + 2).to(gl.float32)  # replicated lanes: several threads read the same entry
acc = gl.fma(x, w3, acc)
gl.barrier()  # every wave has read the history before any wave shifts it
if live:
    gl.store(history + base, h1)
    gl.store(history + base + 1, h2)
    gl.store(history + base + 2, x)
```

Not this: the same stores with replicated lanes and no barrier, which failed 100 of 100 trials *(field, E21)*.

Explained in [triton/gdn](../triton/gdn.md) §3.

### 5. A bf16 rounding that keeps IEEE overflow beside an fp8 convert, two ways (Qwen3-Next table-driven kernel)

From the Qwen3-Next table-driven kernel, whose configuration rows take the first way at 1 row, the second at 2, 4, 8
and 32 rows, and neither at 16, 64, 128 and 256 rows, where its bf16 results saturate (§ Counter-signals); the two
choices are separate knobs ("collapsing the two into one flag silently changes results", in its words). With
MODE.FP16_OVFL set, as Triton's fp8 cast sets it, a finite bf16 overflow saturates to ±0x7F7F instead of ±Inf
*(field, E19)*. First way, at 1 row: keep Triton's fp8 store, so the bit is set, and map |x| ≥ 3.396e38, the finite
values that overflow bf16, to a signed infinity before the bf16 convert (`OVFL_FIXUP`). Second way, at 2, 4, 8 and 32
rows: use no Triton fp8 cast, so Triton adds no setreg; the kernel clears the bit at entry (`ieee_pid`), sets it on
the fp8 scale after the last bf16 rounding (`fp8_mode_scale`), then clamps to ±448 and converts pairs of values with
an inline `v_cvt_scalef32_pk_fp8_f32`, which takes the values divided by that scale and so runs with the bit set.
Both asm helpers are `is_pure=False` and return the value they are given, the program id and the scale: order comes
from data, not from `is_pure=False`, so the dependency is routed through the operands
([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §7).

```python
# first way: the bf16 rounding where Triton's fp8 store has set the bit (OVFL_FIXUP is on at 1 token)
@gluon.jit
def round_bf16_ieee(value, OVFL_FIXUP: gl.constexpr):
    # ...
    if not OVFL_FIXUP:
        return value.to(gl.bfloat16)
    bits = value.to(gl.uint32, bitcast=True)
    infinity = ((bits & 0x80000000) | 0x7F800000).to(gl.float32, bitcast=True)
    value = gl.where(gl.abs(value) >= 3.39617752923046e38, infinity, value)
    return value.to(gl.bfloat16)
# ...
# second way: the kernel's entry runs `pid = ieee_pid(pid)`, which clears the bit
@gluon.jit
def ieee_pid(pid):
    return gl.inline_asm_elementwise(
        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\ns_mov_b32 $0, $1",
        "=s,s", (pid,), dtype=gl.uint32, is_pure=False, pack=1,
    )
# after the last bf16 rounding `quant_scale = fp8_mode_scale(quant_scale)` sets it
@gluon.jit
def fp8_mode_scale(scale):
    return gl.inline_asm_elementwise(
        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\nv_mov_b32 $0, $1",
        "=v,v", (scale,), dtype=gl.float32, is_pure=False, pack=1,
    )
```

Not this: Triton's fp8 cast beside a plain `.to(gl.bfloat16)` where a value can overflow bf16 before the fp8 convert.
The kernel can then return a finite maximum where the reference overflows to ±Inf, which the non-finite rule rejects
([triton/numerics](../triton/numerics.md) §1).

Explained in [triton/numerics](../triton/numerics.md) §2.

### 6. `v_rcp_f32` of an argument rescaled by 2^-64 (Qwen3-Next table-driven kernel)

From the sigmoid of the Qwen3-Next table-driven kernel, `denominator` being `1.0 + exp(-value)`. Compiled for gfx950,
`v_rcp_f32` is 1 instruction where a division is 11; it is within 1 ulp and flushes denormals. The kernel keeps a true
division where the reference expression is kept verbatim (its configuration rows for 128 and 256 rows), a bare
`v_rcp_f32` only where the caller immediately rounds to bf16 (16 and 64 rows), and this form where the result can be
subnormal (1, 2, 4, 8 and 32 rows): `v_rcp_f32` of the argument scaled by 2^-64, the reciprocal scaled by the same
constant again, which keeps the instruction's input and output in the normal range. Where the quotient is itself a
rounding point, add the correction of [triton/numerics](../triton/numerics.md) §2.

```python
# 2**-64, used to keep v_rcp_f32 inputs and outputs in the normal range.
_RESCALE = gl.constexpr(5.421010862427522e-20)
# ... in the sigmoid, `denominator` being 1.0 + exp(-value):
        reciprocal = gl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1", "=v,v", (denominator * _RESCALE,),
            dtype=gl.float32, is_pure=True, pack=1,
        )
        return reciprocal * _RESCALE
```

Not this: the bare instruction where the reciprocal can be subnormal. `v_rcp_f32` flushes denormals: in a probe
`v_rcp_f32(2^127)` returned 0 *(field, E23)*.

Explained in [triton/numerics](../triton/numerics.md) §6.

### 7. The fp8 scale as a quotient in 7 instructions (Qwen3-Next, 64 rows)

How the Qwen3-Next kernel for 64 rows forms its fp8 group scale, max(amax, 1e-10) / 448: `magnitude` is the head's
amax and `quant_max` 448, so n = `numerator` and d = `quant_max`. Where a division compiles to 11 instructions this
quotient takes 7: `v_rcp_f32` of d, a Newton step on the reciprocal, q = n · r, a residual step on q, then
`v_div_fixup_f32 q, d, n` for the special cases (0, ∞, NaN). In a test kernel it matched the compiler's `n / d` on 16
special cases, NaN results by class, and bitwise on 843198 random normal quotients with 2^-126 ≤ |d| ≤ 2^126
*(field, E24)*. Which quotient an output's bound allows: [triton/numerics](../triton/numerics.md) §2.

```python
@gluon.jit
def _quantization_scale(magnitude, quant_max):
    numerator = gl.maximum(magnitude, 1.0e-10)
    inverse = gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", (quant_max,),
        dtype=gl.float32, is_pure=True, pack=1,
    )
    inverse = gl.fma(gl.fma(-quant_max, inverse, 1.0), inverse, inverse)  # Newton step
    quotient = numerator * inverse
    residual = gl.fma(-quotient, quant_max, numerator)
    quotient = gl.fma(residual, inverse, quotient)  # residual step
    return gl.inline_asm_elementwise(
        "v_div_fixup_f32 $0, $1, $2, $3", "=v,v,v,v",
        (quotient, quant_max, numerator), dtype=gl.float32, is_pure=True, pack=1,
    )
```

Not this: this quotient with a divisor at the ends of the fp32 range. At d = ±2^127 and ±2^-127 it was off by more
than 1 ulp in 24 of 24 cases *(field, E24)*; here d is the constant 448.

Explained in [triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §4.

### 8. A state store masked to the 64 B segments whose bits changed (Qwen3.5, bf16 output, a second 64-row kernel)

From a second Qwen3.5 kernel with bf16 output for 64 rows, added to the library after its growth; the 64-row kernel of
§ Mechanism stores every segment. After a token's update it compares the state's bits with the bits it loaded, per
segment of `WORDS` 4 B words (16 here, 64 B), and masks the store: a segment that did not change is not written.
`old_bits` is the state as loaded, `state` the values after the update, `DS` the state pool and `CACHE` the store's
cache hint; `rows` is the number of 128-key state rows the program holds (not this page's row count) and `offsets`
their element offsets in the pool. The kernel keeps nothing between calls and is right on any state, so no check fails
it; its work depends on the state's history.

```python
# Gluon. old_bits: the state's bits as loaded; state: the values after this token's update
new_bits = state.to(gl.uint32, bitcast=True)
ol, oh = gl.split(old_bits.reshape((rows, 64, 2)))
nl, nh = gl.split(new_bits.reshape((rows, 64, 2)))
old_pairs = ol.to(gl.uint64) | (oh.to(gl.uint64) << 32)
new_pairs = nl.to(gl.uint64) | (nh.to(gl.uint64) << 32)
# a segment of WORDS 4 B words (16 here: 64 B) is dirty when any of its bits differ
dirty = gl.max((old_pairs != new_pairs).reshape((rows, 128 // WORDS, WORDS // 2)).to(gl.int32), 2) != 0
groups = old_bits.reshape((rows, 128 // WORDS, WORDS))
dirty = gl.convert_layout(dirty, gl.SliceLayout(2, groups.type.layout))
mask, _ = gl.broadcast(dirty[:, :, None], groups)
# the store is masked: a segment that did not change is not written
gl.amd.cdna3.buffer_store(state, DS, offsets, mask.reshape((rows, 128)), cache=CACHE)
```

Not this: this kernel's speedup timed with the state replayed in place, when a step with new tokens changes the state
on every call. Replayed on one call's tensors the recurrence settles toward a fixed point and the kernel stores less,
from 100% of the `delta_state` segments at calls 1–4 to 52% at call 64 and 20% after 400: it read 2.40× its baseline
in place and 1.62× with the state refreshed before every call, the one to quote *(field, E20)*, both on
another timer than this page's ruler (§ Counter-signals).

Explained in [pitfalls/state_kept_across_calls](../pitfalls/state_kept_across_calls.md) § Root cause.

## Result

- **Speedups** over the baseline above: Qwen3.5 1.282 (256 rows) to 2.507 (2 rows), the bf16-output kernel 1.461 *(field, E25)*.
  Qwen3-Next, in a later run of the ruler: its 9 kernels 1.151 (replicated lanes, 256 rows) to 3.350 (32 rows), the best
  kernel per size 1.457 (256 rows) to 3.350, and the per-size kernels above the table-driven one at 7 of 9 sizes (equal at 4 rows,
  the table-driven one ahead at 256) *(field, E6)*.

Per size against what SGLang runs for the layer, one run of the ruler per value: Qwen3.5 in three runs
*(field, E5)*, Qwen3-Next in one *(field, E6)*.

| Rows | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5, fp8 output (bf16 output at 64 rows) | 2.464 | 2.507 | 2.422 | 2.351 | 2.336 | 1.651 | 1.461 | 1.426 | 1.282 |
| Qwen3-Next, per-size kernel (one-owner at 128 and 256 rows) | 2.443 | 2.538 | 2.370 | 2.506 | 3.222 | 3.350 | 2.462 | 1.782 | 1.425 |
| Qwen3-Next, table-driven kernel | 2.390 | 2.450 | 2.370 | 2.479 | 3.101 | 3.091 | 2.401 | 1.755 | 1.457 |

The Qwen3-Next replicated-lane kernel read 1.466 and 1.151 at 128 and 256 rows *(field, E6)*. A later run of the
ruler read Qwen3.5 at 2.478, 2.409, 2.405 and 2.360 at 1, 4, 8 and 16 rows *(field, E5)*.

The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1. What these
ratios say, and how far one moves with no change to the kernel: [triton/gdn](../triton/gdn.md) §13.

## Evidence

- E1 [2026-11, CPU (target gfx950), Triton 3.8.0, N=1, each kernel's `run` replayed on the host with its launches recorded, no GPU] the 24 fp8 GEMM sizes one launch each, the N 3072 kernel for M = 6144 rejecting independently strided A and B; at the 89 MoE sizes of 58 kernels the gate-up launch, its grid and the source of its W1; the baselines' calls: MiniMax-M2 routers `tgemm.mm` with the first operand cast to fp32, then `moe_fused_gate` and `fused_moe` at `per_1x32` (MXFP4) or `per_128x128` (FP8); GDN decode a conv update, the recurrence, the norm and the quantization; GDN prefill `causal_conv1d_fn`, `fused_gdn_gating`, `chunk_gated_delta_rule`, the norm and the quantization (`per_token_group_quant_fp8` in Qwen3.5)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, each kernel called 5 times at its first and last size on fresh copies of the inputs, outputs compared byte by byte; sources scanned for atomics] all 44 GDN kernels: no atomic call; 61 kernel-sizes, 305 calls, every repeat byte-identical to the first; the 13 sizes between a kernel's first and last not run (the 18 decode and 26 prefill kernels)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, runs on one GPU: the bound formulas applied to built outputs, or the accuracy check run on a library kernel and on an edited copy] a 0/0 tolerance on bf16 in the form B = atol + (rtol + u)·|y|, y the unrounded reference and u = 2^-8 the unit roundoff of bf16: B = 2^-8·|y|, the round-to-nearest output passed, 1% of elements one step further failed (fraction over the bound 0.00995); MXFP4 round trip (ceil e8m0, RNE e2m1) of both expert-GEMM inputs inside the references, first input set: DeepSeek-V2 13.45 dB at 256 tokens, MiniMax-M2 13.18 dB at 128; an fp32 dot-product rounding bound at K = 6144: fp32 sums in 1, 6 and 16 parts passed at peak 0.69; bias test, n = 10^6, every error +0.5·u·|y|: |mean|/rms 0.80, the output-rounding and dot-product bounds failed, the tolerance, the SQNR floor and a bound relative to a same-mean baseline passed; RTZ to bf16 of positive values 0.68; the check's large-magnitude input set, every floating-point input scaled by one exact power of two, the largest that keeps the tensors and the dot-product outputs in range: 2^54 for the N 2624 × K 6144 bf16 GEMM and 2^55 for the N 6144 × K 2048 bf16 GEMM at all 10 sizes; Qwen3.5 GDN decode bounds: 65536 codes moved one e4m3 step and 512 scales moved one ulp all passed; a8w8 block scales, 2400 drawn: log2 from −2.999 to −0.0009, none an integer; failures: `0 · (1 / gl.exp2(-127))` 256 of 256 non-finite mismatches, a 0xFF scale, an RTZ'd NaN, saturated 448, 65504 and fp32 maximum, NaN against 448 one each; a bf16 running sum over K = 2048 fraction over the bound 0.941 at peak 58.4, a stray fp4 nibble 0.438, an unwritten output 1 at peak 128; missing tap rounding, a contracted product and x·(1/s) at injected ties 1.4e-4, 9.5e-5, 1.2e-4 at peaks 1.84, 1.95, 2.67 under bounds chosen for the test; fp8 P on P·V 23.1 dB; MXFP4 18.77 dB, 17.07 with an e5m2 rounding first; the Qwen2-MoE router kernel for 256 tokens, seed 0: fp32 logits rounded once to bf16, then ranked, gave the reference's top-10 ids for all 256 tokens, the unrounded sum ranked changed 97 tokens and failed the id comparison (fraction of ids differing 0.0809) in both checked runs of each run; the a8w8 GEMM N 7168 × K 2048 at m = 64: `dot * sx * sw` against `dot * (sx * sw)` in K stages 0–2 of 8 (stages 3–7 multiply by a precomputed sx · sw in both arms), 2 checked runs per arm in each run: all 12 passed, worst fraction over the bound 0
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush), speedup over sglang's five-call path] Qwen3.5 GDN decode, speedups from three runs: with the fp8 quant 1 row 2.464, 4 rows 2.422, 8 rows 2.351, 16 rows 2.336, 32 rows 1.651 in one (a later run of the ruler 2.478, 2.409, 2.405 and 2.360 at 1–16 rows); 2 rows 2.507 and, with BF16 output, 64 rows 1.461 in a second; 128 rows 1.426 and 256 rows 1.282 in a third
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] the 9 Qwen3-Next GDN decode kernels, one later run of the ruler: 1.151 (the replicated-lane kernel, 256 rows) to 3.350 (the 32-row kernel); best kernel per size 1.457 at 256 rows (the table-driven kernel) to 3.350; per-size kernel against the table-driven one at 1, 2, 4, 8, 16, 32, 64 rows: 2.443 / 2.390, 2.538 / 2.450, 2.370 / 2.370, 2.506 / 2.479, 3.222 / 3.101, 3.350 / 3.091, 2.462 / 2.401; at 128 and 256 rows the one-owner kernel 1.782 and 1.425, the replicated-lane kernel 1.466 and 1.151, the table-driven kernel 1.755 and 1.457 (8 value heads, key and value dim 128, conv dim 2048)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (N=4 for the 1–4-row call times), the ruler (Kineto) with its accuracy check, on the decode kernels and modified copies, 4 alternated rounds per arm, kernel time the mean of the rounds' median kernel-side times, variants that fail accuracy run in-process on random inputs against the unmodified kernel's output; Gluon probe kernels: a 1 GiB tensor streamed once per launch (2048 programs × 4 warps, 16 bytes per lane unless stated), median of 9 rounds × 10 launches, variants interleaved; a [64, 64] 32-bit tile converted A → B → A 1024 times in 1024 programs, median of 7 rounds × 3 launches; ranges over the runs] the 2-row kernel at 2 rows with 16-row value tiles (256 CTAs) 8.59–8.73 µs against 10.55–10.62 µs with whole-head tiles of the same kernel (128 rows, 32 CTAs, 267 VGPRs), 17.7–18.6% less, at 16 rows 14.76–14.78 against 13.19–13.23 µs; tiles of 8, 16 and 32 rows at 2 rows 8.640–8.655, 8.631–8.665 and 8.601–8.714 µs; its history shift moved into launch 1: `conv_state` wrong on 10–11 of 20 inputs per run, `delta_state` off by up to 0.95–1.46, the unmodified kernel exact on all 20 in every run; kernels run at other row counts: at 24 rows the 16-row kernel 1.821–1.833 and the 4-row kernel 1.809–1.823, at 192 rows the 128-row kernel 1.339–1.342 and the 4-row kernel 1.042–1.043; the Qwen3-Next template at 32 rows with inline history retirement: `delta_state` wrong on 10 of 10 inputs, max |d| 3.243 in every run, warm-cache graph replay B/A 0.986–0.999 (7.48–7.62 against 7.58–7.63 µs); the 32-row kernel, state load after the gates with the SiLU early: B/A per pair 1.000–1.018 (17.91–18.04 → 18.16–18.27 µs); the 8-row kernel, q loaded and normalized before the state load 1.018–1.039 (9.47–9.49 → 9.67–9.76 µs), decay and β in each update program 1.062–1.074 (9.47–9.53 → 10.13–10.16 µs); the 1-row kernel against scalar exponentials (7 against 4 `v_exp_f32`) B/A per pair 0.986–1.013 (8.65–8.73 against 8.64–8.71 µs); 8 readers of one chunk, ids congruent mod 8 12.1–12.4 µs against 21.6–22.0 µs consecutive (1.75–1.80×); the 128-row kernel with `token = pid // VH` B/A per pair 1.005–1.011 (45.62–45.93 → 46.09–46.21 µs); state read plus write 5.85–5.89 TB/s by the 128-row kernel (45.6–45.9 µs) and 6.57–6.59 TB/s by the 256-row kernel (81.5–81.7 µs), 1.06–1.20 of a 5.48–5.51 TB/s `.cg` read stream, the 256-row baseline 106.4–106.7 µs (0.91–0.92 of it; ratio 1.30–1.31), at 2 rows 8.7–8.8 and 21.6–21.7 µs (0.087–0.088 and 0.035; ratio 2.47–2.48); the 2-row kernel at 2 rows: two kernels in every Kineto trace, median call time per timing 8.55–8.87 µs (run means 8.65, 8.75, 8.68 µs), its baseline 21.20–21.78 µs, 4 timings per run; the 1-, 2- and 4-row kernels 8.62–9.04, 8.81–9.14 and 8.81–9.13 µs in three runs and 9.09, 8.65 and 9.10 µs in a fourth (t2/t1 0.951–1.022, t4/t1 1.001–1.022), the 2-row call 0.46–0.485 TB/s against 6.72–6.78 TB/s for a 1 GiB `.cg` Gluon read stream (0.068–0.072 of it); 256 programs × 4 warps × 8 loads per lane 7129–7144 GB/s, 2048 × 4 × 1 6674–6684 GB/s (1.067–1.070×); at 3 MB in flight 96 × 4 × 8 3945–3995 GB/s, 384 × 4 × 2 4920–4962 GB/s, 192 × 1 warp × 16 2454–2467 GB/s (0.80–0.81× and 0.495–0.499× of 384 × 4 × 2); reads: no hint 5993–6013 GB/s, `.cg`/`.cs` 6771–6837 GB/s (1.130–1.137×), buffer over pointer 1.002–1.005 hinted and 1.002–1.003 unhinted; writes: no hint 5792–5889 GB/s, `.cs` 5404–5510 (0.928–0.945×; buffer over pointer 0.997–1.012), `.wt` 0.998–1.005× through `buffer_store` (above 1.00 in two runs) and 0.983–0.996× through `gl.store`, below every `gl.store` without cache bits (none, `.wb`, `.cg`) in each run; lane-moving conversions 574–1522 ns per conversion against 73–74 ns for the loop alone, the 96-`ds_bpermute` path 661–663 ns, the vectorized LDS round trip 574–581 ns (Qwen3.5 GDN decode kernels; the Qwen3-Next template; probe kernels)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, per-kernel times on the ruler's timer (tracer), 5 rounds per arm, medians] Qwen3.5 decode, the kernel against the five-call baseline, a trivial kernel 4.26 µs: 2 rows, 2 launches against 5, ratio 2.486, launch floors 0.959 and 0.966 of each side's time; 16 rows floors 0.797 and 0.800
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (slope and two-kernel readings repeated in four more runs), Gluon probe kernels on one GPU: chained trivial 1-warp kernels in a CUDA graph, timed by the ruler's timer (Kineto, `GPU_FORCE_QUEUE_PROFILING=1`) and by the bare replay's wall clock, eager launches by the host clock] the reading of the ruler's timer grows 4.062 µs per added kernel (Kineto sum 3.902), a bare replay 1.815 µs (grid 1) and 1.844 µs (grid 256); two chained kernels read 8.246 µs on the ruler's timer; eager host time 1.004 µs per tensor argument, while the ruler's timer reads 4.162–4.178 µs for kernels with 1, 8 and 32 tensors; the repeats, with and without queue profiling forced: the ruler's timer 4.045–4.082 µs per kernel, bare replay 1.766–1.771 µs, two chained kernels 8.25–8.40 µs (toy kernels)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel run once on the ruler (Kineto) with its accuracy check, on its own Triton cache, resources and opcodes read from each cached `.amdgcn`; for the Qwen3-Next kernels also the measured speedup at each kernel's smallest size against the run of E11 and E12] the 18 GDN decode kernels: no `v_mfma` and no `v_dot2`; Qwen3.5 recurrence VGPRs 35, 61, 61, 142, 146, 126, 47, 55, 54 at 1, 2, 4, 8, 16, 32, 64, 128, 256 rows, no spills, the 128-row kernel 16 warps at `waves_per_eu` 8; all 26 prefill kernels contain an fp32 MFMA (`16x16x4`, or `32x32x2` in 4 Qwen3.5 and 2 Qwen3-Next kernels) and a BF16 one; SEQ32G's chunk-table builder 190 VGPRs at M = 8192, 256 VGPRs with 64 spilled at M = 4097 under `waves_per_eu=2`, 322 and no spill without it; the 20 Qwen3-Next kernels: every measured speedup within 4.7% of the earlier run's (largest deviation of the ratio from 1, 0.04654); no accuracy result of this run was kept for them (the 44 GDN kernels)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush), speedup over the Qwen3-Next baseline of [triton/gdn](../triton/gdn.md) §1] Qwen3-Next GDN decode, per-size kernels: 1 row 2.443, 2 rows 2.538, 4 and 8 rows (one kernel) 2.370 and 2.506, 16 rows 3.222, 32 rows 3.350, 64 rows 2.462, 128 and 256 rows (one kernel) 1.782 and 1.425, the third design at 128 and 256 rows 1.466 and 1.151; the template at 1–256 rows 2.390, 2.450, 2.370, 2.479, 3.101, 3.091, 2.401, 1.755, 1.457; one run
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush), speedup over the baseline of [triton/gdn](../triton/gdn.md) §1] Qwen3.5 GDN prefill, speedups per size as in [triton/gdn](../triton/gdn.md) §8, from five runs: D 1024/1 2.423, D 512/1 2.638 and 1024/2 1.889, RSUM b 16384/2 at `max_seq_len` 8192 1.339 in the first (a later run of the ruler: D 1024/1 2.452, D 512/1 2.681 and 1024/2 1.927); RSUM b 16384/2 at `max_seq_len` 10000 1.444, MAP 10000/1 1.188, CKPT 1025/1 2.256, RSUM b 16384/1 1.741, RSUM a 5000/1 1.199 and 15000/3 1.198, MAP 8192/1 1.193, SEQ16 b 15361/8 1.454 and 16384/8 1.475, CKPT 2048/1 2.092 in the second; D 512/2 2.629 in the third; SEQ32L 2049/2 1.374 and 4096/2 1.452 in the fourth; SEQ32G 4097/3 1.029 and 8192/4 1.424, SEQ16 a 8193/5 0.997 and 15360/8 1.473 in the fifth; Qwen3-Next GDN prefill, one run, source / twin: snapshot scan 1024/1 1.573, 2048/2 1.434; multi-schedule 3072/3 1.592 / 1.554, 4096/4 1.504 / 1.501, 8192/8 1.490 / 1.484, 12288/12 1.288 / 1.282, 16384/16 1.463 / 1.460; compact factors 13312/13 1.232 / 1.233, 15360/15 1.328 / 1.329; multi-schedule 4096/1 1.718 / 1.719, 5120/1 1.727 / 1.741, 10240/2 1.308 / 1.311; reverse summaries 16384/4 1.294 / 1.283
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (Kineto); accuracy against each output's bound, speedup against what sglang runs for the op] every size with a recorded accuracy result passed: 198 of the 242 sizes of the library's 154 kernels, among them gated attention 9, Qwen3.5 GDN decode fp8 8 and prefill 21, MiniMax-M2 fp8 28, Qwen2-MoE fp8 7; the 44 sizes of the 20 Qwen3-Next kernels, which had no recorded result then, all passed in a later run of the ruler, the run their stored speedups are from; speedups: the MM64-FP4 kernel with the software e2m1 encoder 1.19 at 128 tokens and 1.194 at 256; the Qwen2-MoE fp8 router + experts kernels with a float64 router dot 1.172 at 1 token (on the VALU) and 1.063 at 32 (on the fp64 MFMA), 1.152 and 1.063 in a second run of the ruler with unchanged sources (image lmsysorg/sglang:v0.5.22-rocm10-mi35x)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the ruler (Kineto), 2 rounds per arm, arms alternated in a fixed order, one GPU] Qwen3.5 GDN fp8 decode kernels run at row counts between their sizes, the log-nearest kernel against the other neighbour, call time (first run / second): 3 rows the 4-row kernel 8.97 / 8.87 against the 2-row 8.90 / 8.85 µs; 6 rows the 8-row 9.18 / 9.18 against the 4-row 9.10 / 8.98; 12 rows the 16-row 10.37 / 10.35 against the 8-row 10.54 / 10.62; 24 rows the 32-row 16.23 / 16.02 against the 16-row 15.59 / 15.48; 48 rows the 32-row 23.25 / 23.50 against the 128-row 21.53 / 21.60; 96 rows the 128-row 36.39 / 36.38 against the 32-row 39.26 / 39.92
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the prefill graph capture N=3), the ruler (Kineto) and its accuracy check with poisoned scratch, on the kernels named and on all 26 GDN prefill kernels; JIT binaries counted around a CUDA-graph capture (relaxed mode); the sglang integration of the prefill kernels called with mocked sglang layers] a copy of the 8-row kernel that skips the `scales` store of odd rows: a numeric mismatch at the first captured call, 64 non-finite mismatches in `scales`; the unmodified kernel: no finding; the 16-row kernel with the pool size as a `gl.constexpr`: 1, 2, 3, 4 binaries after pool sizes 33, 34, 35 and, under capture, 36; with a runtime pool size 1, 1, 1, 1; no capture error; the 26 prefill kernels, each with the fresh-state zeroing kernel of [triton/gdn](../triton/gdn.md) §7: all timed (capture under `capture_error_mode='global'`, replay with new inputs), no capture, accuracy or mutation finding, exit 0; the mocked Qwen3.5 and Qwen3-Next prefill integrations ran eager, declined the call under capture and ran eager again (call counters 1, 1, 2)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 run, one input set per size, 2 runs of the accuracy check per arm, not independent samples] A the library kernel, B with one rounding changed: the Qwen2-MoE router kernel at 256 tokens ranking the fp32 logit sum instead of its bf16 rounding: weights' relative L2 against an unrounded float64 reference 0.004103544 → 2.836419e-7, fraction of top-k ids differing 0.08086, B failed both; a Qwen3.5 GDN decode kernel at 128 tokens without the per-tap bf16 rounding: values' relative L2 0.015502125 → 0.001760144, B failed both (values and state peaks 1.689 and 2.019); a Qwen3.5 GDN prefill kernel at 16384 tokens quantizing the fp32 precursor of its normalized output: values and scales changed, A and B passed both, peaks 0.353679, 0.64, 0.348193; the same kernel with a bf16 round trip on each conv tap product: failed once, scales peak 12.56; the DS512 kernel of E17 at 8 tokens without its router-logit rounding: passed both, paired mean 1.238747e-9, SE 3.271610e-7; the Qwen3.5 gated attention kernel of E18 at 32 rows with the attention kept in fp32 before the gate: scales' relative L2 0.0032333552 → 0.0032689805, passed both; the router kernel with two ids of token 0 swapped: id mismatch on 2 of 2560 elements, failed
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler with its accuracy check, on one GPU, 4 runs per arm alternated, 2 for each DSA kernel; both arms' launches compiled and their ISA compared] the DS512 kernel for 8 tokens, `enable_fp_fusion` `False` → `True` on all 3 launches: speedup 1.147 → 1.146 (A 1.142, 1.147, 1.15, 1.149; B 1.147, 1.147, 1.151, 1.14), B/A 0.9994, the same instructions and `.amdhsa_` fields (633, 1554, 440 instructions); the Qwen2-MoE router kernel for 256 tokens, `["amdgpu-ieee", "false"]` removed from the routing launch: 1.910 → 1.921 (A 1.905, 1.922, 1.907, 1.904; B 1.926, 1.911, 1.923, 1.927), B/A 1.006, A's (max − min)/mean 0.94%, the routing launch 612 → 734 instructions and `.amdhsa_ieee_mode` 0 → 1, the split-K projection launch unchanged; accuracy passed in all 16 runs of the two; `allow_flush_denorm=True` removed from the 4 DSA 64-head kernels at 12, 16, 24, 32 tokens: speedup ratio 1.0007, 1.0033, 0.9986, 0.997, accuracy passed in every B run, the split and the merge kernel the same instructions and `.amdhsa_` fields in 8 of 8 kernels
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 64 input sets drawn from the op's input distributions at 32 rows, the library kernel and a copy without the attention's bf16 rounding before the gate] scales' relative L2 against a float64 reference without that rounding, the copy's over the library kernel's: geometric mean 0.849, lower on 62 of 64 sets; both passed the accuracy check on all 64 sets (Qwen3.5 gated attention with an fp8 group-quantized output, the kernel for 32 rows)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, kernels compiled for gfx950, VGPRs, waves per SIMD and instructions read from the `.amdgcn`; probes and the kernels run on the GPU against fp64, torch or the op's reference; one A/B on the ruler (Kineto) of 4 rounds] the 2-row kernel with q read together with k: identical instructions, 61 VGPRs; the Qwen3-Next template, state load early against late: +7 VGPRs at 1 row (65 as built), 0 at 32 (160), +77 at 128, +12 at 256, no spills; accumulate against recompute form in its fused-heads path: 138 against 134 VGPRs at 16 rows, 131 against 132 at 64, 3 waves per SIMD, no spills; decay and β before the conv at 128 rows: 200 VGPRs either way, 2 waves per SIMD; at 32 rows the late state load 10.28 µs against 8.39 µs early, 160 VGPRs each (speedup 3.18 → 2.58; the other 8 sizes 0.997–1.006); one-wave product of two 16 × 16 fp32 accumulators: 69 instructions with `k_width = 4`, 69 relabeled, 79 with two permlane steps (4 swaps), 92 direct, 1024 B of LDS each; relabeled result 9.727e-8 relative to fp64 (`k_width = 4` the same, plain route 1.007e-7); `k_width = 1` from the transposed accumulator 4 `ds_write_b32` + 4 `ds_read_b32`, 1024 B, no `ds_bpermute`, a one-transposition step 2 `v_permlane32_swap`; 64 × 64 tiles, 4 waves: only (transposed, `[4, 1]`, A) and (not transposed, `[1, 4]`, B) at `k_width = 4` pass `assert_trivial`; SEQ16's 64 × 128 `[4, 1]` state: direct 8 `ds_write_b128` + 8 `ds_read_b128`, 16384 B, relabeled no LDS, 32 `v_mfma_f32_16x16x4_f32`; 2^20 random BF16 pairs with exponents within ±63: every fp32 product equals the fp64 one; 16×32 @ 32×16 tiles (4194304 elements), the fp32 operand split into BF16 terms: equal to the correctly rounded BF16 in 58.2% (high term), 99.76% (two RNE terms), 100% (RTZ, RTZ, RNE); 300000 random fp32 values with exponents within ±40: two RNE terms split error at most 7.605e-6 relative, three RTZ terms exact; 10000 chunks of 64 tokens, log-decays uniform in [−6, 0]: the prefix ratio NaN at 4,242,274 and 0 at 4,179,454 of 14,078,262 entries whose fp64 value is a normal fp32, a masked add-scan and one exponential within 3.76e-6 relative; padded rows of the 2-row and 8-row kernels: codes zero-valued, about half of the bytes 0x80, every scale 0x2a7b5124 = 2.23214296e-13; the Qwen3-Next snapshot-scan kernel within the accuracy bounds at 128 sequences, at 129 normalized excess 161, with eight halvings within them at 129; one recurrence step on a random 128 × 128 fp32 state with decay 1 and β 0: no bit changed, v NaN at rows 3 and 17 made those rows NaN; the 1-row kernel with its BF16 core scratch replaced by an fp32 scratch of the BF16-rounded core: outputs bitwise equal on 10 of 10 inputs, 4 of them not padding; with the unrounded core equal on the 6 padding inputs only; fp32 `.to(float8e4nv)`, `.to(float8e5)` and an fp32 store into an e4m3 pointer: one MODE bit-23 write (value 1) at instruction 6, before the first VALU; the 2-row kernel's norm and quant kernel: bit 23 set at instruction 6, before its first `v_cvt_pk_bf16_f32` (instruction 173); with the bit set fp32 → BF16 of finite overflowing values 0x7F7F/0xFF7F and of Inf 0x7F80/0xFF80, with it clear 0x7F80/0xFF80; the Qwen3-Next 1-row kernel's IEEE BF16 rounding next to a Triton fp8 store: no bit mismatch against torch on 60416 values; the template's config rows: IEEE fix-up at 1 row, native convert at 2, 4, 8 and 32, neither at 16, 64, 128 and 256; 16 re-splits of `cu_seqlens` at the 8 multi-sequence prefill sizes with unequal lengths changed no grid; 12000 random ragged batches: chunk slots distinct and below `cdiv(m, BT) + batch`; fp32 `torch.empty` scratch per call at the 8–16K-token summary sizes 173.1 (RSUM a, 15000/3) to 294.0 MB (MAP, 10000/1), MAP's factors there 246.2 MB and maps 35.7 MB (Qwen3.5 and Qwen3-Next GDN kernels; probe kernels)
- E20 [2026-11, gfx950 (MI355X), Triton of the sglang 0.5.22 image (no FlyDSL kernel), N=200 rounds, the kernel and the baseline (sglang 0.5.22's GDN decode) alternating with the order swapped on odd rounds, each on its own buffers; before each timed call a 1 GiB write and a GPU spin of at least 300 µs, CUDA events, medians] the Gluon GDN decode kernel at n_tokens 64, n_slots 127: state replayed in place, kernel 20.8 µs, baseline 49.8 (2.40×); `delta_state` and `conv_state` copied back from the initial draw before every call, kernel 30.8, baseline 49.8 (1.62×); share of the live slots' `delta_state` segments one kernel call stored, from the initial state: 100% at calls 1–4, 83% at 16, 52% at 64, 31% at 200, 20% after 400; `conv_state` 0% from call 4 (Qwen3.5 GDN decode, Gluon)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 executions, 20–100 trials each within an execution] Qwen3.5 GDN decode at 1, 2 and 4 rows: the kernels as they are 20/20 pass each, history shift before the reader launch 20/20 fail each, shift inside the reader 14/20, 5/20, 17/20 fail; the one-owner kernels, Qwen3.5 at 8 rows and Qwen3-Next at 128, 40/40 pass; the Qwen3-Next replicated-lane kernel at 128 rows 100/100 pass with its barrier, 100/100 fail without; a 4-wave kernel with a delayed reader: replicated without a barrier 40/40 mismatch, with a barrier or one owner 40/40 exact (an earlier probe without the delay showed no race); GDN prefill, a Qwen3.5 kernel at 2049 tokens / 2 sequences and a Qwen3-Next kernel at 3072/3 and 3072/2: history committed early fails all three inputs in each execution, 12693–29318 fp8 values changed, final conv and delta states unchanged; the unchanged kernels pass
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, probe kernels on one GPU] against an fp32 mirror of the reference and a bound of two output roundings (2·u_out·|y|): the fp32 precursor quantized instead of the bf16 output put 100000 of 100000 scales outside the bound and changed 282120 of 12.8M codes (repeated once on the same data, same result); all-zero group: scale 2.232e-13 and code 0 with the floor, scale 0 and code 0xFE without it; fused norm + quant 12.22 µs against 22.37 µs for the two kernels at 4096 rows, 13.63 against 22.49 at 131072 (median over 9 rounds of 50 back-to-back launches, no flush, the fused kernel timed first); 12.8M fp32 values within 2 ulps of e4m3 midpoints: x/s against x·fl32(1/s) 624683 codes moved, each one step, the fourth form of [triton/numerics](../triton/numerics.md) §2 0 code and 0 quotient mismatches against IEEE division; bf16 inputs, |x|/amax from 2^-10 to 2: 0 of 491520 pairs moved; `amax / 448` against `amax · fl32(1/448)`: one ulp below for 71 of 128 significands in each of 25 binades, above for none; 100000 groups with divided scales: no element over the bound, |mean|/rms 0.309 against 0.0158; VALU 550 (division per value), 236 (reciprocal per group), 252 (fourth form); without a clamp and with scale = amax/448: x/scale above 448 in 325 of 4096 rows, within 448·(1 + 2^-22), Triton's cast 0x7E at every row's amax, no NaN code; a kernel with a Triton fp8 cast against one without: fp16 multiply 0x7BFF against 0x7C00, bf16 and scaled-e4m3 MFMAs 0x7F7FFFFF against 0x7F800000, fp32 → bf16 0x7F7F/0xFF7F against 0x7F80/0xFF80; 2^20 elements: `gl.fma(p * sx, sw, acc)` and `acc + p * sx * sw` 0 elements differ, `gl.fma(p, sx * sw, acc)` 213955 (probe kernels, not library kernels)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, probe kernels on one GPU, errors against float64] floored `v_rcp_f32` scale reciprocals for amax 0 to 1e30 all normal, at least 4.48e-28, and `v_rcp_f32(2^127)` = 0; division spellings 0 mismatches over 2^24 normal pairs; `gl.exp(-100)` = 3.784e-44, 0.45 subnormal ulp off; max relative error × 2^24 of `gl.exp` 2.2, 15.9, 64.8 and of `libdevice.exp` 2.0, 12.35, 45.5 on [0.5, 1], [8, 16], [80, 88]; bf16 mismatches over 2^24 x in [−60, 60]: 2209 (`gl.exp`), 2117 (`libdevice.exp`); `gl.log(1e-40)` = −92.1; `gl.log2` of 2^-140 and 2^-149 exact; `gl.sqrt`, `gl.rsqrt` at most 0.92, 0.86 ulp over 2^24 inputs in 2^±100, 0 and +Inf at 1e-40; an RMS norm with `gl.rsqrt` on 1000 × 1024 randn: every bf16 output within 0.5 ulp; `gl.sqrt_rn` 0 mismatches over 2^24; bare `v_rcp_f32` 10.87% of 2^24 fp32 results off, 124 (7.4e-6) one bf16 ulp off after rounding to bf16; `v_log_f32(1 + gl.exp(a)) · fl(ln 2)` against `gl.log`: 0.2704 of 2^24 a in [−20, 20] differ; `a * b - fl32(a * b)`: default and `gl.fma` exact on all 2^24, `enable_fp_fusion=False` 0; `mfma_scaled` e4m3 with per-row scales 2^-90 to 2^-60: 359 of 1024 outputs differ in bits from float64 rounded once, all subnormal, 23 of 505 expected nonzero subnormals 0; one 0xFF scale byte: its row and column NaN, the rest exact; K = 128 with one 448 × 448 product: 4 elements over an fp32 dot-product rounding bound, peak 1.04; bf16 [16, 16, 32] MFMA: integer data exact at K = 128 and 512, randn peak 0.0081 of the bound (fp8 0.871); rows of A and B permuted: outputs permuted bit for bit (16x16x128, 32x32x64 e4m3, e4m3 × e2m1); a8w8 at K = 1024 with scales from 2^U[-3, 0] and from 2^U[-1, 1]: 0 elements over atol = rtol = 1e-2 in 3 seeds of 4096 × 256 (65536 elements per seed in the first run); the counts the same in all 3 runs (probe kernels, not library kernels)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, results against the compiler's `n / d`] the 7-instruction quotient (`v_rcp_f32`, Newton step, residual step, `v_div_fixup_f32`): 16 special cases equal (non-NaN results bitwise, NaN by class); 843198 random normal quotients with 2^-126 ≤ |d| ≤ 2^126 all bitwise equal and within 0.5 ulp; d = ±2^127 and ±2^-127 off by more than 1 ulp in 24 of 24 (test kernel)
- E25 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] Qwen3.5 GDN decode: lowest 1.282 at 256 rows, highest 2.507 at 2 rows, in two runs of the ruler; bf16-output op 1.461 at 64 rows, in the second of them (16 value heads, key and value dim 128, conv dim 3072)
- S1 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp:433 `ClampFOp` lowers to `ROCDL::FMed3Op` (:433-451)
- S2 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp:719 a barrier on local and global spaces becomes release fence, `s_barrier`, acquire fence (:719-736); TritonGPUToLLVM.cpp:113 Membar analysis runs before lowering (:113-115)
