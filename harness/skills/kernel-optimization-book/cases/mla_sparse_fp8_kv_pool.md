---
title: "Sparse MLA with one query token per sequence on DeepSeek-V4's padded fp8 KV pool, 64 heads, 128 window slots plus 0, 512 or 8256 compressed-cache slots — one dequantised 32-key LDS tile feeds both GEMMs for all 64 heads; a second launch merges the partials"
kind: case
ops: [mla, attention]
dtypes: [fp8, bf16]
regimes: ["one query token per sequence", "sparse top-k", "batch 1-16", "batch 32-64", "batch 128-256"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over sglang's TileLang dpsk_v4_fp8_attention_fwd, one-sided lower bound of the time ratio over 32 rounds of CUDA-graph replays with a 1 GiB cache flush before each call, per size", value: "1.165–2.405; 61 of 61 sizes ≥ 1.03", gpu: MI355X, workload: "window only (128 indices into a pool of 256-slot pages), 64 heads, one query token per row, batch 1–256 × 4–9 pool sizes", date: "2026-10-28"}
  - {metric: "speedup over sglang's TileLang dpsk_v4_fp8_attention_fwd, one-sided lower bound of the time ratio over 32 rounds of CUDA-graph replays with a 1 GiB cache flush before each call, per size", value: "2.396–5.845; 61 of 61 sizes ≥ 1.03", gpu: MI355X, workload: "window plus an extra pool of page size 2 with 8256 indices, batch 1–256 × 4–9 pool sizes", date: "2026-10-28"}
  - {metric: "speedup over sglang's TileLang dpsk_v4_fp8_attention_fwd, one-sided lower bound of the time ratio over 32 rounds of CUDA-graph replays with a 1 GiB cache flush before each call, per size", value: "2.362–5.078; 61 of 61 sizes ≥ 1.03", gpu: MI355X, workload: "window plus an extra pool of page size 64 with 512 indices, batch 1–256 × 4–9 pool sizes", date: "2026-10-28"}
---

# Sparse MLA with one query token per sequence on the DeepSeek-V4 fp8 pool: one dequantised tile for all 64 heads

One FlyDSL source serves three configurations of sparse MLA for DeepSeek-V4-Flash with one query token
per sequence, as in decode. They are called here by their extra pool: "no extra" (window only), "extra 8256"
(extra page size 2, 8256 indices) and "extra 512" (extra page size 64, 512 indices). "batch" is the number of rows,
one query token each. sglang's DSA backend (8 heads, one fp8 pool of 576-byte rows,
top-2048) is another op, with its own one-launch kernel:
[dsa_sparse_mla_8_heads.md](dsa_sparse_mla_8_heads.md).

## Applies when

- Sparse MLA over a paged KV cache in DeepSeek-V4's fp8 pool format:
  one query per row, 64 heads, q bf16 with 512 values per head, `sparse_lens`, attention sinks, lse returned. The
  pool is float8_e4m3fn bytes in pages of 256 slots; a slot is a 576-byte payload (448 fp8 e4m3 values, then 64 bf16
  rope values) plus an 8-byte scale slot (7 UE8M0 exponents, one per 64 values) stored after the page's payloads,
  each page padded to a multiple of 576 bytes. 128 indices per row into it; the extra-pool
  configurations add a second pool of the same format with its own indices and lengths.
- Sizes: batch 1, 2, 4, …, 256, each with 4 to 9 of nine pool sizes (10543–12317 pages, paired with
  one extra pool of 105428–123163 pages in the extra-pool configurations): 61 sizes per configuration. The kernel
  does not depend on the page count. The pools are large (estimated from the shapes: 1.8 GB main pool at 12317
  pages, 4.6 GB extra pool for extra 512).
- Accuracy: bounded; `output` atol = rtol = 1e-2, `lse` atol = rtol =
  1e-3. The reference defines the fp8 → bf16 step as the same bit assembly the kernel
  does, so that step matches it exactly; the error left to the comparison comes from bf16 P, bf16 partials and the
  bf16 output.
- Yardstick: sglang's TileLang `dpsk_v4_fp8_attention_fwd`, not aiter. Another FlyDSL kernel covers extra 8256
  at 19 sizes
  (batch 1–128, pools of 1501–5229 pages) that are not this page's.

## Counter-signals

- **Only nine pool sizes were measured.** The pool of a TP8 decode scenario (12153 main,
  121529 extra pages) is among them at batch 1, 2 and 4 only.
- **The thinnest margin is the window-only configuration at batch ≥ 128**: 1.165–1.188 at batch 128 and 1.193–1.354
  at batch 256 *(field, E1)*, against 1.582–2.405 at batch ≤ 64. The cause is not established.
- **The measurement draws lengths and indices uniformly.** `sparse_lens` and `extra_sparse_lens` are uniform in
  [0, width] and the indices uniform over the whole pool. In serving they follow the
  model's top-k selection, whose index structure moved gather cost on another sparse kernel
  (`../attention/split_kv.md` §7); the speedup on serving inputs is not measured. At batch 1 one
  row's drawn lengths set the work: the nine batch-1 sizes of extra 8256 read 2.809–5.845 *(field, E2)*, and each
  reproduces within 4.1% in the other run *(field, E4)*, so the spread follows each size's inputs
  (drawn from a seed fixed per size), not run noise.
- **`sm_scale` and `kv_scale` are compile-time constants.** The builder is a `functools.cache` keyed by both
  values; a
  new value compiles a new kernel pair at the call that brings it, which may be inside a CUDA-graph capture, where
  launch code must not compile. Every measured size holds `sm_scale` 0.0442 and `kv_scale` 1.0.
- **Workspace kept per batch size.** A `functools.cache` function keyed by batch, the two index widths and the
  device holds a bf16 partial
  output and an fp32 partial lse for every slot a row can write, for the life of the process: estimated 0.26 MB,
  1.3 MB and 8.75 MB per row for no extra, extra 512 and extra 8256, so about 4.5 GB for extra 8256 at its nine batch
  sizes (estimated from the shapes). Every slot the combine reads is written by the same call's partial kernel,
  so no call reads bytes it did not write; two streams calling one size at once would share it.
- **The pool is addressed from the start of its storage.** The kernel takes the pool as
  `torch.as_strided(raw, …, 0)`, offset 0
  of the storage, not `data_ptr()`. A pool view that starts inside a larger allocation would be read
  from the wrong base. The measured pools start at offset 0; sglang's were not checked for this page.
- **The combine's shape suits the extra-pool configurations, not the window-only one.** The combine runs 4 waves per
  workgroup with an LDS merge. `../attention/split_kv.md` §1 and `../gemm/grid_pricing.md` §2 keep one
  wave per combine workgroup unless each output merges many partials: up to 20 per (row, head) for extra 512 and
  133 for extra 8256, against 4 for the window-only one. A one-wave combine gave the whole call 1.33× the time at
  133 partials and 8% more at 20 (batch 8), and 7% less at 4 partials and batch 64 *(field, E5)*.
  Nothing here prices the producer's departures from the split-KV card: a producer workgroup owns
  one 32-key tile, two above 1024 live keys, where §5 of the split-KV card asks for at least two per split; partials
  are bf16, which §5 says adds merge error.
- **Each call goes through a `@flyc.jit` launcher**, which re-binds its arguments on every call
  (`../flydsl/attention.md` § measurement §6, on FlyDSL 0.3.2). The measured ratio is GPU time in graph replays, so
  that host cost is outside it; an eager caller pays it.
- **64 heads are a compile-time constant**: the 128-head DeepSeek-V4-Pro configurations have their own FlyDSL
  kernels.

## Mechanism

Two launches per call: a partial kernel on a grid of (batch, slots) and a combine kernel on (64 heads,
batch), each workgroup 256 threads, 4 waves.

**Slots.** The grid has one slot per 32 indices of each segment: 4 for the 128-wide window, plus 16 (extra 512) or
258 (extra 8256), fixed by the configuration whatever the live lengths. Slot g of a segment covers keys
[g·k, g·k + k): k
= 32, or 64 once that segment's live length passes 1024 (only the 8256-wide segment can reach
it). A slot whose first key is past the live length exits at once. Live partials stay packed at each
segment's base, so the workspace holds 4, 20 or 133 slots per row.

**Gather and dequantise into one LDS tile.** Eight threads per key, 32 keys. Each key's index gives its
page (a shift by log2 of the page size) and slot; seven threads each read 64 fp8 bytes as four 16-byte global
loads plus that block's UE8M0 exponent, and build bf16 values by integer bit assembly: exponent and mantissa
shifted into place, exponent + scale − 7 added in one integer add per pair, sign moved to bit 15. The eighth
thread copies the 64 bf16 rope values. Everything lands in one bf16 LDS tile of 32 keys × 512 dims (32 KiB), stored
as 16 blocks of 32 keys × 32 dims. A key with a negative index or past the live length reads slot 0 and gets a
−1e30 bias in LDS, so its exp2 is 0 (the masking of `../attention/split_kv.md` §4). A payload is
4.5 lines of 128 bytes at phase 0 or 64, so each key touches 5 lines plus its scale (from the layout; §7 of the same
card). The bit assembly of one 64-value block:

```python
# thread (kk, jj): key kk of the tile, block jj of its payload
# the pool is viewed as 32-bit words; pay: the payload's word offset
scw = fx.ptr_load(
    fx.add_offset(
        kvp, page * uwords + scoff + within * 2 + (jj >> 2)
    )
)
scb = (scw >> ((jj & fx.Int32(3)) * 8)) & fx.Int32(255)  # this block's UE8M0 exponent byte
cmv = (scb + fx.Int32(249)) & fx.Int32(255)  # scale − 7, mod 256
c1 = cmv << 7  # at the bf16 exponent field
c2 = c1 | (c1 << 16)  # for both bf16 of a 32-bit pair
nbase = pay + jj * 16
for c in range_constexpr(4):
    v = fx.ptr_load(fx.add_offset(kvp, nbase + c * 4), T.i32x4)  # 16 fp8 bytes
    outs = []
    for q4 in range_constexpr(4):
        w = fx.Int32(v[q4])
        x01 = (w & fx.Int32(255)) | ((w & fx.Int32(65280)) << 8)  # bytes 0 and 1, one per 16-bit half
        x23 = ((w >> 16) & fx.Int32(255)) | (
            (w >> 8) & fx.Int32(16711680)
        )
        # 8323199 = 0x7F007F: exponent and mantissa, up 4 bits; 8388736 = 0x800080: the sign, to bit 15
        o0 = (((x01 & fx.Int32(8323199)) << 4) + c2) | (
            (x01 & fx.Int32(8388736)) << 8
        )
        o1 = (((x23 & fx.Int32(8323199)) << 4) + c2) | (
            (x23 & fx.Int32(8388736)) << 8
        )
        outs.append(o0)
        outs.append(o1)
```

**Logits, heads on the lane.** Each wave owns 16 heads and keeps their q in registers for the whole workgroup (16
loads of 16 bytes per lane). Per 16-key half, 16 `mfma_f32_16x16x32_bf16` with A = keys from LDS and B = q: C holds
one head per lane and the keys on the element index, so the row max and row sum are an in-lane fold over 8 keys
and two `shuffle_xor` (16, 32) (`../attention/mfma_layout.md` §3–4, `../attention/softmax.md`
§5). `sm_scale · kv_scale · log2 e` and the bias go in with one multiply and one add before `exp2`
(`../attention/softmax.md` §3).

```python
# lr = lane % 16, lg = lane // 16; h0 = wave * 16, the wave's first head
sv = []
for t in range_constexpr(2):  # the two 16-key halves of the tile
    sa = Vec.filled(4, 0.0, Float32)
    for ks in range_constexpr(16):  # 16 blocks of 32 dims
        av = fx.ptr_load(
            fx.add_offset(
                vp, ks * 1024 + (16 * t + lr) * 32 + 8 * lg
            ),
            T.bf16x8,
        )
        sa = Vec(
            rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [av, qv[ks], sa])  # A = keys, B = q of head h0 + lr
        )
    bv4 = Vec(
        fx.ptr_load(fx.add_offset(biasp, 16 * t + 4 * lg), T.f32x4)
    )
    for e in range_constexpr(4):
        sv.append(fx.Float32(sa[e]) * scale2 + fx.Float32(bv4[e]))

mblk = sv[0]  # 8 of the tile's 32 keys, all of head h0 + lr
for e in range_constexpr(1, 8):
    mblk = mblk.maximumf(sv[e])
mblk = mblk.maximumf(mblk.shuffle_xor(16, 64))  # the lanes 16 and 32 away hold the head's other keys
mblk = mblk.maximumf(mblk.shuffle_xor(32, 64))
```

**P·V from the same tile.** The values are the first 512 dims of the same latent, so P·V reads the tile again,
transposed, with `ds_read_b64_tr_b16` (`LDSReadTrans16_64b`): 64 `mfma_f32_16x16x16bf16_1k` per 32 keys and wave,
A = P as 4 bf16 per lane straight from the logits layout (keys already on the element index), B = Vᵀ, into 32 f32x4
accumulators per lane, 16 heads × 512 dims per wave (`../attention/lds.md` §1–3, `../flydsl/attention.md` § lds §2,
`../attention/mfma_layout.md` §4). One gather and one dequantisation feed both GEMMs and all 64 heads.

```python
pf = []
for t in range_constexpr(2):  # P of this lane's head, 4 keys per half, rounded to bf16
    pf.append(
        Vec.from_elements(
            [pv[4 * t + e] for e in range(4)], Float32
        ).to(BFloat16).bitcast(Int16)
    )
tr_atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), BFloat16)
tr_layout = fx.make_layout(4, 1)
for n in range_constexpr(32):  # 32 accumulators of 16 dims
    for t in range_constexpr(2):
        base = (n // 2) * 1024 + (16 * t) * 32 + (n % 2) * 16
        elem = base + (lane // 4) * 32 + (lane % 4) * 4
        src = fx.make_view(fx.add_offset(vp, elem), tr_layout)
        dst = fx.make_rmem_tensor(tr_layout, BFloat16)
        fx.copy(tr_atom, src, dst)  # the tile the logits read, now transposed
        bvv = Vec(dst.load()).bitcast(Int16)
        acc[n] = Vec(
            rocdl.mfma_f32_16x16x16bf16_1k(
                T.f32x4, [pf[t], bvv, acc[n]]
            )
        )
```

**Online softmax.** The running max starts at −1e20, so a block of masked keys gives P = 0, not NaN
(`../attention/softmax.md` §6.1). The accumulator holds heads on its element index, so `alpha =
exp2(m_old − m_new)` moves to that layout through 16 floats of the wave's own LDS (an `lgkmcnt(0)` wait, no
barrier) before it scales all 128 accumulators (`../attention/softmax.md` §6.2). Two barriers per
32-key step guard the tile. The row sum adds the fp32 P while P·V takes P rounded to bf16 (§5 of the softmax card
sums the rounded P). At the end a lane scales by 1/l and stores the partial as bf16, 2 bytes per store; lanes 0–15
store lse = (m + log2 l) · ln 2, or −1e30 for an empty slot.

**Combine.** One workgroup per (head, row), 8 dims per lane. Wave w walks slots w, w + 4, … in
order, four per trip: first the max of the partial lse, then Σ exp(lse − max) · partial (bf16 read as fp32); wave 0
merges the four waves' (max, sum, vector) from LDS in wave order. It scales by 1/(1 + exp(sink − lse)) and by
`kv_scale`, writes bf16, and writes the lse without the sink; a row that read nothing gets output 0 and lse +inf.
A wave's two walks:

```python
# msl: this wave; nat: the row's live partials; slot(i): whether partial i is live, and its address
# Each wave takes every MSLICE-th group: its own max, then its own
# scaled sum; the waves are flash-merged through LDS below.
mcur = negb
for ib in range(0, nat, MSLICE * CUNR):  # MSLICE: waves per workgroup; CUNR: slots a wave reads per trip
    for u in range_constexpr(CUNR):
        ok, off = slot(ib + msl + fx.Int32(MSLICE * u))
        lv = fx.ptr_load(fx.add_offset(plp, off))
        mcur = mcur.maximumf(ok.select(lv, negb))

ssum = zerof
macc = Vec.filled(CPT, 0.0, Float32)  # CPT: dims per lane
for ib in range(0, nat, MSLICE * CUNR):
    for u in range_constexpr(CUNR):
        ok, off = slot(ib + msl + fx.Int32(MSLICE * u))
        lv = fx.ptr_load(fx.add_offset(plp, off))
        sc = ok.select(fmath.exp2((lv - mcur) * LOG2E), zerof)
        ov = Vec(
            fx.ptr_load(fx.add_offset(mpo, off * DIM + dim), T.bf16x8)
        ).to(Float32)
        macc = Vec(fx.fma(ov, Vec.from_elements([sc] * CPT, Float32), macc))
        ssum = ssum + sc
```

Not this where each output merges only a few partials: several waves per combine workgroup with an LDS merge. The
one-wave form was the faster one at 4 partials per output and batch 64 (the window-only configuration) and the slower
one at 20 and 133 partials and batch 8 (the extra-pool configurations) *(field, E5)* (§ Counter-signals).

Every reduction, in both kernels, runs in a fixed order, with no atomics. The builder, keyed by the two scales and
each segment's index width and page size, and the workspace are module-level `functools.cache` functions; a call
without an extra pool
passes the window's tensors as the second segment with width 0, which compiles that segment away.

## Result

The yardstick at every size is sglang's TileLang `dpsk_v4_fp8_attention_fwd`; the other extra-8256 kernel covers
pools of 1501–5229 pages, none of these sizes. The speedup is a one-sided lower bound (α = 0.05) of baseline /
kernel time over 32 rounds of CUDA-graph replays with a 1 GiB cache flush before each call; a size counts as won
from 1.03.

The numbers were measured on FlyDSL 0.3.2, a pin installed over a container image that ships
0.3.4.1 (`../pitfalls/measured_a_different_build.md`).

Per batch, the range over that batch's pool sizes (9, 9, 7, 6, 8, 8, 6, 4 and 4 sizes):

| Configuration | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | ≥ 1.03 |
|---|---|---|---|---|---|---|---|---|---|---|
| no extra *(field, E1)* | 1.901–2.405 | 2.257–2.399 | 2.030–2.400 | 2.244–2.367 | 2.274–2.394 | 1.906–2.033 | 1.582–1.646 | 1.165–1.188 | 1.193–1.354 | 61 of 61 |
| extra 8256 *(field, E2)* | 2.809–5.845 | 4.069–5.486 | 3.772–5.269 | 3.342–3.967 | 2.811–3.162 | 2.467–2.734 | 2.396–2.625 | 2.428–2.553 | 2.482–2.560 | 61 of 61 |
| extra 512 *(field, E3)* | 2.620–3.493 | 2.674–3.571 | 3.300–3.585 | 3.114–3.414 | 2.688–2.790 | 2.362–3.614 | 2.738–3.385 | 3.429–3.910 | 4.645–5.078 | 61 of 61 |

183 of the 183 sizes are at or above 1.03, and all pass the accuracy check. A separate run, the earlier of the two and
on the same image, also read all 183 at or above 1.03 *(field, E4)*.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-sided lower bound (α = 0.05) of baseline / kernel time over 32 rounds of CUDA-graph replays with a 1 GiB cache flush before each call] no extra: speedup over sglang's TileLang dpsk_v4_fp8_attention_fwd 1.165–2.405 per size, 61 of 61 ≥ 1.03; batch 1 1.901–2.405, 2 2.257–2.399, 4 2.030–2.400, 8 2.244–2.367, 16 2.274–2.394, 32 1.906–2.033, 64 1.582–1.646, 128 1.165–1.188, 256 1.193–1.354 (61 sizes)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] extra 8256: speedup 2.396–5.845 per size, 61 of 61 ≥ 1.03; batch 1 2.809–5.845, 2 4.069–5.486, 4 3.772–5.269, 8 3.342–3.967, 16 2.811–3.162, 32 2.467–2.734, 64 2.396–2.625, 128 2.428–2.553, 256 2.482–2.560 (61 sizes)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] extra 512: speedup 2.362–5.078 per size, 61 of 61 ≥ 1.03; batch 1 2.620–3.493, 2 2.674–3.571, 4 3.300–3.585, 8 3.114–3.414, 16 2.688–2.790, 32 2.362–3.614, 64 2.738–3.385, 128 3.429–3.910, 256 4.645–5.078 (61 sizes)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=2, two separate runs on the same image, the second the run of E1–E3; the first run's ruler not stated in the record] the two runs, in order: no extra 1.167–2.439 then 1.165–2.405, extra 8256 2.374–5.616 then 2.396–5.845, extra 512 2.363–5.019 then 2.362–5.078, all 183 sizes ≥ 1.03 both times; per-size ratio between the runs 0.924–1.070; extra 8256 at batch 1 by pool size 2.753, 5.616, 5.542, 5.027, 5.246, 5.523, 3.468, 4.491, 2.791 then 2.809, 5.845, 5.595, 5.071, 5.357, 5.526, 3.444, 4.494, 2.861 (the three configurations)
- E5 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=7 rounds, CUDA-graph replay with a 1 GiB flush per call, arms interleaved] whole-call time, one-wave over four-wave combine, 64 heads, every slot live: 133 partials 1.331 and 20 partials 1.080 at batch 8; 4 partials 0.934 at batch 64; every arm within the accuracy tolerances (the same kernel with only the combine's waves and LDS struct changed)
