---
title: "Sparse MLA of sglang's DSA backend with one query token per sequence, 8 heads, fp8 q and 576-byte KV rows, top-2048, at 32, 128 and 256 tokens — a wave-private buffer-to-LDS ring feeds fp8 MFMAs; 8, 2 or 1 splits per token, merged inside the launch by the last split to arrive"
kind: case
ops: [mla, attention]
dtypes: [fp8, bf16]
regimes: ["tokens 32", "tokens 128", "tokens 256"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over sglang's triton_sparse_mla_decode_splitk, one-sided lower bound of the per-round time ratio, per size, measured on an earlier build of this kernel", value: "1.642 / 1.592 / 1.265 at 32 / 128 / 256 tokens", gpu: "MI355X", workload: "8 heads, fp8 q (tokens, 8, 576) and KV pool (slots, 576), top-2048, bf16 output (tokens, 8, 512); (n_tokens, n_slots) = (32, 65536), (128, 262144), (256, 524288)", date: "2026-11-09"}
  - {metric: "speedup over the older Triton (Gluon) kernel of the same op (the yardstick at 256 tokens; at 32 and 128 tokens a newer Gluon kernel is faster than it), same run, one-sided lower bound per size, measured on an earlier build of this kernel", value: "1.536 / 1.319 / 1.198 at 32 / 128 / 256 tokens", gpu: "MI355X", workload: "the same three sizes", date: "2026-11-09"}
---

# Sparse MLA decode for sglang's DSA backend, 8 heads: one launch, splits merged by the last to arrive

One FlyDSL kernel for the sparse MLA decode of sglang's DSA backend at 8 heads, the decode attention of GLM-5.2 at 8
heads per rank (TP8), built for three sizes. The padded DeepSeek-V4 pool is a different op, run by a two-launch kernel
for 64 heads: [mla_sparse_fp8_kv_pool.md](mla_sparse_fp8_kv_pool.md).

## Applies when

- Row-major operands: q (tokens, 8, 576) and the pool (n_slots, 576) float8_e4m3fn, unscaled, a row being 512 latent
  values then 64 RoPE values; top-k (tokens, 2048) int32, distinct, −1 for none; `sm_scale` fp32; output (tokens, 8,
  512) bf16, 0 for a token that reads no row. Two runs must be bit-identical, and the output must reach SQNR ≥ 20 dB
  against the reference.
- Sizes, three pairs and nothing between them: (n_tokens, n_slots) = (32, 65536), (128, 262144) and (256, 524288), each
  with `sm_scale` 0.0625. The kernel is compiled per token count and takes the pool's size as a launch argument, so a
  token count runs at every pool size; only pools of tokens × 2048 rows (36, 144 and 288 MiB) were timed.
- Not covered: the other 33 decode sizes sglang pads TP8 batches to and 16 heads (covered by Gluon kernels and by
  [dsa_sparse_mla_16_heads.md](dsa_sparse_mla_16_heads.md)); another top-k or head count; draft steps of speculative
  decoding and a q that is not one contiguous (tokens, heads, 576) view, which stay on sglang's kernel.
- Yardstick at each size: the faster of the baseline, sglang 0.5.22's `triton_sparse_mla_decode_splitk`, and the fastest
  Triton (Gluon) kernel of the same op there. Those come in two generations (§ Result). The newer one, which reads the
  fp8 q directly, measured 1.226 and 1.343 over the baseline at 32 and 128 tokens in a separate run of § Result's
  ruler, and 0.99 at 256 tokens, so there the older one stays the yardstick *(field, E5)*. The older Gluon
  kernels, one per size, beat the baseline at all three sizes in a repeat measurement *(field, E6)*. So the yardstick
  is the newer Gluon kernel at 32 and 128 tokens and the older one at 256, each faster than the baseline at its size
  (§ Result).

## Counter-signals

- **The pool sizes timed are tied to the token count.** The kernel is compiled once per token count, cached on
  `n_tokens` alone: `sm_scale · log2 e`, the pool's size in bytes and the key sort's bucket shift are launch arguments,
  so a call at a pool size or scale not seen before runs the kernel already compiled for its token count. A bucket of
  the key sort is the smallest power of two of slots that covers the pool in 16 or 64 buckets, so at a pool that is not
  16 or 64 times a power of two the last buckets hold fewer keys or none. The speed at serving pool sizes is not
  measured.
- **Uniform indices.** The timed inputs draw each token's 2048 slots uniformly over the pool. Serving top-k sets overlap
  between adjacent tokens, which moved gather cost on another sparse decode (`../attention/split_kv.md` §7); the key
  sort at 128 and 256 tokens was not measured on such indices.
- **At 256 tokens the margin owes nothing to the merge.** There this kernel and the yardstick, the older Gluon kernel,
  both run one workgroup per token with no split; the 1.198 there comes from the producer and from the bf16 copy of q
  that the Gluon kernel launches first and this kernel does not need.
- **Half of every MFMA is padding.** 8 heads fill 8 of the 16 MFMA columns of both GEMMs; the other 8 carry zero q.
- **P·V is on the legacy fp8 MFMA**, which takes the bf16 cycle count for the same K (`../attention/mfma_layout.md` §8);
  only the latent part of QK uses the wide-K form. P is rounded to fp8 for P·V while its row sum stays fp32. It is
  scaled by 2^8 first, so a weight keeps a normal e4m3 value down to 2^−14 of the running max and rounds to zero only
  under about 2^−18; unscaled the limits are 2^−6 and 2^−10, and a key e^−8 (about 2^−11.5) below the max would add
  nothing. Smaller weights still round to zero, so an input where each wave's other keys score about 12.5 or more below
  one key whose latent values are zero misses the 20 dB bound: about 12 dB at a gap of 12 and 0 dB from 13 (estimated, a
  CPU model of the arithmetic). sglang's kernel, the baseline, multiplies P by the fp8 maximum, 448, before its
  conversion, and meets the same limit within a factor of 1.75.
- **Measured on an earlier build.** Every time, bound and stress result on this page is of an earlier build of this
  kernel. The kernel differs from it in two ways: P is scaled by 2^8 before its fp8 conversion, and `sm_scale · log2 e`,
  the pool's size and the key sort's bucket shift are launch arguments rather than compiled in. At the three timed pools
  the bucket width is the same power of two as before (16384 slots at 128 tokens, 8192 at 256), so the key order is
  unchanged. The kernel in this form has not been run on a GPU: its speed and its output are not measured.
- **One FlyDSL only.** The earlier build was measured on FlyDSL 0.3.2 alone, pinned by hand over the image's 0.3.4.1,
  with aiter installed without its dependencies. An install that takes aiter with its dependencies runs the FlyDSL that
  aiter pins, and that combination was not measured (`../pitfalls/measured_a_different_build.md`).
- **What the cross-workgroup merge was run under.** Besides the accuracy check of the timing run, each try below was
  compared byte for byte with the same call alone, at all three sizes: on a stream limited to 16 CUs and on one limited
  to 64, beside a second call on the same inputs, and in a replayed graph, 20 tries each, none differing and none hung
  *(field, E2)*; two calls at once on two streams with different inputs, unrestricted and with both streams on the same
  64 CUs, 20 tries each, none differing *(field, E4)*. Nothing beyond two concurrent calls of this op was run, and no
  other kernel was resident beside it. No workgroup waits for another (`../attention/split_kv.md` §2), and each call
  gets its own `torch.empty` partials and arrival words. The merge crosses XCDs only at 32 tokens, where the 8 splits of
  a token are consecutive workgroups on 8 XCDs (`../attention/grid_dispatch.md` §2); at 128 tokens the two splits of
  token t are workgroups t and t + 128, on one XCD.
- **Workspace reuse is tested for the op alone, not in a model's graph.** Arrival words are not cleared between calls,
  so whether a word belongs to this launch rests on its tag alone (§ Mechanism). That also sends every split of the next
  launch down the compare-swap path: the 16-head kernel stores zero once a word is complete, which, alone in a 16-head
  port, was 1.7–2.1% faster at 1 token and level at 32 and 256
  ([dsa_sparse_mla_16_heads.md](dsa_sparse_mla_16_heads.md); `../attention/split_kv.md` §3, its E51); this kernel was
  not timed with it. One captured graph per size, replayed 24 times with three different input sets copied into its
  buffers in turn, so every replay meets the words the replay before it left, matched an eager call on the same inputs
  at every replay *(field, E3)*. Inside a model's decode graph the call's workspace comes from the graph's memory pool,
  which other kernels may write between replays; that has not been run.
- **Two queues can share a tag.** A launch on a queue whose object sits at address a and one on a queue at b get equal
  tags exactly when the first's dispatch id minus the second's is δ = (b − a) · H2 · H1⁻¹ mod 2^56, one value fixed by
  the two addresses. A flag buffer last written on one queue and read by a launch on the other at that distance would be
  misread: a split could merge before this launch's partials have landed, with no error. Not checked: where δ falls for
  the queues ROCm creates; whether one flag buffer moves between queues; δ = 0 cases — a queue destroyed and recreated
  at the same address restarts its dispatch ids at 0, and a graph captured on one stream and replayed on another may
  carry the capture-time queue address; and whether the address `llvm.amdgcn.queue.ptr` returns on this build is
  non-zero and distinct per queue.
- **Cross-XCD visibility rests on scope bits, not fences.** Partials, maxima, sums and the merge's loads use `SC1` (aux
  16) and the kernel has no fence. `SC1` stores were bitwise clean across XCDs in another decode kernel
  (`../attention/split_kv.md` §2, its E5); the ISA does not say which L2 case gfx950 is
  (`../hardware/gfx950/memory_and_sync.md` §2). The tests here do not add to that: the timing run flushes 1 GiB before
  each call and the graph test copies the 36 MiB pool or more between replays, so none is known to have left a stale
  partial line in an XCD's L2.

## Mechanism

**Grid.** 4 waves (256 threads) per workgroup; the number of splits per token keeps the grid at 256 workgroups, one per
CU of an MI355X:

| Size | splits per token | workgroups | keys per wave | 32-key chunks per wave | split merge |
|---|---|---|---|---|---|
| 32 tokens | 8 | 256 | 64 | 2 | in the launch, across XCDs |
| 128 tokens | 2 | 256 | 256 | 8 | in the launch |
| 256 tokens | 1 | 256 | 512 | 16 | none: the output is stored directly |

A workgroup takes 148–156 KiB of LDS (from the source's shapes: two ring slots of 4 waves × 32 rows × 592 B, plus the
sort's buffer at 128 and 256 tokens), so one fits per CU and the grid runs as one round. The kernel switches features by
split count: split-fastest workgroup order and a last chunk scored in two halves at 8 splits; the key sort, key validity
from a ballot and `SC1` gathers at 2 and 1; `v_readlane` addressing, a peeled first chunk and each (token, head)'s two
maxima and sums side by side at 2; all of a chunk's `ds_bpermute`s in flight before the first is used at 1.

```python
# S: splits per token (8, 2 or 1); each switch is a compile-time constant of the kernel built for a token count
SORT = S <= 2  # each wave reorders its keys by pool bucket, so every CU sweeps the pool in the same order
NB = 64 if S == 1 else 16  # pool buckets of the address sort
BALLOT = S <= 2  # key validity from a ballot per index register instead of ds_bpermute per chunk
RLADDR = S == 2  # DMA offsets from v_readlane instead of ds_bpermute
BATCH = S == 1  # a chunk's ds_bpermutes all in flight before the first is used
VPRE = 28 if S == 2 else 16  # V^T reads issued before the softmax in a chunk that releases no ring slot
XMAJ = S == 8  # split-fastest workgroup order: a token's splits run on distinct XCDs
KV_AUX = SC1 if S <= 2 else 0  # cache policy of the KV gather DMAs
PLC = S == 2  # a (token, head)'s split maxima and sums side by side: one merge load
PEEL0 = S == 2  # chunk 0 out of the loop: O^T and l are still zero, so its rescale is the wave max alone
SPLITW = S == 8  # the last chunk's scores in two halves, the first on the rows its first DMAs fill
```

**DMA ring.** Two ring slots; each wave owns a 32-row region of each, rows pitched 592 B (576 + 16, so the rows one K or
Vᵀ read touches start in distinct bank groups). A chunk is 18 `buffer_load … lds` of 16 B per lane, 1 KiB each, and a
19th on 32 lanes (512 B), filling the region in byte order: consecutive lanes read consecutive 16 B of one row
(`../attention/split_kv.md` §7), and a lane that falls on a row's 16-byte pad gets an offset past the pool's size, so
the buffer returns zeros and fetches nothing (`../hardware/gfx950/memory_and_sync.md` §2–3). A lane's row comes from the
wave's index registers by `ds_bpermute`, or, from the third chunk on at 2 splits, by `v_readlane` of the chunk's 32
keys; an index of −1 reads row 0 and its score is set to −inf. Chunks 0 and 1 are issued first; chunk c is computed
while c + 1 lands, and once every LDS read of c's slot has returned (`s_waitcnt lgkmcnt(0)`), the DMA of c + 2 into that
slot goes out before the last 16 P·V MFMAs. The loop has no workgroup barrier: each wave waits on its own `vmcnt`. A
576-byte row starts at phase 0 or 64 and touches 5 lines.

```python
RP = D + 16  # LDS pitch of a ring row: the rows one K or V^T read touches start in distinct bank groups

# buffer offsets of this lane's DMAs of a chunk: they fill the wave's region in byte order
for i in range_constexpr(NDMA):
    b = lane * 16 + DMA_BYTES * i
    key = b // RP
    col = b - key * RP
    ri = fx.Int32(
        rocdl.ds_bpermute(T.i32, fx.Int32((key + half * KW) * 4).ir_value(), sel.ir_value())
    )
    ri = (ri < fx.Int32(0)).select(fx.Int32(0), ri)
    # a lane on a row's pad reads past the pool: the buffer returns zeros and fetches nothing
    offs.append((col < fx.Int32(D)).select(ri * D + col, fx.Int32(pool_bytes)))

def one(i):  # one DMA: 16 B per lane, from the pool into this wave's region of the slot (lptr)
    rocdl.raw_ptr_buffer_load_lds(
        rsrc,
        _gep(lptr, DMA_BYTES * i),
        fx.Int32(16),
        fx.Int32(offs[i]),
        fx.Int32(0),
        fx.Int32(0),
        aux=KV_AUX,
    )

for i in range_constexpr(NDMA - 1):
    one(i)
if lane < fx.Int32(NLAST):  # the last DMA only on the lanes still inside the region
    one(NDMA - 1)
```

```python
# ring: chunks 0 and 1 in flight, then compute chunk c while c+1 lands and c+2 is issued
dma(addrs(0), fx.Int32(0))
dma(addrs(1), fx.Int32(1))
for c in range(c0, NC - 2):
    cc = fx.Int32(c)
    slot = cc % 2
    nxt = addrs_rl(cc + 2) if RLADDR else addrs(cc + 2)
    rocdl.s_waitcnt(vmcnt=NDMA)  # chunk c has landed; only chunk c+1's DMAs are still out
    acc, m, l = compute(cc, slot, acc, m, l, functools.partial(dma, nxt, slot))

# inside compute(), ahead of the chunk's last 16 P.V MFMAs; rel is the DMA of chunk c+2 into this slot
if const_expr(rel is not None and n0 + VB == DV // 16):
    # every read of the slot has landed: its next DMA goes out before the last P.V MFMAs
    rocdl.sched_barrier(0)
    rocdl.s_waitcnt(lgkmcnt=0)
    rel()
    rocdl.sched_barrier(0)
```

**Key sort** (128 and 256 tokens). Each wave counting-sorts its keys by pool bucket (16 buckets at 2 splits, 64 at 1; a
bucket is a power of two of slots wide, so it is the slot shifted right by a launch argument), chunk 0's keys, already
in flight, kept first; the source's aim is that every CU sweeps the pool in the same order. The order is fixed by the
indices, so the result stays bitwise repeatable. Its effect was not measured apart.

**fp8 MFMAs and the softmax.** The token's q stays in registers as fp8 for the whole workgroup, the B operand, heads 0–7
on lanes 0–7 of each 16-lane group. Per 16-key tile: 4 `mfma_scale_f32_16x16x128_f8f6f4` with unit E8M0 scales over the
512 latent values and 2 `mfma_f32_16x16x32_fp8_fp8` over the 64 RoPE values, A = key rows from LDS. A lane holds one
head with its keys on the element index, so the row max is an in-lane fold plus two `shuffle_xor`
(`../attention/mfma_layout.md` §4). `sm_scale · log2 e` is one multiply before `exp2`. The running max and the
accumulators are rescaled only when a ballot shows some lane's chunk max above the running max (a wave-uniform branch,
threshold 0, so a skipped rescale is a multiplication by exactly 1; `../attention/softmax.md` §6.3). P is exp2(s − (m −
8)), the weight scaled by 2^8 (at most 256, under e4m3's 448), summed in fp32 and packed to fp8 (`cvt_pk_fp8_f32`) as
the B operand of P·V; the sum carries the same factor and every output is divided by a sum, so it cancels. P·V is 32
`mfma_f32_16x16x32_fp8_fp8` per chunk with A = Vᵀ read from the same ring rows by `ds_read_tr8_b64`, into 32 f32x4
accumulators (512 values × 16 head columns). A chunk issues 16 of those reads (28 at 2 splits) before the softmax, all
32 when its slot is reused next, so they land under it. Nothing is converted to bf16; the older Gluon kernels copy q to
bf16 in a launch of their own first, and the newer ones read the fp8 q directly.

**Workgroup merge.** After its last chunk each wave writes its fp32 Oᵀ to its own ring region and its max and sum to
LDS; one barrier; 32 threads per head then combine the 4 waves in wave order, 16 output columns per thread. At 256
tokens this is the output, stored as bf16 with `SC1` write-through stores.

**Split merge inside the launch** (32 and 128 tokens). Thread h·32 … h·32 + 31 of a workgroup hold head h, so wave w
holds heads 2w and 2w + 1, and each (token, wave) has its own 64-bit arrival word on its own 128-byte line. Its low byte
is the mask of splits that published; the other 56 bits hold the launch's tag, the low 56 bits of `dispatch.id` (which
counts per HSA queue) · H1 + the address of the launch's queue object (`llvm.amdgcn.queue.ptr`) · H2, with two odd
64-bit constants H1 and H2: tag bits 0–23 above the mask, bits 24–55 in the high half. The low bits of a product depend
only on the low bits of its factors, and an odd multiplier permutes the residues modulo 2^56, so the tag changes with
any change in the low 56 bits of either input alone, and on one queue two dispatch ids give the same tag only 2^56
dispatches apart.

```python
MBITS = 8  # low bits of an arrival word: the mask of the splits that arrived; the tag sits above
TAG_H1 = 0x9E3779B97F4A7C15 - (1 << 64)  # odd multipliers of the launch tag (signed i64)
TAG_H2 = 0x632BE59BD9B4E019

# the launch's tag: dispatch id * H1 + queue address * H2; the word keeps its low 64 - MBITS bits
did = llvm.call_intrinsic(T.i64, "llvm.amdgcn.dispatch.id", [], [], [])
qptr = llvm.call_intrinsic(llvm.PointerType.get(4), "llvm.amdgcn.queue.ptr", [], [], [])
qpi = llvm.ptrtoint(T.i64, qptr)
tag64 = arith.addi(
    arith.muli(did, arith.constant(T.i64, TAG_H1)), arith.muli(qpi, arith.constant(T.i64, TAG_H2))
)
tag_lo = fx.Int32(rocdl.readfirstlane(T.i32, arith.trunci(T.i32, tag64))) << MBITS
tag_hi = arith.trunci(T.i32, arith.shrui(tag64, arith.constant(T.i64, 32 - MBITS)))
tag_hi = fx.Int32(rocdl.readfirstlane(T.i32, tag_hi))
mybit = fx.Int32(1) << split
pub = Vec.from_elements([tag_lo | mybit, tag_hi], Int32).bitcast(Int64)[0]  # what a split ORs into the word
```

1. Each wave issues one `SC1` load of its arrival word, then stores its heads' normalized bf16 partials and one
   (max, sum) per head with `SC1` write-through stores.
2. If the word carries this launch's tag and every other split's bit, the wave merges at once and does not publish,
   so no other wave can complete the mask.
3. Otherwise it waits `vmcnt(0)`: its stores are acknowledged, and they are all the word vouches for, since the
   word covers only this wave's heads. That wait is the release; there is no fence and no workgroup barrier.
   Lane 0 then ORs (tag | its bit) into the word with one 64-bit atomic that returns the old word; the wave whose
   OR completes the mask merges.
4. A word that carries another launch's tag is claimed with one 64-bit compare-swap from the value read in step 1;
   a failed swap means a partner claimed it for this launch, and the wave ORs its bit as in step 3 (at 2 splits,
   a returned word already holding the partner's bit lets it merge without publishing).
5. The merge issues all its loads at once, with `SC1`: the splits' maxima and sums and 2 × 16 B of partials per
   split and thread, before its first wait. The merging wave's own max and sum come from registers. It weighs the
   splits in index order, so the output bits do not depend on which split merges, and stores bf16 with `SC1`.

```python
# w_lo, w_hi: the arrival word read in step 1, ahead of the partial stores
tagok = ((w_lo & fx.Int32(~MMASK)) == tag_lo) & (w_hi == tag_hi)
have = tagok.select(w_lo & fx.Int32(MMASK), fx.Int32(0))
fast = (have | mybit) == full

def arrive_or():
    """Set this split's bit (the word carries the tag); the split completing the mask merges."""
    old = llvm.call_intrinsic(
        T.i64,
        "llvm.amdgcn.raw.ptr.buffer.atomic.or.i64",
        [pub.ir_value(), fl_at, aoff.ir_value(), zi, zi],
        [],
        [],
    )
    o_lo = fx.Int32(rocdl.readfirstlane(T.i32, arith.trunci(T.i32, old)))
    if ((o_lo & fx.Int32(MMASK)) | mybit) == full:
        merge()

if fast:
    # every partner already published: this wave merges without publishing (so no other can)
    merge()
else:
    rocdl.s_waitcnt(vmcnt=0)  # the partials have landed before this split's bit says so
    if tagok:
        arrive_or()
    else:
        # a word an earlier launch left: claim it for this launch; a failed claim returns the word
        # a partner claimed, and with two splits its bit lets this wave merge without publishing
        wold = Vec.from_elements([w_lo, w_hi], Int32).bitcast(Int64)[0]
        r = rocdl.raw_ptr_buffer_atomic_cmpswap(
            T.i64, pub.ir_value(), wold.ir_value(), fl_at, aoff.ir_value(), zi
        )
        ...  # r differs from wold: a partner claimed the word first, so arrive_or(); at 2 splits, merge()
        # instead when r already carries this launch's tag and the partner's bit
```

No loop polls, nothing sleeps, and a merge is one round of loads by one wave; the 4 waves of a token merge
independently and may sit in different workgroups. `../attention/split_kv.md` §3 measured a "last workgroup
reduces" fold losing where a row's reduce ran as a loop of dependent memory round trips in one workgroup at a few
rows, or where the reduce's registers sat on a 16-head producer. Here the merge was not measured apart from the
rest of the kernel (§ Result).

## Result

One run on an MI355X with FlyDSL 0.3.2: CUDA-graph replays with a 1 GiB read flush before each call, every participant
timed once per round in a seeded order, 16 rounds at each size. A speedup is the one-sided Student-t lower bound, α
0.025 per look, of the geometric mean of the per-round time ratios (the bound is on the mean of their logarithms); times
are medians of the 16 rounds. The "baseline" participant is sglang 0.5.22's `triton_sparse_mla_decode_splitk`. One
session on one GPU timed the earlier build of this kernel (§ Counter-signals) and the two newer Gluon kernels, each
paired with the baseline and with the older Gluon kernel of its size. "This kernel" in the two tables below is that
earlier build.

| Size | this kernel | baseline | older Gluon kernel | bound over baseline | bound over older Gluon kernel |
|---|---|---|---|---|---|
| 32 tokens *(field, E1)* | 12.04 µs | 19.79 µs | 18.50 µs | 1.642 | 1.536 |
| 128 tokens *(field, E1)* | 27.63 µs | 44.02 µs | 36.46 µs | 1.592 | 1.319 |
| 256 tokens *(field, E1)* | 47.36 µs | 59.98 µs | 56.80 µs | 1.265 | 1.198 |

The older Gluon kernels beat the baseline at every size, so against them the per-size speedup is 1.536, 1.319 and 1.198.
At 256 tokens the older kernel is the yardstick; at 32 and 128 tokens the yardstick is the newer Gluon kernel (§ Applies
when), which this session timed but not paired with this kernel (below). All three sizes passed the accuracy check with
a violation fraction of 0.

The newer Gluon kernels, one for 32 tokens and one for 128 and 256, are a revision of the older ones that reads the fp8
q directly; they are the yardstick at 32 and 128 tokens (§ Applies when). In the same session *(field, E1)*:

| Size | this kernel | newer Gluon kernel | bounds of the newer Gluon kernel, over baseline / older Gluon kernel |
|---|---|---|---|
| 32 tokens | 12.04 µs | 16.15 µs | 1.224 / 1.143 |
| 128 tokens | 27.63 µs | 32.89 µs | 1.336 / 1.113 |
| 256 tokens | 47.36 µs | 61.59 µs | 0.979 / 0.925, slower than both |

The two were not timed against each other: the ratio of their medians, 1.34, 1.19 and 1.30, is estimated, and at 32 and
128 tokens it is the only figure against the yardstick there.

How the kernels split the work at each size, from their sources:

| Size | older Gluon kernel | newer Gluon kernel | this kernel |
|---|---|---|---|
| 32 tokens | q copied to bf16; 16 splits per token; merge launch | 8 splits; merge launch | 8 splits; merged in the launch |
| 128 tokens | q copied to bf16; 2 splits; merge launch | 4 splits; merge launch | 2 splits; merged in the launch |
| 256 tokens | q copied to bf16; one workgroup per token | 2 splits; merge launch | one workgroup per token |

At 32 tokens the newer Gluon kernel splits a token 8 ways as this kernel does but merges in a second launch; the
producers differ as well, so the 1.34 is not the merge's alone. No ablation separates the merge, the ring, the fp8 q or
the key sort. Rows × 576 B over the median time is 3.1, 5.5 and 6.4 TB/s at 32, 128 and 256 tokens (estimated; rows
shared between tokens may hit in cache): at 2 chunks per wave the 32-token size is far from the bandwidth the 256-token
size reaches (`../attention/split_kv.md` §6).

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=1, one-sided Student-t lower bound of the geometric mean of the per-round time ratios, 16 rounds per size of CUDA-graph replays with a 1 GiB read flush before each call, every participant timed once per round in a seeded order] one session on one GPU: the earlier build of this kernel and the two newer Gluon kernels, each paired with the baseline (sglang 0.5.22's `triton_sparse_mla_decode_splitk`) and with the older Gluon kernel of its size; medians of 16 rounds and lower bounds, this kernel / baseline / older Gluon kernel: 32 tokens 12.037 / 19.786 / 18.500 µs, bounds 1.642 and 1.536; 128 tokens 27.625 / 44.020 / 36.465 µs, 1.592 and 1.319; 256 tokens 47.363 / 59.979 / 56.798 µs, 1.265 and 1.198; the older Gluon kernels: three, one per size; the newer Gluon kernel for 32 tokens 16.151 µs, bounds over the baseline / the older Gluon kernel 1.224 / 1.143; the newer Gluon kernel for 128 and 256 tokens at 128 tokens 32.895 µs, 1.336 / 1.113, at 256 tokens 61.587 µs, 0.979 / 0.925, slower than both; all six accuracy passed, worst violation fraction 0.0
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=20, stress runs of the earlier build of this kernel, each try byte-compared with the call alone, on the timing inputs] at each of the three sizes: on a 16-CU stream, on a 64-CU stream, beside a second call on the same inputs, and in a graph replayed with the same inputs, 20 tries each, 0 differ, no hang
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=24, one captured graph per size, three input sets with differing outputs copied into its buffers in turn, each replay byte-compared with an eager call on the same inputs] the earlier build of this kernel: 24 replays at each of the three sizes, 0 differ
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=20, two calls at once on two streams, each on its own input set and outputs, each byte-compared with an eager call alone] the earlier build of this kernel, at each of the three sizes, with the two streams unrestricted and with both on the same first 64 CUs, 20 tries each, 0 differ
- E5 [2026-11, gfx950, Gluon, N=1, a separate run of the ruler of E1] speedups over the baseline of the newer Gluon kernels: the one for 32 tokens at (32, 65536) 1.226; the one for 128 and 256 tokens at (128, 262144) 1.343 and at (256, 524288) 0.99; accuracy passed
- E6 [2026-11, gfx950, Gluon, N=1, a repeat measurement, ruler not stated in the record] speedups over the baseline of the three older Gluon kernels: at (32, 65536) 1.062, at (128, 262144) 1.196, at (256, 524288) 1.05; accuracy passed
