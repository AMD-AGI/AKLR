---
title: "Sparse MLA of sglang's DSA backend with one query token per sequence, 16 heads (GLM-5.2 at TP4), fp8 q and 576-byte KV rows, top-2048, at the token counts sglang pads TP4 batches to — the 8-head kernel at 16 heads; 16, 8, 2 or 1 splits per token, each split merging its own slice of the output after a bounded wait"
kind: case
ops: [mla, attention]
dtypes: [fp8, bf16]
regimes: ["tokens 1-8", "tokens 12-32", "tokens 40-128", "tokens 136-256"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over sglang's triton_sparse_mla_decode_splitk, one-sided lower bound of the per-round time ratio, per size", value: "1.291–1.768 at all 36 sizes", gpu: "MI355X", workload: "16 heads, fp8 q (tokens, 16, 576) and KV pool (slots, 576), top-2048, bf16 output (tokens, 16, 512); tokens 1, 2, 4, 8, 12, 16, 24, 32 and 40–256 in steps of 8, pools of max(65536, 2048 × tokens) rows", date: "2026-11-10"}
  - {metric: "speedup over the Triton (Gluon) kernel of the same op, at the sizes where that kernel is at least 1.03 over sglang's; same run, one-sided lower bound per size", value: "1.025–1.454 at the 27 sizes that have one", gpu: "MI355X", workload: "the same operands, the 27 token counts other than 12, 24, 56 and 136–176", date: "2026-11-10"}
---

# Sparse MLA decode for sglang's DSA backend, 16 heads: the 8-head kernel, with a per-slice merge

One FlyDSL kernel for the sparse MLA decode of sglang's DSA backend at 16 heads, the decode attention of GLM-5.2 at 16
heads per rank, built for 36 of the 36 token counts that sglang pads TP4 batches to. It is
[dsa_sparse_mla_8_heads.md](dsa_sparse_mla_8_heads.md)'s kernel ported to 16 heads; everything that page says of the
ring, the key sort and the fp8 MFMAs holds here unless this page says otherwise.

## Applies when

- The operands of the 8-head page, with 16 heads: q (tokens, 16, 576) and the pool (n_slots, 576) float8_e4m3fn,
  unscaled; top-k (tokens, 2048) int32; output (tokens, 16, 512) bf16. Two runs must give the same bytes, and the output
  must reach SQNR ≥ 20 dB against the reference, violation fraction 0.
- Sizes: (n_tokens, n_slots) with n_slots = max(65536, 2048 · n_tokens), `sm_scale` 0.0625: all 36, n_tokens 1, 2, 4, 8,
  12, 16, 24, 32 and 40–256 in steps of 8. The kernel is compiled per token count and takes the pool's size as a launch
  argument.
- Yardstick at each size: the faster of the baseline, sglang 0.5.22's `triton_sparse_mla_decode_splitk`, and a Triton
  (Gluon) kernel of the same op, which counts at the sizes where it measured at or above 1.03 over the baseline: 27 of
  the 36, all but 12, 24, 56 and 136–176 tokens.
- Not covered: the draft steps of speculative decoding and a q that is not one contiguous (tokens, 16, 576) view (both
  stay on sglang's kernel), another top-k or head count.

## Counter-signals

- **A split count that the sizes timed during tuning do not reach is untested.** Tuning timed 1, 32 and 256 tokens, so
  the 2-split path never ran, and that path's first form returned wrong outputs at every 2-split size (40–128 tokens): a
  waiting split loaded its partner's max and sum before the wait. The load now runs after it (§ Mechanism). Time, or at
  least stress, every split count a kernel runs (`../attention/split_kv.md` §2, its E50).
- **The tag shrinks with the split count.** The arrival word keeps 2S low bits for the two masks, so the tag has
  64 − 2S bits: 32 at 16 splits (1–8 tokens). The flag block is a `torch.empty` per call; if it holds arbitrary
  bytes, a word passes as this launch's with odds of about 2^−32 for the tag times about 1% for the mask checks
  (estimated), and up to 32 words are read per call. A word that passes makes splits merge before their partners
  have published, with no error. A word this kernel leaves is zero, so the 32-bit tag matters only for bytes another
  kernel wrote into the block; `../attention/split_kv.md` §3 asks for a wider tag. The 8-head page lists the tag's
  other risks (two queues sharing a tag).
- **The wait costs time when a partner is not running.** On a stream limited to part of the CUs a waiting split spins up
  to 2000 `s_memrealtime` ticks (20 µs at 100 MHz) before it hands its slice off. The stress test (§ Result) checked the
  output there, not the time.
- **P is scaled by 2^6, not 2^8, before its fp8 conversion**, because the running max may trail by up to 2.75 (lazy
  rescale, below) and 2^(6 + 2.75) is under e4m3's 448. A weight under 2^−12 of the running max loses precision, two
  bits earlier than the 8-head kernel. The timed inputs passed the 20 dB bound; inputs with a few dominant keys were not
  tried.
- **Uniform indices, one FlyDSL build.** As on the 8-head page: the timed inputs draw the top-k slots uniformly, the
  pools have max(65536, 2048 × tokens) rows, and the run had FlyDSL 0.3.2 pinned over the image's version; an install
  that takes aiter with its dependencies runs the FlyDSL that aiter pins, and that combination was not measured
  (`../pitfalls/measured_a_different_build.md`).
- **Workspace reuse is tested for the op alone**, not inside a model's decode graph (8-head page).

## Mechanism

What changed from the 8-head kernel:

- **No padding heads.** 16 heads fill the 16 columns of every MFMA; q is loaded with head h on lane h of every
  16-lane group.
- **Splits per token**: 16 at 1–8 tokens (one 32-key chunk per wave), 8 at 12–32, 2 at 40–128, 1 above, 4 waves per
  workgroup; 16 to 256 workgroups, one round.
- **Switches by split count.** Key validity comes from a ballot and the index registers hold row byte offsets at
  every split count (the 8-head kernel: at 2 and 1); the first chunk is peeled whenever a wave has more than two
  (there: at 2 splits only); split-fastest workgroup order and the last chunk scored in halves at 8 and 16 splits.
- **Wave merge in two ring regions.** Each wave writes its fp32 Oᵀ for heads 0–7 into its region of ring slot 0 and
  heads 8–15 into its region of slot 1 (8 heads × 516 floats fit one region); the 8-head kernel packed two tiles per
  store with a DPP move into one region.

  ```python
  HH = 8  # heads of a wave's O^T per ring region in the LDS wave merge
  HP = DV + 4  # f32 pitch of a head's row in the LDS wave merge: heads in distinct banks

  # f32 offset in the ring of head hh's row of wave w_: slot hh // 8, the wave's region, row hh % 8
  def hrow(hh, w_):
      return (hh // HH) * (SLOT // 4) + w_ * (WSLOT // 4) + (hh % HH) * HP

  obase = hrow(lr, wave) + 4 * lg  # lane lr of a 16-lane group holds head lr; lg is the lane group
  for n in range_constexpr(DV // 16):
      fx.ptr_store(acc[n], fx.add_offset(ring, obase + 16 * n))
  ```
- **Lazy rescale.** The running max moves only when a chunk's max exceeds it by more than 2.75 (in log2 units),
  and P is scaled by 2^6 (Counter-signals).

  ```python
  P_LOG2_SCALE = 6.0  # P is scaled by 2^P_LOG2_SCALE before its fp8 conversion
  # the running max moves only when a chunk beats it by more than RS_SLACK (log2 units): a skipped chunk's
  # scaled P stays at most 2^(P_LOG2_SCALE + RS_SLACK), under 448, the e4m3 maximum
  RS_SLACK = 2.75
  assert 2 ** (P_LOG2_SCALE + RS_SLACK) < 448

  # per chunk after the first: mb is the lane's chunk max, m the running max
  big = mb > m + fx.Float32(RS_SLACK)
  need = arith.cmpi(
      arith.CmpIPredicate.ne, rocdl.ballot(T.i64, big.ir_value()), arith.constant(T.i64, 0)
  )
  # rescale() moves m and multiplies O^T and l by 2^(m - new m); taken only when some lane needs it
  res = _if_yield(need, rescale, [a.ir_value() for a in acc] + [m.ir_value(), l.ir_value()])
  ...
  msafe = (m == fx.Float32(NEG_INF)).select(fx.Float32(0.0), m)
  pv = [_exp2(sv[j] - (msafe - fx.Float32(P_LOG2_SCALE))) for j in range(8)]  # at most 2^(6 + 2.75)
  ```
- **Split merge, per slice, with a bounded wait** (`../attention/split_kv.md` §2, Bounded wait). Each (token, wave) has
  a 64-bit arrival word on its own 128 B line: S bits of arrivals, S bits of abandoned slices, the launch tag above. A
  split publishes its normalized bf16 partial and its (max, sum) per head with `SC1` stores, waits `vmcnt(0)`, then ORs
  (tag | its bit) into the word (a word with another launch's tag is claimed with a compare-swap first). The split whose
  OR completes the mask merges its own slice of the 512 columns (512 / S), every abandoned slice, and stores zero to the
  word, so the next launch ORs into a zero word. Every other split polls the word with an atomic OR of 0, no sleep,
  until the mask is complete or its own bit is gone (the word was zeroed), then merges its own slice; after 2000 ticks
  it sets its abandon bit with a compare-swap instead, unless the word changed, and leaves. A word counts as this
  launch's only if it carries the tag, at least one arrival, and abandons only of splits that arrived. Each slice is
  weighed over the splits in index order, so the bits do not depend on who merged it.

  ```python
  def ours(lo_, hi_):
      """The word carries this launch's tag, at least one arrival, and abandons only of splits that
      arrived (at 16 splits the tag has only 32 bits; the masks lower a stale word's odds)."""
      arr_ = lo_ & fx.Int32(AMASK)
      ab_ = (lo_ >> S) & fx.Int32(AMASK)
      tag_ = ((lo_ & fx.Int32(LOMASK)) == tag_lo) & (hi_ == tag_hi)
      return tag_ & (arr_ != fx.Int32(0)) & ((ab_ & arr_) == ab_)

  def merge_own():
      ms_, ls_ = weights()  # every split's max and sum, loaded once all have published
      cb, parts = slice_loads(split)
      rocdl.sched_barrier(0)  # every load of the slice in one batch, before the first wait
      wsc = weigh(ms_, ls_)
      slice_out(cb, parts, wsc)
      return wsc
  ```

  ```python
  # the word this split's OR of (tag | its bit) returned: arrived is its arrival mask plus this split's
  # own bit, aband its mask of abandoned slices
  if arrived == full:  # this split completed the mask
      wsc = merge_own()
      for s_ in range_constexpr(S):  # and the slices of the splits that gave up waiting
          if ((aband >> s_) & fx.Int32(1)) != fx.Int32(0):
              cb, parts = slice_loads(fx.Int32(s_))
              slice_out(cb, parts, wsc)
      # every split has published: the word goes back to zero, which the next launch's splits OR
      # into without a claim (lane 0 alone stores); the waiters take a zero word as complete
      _bstore(Vec.from_elements([fx.Int32(0), fx.Int32(0)], Int32), fl_at, aoff, SC1)
  else:
      # wait for the mask to fill, bounded: on timeout hand this slice to the split that completes it
      t0 = llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])
      ...  # body of an scf.while that runs until done, or until abandon() wins its compare-swap:
          c_lo, c_hi = split64(atom_or(arith.constant(T.i64, 0)))  # the poll: an atomic OR of 0
          # this split's bit stays in the word until the split completing the mask stores zero
          done = ((c_lo & mybit) == fx.Int32(0)) | ((c_lo & fx.Int32(AMASK)) == full)
          now = llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])
          late = arith.cmpi(
              arith.CmpIPredicate.sgt, arith.subi(now, t0), arith.constant(T.i64, WAIT_TICKS)
          )
          ...  # not done and late: abandon() compare-swaps c | (mybit << S) into the word
      if fx.Int32(wop.results[0]) == fx.Int32(1):  # the loop ended on done
          merge_own()
  ```

  Not this: `weights()` issued before the wait. A waiting split then weighs its slice with
  the max and sum its partner's row held before the partner published; the two-split path did that in its first form and
  returned wrong outputs at every size it runs (Counter-signals).
- **Key sort** at 1 and 2 splits, as in the 8-head kernel, at every pool size: a form that skips it on pools of 16384
  slots or fewer (part of the second step below) applies to no size here and is not in the kernel.

What each step bought, each timed in pairs against the step before (1, 32 and 256 tokens at 16384, 65536 and 262144
slots; speedups) *(field, E1)*:
- the port to 16 heads: 1.44–1.83 over the starting baseline kernel;
- the lazy rescale, a peeled first chunk and no key sort on a pool of 16384 slots: 1.007–1.113, most at 256 tokens
  on that pool;
- the zeroed arrival word, with ballot validity and precomputed row offsets at 8 splits and precomputed offsets at
  1 and 2: 1.029–1.034 at 1 token. Alone, in a second port of the same kernel, the zeroed word was 1.017–1.021 at 1
  token and level at 32 and 256 (`../attention/split_kv.md` §3, its E51);
- the per-slice merge with the bounded wait: 1.086–1.089 at 1 and 32 tokens (then 8 splits), level at 256;
- 16 splits at 1 token: 1.098–1.110.

## Result

One run on an MI355X with FlyDSL 0.3.2, the kernel, the baseline and the Gluon kernel timed in the same rounds:
CUDA-graph replays with a 1 GiB read flush before each call, 16 rounds per size; a speedup is the one-sided lower bound
of the per-round time ratio, as on the 8-head page. "Gluon kernel" is the Triton (Gluon) kernel of the same op, at the
sizes where it measured at or above 1.03 over the baseline (§ Applies when); elsewhere the baseline is the yardstick
*(field, E2)*.

| tokens | splits | bound over baseline | bound over the Gluon kernel | sizes where the baseline is the yardstick |
|---|---|---|---|---|
| 1–8 | 16 | 1.376–1.690 | 1.025–1.076 | – |
| 12–32 | 8 | 1.558–1.768 | 1.100–1.304 | 12, 24 |
| 40–128 | 2 | 1.322–1.523 | 1.058–1.454 | 56 |
| 136–256 | 1 | 1.291–1.508 | 1.206–1.420 | 136–176 |

Every size passed the accuracy check, violation fraction 0. The margin over the Gluon kernel is smallest at 1–4 tokens
(1.025–1.043), where it comes closest; the largest are at 72–96 and 184–200 tokens (1.36–1.45). Each size also passed,
byte for byte against the call alone: a stress test (streams limited to 16 and 64 CUs, a second call on the same
inputs), one graph replayed 24 times with three input sets rotated into its buffers, and two calls at once on two
streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each *(field, E3)*.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=3 pairs, paired timing of CUDA-graph replays, each step against the step before; cases 1, 32, 256 tokens × 16384, 65536, 262144 slots] port: 1.468–1.480 / 1.760–1.827 / 1.439–1.732 at 1 / 32 / 256 tokens; lazy rescale, peeled first chunk, no key sort at 16384 slots: 1.008–1.009 / 1.007–1.008 / 1.017–1.113; zeroed arrival word with ballot validity and precomputed row offsets at 8 splits and precomputed offsets at 1 and 2 splits: 1.029–1.034 / 1.001–1.020 / 0.997–1.023; per-slice merge with bounded wait: 1.086–1.089 / 1.081–1.085 / 0.999–1.000; 16 splits at 1 token: 1.098–1.110 / 1.001 / 1.000–1.001
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, one-sided lower bound of the per-round time ratio over 16 rounds of CUDA-graph replays with a 1 GiB read flush before each call, the kernel, the baseline and the Gluon kernel timed in the same rounds, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] the 16-head kernel, bound over the baseline (sglang 0.5.22's `triton_sparse_mla_decode_splitk`) / over the Gluon kernel of the same op at n_tokens 1, 2, 4, 8, 16, 32: 1.376/1.043, 1.486/1.025, 1.580/1.032, 1.690/1.076, 1.585/1.100, 1.644/1.304; 12, 24: 1.558, 1.768 (baseline only: no Gluon kernel at 1.03 or more over it); 40, 48, 64: 1.357/1.266, 1.358/1.294, 1.373/1.058; 56: 1.362; 72–128: 1.523/1.454, 1.519/1.443, 1.494/1.411, 1.453/1.369, 1.389/1.317, 1.368/1.303, 1.343/1.288, 1.322/1.227; 136–176: 1.508, 1.503, 1.500, 1.490, 1.478, 1.486; 184–256: 1.466/1.420, 1.442/1.394, 1.411/1.361, 1.387/1.339, 1.359/1.311, 1.337/1.296, 1.315/1.272, 1.310/1.268, 1.310/1.262, 1.291/1.206; accuracy passed at all 36, worst violation fraction 0
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=3–24 (3 tries per stress condition, 24 graph replays, 20 tries per concurrent condition); each byte-compared with the call alone on the timing input sets] the 16-head kernel at all 36 sizes: the stress conditions (a stream limited to 16 CUs, one limited to 64 CUs, a second call on the same inputs) passed; one captured graph replayed 24 times with three input sets (outputs differ between sets) rotated into its buffers, 0 differ; two calls at once on two streams with different inputs, unrestricted and both on the same first 64 CUs, 20 tries each, 0 differ
