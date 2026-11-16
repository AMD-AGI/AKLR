---
title: "Sparse MLA of sglang's DSA backend with one query token per sequence, 16 heads (GLM-5.2 at TP4), fp8 q and 576-byte KV rows, top-2048, at the token counts sglang pads TP4 batches to — the 8-head kernel at 16 heads; 16, 8, 2 or 1 splits per token, each split merging its own slice of the output after a bounded wait"
kind: case
ops: [mla, attention]
dtypes: [fp8, bf16]
regimes: ["tokens 1-8", "tokens 12-32", "tokens 40-128", "tokens 136-256"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1, AKLR validate measurement profile 19
measured:
  - {metric: "speedup over the contract's baseline (sglang's triton_sparse_mla_decode_splitk), AKLR validate published lower bound per Point, local run, not CI", value: "1.291–1.768 at all 36 Points", gpu: "MI355X", workload: "dsa_sparse_mla_decode_n_heads16_kv_dim576_topk2048, the 36 Points this file lists", run_ref: "this file's AKLR validate, local, not CI", date: "2026-11-10"}
  - {metric: "speedup over the file main serves at the Point (PR #66's Triton files), same run, published lower bound per Point", value: "1.025–1.454 at the 27 Points where it serves one", gpu: "MI355X", workload: "the Points where main serves one", run_ref: "this file's AKLR validate, local, not CI", date: "2026-11-10"}
sources:
  - aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads16_kv_dim576_topk2048/a8d38f7436dc64c0.json
---

# Sparse MLA decode for sglang's DSA backend, 16 heads: the 8-head kernel, with a per-slice merge

One FlyDSL file, `a8d38f7436dc64c0`, lists 36 of the 36 decode sizes of `dsa_sparse_mla_decode@1.0` that sglang
pads TP4 batches to, the decode attention of GLM-5.2 at 16 heads per rank. It is
[dsa_sparse_mla_8_heads.md](dsa_sparse_mla_8_heads.md)'s kernel ported to 16 heads by a forge-loop campaign and
reviewed; everything that page says of the ring, the key sort and the fp8 MFMAs holds here unless this page says
otherwise. Its workloads carry no `speedup` yet, so dispatch serves them only once CI has written one that
qualifies (`aklr/ops/README.md` § Environment variables). It embeds its source as `kernel.sources[0].content`
(`kernel.py`); function names below are in it.

## Applies when

- Contract `dsa_sparse_mla_decode@1.0` (op type `dsa_backend`), layout `row_major`, as on the 8-head page, with
  16 heads: q (tokens, 16, 576) and the pool (n_slots, 576) float8_e4m3fn, unscaled; top-k (tokens, 2048) int32;
  output (tokens, 16, 512) bf16. `compare`: bitwise determinism, violation fraction 0, SQNR ≥ 20 dB.
- Points: (n_tokens, n_slots) with n_slots = max(65536, 2048 · n_tokens), `sm_scale` 0.0625: all 36, n_tokens 1, 2, 4, 8, 12, 16, 24, 32 and 40–256 in steps of 8.
  Dispatch matches `n_tokens` exactly; `n_slots` is `any_size`.
- Yardstick at each Point: the faster of the baseline, `dsa_sparse_mla_decode_baseline` (sglang 0.5.22's
  `triton_sparse_mla_decode_splitk`), and the file main serves there: PR #66's Triton (Gluon) files, served at the
  Points where CI measured them at or above `AKLR_MIN_SPEEDUP` (`aklr/ops/README.md` § Environment variables): 27 of the 36, all but 12, 24, 56 and 136–176 tokens, as of
  #102.
- Not covered: the draft steps of speculative decoding and a q that is not one contiguous (tokens, 16, 576) view
  (the api sends them back to sglang), another top-k or head count.

## Counter-signals

- **A split count the campaign's scored sizes do not reach is untested by it.** The campaign scored 1, 32 and 256
  tokens, so its 2-split path never ran, and that path's first form returned wrong outputs at every 2-split Point
  (40–128 tokens): a waiting split loaded its partner's max and sum before the wait. The load now runs after it
  (S2). Score, or at least `--stress`, every split count a file serves (`../attention/split_kv.md` §2, its E50).
- **The tag shrinks with the split count.** The arrival word keeps 2S low bits for the two masks, so the tag has
  64 − 2S bits: 32 at 16 splits (1–8 tokens). The flag block is a `torch.empty` per call; if it holds arbitrary
  bytes, a word passes as this launch's with odds of about 2^−32 for the tag times about 1% for the mask checks
  (estimated), and up to 32 words are read per call. A word that passes makes splits merge before their partners
  have published, with no error. A word this kernel leaves is zero, so the 32-bit tag matters only for bytes another
  kernel wrote into the block; `../attention/split_kv.md` §3 asks for a wider tag. The 8-head page lists the tag's
  other risks (two queues sharing a tag).
- **The wait costs time when a partner is not running.** On a stream limited to part of the CUs a waiting split
  spins up to 2000 `s_memrealtime` ticks (20 µs at 100 MHz) before it hands its slice off. `--stress` checked the
  output there, not the time.
- **P is scaled by 2^6, not 2^8, before its fp8 conversion**, because the running max may trail by up to 2.75
  (lazy rescale, below) and 2^(6 + 2.75) is under e4m3's 448. A weight under 2^−12 of the running max loses
  precision, two bits earlier than the 8-head file. Validate's inputs passed the 20 dB bound; inputs with a few
  dominant keys were not tried.
- **Uniform indices, local timing.** As on the 8-head page: bench draws the top-k slots uniformly, times pools of
  max(65536, 2048 × tokens) rows, and ran FlyDSL 0.3.2 pinned over the image's version, not what CI runs
  (`../pitfalls/measured_a_different_build.md`).
- **Workspace reuse is tested for the op alone**, not inside a model's decode graph (8-head page).

## Mechanism

What changed from the 8-head kernel:

- **No padding heads.** 16 heads fill the 16 columns of every MFMA; q is loaded with head h on lane h of every
  16-lane group.
- **Splits per token** (`_config`, S1): 16 at 1–8 tokens (one 32-key chunk per wave), 8 at 12–32, 2 at 40–128, 1
  above, 4 waves per workgroup; 16 to 256 workgroups, one round.
- **Switches by split count.** Key validity comes from a ballot and the index registers hold row byte offsets at
  every split count (the 8-head kernel: at 2 and 1); the first chunk is peeled whenever a wave has more than two
  (there: at 2 splits only); split-fastest workgroup order and the last chunk scored in halves at 8 and 16 splits.
- **Wave merge in two ring regions.** Each wave writes its fp32 Oᵀ for heads 0–7 into its region of ring slot 0 and
  heads 8–15 into its region of slot 1 (8 heads × 516 floats fit one region); the 8-head kernel packed two tiles per
  store with a DPP move into one region.
- **Lazy rescale.** The running max moves only when a chunk's max exceeds it by more than 2.75 (in log2 units),
  and P is scaled by 2^6 (Counter-signals).
- **Split merge, per slice, with a bounded wait** (`../attention/split_kv.md` §2, Bounded wait; S2). Each (token, wave)
  has a 64-bit arrival word on its own 128 B line: S bits of arrivals, S bits of abandoned slices, the launch tag
  above. A split publishes its normalized bf16 partial and its (max, sum) per head with `SC1` stores, waits
  `vmcnt(0)`, then ORs (tag | its bit) into the word (a word with another launch's tag is claimed with a
  compare-swap first). The split whose OR completes the mask merges its own slice of the 512 columns (512 / S),
  every abandoned slice, and stores zero to the word, so the next launch ORs into a zero word. Every other split
  polls the word with an atomic OR of 0, no sleep, until the mask is complete or its own bit is gone (the word was
  zeroed), then merges its own slice; after 2000 ticks it sets its abandon bit with a compare-swap instead, unless
  the word changed, and leaves. A word counts as this launch's only if it carries the tag, at least one arrival,
  and abandons only of splits that arrived. Each slice is weighed over the splits in index order, so the bits do
  not depend on who merged it.
- **Key sort** at 1 and 2 splits, as in the 8-head kernel; the campaign's form skipped it on pools of 16384
  slots or fewer, which no listed Point has, and the review removed that variant.

What each kept round bought, in the campaign (paired keep gate, 1, 32 and 256 tokens at 16384, 65536 and 262144
slots, each round against the one before; speedups) *(field, E1)*:
- the port to 16 heads: 1.44–1.83 over the starting baseline kernel;
- the lazy rescale, a peeled first chunk and no key sort on a pool of 16384 slots: 1.007–1.113, most at 256 tokens
  on that pool;
- the zeroed arrival word, with ballot validity and precomputed row offsets at 8 splits and precomputed offsets at
  1 and 2: 1.029–1.034 at 1 token. Alone, in a second port of the same kernel, the zeroed word was 1.017–1.021 at 1
  token and level at 32 and 256 (`../attention/split_kv.md` §3, its E51);
- the per-slice merge with the bounded wait: 1.086–1.089 at 1 and 32 tokens (then 8 splits), level at 256;
- 16 splits at 1 token: 1.098–1.110.

## Result

AKLR validate, measurement profile 19, on an MI355X with FlyDSL 0.3.2 (local, not CI), with `--base` at main
d7c0a3e plus #94 (since merged), which serves the same files at these Points:
graph replays with a 1 GiB read flush before each call, 16 rounds per Point; a speedup is the published one-sided
lower bound of the per-round time ratio, as on the 8-head page. "Served file" is the Triton file main serves at the
Point (#66's), where CI measured one at or above `AKLR_MIN_SPEEDUP`; elsewhere the baseline is the yardstick
*(field, E2)*.

| tokens | splits | bound over baseline | bound over the served file | Points with no served file |
|---|---|---|---|---|
| 1–8 | 16 | 1.376–1.690 | 1.025–1.076 | – |
| 12–32 | 8 | 1.558–1.768 | 1.100–1.304 | 12, 24 |
| 40–128 | 2 | 1.322–1.523 | 1.058–1.454 | 56 |
| 136–256 | 1 | 1.291–1.508 | 1.206–1.420 | 136–176 |

Every Point passed accuracy, violation fraction 0. The margin over the served file is smallest at 1–4 tokens
(1.025–1.043), where #66's files come closest; the largest are at 72–96 and 184–200 tokens (1.36–1.45). Each Point
also passed, byte for byte against the call alone: the forge driver's `--stress` (streams limited to 16 and 64 CUs,
a second call on the same inputs), one graph replayed 24 times with three input sets rotated into its buffers, and
two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each
*(field, E3)*.

## Reproduce

On an MI355X, in a checkout where this file is not on the base branch, with FlyDSL 0.3.2:

```
python -m aklr validate aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads16_kv_dim576_topk2048/a8d38f7436dc64c0.json --base origin/main
```

It times the kernel against the baseline and against the file main serves at each Point and writes
`aklr_validation/report.md` (`aklr/ops/README.md` § Commands). The local result does not count; CI's measurement
is the one dispatch reads. The stress check is the forge driver's `--stress` (`docs/forge_task.md`). To read the
kernel: `python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["kernel"]["sources"][0]["content"])' <schema>`.

## Rollback

Delete the file in a PR: its Points fall back to the files main serves there (PR #66's) or to sglang's kernel.
`AKLR_DISABLE=dsa_sparse_mla_decode@1.0` sends every head count of the contract back to sglang. Roll back for a
wrong output CI does not catch: a workspace written by other kernels inside a model's graph, a flag word shared
between queues, inputs with a few dominant keys (Counter-signals), or a FlyDSL or LLVM upgrade.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` the schema embeds.

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=3 pairs, forge-loop paired keep gate on AKLR's graph timer, each kept round against the round before; cases 1, 32, 256 tokens × 16384, 65536, 262144 slots] port: 1.468–1.480 / 1.760–1.827 / 1.439–1.732 at 1 / 32 / 256 tokens; lazy rescale, peeled first chunk, no key sort at 16384 slots: 1.008–1.009 / 1.007–1.008 / 1.017–1.113; zeroed arrival word with ballot validity and precomputed row offsets at 8 splits and precomputed offsets at 1 and 2 splits: 1.029–1.034 / 1.001–1.020 / 0.997–1.023; per-slice merge with bounded wait: 1.086–1.089 / 1.081–1.085 / 0.999–1.000; 16 splits at 1 token: 1.098–1.110 / 1.001 / 1.000–1.001
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate measurement profile 19, `--base` main d7c0a3e with #94 on top, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] a8d38f7436dc64c0, bound over the baseline / over the served file at n_tokens 1, 2, 4, 8, 16, 32: 1.376/1.043, 1.486/1.025, 1.580/1.032, 1.690/1.076, 1.585/1.100, 1.644/1.304; 12, 24: 1.558, 1.768 (no served file); 40, 48, 64: 1.357/1.266, 1.358/1.294, 1.373/1.058; 56: 1.362; 72–128: 1.523/1.454, 1.519/1.443, 1.494/1.411, 1.453/1.369, 1.389/1.317, 1.368/1.303, 1.343/1.288, 1.322/1.227; 136–176: 1.508, 1.503, 1.500, 1.490, 1.478, 1.486; 184–256: 1.466/1.420, 1.442/1.394, 1.411/1.361, 1.387/1.339, 1.359/1.311, 1.337/1.296, 1.315/1.272, 1.310/1.268, 1.310/1.262, 1.291/1.206; accuracy passed at all 36, worst violation fraction 0
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=3–24 (the driver's 3 tries per `--stress` condition, 24 graph replays, 20 tries per concurrent condition); each byte-compared with the call alone on AKLR's input sets] a8d38f7436dc64c0 at all 36 Points: the forge driver's `--stress` passed; one captured graph replayed 24 times with three input sets (outputs differ between sets) rotated into its buffers, 0 differ; two calls at once on two streams with different inputs, unrestricted and both on the same first 64 CUs, 20 tries each, 0 differ
- S1 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads16_kv_dim576_topk2048/a8d38f7436dc64c0.json#kernel.py:45 `_config`; `_build(n_tokens)` at :114; P scale and `RS_SLACK` at :29-33, the lazy rescale at :403, P at :432; the wave merge's two ring regions (`hrow`) at :579; pad lanes sent past the pool at :250
- S2 aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads16_kv_dim576_topk2048/a8d38f7436dc64c0.json#kernel.py:645 `dispatch.id` and `queue.ptr` (:646) into the tag; `ours` at :686; the OR at :715 and the claim's compare-swap at :722; `weights` at :738, after the wait; `merge_own` at :805; the word stored back to zero at :821; the bounded wait from :825, its end on this split's bit at :835, the abandon compare-swap at :847; `torch.empty` partials and words at :890-898
