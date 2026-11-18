# How AKLR Times and Judges a Point

Sources: `aklr/core/validation/stages/bench/` (`_measure/_profile.py` for the timing and statistics,
`_plan/` for what a Point is timed against, `_verdict/` for the judging), `aklr/core/platform/env.py`.

## Accuracy

- Each Point is checked on `--num-trials` input sets (default 3), drawn by the schema's `init` with
  a seed from the schema name, the Point's workload key (today its axes) and the trial, rank 0's as a
  single rank's: the bench, the forge driver and the op tests draw the same input sets. Each trial calls the kernel once,
  eagerly, the way dispatch calls it, and compares its outputs with the Reference's by the schema's
  `check`. Only a Point that passes every trial is timed.

## Timing

- Every call is replayed in a CUDA graph and preceded by a 1 GiB cache flush inside the graph, so
  weights and activations come from HBM on every call. A timing is the kernel tracer's GPU time per
  call in a 20 ms graph; a tracer fault is timed again once.
- Each trial's input set is timed once its check passes: the candidate and its opponent once each,
  in a seeded shuffled order (`bench_cudagraph` warms up and repeats the call itself). A trial's
  log ratio is `log(t_opponent / t_candidate)`.
- What a Point is timed against:

  | Case | Opponent | Threshold | Speedup written back |
  |---|---|---|---|
  | the base serves this very kernel | Baseline | Retain | measured |
  | the base serves another kernel, with a stored speedup s | Incumbent | Entry | ratio × s |
  | the base serves another kernel, without one | Incumbent | Entry | none |
  | nothing served, no base or a first release | Baseline | Retain | measured |
  | nothing served otherwise | Baseline | Entry | measured |
  | neither a Baseline nor an Incumbent | nothing | - | none |

## Judging

- The gate is the mean log ratio over the trials, exponentiated: their geometric mean. No confidence
  bound: a ratio near a threshold can flip between runs. For the gate only, each trial's log ratio
  is clamped to ±ln 2: a report's `ratio` never shows more
  than 2.000. At either end, read the medians.
- Thresholds: Entry 1.05 (`DEFAULT_MIN_SPEEDUP`; `not_faster` at or below it), Retain 0.99
  (`regressed` at or below it), and dispatch's `AKLR_MIN_SPEEDUP`, 1.05 by default, the stored
  `speedup` a Point needs to be served.
- A schema is admitted when at least one Point is correct and its speedup over its opponent, to 3
  decimals, reaches Entry (>=, as dispatch compares a stored one), even where its gate was Retain;
  where nothing is timed, correctness is enough.
- The stored `speedup` is the geometric mean of the **unclamped** trials, to 3 decimals. A Point
  whose stored speedup is under 1.05 is never served.
- Without `--base`, nothing is served at the base: Points are held to Retain against the Baseline.
  Validate with `--base origin/main` before trusting a pass.

## What moves a result

- **Per-call weight caches miss.** A kernel that caches a prepared weight keyed on one tensor misses
  on every timed call, since each set brings its own weight, as a model's layers do. Measured on CI:
  a FlyDSL decode GEMM that re-packs `b` on a miss took 17.45 µs against the baseline's 7.12 µs at m = 4, and
  its stored 1.109 did not hold.
- **The graph-node floor.** With the flush, a small kernel's node costs about 4 µs whatever its work:
  add + RMSNorm at hidden 6144 took 3.9 – 4.5 µs from 1 to 256 tokens, for both kernels. A gain of
  0.1 – 0.25 µs there is a fixed latency difference: 1.04 – 1.05 on one node, 1.023 – 1.025 on
  another for the same sources.
- **Spike timings.** Under the former 32-round protocol, two candidate rounds at +35 % and +21 % over
  the median took a Point with a median ratio of 1.024 to a lower bound of 0.995. They come from the
  node: a CI run validating 88 schemas on 8 GPUs at once had spikes on both sides. With one timing
  per trial, a spike moves the geometric mean directly; the stored speedup includes it unclamped.
- **Node and load.** The same kernel sources (add + RMSNorm) validated on a development node and on
  CI's, which ran 88 schemas on 8 GPUs at once, moved per Point by −2.5 % to +2.3 %; the same node
  twice moved them by under 1 %. A margin of a few percent is node-dependent.
- **Sweeps are not the gate.** A config sweep's single-input-set median predicted the former
  protocol's lower bound within −6 % to +2 % in these runs. Settle choices worth 1 – 2 % with validate.
- **Hot caches overstate.** A benchmark that replays without flushing can keep a decode GEMM's weight
  in MI355X's 256 MB MALL; AKLR reads it from HBM every call (a 25 MB weight took ~6 µs, about
  4.2 TB/s). Re-tune a configuration found hot.
