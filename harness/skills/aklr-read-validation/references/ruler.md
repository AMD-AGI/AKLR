# How AKLR Times and Judges a Point

Sources: `aklr/core/validation/stages/_measure.py`, `stages/_bench/_stage.py`,
`stages/_bench/_report.py`, `aklr/core/platform/env.py`.

## Timing

- Every call is replayed in a CUDA graph and preceded by a 1 GiB cache flush inside the graph, so
  weights and activations come from HBM on every call.
- Timed calls rotate through 8 input sets (`N_TIMING_SETS`), each with its own tensors, weights
  included, as a model's layers bring their own.
- One warm-up round, then 32 rounds (`N_ROUNDS`). Each round times the candidate and the versus
  side (the baseline, or a served kernel); its log ratio is `log(t_versus / t_candidate)`.

## Judging

- The gate is the one-sided Student-t lower bound (α = 0.05) of the mean log ratio, exponentiated.
  For the gate only, each round's log ratio is clamped to ±ln 2 (`MAX_ABS_LOG_RATIO`): a report's
  ratio column never shows more than 2.000, and a bound near 0.5 can fall just under it. At either
  end, read the medians.
- Thresholds (`THRESHOLD_RATIOS`, `AKLR_MIN_SPEEDUP`):

  | | ratio | applies to |
  |---|---|---|
  | entry | 1.02 | a new or changed kernel, vs the baseline |
  | retain | 0.99 | a kept kernel vs the baseline, and any kernel vs a served one |
  | dispatch | 1.03 (default) | the stored `speedup` a Point needs to be served |

- The stored `speedup` (written back by CI) is the graph lower bound vs the baseline computed from
  the **unclamped** rounds, to 3 decimals. A Point whose bound is in [1.02, 1.03) passes entry and is
  never served.
- Without `--base`, a named kernel counts as kept: it is held to 0.99, not 1.02, and is not timed
  against a served kernel. Validate with `--base origin/main` before trusting a pass.

## What moves a result

- **Per-call weight caches miss.** A kernel that caches a prepared weight keyed on one tensor misses
  on every timed call, since each set brings its own weight, as a model's layers do. Measured on CI:
  a FlyDSL decode GEMM that re-packs `b` on a miss took 17.45 µs against the baseline's 7.12 µs at m = 4, and
  its stored 1.109 did not hold.
- **The graph-node floor.** With the flush, a small kernel's node costs about 4 µs whatever its work:
  add + RMSNorm at hidden 6144 took 3.9 – 4.5 µs from 1 to 256 tokens, for both kernels. A gain of
  0.1 – 0.25 µs there is a fixed latency difference: 1.04 – 1.05 on one node, 1.023 – 1.025 on
  another for the same sources.
- **Spike rounds.** Two candidate rounds of 32 at +35 % and +21 % over the median took a Point with a
  median ratio of 1.024 to a lower bound of 0.995. They come from the node: a CI run validating 88
  schemas on 8 GPUs at once had spike rounds on both sides. The stored speedup includes them
  unclamped.
- **Node and load.** The same kernel sources (add + RMSNorm) validated on a development node and on
  CI's, which ran 88 schemas on 8 GPUs at once, moved per Point by −2.5 % to +2.3 %; the same node
  twice moved them by under 1 %. A margin of a few percent is node-dependent.
- **Sweeps are not the gate.** A config sweep's single-input-set median predicted validate's lower
  bound within −6 % to +2 % in these runs. Settle choices worth 1 – 2 % with validate.
- **Hot caches overstate.** A benchmark that replays without flushing can keep a decode GEMM's weight
  in MI355X's 256 MB MALL; AKLR reads it from HBM every call (a 25 MB weight took ~6 µs, about
  4.2 TB/s). Re-tune a configuration found hot.
