# How AKLR measures a kernel

What AKLR's timer reads, who measures, and what the stored number does not say. Sources: `MEASUREMENT_PROFILE` and
`Clock` in `aklr/core/validation/stages/_measure.py`, `bench_cudagraph` in `aklr/core/utils/testing.py`,
`aklr/plugins/aiter/README.md` § Inputs and workloads. Thresholds, the gate and what moves a result:
`harness/skills/read-validation/references/ruler.md`. Each general lesson stays in the book, linked from the row
that needs it.

## The ruler

| Part | What AKLR does |
|---|---|
| A reading | The call is captured several times in one CUDA graph, as many as fill 20 ms, each behind a 1 GiB read that empties L2 and MALL. The graph is replayed 10 times. A call's time is the sum of the device times of its kernels, read by AKLR's own rocprofiler-sdk tracer and averaged over the replays. The flush, host time and the gaps between kernels are left out. The reading is the median over the captured calls |
| A round | One reading of the kernel, one of the baseline, and one of the incumbent if there is one, in a seeded random order. One warm-up round, then 16 rounds, then 16 more unless every threshold is already decided |
| `speedup` | The one-sided Student-t lower bound of the mean of log(baseline time / kernel time) over the rounds, to 3 decimals. A bound, not a mean: more spread between rounds lowers it |
| `microseconds` | The kernel's median reading in those rounds |
| The inputs | One input set per Point, drawn on the GPU from the contract's `distribution`s and seeded by the workload: the same tensors in every round, every scalar at its default |
| The baseline | The contract's baseline class, called as the engine calls it, on its own copy of the inputs, timed by the same clock |

`MEASUREMENT_PROFILE` is copied into every `report.json`. Compare numbers only between reports whose
`measurement_profile.version` is the same.

## Who measures

- CI's GPU jobs run on a PR labelled `ci:gpu` and on every push to main. `aiter-test` times `aklr/plugins/aiter`
  against the baseline aiter of `stack.json`. `sglang-test` and `vllm-test` time their plugins' ops against the
  image's own code, and build and check `aklr/plugins/aiter` there without timing
  (`harness/skills/read-validation/references/ci.md`).
- A PR is judged on the merge of its head into its base. The numbers a kernel is served on are measured on main
  after the merge and written into the file by a bot PR. CI rewrites a stored value only when it moves by more
  than 2% (README § Inputs and workloads).
- A local validate runs the same ruler and writes its numbers under `--out`. It decides nothing. Use it to find
  the Points CI will demote, and to A/B ([validate.md](validate.md)).

## The floor

AKLR's timer is a per-kernel profiled timer. On it every kernel of a call reads at least about 4.4 µs on MI355X,
whatever it does. The measurements, taken on this tracer, are in
`../kernel-optimization-book/pitfalls/profiled_timer_floor.md`. In AKLR this means:

- **A Point whose baseline reads 4–5 µs may be at the floor.** A kernel that is faster back to back can read
  parity there and be demoted as `not_faster`. A stored speedup near 1.0 at such a Point is not evidence that the
  Point has no headroom, and a small stored gain is a fixed latency difference that does not carry across nodes.
- **A call of n kernels reads at least n floors.** On this ruler, folding two launches into one is worth one floor,
  and where both sides sit on the floor the ratio is the ratio of their launch counts. A launch that zeroes
  counters, a dtype conversion in `run` and a separate reduce kernel each cost one.
- **A deployment's graph pays less per extra kernel than the floor**
  (`../kernel-optimization-book/triton/authoring.md` §7). A design chosen only to save floors gains less end to end
  than its stored speedup says.
- **Check it at your Point, before a campaign.** Look for baselines at 4–5 µs in a validate report, or in
  `$RUN/baseline.json` of a forge_task run. Time an empty kernel (a one-element `zero_`), the baseline and the
  candidate with AKLR's clock ([validate.md](validate.md) § Reproduce one step outside validate). A side that
  reads about what the empty kernel reads is at the floor.

## What the number leaves out

| AKLR's bench | So | The general lesson |
|---|---|---|
| Host time is not in a reading | A cheaper launch path gains nothing here. An eager caller, such as an uncaptured prefill batch, pays it | `../kernel-optimization-book/pitfalls/eager_call_host_time.md` |
| The kernel has the GPU alone | A wait on another workgroup always completes. Run the stress check | [validate.md](validate.md) § Stress |
| Every call starts with cold caches | A configuration tuned on a hot cache can lose. Tune under the flush | `../kernel-optimization-book/gemm/small_m.md` §1 |
| The timed calls reuse one input set, and a mutable input (a KV cache, a recurrent state) is replayed in place | A kernel that writes only what changed reads faster than it will in serving. Quote its time with the state refreshed before each call | `../kernel-optimization-book/pitfalls/state_kept_across_calls.md` |
| Inputs come from the contract's `distribution`s, not from a model | Lengths, indices and routing can all take the fast path, have no locality, or never leave a tile empty. Read each input's `distribution` in the definition file and ask whether serving looks like it | `../kernel-optimization-book/pitfalls/degenerate_points.md` |
| A var axis that is not `any_size` is served only at the listed values | A speedup stored at a pool or table size the model never sends serves nothing | the same page |
| Timing runs at each scalar's default. The workload's own scalars run in one check step | A scalar that changes the work, such as an iteration count, is timed at its default. A value elsewhere in its range is never run: test it yourself | — |
| The baseline's output is compared with nothing. Only a `NotImplementedError` is reported, as `baseline_failed` | A baseline that computes wrong, or less, becomes the denominator. At a Point with a surprising ratio, check the baseline's output against the reference | `../kernel-optimization-book/pitfalls/baseline_silently_wrong.md` |
| The baseline is the installed build | `aklr/plugins/aiter` is timed against aiter at the commit in `stack.json`, an op of the sglang or vLLM plugin against the engine image's own code and aiter. A ratio taken on another build has another denominator. The report does not record aiter | `../kernel-optimization-book/pitfalls/measured_a_different_build.md` |
| Bench runs the `content` of the definition file | Editing a `kernel.py` beside it changes nothing. Put the file into the definition before you validate | the same page |
| The speedup of another harness is not this one | A KernelForge score, an eager probe and a back-to-back replay each differ from the stored bound | `../kernel-optimization-book/pitfalls/speedup_from_another_ruler.md` |

## How far a number moves by itself

- The same sources at the same Point differ by a few percent between runs, more near the floor. The measured
  spreads are in `../kernel-optimization-book/triton/tuning.md` §1 and in ruler.md § What moves a result.
- Leave a margin over `AKLR_MIN_SPEEDUP`. A Point stored near 1.05 can go in and out of service from one CI run
  to another.
- Compare speedups at one Point only, and report both times. Across Points a ratio mixes the baseline's change
  with the kernel's.
- A gap of a few percent between two files' stored values, measured in different CI runs, says nothing about the
  two kernels. Time them in the same rounds ([validate.md](validate.md) § A/B one change).

## Tracer faults

A `platform_failure` whose detail names `rocprofiler-sdk` or `graph replays`, or says `the kernel tracer failed`
or `calibration could not restore`, is the tracer losing records (`is_tracer_fault` in `_measure.py`), not the
kernel. AKLR has already read again once in place and rerun the task once in a fresh process. Run validate again,
on an idle machine. Do not keep or revert a change on it. A harness of your own gets the in-place retry from
`Clock.graph_reading`; `bench_cudagraph` called directly raises. The general lesson:
`../kernel-optimization-book/pitfalls/profiler_misses_kernels.md`.
