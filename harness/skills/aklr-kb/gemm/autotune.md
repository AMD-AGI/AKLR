---
title: "GEMM — per-shape autotune that cannot regress, and a launch path that costs nothing"
kind: method
lever: gemm_autotune
ops: [gemm]
gens: [gfx950]
dtypes: [fp8, mxfp4, mxfp8, bf16]
regimes: [prefill, training, cuda-graph]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/gemm_helper.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py
---

# Per-shape autotune and the launch path

> No single GEMM configuration is best across shapes. Two loop designs, tile shapes, L2 bands, store
> policy and split-K all change sign between shapes (`mainloop.md` §1,
> `epilogue.md` §5). Turbo's dense GEMMs therefore race a short candidate list on the first call
> for each shape and cache the winner.
> "FP8", "FP4", "MX8" and "GH" are the files under
> `primus_turbo/flydsl/{gemm,utils}` at Primus-Turbo main `7615a5a2`. Numbers marked `*(field, E<n>)*` are
> measured values; their conditions are listed under Evidence at the end. Use them for direction and order of
> magnitude, not as targets.

In FlyDSL: `../flydsl/gemm.md` § autotune Per-shape autotune and the launch path.

## Route here when
- You are adding a configuration axis, or a second kernel design, behind one entry point.
- A benchmark gives a bimodal result for one shape across runs.
- Host launch time shows up next to a short kernel, or a kernel behaves differently under CUDA-graph capture.

## 1. The first-call race (FP8 `_dense_race`, :3040)

For each candidate `(cfg, factory)`:
1. Build it. Run it once on scratch output. Drop it if it raises, or if the first 1024 outputs are not
   finite. A geometry that does not hold for the shape simply leaves the race.
2. Run 200 throwaway launches of the first candidate in the list (the one timed first), so the clock is up before anything is timed
   (`_PICK_RAMP_ITERS`, FP8:80).
3. Time every candidate over forward and reversed passes, three round trips, and keep each candidate's
   minimum (`_pick_dense_candidate`, FP8:3067). This fixed forward-plus-reversed order keeps the first
   candidate at both ends of every pass and the last candidate at the turn, twice in a row. With three or
   more candidates whose times sit within the slot effect, shuffle the order on each pass instead
   (`measurement.md` §2).
4. Cache the winner per static shape key. Call `torch.cuda.empty_cache()` to release the race buffers.

**Keep the race; do not replace it with a model.** *(field, E1)* Replacing the TN race with "sort by an
analytic model, take the first" left the scored shapes unaffected, because the model ranked them right.
Off the scored set, six wgrad cells regressed, three of them by 13–14%. Restoring the race recovered every cell. A static
analysis is a cheap way to find suspect shapes; only a measurement decides. Any change that replaces a race
must be validated on shapes outside the scoring set.

**Keep the candidate list short.** Every candidate is compiled on the first call: "a race compiles every
entry, so the list is what first-call latency costs" (FP8:3188-3191). Turbo's NT 8-wave tables are four
wide, or two wide when whole-loop geometries join the race (`_NT_CANDIDATES`, FP8:3193-3198). Add a
candidate only after it has been shown to win somewhere. *(field, E16)* On a grouped GEMM, a 4-wave
persistent candidate that cleared the adoption margin on none of the deployment shapes only added compile
time.

## 2. Never regress: the incumbent first, challengers win by a margin

- List the heuristic or incumbent configuration first. A challenger must beat it by a margin larger than
  the race's own noise. Turbo's margins:
  - whole-loop depth: 2% (`_WL_MARGIN = 1.02`, FP4:3274);
  - speculative swizzles: 0.5% (`_SWZ_MARGIN = 1.005`, FP4:3278). *(field, E15)* With the race's fixed
    candidate order and minimum over 8 reps (FP4:3261-3273), a replay on one MXFP4 shape gave the
    first-listed candidate a 2.3–2.4% discount, so there this margin acted as an estimated ~2.9%;
  - split-K arms: 1% (`_race_mxfp4_ksplit(margin=0.99)`, FP4:3557);
  - MXFP8 C-store policy and split: 1% (MX8:973-976).
  These are the constants Turbo ships; its source records no measurement behind them. Derive your own
  margin from your race's spread.
- **Pure tile-order changes are safe to add.** A bijection on tile ids cannot change the result, so a
  swizzle candidate needs no correctness re-check (FP4:3186-3187).
- **Every extra candidate is another draw from the race's noise.** Add the margin before widening the
  list. *(field, E2)* One candidate added without a margin cost 0.19% of a 12-shape score in one read, and
  one shape (FC2_wgrad) fell from 1.005 to 0.993 of the reference's throughput.
- **The first list entry is a decision.** With a margin, a challenger that wins by less never lands.
  - *(field, E3)* On a grouped fp8 dgrad, where no `num_xcd` was best everywhere, starting the list at one end of the
    axis let the margin chain step along it. Starting in the middle stranded it, because neither neighbour could clear
    the margin. Start at the end on the side of the measured optimum; one arm that moved the dgrad and the forward
    starting points one step further together read lower, and it moved both lists, so it does not tell which move
    cost.
- **Race noise above the margin makes the winner flip between runs.** *(field, E4)* On a grouped MXFP8 GEMM, one
  shape alternated between two configurations that tied at the race's synthetic balanced point and were about
  3% apart on the skewed distribution the bench scored. Clearing the cache and re-running the same shape
  showed a 2:1 split. The cause was the measurement order, not thermal noise.
  - Time base and candidate in one process, inside one window, as a palindrome `(base, cand, cand, base)` per
    rep, and take the median of `(c1+c2)/(b1+b2)` (`_robust_ab_ratio`, GH:2091). The DVFS ramp is odd in the
    slot index, so it cancels within a rep. This is the two-arm in-process case of
    `measurement.md` §2.
  - *(field, E5)* On a grouped fp8 GEMM, with the palindrome, 48 of 48 re-runs (eight cells × six) picked the
    same configuration, in each cell the arm measured fastest.
  - Check stability itself repeatedly. One stable check is not evidence; use at least 6 cleared-cache
    re-runs and report every adopted configuration.
- **Score at the deployment M.** *(field, E6)* On a grouped fp8 GEMM, an arm that was +2.5–3.0% at the deployed M
  scored −0.94% and +0.54% at the race's two canonical M points, so the race never picked it. When one shape
  family is known, give it a static lead: reorder its candidate list under a predicate that matches only that
  family. Keep the race running, so the lead can still be beaten. Verify the lead is non-negative on every M the
  predicate covers.
- **Hard-coding beats a race that scores the wrong point.** *(field, E7)* With a small candidate list and shape
  families that can be measured offline once, fixing the choice per family also drops the first-call race. On a
  grouped MXFP4 GEMM whose race scored candidates on synthetic balanced group offsets, the candidate it did not
  pick was 7.0% and 7.8% faster on the two cells with the lowest ratio; hard-coding it raised the lowest of the
  18 per-cell ratios (MXFP8 time over MXFP4 time) from 1.1843 to 1.2271. Do not let two dispatch paths share
  one default constant.

## 3. Staged searches

Search one axis family at a time, holding the earlier winners, instead of a full cross product:
- MXFP8 (`_autotune_mxfp8`, MX8:865-981): `(BLOCK_M, GROUP_M, num_xcd)` at the seed band width; then the 2D
  N-band width, adopted only above 1.5%; then the C-store cache policy; then the split plan.
- MXFP4 (FP4:3170-3377 and the dispatch around it): swizzle × whole-loop depth; then the epilogue
  variants; then the launch mode (plain, uniform split, tail split over M or N). Launch modes are timed end
  to end, including the host reduce.

When a search stalls, change the family of the dimension you search (split-K, MFMA shape, cache modifier,
`waves_per_eu`, scale prefetch). Do not re-sweep the same tile grid.

In FlyDSL: `../flydsl/gemm.md` § autotune §3.

## 4. Representative probe data

- **Time on production-magnitude data.** fp8 wall time depends on operand magnitude through power and
  clock (Turbo's grouped fp8 race, `_autotune_np_dispatch`, `grouped_gemm/grouped_gemm_fp8_kernel.py:2558`), so
  race on the caller's operands, as Turbo's dense race does (`_bench_dense_args`, FP8:3008-3013). Zero operands
  are a false ruler. *(field, E8)* On an MXFP4 GEMM they read 155 µs against 205 µs for the same build and
  shape, while operand entropy moved a microbenchmark's peak by only 10.7%; the rest of the gap is put,
  untested, on allocation history. Time on tensors the deployment path produced.
- **Choose the beta the race ranks at.** Turbo's dense races time the beta=0 build on a scratch output and rebuild the
  winner with the accumulate epilogue (`epilogue.md` §7), so beta=0 and beta=1 callers of a shape share
  one configuration. *(field, E9)* Ranking at the caller's beta costs one more build per surviving candidate. On a
  grouped per-tensor wgrad, the beta=1 ranking had another winner than the beta=0 ranking on two of four shapes;
  ranked at beta=0, the race reached the beta=1 best on one of them in 2 of 6 runs, ranked at beta=1 in 6 of 6.
- **Check numerics, not only finiteness, for candidates that change arithmetic order** (split-K, different
  atoms). Compare each against the base configuration's output.

## 5. Cache keys

- **The key holds every input that changes the compiled code or the candidate list, and nothing else.**
  Never key on `id(tensor)`. A freed tensor's id is reused, so the cache returns another tensor's result.
  Turbo's compiled-launch key is the launch's identity, then `(shape, stride, dtype)` per tensor, integer
  arguments by value and any other argument by type name, leaving out the stream handle; strides are in
  because the compile pins the operand layout (FP8:2988-3005).
- **A missing discriminator gives a wrong answer, not a slow one.** Turbo's MXFP4 artifact key carries
  `row_bytes` (one logical shape, two allocations, two kernels) and `scales_prepacked` (the two builds
  differ in whether the launch runs the scale preshuffle) beyond the shape (FP4:4518-4519, FP4:4650-4651).
  Leave either out and one call is handed the other's module.
- **A value used as a compile-time constant is part of the specialization**, so pass values that vary per call as
  runtime arguments, and put every compile-time value in the key.
- Build knobs that select a code path (tile order, store policy) must be in the key. Otherwise two policies
  share one cache slot and silently run as one binary.

In FlyDSL: `../flydsl/gemm.md` § autotune §5.

## 6. The launch path

- **Skip no-op tensor ops on the hot path.** A no-op `.to()` or `.reshape()` still costs a dispatch
  (`_scalar_scale`, FP8:3098-3105). Turbo reads the stream through `torch._C._cuda_getCurrentRawStream`
  (FP8:2970).
- **Under CUDA-graph capture: no timing.** A race cannot time inside capture, so fall back to the static heuristic
  configuration. Warm up and synchronize the kernel with the same compile-time key before capture starts.
- Dispatch logic in front of a launch adds host time to every call: re-measure the host path after adding any.

In FlyDSL: `../flydsl/gemm.md` § autotune §6.

## 7. Failures the race must not hide

- **`except Exception: continue` turns a build bug into "no working config".** A `NameError`, a signature
  mismatch or a key-tuple mismatch looks exactly like an incompatible configuration.
  - *(field, E12, E13, E14)* This has removed a whole launch path or optimization axis three times while every
    benchmark stayed green: the dense fp8 GEMM's NT forward path, MXFP4's epilogue-variant stage and MXFP4's
    tail-split launch mode. In one case, a launch mode failed to build for a whole 16-round campaign.
  - Record `(candidate, repr(exception))` in a module-level list that a verbose path prints.
  - Add a smoke test that builds every launch mode once without timing.
  - Assert the derived knobs (tiles per workgroup, split factor, block_n) on one host-known shape.
- **Check a configuration's LDS bytes against 160 KiB before timing it.** Depending on the toolchain, one over the
  limit may fail only at launch, without an exception, and leave the output unwritten (the signature: LLVM's
  `local memory (N) exceeds limit` on stderr, `hipErrorIllegalState` at launch).
- **A backend that raised once is disqualified for that shape.** Do not retry it on later calls.
- **A timing defect in the race can be load-bearing.** It may compensate for a cache-state difference
  between the race and the scored step. *(field, E15)* Fixing the race's warm-up and ordering lost 0.68% in one
  read of a six-shape score that ran the quantization kernels before each GEMM, per shape from −3.2% (QKV_fwd)
  to +2.0% (QKV_wgrad); on a GEMM-only score a related change to the race's timing read flat. Change race
  timing only when the race and the scorer measure the same way.

In FlyDSL: `../flydsl/gemm.md` § autotune §7.

## Related
`mainloop.md` · `epilogue.md` · `measurement.md` (rulers, null controls) ·
`correctness.md` · `../flydsl/debug-flydsl-kernel.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The month is the run's: the round timestamps in each campaign's `run.jsonl`, the commit date (E1, E9),
the date the note gives (E8, E12) or the date of the commit that first records the note (E16). "Turbo" is Primus-Turbo.
Each line names its kernel; "grouped" and "fused MoE" in a claim mark a number the card states for another kind of
kernel.

- E1 [2026-09, gfx950, FlyDSL ?, N=3, STREAM chain ruler (each GEMM timed inside a back-to-back chain), A/B alternated in three pairs, median; the four-arm localization two rounds per arm] taking the TN plan's first entry instead of racing, on 48 Llama-2 cells outside the five scored ones: TN layer 0.9163 against the race (0.9145 in the four-arm run), worst cell 0.8599 (7B gate_up wgrad, m=4096, −14%), 70B gate_up 0.8645 and 70B down 0.8686 at m=4096; the six regressing cells were all gate_up/down wgrad; the only scored TN shape picked the same tile either way; with the race restored the TN layer read 1.0237, its worst cell 1.0098, all 48 cells 1.0079 (Turbo dense per-tensor fp8 GEMM `gemm_fp8_tensorwise_flydsl_kernel`, TN wgrad, Llama-2 7B/70B at m=4096/8192; Primus-Turbo d9809c28; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_fp8_dense_gemm_campaign.md:39; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/06-autotune-dispatch.md:241)
- E2 [2026-09, gfx950, FlyDSL ?, N=1, campaign score `mxfp4` (geomean over the 12 table rows of our TF/s over a frozen reference's TF/s), one read] one more swizzle candidate (`num_xcds` 16 next to 8 below K=28672) with no margin, in a race that keeps each candidate's min over `ITERS, REPS = 20, 8`: 0.9884 → 0.9872 (−0.19%), FC2_wgrad 1.005 → 0.993 (Turbo MXFP4 dense GEMM `gemm_mxfp4_flydsl_kernel`, NT, round 3 arm S6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_112823/rounds/round-03/agent.txt:70; :41)
- E3 [2026-07, gfx950, FlyDSL ?, N=1–3, campaign score `_bench_tw_vs_mx.py` (geomean over 14 (op, projection, distribution) cells of the MXFP8 time over ours); the kept arm read three times, the baseline and the other arms once] the NN dgrad race's `(num_xcd, group_m)` list reordered from `[(4,8),(8,4),(8,2)]` (xcd 4 first) to `[(2,8),(4,8),(8,4)]` (xcd 2 first), in the same arm as a base for the forward `down` branch's empty slot: gm 0.98628 → 0.98770 / 0.98738 / 0.98700, dgrad group 0.9862 → 0.9897 / 0.9870 / 0.9880; pushing the bases one step further (dgrad xcd 1, forward xcd 2) 0.98564; adoption margin 1.5% (gpt-oss per-tensor fp8 grouped GEMM, NN dgrad, round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260731_073404/rounds/round-09/agent.txt:13; :24; myskill methodology/13-autotune-fleet.md:20)
- E4 [2026-07, gfx950, FlyDSL ?, N=3, config cache cleared and the shape re-run three times; campaign score `_bench_campaign_mxfp8.py` (geomean of tw/mx over 14 (op, projection, distribution) cells)] the NT race on `down` (N=2944) picked `xcd=4` or `xcd=8`, 2:1; the two tie at the race's synthetic balanced point and are ~3% apart on the heavy distribution, where `down heavy` read 1.010 or 0.976; adoption margin 1.5%. The next round timed base and candidate alternately in one window: three cleared-cache runs picked the same configuration for all three NT shapes, `down heavy` 1.001–1.007, gm range over four runs 0.62% → 0.12% (gpt-oss MXFP8 grouped GEMM, forward and dgrad NT; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/rounds/round-04/agent.txt:20; memory.md:142)
- E5 [2026-09, gfx950, FlyDSL 0.2.2, N=6, cleared-cache re-runs of the race per cell] race paired against the incumbent as `(base, cand, cand, base)` per rep (`_robust_ab_ratio`), every canonical M below 0.985: B=24 eight cells × 6 = 48/48 stable, B=8/32 four cells × 4 = 32/32, each pick the arm measured fastest; with the unpaired race the same tree had picked (4,8)/(8,4)/(8,4) on Down fwd N=2944, where (8,4) is 2.2% faster at the deployment point; one stability probe read 4/4 at the start of the round and 2:1 two hours later (gpt-oss-20b grouped per-tensor fp8 GEMM `grouped_gemm_fp8_tensorwise_flydsl_kernel`, forward and dgrad, round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/memory.md:268; :271; myskill pitfalls/06-autotune-dispatch.md:199)
- E6 [2026-08, gfx950, FlyDSL ?, N=4, isolated probe `_probe_r9_nnknob.py`, palindrome, median of 4 probe rounds at M = 1024/2048/8192 and of 5 at the deployed M, bitwise-identical arms] `(num_xcd, group_m)` = (8, 4) against the race's pick (4, 8): at the deployed 4096 tokens per expert +2.48% and +2.95% (two balanced sweeps), heavy skew +2.49%; at the race's two canonical M points 1024 −0.94% and 8192 +0.54%, geomean about +0.2%, under the 1.5% adoption margin; M = 2048 +0.32% (gpt-oss-20b down-projection padN dgrad NN, grouped tensorwise fp8, G=32; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/remote_data.md:28; rounds/round-06/agent.txt:21; :25; rounds/round-06/deep_raw.jsonl:2142)
- E7 [2026-08, gfx950, FlyDSL ?, N=2, race disabled and each config timed twice (repeatability under 0.15%); campaign score `gm_ratio` (geomean over 18 configs of the MXFP8 time over ours), one read per arm] the NT race scored candidates on synthetic balanced group offsets and did not pick `(2,8,0)`, which was in its list; timed per config, `(2,8,0)` was 7.0% and 7.8% faster on the two configs holding `min_ratio`; a hard-coded NT constant, the race code deleted with it: gm 1.3529 → 1.37619, min 1.1843 → 1.22707; the same constant on the wgrad path, which shared the default, collapsed wgrad to 0.38–0.55× (gpt-oss MXFP4 grouped GEMM `mxfp4_grouped_kernel.py`, round 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/rounds/round-02/agent.txt:22; :24; memory.md:162)
- E8 [2026-09, gfx950, FlyDSL ?, N=?, two probes in separate processes on one build and shape] `torch.zeros` operands 151–155 µs (`_occ.py`) against bf16 randn quantized through `QuantizedTensor.quantize` 205 µs (`_cst.py`), 32%; in a microbenchmark high-entropy operands lowered the peak 10.7% (4044 → 3614 TMAC/s) against constant ones; the note puts the rest on allocation history and page placement, untested (Turbo MXFP4 dense GEMM, out_proj_wgrad 4096×4096×32768; note, the probe runs are not recorded on this machine; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_zero_operands_fake_gemm_timing.md:8; repeated as a ruler note in local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260915_160527/memory.md:78)
- E9 [2026-09, gfx950, FlyDSL ?, N=?, each candidate pinned and timed at both betas on balanced draws; the race's choice observed over 6 scored runs] winners at beta=0 and beta=1: down G=24 (1,4) 2623.1 and (4,2) 2215.9 TF/s, gate_up G=24 (1,4) 2677.3 and (1,3) 2272.6, the same rectangle at G=32; beta=1 16% slower outright; ranked at beta=0 the race reached the beta=1 best on down in 2 of 6 runs, ranked at the caller's beta in 6 of 6; each surviving candidate is compiled twice, about doubling a cold autotune (gpt-oss-20b grouped per-tensor fp8 wgrad, variable K, down and gate_up at G=24 and G=32; Primus-Turbo d0cd4b00; local:/workspace/code/gpt_oss_docker/.claude/memory/project_b24_pad2944_merge.md:100)
- E12 [2026-08, gfx950, FlyDSL ?, N=?, campaign bench that scored NN and TN only] `cands.append([launch, (bm, gm, gn, xcd, ag), c])` inside `for bm, gm, xcd, ag in _NT_CANDIDATES`: `gn` unbound, every NT candidate dropped by `except Exception: continue`, `NT autotune found no working cfg`; the dense fp8 forward was broken on the delivered commit while ten rounds of bench stayed green (Turbo dense fp8 GEMM, NT path, found after the NN dgrad campaign 20260809_051251; notes, no run record on this machine; local:/workspace/code/gpt_oss_docker/.claude/memory/feedback_bench_scope_blind_sibling_path.md:13; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/06-autotune-dispatch.md:48)
- E13 [2026-09, gfx950, FlyDSL 0.2.4, N=1, code reading, then one scored read per arm] the TACCW/COOP epilogue-twin stage looks its base up in `_MXFP4_AT_CACHE` under a 13-field key while entries are stored under 15 fields; the `KeyError` goes to `except Exception: pass`, so the twin axis had not run in the whole campaign; with the key fixed in a later campaign, all 18 cells of the `gemm18` table still picked `(taccw, coop) = (False, False)` (Turbo MXFP4 dense GEMM for gpt-oss, `_autotune_mxfp4_config`, round 16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_085930/rounds/round-16/agent.txt:8; myskill pitfalls/06-autotune-dispatch.md:827; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/memory.md:199)
- E14 [2026-09, gfx950, FlyDSL 0.2.4, N=?, code reading and a probe of the race] the four kernel calls in `launch_mxfp4_tail` lacked the `ACT`/`PROBS` arguments, so launch modes 2 and 3 (tail split) raised `TypeError: missing a required argument: 'ACT'` and `except Exception: continue` dropped them: the tail-split leg had not been raced; myskill counts mode 2/3 as unbuilt through the whole preceding 16-round campaign `20260923_085930`; the two arguments came with `9f05cd85` (2026-09-17) (Turbo MXFP4 dense GEMM, 18-cell `gemm18` campaign, round 1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/rounds/round-01/agent.txt:11; myskill pitfalls/06-autotune-dispatch.md:50; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_085930/run.jsonl:74)
- E15 [2026-09, gfx950, FlyDSL 0.2.4, N=1, campaign score `gptoss_q` (six gpt-oss training-step cells: dual quantization of A and B, scale handoff and the packed GEMM), one read against the same day's 1.2301] race timing fixed as the KB prescribed (0.30 s wall-clock warm-up, candidate order reversed each rep): 1.2217 (−0.68%) on a clean run (`pass_drift` 0.0102); per cell QKV_wgrad +2.0%, QKV_fwd −3.2%; replaying the old timing (fixed order, min over 8 reps) on the compiled candidates gave the first-listed candidate a 2.3–2.4% discount on QKV_dgrad 32768×2880×5120, which the round reads as an effective `_SWZ_MARGIN` of about 2.9%; on the GEMM-only `gemm18` score a later change to the race's timing estimator (min over 8 reps → median of per-rep ratios) read flat, 1.0087 / 1.0123 against 1.0103 (Turbo MXFP4 dense GEMM for gpt-oss, `_autotune_mxfp4_config`, round 16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_085930/memory.md:345; rounds/round-16/agent.txt:23; :30; :58; myskill pitfalls/06-autotune-dispatch.md:788; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/memory.md:198)
- E16 [2026-07, gfx950, FlyDSL ?, N=?, ruler not stated; adoption margin >4% for a 4-wave candidate over the real 8-wave dispatch best] a 4-wave persistent NT candidate came out only ~3% over the real 8-wave dispatch best, under the 4% margin, and lost to the 4-wave non-persistent kernel at short K, where persistence was to pay; adopted on no shape, withdrawn, net change zero (Turbo grouped per-tensor fp8 GEMM, tensorwise branch, NT; note, no run record on this machine; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/06-autotune-dispatch.md:80; first recorded in myskill 99c3288, pr-merge-gate/SKILL.md:98)
