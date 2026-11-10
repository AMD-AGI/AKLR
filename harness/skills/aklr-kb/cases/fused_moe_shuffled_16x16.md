---
title: "MXFP4 fused MoE, hidden 6144, top-9, 16×16-shuffled weights (257 experts at inter 256 or 512, 65 and 33 experts) — no sort and two launches at 1–2 tokens; the sort folded into stage 1 or one prelude launch, mostly with an atomic stage 2, at 4–128; from 256 tokens sorted row tiles whose bf16, fp8 or MXFP4 route rows a separate launch sums in a fixed order, or an atomic stage 2"
kind: case
ops: [moe]
dtypes: [mxfp4]
regimes: ["tokens 1-2", "tokens 4-128", "tokens 256-4096"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, AKLR CI measurement profile 19, per Point", value: "0.137–2.489; 2 of 13 Points ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256 file 1fee59821837c26f, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (CI run 42549655045, #102)", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR CI measurement profile 18, per Point", value: "1.079–1.113; 4 of 4 Points ≥ 1.03 (m 8, 16, 32, 64)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256 file 4fbd9d02b4c053c7, m 8, 16, 32, 64", run_ref: "90b9d50 (CI run 41976547645)", date: "2026-11-05"}
  - {metric: "speedup over aiter, AKLR CI measurement profile 19, per Point", value: "0.973–1.670; 10 of 13 Points ≥ 1.03 (all but m 8, 16, 32)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048 file ef15e87894a8ae06, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (CI run 42549655045, #102)", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR CI measurement profile 19, per Point", value: "0.811–1.316; 4 of 13 Points ≥ 1.03 (m 1, 2, 4, 1024)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048 file 1bf41ccba6750300, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (CI run 42549655045, #102)", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point", value: "1.246–1.800; 4 of 4 Points ≥ 1.03 (m 512, 1024, 2048, 4096)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256 file 2ff617744254b896 as its predecessor aa1aaac0bdd0068f (same GPU code, not re-measured), m 512, 1024, 2048, 4096", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_aa1aaac0bdd0068f/report.json (base 0e02a81)", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point", value: "1.049; 1 of 1 listed Point ≥ 1.03 (m 32)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512 file 4ea93103a078dc17 as its predecessor 2a0d8c184a6adab8 (same GPU code, not re-measured), m 32", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_1ddb007af161404d/report.json, the same kernel with m 64 (base 0e02a81)", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point", value: "1.069–1.668; 4 of 4 Points ≥ 1.03 (m 512, 1024, 2048, 4096)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512 file 71df350d424c8b92 as its predecessor 98672f45ef4eedf1 (same GPU code but for one buffer bound, not re-measured), m 512, 1024, 2048, 4096", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_f2e2b374be1c9f8b/report.json, the same kernel with m 128 (base 0e02a81)", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point", value: "1.123–1.193; 6 of 6 Points ≥ 1.03 (m 4, 8, 16, 32, 64, 128)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048 file 4535d4f791472506 as its predecessor 2c7c5bacd06c8e29 (same GPU code, not re-measured), m 4, 8, 16, 32, 64, 128", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_2c7c5bacd06c8e29/report.json (base 0e02a81)", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point", value: "1.230–1.365; 3 of 3 Points ≥ 1.03 (m 256, 2048, 4096)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048 file 78f98e0bf6d4c552 as its predecessor 40937d155ca6aa98 (same GPU code, not re-measured), m 256, 2048, 4096", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_40937d155ca6aa98/report.json (base 0e02a81)", date: "2026-11-09"}
sources:
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/1fee59821837c26f.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/4fbd9d02b4c053c7.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/2ff617744254b896.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/4ea93103a078dc17.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/71df350d424c8b92.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048/ef15e87894a8ae06.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/1bf41ccba6750300.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/4535d4f791472506.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/78f98e0bf6d4c552.json
---

# MXFP4 fused MoE with 16×16-shuffled weights, hidden 6144, top-9

Nine FlyDSL kernel files for `fused_moe@1.0` under four schema names: 257 experts with inter dim 256 (`1fee5982…`,
`4fbd9d02…`, `2ff61774…`) and with inter dim 512 (`4ea93103…`, `71df350d…`), 65 experts (`1bf41ccb…`,
`4535d4f7…`, `78f98e0b…`) and 33 experts (`ef15e878…`) with inter dim 2048. Four are on main (`1fee5982…`,
`4fbd9d02…`, `ef15e878…`, `1bf41ccb…`); the other five are new and carry no CI measurement yet. Each schema embeds
its kernel in `kernel.sources`: `1fee5982…`, `ef15e878…`, `1bf41ccb…` and `78f98e0b…` as one `kernel.py`,
`4fbd9d02…` as eight files, and `4ea93103…`, `4535d4f7…`, `2ff61774…` and `71df350d…` as their `kernel.py` beside
copies of the aiter modules they use (10, 12, 16 and 20 files in all); the function names below are in those
sources. "m" is the Point's `n_tokens`: the tokens in the call, the input batch, not the rows an expert receives. A
route is one (token, expert) pair, 9m per call.

The five new files replace earlier versions that this page's runs measured and that imported aiter at run time:
`aa1aaac0…` → `2ff61774…`, `2a0d8c18…` → `4ea93103…`, `98672f45…` → `71df350d…`, `2c7c5bac…` → `4535d4f7…` and
`40937d15…` → `78f98e0b…`. Their own kernel builders are unchanged, the aiter modules the earlier versions imported
are now embedded from the aiter commit they were measured with (dbd8bf5bd), and the code paths no listed Point
reached are removed. So their GPU code is the measured versions', except that `71df350d…`'s head launch now bounds
its sorted-id reads and its stage 1 at 1024 tokens re-sorts the activation scales (Counter-signals). The measurements
below are of the earlier versions, but for E18 on `71df350d…`.

## Applies when

- Contract `fused_moe@1.0`, layout variant `shuffled_16x16`: `w1` (experts, 2 × inter, 6144) and `w2` (experts,
  6144, inter) as fp4 e2m1 in aiter's 16×16 preshuffle, one E8M0 scale per 1×32, shuffled and padded to (256, 8);
  bf16 `hidden_states`, fp32 `topk_weights`, int32 `topk_ids`, no expert mask; SiLU on separate gate and up rows,
  routing weight applied in stage 2, `act_quant` 4 (activations quantized to MXFP4); bf16 `out`
  (`aklr/ops/fused_moe/contract.py`, the `shuffled_16x16` variant). A token's 9 expert ids are distinct (the
  `topk_ids` contract); none of these names has an expert mask or `num_local_tokens`.
- Compare: `determinism: bounded`, `max_violation_fraction` 1e-3, and one bound on the whole output: SQNR ≥ 13 dB
  against the reference (float32 in validation) rounded to bf16, with no per-element bound and no bias test (`SqnrBound`,
  `aklr/core/contract/semantics/_compare.py`). Atomic accumulation is allowed.
- The model: the commit that adds the 257-expert `4fbd9d02…` names it GLM-5.2's MXFP4 MoE (256 routed experts plus
  the fused shared one, top-9) at decode sizes. The four names fit that layer (hidden 6144, MoE intermediate 2048,
  `aklr/benchmarks/models/glm52/README.md`): inter 256 and 512 are 2048 split 8 and 4 ways, 65 and 33 experts are
  64 or 32 routed experts plus the shared one. No other file names the model.
- Points. Dispatch serves a Point with the file of the highest `speedup` among the files of its schema name, from
  1.03; a workload without a CI `speedup` is not served (`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` § Environment
  variables; S6). Served on main as of #102 (6d7cfec, CI run 42549655045), by the speedups CI wrote into the files
  *(field, E1)*:

  | Name | Served on main | Not served on main |
  |---|---|---|
  | 257 experts, inter 256 | m 1, 2 `1fee5982…`; m 8–64 `4fbd9d02…` | m 4, 128–4096 |
  | 257 experts, inter 512 | none (only the kernel-less definition `c97f6644…`) | all 13 Points |
  | 65 experts, inter 2048 | m 1, 2, 4, 1024 `1bf41ccb…` | m 8–512, 2048, 4096 |
  | 33 experts, inter 2048 | m 1, 2, 4, 64–4096 `ef15e878…` | m 8, 16, 32 |

  The new files list these Points, served once CI records a `speedup` of 1.03 or more for them (this page's runs of
  their earlier versions *(field, E4, E5, E6, E7, E8)*); they raise on any other token count:

  | Name | Listed by the new files |
  |---|---|
  | 257 experts, inter 256 | m 512–4096 `2ff61774…` |
  | 257 experts, inter 512 | m 32 `4ea93103…`; m 512–4096 `71df350d…` |
  | 65 experts, inter 2048 | m 4–128 `4535d4f7…`; m 256, 2048, 4096 `78f98e0b…` |

  At every listed Point but one the name has no kernel file on main, or its main file reads below 1.03
  *(field, E1)*, so the yardstick there is aiter. The exception is m 4 of the 65 experts, which `4535d4f7…` lists
  and main's `1bf41ccb…` serves at 1.031: of the qualifying files of a name, dispatch serves the one whose workload
  ranks highest, by `speedup` and then by kernel content hash (S6). Until CI writes a `speedup` for `4535d4f7…`,
  `1bf41ccb…` keeps m 4; after that, the file with the higher CI speedup serves it if that speedup reaches 1.03, and
  `1bf41ccb…`, not aiter, is the yardstick there while it qualifies.

## Counter-signals

**Every regime**

- **Output bits vary where stage 2 adds with atomics.** `4fbd9d02…` (m 8–64), `4ea93103…` (m 32), `4535d4f7…` (m
  4–128) and `71df350d…` at m 512 add the weighted rows into `out` with bf16 atomics in arrival order; `4fbd9d02…`'s
  output read a run-to-run SQNR of 48.4–51.9 dB *(field, E3)*, the others' is not measured. Under a bitwise contract
  they need a fixed-order stage 2 (`../moe/few_tokens.md`, Counter-signals). The other files and Points sum each
  token's routes in a fixed order.
- **Precision is spent where the SQNR bound allows it.** `ef15e878…` stores its stage-2 rows as fp8 e4m3 with no
  scale at m 16, 32, 64 and 128–4096 (`cvt_pk_fp8_f32`) before the route sum; `1bf41ccb…` truncates, rather than
  rounds, its routing-weighted bf16 rows and its bf16 output at every m (`bits >> 16`); `1fee5982…` at m ≥ 512, where
  it is not served, stores int8 rows with one fp32 scale per 32 values. `2ff61774…` (every Point), `71df350d…` (m
  1024–4096) and `78f98e0b…` (every Point) store each route's stage-2 row as MXFP4, e2m1 with an E8M0 scale per 32,
  before the route sum. The contract notes that the 33- and 65-expert kernels keep partials far off aiter's error
  (`contract.py`, the FIXME at `out`'s `error_bounds`). The runs record `accuracy: passed`, not the SQNR; the error
  itself is not measured here. Under a tighter bound these Points need wider partials and rounding (code read).
- **Not the compile-function form in three files.** `1fee5982…`, `ef15e878…` and `1bf41ccb…` keep built launches in
  a module-level dict `_RUN_CACHE` filled from `run`, not in a `functools.cache` compile function, allocate their
  per-shape scratch on device `"cuda"` rather than an argument's device, and `1fee5982…` hashes its own source file
  (`Path(__file__).read_bytes()`) on a Point's first call: file I/O in launch code (`aklr/ops/README.md` § The kernel
  section and ABI). Their scratch is written by every call before it is read (`1fee5982…` zeroes its expert counters
  in `quant_input` at m ≥ 32; the others store their counts per call). The other six files use `functools.cache` and
  allocate their workspace per call.
- **Concurrent calls of the three `_RUN_CACHE` files give wrong output.** Their per-shape scratch sits in the launch
  that `_RUN_CACHE` keeps (S2, S4, S5), so two calls of one Point at once on two streams share it. With different
  inputs (20 tries per Point, on two unrestricted streams and on two streams limited to the same first 64 CUs, each
  compared with an eager call alone), `ef15e878…` gave wrong output at every Point in 16–20 of 20 tries, `1bf41ccb…`
  at every Point from m 2 in 18–20 of 20 and at m 1 on the shared CUs, and `1fee5982…` at m 2 (3 of 20) and m 8
  (20 of 20) on the shared CUs and at m 4 and from m 16 up in 19–20 of 20; at m 4096 a GPU memory access fault ended
  its process after the unrestricted tries. `4fbd9d02…` and the earlier versions of the five new files gave 0 of 20
  at every Point *(field, E16, E15)*.
- **No aiter at run time in the new files.** None of the five imports aiter. `4ea93103…`, `4535d4f7…`, `2ff61774…`
  and `71df350d…` carry the aiter modules they use as copies from aiter dbd8bf5bd: whole files, except three host
  functions of aiter's `fused_moe.py` in `71df350d…` (`fused_moe_v2_stage2.py`), changed only in imports and host
  code: environment knobs fixed at their default values, a cross-call cache removed, a `.tolist()` written as a
  list, an argument named `input` renamed, streams taken from the input's device, a few package-level names from a
  small `aiter_stubs.py`, and imports of modules not copied, in code no Point reaches, replaced by a
  `NotImplementedError`. The measured runs' environment is not
  recorded; if it set one of those knobs, what they ran differs from the new files (code read). Each file's
  configuration per m is a table in its own source, and a token count it does not list raises instead of falling
  back to aiter or the baseline. `2ff61774…` fixes the block sizes and stage-1 kernels that `aa1aaac0…` took from
  aiter's tuned table (`glm5_fp4_tuned_fmoe.csv` at dbd8bf5bd) where its own tables named none; `71df350d…` fixes the
  plan that `98672f45…` checked against aiter's configuration, runs its own driver where that ran aiter's
  `fused_moe_2stages`, and carries aiter's stage-1 builder as three edited copies
  (`mixed_moe_gemm_2stage_common_vb.py`, `…_vbgs4.py`, `…_vbgs8.py`) where `98672f45…` edited aiter's source at run
  time (S7, S9). Installing another aiter no longer changes what runs.
- **`71df350d…`'s head launch bounds its sorted-id reads.** At m 2048 and 4096 the last 32-row tile of
  `_head_resort_kernel`'s scale re-sort reads sorted ids up to index 51,327 and 86,207 (9 past the end) from a buffer
  of 51,319 and 86,199. `98672f45…` read them through an unbounded buffer descriptor; `71df350d…` gives the
  descriptor the buffer's size, so those reads return 0. The rows they feed lie past the valid count and their scales
  were already stored as 0, so the output does not change (code read, S9).
- **At 1024 tokens stage 1 no longer reads the activation scales in token order.** With that read (`001a2d8b…`, the
  review revision), two calls with different inputs on two streams gave a wrong output in about 0.35% of outputs: a
  few tokens each with one route row wrong, by up to the output's mean magnitude. A single call beside an unrelated
  GEMM did too, and stage 2 rerun alone on the captured stage-1 output reproduced the wrong rows, so the error arises
  in stage 1 or before it; the cause inside that path is not found. `71df350d…` re-sorts the scales at 1024 as at
  2048 and 4096: 0 wrong outputs in 1000 at the same speed, 1.161 at m 1024 *(field, E18)*. m 512 keeps the
  token-order read: 0 of 1200 concurrent outputs had a row off by more than 4, where the atomic stage 2 moves rows by
  up to 2.5 alone *(field, E18)*.
- **Code no Point reaches:** in `ef15e878…`, `_route` and `_route_block` (route ranks taken with atomics),
  `_route_fused`, `_quantize` and `_activate_split`; in `1bf41ccb…`, the torch sort fallback, `quant_kernel` and
  `_make_split_quant`; in `4fbd9d02…`, the split-K exchange, n-fusion, item-direct head and k-rotation of `g1_fold.py`
  and the non-atomic epilogues of `g2_port.py`; in `2ff61774…`, the fp8-row, atomic and persistent branches of its
  stage-2 GEMM copy, which its constants turn off at every Point; in the four files with copied aiter modules, most
  of those modules' functions (those that raise `NotImplementedError` included). CI measures none of them.
- **Docstrings and names.** `ef15e878…`'s docstring promises "reference-compatible rounding" while its stage-2 rows
  are fp8 at most Points; `1bf41ccb…`'s names a fallback torch sort no Point runs. The builders are named for inter
  128 (`build_mxfp4_moe_e257_i128_module`) and 1024 (`…_e33_i1024_…`, `…_e65_i1024_…`) on shapes with 256 and 2048.

**1–2 tokens**

- **65 experts at m 2 and m 4 sit near the threshold.** `1bf41ccb…` reads 1.085 at m 2 and 1.031 at m 4 as of #102
  *(field, E1)*, 1.050 and 0.979 in the CI run before *(field, E17)*, and 0.794, 1.049, 0.989, 1.014, 1.024 at m 2
  across five earlier runs in the SIKL repository's CI *(field, E2)*: which run wrote the file decides whether
  dispatch serves it.

**4–128 tokens**

- **257 experts, inter 256, at m 4 and 128: served by no file.** `4fbd9d02…` has no Point there and `1fee5982…`
  reads 0.137 and 0.314 *(field, E1)*. At m 4, 16, 128, 512 and 1024 `1fee5982…` repacks `w1`, `w2` and their scales
  into a K-major record layout on every call (`pack_weights`, two extra `pack_weight_scales` launches): about 0.64 GB
  read and as much written per call (estimated from the tensor sizes). Those five Points read 0.137–0.431, its other
  Points below 1.03 0.741–0.925 *(field, E1)*. A kernel may not keep a repacked weight between calls (`../SKILL.md`,
  Stop table), and the `shuffled_16x16` layout is the one its stages read without repacking.
- **At m 8–64 the two 257-expert inter-256 files on main compete:** `4fbd9d02…` serves at 1.079–1.113;
  `1fee5982…` reads 0.180–0.925 there *(field, E1)*, so `4fbd9d02…` is 1.20–6.0× faster (estimated from the two CI
  ratios).
- **257 experts, inter 512: m 64 and m 128 are not listed.** The kernel of `4ea93103…`, as its earlier version,
  read 1.025 at m 64 (median 181.7 µs against aiter's 186.2) and that of `71df350d…` 1.014 at m 128 (203.9 against
  207.0 µs, below the bench's 1.02 entry gate), both below 1.03 *(field, E5, E6)*; both files raise at those
  Points.
- **`4ea93103…`'s m 32 is the Point closest to the threshold.** Its earlier version's lower bound read 1.049 and
  1.040 in two runs, the second 0.010 above 1.03; a CI run below 1.03 leaves m 32 unserved, and the name has no other
  file *(field, E5)*. `71df350d…`'s m 512 read 1.069 and 1.067 *(field, E6)*. The 33- and 65-expert runs below show
  how far such margins move between runs.
- **33 experts: m 8, 16 and 32 are not served; m 4 and 64 sit near the threshold.** As of #102: m 4 1.052, m 8
  1.021, m 16 0.973, m 32 1.022, m 64 1.053 *(field, E1)*; in the CI run before: 1.031, 1.024, 0.967, 1.018, 1.031
  *(field, E17)*. Across five earlier runs in the SIKL repository's CI: m 8 0.994, 1.031,
  1.009, 1.012, 1.019; m 32 0.974, 1.024, 1.003, 1.002, 0.991; m 64 1.008, 1.033, 1.025, 1.022, 1.030; m 2 1.011,
  1.039, 1.033, 1.046, 1.049; m 16 0.983, 1.045, 1.036, 1.043, 1.039 *(field, E2)*. Which of these Points dispatch
  serves depends on the run that wrote the file. m 1 read 0.921 in the first of the five and 1.258–1.279 in the other
  four.
- **`4535d4f7…` splits stage 1's last round across workgroups at m 32–128.** Expert blocks from the 64th on run as 6
  K parts in separate workgroups; each stores its fp32 accumulators to a workspace with coherent stores and counts in
  on a per-tile counter the prelude zeroed, and the last to arrive adds the parts in part order and runs the
  epilogue. No workgroup waits for another, so it needs no co-residency; a stress run on CU-limited streams and beside
  a second call agreed with the call alone *(field, E14)*.

**256–4096 tokens**

- **257 experts, inter 256, at m 256: listed by no file.** `2ff61774…` starts at 512 and `1fee5982…` reads 0.855
  *(field, E1)*.
- **65 experts at m 512: listed by no new file and below 1.03 on main.** `1bf41ccb…` reads 1.020 as of #102
  *(field, E1)*, 1.007 in the CI run before *(field, E17)* and 1.012–1.147 across the five earlier runs
  *(field, E2)*; neither `4535d4f7…` nor `78f98e0b…` lists the Point. At the other Points those two list it reads
  0.811–1.012, and 1.031 at m 4 *(field, E1)*.
- **`78f98e0b…` reuses constants from another load point at 256 and 2048 tokens.** It is `ef15e878…`'s pipeline
  with branches written for 65 experts (`CACHE1` at 256 tokens, `GROUP_M1`/`GROUP_M2`, `PIPE1`, `PIPE2`,
  `GATHER_FUSED`, `DERIVE_TILES`, `TAILS`, `W2NT`: S11), and `_OWNED` applies the constants of the 33-expert layer's
  2048-token row at all three Points. Only at 4096 tokens do the routes per expert match that row (9m divided by the
  experts: about 567 against 558); at 2048 tokens they are about 284 and at 256 about 35 (estimated).

## Mechanism

What the nine kernels share:

- **Scaled fp4 MFMA.** `mfma_scale_f32_16x16x128_f8f6f4` with fp4 on both operands and the E8M0 scale bytes as
  operands (`ef15e878…` also its 32×32×64 form in stage 2 at m 64, 256, 512), fp32 accumulation
  (`../gemm/mainloop.md` §5). The preshuffled weights go from global straight into MFMA operand registers with 16-byte
  loads (`../gemm/datapath.md` §10, §11).
- **Activations quantized twice on the GPU:** the input rows to MXFP4 (amax of each 32-value group across lanes, an
  E8M0 exponent from it, `cvt_scalef32_pk_fp4_*`), and the SiLU(gate)·up intermediate again before stage 2: in stage
  1's epilogue, or, in `1fee5982…` at m 1 and 2, at the start of stage 2. The reference rounds both to MXFP4 too
  (`aklr/ops/fused_moe/reference.py`, its docstring).
- **Configuration per m** from constants in the source, compiled on a Point's first call. In `2ff61774…` they
  include the block sizes and stage-1 kernels its earlier version took from aiter's tuned table, in `71df350d…` the
  stage wrappers and stage-1 output format its earlier version took from aiter's configuration.

**1–2 tokens: no sort, one route per workgroup set.**

- **At m = 1 the three files with that Point run two launches and no routing sort.** Each route gets its own
  workgroups, so the expert is read straight from `topk_ids`; the token row is quantized inside each stage-1
  workgroup; stage 1 splits K over waves or workgroups and adds the parts in a fixed order; stage 2 runs one wave per
  route and adds the 9 weighted rows of a token in LDS in route order. Every row of the 16-row MFMA tile is the same
  token and only row 0 is kept. This is the few-M-block shape of `../moe/grouped_sorted_tokens.md` §3
  (the N blocks are the parallelism) taken to one route per workgroup.
- `1fee5982…` (257 experts, inter 256), m 1 and 2, two launches (flags at `kernel.py:655–707`). `stage1`: 8 column
  blocks of 64 × 9m routes × 4 K parts (288 and 576 workgroups of 4 waves); the workgroup quantizes its token's
  1536-value K quarter into LDS once (`_quant8`), then each wave runs 12 scaled MFMAs on 16 gate or up columns and
  stores row 0 as an fp32 partial per K part. `stage2_reduce`: 192 blocks of 32 output columns × m tokens, 9 waves,
  one per route; each wave adds its route's 4 partials in part order, applies SiLU·up, rounds to bf16, quantizes 4
  values per lane (`_quant4`) and moves the packed words into MFMA operand order with `ds_bpermute`, so the
  intermediate never goes to memory; 32 threads then add the 9 routes × routing weight in route order and store bf16.
- `1bf41ccb…` (65 experts), m 1, two launches: stage 1 with 9 routes × 64 tiles of 32 gate + 32 up columns (576
  workgroups), each workgroup quantizing the token row into LDS and its 4 waves taking a K quarter each; the quarters
  are added in LDS in quarter order before SiLU·up and MXFP4 (`FUSE_SPLIT_QUANT`). Stage 2 `direct_reduce`: 192
  workgroups of 32 columns, 9 waves, one per route, K = 2048, the weighted rows added in LDS in route order. At m 2
  it also runs per route without a sort, stage 1 without the K split and stage 2's rows added by a separate `reduce`
  launch in route order (`_Runner`: `direct` up to 2 tokens, `fuse_stage2_reduce` at m 1 only).
- `ef15e878…` (33 experts), m 1 and 2: the first two rows of its table below.

**4–128 tokens: sorted expert blocks, the sort inside stage 1 or in one prelude launch.**

- `4fbd9d02…` (257 experts, inter 256), m 8, 16, 32, 64: two launches with the routing sort folded into stage 1:
  `../moe/few_tokens.md` describes the fold.
  - Stage 1 (`g1_fold.py`): 16-row expert blocks × 2 blocks of 128 gate + 128 up columns, 4 waves of 4 16-column
    tiles; K = 6144 in 24 tiles of 256. Weights non-temporal into a 2-stage register ring; the 16 hidden rows gathered
    by token id two loop bodies ahead, quantized inline into a 3-stage XOR-swizzled LDS ring; 8 scaled MFMAs per tile
    and wave at raised priority (`s_setprio`); accumulators through LDS, SiLU·up, MXFP4 with shuffled E8M0 scales in
    sorted-row order. At m = 32 the tile order is grouped per XCD after the bound test
    (`../gemm/schedule_and_locality.md` §2). 144–554 workgroups.
  - Stage 2 (`g2_port.py`, aiter's `mxfp4_gemm2.py` with a CK-style tile remap `_g2_spart_tile`, the expert id read
    before the A loads, and `amdgpu-kernarg-preload-count` 16, `../gemm/small_m.md` §6): per 16-row block,
    24 workgroups of 256 output columns, K = 256 in one tile, the MXFP4 intermediate by buffer→LDS DMA, weights
    non-temporal into registers, × routing weight, bf16x2 `atomicrmw fadd` into `out`. Upper-bound grids of
    1728–6648 workgroups; those past the device bound exit (`../moe/grouped_sorted_tokens.md` §2, §6).
- `4ea93103…` (257 experts, inter 512), m 32: the same fold in two launches of its own (`_fold_run`). Every stage-1
  workgroup sorts all routes itself in LDS (no grid-wide barrier) into 16-row expert blocks, starts its weight stream
  before the sort's rank and publish steps since it needs only the expert id, loads the gathered hidden rows two
  k-tiles ahead of their MXFP4 quantization in the loop, and runs 256-column gate/up tiles; the grid zeroes `out`.
  Stage 2 walks (m-block, n-block) tiles in interleaved groups, loads each row's sorted id and routing weight before
  the GEMM and adds pairs of bf16 columns into `out` atomically. At m 32 stage 1 takes XCD-contiguous tiles and
  stage 2 an XCD-major block order; both preload 16 kernel arguments. Paired against the version before the fold was
  added, about +3.9% at m 32 *(field, E11)*.
- `4535d4f7…` (65 experts), m 4–128. Stage 1 is aiter's builder copied (32-row tiles of 128 gate/up columns), with
  w1 loaded two K tiles ahead and SiLU·up quantized to MXFP4 in its epilogue with 4-byte stores; stage 2 is aiter's
  copied too, adding into `out` with bf16 atomics, each expert block's K split over two workgroups (the atomics sum
  them too), w2 non-temporal in 64-column tiles, the routing loads issued at kernel start.
  - m 4, 8, 16: no prelude. Stage 1 sorts the routes, quantizes its A rows straight into LDS and zeroes `out`
    itself; below 32 tokens every expert has at most one 32-row block, since a token's expert ids are distinct.
  - m 32, 64, 128: a prelude launch sorts (route counts by LDS atomic adds, a one-wave scan), quantizes the
    input to MXFP4, zeroes `out` and the split-tail counters; stage 1's last, mostly empty round of blocks is split
    six ways in K (Counter-signals).
  - Paired against the version before each was added: the one FlyDSL prelude in place of aiter's sort, quantization
    and zeroing about +3.1% to +3.8% per listed Point, the non-temporal w2 with 64-column stage-2 tiles +2.3% to
    +3.1% *(field, E12)*.
- `ef15e878…` (33 experts), served at m 4, 64 and 128: the rows m 4–32 and 64–1024 of its table below.
  `1bf41ccb…` runs m 4–128 on sorted 16-row tiles (32 at m 128) in four launches and is served only at m 4.

**256–4096 tokens: sorted row tiles whose route rows (bf16, fp8 or MXFP4) a separate launch sums in a fixed order,
or an atomic stage 2.**

- `2ff61774…` (257 experts, inter 256), m 512–4096: a routing launch, stage 1, stage 2 and a route sum.
  - A routing launch sorts the routes per expert and, in extra workgroups, quantizes the activations to MXFP4 in
    token order (bit-exact with aiter's unshuffled MX quantization, by the code's comment); at 512 and 1024 tokens
    more workgroups read the first and last experts' W1 and W1 scales while HBM is otherwise idle, so stage 1's first
    workgroups find them in the memory-side cache. At 2048 and 4096 (`_SPREAD`), when stage 1's kernel is
    XCD-swizzled, the sort deals the valid blocks evenly over the 8 XCD slices of its grid.
  - Stage 1 is aiter's FlyDSL builder (a vendored copy) building its sorted A-scale tiles in LDS from the token-order
    scales, so no scale-shuffle launch runs.
  - Stage 2 (`mxfp4_moe_gemm2`, a port of aiter's a4w4 down GEMM) writes each route's row as MXFP4; 256-column tiles
    let an expert's M tiles share w2, kept in L2 (not non-temporal) at 2048 and 4096 with an output-tile order of
    group 8 (`_sp801`); a wave whose rows are all padding skips its passes.
  - `ws_route_sum_fp4_kernel`: one workgroup per token decodes its 9 rows, weights them and adds them in route order
    in fp32, stores bf16.
  - Paired against the version before each was added (a step may bundle more than the change named): MXFP4 route
    rows with the FlyDSL route sum, replacing fp8 rows and aiter's reduction, about +5.0%, +6.1% and +7.8% at 1024,
    2048 and 4096; the in-LDS scale tiles, replacing a separate scale-shuffle launch, +2.2% to +4.2% per listed Point
    *(field, E9)*.
- `71df350d…` (257 experts, inter 512), m 512–4096: its own driver (`_into_out`) around launches of its own and
  copies of aiter's FlyDSL stage kernels.
  - `_route_kernel`, one launch: the sort into block-padded expert ranges, the activations' MXFP4 quantization in
    token order, and at m 512 the zeroing of `out`.
  - Stage 1 is aiter's FlyDSL kernel named per m in `_OVERRIDE`, built by one of three edited copies of aiter's
    stage-1 builder (at 1024–4096 its XCD tile remap covers the valid blocks only; at 512 it reads the activation
    scales in token order, so no scale re-sort runs). At 1024–4096 it runs as two launches, the first sort blocks by
    `_head_resort_kernel`, fused with the scale re-sort, and the rest in whole dispatch rounds; 4096 uses 192-row
    tiles.
  - Stage 2 at m 512: aiter's v2 atomic kernel into `out` (`_flydsl_v2_stage2_wrapper`, copied). At 1024–4096
    `_es_stage2_kernel`, persistent on 512 workgroups and expert-stationary: a workgroup holds one panel of w2 (an
    expert × 1/24 of the output columns) in registers while the next panel's loads are in flight and streams the
    expert's sorted rows 32 at a time through an LDS double buffer, writing unweighted MXFP4 route rows; at 2048 and
    4096 it takes the experts in descending route-count order. `_fp4_reduce_kernel` then adds each token's rows,
    weighted, in route order in fp32.
  - The persistent expert-stationary stage 2, paired against the version before it was added: about +1.5%, +6.1%
    and +4.7% at 1024, 2048 and 4096 *(field, E10)*.
- `78f98e0b…` (65 experts), m 256, 2048, 4096: `ef15e878…`'s pipeline with the constants of its 2048-token row (96-row
  tiles) and branches of its own for 65 experts. One launch routes per expert and quantizes the input in separate
  workgroups; both GEMMs derive their tiles from the expert counts, so no tile-list launch runs. Stage 1 fills its LDS
  panels straight from the routed rows of the quantized input, the next panel prefetched into registers in the last
  slice of each K step into double-buffered LDS; experts' short last tiles run 32- and 64-row copies of the tile body.
  Stage 2 is software-pipelined, the next activation slab and weights in flight during the current step's MFMAs, 8
  waves of 32 columns per 256-column tile, and writes MXFP4 route rows that `_reduce_fp4` adds in route order in
  fp32. At 256 tokens both weights are streamed non-temporally. The direct copy of the activations into LDS with the
  first weight slice loaded a step ahead, paired against the version before it was added: about +7.0% and +7.2% at
  2048 and 4096 *(field, E13)*.
- `1bf41ccb…` (65 experts), m 512 and 1024 (served at 1024), five launches: `collect` (per expert, ballot ranks in
  input order; extra workgroups quantize the input) and `compact` (expert-major 64-row blocks padded with a sentinel
  row); stage 1 on 64-row tiles, 4 waves of 32 + 32 columns sharing the tile's activation rows through two alternating
  LDS pages, the next K chunk loaded through a loop-carried `range(..., init=...)`, SiLU·up through LDS and one lane
  per row quantizing its 32 values; stage 2 on 64-row tiles, 4 waves × 128 columns, with the tile's whole K = 2048
  activation panel loaded into LDS once so the MFMA loop has no cross-wave barrier (`FULL_STAGE_A`), MFMA operands
  swapped so a lane holds 4 adjacent columns, 8-byte stores of the weighted rows; `reduce` adds each token's 9 rows in
  route order. No atomics on data at any served Point (the m 4–128 routing takes route ranks with LDS atomics;
  results stay indexed by route).
- `ef15e878…` (33 experts) changes structure with m across all three regimes. Every GEMM maps its workgroup id with
  `_chiplet_order`, a contiguous slice of the tile order per XCD (`../gemm/schedule_and_locality.md` §2); one aligned
  scale dword serves two 16-column halves and two K slices (`PACKED_SCALES1/2`); the intermediate is rounded to bf16
  before its MXFP4 quantization. Stage-1 weights load non-temporal at m 2–256, cached at m 1 and m ≥ 512 (`CACHE1`).
  Per m (`build_mxfp4_moe_e33_i1024_module`):

| m | Launches | Routing | Stage 1 | Stage 2 rows and route sum |
|---|---|---|---|---|
| 1 | 2 | none (`DIRECT`) | `_gemm1_split_fused`: 144 workgroups of 16 waves, 8 per K half; each quantizes the token row into LDS; halves added in order, SiLU, MXFP4 | `_gemm2_route_reduce`: a wave per route, the weighted rows added in LDS in route order |
| 2 | 4 | `_quantize_route_fused`: the grid quantizes the input; workgroup 0 ranks each expert's routes with ballots in input order and writes the 16-row tile table | `_gemm1_grouped_split_fused`: 16-row tiles, 2 K halves × 4 waves | `_gemm2`, fp32 rows; `_reduce` |
| 4, 8, 16, 32 | 4 | the same | `_gemm1`, 16-row tiles gathered from the quantized input by route; 64 + 64 columns and 4 waves (m 4), 128 + 128 and 8 waves (8, 16), 4 waves with K steps in pairs (32) | fp32 rows and `_reduce` at m 4, 8; fp8 rows through LDS and `_reduce_fp8x4` at 16, 32 |
| 64–1024 | 5 | `_route_expert`: one workgroup per expert compacts its routes in input order while the rest of the grid quantizes the input | `_gather_quant` (a coalesced XOR-swizzled LDS transpose from m 256) copies each tile's rows in tile-major order; `_gemm1` with 32-, 48- or 96-row tiles (m 64, 128, 256–1024) staged through LDS two K steps at a time, the next weights prefetched at m 64, 128, the quantized output staged in LDS for 16-byte stores except at m 128; groups of 2–3 row tiles of one expert per column sweep at m 512, 1024 (`GROUP_M`) | `_gemm2`, fp8 rows, `_reduce_fp8x4`; 32×32×64 MFMA at m 64, 256, 512 |
| 2048, 4096 | 6 | the same plus `_route_tiles` | 96- and 128-row tiles, `GROUP_M` 3 and 9 | the same, 16×16 MFMA |

## Result

Both measurements use the ruler of `MEASUREMENT_PROFILE` in `aklr/core/validation/stages/_measure.py`, a one-sided
Student-t lower bound of aiter's time over the kernel's after 16 or 32 rounds of CUDA-graph replays with L2 flushed
before each call. #102 measured main's files and this page's runs measured the new files under its version 19, which
collects kernel times with AKLR's own rocprofiler-sdk tracer (#33); `4fbd9d02…`'s numbers and the CI run before #102
are from version 18, which collected them through Kineto.

**On main** as of #102 (6d7cfec, CI run 42549655045; `4fbd9d02…`: 90b9d50, CI run 41976547645), the speedups
dispatch reads *(field, E1)*. Bold: 1.03 or above.

| File | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 257 experts, inter 256 `1fee5982…` | **2.489** | **2.016** | 0.137 | 0.925 | 0.180 | 0.820 | 0.813 | 0.314 | 0.855 | 0.372 | 0.431 | 0.793 | 0.741 |
| 257 experts, inter 256 `4fbd9d02…` | – | – | – | **1.113** | **1.079** | **1.112** | **1.094** | – | – | – | – | – | – |
| 33 experts `ef15e878…` | **1.272** | **1.109** | **1.052** | 1.021 | 0.973 | 1.022 | **1.053** | **1.186** | **1.237** | **1.285** | **1.670** | **1.090** | **1.072** |
| 65 experts `1bf41ccb…` | **1.316** | **1.085** | **1.031** | 0.984 | 0.965 | 0.998 | 1.012 | 0.898 | 0.958 | 1.020 | **1.238** | 0.868 | 0.811 |

"–": no workload in that file. Every workload reads `accuracy: passed`. Earlier runs of the 33- and 65-expert kernels
moved the Points near 1.03 across it *(field, E2, E17)*: `1bf41ccb…`'s m 4 read 0.979 in the CI run before #102.

**The new files' earlier versions** (`python -m aklr validate` at main 0e02a81 on an MI355X, FlyDSL 0.3.2, aiter
0.1.21.dev66+gdbd8bf5bd, torch 2.11.0+rocm10.0.0). The new files keep their GPU code (`71df350d…` but for its
sorted-id bound and its scale path at m 1024, which E18 measured: 1.061, 1.161, 1.425, 1.649). CI writes `speedup`
into the new files after they merge; until then dispatch serves none of their Points. Bold: 1.03 or above. m 64
(1.025) and m 128 (1.014) are below 1.03, so the files do not list them. Each row is the first of two runs; the
second is in the evidence line.

| File (measured as) | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 257 experts, inter 256 `2ff61774…` (`aa1aaac0…`) *(field, E4)* | – | – | – | – | – | – | – | **1.246** | **1.505** | **1.587** | **1.800** |
| 257 experts, inter 512 `4ea93103…` (`2a0d8c18…`) *(field, E5)* | – | – | – | **1.049** | 1.025 | – | – | – | – | – | – |
| 257 experts, inter 512 `71df350d…` (`98672f45…`) *(field, E6)* | – | – | – | – | – | 1.014 | – | **1.069** | **1.161** | **1.409** | **1.668** |
| 65 experts `4535d4f7…` (`2c7c5bac…`) *(field, E7)* | **1.193** | **1.135** | **1.123** | **1.156** | **1.179** | **1.152** | – | – | – | – | – |
| 65 experts `78f98e0b…` (`40937d15…`) *(field, E8)* | – | – | – | – | – | – | **1.230** | – | – | **1.365** | **1.276** |

Every measured Point passed accuracy with a worst violation fraction of 0. Median call times: `2ff61774…` 120.9 µs
at 512 and 247.2 at 4096 against aiter's 150.8 and 445.1; `71df350d…` 225.8 and 363.6 against 241.4 and 607.0;
`4535d4f7…` 96.7 at 4 and 212.9 at 128 against 115.6 and 245.3; `78f98e0b…` 234.8 at 256 and 855.4 at 4096 against
289.0 and 1093.5; `4ea93103…` 132.9 at 32 against 139.9 *(field, E4, E5, E6, E7, E8)*.

## Reproduce

On an MI355X, from the repository root, one schema file at a time:

```
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/2ff617744254b896.json
```

and the same for the other paths under `sources`. Each runs the audit and bench stages and writes
`aklr_validation/report.md` with the per-Point speedup (`aklr/ops/README.md` § Commands). Add `--base origin/main`
only on a branch that edits the file: with the base set, bench measures only schemas that are not at the base, so on
an unchanged checkout it skips the bench (`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). The
local result does not count; CI's measurement is the one dispatch reads (`aklr/ops/README.md` § Workflow).

## Rollback

Delete a schema file (in a PR): its Points fall back to another file of the same name where that one reaches 1.03,
else go unserved: the call returns to the caller's own path, and AKLR runs no baseline in its place. Deleting
`4fbd9d02…` leaves m 8–64 of the 257-expert inter-256 name unserved, since `1fee5982…` is below 1.03 there. At every
Point a new file lists but m 4 of the 65 experts, its name has no kernel file on main or the main file is below 1.03
*(field, E1)*, so deleting a new file leaves those Points unserved; m 4 goes back to `1bf41ccb…` while it reads 1.03
or more. `AKLR_DISABLE=fused_moe@1.0` stops serving the whole contract, every other MoE kernel included. Points CI
measures below 1.03 are not served without any rollback. If the contract becomes bitwise, remove `4fbd9d02…`,
`4ea93103…`, `4535d4f7…` and `71df350d…`'s m 512 workload; if the error bound tightens, remove the files and Points
that keep fp8 or MXFP4 route rows or truncate (`ef15e878…`, `1bf41ccb…`, `2ff61774…`, `71df350d…` at 1024–4096,
`78f98e0b…`).

## Evidence

`<file>#kernel.py:<line>` is a line of the `kernel.py` source the schema file embeds; `g1_fold.py` and `g2_port.py`
are sources of `4fbd9d02b4c053c7.json`. E4–E15 measured the earlier versions of the five new files (aa1aaac0bdd0068f
of 2ff617744254b896, 2a0d8c184a6adab8 of 4ea93103a078dc17, 98672f45ef4eedf1 of 71df350d424c8b92, 2c7c5bacd06c8e29 of
4535d4f791472506, 40937d155ca6aa98 of 78f98e0bf6d4c552); the new files keep their GPU code, `71df350d…` but for its
head launch's sorted-id bound (S9) and its scale path at m 1024 (E18).

- E1 [2026-11, gfx950, FlyDSL ?, N=1, AKLR CI, measurement profile 19; 4fbd9d02b4c053c7: profile 18] speedup over aiter in the committed files at main 6d7cfec (#102), per Point m 1–4096: 1fee59821837c26f 2.489, 2.016, 0.137, 0.925, 0.180, 0.820, 0.813, 0.314, 0.855, 0.372, 0.431, 0.793, 0.741; ef15e87894a8ae06 1.272, 1.109, 1.052, 1.021, 0.973, 1.022, 1.053, 1.186, 1.237, 1.285, 1.670, 1.090, 1.072; 1bf41ccba6750300 1.316, 1.085, 1.031, 0.984, 0.965, 0.998, 1.012, 0.898, 0.958, 1.020, 1.238, 0.868, 0.811 (6d7cfec, CI run 42549655045); 4fbd9d02b4c053c7 at m 8, 16, 32, 64: 1.113, 1.079, 1.112, 1.094, not re-measured by #102 (90b9d50, CI run 41976547645)
- E2 [2026-10, gfx950, FlyDSL ?, N=5, CI of the SIKL repository, before these kernels moved to AKLR; FlyDSL of the runs: not recorded, 0.3.1, then 0.3.4.1 three times] the five committed measurements of an unchanged kernel, in commit order: 33 experts m 1 0.921, 1.279, 1.261, 1.278, 1.258; m 2 1.011, 1.039, 1.033, 1.046, 1.049; m 8 0.994, 1.031, 1.009, 1.012, 1.019; m 16 0.983, 1.045, 1.036, 1.043, 1.039; m 32 0.974, 1.024, 1.003, 1.002, 0.991; m 64 1.008, 1.033, 1.025, 1.022, 1.030; 65 experts m 2 0.794, 1.049, 0.989, 1.014, 1.024; m 256 1.034, 0.968, 0.917, 1.026, 1.023; m 512 1.147, 1.092, 1.012, 1.107, 1.113 (files ef15e87894a8ae06 and 1bf41ccba6750300; SIKL commits baa858028, cc6e3b96c, cca5b12df, 102877ca0, 5571f4589)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=3, three of the bench's input sets per token count, SQNR of the bf16 output between two runs] 48.35–51.87 dB at m 8, 16, 32, 64 (the 4fbd9d02b4c053c7 sources, byte-identical; local:/workspace/code/gpt_oss_docker/scratch/glm52_sikl_port/evidence/moe_check2_result.json)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=2, two AKLR validate runs of 16 timing rounds each, measurement profile 19] speedup over aiter 1.246, 1.505, 1.587, 1.800 at m 512, 1024, 2048, 4096; median kernel 120.85, 134.01, 179.83, 247.16 µs, aiter 150.82, 201.94, 285.59, 445.12 µs; accuracy passed, worst violation fraction 0 (…_n_experts257_topk9_inter_dim256, file aa1aaac0bdd0068f, the earlier version of 2ff617744254b896, base 0e02a81; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_aa1aaac0bdd0068f/report.json); a second profile-19 run of the same file read 1.248, 1.473, 1.592, 1.812 (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/results/g2/glm_aa1aaac0bdd0068f/validation/report.json)
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=2 at m 32, two AKLR validate runs of 16 timing rounds each; N=1 at m 64, 32 timing rounds; measurement profile 19] speedup over aiter 1.049 at m 32, 1.025 at m 64; median kernel 132.86 and 181.68 µs, aiter 139.94 and 186.19 µs; accuracy passed, worst violation fraction 0 (…_n_experts257_topk9_inter_dim512, the kernel of file 2a0d8c184a6adab8, the earlier version of 4ea93103a078dc17, measured as 1ddb007af161404d, which also held m 64; base 0e02a81; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_1ddb007af161404d/report.json); a second profile-19 run of 2a0d8c184a6adab8 read 1.040 at m 32 (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/results/g6/glm_2a0d8c184a6adab8/validation/report.json)
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=2 at m 512–4096, two AKLR validate runs of 16 timing rounds each; N=1 at m 128, 16 timing rounds; measurement profile 19] speedup over aiter 1.014, 1.069, 1.161, 1.409, 1.668 at m 128, 512, 1024, 2048, 4096 (m 128 below the 1.02 entry gate: `not_faster`); median kernel 203.92, 225.77, 247.51, 290.50, 363.60 µs, aiter 206.97, 241.43, 287.56, 410.69, 607.03 µs; accuracy passed, worst violation fraction 0 (…_n_experts257_topk9_inter_dim512, the kernel of file 98672f45ef4eedf1, the earlier version of 71df350d424c8b92, measured as f2e2b374be1c9f8b, which also held m 128; base 0e02a81; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_f2e2b374be1c9f8b/report.json); a second profile-19 run of 98672f45ef4eedf1 read 1.067, 1.147, 1.393, 1.662 (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/results/g3/glm_98672f45ef4eedf1/validation/report.json)
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=2, two AKLR validate runs of 16 timing rounds each, measurement profile 19] speedup over aiter 1.193, 1.135, 1.123, 1.156, 1.179, 1.152 at m 4, 8, 16, 32, 64, 128; median kernel 96.73, 139.97, 179.29, 203.78, 206.31, 212.87 µs, aiter 115.55, 158.94, 201.43, 235.75, 243.53, 245.31 µs; accuracy passed, worst violation fraction 0 (…_n_experts65_topk9_inter_dim2048, file 2c7c5bacd06c8e29, the earlier version of 4535d4f791472506, base 0e02a81; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_2c7c5bacd06c8e29/report.json); a second profile-19 run of the same file read 1.182, 1.137, 1.117, 1.151, 1.173, 1.150 (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/results/g7/glm_2c7c5bacd06c8e29/validation/report.json)
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=2, two AKLR validate runs of 16 timing rounds each, measurement profile 19] speedup over aiter 1.230, 1.365, 1.276 at m 256, 2048, 4096; median kernel 234.80, 487.23, 855.38 µs, aiter 288.96, 665.60, 1093.51 µs; accuracy passed, worst violation fraction 0 (…_n_experts65_topk9_inter_dim2048, file 40937d155ca6aa98, the earlier version of 78f98e0bf6d4c552, base 0e02a81; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/glm/glm_40937d155ca6aa98/report.json); a second profile-19 run of the same file read 1.183, 1.382, 1.274 (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/results/g4/glm_40937d155ca6aa98/validation/report.json)
- E9 [2026-11, gfx950, FlyDSL ?, N=?, campaign paired timing: the version without the change over the version with it, minus one] aa1aaac0bdd0068f's campaign: fp8 route rows and aiter's reduction in place of MXFP4 route rows and the FlyDSL route sum +0.0, +5.0, +6.1, +7.8% at m 512, 1024, 2048, 4096 (round cb1d5c9a); a separate scale-shuffle launch in place of the sorted A-scale tiles built in LDS +4.2, +3.4, +3.3, +2.2% (round afac3503) (local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_tp8_l/run/workspace/forge_experiments/ablation.json)
- E10 [2026-11, gfx950, FlyDSL ?, N=?, campaign paired timing, as E9] 98672f45ef4eedf1's campaign: the version before the persistent expert-stationary stage 2 with next-panel prefetch +1.5, +6.1, +4.7% at m 1024, 2048, 4096 (round 03495731; local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/glm_l/run/workspace/forge_experiments/ablation.json)
- E11 [2026-11, gfx950, FlyDSL ?, N=?, campaign paired timing, as E9] 2a0d8c184a6adab8's campaign: the routing sort and zeroing of `out` outside stage 1 +3.9% at m 32 (round 8a18ac84; local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/glm_s/run/workspace/forge_experiments/ablation.json)
- E12 [2026-11, gfx950, FlyDSL ?, N=?, campaign paired timing, as E9] 2c7c5bacd06c8e29's campaign: aiter's sort, quantization and zeroing in place of one FlyDSL prelude +3.8, +3.1, +3.1, +3.3, +3.3, +3.3% at m 4, 8, 16, 32, 64, 128 (round a89ca455); cached stage-2 weights without the 64-wide tiles +2.8, +2.5, +2.6, +2.8, +3.1, +2.3% (round 4085ec54) (local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_ep4_s/run/workspace/forge_experiments/ablation.json)
- E13 [2026-11, gfx950, FlyDSL ?, N=?, campaign paired timing, as E9] 40937d155ca6aa98's campaign: stage-1 activations not copied straight into LDS and the first weight slice not loaded a step ahead +7.0, +7.2% at m 2048, 4096 (round 79e1b259; local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_ep4_l/run/workspace/forge_experiments/ablation.json)
- E14 [2026-11, gfx950, FlyDSL ?, N=3, campaign stress check, 3 tries per condition] every Point, including m 64 and m 128, on a stream limited to 16 and to 64 CUs and beside a second call on another stream agreed with the call alone within the bounded comparison: PASS for the five delivered kernels, byte-identical to the sources of the new files' earlier versions (local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_tp8_l/stress.log, local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_ep4_s/stress.log, local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_ep4_l/stress2.log, local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/glm_l/stress.log, local:/workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/glm_s/stress.log)
- E15 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and Point (24 graph replays), stress checks on MI355X at main 0e02a81] 0 differing tries for aa1aaac0bdd0068f, 2a0d8c184a6adab8, 98672f45ef4eedf1, 2c7c5bacd06c8e29 and 40937d155ca6aa98 (the earlier versions of the five new files) at every listed Point under (a) the forge driver's conditions: a stream limited to 16 CUs, one limited to 64 CUs, a second call with the same inputs on another stream, replays of one captured graph; (b) one CUDA graph replayed 24 times with three input sets copied into its buffers, each replay compared with an eager call on the same inputs; (c) two calls at once on two streams, each with its own input set and outputs, on unrestricted streams and on two streams limited to the same first 64 CUs, each compared with an eager call alone (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/README.txt; per file under local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/results/, local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/graph/results/ and local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/concurrent/results/)
- E16 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and Point, check (c) of E15 on main's unmodified files] wrong output, tries out of 20 on unrestricted streams / on the shared 64 CUs: ef15e87894a8ae06 m 1 16/20, m 2 19/20, m 4 18/20, m 8 19/20, m 16 19/20, m 32–4096 20/20; 1bf41ccba6750300 m 1 0/20, m 2 18/20, m 4–32 19/20, m 64–4096 20/20; 1fee59821837c26f m 1 0/0, m 2 0/3, m 4 20/20, m 8 0/20, m 16 20/20, m 32 19/20, m 64–128 20/20, m 256 19/20, m 512–2048 20/20, m 4096 20 of 20 unrestricted, then a GPU memory access fault in `grouped_gemm_3`; 4fbd9d02b4c053c7 0 at every Point (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/main_control/README.txt; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/main_control/results/g7/conc_main_ef15e87894a8ae06/run.log, local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/main_control/results/g7/conc_main_1bf41ccba6750300/run.log, local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/main_control/results/g5/conc_main_1fee59821837c26f/run.log, local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/main_control/results/g7/conc_main_4fbd9d02b4c053c7/run.log)
- E17 [2026-11, gfx950, FlyDSL ?, N=1, AKLR CI, measurement profile 18] speedup over aiter in the committed files before #102 (at 0e02a81), per Point m 1–4096: 1fee59821837c26f 2.460, 2.024, 0.136, 0.909, 0.178, 0.827, 0.828, 0.311, 0.861, 0.372, 0.426, 0.808, 0.758; ef15e87894a8ae06 1.217, 1.090, 1.031, 1.024, 0.967, 1.018, 1.031, 1.192, 1.170, 1.263, 1.720, 1.073, 1.077; 1bf41ccba6750300 1.282, 1.050, 0.979, 0.960, 0.941, 0.967, 0.969, 0.863, 0.954, 1.007, 1.221, 0.867, 0.803 (d908d01, CI run 42000899019)
- E18 [2026-11, gfx950, FlyDSL 0.3.2, N=500 tries per run (two streams, different inputs, AKLR timing inputs), each output compared with the call alone, bitwise at m 1024, rows off by more than 4 at m 512] 001a2d8b86488f45 at m 1024: 8, 6 and 8 wrong outputs of 2000 in three runs, one route row wrong per affected token, up to 45.75 off; stage 2 rerun alone on the captured stage-1 output reproduced them; 759db91b501a7b1d (the same without the split stage 1 at 1024) 2 of 1000; 71df350d424c8b92 0 of 1000, and AKLR validate (measurement profile 19, 16 rounds) 1.061, 1.161, 1.425, 1.649 at m 512, 1024, 2048, 4096, accuracy passed; 71df350d424c8b92 at m 512: 0 of 1200 outputs with a row off by more than 4, worst 3.0, alone repeats up to 2.5 (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/glm_concurrency_1109/README.txt)
- S1 aklr/ops/fused_moe/contract.py:271 `SqnrBound(min_sqnr_db=13.0)` under the FIXME at :269–270 (e33/e65 kernels keep fp8/MXFP4 partials); compare at :299; the `shuffled_16x16` variant at :278–286
- S2 fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/1fee59821837c26f.json#kernel.py:655 `pack_weights` at m 4, 16, 128, 512, 1024, launched at :763–785 (`pack_weight_scales`, :138); stage 1 at :245, stage-2 gather by `ds_bpermute` at :374–381, route sum in route order at :394–400; `_RUN_CACHE` at :801, `read_bytes()` at :653, device `"cuda"` at :708; counters zeroed per call at :234–236
- S3 fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/4fbd9d02b4c053c7.json#kernel.py:58 `_plan` (functools.cache) builds the fold for m in {8, 16, 32, 64}, XCD grouping at m 32 (:37); workspace per call at :144; g1_fold.py:518 non-temporal weights, :1143 `s_setprio(1)`; g2_port.py:328 kernarg preload 16, :899–915 bf16x2 `atomicrmw fadd`
- S4 fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048/ef15e87894a8ae06.json#kernel.py:1340 per-m flags (to :1493) and launch chain at :1527–1574; fp8 rows at :1231 and :1238, `_reduce_fp8x4` at :1320; `_chiplet_order` at :156; docstring at :1; `_RUN_CACHE` at :1593, device `'cuda'` at :1494
- S5 fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/1bf41ccba6750300.json#kernel.py:1380 per-m flags in `_Runner` (to :1472); bf16 truncation of the stage-2 rows at :777 (m ≥ 2) and :974 (m 1), of the output at :982 (m 1) and :1080 (m ≥ 2); the other stage-2 store paths (:782–803, :843–867) are reached by no Point; `FULL_STAGE_A` at :293 and :455–467; torch sort fallback at :1493–1509 behind `fused_sort` (:1435); per-shape scratch in `_Runner.__init__` at :1421–1445, kept by `_RUN_CACHE` at :1550
- S6 aklr/core/runtime/dispatch/_dispatcher.py:112 a workload qualifies at `speedup >= AKLR_MIN_SPEEDUP` (`aklr/core/contract/spec/_workloads.py:38`; a workload without `speedup` does not, :41); the highest-ranked file serves a Point (:114–117); `rank_key` (aklr/core/contract/op.py:948) orders by accuracy not failed, then `speedup`, then the kernel's content hash (:951–952)
- S7 fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/2ff617744254b896.json#kernel.py:2337 `_BLOCK_M` and :2341 `_STAGE1_KERNEL` (aiter dbd8bf5bd's tuned block sizes and stage-1 kernels, 1024's kernel its own), `_g2` at :2349; `_STAGE2_KERNEL` at :2369 (MXFP4 route rows, W2 kept in L2 and `_sp801` at 2048, 4096; comment at :2366–2368), `_SORT_CFG` at :2378, `_SPREAD` at :2381, `_W1_PREFETCH` at :2385 and its tail at :2388; `_plan` (functools.cache) at :2392, raising for other m at :2394, its fused-sort and quantization asserts at :2411–2412, the route sum of MXFP4 rows at :2402; `compile_sort_fused` at :2002, the activation quantization at :1903–1909, the W1 prefetch at :1951; `ws_route_sum_fp4_kernel` at :1838, routes in route order at :1864; `_stage1_tsl` at :5453; aiter's FlyDSL modules as copies in the other 15 sources (`moe_kernels.py`, `mixed_moe_gemm_2stage_common.py`, `mxmoe_dispatcher.py`, ...)
- S8 fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/4ea93103a078dc17.json#kernel.py:94 `_FOLD` (m 32: XCD-contiguous stage 1, XCD-major stage 2) and the block and tile sizes at :95–103; `_fold_stage1` at :466, every workgroup sorting all routes at :517, the weight stream before the sort's publish at :346; `_atomic_bf16_epilog` at :730; `_fold_plan` at :1044 and `_fold_run` (workspace per call at :1079) at :1076; `run` raises for other m at :1137 and takes the fold at :1139
- S9 fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/71df350d424c8b92.json#kernel.py:183 `_route_kernel`; `_ES_RANK` at :387, `_ROUTE` at :392, `_ROUTE_QUANT` at :614, `_S1_GATHER` at :619, `_S1_SPLIT` at :624, `_FP4_REDUCE` at :627, `_OVERRIDE` at :634, `_S1_LOCAL` at :642; `_S1_MODULES` at :661 (the stage-1 builder copies `mixed_moe_gemm_2stage_common_vb.py`, `…_vbgs4.py`, `…_vbgs8.py`); `_head_resort_kernel` at :396, its sorted-id descriptor bounded at :431, the bound passed at :1256, rows past the valid count masked at :459 and tiles stored only below it at :461; `_ES_GRID` at :677, `_es_stage2_kernel` at :695; `_fp4_reduce_kernel` (route order, fp32) at :1004; `_plan` at :1283, raising for other m at :1291; `_into_out` at :1306 (routing launch at :1333, m 512's stage 2 through `_flydsl_v2_stage2_wrapper` of `fused_moe_v2_stage2.py` at :1384); `run` at :1396
- S10 fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/4535d4f791472506.json#kernel.py:49 `_S1_KERNEL` (a name in aiter's registry, read by the copied `moe_kernels.py`), stage-2 constants at :51–54 (`_S2_KSPLIT` 2), `_FOLD` at :58, `_S1_TAIL_PARTS` and `_S1_TAIL_FULL` at :61–62, `_TAIL_CPOL` at :65, `_N_TOKENS` at :68; `_cfg` (functools.cache) at :328 with stage 2 `mode="atomic"` at :342; the prelude at :77, zeroing the split-tail counters at :178; `run` raises for other m at :427; the copied builders at :510, the split-tail merge in part order at :555–557
- S11 fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/78f98e0bf6d4c552.json#kernel.py:3194 `_OWNED` (the 2048-token constants, comment at :3192–3193), `run` raising for other m at :3224; `build_moe_module` at :2531 with branches written for 65 experts at :2543–2545 (`CACHE1` at 256 tokens), :2555–2557 (`GROUP_M1`, `GROUP_M2`), :2705–2751 (`PIPE1` :2705, `GATHER_FUSED` :2719, `PIPE2` :2724, `DERIVE_TILES` :2743, `TAILS` :2746, `W2NT` :2751) and :2769; `BM_LARGE` 96 at :2584, `FP4_P` at :2782, scratch per call at :2793; `_reduce_fp4` (slot order, fp32) at :2474
