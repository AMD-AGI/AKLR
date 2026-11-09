# KernelForge state of the MiniMax-M3.1 routed MoE large-M campaign (temporary)

What a large-M forge-ablate / forge-finalize needs, taken from b3-3 (smci355-128-b3-3) on 2026-11-09 with the
campaign stopped. Campaign: `fused_moe_swiglu_has_swiglu_limit_shuffled_16x16_model_dim6144_n_experts128_topk4_inter_dim768`,
the 25 Points with n_tokens >= 256 (regime file `../large_m_round5/large_m.json`, `kernel.targets` `[{arch: gfx950}]`).
Best kept round: 5, workspace commit `5bfbea6` (forge mean 1.067x over aiter, every Point >= 1.05x, against the
prepare-time baseline; no paired `aklr validate` yet). The loop stopped in round 6 (GPU guard, 2026-11-06 15:42 UTC)
with about 30 min of search budget left; nothing of round 6 was kept.

## Files and their b3-3 paths

Run dir `R=/home/ruibzhan/aklr-moe-m31-20261105/run-large`, campaign root `W=/home/ruibzhan/aklr-moe-m31-20261105`.

| Here | b3-3 path | Notes |
|---|---|---|
| `workspace-git.bundle` | `$R/workspace/.git` | `git bundle create --all`: `forge-optimize` = `5bfbea6` (rounds 2-5 on `24ef1b7` "forge_task: the task"), `master` = `24ef1b7`, and the incumbent worktree's HEAD (`f25779a`). Restore: `git clone -b forge-optimize workspace-git.bundle workspace` |
| `workspace/kernel.py` | `$R/workspace/kernel.py` at `5bfbea6` | the round-5 kernel (same bytes as `../large_m_round5/kernel.py`) |
| `workspace/kernel.round6_wip.py` | `$R/workspace/kernel.py` as found | round-6 agent edits in progress when the guard stopped the container; not kept, for reference only |
| `workspace/driver.py`, `workspace/program.md`, `workspace/.gitignore` | `$R/workspace/` | written by `forge_task prepare`; the agents may not edit `driver.py` |
| `baseline.json`, `schema.json` | `$R/` | aiter baseline per Point from AKLR validate at prepare (2026-11-06 ~08:10 UTC), the schema as read |
| `forge_loop.sh`, `forge_loop_resume.sh` | `$R/` | KernelForge's start and resume commands as prepare wrote them |
| `forge_experiments/` | `$R/workspace/forge_experiments/` | `run_state.json`, `noise_floor.json`, `campaign_config.json`, `events.jsonl`, `lessons/`, `directions/`, `directions.json`, `orchestration/`, `candidates/`, plus the per-session experiment files (`0482255b.json`, `0c8f5098.json`, `2abb24e1.json`, `7a7d7aba.json`), `best_result.json`, `experience.jsonl`, `optimization_history.md`, `optimization_report.md`, `llm_usage.json` (token counts and cost only). Left out: `analysis/` (305 MB of profiling bundles), `aiter_cache/` (147 MB JIT cache, rebuilt), `dashboard.*`, lock files |
| `run_small_review_guide/` | `$W/run-small/review_guide/` | forge-finalize's `--review-guide` files: `flydsl_code_cleanup.md` (FlyDSL v0.3.2 kernel-code-cleanup skill) and `guides.md` (`harness/skills/aklr-kb/flydsl/guides.md`) |
| `venv_patch/moe_kf_patch.py`, `venv_patch/zzz_moe_kf_patch.pth` | `$W/moe_kf_patch.py`, `$W/venv/lib/python3.12/site-packages/` | raises KernelForge's hard-coded 300 s single-bench timeout to 1800 s (`MOE_BENCH_TIMEOUT_SEC`); install both into the campaign venv's site-packages |
| `venv_patch/zz_kernelforge_venv.pth` | `$W/venv/lib/python3.12/site-packages/` | layers the campaign venv on the shared KernelForge venv (`/home/ruibzhan/.venvs/kernelforge`) |

## Paths written into the files

- `workspace/driver.py`: `AKLR_ROOT = /home/ruibzhan/AKLR-wt-minimax-moe` (the AKLR checkout validate runs in and
  publishes candidates into), `SCHEMA_JSON = $R/schema.json`, `VALIDATE_DIR = $R/aklr_validate`. Edit these three
  for another machine; KernelForge pins the driver's hash, so edit before the first resume and keep the copy in the
  incumbent worktree identical.
- `forge_loop.sh` / `forge_loop_resume.sh`: the python `$W/venv/bin/python`, `--workspace $R/workspace`, `--kernel`,
  `--driver`, `--program-md-file` under `$R/workspace/`, `--baseline-json $R/baseline.json`, and
  `--skill-dir /home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb` (and `.../aklr-kb/flydsl`).
- `forge_experiments/campaign_config.json` and `run_state.json` hold `$R/workspace/...` paths (kernel, driver,
  program.md) recorded at the first start.
- `venv_patch/zz_kernelforge_venv.pth`: `/home/ruibzhan/.venvs/kernelforge/lib/python3.12/site-packages`.

Flags the campaign ran with: `--max-hours 8 --model claude-opus-5-5 --gpu 0 --lanes 1 --no-experience-kb
--no-kb-warmstart --no-pr-kb --null-control-pairs 3 --null-control-every 3` (plus prepare's `--paired-keep
--early-stop`). Environment: image `lmsysorg/sglang:v0.5.22-rocm10-mi35x`, aiter `dbd8bf5` (AKLR_BASELINE_AITER)
installed over the image's, FlyDSL 0.3.2, KernelForge (Hyperloom AKLR branch) `fa94e001b`, one MI355X.

## Hand edit in run_state.json

On 2026-11-06 at about 10:55 UTC, with the loop stopped (coordinator-approved). After a host reboot the resume had
set `best_case_times` to the aiter baseline times, because the starting kernel's baseline bench never completed;
the in-session gate then demanded 1.0x over aiter. Fix:

- measured the torch starting kernel with `driver.py --bench-mode --warmup 10 --iters 30` on the campaign card
  (sum 501.4 ms over the 25 Points);
- wrote those per-Point times into `best_case_times`, `baseline_wall_ms` = 501.40194 and
  `search_start_mean_case_speedup` = 0.014490173973418353; nothing else.

The loop has rewritten `best_case_times` with each keep since; it now holds round 5's times. The noise floor in
`noise_floor.json` (0.0993% at iteration 4, 3 pairs) is the loop's own. The round-1 WIP stopped by that edit and
round 3's interrupted WIP (another reboot) were kept on b3-3 only (`$W/run-large-backup-1031/`,
`$W/run-large-backup-1230/`).

## Provenance

The kernel's FlyDSL stage-1/stage-2 builders and helpers are vendored from ROCm/aiter at `dbd8bf5`
(`aiter/ops/flydsl/kernels/`), Apache-2.0, FlyDSL Project Contributors, with their SPDX headers kept; routing,
quantization and dispatch were written by the campaign's agents (claude-opus-5-5 through KernelForge).

No API key, token or `~/.config` file is included; the files were scanned for key values before the push.
