---
title: "Profiling a kernel: capture and read a trace, find a regression's commit"
kind: method
ops: [gemm, moe, attention]
gens: [gfx950, gfx942]
flydsl: 0.3.2
---

# Profiling a kernel

How to capture a kernel trace on gfx950, find its stall hotspots and read its cache counters, and how to find the
commit behind a performance regression. FlyDSL's own skills at v0.3.2 are linked, not copied: each section gives the
link and what differs on gfx950 with FlyDSL v0.3.2, or what this repository adds. Paths are relative to this folder.

## Route here when

- You need a kernel trace or its stall hotspots: § capture-kernel-trace, then § kernel-trace-analysis and § Scripts.
- You need the L2 / HBM counters of a kernel: § kernel-trace-analysis.
- A kernel got slower between two commits: § bisect-perf-regression.
- The trace is from gfx942 (MI300X): § gfx942 (MI300X).

## capture-kernel-trace

FlyDSL's guide: https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/capture-kernel-trace/SKILL.md. Read it to
capture an ATT trace with rocprofv3 (kernel discovery, `input.yaml`, debug info for source lines) and, as a separate
run, PMC cache and HBM counters. What this repository adds:

- **§Error Handling** (added row): `Option 'spirv-expand-step' registered more than once!`, then `caught signal 6`
  and a hang. `import torch` alone triggered it, seen once (torch 2.11.0+rocm10.0.0, venv and `/opt/rocm` rocprofv3 alike): wrap
  the run in `timeout -s KILL` and time kernels with `torch.profiler`.

## kernel-trace-analysis

FlyDSL's guide: https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/kernel-trace-analysis/SKILL.md. Read it to
turn an ATT trace into ranked stall hotspots mapped to source lines, for its hotspot patterns and plan template, the
PMC L2 / HBM decision tree, and the MFMA latency and dependency tables. Run its scripts from `analysis/` (§ Scripts
below). Where it differs on gfx950 with FlyDSL v0.3.2, and what this repository adds:

- **§Register pressure check, reading the real counts**: `out_kernel_trace.csv` is not authoritative for registers
  either: rocprofv3's `VGPR_Count` has read half the true value and `Accum_VGPR_Count` 0
  (`gemm/attribution.md` §9). Read `.vgpr_count` and `.agpr_count` from the kernel's dumped ISA
  (`FLYDSL_DUMP_IR=1`, `NN_final_isa.s`). The analyzer's register counts are then a floor, so treat its
  register-based occupancy as an upper bound. The §Error Handling row "PMC `accum_vgpr=0` but kernel uses MFMA"
  has the same fix: the CSV reads `Accum_VGPR_Count` as 0, so read the two counts from the ISA rather than adding
  `VGPR_Count + Accum_VGPR_Count`.
- **§Register pressure check, the warning**: in FlyDSL v0.3.2 `maxnreg` does not cap registers, and a cap set
  through the `amdgpu-num-vgpr` attribute turns into spills (`flydsl/language/limits.md`).
- **§L2 / HBM efficiency analysis**: a 32B fraction ≈ 0% means full-size requests, not full 64B cache lines (the
  gfx950 cache line is 128 B: `gemm/datapath.md` §6). Judge the achieved bandwidth against a stream
  bandwidth measured on the same machine with the same read/write ratio, not against the theoretical HBM peak
  (`gemm/measurement.md` §7).

## bisect-perf-regression

FlyDSL's guide: https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/bisect-perf-regression/SKILL.md. Use it
when a kernel got slower between two commits and a benchmark command prints the metric: it bisects the range with
git and reports the commit that caused the regression. This repository adds nothing to it.

## Scripts

`analysis/hotspot_analyzer.py` and `analysis/pmc_l2_analyzer.py` are this repository's copies of the two scripts that
FlyDSL's kernel-trace-analysis ships in `scripts/`. Where FlyDSL's skills run `${CLAUDE_SKILL_DIR}/scripts/<script>`
or `kernel-trace-analysis/scripts/<script>`, run them from this folder with the same arguments:

```bash
python analysis/hotspot_analyzer.py <dispatch_dir> --topk 15 --mode both
python analysis/hotspot_analyzer.py <dispatch_dir> --topk 5 --mode src --detail --context 4
python analysis/pmc_l2_analyzer.py <l2_counters.csv> <ea_counters.csv> --kernel <kernel> --ideal-gb <GB> --ea-channels 2
```

- `hotspot_analyzer.py` takes every trace to be gfx950: it reports `gfx950 (CDNA4)` and its LDS limiter uses 160 KB
  per CU, where FlyDSL's copy detects the architecture from the trace's MFMA instructions (a gfx942 trace: § gfx942
  (MI300X) below). Like FlyDSL's copy, it reads accum VGPRs, LDS, SGPRs and workgroup size from the
  `out_kernel_trace.csv` next to the dispatch dir; for registers read the ISA instead (§ kernel-trace-analysis
  above). Otherwise the two copies differ only in formatting and comments.
- `pmc_l2_analyzer.py` differs from FlyDSL's copy only in formatting: same arguments, same output. Read its
  "~0% = full 64B lines" note as full-size requests (§ kernel-trace-analysis above).

## gfx942 (MI300X)

FlyDSL's skills at v0.3.2 cover gfx942 (CDNA3: MI300X, MI308) next to gfx950. They are linked here, not copied, and
this repository has not re-measured them on gfx942.

| You need | Read (FlyDSL v0.3.2) |
|---|---|
| gfx942's occupancy inputs (64 KB LDS, 256-byte LDS blocks), how FlyDSL's analyzer tells gfx942 from gfx950, a PA decode register example | kernel-trace-analysis [Register pressure check](https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/kernel-trace-analysis/SKILL.md) |
| The PA decode L2 / HBM worked example on gfx942, and the hang that multi-pass counter collection caused there | kernel-trace-analysis [L2 / HBM efficiency analysis](https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/kernel-trace-analysis/SKILL.md) |
| fp8 MFMA latencies on gfx942 | kernel-trace-analysis [MFMA latency reference](https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/kernel-trace-analysis/SKILL.md) |
| One hardware pass per PMC job (multi-pass collection hung gfx942) | capture-kernel-trace [PMC Mode](https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/capture-kernel-trace/SKILL.md) |

**Analyze a gfx942 trace with FlyDSL's `scripts/hotspot_analyzer.py`, not `analysis/hotspot_analyzer.py`** (`curl -sSfL
https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py
-o hotspot_analyzer.py`). Ours labels
every trace gfx950 and sizes LDS at 160 KB. FlyDSL's takes a trace with none of `v_mfma_scale_f32_*`,
`v_mfma_f32_16x16x128_*` and `v_mfma_f32_32x32x64_*` as CDNA3, reports it as `gfx942 (CDNA3)` and uses
`lds_total = 65536` (64 KB) in its LDS limiter (`scripts/hotspot_analyzer.py:355-356`, `:422` at v0.3.2).
