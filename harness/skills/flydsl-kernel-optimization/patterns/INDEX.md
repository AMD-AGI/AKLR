---
kind: index
---

# Patterns

Symptom pages: from what a trace or counter shows to the techniques worth trying.

<!-- generated:start (tools/kb_index.py) -->
| Page | Signals | Ops | Basis |
|---|---|---|---|
| [Small-M launch floor: a decode row bound by launch, reduction and latency, not by reading B](launch_floor_small_m.md) | M ≤ 256 on a row bound by launch, reduction and memory latency, a large per-tile intercept F in a K-sweep fit, one CUDA-graph node or an empty kernel costs a large share of the measured time, s_load then lgkmcnt(0) in front of the first B load, .amdhsa_user_sgpr_kernarg_preload_length = 0 | gemm | source-semantics |
| [LDS bank conflicts and LDS-side stalls](lds_bank_conflicts.md) | SQ_LDS_BANK_CONFLICT above the hand-computed floor, stall on the ds_read / ds_write instructions themselves, s_waitcnt lgkmcnt(0) stall right after a ds_write, SQ_WAIT_INST_LDS, SQ_LDS_IDX_ACTIVE, ds_read2 offsets that are multiples of 64, ds_read2_b32 where a ds_read_b128 was meant | gemm | source-semantics |
| [Low CU utilization: too few workgroups for 256 CUs, or a mostly empty last dispatch round](low_cu_utilization.md) | tiles < NCU (256 on gfx950), last_round < slots in the dispatch-round arithmetic, time flat across M while the launched grid is unchanged, launched grid read from a kernel trace | gemm | source-semantics |
| [Memory-bound: the kernel is limited by global memory traffic](memory_bound.md) | bytes moved / wall time near the stream bandwidth measured on the same box, wall time unchanged when MFMA work is added (compute-additivity probe), s_waitcnt vmcnt(0) in front of the MFMAs, VMEM-load stalls (hotspot_analyzer.py), MemUnitStalled | gemm, moe | source-semantics |
| [Register pressure: VGPR / AGPR spills, or occupancy below the plan](register_pressure.md) | vgpr_spill_count > 0, scratch_ loads and stores in the hot loop, v_accvgpr_read / v_accvgpr_write in the hot loop, .vgpr_count + .agpr_count above a reachable occupancy tier, .private_segment_fixed_size > 0, one wave per SIMD | gemm | source-semantics |
| [Shape divergence: one kernel or configuration wins on some shapes and loses on others](shape_divergence.md) | per-shape deltas of opposite sign under one change, geometric mean flat while single cells move, race winner flips between cleared-cache re-runs, a tuned row serving every M that pads to it | gemm | source-semantics |
<!-- generated:end -->
