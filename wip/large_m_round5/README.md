# Temporary: MiniMax-M3.1 TP4 routed MoE, n_tokens >= 256 (round-5 forge-loop kernel)

Not a delivered kernel and not loaded by AKLR: `wip/` is outside `aklr/ops/`. This parks the best kernel of
a KernelForge forge-loop campaign so it can be finalized elsewhere.

- `kernel.py`: the campaign's round-5 kept kernel (forge workspace commit 5bfbea6), unreviewed (no forge-ablate,
  forge-finalize or stress run yet). Forge-loop's own driver read it at a mean 1.067x over aiter on the 25
  Points, every Point >= 1.05x, against baseline times recorded at prepare; no paired `aklr validate` yet.
- `large_m.json`: the schema regime file it was tuned on: the 25 Points of
  `fused_moe_swiglu_has_swiglu_limit_shuffled_16x16_model_dim6144_n_experts128_topk4_inter_dim768` with
  n_tokens >= 256, `kernel.targets` = `[{arch: gfx950}]` (the committed definition says MI350X).
- Provenance: the FlyDSL stage-1/stage-2 builders and helpers are vendored from ROCm/aiter at
  `AKLR_BASELINE_AITER` (dbd8bf5, `aiter/ops/flydsl/kernels/`), Apache-2.0, FlyDSL Project Contributors, with the
  SPDX headers kept on each copied block; routing, quantization and dispatch were written by the campaign.
