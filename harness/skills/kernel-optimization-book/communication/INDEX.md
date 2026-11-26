---
title: "Communication between GPUs: EP against TP at a few tokens per GPU, and reading a collective in a trace"
kind: index
gens: [gfx950]
---

# Communication

The collectives of a multi-GPU step: the allreduce that joins TP's partial results, and the all-to-all dispatch
and combine that move tokens to their experts and back under EP. One kind of kernel moves data between GPUs
inside the op: a fused EP MoE, whose FlyDSL kernels, taken from aiter's `MegaMoEV2`, dispatch, run both GEMMs
and combine in one launch per rank. The allreduce and standalone dispatch/combine kernels are aiter's (custom
allreduce, the FlyDSL EP dispatch/combine) and RCCL's. These pages are for choosing a parallel layout and for
reading a collective's time in a trace. Design pages for kernels that move data between GPUs, the fused EP MoE's
included, go here.

| Question | Read |
|---|---|
| EP or TP for a MoE layer at a few tokens per GPU (as in decode); what EP's dispatch pays for, and which changes moved it | `ep_dispatch_combine.md` |
| A trace gives the allreduce a large share of the step | `../pitfalls/nograph_trace_inflates_collectives.md` |
| aiter's FlyDSL dispatch/combine and `MegaMoEV2` kernels: files, entry points, tuning tables | `../flydsl/kernels/flydsl_kernel_library.md` |
| The MoE GEMMs on either side of the dispatch | `../moe/INDEX.md` |

## Elsewhere

- RCCL's environment variables (protocol, channels, algorithm):
  https://rocm.docs.amd.com/projects/rccl/en/latest/api-reference/env-variables.html
- aiter's custom allreduce: `csrc/include/custom_all_reduce.cuh` in ROCm/aiter.
