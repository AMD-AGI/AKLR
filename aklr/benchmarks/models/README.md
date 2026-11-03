# Model-level micro-benches

Each directory benchmarks one model on sglang: it times every operator on its
own, sums them over the layer stack, and checks the sum against a full forward.
How to run one, and how to read its report, is in that model's README.

| file | holds |
|---|---|
| `<model>/profile_decode.py`, `<model>/profile_prefill.py` | The entry points, one per phase. Copied per model on purpose: each reports its model's own sections. |
| `<model>/model_bench.py` | That model's parts: its layer tree, input recording, batch and request setup, memory accounting, server arguments. |
| `sglang_bench.py` | Code that is the same in every model: forward setup, batch preparation, CUDA-graph and eager timing, MoE parallel layouts, version and table output. |
| `kernel_breakdown.py` | The per-kernel breakdown of an operator's replay, from Kineto. Used by the decode scripts that report kernels. |

Each profile script appends this directory to `sys.path` before importing
`sglang_bench` and `kernel_breakdown`. Code that differs between models stays
in that model's `model_bench.py`.
