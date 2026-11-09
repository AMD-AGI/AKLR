# Program: implement fused_moe_swiglu_has_swiglu_limit_shuffled_16x16_model_dim6144_n_experts128_topk4_inter_dim768 in flydsl

## Objective

Make `run` in `kernel.py` faster than AKLR's baseline on as many of the 25 workloads below as
possible, keeping every one of them correct: a candidate that wins on time and loses a workload is rejected.

## Contract

- `fused_moe`: Route each token to its top-k experts, run both projections, combine.
- Attributes: `{"activation": "swiglu", "quant_type": "per_1x32", "doweight_stage1": false, "gate_mode": "separated", "has_expert_mask": false, "has_swiglu_limit": true, "act_quant": 4}`. Inputs: `hidden_states`, `w1`, `w2`, `topk_weights`, `topk_ids`, `w1_scale`, `w2_scale`, `swiglu_limit`. Outputs: `out`.
- Fixed axes: `model_dim=6144, n_experts=128, topk=4, inter_dim=768, gate_up_rows=1536, model_blocks=192, inter_blocks=24`. Varying axes: `n_tokens`.
- `run` takes AKLR's call keys as keyword arguments and writes each output into its preallocated argument;
  it returns nothing and allocates no output.
- `kernel.py` may import modules beside it relatively (`from .helpers import f`): those modules are part of the
  kernel AKLR publishes; a file `kernel.py` does not reach, such as a probe, is not.
- AKLR compares the outputs with `{"max_violation_fraction": 0.001, "determinism": "bounded", "element_order": "fixed"}`.
- Every shape, dtype, layout and error bound is in the schema: `/home/ruibzhan/aklr-moe-m31-20261105/run-large/schema.json`.
- What a kernel may and may not do (the audit, the ABI, determinism) is AKLR's rule:
  `/home/ruibzhan/AKLR-wt-minimax-moe/aklr/ops/README.md`.

## Workloads

| case | `n_tokens` |
|---|---|
| `n_tokens=256` | 256 |
| `n_tokens=272` | 272 |
| `n_tokens=288` | 288 |
| `n_tokens=304` | 304 |
| `n_tokens=320` | 320 |
| `n_tokens=336` | 336 |
| `n_tokens=352` | 352 |
| `n_tokens=368` | 368 |
| `n_tokens=384` | 384 |
| `n_tokens=400` | 400 |
| `n_tokens=416` | 416 |
| `n_tokens=424` | 424 |
| `n_tokens=432` | 432 |
| `n_tokens=448` | 448 |
| `n_tokens=464` | 464 |
| `n_tokens=480` | 480 |
| `n_tokens=496` | 496 |
| `n_tokens=512` | 512 |
| `n_tokens=576` | 576 |
| `n_tokens=640` | 640 |
| `n_tokens=3112` | 3112 |
| `n_tokens=3816` | 3816 |
| `n_tokens=10000` | 10000 |
| `n_tokens=13152` | 13152 |
| `n_tokens=32768` | 32768 |

## Where the numbers come from

`driver.py` judges correctness with `python -m aklr validate` and times the candidate with AKLR's graph
timer on AKLR's timing inputs. It is the measurement boundary and is not editable.

A score is AKLR's baseline time over the candidate's, per workload, averaged over the workloads: above 1.0 is faster than the baseline on average, not on every workload.
