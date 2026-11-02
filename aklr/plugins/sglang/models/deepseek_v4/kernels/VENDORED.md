# Vendored kernels

`common/` (the HIP device runtime) and `deepseek_v41/gfx950/` (the DeepSeek-V4.1-Flash K1 / K2 / K2i / FFN
launches and their host runner) are SGLang-AMD-Mono's (https://github.com/sgl-project/sgl-amd-mono at
0bce1886519813a8086488e599dfa155a69e74a1, Apache-2.0: `LICENSE`, `NOTICE`), as Infera-Flash vendors them in
`infera_flash/third_party/sgl_amd_mono` at Infera-Flash 2886a33, with
Infera-Flash's kernel changes (each marked `infera-flash:` where it changes a constant):

| Infera-Flash commit | change |
|---|---|
| d86c6fd | borrow from SGLang-AMD-Mono (opt26-opt29) |
| 9ee522e | K1's wq_b weights issued right behind wqkv's; the seam norm's polls in one batch; norms on free CTAs (opt37) |
| caa85c5 | FFN seam slice loads ahead of the all-reduce wait; the FFN norm's output in 16 B stores (opt38) |
| e3a837e | K1's q / kv norms on CTAs without a slice |
| 5733606 | K1's QRDY published once QX8 landed, ahead of the qr / KV record stores (opt46) |
| 99cce4b | spins sleep 4 (not 1) between polls (opt47) |

AKLR's changes: the package path in the imports and in `common/plan.py`'s `_TRACED`, and the aiter FlyDSL
helpers the kernels import (`aiter.ops.flydsl.kernels.{buffer_ops,dpp_utils,kernels_common}`) read from
`common/aiter_flydsl/`, a copy of aiter 285579b6's (the aiter of the v0.5.22-rocm720-mi35x-20261022 image, for
FlyDSL 0.4.0; MIT / Apache-2.0 as each file's header says): AKLR's aiter fork carries older ones (no
`kernel_signature`, FlyDSL 0.3.2's buffer API).

`vllm_mono/` is vLLM's DeepSeek-V4.1-Flash mono layer as Infera-Flash vendors and tunes it at 2886a33
(`infera_flash/third_party/vllm_mono`, Apache-2.0; its `VENDORED.md` lists the origin and AKLR's changes). The
target's layers run it (``../target_vllm.py``) on SGLang's caches and tables; the DSpark draft keeps the
SGLang-AMD-Mono launches.

`deepseek_v41/gfx950/hipops/` is Infera-Flash's own HIP kernels at 2886a33: `csrc/common/{device,torch_utils}.h`
and `csrc/dsv41/{markov.h,markov.hip,quant.h}` as they are there, but for a `C10_HIP_KERNEL_LAUNCH_CHECK` fallback in
`markov.hip` (the rocm10 image's torch names it `C10_CUDA_`). AKLR's:
- `csrc/dsv41/index_q.{h,hip}`: Infera-Flash's `index_q` (`csrc/dsv41/indexer.hip`), its epilogue writing K2i's
  FP4 query, scales and head weights as sglang's `index_q_rope_pack_weights_flydsl` does (`../spec.py`'s
  `_index_q`); its optional gate GEMV (`gate_gemv_wave` of `csrc/dsv41/moe_route.h`) takes a ratio-2 compressor's
  `wkv_gate` (`../compressor_routes.py`).
- `csrc/ops.hip`: Infera-Flash's `markov_step` binding, an `argmax_xchg` binding without its acceptance outputs, and
  `index_q_pack`, registered as their own library.

The sglang glue in `../spec.py` and `../draft.py` is SGLang-AMD-Mono's `deepseek_v41/spec.py` and `draft.py` at
b27d604 (the package in that image), whose runner interface is the same as 0bce188's.
