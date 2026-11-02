vLLM's DeepSeek-V4.1-Flash mono decode layer (`vllm/models/deepseek_v41/amd/mono`, Apache-2.0; parts adapted
by vLLM from ROCm/ATOM, MIT), from image `vllm/vllm-openai-rocm:nightly-rocm100-5555a2d8f7116b2b4877f66516f4128176cf722a`
(vLLM 0.31.1rc1.dev376+g5555a2d8f). Unchanged but for the package path in its imports, and in `runner.py`:
`DSV41MonoLayer.forward`'s `select` callback (an index source's top-k between K1 and K2), and
`INFERA_FLASH_MONO_TIMELINE=1` (the kernels' own timeline stamps, the last layer's, in `runner.tl`).
Also (infera-flash): `MonoBuild.experts` / `topk` and `MonoLayerWeights.topk` so the FFN launch serves the DSpark
draft's layers (128 experts, top-3; the MoE stages took them already), and the timeline builds stamp the FFN seam
and the MoE stages too (`layer.TL_STRIDE` points a CTA: K2's attention back from 0, the MoE from 16, the seam 24-26).

AKLR took this directory from Infera-Flash `infera_flash/third_party/vllm_mono` at 2886a33 (with
Infera-Flash's kernel changes: opt37-39, opt46-48 and the vLLM-only `00329c1`,
`42f6561`, `931c29d`, `6cf5b17`; `docs/dsv41.md` there). AKLR's changes:
- the package path in the imports; the aiter FlyDSL helpers (`buffer_ops`, `dpp_utils`) read from
  `../common/aiter_flydsl/` as the SGLang-AMD-Mono kernels' do;
- the symmetric peer buffer is AKLR's `../common/peer.py` (`common/peer_memory.py`, which imports aiter's
  `quick_allreduce_int4_ipc`, removed);
- `attention/back.py` `stage_combine`: on an index source it waits for the key table K2 built (`krdy`) and reads
  the key count past the CU's cache, as SGLang-AMD-Mono's combine does; before, a count left from an earlier layer
  could wait for chunks no split publishes (a hang at short contexts, where the layers' counts differ).
