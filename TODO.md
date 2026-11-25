# TODO

- **aklr/plugins**: kept as on main, `vllm/` included. It still imports the removed
  `aklr.core.contract`, and its tests (`tests/plugins/`, those main added since included) were removed;
  the sglang and vLLM entry points stay on `aklr.core.runtime.dispatch`. Move it to the schema and the
  seam: vLLM's ROCM_AITER_FA route to `unified_attention` is served by the aiter seam adapter now,
  and sglang's MLA absorb `torch.bmm` (`bmm_bf16_nn`) has no seam and no baseline yet.
  Main's #148 (vLLM hook lifecycle, `AKLR_TRACE`) came in with the plugins: its
  `aklr/core/runtime/_vllm_hook_registry.py` and `aklr/core/runtime/dispatch/_trace.py` are as on main,
  and `_trace.py` still imports `aklr.core.contract`. The schema dispatcher does not call it, so
  `AKLR_TRACE=op` logs nothing, and the vLLM entry point keeps its own wrapper instead of going through
  `aklr.plugins.vllm._register`.
- **aiter plugin** (main's #226, needs approval: `aklr/core/runtime/` is protected): the ops are under
  `aklr/plugins/aiter` (main's #327), and core reads only its sub-packages that have a `schema.py`
  (`OpVariants.ops`, `DefaultLayout.paths`). `provider`, `mla_absorb` and `situv2_a4w4_moe` there are as on
  main and move with the other plugins; the two conditions go then. A changed schema file of those two ops is
  outside the default validation scope until then. The `aiter.plugins` entry point is out of `pyproject.toml`
  until then: the provider imports the removed `aklr.core.contract`. It asks an op's api `serving(...)`, the kernel
  `dispatch` would run or the fallback reason, which the schema dispatcher's apis do not have yet, and starts
  AKLR from several threads, which needs one init at a time. `AKLR_AITER_PLUGIN` went with the other plugin
  flags of `RuntimeEnv`. `RunnableMetadata.kernel_ref` is in.
- **Plugin ops main added since** move with the plugins: the GLM-5.2 DSA sparse MLA prefill (#83), whose
  `causal_topk` input distribution lived in the contract semantics and becomes its template's init, and the
  routes of Kimi-K3.1-Pro (#224). `fp8_paged_mqa_logits_verify` is in `aklr/plugins/aiter/mla` with its kernel; no route
  calls it yet, and its seam adapter (on aiter's `deepgemm_fp8_paged_mqa_logits_schedule`) gives the bench its
  baseline only. So do the GLM-5.2 MLA absorb decode ops (#49), Qwen3-MoE's router and experts (#56), the
  Kimi-K3.1 KDA chain verify (#256), `situv2_a4w4_moe` (#261) and the aiter provider's routes of the GEMM, MoE,
  MLA and mHC ops (#271). Of the ops under `aklr/plugins/aiter`, `mla_decode_fwd_cprr` (#259) and `kimi_k3_kda_conv` are
  in with their kernels and a baseline-only seam adapter each: vLLM reaches them through its plugin.
  aiter's `gemm_a8w8_bpreshuffle` is served through the seam.
- **`kimi_k3_kda_conv`'s state**: the op writes `conv_state` in place, and a Standard's check sees the outputs
  only, so no check reads the state a kernel leaves (main compared it exactly). Give the check the inputs
  after the call (needs approval: `aklr/core/schema/` is protected).
- **Compile tripwire**: the bench does not fail a kernel that compiles in a timed call. Main's tripwire, with
  its hooks for FlyDSL 0.5's launch cache (#271), went with the old bench.
- **`tests/core/jit/test_flydsl_cache.py`** (main's #271) reaches `flydsl_cache`'s private functions under five
  `noqa: SLF001`: rewrite it at `install` and `is_needed` on an image with FlyDSL 0.5, where it runs. The cache
  now tells AKLR's kernels by the `flydsl_` package prefix `FlyDSLBuilder` gives them here.
- **CI**: `sglang-test` and `vllm-test` run `tests/seam` and check `aklr/plugins/aiter` untimed. The plugin tests and the
  plugins' timed validation against sglang's and vLLM's own code come back with the plugins. Main's scoping of
  the op reference tests to the changed kernels' contract IDs (#301) is not carried: there are no contract IDs
  and no `tests/plugins/aiter/test_reference.py` to scope.
- **Left on the contract layer**: `harness/scripts/forge_task/assets/aklr_forge_driver.py` and
  `aklr_api_params.py` (with `harness/tests/forge_task/fake_aklr.py`) still import the removed
  `aklr.core.contract`. So does main's kernels report (#319, `benchmarks/kernels/kernels_report`): its test
  and the `publish-kernels` job of `ci.yaml` are out until it reads the schema files, and the site's kernels
  section keeps its last data.
- **Main's #293 and #322 in the ops**: only the kernels are carried. The references stay float64, except
  `biased_grouped_topk`'s, whose kernels rank near ties by main's float32 key. Each op keeps its own `check`:
  main's common tolerance and SQNR bounds are not in. Core's `default_tolerances` differ from main's table
  (float32 1e-5 against 1e-4, float16 1e-3 against 1e-2) and `default_check` has no SQNR (needs approval:
  `aklr/core/schema/` is protected). Main's shorter `bench_cudagraph` (`aklr/core/utils/testing.py`) is not
  carried either.
- **Incumbent serve rule** (needs approval: `aklr/core/schema/` is protected): whether dispatch serves a
  `(schema, workload)` on a target (kernel present, `kernel.build.runs_on(target)`, `workload.qualifies(...)`)
  is written twice, in `SchemaCollection.filter_schemas` and by hand in `IncumbentPool.incumbent`
  (`aklr/core/validation/stages/bench/_admission/_plan.py`), which had left out `runs_on`. Give `Schema` a
  `serves(workload, target, min_speedup, baseline)` used by both, then delete the `WORKAROUND:` `runs_on`
  check in `_plan.py`. Regression test: `test_only_a_base_kernel_that_runs_on_the_target_is_an_incumbent`.
