# TODO

- **aklr/plugins**: kept as on main, `vllm/` included. It still imports the removed
  `aklr.core.contract`, and its tests (`tests/plugins/sglang/`, `tests/plugins/vllm/`) were removed;
  the sglang and vLLM entry points stay on `aklr.core.runtime.dispatch`. Move it to the schema and the
  seam: vLLM's ROCM_AITER_FA route to `unified_attention` is served by the aiter seam adapter now,
  and sglang's MLA absorb `torch.bmm` (`bmm_bf16_nn`) has no seam and no baseline yet.
  Main's #148 (vLLM hook lifecycle, `AKLR_TRACE`) came in with the plugins: its
  `aklr/core/runtime/_vllm_hook_registry.py` and `aklr/core/runtime/dispatch/_trace.py` are as on main,
  and `_trace.py` still imports `aklr.core.contract`. The schema dispatcher does not call it, so
  `AKLR_TRACE=op` logs nothing, and the vLLM entry point keeps its own wrapper instead of going through
  `aklr.plugins.vllm._register`.
- **Incumbent serve rule** (needs approval: `aklr/core/schema/` is protected): whether dispatch serves a
  `(schema, workload)` on a target (kernel present, `kernel.build.runs_on(target)`, `workload.qualifies(...)`)
  is written twice, in `SchemaCollection.filter_schemas` and by hand in `IncumbentPool.incumbent`
  (`aklr/core/validation/stages/bench/_admission/_plan.py`), which had left out `runs_on`. Give `Schema` a
  `serves(workload, target, min_speedup, baseline)` used by both, then delete the `WORKAROUND:` `runs_on`
  check in `_plan.py`. Regression test: `test_only_a_base_kernel_that_runs_on_the_target_is_an_incumbent`.
