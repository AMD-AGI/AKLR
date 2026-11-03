# Downstream Qualification Workflow

1. **Define the matrix.** Pin the AKLR image, GPU, model, workload, and exact
   SGLang, vLLM, and ATOM revisions required by the current gate configuration.
2. **Establish the baseline.** Choose the approved AITER or prior AKLR runtime
   and keep all non-AKLR inputs identical between baseline and candidate.
3. **Install and probe.** Verify each downstream can import and call the expected
   AKLR interface (`aklr.ops` or the SGLang plugin) before starting expensive
   workloads.
4. **Run correctness.** Exercise representative end-to-end requests, compare
   outputs or accuracy metrics with declared tolerances, and retain failures and
   logs.
5. **Confirm kernel selection.** Use available tracing, logging, or controlled
   mutation to distinguish the intended AKLR path from a passing fallback.
6. **Run performance.** Measure candidate and baseline using the same warm-up,
   concurrency, input/output lengths, parallelism, and repetition policy.
7. **Report per engine.** Mark each engine `PASS`, `FAIL`, `SKIPPED`, or
   `BLOCKED`; one passing engine does not qualify another.
