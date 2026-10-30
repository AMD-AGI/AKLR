# Kernel Pull Request Review Workflow

1. **Define the diff.** Identify kernel bodies, host integration, tests,
   benchmarks, generated artifacts, and unrelated changes. Resolve the claimed
   support and performance target.
2. **Review integration and preview quality.** Check meaningful names,
   provenance, placement, public reachability, dispatch keys, fallback, and
   whether the host diff is easy to distinguish from generated code.
3. **Review correctness.** Run the smallest useful tests first, then
   representative and edge-case comparisons with a trusted reference. Include
   unsupported inputs and verify that failures or fallbacks are explicit.
4. **Review runtime safety.** Run available sanitizer or memory-checking paths
   appropriate to the kernel and inspect synchronization, bounds, aliasing,
   dtype conversion, and error propagation.
5. **Review performance.** Reproduce baseline and candidate in the same
   environment, confirm the selected kernel, include warm-up and repeated
   measurements, disclose every tested case, and preserve unfavorable results.
6. **Audit reward hacking.** Check for weak baselines, benchmark-only dispatch,
   cherry-picked shapes, hidden skips, swallowed failures, cache asymmetry,
   timeout bias, missing synchronization, or summaries that hide regressions.
7. **Issue a verdict.** Findings come before summaries. State what was tested,
   what could not be tested, and the exact evidence needed to clear blockers.
