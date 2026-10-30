# AITER Synchronization Workflow

1. **Identify the candidate.** Pin the AITER commit and determine the affected
   API, kernel, architecture, dtype, shape range, and claimed improvement.
2. **Check AKLR.** Find the corresponding AKLR path and determine whether the
   change is already present, conflicts with generated kernels, or affects only
   an unsupported regime.
3. **Establish the incumbent.** Record the current AKLR implementation selected
   for the relevant workload and verify its correctness before measuring it.
4. **Reproduce the gain.** Run the AITER candidate and AKLR incumbent with the
   same hardware, software, inputs, warm-up, repetitions, synchronization, and
   aggregation. Treat noise-sized or irrelevant gains as no gain.
5. **Port only the useful delta.** Preserve AKLR interfaces and stronger
   implementations. Integrate the AITER path only for regimes where the
   evidence supports it, then repeat correctness and performance verification.
6. **Decide.** Use `PORT` for a verified relevant gain, `REJECT` when the gain
   is absent or redundant, and `ESCALATE` for a correctness or security issue
   outside this performance-sync rule.

Every decision should be traceable so a later monitor run does not repeatedly
evaluate the same rejected commit without new evidence.
