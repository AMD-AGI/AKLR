# Upstream Monitoring Workflow

1. **Load scope and checkpoint.** Read configured repositories, branches, and
   last examined revisions. If these do not exist, report the missing
   configuration instead of scanning an arbitrary range.
2. **Collect changes.** Inspect commits and pull requests in the new range for
   kernel bodies, dispatch, configuration, interface changes, tests, and
   benchmark evidence.
3. **Filter candidates.** Exclude documentation-only changes and unrelated
   refactors. Preserve source links and claimed evidence without treating claims
   as verified results.
4. **Deduplicate.** Search AKLR, active pull requests, and prior candidate
   records for equivalent operator, algorithm, architecture, and shape work.
5. **Classify and route.** Send generated implementations to
   `port-generated-kernels`. Flag interface-only, correctness, or security
   changes for human triage.
6. **Report and checkpoint.** Rank candidates by supported-workload relevance
   and evidence quality. Update external monitoring state only when the
   scheduled workflow explicitly authorizes that mutation.
