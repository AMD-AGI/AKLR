# Kernel benchmark admission

Each kernel schema qualifies when at least one **same workload** passes all
correctness checks and its graph-mode baseline speedup lower bound is strictly
greater than 1.02. This restores the pre-refactor per-kernel admission rule while
retaining the current correctness checks and statistical timing bound. When no
baseline exists, a measured correctness pass is sufficient.

Other workloads may fail correctness or performance without rejecting a kernel
that qualifies elsewhere. Their findings and individual measurements remain in
the report and measured schemas; correctness failures are written as
`accuracy: failed` and cannot qualify for runtime dispatch. A kernel with no
qualifying workload fails with `no_qualifying_workload`, including when every
workload is correct but too slow. Candidate build failures, crashes, and timeouts
also exclude only the affected workloads, even when a whole batch stops.
Platform failures and invalid evaluations still block the stage.

The unused `is_kept` and `known_failed` workload-record fields are removed.
The report version remains 2; admission uses this run's measurements.

The audit gate is unchanged. A normal validation pipeline still stops after a
failed stage.
