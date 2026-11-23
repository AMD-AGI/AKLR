# Do not call the reference or the baseline from a kernel

## Rule

A kernel does not import or call its op's reference or baseline. For inputs the kernel does not run faster than
the baseline, leave them out of the schema's `workloads`; do not hand them to the baseline from inside the kernel.

## Bad case

The FlyDSL kernel for `fused_moe@1.0` at model dim 7168, 385 experts (PR #237) ran its own code up to 70 tokens.
Above that it imported `fused_moe_baseline` from `aklr.ops.fused_moe.baseline` and called it, and its schema listed
workloads up to 224 tokens. When the baselines became classes, the import failed and the runtime could not load
any kernel.

## Why it is wrong

- AKLR checks a kernel against its reference and times it against its baseline. A kernel that calls them depends
  on what it is measured against: a change to the baseline breaks the kernel.
- Above 70 tokens the kernel ran the baseline itself, so it could not be faster (0.98x); those workloads were
  listed only to be demoted.

## Do instead

Remove the call and the workloads it served: here the workloads above 70 tokens were removed, and the runtime
falls back to the original implementation for them. Audit rule SA5 now rejects such an import.

## Check

- Does a kernel source import or call the reference or the baseline?
- Does the kernel take another path for some inputs, and are those inputs in its schema's `workloads`?
