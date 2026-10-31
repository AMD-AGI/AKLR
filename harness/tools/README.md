# Tools

This directory is reserved for reusable tools that support the AKLR kernel
lifecycle. Expected examples include kernel integration helpers, shape
inference, benchmarking, and reachability analysis.

Tools should be added only when their behavior and ownership are defined.

- [`forge_task/`](forge_task/README.md): turns one kernel-less schema into a task
  for KernelForge's forge-loop, and puts the kernel it ends with back into AKLR.
