# Do not change a contract to fit a kernel

## Rule

A contract describes the op: what the caller passes in and what it gets back. If a kernel cannot handle some
inputs, fix the kernel. Do not change the contract to exclude those inputs.

## Bad case

The DeepSeek-V4 FlyDSL kernel for `fp8_paged_mqa_logits@1.0` used 32-bit offsets, so it could not read a KV cache
larger than 2 GiB. Instead of fixing the kernel, the agent added a size limit (`_BYTE_LIMIT`) to
`aklr/ops/mla/contract.py`. The user caught it before PR #96 was opened.

## Why it is wrong

All kernels of the op, its baseline and its reference share the contract. The limit came from one kernel, not from
the op, so it took those inputs away from every kernel and hid the kernel's bug.

## Do instead

Fix the kernel so it handles every input it can be given. Here the kernel switched to 64-bit addressing for caches
of 2 GiB or more, and test cases larger than 2 GiB were added. Some inputs can fall back to the original
implementation instead ([slow_fallback_inside_kernel.md](slow_fallback_inside_kernel.md)). If neither works, tell
the user and ask; do not change the contract.

## Check

- Would this limit still be needed if this kernel did not exist? If not, it belongs to the kernel.
- Does a contract comment explain how a kernel works (32-bit offsets, LDS size)?
