# Do not hide a slow fallback inside a kernel

## Rule

If a kernel is slow on some inputs or cannot handle them, fix the kernel, or let those inputs fall back to the
original implementation. Do not add a slow path inside the kernel for them.

## Bad case

The DeepSeek-V4 FlyDSL kernel for `fp8_paged_mqa_logits@1.0` could not read a KV cache larger than 2 GiB. After
the contract limit was rejected (see [contract_changed_to_fit_kernel.md](contract_changed_to_fit_kernel.md)),
the agent made the kernel compute those inputs in plain PyTorch instead. The results were correct, but 20-60x
slower than aiter. No test case was larger than 2 GiB, so validation did not see it. The user caught it.

## Why it is wrong

AKLR picks a kernel from how it did on its test cases. Here every test case was under 2 GiB, where the kernel was
faster than aiter, so AKLR used it for larger caches too, where it was much slower.

## Do instead

- If the size is matched exactly, fix the kernel, or leave that size out of the schema's `workloads` so it falls
  back to the original implementation.
- If the size is `any_size`, the kernel is used at every size, so fix the kernel.

Here the size was `any_size`, so the kernel was fixed: 64-bit addressing for caches of 2 GiB or more, the PyTorch
path removed, and test cases larger than 2 GiB added.

## Check

- Does the kernel take a different path for some inputs (PyTorch code, a simpler loop, a call to another
  library)?
- Do test cases cover that path, and was it timed against the baseline?
