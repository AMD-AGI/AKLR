---
title: FlyDSL — what is tunable on the baseline aiter's FlyDSL GEMMs, and what is fixed
kind: topic
lever: flydsl_knob_space
ops: [gemm]
dtypes: [bf16, fp8]
regimes: [any]
gens: [gfx950]
flydsl: 0.3.2
sources:
  - aiter/ops/flydsl/gemm_kernels.py
  - aiter/ops/flydsl/kernels/hgemm_dispatch.py
  - aiter/ops/flydsl/kernels/splitk_hgemm.py
  - aiter/ops/flydsl/kernels/small_m_hgemm.py
  - aiter/ops/flydsl/kernels/preshuffle_gemm.py
  - csrc/gemm_a16w16/gemm_a16w16_tune.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/autotune.py
---

# The tunable surface of aiter's FlyDSL GEMMs

## Route here when
You are reading a tuned row for, or racing against, a FlyDSL GEMM that the baseline aiter ships
(`flydsl_kernel_library.md`), and want to know which arguments exist, which values are legal, and which ones
will raise.

**Not here if you are authoring.** In your own kernel, knobs are the last thing you touch;
`flydsl_authoring_method.md` explains why reaching for them early wastes the run.

A path given in full is from aiter's root; after that a file is named without its directory, and
`kernels/…` is `aiter/ops/flydsl/kernels/…`. The signatures and the validation functions named below are the
specification; nothing else is.

## bf16/fp16 HGEMM: `flydsl_hgemm` (`aiter/ops/flydsl/gemm_kernels.py:859`)

| Argument | Default | What it is |
|---|---|---|
| `tile_m`, `tile_n`, `tile_k` | 128, 128, 64 | output tile and K step |
| `stages` | 2 | LDS pipeline depth |
| `split_k` | 1 | K partitions across workgroups, added into C by atomics |
| `block_m_warps`, `block_n_warps`, `block_k_warps` | 2, 2, 1 | waves along M, N and K; `block_k_warps > 1` reduces through LDS inside the workgroup |
| `b_to_lds` | False | B through LDS; False loads B straight into registers and allows only 2 stages (`kernels/splitk_hgemm.py:949`). The catalog always sets True. |
| `kernel_family` | `None` (generic) | `"small_m"` selects the small-M kernel |
| `n_tile_repeat`, `persistent_n_tiles`, `waves_per_eu`, `b_to_lds_unroll` | 1, 1, 0, 0 | small-M kernel only |
| `out`, `bias` | — | `out` in the input dtype; `bias` is `[N]` in the input dtype (`gemm_kernels.py:249-301`) |

**Accepted but without effect, or rejected:** `async_copy` is dropped by the dispatcher
(`kernels/hgemm_dispatch.py:37`), and the kernel always uses 16-byte DMA on gfx950
(`splitk_hgemm.py:128-137`); `pack_n` must be 1; `b_preshuffle=True`, `auto_shuffle_b=True` and
`c_to_lds=True` raise `ValueError`. A kernel name is stricter: `flydsl_kernel_name` fixes its `async_copy` field
to `True` on gfx950 (it is derived from `get_rocm_arch()`) and raises `ValueError` on any other value
(`gemm_kernels.py:42, 98-116`).

**Fixed by the kernel:** bf16 or fp16 in, the same dtype out; the 16×16×32 MFMA; at most 16 waves per
workgroup (`splitk_hgemm.py:113, 133-137`).

**Legality** (`_validate_hgemm_tiling`, `gemm_kernels.py:346-486`, and the kernel's own asserts,
`splitk_hgemm.py:113-186`):
- `tile_k ≥ 32` and a multiple of 32; `tile_m` a multiple of `block_m_warps·16`, `tile_n` of
  `block_n_warps·16`; `tile_k / block_k_warps` a multiple of 32.
- **N ≥ `tile_n` and N a multiple of `tile_n`**: there is no N tail (N=200 with `tile_n` 128 raises; N=192 with
  `tile_n` 64 runs).
- **`split_k` divides K**, and `K / split_k` is a multiple of `tile_k` holding at least `stages` K tiles (the
  last is a bare `AssertionError` from `splitk_hgemm.py:155`).
- `tile_m·tile_k`, `tile_n·tile_k` and `tile_m·tile_n` are multiples of 8 × the workgroup's thread count (a
  whole number, at least one, of 16-byte vectors per thread).
- LDS: `max(stages·(tile_m + tile_n·[b_to_lds])·tile_k·2, block_k_warps·tile_m·tile_n·2)` within the device
  limit.
- A 256×256 tile needs `tile_k` 64, `split_k` 1 and 2 stages (`splitk_hgemm.py:115-116`); with M, N and K all
  ≥ 4096 only that tile, with waves 2×4×1, passes (`selection_filter`, `gemm_kernels.py:304-343`).
- Split-K: `ceil(M / tile_m)·(N / tile_n) ≤ 256` tile counters, checked at launch
  (`gemm_kernels.py:712-724`).

Most violations raise `ValueError` naming the rule; the kernel's asserts raise `AssertionError`, some without
a message.

**The small-M kernel** (`kernel_family="small_m"`, `kernels/small_m_hgemm.py:386`) is bf16 only, fixes
`tile_m` 16 and one wave along M, needs N a multiple of `tile_n` and `split_k` dividing K, and launches
`ceil(M / 16)` row blocks (`gemm_kernels.py:781-791`, `small_m_hgemm.py:403-445, 1402-1407`). Its catalog,
which only enumerated M ≤ 16, is disabled.

### What the tuner searches

`get_flydsl_splitk_hgemm_kernels` (`gemm_kernels.py:589`) enumerates

| Knob | Values |
|---|---|
| `tile_m` | 16, 32, 48, 64, 80, 96, 128, 256, capped at `max(96, align_up(2M, 16))` (:209-214) |
| `tile_n` | 64, 128, 256, dividing N |
| `tile_k` | 64, 128, 256 |
| `stages` | 2–8 |
| `split_k` | 1–13, dividing K with `K / split_k` a multiple of `tile_k` (:216-226) |
| waves (M, N, K) | each of 1, 2, 4, product ≤ 16 |
| `b_to_lds` | True |

The generic kernel only; the small-M family is not generated (:652-685). The offline tuner
(`csrc/gemm_a16w16/gemm_a16w16_tune.py:723-790`) builds FlyDSL candidates for bf16 input without `scaleAB`,
then drops every `tile_m` above `max(M, smallest tile_m)`, every `tile_n` not dividing N, and every split-K
candidate with more than 128 tile counters; it accepts a candidate within 5e-2 of the reference for a bf16
output (:109-112). Measured, at N=6144 and K=4096: M=1, 16 and 24 keep only `tile_m` 16; M=72 keeps 16–64;
M=104 keeps 16–96; `split_k` is 1, 2, 4 or 8 throughout.

The winner is written as `flydsl_gemm{stages}_a{dt}_w{dt}_{out}_t{tm}x{tn}x{tk}_split_k{s}_
block_m_warp{m}_block_n_warp{n}_block_k_warp{k}_async_copy{…}_b_to_lds{…}_b_preshuffle{…}_c_to_lds{…}_{gfx}`
(`flydsl_kernel_name`, `gemm_kernels.py:118-175`; the small-M family adds `_small_m` and its knobs before the
arch). The launched kernel is named like `hgemm_bf16_16x64x128x4_SPK4_W1x2x1_BLDS1_TN_AS1_0`.

## fp8/int8 preshuffle: `flydsl_preshuffle_gemm_a8` (`gemm_kernels.py:1022`)

| Argument | Default | Rule |
|---|---|---|
| `tile_m`, `tile_n`, `tile_k` | required | N a multiple of `tile_n` |
| `use_async_copy` | 0 | DMA of A into LDS; 8-bit inputs only |
| `waves_per_eu` | 0 | ≤ 0 means no occupancy hint |
| `xcd_swizzle` | 0 | XCD-aware tile order |
| `lds_stage` | 2 | 1 or 2 |
| `enable_scheduler` | True | hot-loop scheduling hints |
| `split_k` | 1 | divides K, with `(K / split_k) % tile_k == 0` |

Further rules (`gemm_kernels.py:1005-1060`, `kernels/preshuffle_gemm.py:148-183`): M ≤ 65536; split-K needs at
most 256 tiles and `split_k·M·N ≤ 4·256·32·128` fp32 workspace elements. The kernel can fuse a bias and
ReLU/SiLU/GELU, but not under split-K, and the aiter wrapper does not pass an epilogue.

## fp8 8-wave: `run_gemm_a8w8_bpreshuffle_8wave` (`aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py:265`)

Configured only through its kernel name (`parse_8wave_kernel_name`, :247-262): `block_m`, `block_n`, `block_k`,
the A/W/output dtypes, `waves_per_eu` and `xcd_swizzle`. It has no split-K.

## FlyDSL's built-in autotuner

FlyDSL v0.3.2 ships a Triton-shaped autotuner (`python/flydsl/autotune.py`):

```python
from flydsl import Config, autotune
Config(num_warps=4, waves_per_eu=3, maxnreg=128, **kernel_kwargs)
```

`Config.compiler_opts()` separates the compiler options (`waves_per_eu`, `maxnreg`) from the kwargs injected
into the `@jit` call; `@autotune` times the candidates and caches the winner on disk. In v0.3.2 `maxnreg` reaches
only the binary's options and leaves the code unchanged (`../language/limits.md`).

**aiter does not use it for GEMM.** It sweeps offline and writes the per-shape CSV that `tuned_gemm` reads at
dispatch: decide offline, bake the answer, never search in the serving path.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Invalid N for this kernel` | N is not a multiple of `tile_n` | pick a `tile_n` that divides N |
| `ValueError: Invalid split-K` / `Invalid K for this kernel` | `split_k` does not divide K, or a split is not whole `tile_k` steps | change `split_k` or `tile_k` |
| bare `AssertionError` at compile | a kernel assert, e.g. fewer K tiles per split than `stages` | lower `stages` or `split_k` |
| `ValueError: Split-K semaphore capacity exceeded` | more than 256 tiles with `split_k > 1` | larger tiles or `split_k` 1 |
| The shape still runs another backend | no tuned row, or a row whose name does not parse | a FlyDSL GEMM runs only from a tuned row (`flydsl_kernel_library.md`) |
| Outputs differ between identical runs | HGEMM split-K ≥ 3, or split-K 2 with a bias, adds partials by atomics in arrival order | split-K 1, or 2 without a bias, where runs must match bitwise |
| A tuned row regressed after an upgrade | the CSV was raced on another build or machine | re-tune for the build you deploy |

## Related
`flydsl_kernel_library.md` (which family to call in the first place) ·
`flydsl_authoring_method.md` (what to do when no knob setting is enough) ·
`../../gemm/small_m.md` (choosing these arguments at small M)
