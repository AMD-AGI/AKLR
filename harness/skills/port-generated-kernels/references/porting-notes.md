# Porting Notes

Traps met turning agent-generated kernels into AKLR schemas, and what settled each.

## Know what actually ran

- **Triage on AKLR's Points first.** Generated kernels are often specialized to the source's
  workload. A MoE kernel whose fast path (the routing sort folded into stage 1) was enabled only for
  token counts 6, 12, …, 42 ran at 1.00 – 1.06× the baseline on AKLR's 8 – 64 Points as shipped, and 1.09 – 1.15×
  once the fast path was enabled for them. Read every environment gate and per-shape table before
  timing.
- **Confirm the fast path ran.** Sources wrap optimizations in `try` / `except` that fall back to the
  original silently. Check the compiled kernel's symbol name (generators often encode the variant in
  it) in a trace, in the source's run and in the port's.
- **Take the knob values that ran**, not a guess: the environment defaults the source shipped with,
  cross-checked against what its evidence encodes (a kernel symbol such as `..._hpf2_..._bef3_...`).

## Make it self-contained

- **Freeze generated code.** When the source builds kernels at run time (text-cloning another
  module and `exec`-ing it, monkeypatching), run it once, keep the generated files, and prove they
  are what ran: the hash the source itself uses (often in the generated file's name), and a
  definition-by-definition AST comparison. Audit the frozen file itself as a negative control: it
  should be rejected (one was, with 21 findings: `os` and `aiter` imports, environment reads, a
  `__name__` write), and the vendored copy made from it should pass.
- **Vendor only what is reached.** Copy the helper definitions the kernel uses, import siblings
  relatively (SA4), and compare each kept definition's AST with the upstream file. A helper's `re`,
  `os`, `json` or `builtins` import is outside SA4's list: replace the use with plain code and test it
  against the original on real inputs.
- **Knobs become constants**, and every `os.environ` read goes (SA5, SA8).
- **Scratch**: one `torch.empty(n_bytes, dtype=torch.uint8, device=...)` per call, sliced at aligned
  offsets (256 B), replaces a source's persistent workspace. Drop a `zero_()` only after showing each
  buffer is written before it is read.
- **No cache across calls.** Besides breaking SA, a weight cache keyed on one tensor misses on every
  timed call (`read-validation`, ruler.md); prepare in the kernel or not at all.
- **Compile functions** are module-level `@functools.cache` functions of shapes, dtypes and
  constants; a kernel specialized on a size (a token count as a compile-time constant) compiles once
  per Point.

## FlyDSL v0.3.2

- `flyc.compile(fn, *args)` runs the kernel once with `args` before it returns; do not call the
  result again on that call.
- The JIT disk cache key includes closure scalars (`int`, `float`, `bool`, `str`, `None`, `tuple`,
  `Enum`), so variants of one `@flyc.jit` factory built with different constants get their own
  entries; renaming a launcher through `__name__` (rejected by the audit) is not needed.

## Determinism and the gate

- A `bitwise` contract (every GEMM but `gemm_a8w8_blockwise_scaled`) rejects atomic split-K. A
  fixed-order two-stage split (fp32 partials in scratch, a second kernel summing them in order)
  costs one more graph node: it paid at m = 64 of a K = 6144 decode GEMM and lost at m = 16 and 32.
- For a `bounded` contract with atomics, call a port numerically equivalent to its source when their
  cross-SQNR lies within the port's own run-to-run SQNR (measured: 48.5 – 51.5 dB across, 48.4 –
  51.9 dB run to run).
- Validate with `--base origin/main` and name only the new schema: without `--base`, every named
  kernel is audited in full, and most committed kernels predate the audit rules.
