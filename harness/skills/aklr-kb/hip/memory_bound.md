# Memory-Bound HIP Kernels: What Moved add + RMSNorm

The case: residual add + RMSNorm, bf16, hidden 6144, one workgroup of 256 threads per row, against
AITER's `rmsnorm2d_fwd_with_add` on MI355X. Ratios are AITER time over kernel time; the ablations ran
in one process, 5 rounds per arm, arm order shuffled per round, after a 60 s warm-up (§5).

## 1. Try the output stores' cache policy first

Both outputs (`residual_out`, `normalized`) are written once and not read again by the kernel. AITER
stores `normalized` with no cache bits and `residual_out` with `sc0 nt`; storing both with `sc0 sc1`
(system scope; `aux = 17` in `__builtin_amdgcn_raw_buffer_store_b128(v, rsrc, voffset, soffset, aux)`)
carried nearly all of the gain: reverting only those two constants, all else equal, left none of it at
16, 128, 4096 and 16384 tokens and 1.7 of 4.6 points at 1024:

| n_tokens | 16 | 128 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| `sc0 sc1` on both | 1.052 | 1.085 | 1.046 | 1.057 | 1.035 |
| AITER's store bits | 0.996 | 0.997 | 1.017 | 0.992 | 1.002 |

The ISA's store table (CDNA4 Table 50; `../hardware/gfx950/memory_and_sync.md`): system
scope is "Coherent Cache Bypass" in L2 and "Hit LRU" in the last-level cache; wave scope with no bits
is "Hit LRU" in L2, and `sc0 nt` is "Hit Stream". Why bypassing L2 is faster here is inferred, not
measured with counters: the stores leave no dirty lines in the L2s to write back. Load cache bits
were not ablated, and no other op was tried.

## 2. Ablate schedule changes before crediting them

Each change reverted alone, the rest as shipped:

| variant | 16 | 128 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| as shipped | 1.052 | 1.085 | 1.046 | 1.057 | 1.035 |
| `x` / `residual` loads grouped, not interleaved | 1.054 | 1.091 | 1.046 | 1.060 | 1.037 |
| weight loaded before `x` / `residual` | 1.052 | 1.085 | 1.074 | 1.053 | 1.027 |
| weight preparation moved early | 1.048 | 1.084 | 1.052 | 1.056 | 1.036 |
| `s_nop` after each store instead of live store data | 1.048 | 1.086 | 1.057 | 1.055 | 1.034 |

Interleaving the loads, hiding the weight preparation under the LDS read and the store-hazard handling
are within ±1.1 % everywhere; loading the weight first gains 2.7 % at 1024 tokens only and loses at
16384. None of them is where the speedup came from.

## 3. Match AITER bit for bit from its built ISA

AITER compiles with `-ffast-math`, so its source order is not its arithmetic. Disassemble the library
AITER actually built (§4) and reproduce what the ISA does, in the source, since AKLR builds with
`-O3` only. For this kernel:

- squares rounded one by one and summed in element order: `#pragma clang fp contract(off)` keeps the
  compiler from fusing them;
- the same in-wave DPP tree and cross-wave order (`(W0 + W1) + (W2 + W3)`);
- `1/H` as `v_rcp_iflag_f32` through inline asm, which the compiler would otherwise fold to the
  correctly rounded constant;
- epsilon added in double, then rounded to fp32;
- `normalized = (s * r) * (w + 0.0f)` in that order (`+ 0.0f` turns a −0 weight into +0);
- the fp32 denormal mode set at entry to AITER's (flush, `s_setreg` of `MODE.FP_DENORM`), where
  AKLR's build keeps denormals.

Check on enough elements: the source kernel, which differs in its arithmetic, matched AITER at every
element up to 32 rows × 6144 and differed from 64 rows on (2.5e-6 – 8.9e-6 of elements, 1 ulp). The
exact kernel matched both outputs byte for byte at 14 row counts up to 16384, on 32 edge inputs (±0,
denormals, squares that overflow fp32) and under graph replay; its error against float64 was at most
0.996 output ulp. NaN payloads are not guaranteed.

The extra arithmetic was free: the source's shorter arithmetic was within ±0.7 % of the exact kernel at
every row count but 512, where it was 3 % slower.

## 4. Disassemble a built library

```bash
cp lib.so /tmp/lib.so   # llvm-objcopy with no output file rewrites its input in place
llvm-objcopy --dump-section=.hip_fatbin=/tmp/fatbin /tmp/lib.so /tmp/lib_out.so
clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
  --input=/tmp/fatbin --output=/tmp/co
llvm-objdump -d --mcpu=gfx950 /tmp/co                  # code
llvm-objdump -D --mcpu=gfx950 -j .rodata /tmp/co       # kernel descriptors (denorm mode, ...)
```

AITER's JIT modules are under `aiter/jit/` once built (`module_rmsnorm_quant.so` for this kernel).
Diff the instruction counts per kernel (`v_fma`, `v_add_f32`, ...) between yours and AITER's.

## 5. Ablate one change at a time

Generate each variant by a text substitution on the shipped source, build it, check that its outputs
are byte-identical to the shipped kernel's before timing it, and time all arms in one process with the
arm order shuffled every round after a 60 s warm-up (MI355X runs cold clocks ~20 % low for 1 – 2
minutes). Five rounds resolved 0.1 – 0.5 %.

## 6. Store-data hazard

CDNA4 Table 11 (required software wait states): after a 12- or 16-byte store
(`BUFFER_STORE_DWORD_X3` / `_X4`, the FLAT, global and scratch ones too), writing the VGPRs that hold its
data needs 1 wait state, or 2 when a VALU writes them; a `BUFFER_STORE` taking its offset from an SGPR
needs none. Two ways out both gave byte-exact output at the same speed (§2): keep every store's data
registers live to the end of the kernel (an ISA scan found no write to them after their store; the
kernel used 80 VGPRs), or put an `s_nop` after each store.
