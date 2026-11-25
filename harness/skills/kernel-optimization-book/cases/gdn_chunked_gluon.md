---
title: "GatedDeltaNet over chunked sequences of many tokens in Gluon (Qwen3.5: 16 value heads, conv dim 3072; Qwen3-Next: 8 value heads, conv dim 2048), fp8 group-quantized output — parallel chunk phases, one sequential launch, a parallel epilogue; 5–8 launches a call"
kind: case
ops: [gdn]
dtypes: [bf16, fp8, fp32]
regimes: ["512-16384 tokens in 1-8 sequences (Qwen3.5)", "1024-16384 tokens in 1-16 sequences (Qwen3-Next)"]
accuracy: bitwise
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup over what SGLang runs for the layer, per size, one reference run of the ruler per kernel", value: "0.997 (8193 tokens in 5 sequences) to 2.638 (512 in 1); 0.997 and 1.029 (4097 in 3) are below the bar of these runs", gpu: MI355X, workload: "Qwen3.5 layer, 21 sizes of 512-16384 tokens in 1-8 sequences", date: "2026-11"}
  - {metric: "speedup of the best kernel per size over what SGLang runs for the layer, one later run of the ruler", value: "1.233 (13312 tokens in 13 sequences) to 1.741 (5120 in 1)", gpu: MI355X, workload: "Qwen3-Next layer, 13 sizes of 1024-16384 tokens in 1-16 sequences", date: "2026-11"}
---

# GatedDeltaNet over chunked sequences of many tokens in Gluon (Qwen3.5, Qwen3-Next)

Twenty-six Gluon kernels, a family of the Triton and Gluon kernel library that
[triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps: the GatedDeltaNet (linear attention)
layer of Qwen3.5 and of Qwen3-Next over sequences of many tokens, its recurrence run in chunks on the matrix cores,
the gated RMSNorm and the fp8 group quantization fused into the output epilogue. That page says which family to
start from; [triton/gdn](../triton/gdn.md) §7–12 explains the techniques and names the Qwen3.5 designs by the labels
used below (§ Mechanism). This page has the family's kernels by size, their code, and what was measured.

The kernels and the path they replace:

Packed sequences of the two layers: Qwen3.5, 15 kernels over 21 sizes (512–16384 tokens in 1–8 sequences);
Qwen3-Next, 11 kernels over 13 sizes (1024–16384 in 1–16). Source files: the kernel, a module that zeroes fresh state
slots and, in 9 Qwen3.5 kernels, a module that repacks the projections. The baseline chains `causal_conv1d_fn`,
`fused_gdn_gating`, `chunk_gated_delta_rule`, the norm and the quantization (`per_token_group_quant_fp8` in
Qwen3.5) *(field, E1)*.

## Applies when

- **The op.** The GatedDeltaNet layer of Qwen3.5 (16 value heads, conv dim 3072, contiguous projections) or of
  Qwen3-Next (8 value heads, conv dim 2048, projections grouped by key head), both with 4 key heads and a key and
  value width of 128, over a ragged batch of sequences (`cu_seqlens`), each with its own state slot. Per token: a
  causal width-4 convolution of the Q, K and V channels with the slot's 3-token history, plus `conv_bias`, then SiLU,
  rounded to bf16; q and k L2-normalized and rounded to bf16; β and the decay from the B and A projections; the gated
  delta-rule update of the fp32 128 × 128 state of the (slot, value head), its output rounded to bf16; a gated
  RMSNorm, rounded to bf16; then, per value head, 128 e4m3 codes and one fp32 scale. A call writes `normalized`,
  `values` and `scales` and updates `conv_state` and `delta_state` in place ([triton/gdn](../triton/gdn.md) §1).
- **What it replaces.** SGLang's `Qwen3_5GatedDeltaNet.forward` and `Qwen3GatedDeltaNet.forward` where they take
  sequences of many tokens: the chain of calls named above. A speedup is against that chain.
- **Sizes.** A kernel is written for a fixed list of sizes, a size being a token count with a sequence count, under
  the maximum sequence length its op states or under none (§ Mechanism). Another size has no kernel.
- **Accuracy.** The family is held to `bitwise`: every run on the same inputs gives the same bytes as the first. Each
  output must also stay within its own error bound against the reference ([triton/numerics](../triton/numerics.md) §1).
  No kernel uses an atomic, and five calls on fresh copies of the inputs were byte-identical at each kernel's first
  and last size; the sizes between them were not run *(field, E2)*.

## Counter-signals

- **Two Qwen3.5 sizes are below the bar of these runs.** 8193 tokens in 5 sequences read 0.997 and 4097 in 3 read
  1.029 *(field, E3)*. Each is the smaller of its kernel's two sizes: the kernel's value strip gives the
  sequential launch 256 programs at the larger size, 160 and 192 at these ([triton/gdn](../triton/gdn.md) §8). The
  lowest Qwen3-Next size is 13312 tokens in 13 sequences, 1.233 *(field, E4)*.
- **The Qwen3-Next speedups come from one run.** A repeat measurement of each of the library's 20 Qwen3-Next
  GatedDeltaNet kernels, these 11 among them, read its speedup within 4.7% of that run at the kernel's smallest
  size, but none of the 20 passed the repeat: each failed the accuracy check or exited nonzero *(field, E5)*. In the
  run of the ruler that recorded their accuracy, all 44 sizes of those 20 kernels passed *(field, E8)*. No cause of the
  failed repeat is recorded.
- **Which design a size got is not a selection rule confirmed by measurement** ([triton/gdn](../triton/gdn.md) §8).
  At 8192 tokens in 1 sequence a reverse-summary kernel for 16384 tokens, run there, took 0.73–0.74× the time of the
  size's own kernel, the one entry 1 of § Code is from, in 2 runs, the transplanted kernel's accuracy not checked
  *(field, E9)*. Sizes that share a source differ too: the source of the kernels for up to 1024 tokens read 2.65 at
  512 tokens in 1 sequence and 1.91 at 1024 in 2, that of the reverse summaries for 16384 tokens 1.73–1.74 in 1
  sequence and 1.36 in 2 of up to 8192, in two runs of the ruler other than those of § Result *(field, E9)*. Measure
  candidate designs at the size in question.
- **Branches that no size takes were not run.** The Qwen3.5 kernel for 2049–4096 tokens in 2 sequences has a path
  for more than 4096 tokens that static-asserts a 3072-element row stride of `normalized`, which the
  (tokens, 16, 128) output lacks: compiled at that output's stride, the path fails its `static_assert`. The path
  that the kernel for 4097–8192 tokens in 3–4 sequences keeps for up to 4096 tokens, and the one that the 1025- and
  2048-token kernels keep for up to 1024, do not run at their sizes: no run checked or timed them. The Qwen3-Next
  kernel for 1024 tokens in 1 sequence and 2048 in 2 finds a token's sequence in seven unrolled halvings, right for
  at most 128 sequences: it fails at 129, where eight halvings pass *(field, E10)*.
- **The repack launch is avoidable, and its cost is not measured.** Nine Qwen3.5 kernels copy the contiguous
  projections into the grouped order first, 10 KiB per token each way (estimated); the six MAP and RSUM kernels read
  the contiguous layout directly ([triton/gdn](../triton/gdn.md) §7).
- **What the capture test covers, and what the ruler leaves out.** All 26 kernels were captured into a CUDA graph
  and replayed with new inputs, matching the reference; that does not independently test the flag zeroing, the ragged
  handling or the stream-only synchronization *(field, E11)*, and the zeroing itself was tested on one kernel
  (§ Mechanism). A call whose launches depend on its lengths must stay out of a caller's capture
  ([triton/authoring](../triton/authoring.md) §8): the SGLang integration of these kernels declines the call while a
  stream is capturing, which was checked with mocked SGLang layers, not in a running server *(field, E11)*. Run
  eager, a launch pays about 1.0 µs of host time per tensor argument (measured on toy kernels), which a graph-replay
  timer, the ruler's included, omits *(field, E12)*.
- **The conv history is written after its last reader.** An early write passes a check of the final states and
  shows only in the outputs (§ Mechanism). Where readers and the writer sit in different waves, a barrier goes
  between them. A sequence of one or two tokens keeps old entries, and the zeroing launch clears only the delta
  state: a fresh sequence's reads of the old history are masked with its flag ([triton/gdn](../triton/gdn.md) §7).
- **Scratch is `torch.empty` per call**, so it holds undefined bytes: 173–294 MB of fp32 at 8–16K tokens in the
  summary designs *(field, E10)*. Write every element a later launch reads in the same call, padding rows and
  ragged tails included: a NaN left in old memory survives a zero β or a decay of 1. The 26 kernels passed the
  ruler's accuracy check with poisoned scratch *(field, E11)*.
- **The fp32 state algebra passed its accuracy checks, with no general guarantee.** Of a random and a cancellation
  input for each of two Qwen3.5 kernels, only the random input of the kernel for 512 tokens in 2 sequences kept the
  delta state within a preset limit; that of the kernel for 8193 tokens in 5 exceeded it in max-abs, at 37.3 times
  the error of a GPU fp32 recurrence, and on the cancellation inputs the L2 error reached 721 (8193 in 5) and 1192
  (512 in 2) times that recurrence's *(field, E13)*. Finite inputs only.
- **The K relabel of entry 3 was not timed against the direct conversion in a GatedDeltaNet kernel.** What was
  timed is a toy chain of 256 products on `k_width = 4` operands, whose code in one wave is the relabel's: with 4
  waves and 16 programs per CU it took 1.313× the time of the direct `k_width = 1` conversion, one run *(field, E14)*,
  at 4 programs per CU 0.54–0.89× of it in all five tile and wave shapes tried, one run *(field, E15)*.
- **The swap of entry 6 pads for a register that a VALU instruction wrote.** In test kernels, on an accumulator read
  straight from a bf16 or scaled-fp8 MFMA, a `v_permlane16_swap` behind `s_nop 1` was wrong in 100 of 100 trials, and
  right behind `s_nop 7`, `s_nop 11` or `s_nop 15` + `s_nop 3`, by MFMA opcode *(field, E16)*.

## Mechanism

Every call zeroes the state slots of sequences without an initial state (tested on the Qwen3.5 kernel for 15361–16384 tokens in 8 sequences), and 9 Qwen3.5 kernels repack the projections into grouped order. In launch order at all 34 sizes: prepare, chunk factors, summaries for long sequences, one sequential launch, a parallel output epilogue; 5–8 launches *(field, E17)*. At the 21 Qwen3.5 sizes the conv history left after the call was the last three entries of the old history (zeros when fresh) followed by the sequence's inputs, so a sequence shorter than three tokens keeps old or zero entries *(field, E17)*. Committing the history early in a Qwen3.5 kernel (2049 tokens in 2 sequences) and a Qwen3-Next kernel (3072 in 3, 3072 in 2) changed 12,693–29,318 fp8 values and failed accuracy on all three inputs in each of two executions, while the unchanged kernels passed and the final conv and delta states were unchanged: only the outputs expose the early read *(field, E18)*.

Kernels with their sizes as tokens/sequences, then the op's stated maximum sequence length after the semicolon (∅: none stated).

| Sequences | Qwen3.5 | Qwen3-Next | Sequential pass |
|---|---|---|---|
| up to about 2K tokens per sequence | five kernels: (512/2; 256), (512/1, 1024/2; 512), (1024/1; 1024), (1025/1; 1025), (2048/1; ∅) | one kernel (1024/1, 2048/2; ∅) | 64-token chunks, chunk-start checkpoints, output in parallel per (head, chunk) |
| several sequences | four kernels: (2049/2, 4096/2), (4097/3, 8192/4), (8193/5, 15360/8), (15361/8, 16384/8), all ∅ | two twin pairs: (3072/3, 4096/4, 8192/8, 12288/12, 16384/16; ∅), (13312/13, 15360/15; ∅) | per (sequence, head, value-row strip); the Qwen3.5 pass has 160–256 programs (at 8 sequences: 16 heads × 2 strips of 64 value rows × 8 sequences = 256; at 5 sequences: 160) |
| long sequences, 4096–16384 tokens in 1–4 | maps from an identity seed: (8192/1; 8192), (10000/1; 10000); reverse summaries: (5000/1, 15000/3; 5000), (16384/2; 8192), (16384/2; 10000), (16384/1; 16384) | three twin pairs: (4096/1; 4096), (5120/1, 10240/2; 5120), (16384/4; 5120) | a parallel summary launch per run of chunks (320–864 CTAs), one sequential launch (64–384), a parallel replay of the chunks (320–1024) |

The Qwen3.5 kernels under the labels of [triton/gdn](../triton/gdn.md) §8, which the code comments below use: D the
three kernels for 512–1024 tokens and CKPT the two for 1025 and 2048 (first row); SEQ32L, SEQ32G and SEQ16 the
kernels for 2, 3–4 and 5–8 sequences (second row; SEQ16 has two sources, a for 8193–15360 tokens and b for
15361–16384); MAP the maps from an identity seed and RSUM the reverse summaries (third row; RSUM a for sequences of up
to 5000 tokens, RSUM b the three kernels for 16384 tokens). Kernels of one label and letter run byte-identical code.

- **Techniques.** bf16 MFMA for the Gram products of the bf16-rounded Q and K, fp32 MFMA for the state algebra
  ([triton/gdn](../triton/gdn.md) §9, [triton/numerics](../triton/numerics.md) §5). The Qwen3.5 kernel for 15361–16384 tokens in 8 sequences permutes K inside every 16-block of both operands
  so that an fp32 accumulator becomes the next MFMA's A operand by a trivial `convert_layout` ([triton/gdn](../triton/gdn.md)
  §11, [triton/gluon/layouts](../triton/gluon/layouts.md) §7). Every Qwen3-Next kernel but the one for 1024–2048 tokens has a twin, the same source with the token count
  and the sequence count (in the 13312–15360-token pair also a row total) as constexprs in one and runtime ints in the other: the runtime twin can reuse one binary only across sizes
  whose complete JIT specialization key is the same. Measured, the twins came within 2.4% of each other,
  either way *(field, E4)*.
- **Resources** *(field, E19)*: Qwen3.5 123 compiled kernels, `num_warps` 1, 2, 4 or 8, `waves_per_eu` 2, 4 or 5 in 15, VGPR
  3–256, no AGPRs, LDS 0–32 KiB, spills in 2 (64 VGPRs in a kernel that builds paired chunk tables, 1 SGPR); Qwen3-Next 117
  compiled kernels, 1–4 warps, VGPR 2–316, AGPRs in 5 (20–60), LDS 0–32 KiB, no spills. fp32 16 × 16 × 4 MFMA in 57 compiled kernels
  of each family, bf16 16 × 16 × 32 in 24 of each; fp32 32 × 32 × 2 in 4 (Qwen3.5) and 2 (Qwen3-Next), bf16
  32 × 32 × 16 in 2 (Qwen3.5). All 26 kernels hold both bf16 16 × 16 × 32 and fp32 16 × 16 × 4 MFMAs.

## Code

### 1. The chunked form of the recurrence, as a kernel's docstring states it (Qwen3.5, 8192 tokens in 1 sequence)

The kernels run the recurrence over each sequence in the chunked WY form. The chunk-factor function of the Qwen3.5
kernel for 8192 tokens in 1 sequence (MAP) states it for the 16-token blocks it solves: a unit-lower-triangular
system gives W and U, after which a block's outputs Y and its outgoing state follow from the incoming state S by
matrix products. G is the relative decay and c the cumulative decay; R = Uᵀ − S Wᵀ is the residual; `response` is the
transpose of the block's causal decayed attention, tril(Q Kᵀ ⊙ G)ᵀ (the untransposed matrix has i ≥ j); `Ktail` is K
scaled by each token's decay to the block's end, G(last, j). The docstring's last line is the rule for G: each
relative decay is a product over its own interval, here a multiply `gl.associative_scan` down the columns of
`gl.where(i[:, None] > j[None, :], sg[:, None], 1.0)`, `i` and `j` being the block's row and column token indices and
`sg` its per-token decays.

```python
# docstring of a Qwen3.5 prefill kernel's chunk-factor function: G relative decay, c cumulative decay
    With cumulative decay c and relative decay G, solve
      (I + tril(beta * G * (K @ K.T), -1)) [W, U]
          = [beta * c * K, beta * V].
    Replay uses R = U.T - S @ W.T, then
      Y = S @ (c * Q).T + R @ response,
      S_next = c_last * S + R @ Ktail.
    Products of decays are formed directly, without dividing tiny prefixes.
```

Not this: the relative decay as a ratio of two prefix products, `prefix_i / prefix_j` (the rule and both scan forms:
[triton/gdn](../triton/gdn.md) §10). In a test on chunks of 64 tokens with log-decays uniform in [−6, 0], the ratio
underflowed to NaN or 0 at 60% of the normal-valued entries, where an add-scan of the masked log-decays followed by
one exponential, the form of the Qwen3-Next kernel for 1024 tokens in 1 sequence and 2048 in 2, stayed within 3.8e-6
*(field, E10)*; this kernel's multiply scan was not in that test.

Explained in [triton/gdn](../triton/gdn.md) §1, §10.

### 2. A 16 × 16 unit-lower-triangular inverse by repeated squaring on the fp32 MFMA (Qwen3.5 reverse summaries, 16384 tokens)

The chunk factors need (I + A)⁻¹ with A = tril(β · G ⊙ K Kᵀ, −1), the matrix of entry 1. On a 16 × 16 block the
RSUM b kernels multiply out (I + N)(I + N²)(I + N⁴)(I + N⁸), which is (I − N)⁻¹ for a strictly lower N, since
N¹⁶ = 0, and pass N = −A: the caller negates the Gram term. `lower` is that N, `_fp32_mfma(a, b, acc)` is
`acc + a @ b` on the fp32 16 × 16 × 4 MFMA, `MMA` its accumulator layout in one wave, `rr` and `cc` the block's row
and column indices in that layout. The kernels' 32-token tiles take two such inverses and join them with two more
fp32 MFMAs.

```python
# RSUM b: `lower` is N, the negated strictly lower Gram term of a 16 × 16 block; _fp32_mfma(a, b, acc) is acc + a @ b
power = gl.convert_layout(lower, MMA)
inverse = power + (rr[:, None] == cc[None, :]).to(gl.float32)  # I + N
for stage in gl.static_range(3):
    power = _fp32_mfma(power, power, gl.zeros((16, 16), gl.float32, MMA), MMA)  # N², N⁴, N⁸
    inverse = _fp32_mfma(power, inverse, inverse, MMA)  # (I + power) @ inverse
```

Explained in [triton/gdn](../triton/gdn.md) §9.

### 3. K relabelled inside every 16-block: the accumulator as the next MFMA's A operand (Qwen3.5, 15361–16384 tokens in 8 sequences)

Each chunk step of the per-sequence pass feeds the fp32 MFMA accumulator that holds the state into the next MFMA as
its A operand. This kernel (SEQ16 b) permutes K inside every 16-block, k = 4y + z → 4z + y, on both operands: the
columns of A with the function below, the rows of B with the same reshape on `(rows // 16, 4, 4, columns)` permuted
by `(0, 2, 1, 3)`. The product is unchanged, and the conversion of the relabelled accumulator to the `k_width = 1`
A operand passes `assert_trivial=True`, which fails the compile when a conversion moves data across lanes or waves
(S1); the kernel holds the state K-permuted for the whole sequence (permuting it again lands it in the A layout) and
un-permutes it for the final store. In a probe of its 64 × 128 state tile the direct conversion took 8
`ds_write_b128`, 8 `ds_read_b128` and 16 KB of LDS; relabelled, no LDS access or lane shuffle, only 32
`v_mfma_f32_16x16x4_f32` *(field, E10)*. The per-sequence pass runs one program per (head, strip of 64 value rows,
sequence) (§ Mechanism).

```python
# SEQ16, the columns of A, x of shape (rows, columns): swap the two base-4 digits of k inside every 16-block
    return gl.reshape(
        gl.permute(gl.reshape(x, (rows, columns // 16, 4, 4)), (0, 1, 3, 2)),
        (rows, columns),
    )
```

Explained in [triton/gdn](../triton/gdn.md) §11.

### 4. A launch that zeroes the state slots of fresh sequences from a flag read on the device (all 26 kernels)

State that must start fresh is cleared by a small launch that reads its flag on the device. The first launch of
every call here zeroes the delta-state slot of each sequence that brought no initial state, one program per
(sequence, block of the slot): `STATE` is the state pool, `INDICES` the sequences' slots, `INITIAL` their flags,
`WIDTH` the element count of one slot, `BLOCK` the 1024 elements a program stores, and the host sizes the grid from
shapes alone. The module is byte-identical in all 26 kernels; the excerpt is the copy in the Qwen3-Next kernel for
3072–16384 tokens in 3–16 sequences. It clears the delta state only: reads of a fresh sequence's old conv history
are masked with the same flag.

```python
@gluon.jit
def _initialize_fresh(
    STATE, INDICES, INITIAL, WIDTH: gl.constexpr, BLOCK: gl.constexpr
):
    seq = gl.program_id(0)
    if not gl.load(INITIAL + seq):  # the flag is read on the device, not by the host
        slot = gl.load(INDICES + seq)
        offset = gl.program_id(1) * BLOCK + gl.arange(
            0, BLOCK, gl.BlockedLayout([1], [64], [4], [0])
        )
        gl.store(STATE + slot * WIDTH + offset, 0.0, offset < WIDTH)


def initialize_fresh(state, indices, initial):
    width = state.shape[1] * state.shape[2] * state.shape[3]
    _initialize_fresh[(indices.numel(), triton.cdiv(width, 1024))](
        state, indices, initial, width, 1024
    )
```

Not this: `.item()` or `.tolist()` on the flag to pick the slots on the host, which fails a CUDA-graph capture
([triton/authoring](../triton/authoring.md) §8).

Explained in [triton/authoring](../triton/authoring.md) §6.

### 5. The fused fp8 group quantization at the end of the gated RMSNorm (Qwen3.5 reverse summaries, 16384 tokens)

The output goes to the next block-fp8 linear as e4m3 values with an fp32 scale per 128 values. The last launch of the
RSUM b kernels ends its gated RMSNorm, after `normalized = (...).to(gl.bfloat16)`, with the recipe: store the bf16
output, widen it back to fp32, floor each group's amax at 1e-10, form the scale as a product with `inv_quant_max`,
multiply by one reciprocal per group, clamp, and let the store into the e4m3 pointer convert. That store makes Triton
set MODE.FP16_OVFL at kernel entry, and with the bit set a finite bf16 overflow saturates to ±0x7F7F instead of ±Inf
(measured on probe kernels) *(field, E20)*: a kernel with a Triton fp8 cast can return a finite maximum where the
reference overflows, which the non-finite rule rejects ([triton/numerics](../triton/numerics.md) §1, §2). `row` is a
(token, value head) pair and `col` the 128 values of its group, slices of `BlockedLayout([1, 8], [8, 8], [1, 1], [1, 0])`
at `num_warps=1`, so a group stays inside one wave and its amax needs no LDS; `valid` masks the rows past the last
(token, head) pair; the host passes `quant_max` = 448, `min_quant` = −448 and `inv_quant_max` = `1.0 / 448`. In the
kernel for 2 sequences of up to 10000 tokens this finish matched a reference quantization of its own bf16 output bit
for bit on 100000 random groups *(field, E21)*.

```python
gl.store(Normalized + row[:, None] * 128 + col[None, :], normalized, valid[:, None])
x = normalized.to(gl.float32)
quant_scale = gl.maximum(gl.max(gl.abs(x), 1), 1e-10) * inv_quant_max
encoded = gl.clamp(x * gl.div_rn(1.0, quant_scale)[:, None], min_quant, quant_max)
gl.store(Quantized + row[:, None] * 128 + col[None, :], encoded, valid[:, None])
gl.store(Scales + row, quant_scale, valid)
```

Not this: the scale formed from an amax with no floor. In a probe kernel an all-zero group got scale 0 and code 0xFE
without the floor, the clamp turning the NaN into −448, and a finite, nonzero scale and zero codes with it
*(field, E20)*.

Explained in [triton/numerics](../triton/numerics.md) §2.

### 6. A register bit exchanged with lane bit 4 before a store, with no LDS exchange (Qwen3.5, D kernels at 512 tokens)

The D kernels store each chunk's incoming state in their sequential launch, for the parallel launch that rebuilds the
outputs; at 512 tokens a program owns a stripe of 16 value rows × 128 keys in an fp32 MFMA accumulator (`state`;
`BLOCK_V` is the stripe's value rows) and stores it as bf16 high and low parts. The four `v_permlane16_swap_b32`
exchange registers 0–3 with 4–7 of each 8-register block across lane bit 4, so that K bits 0–2 land in consecutive
registers for 8-value bf16 stores; the `reshape` and `permute` then relabel the axes, and `assert_trivial=True`
asserts that the conversion to `L` moves no data. A swap changes both of its registers, so the 8 outputs are tied to
the 8 inputs (the digits `0` to `7` of the constraints), and with `pack=8` an asm instance takes 8 consecutive
registers of a thread, not 8 consecutive tensor elements *(field, E22)*; in a test kernel this swap matched the
ISA's pseudocode bit for bit *(field, E23)*. The `s_nop 1` at both ends is the kernel's own padding (its docstring:
"Entry and exit delays cover GFX950 hazards."): compiled, the swap reads `v_mov_b32` copies written right before it,
which LLVM leaves unpadded.

```python
gl.static_assert(BLOCK_V == 16)
shuffled = gl.inline_asm_elementwise(
    "\n        s_nop 1\n        v_permlane16_swap_b32 $0, $4\n        v_permlane16_swap_b32 $1, $5\n        v_permlane16_swap_b32 $2, $6\n        v_permlane16_swap_b32 $3, $7\n        s_nop 1\n        ",
    "=v,=v,=v,=v,=v,=v,=v,=v,0,1,2,3,4,5,6,7", [state], dtype=gl.float32, is_pure=True, pack=8)
matrix = shuffled.reshape((16, 2, 2, 2, 2, 2, 2, 2))
matrix = matrix.permute((0, 5, 2, 3, 4, 1, 6, 7)).reshape((16, 128))
L: gl.constexpr = gl.DistributedLinearLayout([[0, 1], [0, 2], [0, 4]], [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64], [0, 8]],
                                             [[0, 16], [0, 32]], [], [16, 128])
matrix = gl.convert_layout(matrix, L, assert_trivial=True)
```

Not this: the same asm where a thread holds 4 fp32. Compiled, it swapped the 4 real registers with 4 unwritten
VGPRs, without a message; `gl.static_assert` pins the shape the asm assumes.

Explained in [triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §6.7.

## Result

- **Speedups** over the baseline above: Qwen3.5 0.997 (8193 tokens in 5 sequences) to 2.638 (512 in 1); 0.997 and 1.029
  (4097 in 3) are below the bar of these runs ([triton/tuning](../triton/tuning.md) §1) *(field, E3)*. Qwen3-Next, in a later
  run of the ruler: the best kernel per size 1.233 (13312 tokens in 13 sequences) to 1.741
  (5120 in 1) *(field, E4)*.

Per size against what SGLang runs for the layer, sizes as tokens/sequences. Qwen3.5, one run of the ruler per size,
five runs in all *(field, E7)*; the kernel is named by its label of § Mechanism:

| Size | Kernel | Speedup |
|---|---|---|
| 512/1 | D | 2.638 |
| 512/2 | D | 2.629 |
| 1024/1 | D | 2.423 |
| 1024/2 | D | 1.889 |
| 1025/1 | CKPT | 2.256 |
| 2048/1 | CKPT | 2.092 |
| 2049/2 | SEQ32L | 1.374 |
| 4096/2 | SEQ32L | 1.452 |
| 4097/3 | SEQ32G | 1.029 |
| 8192/4 | SEQ32G | 1.424 |
| 8193/5 | SEQ16 a | 0.997 |
| 15360/8 | SEQ16 a | 1.473 |
| 15361/8 | SEQ16 b | 1.454 |
| 16384/8 | SEQ16 b | 1.475 |
| 5000/1 | RSUM a | 1.199 |
| 15000/3 | RSUM a | 1.198 |
| 8192/1 | MAP | 1.193 |
| 10000/1 | MAP | 1.188 |
| 16384/1 | RSUM b | 1.741 |
| 16384/2 (maximum sequence length 8192) | RSUM b | 1.339 |
| 16384/2 (maximum sequence length 10000) | RSUM b | 1.444 |

A later run of the ruler read 2.452 at 1024/1 and 2.681 and 1.927 at 512/1 and 1024/2 *(field, E7)*.

Qwen3-Next, one run of the ruler; where the kernel has a twin, the kernel with constexpr counts / its twin with
runtime counts *(field, E7)*:

| Size | Speedup |
|---|---|
| 1024/1 | 1.573 |
| 2048/2 | 1.434 |
| 3072/3 | 1.592 / 1.554 |
| 4096/4 | 1.504 / 1.501 |
| 8192/8 | 1.490 / 1.484 |
| 12288/12 | 1.288 / 1.282 |
| 16384/16 | 1.463 / 1.460 |
| 13312/13 | 1.232 / 1.233 |
| 15360/15 | 1.328 / 1.329 |
| 4096/1 | 1.718 / 1.719 |
| 5120/1 | 1.727 / 1.741 |
| 10240/2 | 1.308 / 1.311 |
| 16384/4 | 1.294 / 1.283 |

The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1. Each size's
chunk size and the programs of its sequential launch: [triton/gdn](../triton/gdn.md) §8.

## Evidence

- E1 [2026-11, CPU (target gfx950), Triton 3.8.0, N=1, each kernel's `run` replayed on the host with its launches recorded, no GPU] the 24 fp8 GEMM sizes one launch each, the N 3072 kernel for M = 6144 rejecting independently strided A and B; at the 89 MoE sizes of 58 kernels the gate-up launch, its grid and the source of its W1; the baselines' calls: MiniMax-M2 routers `tgemm.mm` with the first operand cast to fp32, then `moe_fused_gate` and `fused_moe` at `per_1x32` (MXFP4) or `per_128x128` (FP8); GDN decode a conv update, the recurrence, the norm and the quantization; GDN prefill `causal_conv1d_fn`, `fused_gdn_gating`, `chunk_gated_delta_rule`, the norm and the quantization (`per_token_group_quant_fp8` in Qwen3.5)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, each kernel called 5 times at its first and last size on fresh copies of the inputs, outputs compared byte by byte; sources scanned for atomics] all 44 GDN kernels: no atomic call; 61 kernel-sizes, 305 calls, every repeat byte-identical to the first; the 13 sizes between a kernel's first and last not run (the 18 decode and 26 prefill kernels)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] Qwen3.5 GDN prefill: lowest 0.997 at 8193 tokens in 5 sequences and 1.029 at 4097 in 3, both in one run; highest 2.638 at 512 tokens in 1 (maximum sequence length 512), in another run (16 value heads, conv dim 3072)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] the 11 Qwen3-Next GDN prefill kernels, one later run of the ruler: best kernel per size lowest 1.233 at 13312 tokens in 13 sequences (the runtime twin), highest 1.741 at 5120 in 1 (maximum sequence length 5120, the runtime twin); twins, constexpr / runtime: 4096 tokens in 1 sequence 1.718 / 1.719; 5120 in 1 and 10240 in 2 1.727 / 1.741 and 1.308 / 1.311; 3072 in 3, 4096 in 4, 8192 in 8, 12288 in 12 and 16384 in 16 1.592 / 1.554, 1.504 / 1.501, 1.490 / 1.484, 1.288 / 1.282, 1.463 / 1.460; 16384 in 4 1.294 / 1.283; 13312 in 13 and 15360 in 15 1.232 / 1.233 and 1.328 / 1.329 (8 value heads, conv dim 2048)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel run once on the ruler (Kineto) with its accuracy check, on its own Triton cache, resources and opcodes read from each cached `.amdgcn`; for the Qwen3-Next kernels also the measured speedup at each kernel's smallest size against the run of E6 and E7] the 18 GDN decode kernels: no `v_mfma` and no `v_dot2`; Qwen3.5 recurrence VGPRs 35, 61, 61, 142, 146, 126, 47, 55, 54 at 1, 2, 4, 8, 16, 32, 64, 128, 256 rows, no spills, the 128-row kernel 16 warps at `waves_per_eu` 8; all 26 prefill kernels contain an fp32 MFMA (`16x16x4`, or `32x32x2` in 4 Qwen3.5 and 2 Qwen3-Next kernels) and a BF16 one; SEQ32G's chunk-table builder 190 VGPRs at M = 8192, 256 VGPRs with 64 spilled at M = 4097 under `waves_per_eu=2`, 322 and no spill without it; the 20 Qwen3-Next kernels: every measured speedup within 4.7% of the earlier run's (largest deviation of the ratio from 1, 0.04654), and every one of them failed this run, by an accuracy failure or a nonzero exit (the 44 GDN kernels)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush), speedup over the Qwen3-Next baseline of [triton/gdn](../triton/gdn.md) §1] Qwen3-Next GDN decode, per-size kernels: 1 row 2.443, 2 rows 2.538, 4 and 8 rows (one kernel) 2.370 and 2.506, 16 rows 3.222, 32 rows 3.350, 64 rows 2.462, 128 and 256 rows (one kernel) 1.782 and 1.425, the third design at 128 and 256 rows 1.466 and 1.151; the template at 1–256 rows 2.390, 2.450, 2.370, 2.479, 3.101, 3.091, 2.401, 1.755, 1.457; one run
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush), speedup over the baseline of [triton/gdn](../triton/gdn.md) §1] Qwen3.5 GDN prefill, speedups per size as in [triton/gdn](../triton/gdn.md) §8, from five runs: D 1024/1 2.423, D 512/1 2.638 and 1024/2 1.889, RSUM b 16384/2 at `max_seq_len` 8192 1.339 in the first (a later run of the ruler: D 1024/1 2.452, D 512/1 2.681 and 1024/2 1.927); RSUM b 16384/2 at `max_seq_len` 10000 1.444, MAP 10000/1 1.188, CKPT 1025/1 2.256, RSUM b 16384/1 1.741, RSUM a 5000/1 1.199 and 15000/3 1.198, MAP 8192/1 1.193, SEQ16 b 15361/8 1.454 and 16384/8 1.475, CKPT 2048/1 2.092 in the second; D 512/2 2.629 in the third; SEQ32L 2049/2 1.374 and 4096/2 1.452 in the fourth; SEQ32G 4097/3 1.029 and 8192/4 1.424, SEQ16 a 8193/5 0.997 and 15360/8 1.473 in the fifth; Qwen3-Next GDN prefill, one run, source / twin: snapshot scan 1024/1 1.573, 2048/2 1.434; multi-schedule 3072/3 1.592 / 1.554, 4096/4 1.504 / 1.501, 8192/8 1.490 / 1.484, 12288/12 1.288 / 1.282, 16384/16 1.463 / 1.460; compact factors 13312/13 1.232 / 1.233, 15360/15 1.328 / 1.329; multi-schedule 4096/1 1.718 / 1.719, 5120/1 1.727 / 1.741, 10240/2 1.308 / 1.311; reverse summaries 16384/4 1.294 / 1.283
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (Kineto); accuracy against each output's bound, speedup against what sglang runs for the op] every size with a recorded accuracy result passed: 198 of the 242 sizes of the library's 154 kernels, among them gated attention 9, Qwen3.5 GDN decode fp8 8 and prefill 21, MiniMax-M2 fp8 28, Qwen2-MoE fp8 7; the 44 sizes of the 20 Qwen3-Next kernels, which had no recorded result then, all passed in a later run of the ruler; speedups: the MM64-FP4 kernel with the software e2m1 encoder 1.19 at 128 tokens and 1.194 at 256; the Qwen2-MoE fp8 router + experts kernels with a float64 router dot 1.172 at 1 token (on the VALU) and 1.063 at 32 (on the fp64 MFMA), 1.152 and 1.063 in a second run of the ruler with unchanged sources (image lmsysorg/sglang:v0.5.22-rocm10-mi35x)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the ruler (Kineto) with its accuracy check, on the Qwen3.5 prefill kernels, modified copies and kernels run at another kernel's size, 4 rounds per arm alternated, time = the median kernel-side call time; compiles for gfx950 read from the `.amdgcn`; values given first run / second] 2048/1: CKPT 107.3 / 106.9 µs, RSUM b run at that size 214.2 / 212.9 µs; SEQ16 b over MAP at 8192/1 2.712 / 2.708×, over RSUM b at 16384/1 3.977 / 3.968×; mean call time, the size's own kernel against another design's kernel run there (the transplanted kernels' accuracy not checked): 1024/1 D 67.4 / 67.2 against CKPT 68.3 / 67.9 µs, 4096/2 SEQ32L 193.7 / 192.9 against SEQ16 b 374.5 / 375.5 µs, 8192/1 MAP 495.3 / 497.2 against RSUM b 363.9 / 366.0 µs, 16384/8 SEQ16 b 507.4 / 507.1 against RSUM 651.1 / 652.3 µs; t(4097/3)/t(8192/4) 0.974 / 0.974 (SEQ32G), t(8193/5)/t(15360/8) 0.855 / 0.856 (SEQ16 a), work ratios 0.50 and 0.53; the other value strip, BV 16 at 4097/3 1.317 / 1.317× and BV 32 at 8193/5 1.153 / 1.153× the kernel's own time; SEQ16 b without the `gl.barrier()` before its conv-history write (the one-wave boundary program, 74 VGPRs either way), 10 runs per arm at 15361/8 and 16384/8: no accuracy finding in either arm, `conv_state` exact in all 10 B runs / all 10; SEQ16 b at 16384/8, B/A per pair: next chunk's loads after the math (218 against 222 VGPRs) 1.0196, 1.0211, 1.0199, 1.0233 / 1.0223, 1.0242, 1.0224, 1.0231, `.cg` dropped on the value tile 0.9986, 1.0032, 0.9984, 1.0011 / 1.0023, 1.0007, 1.0006, 1.0012; CKPT at 2048/1, B/A call time per pair: one 128-deep chain (576 MFMAs, 108 against 114 VGPRs) 1.0453, 1.0405, 1.0313, 1.0459 / 1.0386, 1.0426, 1.0426, 1.0421; unroll 1 (64 MFMAs against 576) 1.0675–1.0694 / 1.0629–1.0701; `iterative-ilp` (present in A's LLVM IR) removed 1.1183–1.1334 / 1.1224–1.1294; masked branch forced 1.0237, 1.0166, 1.0124, 1.0202 / 1.0087, 1.0158, 1.0144, 1.0113 (9 dwordx4 and 27 ushort loads, 58 VGPRs in both arms); SEQ32G's chunk-table builder at M = 8192, half tables stored once final 190 VGPRs, all stores at the end 194, no spill (compile only, the same in every session); speedups D 2.646 / 2.654 at 512/1 and 1.913 / 1.906 at 1024/2, RSUM b 1.737 / 1.730 at 16384/1 and 1.359 / 1.360 at 16384/2; SEQ32L at 4096/2: K Kᵀ and Q Kᵀ in each value head (the chunk-factor kernel 160 against 136 MFMAs, 254 against 170 VGPRs), accuracy passed, B/A 0.9777, 1.0235, 1.0225, 1.0221 / 1.0173, 1.0236, 1.0233, 1.0210 (means 1.011 / 1.021); the zero blocks multiplied too (184 against 136 MFMAs, 182 against 170 VGPRs) 0.9655, 1.0063, 1.0060, 1.0077 / 1.0005, 1.0059, 1.0062, 1.0017; SEQ32G at 4097/3 with zeros stored into the packed q/k scratch's tail rows and unmasked loads: builder 24 `global_load_dwordx4` and 3 ushort loads, 190 VGPRs, no spill, against 0, 195 and 256 + 64 spilled, `waves_per_eu` 2 in both, accuracy passed in every B run, B/A 0.8957, 0.8965, 0.9024, 0.8965 / 0.8966, 0.8957, 0.8983, 0.8950 (Qwen3.5 GDN prefill kernels)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, kernels compiled for gfx950, VGPRs, waves per SIMD and instructions read from the `.amdgcn`; probes and the kernels run on the GPU against fp64, torch or the op's reference; one A/B on the ruler (Kineto) of 4 rounds] the 2-row kernel with q read together with k: identical instructions, 61 VGPRs; the Qwen3-Next template, state load early against late: +7 VGPRs at 1 row (65 as built), 0 at 32 (160), +77 at 128, +12 at 256, no spills; accumulate against recompute form in its fused-heads path: 138 against 134 VGPRs at 16 rows, 131 against 132 at 64, 3 waves per SIMD, no spills; decay and β before the conv at 128 rows: 200 VGPRs either way, 2 waves per SIMD; at 32 rows the late state load 10.28 µs against 8.39 µs early, 160 VGPRs each (speedup 3.18 → 2.58; the other 8 sizes 0.997–1.006); one-wave product of two 16 × 16 fp32 accumulators: 69 instructions with `k_width = 4`, 69 relabeled, 79 with two permlane steps (4 swaps), 92 direct, 1024 B of LDS each; relabeled result 9.727e-8 relative to fp64 (`k_width = 4` the same, plain route 1.007e-7); `k_width = 1` from the transposed accumulator 4 `ds_write_b32` + 4 `ds_read_b32`, 1024 B, no `ds_bpermute`, a one-transposition step 2 `v_permlane32_swap`; 64 × 64 tiles, 4 waves: only (transposed, `[4, 1]`, A) and (not transposed, `[1, 4]`, B) at `k_width = 4` pass `assert_trivial`; SEQ16's 64 × 128 `[4, 1]` state: direct 8 `ds_write_b128` + 8 `ds_read_b128`, 16384 B, relabeled no LDS, 32 `v_mfma_f32_16x16x4_f32`; 2^20 random BF16 pairs with exponents within ±63: every fp32 product equals the fp64 one; 16×32 @ 32×16 tiles (4194304 elements), the fp32 operand split into BF16 terms: equal to the correctly rounded BF16 in 58.2% (high term), 99.76% (two RNE terms), 100% (RTZ, RTZ, RNE); 300000 random fp32 values with exponents within ±40: two RNE terms split error at most 7.605e-6 relative, three RTZ terms exact; 10000 chunks of 64 tokens, log-decays uniform in [−6, 0]: the prefix ratio NaN at 4,242,274 and 0 at 4,179,454 of 14,078,262 entries whose fp64 value is a normal fp32, a masked add-scan and one exponential within 3.76e-6 relative; padded rows of the 2-row and 8-row kernels: codes zero-valued, about half of the bytes 0x80, every scale 0x2a7b5124 = 2.23214296e-13; the Qwen3-Next snapshot-scan kernel within the accuracy bounds at 128 sequences, at 129 normalized excess 161, with eight halvings within them at 129; one recurrence step on a random 128 × 128 fp32 state with decay 1 and β 0: no bit changed, v NaN at rows 3 and 17 made those rows NaN; the 1-row kernel with its BF16 core scratch replaced by an fp32 scratch of the BF16-rounded core: outputs bitwise equal on 10 of 10 inputs, 4 of them not padding; with the unrounded core equal on the 6 padding inputs only; fp32 `.to(float8e4nv)`, `.to(float8e5)` and an fp32 store into an e4m3 pointer: one MODE bit-23 write (value 1) at instruction 6, before the first VALU; the 2-row kernel's norm and quant kernel: bit 23 set at instruction 6, before its first `v_cvt_pk_bf16_f32` (instruction 173); with the bit set fp32 → BF16 of finite overflowing values 0x7F7F/0xFF7F and of Inf 0x7F80/0xFF80, with it clear 0x7F80/0xFF80; the Qwen3-Next 1-row kernel's IEEE BF16 rounding next to a Triton fp8 store: no bit mismatch against torch on 60416 values; the template's config rows: IEEE fix-up at 1 row, native convert at 2, 4, 8 and 32, neither at 16, 64, 128 and 256; 16 re-splits of `cu_seqlens` at the 8 multi-sequence prefill sizes with unequal lengths changed no grid; 12000 random ragged batches: chunk slots distinct and below `cdiv(m, BT) + batch`; fp32 `torch.empty` scratch per call at the 8–16K-token summary sizes 173.1 (RSUM a, 15000/3) to 294.0 MB (MAP, 10000/1), MAP's factors there 246.2 MB and maps 35.7 MB (Qwen3.5 and Qwen3-Next GDN kernels; probe kernels)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the prefill graph capture N=3), the ruler (Kineto) and its accuracy check with poisoned scratch, on the kernels named and on all 26 GDN prefill kernels; JIT binaries counted around a CUDA-graph capture (relaxed mode); the sglang integration of the prefill kernels called with mocked sglang layers] a copy of the 8-row kernel that skips the `scales` store of odd rows: a numeric mismatch at the first captured call, 64 non-finite mismatches in `scales`; the unmodified kernel: no finding; the 16-row kernel with the pool size as a `gl.constexpr`: 1, 2, 3, 4 binaries after pool sizes 33, 34, 35 and, under capture, 36; with a runtime pool size 1, 1, 1, 1; no capture error; the 26 prefill kernels, each with the fresh-state zeroing kernel of [triton/gdn](../triton/gdn.md) §7: all timed (capture under `capture_error_mode='global'`, replay with new inputs), no capture, accuracy or mutation finding, exit 0; the mocked Qwen3.5 and Qwen3-Next prefill integrations ran eager, declined the call under capture and ran eager again (call counters 1, 1, 2)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (slope and two-kernel readings repeated in four more runs), Gluon probe kernels on one GPU: chained trivial 1-warp kernels in a CUDA graph, timed by the ruler's timer (Kineto, `GPU_FORCE_QUEUE_PROFILING=1`) and by the bare replay's wall clock, eager launches by the host clock] the reading of the ruler's timer grows 4.062 µs per added kernel (Kineto sum 3.902), a bare replay 1.815 µs (grid 1) and 1.844 µs (grid 256); two chained kernels read 8.246 µs on the ruler's timer; eager host time 1.004 µs per tensor argument, while the ruler's timer reads 4.162–4.178 µs for kernels with 1, 8 and 32 tensors; the repeats, with and without queue profiling forced: the ruler's timer 4.045–4.082 µs per kernel, bare replay 1.766–1.771 µs, two chained kernels 8.25–8.40 µs (toy kernels)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the library kernels on their ops' input sets; for GDN prefill also a GPU fp32 eager recurrence and an fp64 one on the same inputs] Qwen2-MoE router + experts at 1, 2, 4 and 32 tokens: all 32 SQNR comparisons passed, the once-rounded bf16 logits matched at 1–4 tokens and the fp64 MFMA router partials at 32 had zero measured error; GDN: all 168 recorded accuracy comparisons passed; GDN prefill delta state of the Qwen3.5 kernels for 512 tokens in 2 sequences and for 8193 in 5, a random and a cancellation input each: within a preset limit of 4 × the fp32 recurrence's error + 4 × eps32 × the fp64 norm only on the random input at 512 in 2 (L2 error 1.75 and max-abs 1.93 times the fp32 recurrence's), the random input at 8193 in 5 over it in max-abs only (4.69 and 37.3 times); on the cancellation inputs L2 and max-abs error 721 and 6116 times the fp32 recurrence's at 8193 in 5, 1192 and 9171 times at 512 in 2; all four accuracy checks passed
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 8 balanced repeat clusters] a 256-step chain of products on 64 × 16 fp32 tiles, 4 waves, 16 programs per CU: `k_width = 4` operands 337.2 µs against 256.8 µs with the direct `k_width = 1` conversion (1.313×), values correct, largest relative error 2.587e-6 (toy Gluon kernel)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, CUDA-graph replay, 7 rounds, median of the per-round ratios] E14's 256-step chain, `k_width = 4` over direct `k_width = 1` time: at 4 programs per CU 0.543–0.890 in all five tile and wave shapes; 1 wave 32 × 16 0.543, 0.708, 0.803 at 4, 8 and 16 programs per CU (toy Gluon kernel)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, values read back against the host] `v_permlane16_swap` of an MFMA accumulator: wrong in 100 of 100 trials with `s_nop 1`, right in 100 of 100 with `s_nop 7` / `s_nop 11` / `s_nop 15` + `s_nop 3` after 16x16x32 bf16 / 32x32x16 bf16 / 32x32x64 f8; `v_readfirstlane` then `v_readlane` lane select: 0 wrong over 4096 programs × 5 runs with `s_nop 3` and with none; two OPSEL fp4 converts into one register: 0 mismatches over 65536 inputs with `s_nop 0` and without; two setregs of MODE[23]: 0 wrong over 20480 program runs with `s_nop 1` and without (test kernels)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one call per size with its device launches recorded by the profiler] MiniMax-M2 MXFP4: 3 launches at the 10 sizes of 1–16 tokens, the last a down with the weighted combine, the router over K = 3072 in every kernel but MM256-FP4 for 4–16 tokens; 5 at 32–256 tokens; 6–8 at 1024–16384, the down one design in every MM256-FP4 kernel but the one for 2048–4096 tokens, which has its own; MiniMax-M2 FP8: MM256-FP8 3 at 1–16 tokens, its down-and-combine launch 8 warps on a (token, column tile) grid, MM64-FP8 4 at 1–4 and 6 at 8–16, 5–6 at 32–256 with MM64-FP8 at 256 launching a persistent gate-up on 48 × 96 programs, MM256-FP8 6 at 1024–4096 and 7 at 8192–16384, MM64-FP8 8; GDN prefill at all 34 sizes 5–8 launches in the order prepare, chunk factors, summaries for long sequences, one sequential launch, epilogue; at the 21 Qwen3.5 sizes the final conv history the last three entries of the old history (zeros when fresh) followed by the inputs (MoE and GDN prefill kernels)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 executions, 20–100 trials each within an execution] Qwen3.5 GDN decode at 1, 2 and 4 rows: the kernels as they are 20/20 pass each, history shift before the reader launch 20/20 fail each, shift inside the reader 14/20, 5/20, 17/20 fail; the one-owner kernels, Qwen3.5 at 8 rows and Qwen3-Next at 128, 40/40 pass; the Qwen3-Next replicated-lane kernel at 128 rows 100/100 pass with its barrier, 100/100 fail without; a 4-wave kernel with a delayed reader: replicated without a barrier 40/40 mismatch, with a barrier or one owner 40/40 exact (an earlier probe without the delay showed no race); GDN prefill, a Qwen3.5 kernel at 2049 tokens / 2 sequences and a Qwen3-Next kernel at 3072/3 and 3072/2: history committed early fails all three inputs in each execution, 12693–29318 fp8 values changed, final conv and delta states unchanged; the unchanged kernels pass
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, probe kernels on one GPU] against an fp32 mirror of the reference and a bound of two output roundings (2·u_out·|y|): the fp32 precursor quantized instead of the bf16 output put 100000 of 100000 scales outside the bound and changed 282120 of 12.8M codes (repeated once on the same data, same result); all-zero group: scale 2.232e-13 and code 0 with the floor, scale 0 and code 0xFE without it; fused norm + quant 12.22 µs against 22.37 µs for the two kernels at 4096 rows, 13.63 against 22.49 at 131072 (median over 9 rounds of 50 back-to-back launches, no flush, the fused kernel timed first); 12.8M fp32 values within 2 ulps of e4m3 midpoints: x/s against x·fl32(1/s) 624683 codes moved, each one step, the fourth form 0 code and 0 quotient mismatches against IEEE division; bf16 inputs, |x|/amax from 2^-10 to 2: 0 of 491520 pairs moved; `amax / 448` against `amax · fl32(1/448)`: one ulp below for 71 of 128 significands in each of 25 binades, above for none; 100000 groups with divided scales: no element over the bound, |mean|/rms 0.309 against 0.0158; VALU 550 (division per value), 236 (reciprocal per group), 252 (fourth form); without a clamp and with scale = amax/448: x/scale above 448 in 325 of 4096 rows, within 448·(1 + 2^-22), Triton's cast 0x7E at every row's amax, no NaN code; a kernel with a Triton fp8 cast against one without: fp16 multiply 0x7BFF against 0x7C00, bf16 and scaled-e4m3 MFMAs 0x7F7FFFFF against 0x7F800000, fp32 → bf16 0x7F7F/0xFF7F against 0x7F80/0xFF80; 2^20 elements: `gl.fma(p * sx, sw, acc)` and `acc + p * sx * sw` 0 elements differ, `gl.fma(p, sx * sw, acc)` 213955 (probe kernels, not library kernels)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the finish of the Qwen3.5 GDN prefill kernel for 16384 tokens in 2 sequences of up to 10000, on random bf16 groups against an fp32 mirror of the reference quantization] 100000 groups of 1 × 128: 0 scale and 0 code mismatches in each of two runs
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, probe kernels, results read back against the host] at `pack` 2, 4 and 8 on two layouts each instance took consecutive registers, not consecutive tensor elements (0 values off that prediction); `s_mov_b32 $0, 5` then `s_add_u32 $0, $0, $1` on an SGPR input of 3007: 3012 with `=&s`, 10 with `=s` (output and input in one register); two tied outputs, digits 0 and 1: right sums; `v_writelane_b32` with the lane as `n`: a constexpr 37 wrote lane 37, a `gl.static_range` loop and `range(2)` lanes 0 and 1, `range(constexpr=3)` lanes 0–2, a runtime-bound `range` no lane (asm dropped, "constraint 'n' expects an integer constant expression" on stderr); pure `ds_bpermute_b32` asm with its own wait, merged and in a loop-variant and a loop-invariant loop (9 asm blocks each in the compiled loops): 0 mismatches over 100 launches each (64-lane test kernels)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=5 launches × 512 one-wave programs per arm, results read back bitwise] tied outputs `=v×8,0,…,7` (`pack=8`) and `=v×16,0,…,15` (`pack=16`, `is_pure=False`), an empty asm and a `v_add_u32` on each output: 0 mismatches; the [triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §6.7 swap on the 8 outputs: 0 mismatches against the ISA's pseudocode (test kernels)
- S1 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245 `assert_trivial` raises "layout conversion from … is not trivial" with both linear layouts unless the conversion is trivial (:245-248)
