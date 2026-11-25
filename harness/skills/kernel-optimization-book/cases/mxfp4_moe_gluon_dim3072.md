---
title: "MiniMax-M2 MoE layer with MXFP4 experts in Gluon, model dim 3072, top-8 of 256 routed experts, all 256 on the rank at intermediate dim 512 or 64 of them at 1536 — an fp32 VALU router and three launches at 1–16 tokens, job queues from relaxed atomic tickets at 32–256, a ticket or wave-ballot sort before the expert GEMMs from 1024"
kind: case
ops: [moe, fused_moe, grouped_gemm, gemm, topk]
dtypes: [mxfp4, fp32, bf16]
regimes: ["tokens 1-16", "tokens 32-256", "tokens 1024-16384"]
accuracy: bounded
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup over SGLang's path on aiter for the layer, per size, one reference run of the ruler per kernel", value: "MM256-FP4 1.178 (256 tokens) to 3.444 (1); MM64-FP4 1.190 (128) to 3.633 (1)", gpu: MI355X, workload: "15 token counts per rank shape, 1 to 16384; MM64-FP4 at expert offset 192", date: "2026-11"}
---

# MiniMax-M2 MXFP4 MoE layer in Gluon: a rank of all 256 experts or of 64 (model dim 3072, top-8 of 256)

Twenty-three Gluon kernels for one op, MiniMax-M2's MoE layer with MXFP4 experts on one rank: a family of the Triton
and Gluon kernel library that [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps. That page
says which family to start from; [triton/moe](../triton/moe.md) explains the techniques and names kernels by the
labels used below: MM256-FP4 on a rank that holds all 256 experts, MM64-FP4 on a rank that holds 64 of them, then the
token count (`MM256-FP4/m16384`; a range, as in `MM64-FP4/m128–m256`, is one kernel written for several token
counts). The MXFP4 quantization of the two expert-GEMM inputs is explained in
[triton/numerics](../triton/numerics.md) §3, scratch and graph capture in [triton/authoring](../triton/authoring.md)
§6 and §8. This page has the family's kernels by size, their code, and what was measured.

The layer:

MiniMax-M2's MoE layer on one rank: an fp32 router (sigmoid plus correction bias, top-8 of 256, renormalized), MXFP4 experts, no shared expert. MM256-FP4 holds all 256 experts on this rank at intermediate dim 512 (12 kernels); MM64-FP4 holds the 64 from an expert offset at intermediate dim 1536, routes to other experts adding nothing (11 kernels); 30 sizes. The baseline runs the router as `tgemm.mm` with its first operand cast to fp32, then `moe_fused_gate` and `fused_moe` at `per_1x32` *(field, E1)*.

## Applies when

- **The op** is that layer on one rank, at this shape: bf16 hidden states of model dim 3072; an fp32 router weight of
  256 rows and an fp32 correction bias; each token's top-8 experts by sigmoid plus bias, weighed by the unbiased
  sigmoids renormalized to sum 1, with no scaling factor and no shared expert; gate-up and down weights in e2m1 with
  an e8m0 scale per 32, in aiter's 16 × 16 shuffled layout; a bf16 `out`, the rank's share of the layer's output.
- **Two rank shapes.** MM256-FP4 holds all 256 experts at intermediate dim 512. MM64-FP4 is an expert-parallel rank
  that holds the 64 experts from `expert_offset` on, at intermediate dim 1536: the top-8 runs over all 256 experts and
  the weights are normalized over all 8 routes, the routes to experts held elsewhere add nothing, and a token keeps
  about 2 routes (estimated, uniform routing) ([triton/moe](../triton/moe.md) §1).
- **The path replaced** is SGLang's `MiniMaxM2MoE.forward_normal`. A speedup is against a baseline that reconstructs
  SGLang's path on aiter, the calls named above: the router GEMM on fp32 activations, a top-k kernel and aiter's
  `fused_moe`, which sorts and runs its two stages; it does not replay SGLang's process-wide small-batch sort patch.
  All-to-all backends, EPLB and deterministic inference are not covered and stay on SGLang's path
  ([triton/moe](../triton/moe.md) §1).
- **Sizes.** Each rank shape has 15 token counts: 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 6144, 8192 and
  16384. A kernel is written for one count or a few neighbours (§ Mechanism).
- **Accuracy.** The family is held to `bounded` accuracy: `out` stays within its error bound against the reference,
  an SQNR floor of 10 dB ([triton/numerics](../triton/numerics.md) §1), and nothing is asked across runs. Eleven of
  the 23 kernels hold relaxed integer atomics, and every kernel of the library with an atomic is `bounded`: the
  atomics place routes (tickets, job slots) in an order that differs between runs, while every float sum is made in a
  fixed order ([triton/moe](../triton/moe.md) §6, §10). The library's 31 kernels with an atomic, these 11 among them,
  still gave byte-identical outputs at every size in 10 runs *(field, E2)*.

## Counter-signals

- **Each rank shape's lowest value is at 32–256 tokens**: that band measured 1.178–1.230 at MM256-FP4 and
  1.190–1.275 at MM64-FP4, against 1.252–1.735 from 1024 tokens and 1.555–3.633 up to 16
  *(field, E4)*. Inside a 1–16-token kernel written for several counts the value falls as tokens grow
  (MM256-FP4/m4–m16: 2.549, 1.977, 1.595; MM64-FP4/m8–m16: 2.113, 1.555). Per size: § Result.
- **No kernel at another token count, and arms no size takes.** A token count outside the 15 has no kernel. Kernels
  keep host arms for counts they are not written for, as the host code of the 1-token kernel does for 2, 4, 8 and 16
  tokens (§ Code, entry 7); only the arm a kernel's size takes was compiled and checked
  ([triton/moe](../triton/moe.md) §12).
- **Untested.** Under graph capture, other routings, the worst-case sizing and the exit on the device's count were
  not tested (§ Mechanism). For a kernel started from this family,
  [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §3 lists the expert offset and the job
  heights (a job's rows, § Mechanism) as untested; `expert_offset` is 192 in every timed workload, while every
  accuracy input set but one runs 0 ([triton/moe](../triton/moe.md) §1). The ticket and its scatter were checked with
  one count block, not several (§ Code, entry 2).
- **Code that no measurement exercises.** MM256-FP4/m2048–m4096 holds a warp-aggregated atomic scatter; at both of
  its sizes the kernel takes the ticket path instead ([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §6).
- **Designs not compared.** No measurement compares the wave-ballot sort with a ticket sort, a job list written by
  the selection launch with a job derived from the counts, or the two router forms
  ([triton/moe](../triton/moe.md) §4, §5, §6). A speedup belongs to a whole kernel at one size.
- **A tie can select 9 experts at 8 tokens on the 64-expert rank.** MM64-FP4/m8–m16 selects by counting, per
  candidate expert, the experts that beat it; at 8 tokens it counts only strictly higher scores, so a tied 8th and
  9th score selects 9, and 16 tokens was not run *(field, E5)*.
- **Shape constants are baked in.** The MM256-FP4 kernels ignore `expert_offset`, which is right while the rank
  holds all 256 experts; the MM64-FP4 kernels from 1024 tokens read an expert from 6 bits (`descriptor & 63`)
  ([triton/moe](../triton/moe.md) §12), and the MM64-FP4 kernels take `expert_offset` as a constexpr, one value per
  process ([triton/authoring](../triton/authoring.md) §5).
- **Every token's row is written at the 64-expert rank.** About 10% of tokens have no route to a held expert
  (estimated, uniform routing); their `out` row is zero and must still be written. A copy of MM64-FP4/m1024 that
  stored only the rows with a local route left 239616 non-finite mismatches in `out`
  *(field, E6)*.
- **Sentinels for the rows no route fills.** A worst-case grid reaches sorted rows that no route fills, so an
  earlier launch of the call fills `Sorted` with an out-of-range route id that the consumer masks: the route count at
  MM256-FP4/m16384, the capacity at MM256-FP4/m64. A copy of MM256-FP4/m64 without that fill crashed, exit −6
  *(field, E6)*.
- **Buffer accesses stop at 2 GiB from one base.** MM256-FP4/m6144, m8192 and m16384 guard their router and
  scale-buffer paths on the host, MM256-FP4/m2048–m4096 in device code and MM256-FP4/m1024 by token count; these are
  not a guard for every buffer base, and the route-row scratch, routes × model dim, would cross the limit at a larger
  token count or model dim ([triton/moe](../triton/moe.md) §12).
- **A size's constants are coupled.** The fold of the 1-token gate-up changed alone crashes the compile (§ Code,
  entry 7).
- **Knobs measured on these kernels.** The gate-ups of MM256-FP4/m8192 and m16384 set `waves_per_eu` 3; removing it
  at 8192 tokens left the speedup at 1.44 (1.441 → 1.440) *(field, E7)*. The weights' cache hint is
  per size ([triton/moe](../triton/moe.md) §7): in one run, `.cg` on the weight loads was faster in every pair at
  both sizes of MM64-FP4/m128–m256, whose schedule reads each weight once, and 17% slower at MM256-FP4/m16384, whose
  schedule re-reads them *(field, E8)*. MM256-FP4/m4–m16 passes `enable_fp_fusion=False` on its down
  launch at 16 tokens only, and no comment explains the setting ([triton/numerics](../triton/numerics.md) §6).
- **The bound leaves little room.** The kernels quantize both expert-GEMM inputs to MXFP4, which the reference does
  not: that alone measured 13.2 dB at 128 tokens against the 10 dB floor *(field, E9)*, so a
  further independent error has to stay above about 12.8 dB (estimated).

## Mechanism

A kernel is one host entry with the Gluon functions it launches, in one source file, written for a fixed list of
token counts. A route is one (token, expert) pair, and its slot its place 0–7 in the token's top-8; a ticket is the
position that a relaxed `atomic_add` on its expert's counter returns; a job is a run of up to BM routes of one expert,
the rows of one gate-up tile; a route's row is its place in the sorted buffer. Launches are the kernel nodes of one
call. Kernels with their token counts in parentheses:

| Tokens | MM256-FP4 | MM64-FP4 | Launches | Decomposition |
|---|---|---|---|---|
| 1–16 | three kernels (1), (2), (4, 8, 16) | three kernels (1, 2), (4), (8, 16) | 3 *(field, E10)* | router on fp32 VALU dots, no MFMA, over the full K of 3072 in every kernel but MM256-FP4 for 4–16 tokens; 492 sampled projections matched an exact sum *(field, E11)*; the last launch runs the down and the weighted combine *(field, E10)* |
| 32–256 | a kernel per count (32), (64), (128), (256) | two kernels (32, 64), (128, 256) | 5 *(field, E10)* | split-K router tiles, counter and job initialization and the hidden quantization in the first two launches; per-token top-8 with relaxed atomic tickets, a job opened every 16th ticket (every 64th at MM256-FP4, 128 and 256 tokens); gate-up over jobs (MM64-FP4 at 128–256 tokens: each program finds its job by scanning the counts); down; combine |
| 1024–16384 | five kernels (1024), (2048, 4096), (6144), (8192), (16384) | a kernel per count (1024), (2048), (4096), (6144), (8192), (16384) | 6–8 *(field, E10)* | router on fp32 MFMAs at MM256-FP4 and at MM64-FP4 for 1024 tokens, on bf16 MFMAs (three bf16 terms of the weight, checked on samples) in the other five MM64-FP4 kernels; 912 sampled projections matched an exact sum *(field, E11)*; at MM64-FP4 too the router scores all 256 experts. Then selection, a sort (tickets at MM256-FP4, wave ballots at MM64-FP4), gate-up with requantization, the down (one design in every MM256-FP4 kernel but the one for 2048–4096 tokens, which has its own) and the combine *(field, E10)* |

The launches of MM256-FP4/m16384, the kernel of entries 1, 2 and 6 of § Code: router projection, which also zeroes
the counters and fills the sentinels; selection, top-8 and tickets; scatter, rows and the hidden quantization;
gate-up; down; combine *(field, E5)*.

- **Techniques.** Job queues from relaxed atomics; a publish tensor built on one wave masks its copies so each route takes one ticket
  ([triton/moe](../triton/moe.md) §6). At MM64-FP4 a counting sort from wave ballots and popcounts
  ([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §6), atomic-free from 2048 tokens by a count of the compiled
  instructions *(field, E12)* and at 1024 tokens by its source (§ Code, entry 3). The MM256-FP4 kernel for 16384 tokens replays a captured graph with new inputs
  correctly and keeps identical grids over its 6 eager launches at two input sets *(field, E13)*; other routings,
  worst-case sizing and the device-count exit are untested ([triton/authoring](../triton/authoring.md) §8).
  Router accuracy: on complete logits at MM64-FP4 for 2048 and 4096 tokens and MM256-FP4 for 1024, over four input
  distributions and eight seeds, the error against an fp64 matmul of the same inputs was no larger than a GPU fp32
  matmul's in L2 in 96 of 96 cases and in max-abs in 93 *(field, E14)*; at 13 more branches (the 1–16-token VALU and the
  32-token fp32 MFMA among them), on a seed-0 random and a cancellation input each, in L2 on 25 of the 26 inputs
  (the cancellation input of MM256-FP4 for 6144 tokens 1.00009×), in max-abs on 23 and element by element on 3; on two more inputs with a
  nonzero weight residual, the fp64 correction of MM64-FP4 for 6144 and for 8192 tokens ran and both errors were 0 *(field, E15)*. Finite
  fp32-level evidence: not every tile or split specialization, nor the same top-8 for every input.
- **Resources** *(field, E12)*: 183 compiled kernels; `num_warps` 1 (35), 2 (7), 4 (139), 8 (2); `waves_per_eu` 3 in 4, 4 in 1;
  VGPR 9–244, no AGPRs; LDS 0–35 KiB; no spills; scaled 16 × 16 × 128 MFMA in 69, fp32 16 × 16 × 4 in 12, bf16
  16 × 16 × 32 in 5, fp32 32 × 32 × 2 in 3, scaled 32 × 32 × 64 in 1.
- **Constructs**, by the number of the 23 kernels whose sources use them (a text scan, dead branches included;
  [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §4): `mfma_scaled` on e2m1 with e8m0 scales
  23, `gl.inline_asm_elementwise` 22, roles split by program id in one grid 19 (a scan that also counts bounds
  branches), padded LDS layouts 11, relaxed global integer atomics 11, `llvm_fn_attrs` 4. Like MM256-FP4/m16384,
  MM256-FP4/m8192 and m2048–m4096 were captured into a graph and replayed with new inputs, accurately, 6 launches
  each *(field, E13)*.

## Code

### 1. The prefix of the padded counts, recomputed in every scatter program (MM256-FP4, 16384 tokens)

A route's row in the sorted buffer needs the exclusive prefix of every expert's padded count. Each program of this
kernel's scatter launch recomputes it rather than read it from a launch of its own, so the ticket sort costs two
launches, selection and scatter. `counts` holds one row per count chunk (`c`) and one column per expert: the program
sums the chunks, rounds each expert's total up to the tile height `BM`, scans, and adds the counts of the chunks
before `chunk`, the one its own routes fall in; a scatter program handles 256 routes and `CHUNK_BLOCK` is the routes
per count chunk (the `COUNT_BLOCK` of §2), hence `gl.program_id(0) * 256 // CHUNK_BLOCK`, and `bases` is each
expert's first row for that chunk. Program 0 stores the padded total in `Total`, which the expert GEMMs read (§6).
`gl.sum` returns a `SliceLayout`, so the scan runs on a 1-D `BlockedLayout`, between two `convert_layout`s.

```python
    padded = gl.cdiv(gl.sum(counts, 0), BM) * BM
    vector: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    padded_vector = gl.convert_layout(padded, vector)
    start_vector = gl.associative_scan(padded_vector, 0, _add) - padded_vector
    start = gl.convert_layout(start_vector, gl.SliceLayout(0, layout))
    chunk = gl.program_id(0) * 256 // CHUNK_BLOCK
    bases = start + gl.sum(gl.where(c[:, None] < chunk, counts, 0), 0)
    if gl.program_id(0) == 0:
        gl.store(Total, gl.sum(padded_vector, 0))
```

Not this: `gl.associative_scan` on that `SliceLayout`. In Triton 3.8.0 it fails to lower ("error: TODO: unsupported
scan layout") and compiles after a `convert_layout` to a 1-D blocked layout *(field, E16)*.

Explained in [triton/moe](../triton/moe.md) §4.3.

### 2. A relaxed atomic ticket per route and the scatter that turns it into a row (MM256-FP4, 16384 tokens)

In the selection launch, one program per token, each of the token's 8 routes takes a ticket: a relaxed
`gl.atomic_add` on its expert's counter returns the route's place among that expert's routes, and the record
`ticket << 8 | ids` is stored in the route's entry of the ids buffer. `Counts` holds a block of 256 counters, one per
expert, for every `COUNT_BLOCK` routes; `row` is the token and `ids` its 8 experts, held in a 64-lane layout whose
redundant copies are predicated off, so the 8 ids take 8 tickets *(field, E6)*. The 8-bit field holds
ids 0–255, and in an int32 record the ticket must stay below 2^23. `sem="relaxed"` suffices because the consumers run
in a later launch; the default `acq_rel` makes the AMD lowering add cache invalidates and L2 write-backs around the
atomic (S1). The ticket and the scatter were checked with one count block, not this kernel's several.

```python
    ticket = gl.atomic_add(
        Counts + row * TOPK // COUNT_BLOCK * 256 + ids, 1, sem="relaxed"
    )
    record = ticket << 8 | ids
```

The scatter launch then turns each record into a row: it reloads the record, splits it into `expert`, the low 8
bits, and `ticket`, and adds the ticket to the expert's entry of `base_table`, the prefix of §1. The prefix is padded
to the tile height `BM`, so a tile holds the routes of one expert, and the route on a tile's first row writes that
expert to `Experts`.

```python
    offset = ticket + gl.gather(base_table, expert, 1)
    gl.store(Sorted + offset, route, valid)
    gl.store(Experts + offset // BM, expert, valid & (offset % BM == 0))
```

Not this: a result that depends on the ticket order, such as a float sum made in arrival order. On a copy of the
64-token kernel's form at 256 tokens the tickets differed in 19 of 20 runs: the slot-order combine kept one bit
pattern, a float `atomic_add` in arrival order gave 20 *(field, E6)*.

Explained in [triton/moe](../triton/moe.md) §6.

### 3. A counting sort from seven wave ballots, no atomic (MM64-FP4, 1024 tokens)

From 1024 tokens the 64-expert rank sorts its routes with wave ballots and popcounts instead of tickets. A wave holds
a stripe of 64 routes, one per lane, and lane j also stands for the rank's expert j: seven ballots leave in lane j the
64-bit mask of the stripe's routes whose expert is j, and the popcounts of the masks give the experts' counts.
`_wave_ballot_2` is an inline-asm `v_cmp_ne_u32_e64` that returns the wave's lane mask as a uint64
([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §6). `ids` are the routes' expert ids less the rank's expert
offset, so a route to an expert held elsewhere falls outside 0–63, and `r < R` masks the rows past the route count:
every lane votes, so validity is folded into the first predicate. Unlike tickets, the masks fix the row order; the
source of this kernel holds no atomic call, and "atomic-free from 2048 tokens" in § Mechanism rests on a count of
the atomic instructions in the compiled kernels, which its evidence line gives for 2048, 4096, 6144, 8192 and 16384
tokens *(field, E12)*.

```python
    # each ballot is a wave's 64-bit lane mask; a wave holds a stripe of 64 routes, and lane j also stands for expert j
    mask = _wave_ballot_2((r < R) & (ids >= 0) & (ids < 64))  # ballot 1: the routes to an expert of this rank
    for bit in gl.static_range(6):  # ballots 2-7, one per bit of the expert id
        bits = _wave_ballot_2(ids & 1 << bit != 0)
        mask &= gl.where(expert[None, :] & 1 << bit != 0, bits, ~bits)  # lane j keeps the routes whose expert is j
```

Explained in [triton/moe](../triton/moe.md) §6.

### 4. SwiGLU and MXFP4 requantization in the gate-up epilogue (MM64-FP4, 4096 tokens)

A gate-up program of this kernel owns full K for its gate and up columns and whole quantization groups of its output,
so the activation and its requantization run in its epilogue, not in a launch of their own
([triton/moe](../triton/moe.md) §8). `acc` is the fp32 accumulator of a BM × BN tile whose first BN/2 columns are gate
columns and the rest the matching up columns. It is rounded to bf16 and split into (gate, up) pairs, and
`gate · sigmoid(gate) · up` is rounded to bf16 again. `_quantize_groups_3`, the kernel's form of §8, then quantizes
the tile's `groups` groups of 32 to MXFP4, taken in a `layout` that keeps at least two consecutive values of a group
in one thread, which its `gl.split` needs. The two bf16 roundings are the kernel's, where its reference stays in
fp32; such roundings can use up the error budget ([triton/numerics](../triton/numerics.md) §1).

```python
    rounded = acc.to(gl.bfloat16).to(gl.float32)
    pairs = gl.permute(gl.reshape(rounded, (BM, 2, BN // 2)), (0, 2, 1))
    gate, up = gl.split(pairs)
    value = (gate * _sigmoid(gate) * up).to(gl.bfloat16)
    groups: gl.constexpr = BM * BN // 64
    grouped = gl.convert_layout(gl.reshape(value, (groups, 32)), layout).to(gl.float32)
    q, scale = _quantize_groups_3(grouped)
```

Explained in [triton/moe](../triton/moe.md) §8.

### 5. Program 0 of the first launch writes what the next launch's atomics start from (MM256-FP4, 32 tokens)

Scratch comes from `torch.empty`, which holds whatever was there, in an eager call and in every replay; none of the
library's kernels calls `torch.zeros` or `fill_`, so an earlier launch of the call writes the initial values. Here
program 0 of the first launch stores 0 into the expert counts `Counts` and into `Total`, this kernel's job counter,
and −1 into the `Jobs` slots, on which the next launch takes its atomic tickets (§10). `E` is the 256 experts and
`MAX_JOBS` the host's bound on the jobs one call can open; the index vector is rounded up to a power of two and
masked. Every program, 0 included, then goes on to its router or quantization tile.

```python
pid = gl.program_id(0).to(gl.uint32)
linear_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
if pid == 0:  # one program of the first launch writes what the next launch's atomics start from
    expert = gl.arange(0, _next_power_of_2(max(E, MAX_JOBS)), layout=linear_layout)
    gl.store(Counts + expert, 0, expert < E)
    gl.store(Jobs + expert, -1, expert < MAX_JOBS)
    gl.store(Total, 0)
# ... every program, 0 included, then goes on to its router or quantization tile
```

Not this: counters that no launch of the call writes before they are read. In Gluon fold probes, a fold whose
counters no node of the graph zeroed was wrong in every replay after the first, 1999 of 1999 at each of 9 sizes
*(field, E17)*.

Explained in [triton/authoring](../triton/authoring.md) §6.

### 6. A worst-case grid from shapes and an exit on the device's row count (MM256-FP4, 16384 tokens)

A host decision on a count the GPU computed needs a synchronization, and a replay of a captured graph keeps the
captured decision, so the grid of the expert GEMMs is sized from shapes for the worst case
([triton/authoring](../triton/authoring.md) §8). `routes` is tokens × 8, `e` the 256 experts and `block_m` the 64-row
tile: `capacity` is the smaller of `routes` tiles of one route each and `routes` rows plus `block_m - 1` rows of
padding per expert, rounded up to whole tiles. It sizes the sorted buffer and the grids of the gate-up and the down.
What was checked of this kernel under capture, and what was not: § Mechanism.

```python
# host: the grid covers the most rows of tiles any routing can need, from shapes alone
capacity = (
    triton.cdiv(min(routes * block_m, routes + e * (block_m - 1)), block_m)
    * block_m
)
```

On the device, a gate-up program maps its linear id to a row tile and a column tile (`_grouped_tile_2` gives
`M_GROUP` row tiles of one column tile consecutive ids) and returns at once when its tile starts at or past `Total`,
the padded row count that program 0 of the scatter launch stored (§1).

```python
# kernel: a program past the rows the GPU counted returns at once
linear = gl.program_id(0) + gl.program_id(1) * gl.num_programs(0)
tile, column_tile = _grouped_tile_2(linear.to(gl.uint32), column_tiles, M_GROUP)
if tile * BM >= gl.load(Total):
    return
```

Not this: `torch.nonzero` or `.item()` on the count to size the grid on the host. Called alone on a CUDA tensor, not in
this kernel's launch code, each failed a capture with `capture_error_mode="global"` *(field, E19)*.

Explained in [triton/authoring](../triton/authoring.md) §8.

### 7. The gate-up fold and the constants it is coupled to (MM256-FP4, 1 token)

The gate-up of this kernel packs several K chunks into the 16 rows of its MFMA, their number set per token count
([triton/gluon/mfma](../triton/gluon/mfma.md) §8): with a fold of 4, row partition p and column block p read the same
K chunk, and the diagonal blocks are summed ([triton/moe](../triton/moe.md) §7). The excerpt is the host code that
sets it: `gate_fold` becomes the launch's constexpr `FOLD`, beside `NATIVE_FOLD` and, at m ≤ 2, `llvm_fn_attrs` with
`amdgpu-sched-strategy=iterative-ilp`. The kernel is written for 1 token; the arms for other m are branches its size
does not take. A size's constants were chosen together: change them together, from a combination that compiles
elsewhere, here the same source's 16-token branch.

```python
gate_fold = 4 if m <= 2 else 2 if m == 4 else 8 if m == 8 else 1
gate_unroll = 6 if m == 2 else 3
gate_attrs = (["amdgpu-sched-strategy", "iterative-ilp"],) if m <= 2 else ()
_gate_up[gate_grid](
    # ... the tensors and sizes ...
    K_UNROLL=gate_unroll,
    FOLD=gate_fold,  # 1 in place of 4 at m = 1, nothing else changed: the compile segfaults
    PROJECT_ROWS=8 if m == 8 else 4 if m == 16 else 16,
    TRANSPOSED=m not in (1, 2, 8),
    NATIVE_FOLD=m in (1, 8),
    llvm_fn_attrs=gate_attrs,
)
```

Not this: `FOLD` 1 in place of 4 at 1 token with nothing else changed. Every changed run crashed, exit −11, in 4 of
4, and that launch compiled alone segfaults in LLVM's code generation *(field, E21)*; a fold of
1 compiles with 1 warp, with a 256-wide K tile, or without the scheduler attribute *(field, E16)*.

Explained in [triton/debugging](../triton/debugging.md) §6.

### 8. MXFP4 quantization of groups of 32 in registers (MM256-FP4, 64 tokens)

All 23 kernels quantize both expert-GEMM inputs to MXFP4 in registers, e2m1 values with one e8m0 scale per 32; the
reference leaves the activations unquantized, so this rounding is the kernel's, its error left to the 10 dB bound
([triton/numerics](../triton/numerics.md) §3). The excerpt is the body of the group quantizer of the 64-token
MM256-FP4 kernel, a `@gluon.jit` function of `x`, fp32 of shape (groups, 32) with one MX block per row; it returns the
packed codes and the scale bytes. The exponent follows the ceil rule, the block scale rounded up from its maximum / 6;
the values are multiplied by the reciprocal of that power of two, NaN and ±0 are mapped to +0 (an all-zero group gives
0 · Inf), and `v_cvt_scalef32_pk_fp4_f32` packs two values into one byte, round to nearest even. In probe kernels,
not library kernels, this recipe gave no wrong code or scale byte on 100000 groups against a float64 model
*(field, E22)*. `gl.split` needs each pair in one thread, at least 2 consecutive elements per
thread along the 32, or the compile fails.

```python
peak = gl.max(gl.abs(x), 1)
ratio = gl.div_rn(peak, 6.0)
bits = ratio.to(gl.uint32, bitcast=True)
exponent = (bits >> 23 & 255).to(gl.int32) - 127 + (bits & 8388607 != 0)
exponent = gl.minimum(127, gl.maximum(-127, exponent))
scale = gl.exp2(exponent.to(gl.float32))
normalized = x * gl.div_rn(1.0, scale)[:, None]
normalized = gl.where((normalized == 0) | (normalized != normalized), 0.0, normalized)
even, odd = gl.split(gl.reshape(normalized, (x.shape[0], 16, 2)))
packed = gl.inline_asm_elementwise(
    "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0 op_sel:[0,0,0]", constraints="=v,v,v",
    args=[even, odd], dtype=gl.uint32, is_pure=True, pack=1,
).to(gl.uint8)
return (packed, (exponent + 127).to(gl.uint8))
```

Not this: the asm result declared `dtype=gl.uint8`. With `pack=1` LLVM prints `error: could not allocate output
register for constraint 'v'` to stderr, Triton 3.8.0 raises nothing and the asm compiles to nothing: the kernel
launches and writes no codes, or wrong ones; an output left unwritten put every element over its bound
*(field, E9)*.

Explained in [triton/numerics](../triton/numerics.md) §3.

### 9. A software e2m1 encoder: comparisons against the code midpoints (MM64-FP4, 128–256 tokens)

The MM64-FP4 kernel for 128 and 256 tokens encodes e2m1 in software; the sources of the other 22 kernels of the
family hold the hardware convert of §8 ([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §1). `x` is a value
already divided by its group's scale: the seven comparisons count the code midpoints that |x| has passed, alternating
`>` and `>=` so that a tie goes to the even code, and bit 3 is the sign. A NaN compares false and becomes code 0, and
the kernel passes `x * (1 / scale)` without a NaN guard: an all-zero group's 0 · Inf comes out as code 0. At its two
sizes the kernel measured 1.190 and 1.194 over the baseline *(field, E4)*, the whole kernel's value,
not the encoder's.

```python
@gluon.jit
def _fp4_code(x):
    a = gl.abs(x)
    code = (a > 0.25).to(gl.uint8) + (a >= 0.75).to(gl.uint8)
    code += (a > 1.25).to(gl.uint8) + (a >= 1.75).to(gl.uint8)
    code += (a > 2.5).to(gl.uint8) + (a >= 3.5).to(gl.uint8)
    code += (a > 5.0).to(gl.uint8)
    return code | gl.where(x < 0, 8, 0).to(gl.uint8)
```

Explained in [triton/numerics](../triton/numerics.md) §3.

### 10. A ticket as the route's row in its expert's region, a job every 16th ticket (MM256-FP4, 32 tokens)

At 32–256 tokens each expert owns a fixed region of the sorted buffer, as long as the routes it can receive, so a
route's row is `expert · stride + ticket` and the selection launch writes it with no scan
([triton/moe](../triton/moe.md) §6). In this kernel `experts` holds a token's 8 selected ids, `valid` is
`slot < TOPK`, `CAPACITY` is the call's route count and `row * TOPK + slot` the route. The atomic is a relaxed integer
add: it hands out a position and sums nothing ([triton/numerics](../triton/numerics.md) §8). A ticket that is a
multiple of 16 opens a job: the lines after the excerpt take a second relaxed atomic on `Total` and write the job's
first row to `Jobs`.

```python
ticket = gl.atomic_add(Counts + experts, 1, valid, sem="relaxed")
gl.store(Sorted + experts * CAPACITY + ticket, row * TOPK + slot, valid)
new_job = valid & (ticket % 16 == 0)
```

Not this: `Counts` and `Total` left as the previous call wrote them: they must read zero at the start of every call,
graph replays included, which an earlier launch of the call sees to (§5). In a 4096-program ticket kernel replayed
in a graph without a zeroing node, the second replay's tickets started at 4096 *(field, E2)*.

Explained in [triton/numerics](../triton/numerics.md) §8.

## Result

- **Speedups** over the baseline above *(field, E23)*: 1.178 (MM256-FP4, 256 tokens) to 3.633 (MM64-FP4, 1 token) over the 30 sizes.

Per size, each kernel's values from one run of the ruler:

| Tokens | MM256-FP4 *(field, E4)* | MM64-FP4, expert offset 192 *(field, E4)* |
|---|---|---|
| 1 | 3.444 | 3.633 |
| 2 | 3.309 | 2.797 |
| 4 | 2.549 | 2.554 |
| 8 | 1.977 | 2.113 |
| 16 | 1.595 | 1.555 |
| 32 | 1.179 | 1.231 |
| 64 | 1.230 | 1.275 |
| 128 | 1.210 | 1.190 |
| 256 | 1.178 | 1.194 |
| 1024 | 1.352 | 1.252 |
| 2048 | 1.272 | 1.318 |
| 4096 | 1.261 | 1.590 |
| 6144 | 1.415 | 1.689 |
| 8192 | 1.423 | 1.735 |
| 16384 | 1.389 | 1.619 |

Each value belongs to the whole kernel at that size. Over 170 sizes, not this family's alone, repeat runs of the
ruler on unchanged sources moved the ratio at a size by 2% at the median and 6% at p90, and a later run moved the 57
MoE sizes it re-measured by −6.3% to +7.0%, 0.9% at the median *(field, E4)*: gaps of that order
between sizes or kernels say nothing about designs. The kernels for 8192 and 16384 tokens share their source
but for the docstring on both rank shapes, so 1.423 against 1.389 and 1.735 against 1.619 are one source at two sizes
([triton/moe](../triton/moe.md) §11).

Call times on the ruler's timer: MM64-FP4/m128–m256 at 256 tokens 103.6–105.4 µs and MM256-FP4/m8192 505.7–518.6 µs,
4 runs each *(field, E5)*. With routing simulated from the workload's router distribution,
MM256-FP4/m64 took 99.09–101.7 µs against the baseline's 127.0–127.8 µs, a speedup of 1.249–1.285, and touched 77.2%
of the experts on average, in 3 runs *(field, E24)*.

The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1.

## Evidence

- E1 [2026-11, CPU (target gfx950), Triton 3.8.0, N=1, each kernel's `run` replayed on the host with its launches recorded, no GPU] the 24 fp8 GEMM sizes one launch each, the N 3072 kernel for M = 6144 rejecting independently strided A and B; at the 89 MoE sizes of 58 kernels the gate-up launch, its grid and the source of its W1; the baselines' calls: MiniMax-M2 routers `tgemm.mm` with the first operand cast to fp32, then `moe_fused_gate` and `fused_moe` at `per_1x32` (MXFP4) or `per_128x128` (FP8); GDN decode a conv update, the recurrence, the norm and the quantization; GDN prefill `causal_conv1d_fn`, `fused_gdn_gating`, `chunk_gated_delta_rule`, the norm and the quantization (`per_token_group_quant_fp8` in Qwen3.5)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, runs on one GPU] the accuracy check on a `bitwise` and a `bounded` copy of one kernel ran the first input set 3 times (a first capture, the accuracy run, the repeat run) and 2 times, and bit flips planted in those runs produced no finding; bf16 × bf16 products in fp32's normal range exact for 748591 VALU and 16384 MFMA products; 512 crafted values near bf16 midpoints, alone and with 512 random values (1024): plain fp64 → fp32 → bf16 wrong on 260 either way, the nudge of the 1-token Qwen2-MoE router + experts kernel on 0; repeated calls on the same inputs: the 24 a8w8 blockwise kernels 0 non-identical outputs in 20 runs per size, the bf16 GEMM N 2624 × K 6144 1 bit pattern in 20 runs at m = 2, 4, 8, 64, the 31 bounded kernels with atomics byte-identical at every size in 10 runs (the DS512 kernel for 4096 tokens of E3 in 50); a 4096-program ticket kernel in a graph: second replay offset by 4096 without a zeroing node, identical with one (3 runs)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the kernel called 100 times with synchronization, its runtime records saved and their bytes hashed on CPU] the target block on the 128-row body (32 × 32 × 64 MFMAs) in all 100 calls, one hash of its route partials; at 15, 16, 17, 31, 32 and 33 live rows, each with the body that count selects and with the next larger one forced, the target's up codes, scales and down partials and token 0's 6144 outputs identical within each live count (1,040 hashes) (the DS512 kernel for 4096 tokens; the same check, in a run of its own, on the DS512 kernel for 256 tokens: the target on the 16-row body in all 100 calls, one hash of its route partials, no bit of its up codes, scales or down partials changed at those live rows)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush, Kineto), speedup over the baseline of SGLang's path on aiter] speedup per size, the implementation in parentheses: DeepSeek-V2/GLM DS256: 1 token 2.139 (DS256/m1), 2 2.186 (DS256/m2), 4 1.867 (DS256/m4), 8 1.521 and 16 1.245 (DS256/m8–m16), 32 1.086 (DS256/m32), 128 1.030 (DS256/m128), 256 1.160 (DS256/m256), 1024 1.402, 2048 1.329 and 3072 1.407 (DS256/m1024–m3072), 4096 1.420 and 4192 1.378 (DS256/m4096–m4192); DS512: 1 token 2.093 and 2 1.829 (DS512/m1–m2), 4 1.324 (DS512/m4), 8 1.184 (DS512/m8), 16 1.097 (DS512/m16), 256 1.043 (DS512/m256), 1024 1.152 (DS512/m1024), 2048 1.391 (DS512/m2048), 3072 1.235 (DS512/m3072), 4096 1.161 (DS512/m4096), 4192 1.166 (DS512/m4192); MiniMax-M2 FP8, 256 experts: 1, 2, 4, 8, 16 tokens 4.115, 3.755, 2.452, 1.841, 1.732 (MM256-FP8/m1–m16); 32 1.158 and 64 1.173 (MM256-FP8/m32–m64); 128 1.106 (MM256-FP8/m128); 256 1.133 (MM256-FP8/m256); 1024–16384 1.257, 1.248, 1.348, 1.372, 1.388 (MM256-FP8/m1024–m16384); FP8, 64 of 256 experts (expert_offset 192): 1, 2, 4, 8, 16 tokens 3.137, 2.912, 1.672, 1.466, 1.325 (MM64-FP8/m1–m16); 32 1.167 and 64 1.102 (MM64-FP8/m32–m64); 128 1.146 and 256 1.164 (MM64-FP8/m128–m256); 1024–16384 1.216, 1.485, 1.681, 1.770, 1.785 (MM64-FP8/m1024–m16384); MXFP4, 256 experts: 1 token 3.444 (MM256-FP4/m1), 2 3.309 (MM256-FP4/m2), 4, 8, 16 2.549, 1.977, 1.595 (MM256-FP4/m4–m16), 32 1.179 (MM256-FP4/m32), 64 1.230 (MM256-FP4/m64), 128 1.210 (MM256-FP4/m128), 256 1.178 (MM256-FP4/m256), 1024 1.352 (MM256-FP4/m1024), 2048 1.272 and 4096 1.261 (MM256-FP4/m2048–m4096), 6144 1.415 (MM256-FP4/m6144), 8192 1.423 (MM256-FP4/m8192), 16384 1.389 (MM256-FP4/m16384); MXFP4, 64 of 256 experts (expert_offset 192): 1 token 3.633 and 2 2.797 (MM64-FP4/m1–m2), 4 2.554 (MM64-FP4/m4), 8 2.113 and 16 1.555 (MM64-FP4/m8–m16), 32 1.231 and 64 1.275 (MM64-FP4/m32–m64), 128 1.190 and 256 1.194 (MM64-FP4/m128–m256), 1024 1.252 (MM64-FP4/m1024), 2048 1.318 (MM64-FP4/m2048), 4096 1.590 (MM64-FP4/m4096), 6144 1.689 (MM64-FP4/m6144), 8192 1.735 (MM64-FP4/m8192), 16384 1.619 (MM64-FP4/m16384); Qwen2-MoE router+experts: 1 token 1.172 (Q2-RE/m1), 2 1.145 (Q2-RE/m2), 4 1.119 (Q2-RE/m4), 32 1.063 (Q2-RE/m32); router: 8 tokens 1.074 (Q2-R/m8), 16 1.219 (Q2-R/m16), 64 1.295 (Q2-R/m64), 128 1.417 (Q2-R/m128), 256 1.967 (Q2-R/m256), 2232 1.416 (Q2-R/m2232), 4096 1.226 (Q2-R/m4096); fp8 experts: 8 tokens 1.208 (Q2-E/m8), 64 1.171 (Q2-E/m64), 128 1.076 (Q2-E/m128); the values come from eight runs of the ruler, each implementation's from one run; a later run re-measured 57 of the 96 sizes, by −6.3% (MM256-FP8 at 1 token, 4.115 → 3.855) to +7.0% (Q2-R at 16 tokens, 1.219 → 1.304), median change 0.9%; over repeat runs of the ruler on unchanged sources, the ratios at the same size differ by 2.03 % at the median, 6.2 % at p90 and by more than 5 % in 16 % of 1716 pairs at 170 sizes, at most 54.8 % (Q2-RE at 1 token)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the implementations' call times N=4, the top-k probes N=3), one GPU: launch traces with torch.profiler, the ruler's timer (Kineto) and bare graph replay on test kernels, the ruler on the implementations and on copies with one change (A/B: 4 alternated rounds per arm)] launch traces of the 96 MoE sizes: launches per call as in [triton/moe](../triton/moe.md) §1; every size compiles on its first call, none on its second, and 136 of 471 launches carry no token-count constexpr; DS256/m2 at 2 tokens grids 304, 2306, 72, 768; DS256/m1 at 1 token router launch 216 = 192 + 24 programs, selection and gate-up launch 1153; all 32 DeepSeek and MiniMax sizes from 1024 tokens sort before the gate-up; none of 185 kernel names is a quantizer alone; routers of 512 programs in MM256-FP4/m128, MM256-FP4/m256, MM256-FP8/m256, MM64-FP8/m128–m256 and of one full-K program per (token block, expert) in MM256-FP8/m1–m16, MM256-FP4/m2, MM64-FP4/m8–m16 above 1 token; Q2-E/m64 at 64 grids of 3072 and 4096, MM64-FP8/m128–m256 at 256 a persistent W13 kernel on 48 × 96; the same grids on two routings and graph replay equal to eager at 82 sizes; split-K finish launches in DS256/m2, DS256/m1, DS512/m1–m2 at 1 token and Q2-E/m8; the baseline enqueues 8–11 device operations at 1–2 tokens, these implementations 3–4; docstring lists differ from the trace at 7 sizes in 5 of 11 implementations; one trivial kernel reads 4.21 µs on the ruler's timer, each added one 4.06 µs, and a bare graph replay pays 1.815–1.844 µs per extra kernel; call times of the unchanged implementations, mean of 2 measurements of the ruler's timer, 4 runs: DS256/m2 at 2 tokens 19.06–19.61 µs (4.76–4.90 µs per launch), DS512/m256 at 256 225.5–236.8 µs, MM64-FP4/m128–m256 at 256 103.6–105.4 µs, MM256-FP4/m8192 at 8192 505.7–518.6 µs, an earlier batch of runs 225.8, 102.9 and 515.6 µs for DS512/m256, MM64-FP4/m128–m256 and MM256-FP4/m8192; DS256/m2 at 2 tokens, gate-up K splits 4 → 2 (grid 2306 → 1154): 2.183 → 2.175, 19.20 → 19.28 µs, runs 2.167–2.195 against 2.157–2.191, paired 95 % confidence interval of the ratio 0.991–1.001; top-8 of 256 at 16384 tokens: one wave per token 12.96–13.18 µs with no `s_barrier`, four waves 37.68–38.32 µs with 31, the same ids, no scratch in either; the keyed min (retired scores at −inf) 0 wrong of 4000 rows; MM64-FP8/m1–m16 exact at 8 and 16 tokens; MM64-FP4/m8–m16 at 2 tokens keeps 8 through its id tie rule, at 8 tokens selects 9 on a tied 8th and 9th (16 tokens not run); DS256/m2's election (the form of [triton/moe](../triton/moe.md) §4.1) over 16384 programs 9.7–10.7 µs at slots 0–4, 11.3–11.4, 12.8–13.0 and 14.3–14.4 µs at slots 5–7, R² of a linear fit 0.724–0.774; with MM64-FP4/m4096's combine lanes disagree on 10000 of 10000 NaN rows and 0 of 10000 finite ones; MM256-FP8/m1–m16 and MM256-FP4/m2 give ±0 and every NaN payload one key each, Q2-RE/m1 gives 7 bf16 NaNs 7 keys; an MFMA-only fp16 loop with 4 useful rows: `instr_shape=[4, 64, 64]` 2432 µs, [16, 16, 32] 2810 µs; every finite E4M3FN code converts exactly to bf16; integer-scratch patterns over 100 calls: Q2-E/m64 at 64 tokens 9 and Q2-E/m128 at 128 4 in one run, 15 and 8 in another; 31 implementations with atomics, 20 runs each, one output each (the 96 MoE sizes; probe kernels)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the toy MoE N=4), test kernels and copies of the implementations' code on one GPU] DeepSeek's reference with both GEMMs' activations MXFP4-quantized moves `out` by rel L2 0.2103, max abs 31.19; a MM256-FP4/m64-form copy at 32, 128 and 256 tokens puts each route in a job of up to 16 rows of one expert, and over 20 runs at 256 tokens the tickets differ in 19 while the slot-order combine keeps one bit pattern (a float `atomic_add` in arrival order gives 20); 1000 programs of 8 ids in a 64-lane layout leave every counter at 1000; 512 crafted near-midpoint logits: f64 → f32 → bf16 wrong on 260, with the midpoint fix on 0; those 512 and 512 random values through Q2-RE/m1's helper alone: f64 → f32 → bf16 wrong on 260 of the 1024, the helper on 0; an MM64-FP4/m1024 copy storing only the rows with a local route: 239616 non-finite mismatches in `out`; a MM256-FP4/m64 copy without its fill of `Sorted`: the copy crashes, exit −6; under `launch_cooperative_grid` a counter barrier over 256 workgroups of 1024 threads finished in 0.2 ms, every program seeing every store; 4096 workgroups were refused (`hipErrorCooperativeLaunchTooLarge`); on a 2.5 GiB tensor `buffer_load` read the data at byte 2^31 − 64 and 0 at 2^31 + 64 with no fault, `gl.load` both; a toy decode MoE (32 experts, top-4, H 256, I 128, every gate-up program owning full K, no requantization) at 2 and 16 tokens against no fusion, 4 runs: each fusion alone, the other two off, saves on the ruler's timer (Kineto) top-k recomputed in the gate-up, down and combine programs 3.30–4.19 µs, SwiGLU in the gate-up epilogue 4.04–4.64 µs, down and combine in one program 4.19–4.43 µs (their replay times not recorded); the three together give call-time ratios of 0.667–0.751 replayed and 0.426–0.462 on the ruler's timer, rel L2 against no fusion 4.5e-8 and 5.1e-8; in the same runs, each taken as the last of the three with the other two on in both arms, top-k 2.89–3.87 µs and down and combine 2.94–3.73 µs (test kernels)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 sessions (the `.cg` A/Bs three runs in the second), 4 alternated rounds per arm, the ruler, one GPU] first session: DS256/m2 at 2 tokens, `.cg` → none on the weight loads 2.164 → 2.170 (18.98 → 19.05 µs); MM64-FP4/m128–m256 at 256 tokens `.cg` → none 1.180 → 1.156, all 4 pairs (102.93 → 104.65 µs), its unchanged 128-token size 1.183 → 1.185; DS512/m256 at 256 tokens `.cg` added on the 128-row gate-up tiles 1.043 → 1.033, all 4 pairs (225.81 → 227.67 µs); DS512/m8 at 8 tokens `enable_fp_fusion=False` → `True` on all 3 launches 1.150 → 1.150 (56.63 → 56.66 µs), accuracy passed; MM256-FP4/m1 at 1 token K fold 4 → 1 alone: every changed run crashed (exit −11), unchanged runs 3.50–3.59; MM256-FP4/m8192 at 8192 tokens gate-up `waves_per_eu` 3 → 0 1.442 → 1.443 (515.56 → 514.69 µs); second session, three runs: DS256/m2 without `.cg` 2.171 → 2.171 (19.30 → 19.32 µs, pairs both ways), 2.205 → 2.180 and 2.217 → 2.195 (×1.009 and ×1.010, every pair slower); MM64-FP4/m128–m256 without `.cg` 1.189 → 1.150, 1.181 → 1.145, 1.180 → 1.144 (×1.029–1.035); DS512/m256 with `.cg` 1.040 → 1.034, 1.039 → 1.033, 1.032 → 1.027 (×1.005–1.007), each slower in all 4 pairs; a toy kernel whose 64 KiB blocks are each read by two adjacent programs 39.44–39.95 µs unhinted against 41.35–41.74 µs with `.cg`, one read of each block 27.17–27.93 against 24.19–25.39 µs; DS512/m8 1.147 → 1.146 (B/A 0.9994), accuracy passed in all 8 runs, its 3 launches compiled for gfx950 with and without the switch giving the same code (633, 1554 and 440 instructions); MM256-FP4/m8192 1.441 → 1.440 (510.9 → 511.8 µs); the K-fold crash in 4 of 4 changed runs, the unchanged implementation 3.535–3.570
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, 4 alternated rounds per arm, 12 sizes of seven implementations, A/A controls at four sizes; reuse classified from the schedule and the measured routing, not from cache counters] `.cg` on the weight loads against none: read once at MM64-FP8/m1–m16 1, 4, 8, 16 tokens, DS256/m1 1, MM256-FP8/m1–m16 1, 2, MM64-FP4/m128–m256 128, 256 and DS256/m1024–m3072 2048, re-read at DS512/m256 256 and MM256-FP4/m16384 16384; `.cg` faster in every pair in call time and in speedup at six read-once sizes (MM64-FP8/m1–m16 at 4, 8, 16, MM64-FP4/m128–m256 at 128, 256, DS256/m1024–m3072 at 2048; 1.6–9.6%), MM256-FP8/m1–m16 at 2 tokens in aggregate only, no consistent effect at the three 1-token sizes; without `.cg` faster in every pair at both re-read sizes (DS512/m256 1.012 → 1.009, 0.2–0.5%; MM256-FP4/m16384 17%); A/A offsets within 0.37%; every arm accurate
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, runs on one GPU: the bound formulas applied to built outputs, or the accuracy check run on a library kernel and on an edited copy] a 0/0 tolerance on bf16 in the form B = atol + (rtol + u)·|y|, y the unrounded reference and u = 2^-8 the unit roundoff of bf16: B = 2^-8·|y|, the round-to-nearest output passed, 1% of elements one step further failed (fraction over the bound 0.00995); MXFP4 round trip (ceil e8m0, RNE e2m1) of both expert-GEMM inputs inside the references, first input set: DeepSeek-V2 13.45 dB at 256 tokens, MiniMax-M2 13.18 dB at 128; an fp32 dot-product rounding bound at K = 6144: fp32 sums in 1, 6 and 16 parts passed at peak 0.69; bias test, n = 10^6, every error +0.5·u·|y|: |mean|/rms 0.80, the output-rounding and dot-product bounds failed, the tolerance, the SQNR floor and a bound relative to a same-mean baseline passed; RTZ to bf16 of positive values 0.68; the check's large-magnitude input set, every floating-point input scaled by one exact power of two, the largest that keeps the tensors and the dot-product outputs in range: 2^54 for the N 2624 × K 6144 bf16 GEMM and 2^55 for the N 6144 × K 2048 bf16 GEMM at all 10 sizes; Qwen3.5 GDN decode bounds: 65536 codes moved one e4m3 step and 512 scales moved one ulp all passed; a8w8 block scales, 2400 drawn: log2 from −2.999 to −0.0009, none an integer; failures: `0 · (1 / gl.exp2(-127))` 256 of 256 non-finite mismatches, a 0xFF scale, an RTZ'd NaN, saturated 448, 65504 and fp32 maximum, NaN against 448 one each; a bf16 running sum over K = 2048 fraction over the bound 0.941 at peak 58.4, a stray fp4 nibble 0.438, an unwritten output 1 at peak 128; missing tap rounding, a contracted product and x·(1/s) at injected ties 1.4e-4, 9.5e-5, 1.2e-4 at peaks 1.84, 1.95, 2.67 under bounds chosen for the test; fp8 P on P·V 23.1 dB; MXFP4 18.77 dB, 17.07 with an e5m2 rounding first; the Qwen2-MoE router kernel for 256 tokens, seed 0: fp32 logits rounded once to bf16, then ranked, gave the reference's top-10 ids for all 256 tokens, the unrounded sum ranked changed 97 tokens and failed the id comparison (fraction of ids differing 0.0809) in both checked runs of each run; the a8w8 GEMM N 7168 × K 2048 at m = 64: `dot * sx * sw` against `dot * (sx * sw)` in K stages 0–2 of 8 (stages 3–7 multiply by a precomputed sx · sw in both arms), 2 checked runs per arm in each run: all 12 passed, worst fraction over the bound 0
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one call per size with its device launches recorded by the profiler] MiniMax-M2 MXFP4: 3 launches at the 10 sizes of 1–16 tokens, the last a down with the weighted combine, the router over K = 3072 in every kernel but MM256-FP4 for 4–16 tokens; 5 at 32–256 tokens; 6–8 at 1024–16384, the down one design in every MM256-FP4 kernel but the one for 2048–4096 tokens, which has its own; MiniMax-M2 FP8: MM256-FP8 3 at 1–16 tokens, its down-and-combine launch 8 warps on a (token, column tile) grid, MM64-FP8 4 at 1–4 and 6 at 8–16, 5–6 at 32–256 with MM64-FP8 at 256 launching a persistent gate-up on 48 × 96 programs, MM256-FP8 6 at 1024–4096 and 7 at 8192–16384, MM64-FP8 8; GDN prefill at all 34 sizes 5–8 launches in the order prepare, chunk factors, summaries for long sequences, one sequential launch, epilogue; at the 21 Qwen3.5 sizes the final conv history the last three entries of the old history (zeros when fresh) followed by the inputs (MoE and GDN prefill kernels)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one input set per size, the router launch's projection sampled against `math.fsum` with a tolerance per sample, its ISA read; not timed] 58 MiniMax-M2 router projections (hidden bf16 [M, 3072], weight fp32 [256, 3072], M from 1 to 16384): all 8520 sampled comparisons within tolerance, none non-finite; by subset (overlapping): MXFP4 1–16 tokens 10 sizes / 492 comparisons, MXFP4 1024–16384 12 / 912, FP8 1–16 10 / 180, FP8 32–256 8 / 3456, bf16-MFMA routers 18 / 1896; no MFMA in the routers at 1–16 tokens; bf16 MFMA in the MM64-FP4 kernels for 2048, 4096, 6144, 8192 and 16384 tokens, the MM256-FP8 kernels for 256 and for 1024–16384, and the MM64-FP8 kernels for 128–256 and for 1024–16384; fp32 MFMA in every other router above 16 tokens; the three bf16 terms of the MM64-FP4 routers checked on samples, the FP8 routers' inline terms not (MiniMax-M2 MoE kernels)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, each kernel captured into a CUDA graph, replayed and checked for accuracy; eager launches recorded by the profiler] MM256-FP4 for 16384, for 8192 and for 2048–4096 tokens and MM64-FP8 for 1024–16384: captured graphs replayed with new inputs, accurate; eager grids identical at two input sets, 6, 6, 6 and 8 launches; other routings, worst-case sizing and the device early exit not tested
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per input, complete router logits of the library kernel, of a GPU fp32 matmul (IEEE fp32 MFMA, no TF32) and of an fp64 matmul on the same inputs] MM64-FP4 at 2048 and at 4096 tokens and MM256-FP4 at 1024, 4 input distributions × 8 seeds each: the kernel's L2 error against fp64 no larger than the fp32 matmul's in 96 of 96 inputs, its max-abs error in 93 (MiniMax-M2 MXFP4 MoE kernels)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per input, as in E14] 13 router branches at selected sizes (among them the VALU routers of MM256-FP8 at 1, 2, 4 and 16 tokens, the fp32 MFMA of MM256-FP8 at 32 and of MM256-FP4 for 2048–4096, for 6144 and for 8192 tokens, the bf16 three-term MFMA of MM64-FP4 for 6144 and for 8192 and of MM256-FP8 and MM64-FP8 for 1024–16384), a seed-0 random and a cancellation input each (26 inputs), plus two inputs built with nonzero weight residuals; on the 26 the kernel's error against fp64 no larger than the fp32 matmul's in L2 on 25 (the cancellation input of MM256-FP4 for 6144 tokens 1.0000934×), in max-abs on 23 (the cancellation inputs of MM256-FP4 for 2048–4096, for 6144 and for 8192 tokens 1.0319×, 1.0018×, 1.0451×), element by element on 3; in the two residual inputs (MM64-FP4 at 6144 and at 8192 tokens) 16 flags and 256 residual entries set, the fp64 correction ran, logits 0 without it and 2^-23 with it, equal to fp64 and to the fp32 matmul: error 0 on both (MiniMax-M2 MoE kernels)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compile only: `triton.compile` for gfx950 without a GPU, each size's constexprs and launch options; rechecked in later compiles] MM256-FP4/m1's gate-up at 1 token: K fold 4 compiles (160 VGPRs, 24 scaled MFMAs); K fold 1 with 4 warps and K tile 128 kills the compiling process with SIGSEGV (with 16 or 4 rows, transposed or not); K fold 1 compiles with 1 warp (K tile 128 or 256), with 4 warps and K tile 256, or without `llvm_fn_attrs` (29 VGPRs, 24 scaled MFMAs), and every crashing variant carried the `iterative-ilp` attribute MM256-FP4/m1 passes at m ≤ 2; MM256-FP4/m8192's gate-up at 8192 tokens 146 VGPRs with `waves_per_eu` 3 and 150 without, no spills (two compiles); `gl.atomic_add` gives one `global_atomic_add`, no buffer atomic; `gl.amd.cdna4.buffer_atomic_add` on uint32 or uint64 and `buffer_atomic_or`, `and`, `xor` on int32 or int64 compile, all 10 dtype mismatches raise at trace time, `buffer_atomic_xchg` on int32 or int64 traces and then fails in LLVM translation (no `llvm.amdgcn.raw.ptr.buffer.atomic.exch`); `gl.associative_scan` on the `SliceLayout` of MM256-FP4/m16384's scatter prefix (§1; [triton/moe](../triton/moe.md) §4.3) fails to lower ("TODO: unsupported scan layout") and compiles after a `convert_layout` to a 1-D blocked layout, equal to `torch.cumsum`; DS512/m1–m2's expert kernel at 2 tokens compiles with `assert_trivial` on both operand conversions, row-major words raise; shuffled MXFP4 words become the [16, 16, 128] A operand with no instruction and no LDS, row-major ones take 8 `ds_write_b128`, 8 `ds_read_b128` and 8192 B of LDS; a `.wt` `gl.store` is followed by a wait after each of 4 stores, a `.wt` `buffer_store` by none
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the self-resetting counter N=3), Gluon fold and merge kernels in CUDA graphs over 4–8 XCDs, replay time and the ruler's timer (Kineto)] 4, 16, 64 splits × 4096, 16384, 65536 fp32 outputs, 64 outputs per one-wave program: counters zeroed by an earlier node, 2000 of 2000 replays right with `acq_rel` and with `relaxed` at each size, the `relaxed` fold reading a scratch the reference had just filled with the same partials (E18); without a reset, 1999 of 1999 later replays wrong; the merge launch 0.03–0.65× the replay time of the `acq_rel` fold and 0.51–0.92× of the `relaxed` one, 0.12–0.82× on the ruler's timer at up to 16 splits and 16384 outputs, each fold's time including its zeroing node; a counter the last arrival reset itself: 1000 of 1000 replays right in each of 3 runs (Gluon fold probes)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, Gluon fold kernels in CUDA graphs, 3000 replays per arm and size, the partials set to NaN before every replay, counters zeroed by an earlier node] 8 and 32 splits × 16384 and 262144 fp32 outputs: `relaxed` 3000 of 3000 replays wrong at each size, `acq_rel` and `relaxed` with `.wt` partial stores 0 of 3000; one `buffer_wbl2` and one `buffer_inv` in each `acq_rel` fold, none in the `relaxed` ones (Gluon fold probes)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (input mutation and capture N=3), a source scan of the launch code for writes to inputs by name, `global` statements and synchronizing calls (CPU), and the captured checks of E20 on one GPU of planted copies, the read-only inputs compared byte for byte around each call; `torch.cuda.graph` captures of single torch calls with `capture_error_mode="global"`] `x.add_(1)` in `run` and `weight.add_(1)` in a helper taking the input's name were found by the scan, and the same write through a helper parameter named `buf` passed it and failed the byte comparison at the first capture on all 14 RMSNorm sizes; `global` and `nonlocal` were found by the scan, a module-level dict written in `run` passed it; planted one at a time at the top of `run`, `.item()`, `.cpu()`, `.tolist()` and `torch.cuda.synchronize()` were found by the scan and `torch.nonzero`, `x[mask]`, `torch.argwhere` and `repeat_interleave` without `output_size` passed it; each of the eight, called alone on a CUDA tensor after one eager call, failed the capture, two controls (`repeat_interleave` with `output_size` among them) captured (plain-Triton RMSNorm copies for the scan and the byte comparison; the captures ran the torch calls alone, in no kernel's launch code)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the two-phase grid N=4), launches of small Gluon probe kernels with the binaries counted in the JIT cache; planted copies on one GPU checked against the op's reference in captured graphs with poisoned outputs (a first capture and two replays with new inputs, fresh captures per input set, a repeat, a check after the timing) and timed by the ruler, and direct calls against the op's reference; two toy Triton elementwise phases of 131072 elements as two launches and as one grid split by program id, timed as a CUDA graph replayed back to back (wall time) and by the ruler's timer (Kineto)] `INSTR=[16, 16, 32]` as a constexpr failed at the launch with `unhashable type: 'list'` while `triton.compile` of the same kernel succeeded, and a tuple launched; scales 0.5, 1e-6, 3.0, 2^-20 with flags True and False: 1 binary; a 1 KiB view at offset 0 of a 4 KiB buffer got `tt.divisibility = 16` and `tt.pointer_range = 32`, at offset 8 only the range, a 2 GiB + 16 KiB tensor only the divisibility, 3 binaries, `is_within_2gb` called twice per launch of a two-tensor kernel, cache hits included; an RMSNorm copy with `epsilon` hard-coded to 1e-5 passed every check at its sizes, and called with `epsilon` 1e-3 on 64 × 6144 inputs scaled by 0.01 had a maximum relative error of 2.235, the original kernel 0; a copy of an fp8 GEMM (N 3072, m = 16) storing 8 of its 16 rows: wrong values at the first capture with 24576 = 8 × 3072 non-finite mismatches, the original none (one run); the poison filled uint8, int32 and fp4 outputs with 0xFF bytes, fp32 and bf16 with NaN; a `run` keeping its first output in a module-level dict passed a source scan and gave wrong values at the first capture, the fresh captures, the repeat and the check after the timing; a math error: wrong values at the captures and after the timing; a changed reduction order: the same in one run, only a slower time in the other; state kept across calls: wrong values after the timing only; no difference between repeated runs in any; the two phases as one grid 6.33 / 4.30, 5.66 / 4.38, 5.66 / 4.33, 9.191 / 4.443 µs (replayed / the ruler's timer) against two launches 7.42 / 8.49, 7.39 / 8.57, 7.43 / 8.50, 8.471 / 8.610 µs, replay ratio 0.854, 0.766, 0.762, 1.085, outputs equal in every run (copies of a plain-Triton RMSNorm at hidden 6144 and of an fp8 GEMM, probe kernels)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler on one GPU (kernel and baseline in CUDA-graph replays behind a 1 GiB read flush, Kineto kernel sum): a copy with one change alternated with the unchanged kernel, 4 runs each (2 per DSA kernel for `allow_flush_denorm`); an unchanged kernel 3 times against the baseline and 3 times against itself; a slower copy in 4 alternated pairs; offline compiles for gfx950 of the crashing launch] DSA sparse MLA decode, 64 heads, 24 tokens with KV splits 8 → 4: `CompilationError … NameError('cache_shared is not defined')`, in 4 of 4 runs; MM256-FP4 at 1 token with its gate-up fold 4 → 1: the child process exited with code −11, in 4 of 4; the unchanged kernels passed 4 of 4; that FOLD = 1 launch compiled alone without a GPU segfaulted in `make_amdgcn` (2 of 2) and compiled without `llvm_fn_attrs` (2 of 2), as did the kernel's 16-token launch; fp8 GEMM N 3072 × K 1536 at m = 16 with 5 → 3 prefetched K groups: speedup 1.475 → 1.285, kernel 4.232 → 4.851 µs, every pair 1.143–1.150× slower; gated attention decode at 32 tokens with KV splits 16 → 8: 1.79 → 1.129, 44.18 → 69.77 µs, every pair 1.574–1.586× slower; Qwen2-MoE router at 256 tokens without `["amdgpu-ieee", "false"]` on its routing launch: 1.910 → 1.921 (B/A 1.006), the unchanged kernel's four runs spread 0.94 %; DS512 at 8 tokens with `enable_fp_fusion` False → True on its 3 launches: 1.147 → 1.146 (B/A 0.9994); `allow_flush_denorm=True` removed from the 4 DSA 64-head decode kernels (12, 16, 24, 32 tokens): B/A 1.0007, 1.0033, 0.9986, 0.997, instructions and `.amdhsa_` fields identical in 8 of 8 kernels; accuracy passed in every run; the m = 16 fp8 GEMM's ratio to its baseline 1.4784, 1.4386, 1.4749 (max over min 1.028), the 256-token Qwen2 router's 1.9259, 1.9242, 1.9247 (1.001); an identical copy of each timed against the unchanged kernel: 0.999, 0.999, 1.000 for both; fp8 GEMM N 3072 × K 1536 at m = 4096 with 4 → 1 M tiles per group of its tile order: 25.648, 25.643, 25.661, 25.648 against 25.932, 25.974, 25.971, 25.951 µs, slower in 4 of 4 pairs (+1.20 %), timed against the unchanged kernel 0.988 (the kernels named)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, probe kernels on one GPU, against a float64 model] the recipe of §8 ([triton/numerics](../triton/numerics.md) §3) on 100000 groups (randn · 2^k, k from −100 to 100, tie rows): 0 wrong code and scale bytes against ceil e8m0 + RNE e2m1; ceil exponent on 104000 maxima from 2^-122 to 2^127: 0 wrong, amax/scale in (3, 6]; even exponent on 105564 maxima from 2^-126 to below 1.75·2^127: 0 mismatches against aiter's rule, amax/scale in [3.5, 7) from 2^-124 to 2^125; 100256 maxima from 2^-120 to 2^121: the rules differ on exactly the 25551 with mantissa in (1.5, 1.75), ceil at (3.0000002, 3.4999998)·scale, even at (6.0000005, 6.9999995)·scale; 99900 Gaussian groups scaled by 2^-100 to 2^100: ceil 0 and even 19937 elements above 6·scale, SQNR 18.74 and 18.99 dB; the MXFP8 group quantization's reference formula on 100000 groups 0 wrong bytes and codes, `amax / 448` giving the same bytes; power-of-two scaling 0 differ on 200000 values with normal results (e in [−60, 60]); `gl.exp2(e)` exact for all 254 integers e in [−126, 127]; `v_cvt_scalef32_pk_fp4_f32` with S2 = 1.0: 0 mismatches over 73347 pairs against ties-to-even, byte select 0 to 3 right with the other bytes unchanged, ±100 to 0x7/0xF with FP16_OVFL 0 and 1; S2 = 4.0 gave the codes of x/4; attention P at 8 tokens (DSA, Qwen3.5): in fp16 max relative error 4.865e-4 for P ≥ 2^-14 (282 of 131072 below); all 254 finite E4M3FN codes exact in bf16 and fp16; e = −127: `0 · gl.div_rn(1, gl.exp2(e))` NaN, the two-factor and from-bits forms 0; decay 0.5 per token: prefixes 0 at 170 and 200 tokens, their ratios NaN, the direct products 2^-10; 65536 decays below 2^-126: |state·exp − state·exp2| at most 0.94 of 2^-126·|state|; x = −88.5: `gl.exp` 3.672e-39 with reciprocal 2.723e38, `gl.exp2` 0 with Inf; x = −100: both reciprocals Inf; 16384 merges of an empty split with a finite one, both orders: weight 0, result bit-equal; `gl.maximum` of [qNaN, sNaN, 1.0] with a −inf loaded from memory: [−inf, −inf, 1.0]; 1000064 normal fp32 values, none ±0: `gl.maximum`, `gl.max` and `gl.clamp(±448)` bit-equal to torch; a 6-program fp32 atomic sum into one cell: 3 bit patterns in 300 runs, 281 of 299 consecutive pairs identical (probe kernels, not library kernels)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] MiniMax-M2 MXFP4 MoE, both in one run: lowest 1.178 at MM256-FP4 (intermediate dim 512), 256 tokens, highest 3.633 at MM64-FP4 (intermediate dim 1536), 1 token (model dim 3072, 256 routed experts)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, the ruler twice per run; routing simulated from the workload's router distribution and bias with random normal hidden states] MiniMax MXFP4, 256 experts, at 64 tokens: experts touched per call 77.19% on average, 74.61% at the least (uniform-routing estimate 86.9%), 495.3 MB of expert weights; call 99.09–101.7 µs against the baseline's 127.0–127.8 µs (speedup 1.249–1.285), so 4.87–5.00 TB/s against 3.88–3.90 TB/s for those bytes
- S1 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:327 `gluon_to_ttgir`: no buffer-op conversion and no pipeliner (to :348; `make_ttgir` converts at :306); `launch_cooperative_grid: bool = False` (:95), launched with `hipModuleLaunchCooperativeKernel` (third_party/amd/backend/driver.c:674-677); triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:2278 the redundant-thread predicate of an atomic, ANDed into its mask at :2338; `acq_rel`, the default of `atomic_add` (python/triton/language/core.py:2731), makes the backend insert buffer invalidates and L2 write-backs (:2354)
