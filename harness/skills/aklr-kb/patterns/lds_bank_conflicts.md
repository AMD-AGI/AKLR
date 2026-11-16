---
title: "LDS bank conflicts and LDS-side stalls"
kind: pattern
signals: ["SQ_LDS_BANK_CONFLICT above the hand-computed floor", "stall on the ds_read / ds_write instructions themselves",
  "s_waitcnt lgkmcnt(0) stall right after a ds_write", "SQ_WAIT_INST_LDS", "SQ_LDS_IDX_ACTIVE",
  "ds_read2_b32 where a ds_read_b128 was meant"]
ops: [gemm]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
---

# LDS bank conflicts and LDS-side stalls

## Symptom

- `SQ_LDS_BANK_CONFLICT` above the floor computed by hand for the access pattern (S1).
- In the ATT trace, stall on the `ds_read_*` / `ds_write_*` instructions themselves (S2), or on `s_waitcnt lgkmcnt(0)`
  right after a `ds_write` (S3). `ds_read2` offsets that are multiples of 64 are not a conflict on gfx950, although
  FlyDSL's lds-optimization skill lists them as one (S4; `../flydsl/guides.md` § lds-optimization).
- A high `SQ_WAIT_INST_LDS` share of the wave-cycle budget (S5).

## Likely causes

1. **Timing, not layout.** The three LDS failure modes each have their own fix (S6). A bank conflict inflates the
   `lgkmcnt` wait after a `ds_write` as a too-short write-to-read distance does: check the `ds_*` rows and
   `SQ_LDS_BANK_CONFLICT` before changing the distance (S7).
2. **Layout.** On gfx950, `b64`/`b128` reads see bank = `(byte_addr / 4) mod 64` (S8), but a `ds_read_b32` conflicts as
   two 32-lane passes over 32 banks: a lane stride of `s` dwords is a `gcd(s, 32)`-way conflict, so a 128-byte row
   stride and every multiple of it cost the same 32-way conflict in a HIP microbenchmark (S9); LDS writes also
   conflict as on 32 banks (`../hardware/gfx950/memory_and_sync.md` §4.2; `b128` in
   `../flydsl/kernels/flydsl_authoring_method.md`). An XOR inside a vector scrambles its lanes (S11).
3. **Not LDS:** `lgkmcnt` also counts scalar loads, so a long `lgkmcnt(0)` stall may be an `s_load`; keep scalar
   loads out of that window (S12; `gemm/datapath.md` §4).
4. **Co-resident wave mapping.** In one grouped fp8 kernel the wave bijection that put both waves of a SIMD on the
   same fragment of the heavier operand ran ~25% slower, with only `SQ_WAIT_ANY / SQ_WAVE_CYCLES` moving (S13, S14);
   in a double-buffered HIP bf16 GEMM *(field, E2)* and in LDS→MFMA loops (`gemm/schedule_and_locality.md` §1) the same
   mapping changed nothing, so the cause is not established: time the bijections.
5. **Capacity:** LDS per workgroup caps occupancy (S15).
6. **Split reads:** `ds_read2_b32` or `ds_read2_b64` in the ISA where a `ds_read_b128` was meant: the 16-byte load was
   declared only 4- or 8-byte aligned; declare 16 (`../flydsl/language/limits.md`, S32).

## Rule out first

- **A wait count is not a bound.** Co-resident waves can hide `SQ_WAIT_INST_LDS`; decide with a halve-the-reads probe
  (S5, S16). `SQ_LDS_IDX_ACTIVE` over `SQ_VALU_MFMA_BUSY_CYCLES` only gives the LDS-to-MFMA instruction mix; over
  CU-cycles it tracked the LDS port load in a HIP microbenchmark (`gemm/attribution.md` §6). Zero conflicts is not a
  speedup (S17).
- **The floor's lane phases** for `b64` / `b128` are `gemm/datapath.md`'s, not the ISA's (S18).
- **LDS placement.** Reordering `@fx.struct` fields moves only fields of equal size: read `group_segment_fixed_size`
  first (S19, S33).

## Candidate techniques

- More write-to-read distance — `../flydsl/guides.md` § lds-optimization (Method 3) — for cause 1 (S20); with little VGPR
  headroom, moving the write paid where inserting work lost on one register-full kernel,
  `gemm/schedule_and_locality.md` §5 (S21): race both.
- XOR swizzle on 16-byte slots, composed into the layout — `gemm/datapath.md` §2 — one function keys the
  writer and the reader (S22); re-measure it, a derived swizzle is not automatically better (S23).
- Padding — `gemm/datapath.md` §3, `../flydsl/guides.md` § lds-optimization (Method 2) — when a small stride bump
  removes the conflict (S24); pad by whole 16-byte slots to keep `b64`/`b128` alignment (S25); for column-wise
  MFMA fragment reads the swizzle is easier (S26).
- A wave term in the transpose-read key — `gemm/datapath.md` §2 — when a per-wave chunk stride puts
  every wave on the same banks (S27); change a transpose read and its LDS layout together (§9) (S28).
- Re-map waves so co-resident pairs do not share the heavy fragment — `gemm/schedule_and_locality.md`
  §1 — state the `wave_id → (wave_m, wave_n)` mapping explicitly (S29).
- Fewer LDS reads per MAC with a 2×2 wave split — `gemm/mainloop.md` §1 — on long K; race it against
  the 8-wave design (S30, S31).
- Attention's transposed reads: whole 16-element blocks, 16-byte-aligned row strides, and why its bank-conflict
  counters mislead — `attention/lds.md` §4–6; keep write bursts out of transposed-read bursts — §9.

## Evidence

- E2 [2026-11, gfx950 (MI355X), HIP 7.15 (hipcc), rocprofv3, N=15 rounds in one process, arms rotated; A-vs-A within 0.4–1.3%] co-resident waves w and w + 4 sharing the A fragment (`wave_m`, 8 of each wave's 12 `ds_read_b128`) against sharing `wave_n`: −0.15% to +0.06% at three shapes, `SQ_LDS_BANK_CONFLICT` and `SQ_LDS_IDX_ACTIVE` identical (hand-written bf16 GEMM, 128×128 tile of 8 waves, A and B double-buffered in LDS)
- S1 gemm/datapath.md:94 compare SQ_LDS_BANK_CONFLICT against the floor computed by hand, not zero
- S2 ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md:143 Type A bank conflicts: high stall on ds_read/ds_write themselves
- S3 ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md:157 Type B write-read latency exposed: lgkmcnt(0) stall after ds_write
- S4 ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md:155 gfx950: ds_read2 offsets that are multiples of 64 hit the same bank
- S5 gemm/attribution.md:142 SQ_WAIT_INST_LDS counts dependent waits; co-resident waves can hide them
- S6 gemm/datapath.md:65 tell the three LDS failure modes apart; each has a different fix
- S7 gemm/datapath.md:73 a wait spike after a ds_write does not prove exposed write-to-read latency: a bank conflict also inflates that lgkmcnt wait
- S8 gemm/datapath.md:76 64 banks of 4 bytes; wide reads see bank = (byte_addr / 4) mod 64
- S9 gemm/datapath.md:80 ds_read_b32 at a lane stride of s dwords; a 128-byte row stride (s = 32) is a 32-way conflict
- S11 gemm/datapath.md:47 XOR at element granularity scrambles the lanes of ds_read_b64/b128
- S12 gemm/datapath.md:115 a long lgkmcnt(0) stall may be an s_load; keep scalar loads out of that window
- S13 gemm/schedule_and_locality.md:47 one grouped fp8 kernel: the co-resident pair sharing wave_m (the A fragment) ~25% slower
- S14 gemm/schedule_and_locality.md:51 LDS, bank-conflict and MFMA counters identical, cache counters within 0.3%; only SQ_WAIT_ANY / SQ_WAVE_CYCLES moved
- S15 gemm/datapath.md:69 capacity: LDS per workgroup caps occupancy
- S16 gemm/attribution.md:144 halve-the-reads probe; SQ_LDS_IDX_ACTIVE over SQ_VALU_MFMA_BUSY_CYCLES follows the instruction mix, not the LDS port load
- S17 gemm/datapath.md:95 zero conflicts is not a speedup; LDS is not the bound until a probe says so
- S18 hardware/gfx950/memory_and_sync.md:284 gemm/datapath.md's per-width lane phases are not from the ISA
- S19 flydsl/gemm.md:507 reordering @fx.struct fields moves only fields of equal size; each leaf is its own symbol placed by the backend
- S20 ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md:384 Method 3: increase write-read distance
- S21 gemm/schedule_and_locality.md:247 rule of thumb from one register-full kernel: below ~15% VGPR headroom, move the write, do not insert work
- S22 gemm/datapath.md:54 one function computes the swizzle key for both sides
- S23 gemm/datapath.md:51 re-measure a swizzle change; derived is not automatically better
- S24 gemm/datapath.md:84 padding when a small stride bump removes the conflict
- S25 gemm/datapath.md:90 padding must keep the 8- or 16-byte alignment; pad by whole 16-byte slots
- S26 ../flydsl/guides.md:45 column-wise reads such as MFMA fragments: padding hard to use, prefer XOR
- S27 gemm/datapath.md:57 transpose reads need a wave term: per-wave chunk stride hits the same banks
- S28 gemm/datapath.md:271 the transpose instruction and the LDS layout must change together
- S29 gemm/schedule_and_locality.md:54 state the wave_id → (wave_m, wave_n) mapping explicitly
- S30 gemm/mainloop.md:78 why the 2×2-wave tile wins on long K: fewer LDS reads per MAC
- S31 gemm/mainloop.md:89 compile both designs and race them; the crossover depends on the shape
- S32 ../flydsl/language/limits.md:120 a 16-byte LDS load declared 4- or 8-byte aligned becomes `ds_read2_b32` / `ds_read2_b64`; `Vector.load` passes no alignment; declaring 16 cut a MoE epilogue's bank conflicts 4.2–4.9×
- S33 gemm/correctness.md:158 separate LDS buffers are laid out by the backend; read group_segment_fixed_size first
