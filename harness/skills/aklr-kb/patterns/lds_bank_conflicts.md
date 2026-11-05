---
title: "LDS bank conflicts and LDS-side stalls"
kind: pattern
signals: ["SQ_LDS_BANK_CONFLICT above the hand-computed floor", "stall on the ds_read / ds_write instructions themselves",
  "s_waitcnt lgkmcnt(0) stall right after a ds_write", "SQ_WAIT_INST_LDS", "SQ_LDS_IDX_ACTIVE", "ds_read2 offsets that are multiples of 64",
  "ds_read2_b32 where a ds_read_b128 was meant"]
ops: [gemm]
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
---

# LDS bank conflicts and LDS-side stalls

## Symptom

- `SQ_LDS_BANK_CONFLICT` above the floor computed by hand for the access pattern (S1).
- In the ATT trace, stall on the `ds_read_*` / `ds_write_*` instructions themselves (S2), or on `s_waitcnt lgkmcnt(0)`
  right after a `ds_write` (S3); `ds_read2` offsets that are multiples of 64 on gfx950 (S4).
- A high `SQ_WAIT_INST_LDS` share of the wave-cycle budget (S5).

## Likely causes

1. **Timing, not layout.** The three LDS failure modes each have their own fix (S6); first check whether a `ds_write`
   sits too close to the `ds_read` that depends on it (S7).
2. **Layout.** On gfx950, bank = `(byte_addr / 4) mod 64` (S8); a `ds_read_b32` whose lane stride is `s` dwords
   touches `64 / gcd(s, 64)` banks, so a 128-byte row stride puts every lane on two banks (S9). An XOR inside a
   vector scrambles its lanes (S11).
3. **Not LDS:** `lgkmcnt` also counts scalar loads, so a long `lgkmcnt(0)` stall may be an `s_load`; keep scalar
   loads out of that window (S12; `gemm/datapath.md` §4).
4. **Co-resident waves in phase.** Two waves on one SIMD reading the same fragment of the heavier operand at the same
   moment hide nothing for each other (S13); LDS and bank-conflict counters stay identical and only
   `SQ_WAIT_ANY / SQ_WAVE_CYCLES` moves (S14).
5. **Capacity:** LDS per workgroup caps occupancy (S15).
6. **Split reads:** `ds_read2_b32` or `ds_read2_b64` in the ISA where a `ds_read_b128` was meant: the 16-byte load was
   declared only 4- or 8-byte aligned; declare 16 (`../flydsl/language/limits.md`, S32).

## Rule out first

- **A wait count is not a bound.** Co-resident waves can hide `SQ_WAIT_INST_LDS`; decide with a halve-the-reads probe
  and compare `SQ_LDS_IDX_ACTIVE` with `SQ_VALU_MFMA_BUSY_CYCLES` (S5, S16). Zero conflicts is not a speedup (S17).
- **The floor's lane phases** for `b64` / `b128` are the card's, not the ISA's (S18).
- **LDS placement.** Reordering `@fx.struct` fields does not move LDS: read `group_segment_fixed_size` first (S19, S33).

## Candidate techniques

- More write-to-read distance — `../flydsl/guides.md` § lds-optimization (Method 3) — for cause 1 (S20); with little VGPR
  headroom move the write instead of inserting work, `gemm/schedule_and_locality.md` §5 (S21).
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
  counters mislead — `attention/flydsl_attn_lds.md` §4–6; keep write bursts out of transposed-read bursts — §9.

## Evidence

- S1 gemm/datapath.md:92 compare SQ_LDS_BANK_CONFLICT against the floor computed by hand, not zero
- S2 ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md:143 Type A bank conflicts: high stall on ds_read/ds_write themselves
- S3 ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md:157 Type B write-read latency exposed: lgkmcnt(0) stall after ds_write
- S4 ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md:155 gfx950: ds_read2 offsets that are multiples of 64 hit the same bank
- S5 gemm/attribution.md:137 SQ_WAIT_INST_LDS counts dependent waits; co-resident waves can hide them
- S6 gemm/datapath.md:65 tell the three LDS failure modes apart; each has a different fix
- S7 gemm/datapath.md:73 check the write-to-read distance before the layout
- S8 gemm/datapath.md:76 64 banks of 4 bytes; bank = (byte_addr / 4) mod 64
- S9 gemm/datapath.md:80 64 / gcd(s, 64) banks; a 128-byte row stride lands every lane on two banks
- S11 gemm/datapath.md:47 XOR at element granularity scrambles the lanes of ds_read_b64/b128
- S12 gemm/datapath.md:113 a long lgkmcnt(0) stall may be an s_load; keep scalar loads out of that window
- S13 gemm/schedule_and_locality.md:37 co-resident pair reading the same heavy fragment: in phase
- S14 gemm/schedule_and_locality.md:51 counters bit-identical; only SQ_WAIT_ANY / SQ_WAVE_CYCLES moved
- S15 gemm/datapath.md:69 capacity: LDS per workgroup caps occupancy
- S16 gemm/attribution.md:139 halve-the-reads probe; SQ_LDS_IDX_ACTIVE vs SQ_VALU_MFMA_BUSY_CYCLES
- S17 gemm/datapath.md:93 zero conflicts is not a speedup; LDS is not the bound until a probe says so
- S18 hardware/gfx950/memory_and_sync.md:271 the card's per-width lane phases are not from the ISA
- S19 flydsl/gemm.md:412 reordering @fx.struct fields does not move LDS; each leaf is its own symbol placed by the backend
- S20 ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md:384 Method 3: increase write-read distance
- S21 gemm/schedule_and_locality.md:220 when VGPR headroom is small, any insertion lengthens a live range
- S22 gemm/datapath.md:54 one function computes the swizzle key for both sides
- S23 gemm/datapath.md:52 re-measure a swizzle change; derived is not automatically better
- S24 gemm/datapath.md:84 padding when a small stride bump removes the conflict
- S25 gemm/datapath.md:88 padding must keep the 8- or 16-byte alignment; pad by whole 16-byte slots
- S26 ../flydsl/guides.md:44 column-wise reads such as MFMA fragments: padding hard to use, prefer XOR
- S27 gemm/datapath.md:57 transpose reads need a wave term: per-wave chunk stride hits the same banks
- S28 gemm/datapath.md:251 the transpose instruction and the LDS layout must change together
- S29 gemm/schedule_and_locality.md:54 state the wave_id → (wave_m, wave_n) mapping explicitly
- S30 gemm/mainloop.md:79 why the 2×2-wave tile wins on long K: fewer LDS reads per MAC
- S31 gemm/mainloop.md:89 compile both designs and race them; the crossover depends on the shape
- S32 ../flydsl/language/limits.md:103 a 16-byte LDS load declared 4- or 8-byte aligned becomes `ds_read2_b32` / `ds_read2_b64`; `Vector.load` passes no alignment; declaring 16 cut a MoE epilogue's bank conflicts 4.2–4.9×
- S33 gemm/correctness.md:147 separate LDS buffers are laid out by the backend; read group_segment_fixed_size first
