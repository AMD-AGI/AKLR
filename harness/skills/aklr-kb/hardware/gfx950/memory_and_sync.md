---
title: CDNA4 ISA — memory instructions, LDS and synchronization for a GEMM data path on gfx950
kind: topic
ops: [gemm, moe, attention]
dtypes: [any]
regimes: [any]
gens: [gfx950]
flydsl: 0.3.2
---

# Memory instructions and synchronization in the CDNA4 ISA

> Everything here comes from the ISA chapters in `cdna4/`. `chNN` / `ch12_NN` stand for `cdna4/chNN_*.md`, `p.N` for
> the `<!-- page N -->` marker. Quoted strings are exact. "Not in the ISA" means the chapters were searched and say
> nothing about it. Reading caveat: the page markers of tables that cross pages are not reliable and some tables
> are misaligned (`lookup.md` §6); check signs that matter, and the tables flagged below, against the PDF page.

## Route here when
- You are choosing or auditing an `s_waitcnt`: what `vmcnt`/`lgkmcnt` count, in what order they retire, their limits.
- A buffer descriptor, `soffset`/`voffset`/immediate, or out-of-range behaviour is the question.
- You are picking `sc0`/`sc1`/`nt` for an operand load or a C store.
- You are writing a global→LDS DMA (`buffer_load … lds`, `global_load_lds_*`) and its M0.
- LDS capacity, banks, `ds_read`/`ds_write` widths, transpose reads or LDS atomics.
- A wait state around M0, EXEC, VCC, an SGPR read by VMEM, or store data.

**Somewhere else if:** encodings, operand fields or opcode numbers → `lookup.md`; MFMA dependencies and
hazards → `matrix_core.md`; what to do about it in a FlyDSL GEMM → the cards in §7.

## 1. `s_waitcnt` and the counters

"Instructions of a given type return in order, but instructions of different types can complete
out-of-order." (ch04 §4.4). Widths are from ch03 §3.1 Table 2.

| Counter | Bits | Incremented | Decremented | Order |
|---|---|---|---|---|
| VM_CNT | 6 | "every time a vector-memory read or write (MUBUF, MTBUF, or FLAT format) instruction is issued" | "for reads when the data has been written back to the VGPRs, and for writes when the data has been written to the L2 cache" | "Memory reads and writes return in the order they were issued, including mixing reads and writes." |
| LGKM_CNT | 4 | 1 per LDS instruction; SMEM reads "1 for 1-dword loads, 2 for 2-dword or larger loads"; 1 per FLAT instruction; 1 per S_SENDMSG | LDS read or atomic-with-return: data returned to VGPRs; LDS write: data written to LDS; SMEM: "for each Dword returned"; message: when sent | same type in order, "except scalar-memory-reads, which can return out-of-order (in which case only S_WAITCNT 0 is the only legitimate value)" |
| EXP_CNT | 3 | — | — | "VGPR-export count: unused" |

- **SMEM.** "the only sensible way to use this counter is to implement S_WAITCNT 0" (ch08 §8.3). While an
  `s_load` is outstanding, the ISA admits only `lgkmcnt(0)`, even for LDS operations issued next to it.
- **FLAT** instructions "increment both VM_CNT and LGKM_CNT and are not considered done until both have been
  decremented", "can complete out of order with each other", and "the only sensible S_WAITCNT value to use
  after FLAT instructions is zero" (ch10 §10.2–§10.2.2). **GLOBAL and SCRATCH:** "only VM_CNT is used, not
  LGKM_CNT" (§10.4, §10.5).
- **Encoding** (ch12_05 S_WAITCNT, opcode 12): `SIMM16[3:0]` vmcnt low bits, `[6:4]` "export/mem-write-data
  count", `[11:8]` LGKMcnt, `[15:14]` vmcnt bits [5:4]. The ISA gives widths, not maxima; they give vmcnt ≤ 63,
  lgkmcnt ≤ 15, expcnt ≤ 7. The live counts can be read through `S_GETREG` hwreg 7 `IB_STS` (ch05 Tables 19–20).
- **No split counters.** The SOPP opcode table (ch13 §13.1.5 Table 74, opcodes 0–29) has one wait, S_WAITCNT;
  `s_waitcnt_vscnt` and `s_wait_*` are not in this ISA. Stores share VM_CNT and its issue order with loads.
  S_ENDPGM "implicitly executes S_WAITCNT 0"; S_BARRIER waits on no counter (§5 below).
- **Gap.** When VM_CNT decrements for a load whose data goes to LDS (ch04 §4.4 speaks only of "written back to
  the VGPRs"), and whether such a load touches LGKM_CNT, are not in the ISA (§3 below).

## 2. Buffer and global loads and stores

### 2.1 Buffer resource (V#)
128 bits in "four consecutive SGPRs (four aligned SGPRs)" (ch09 §9.1.8 Table 48):

| Bits | Field | ISA |
|---|---|---|
| 47:0 | Base address | "Byte address." |
| 61:48 | Stride | "Bytes 0 to 16383"; 18 bits with add-tid on non-format MUBUF (§9.1.5 Table 45) |
| 62 / 63 | Cache swizzle / Swizzle enable | swizzle "TC L1 cache banks" / swizzled AOS, else "linear (stride * index + offset)" |
| 95:64 | Num_records | "In units of stride or bytes." |
| 110:108, 114:111 | Num format, Data format | used by FORMAT ops; BUFFER_LOAD/STORE_<size> derive both from the opcode (§9.1.4 Table 43) |
| 119 / 127:126 | Add tid enable / Type | adds the lane id to the index / "Value == 0 for buffer." |

"A resource set to all zeros acts as an unbound texture or buffer (return 0,0,0,0)." And "The resource’s data
format must not be INVALID; that format has specific meaning (unbound resource)" — for that case the opcode's
implied format does not replace it (§9.1.4), so check Data format on a hand-built descriptor.

### 2.2 Address
§9.1.1 gives the formula only as an image (`cdna4/images/x346.png`, transcribed; every term is added):
`ADDR = Base(V#) + baseOffset(SGPR) + Inst_offset(instr) + Voffset(VGPR) + Stride(V#) * (Vindex(VGPR) + TID)`.
"Voffset is ignored when instruction bit "OFFEN" == 0"; Vindex is ignored when IDXEN == 0; TID (0..63) "is
ignored when resource bit ADD_TID_ENABLE == 0". Figure 4 in §9.1.5 has the same composition.
- Address VGPRs (Table 42): `offen` only → offset in VADDR; `idxen` only → index; both → index in VADDR,
  offset in VADDR+1. `inst_offset` is a 12-bit "Unsigned byte offset" (Table 41) and "is present, regardless
  of" OFFEN (Table 44). SOFFSET "Must be an SGPR, M0, or inline constant" (Table 41). SGPR and VGPR offsets are unsigned 32-bit (Table 46).
- "For Dword or larger reads or writes, the two LSBs of the byte-address are ignored, thus forcing Dword
  alignment." (§9.1.7). No larger alignment is stated for `dwordx2`/`x4`.

### 2.3 Out of range (§9.1.5.1)
"When an address is out of range, reads return zero, and writes and atomics are dropped."

| Buffer type | Used when | Out of range if |
|---|---|---|
| Raw | `AddTID==0 && SWizzleEn==0 && IdxEn==0` | `(InstOffset + (OffEN ? vgpr_offset : 0)) >= NumRecords` |
| Structured | `AddTID==0 && Stride!=0 && IdxEn==1` | `Index(vgpr) >= NumRecords` |
| Private (scratch) | `AddTID==1 && IdxEn==0` | "no range checking" |

- The base and `soffset` take no part in the raw range check: an address raised only by `soffset` is never judged
  out of range.
- "Load/store-format-* instruction and atomics are range-checked "all or nothing""; "Load/store-Dword-x{2,3,4}
  and range-check per component", so a `dwordx4` that straddles the end keeps its in-range dwords.
- Check p.86: Table 45 says num_records is "In units of: (inst_idxen == 1) ? Bytes : Stride", the reverse of the
  sentence before it ("Bytes for raw buffers, units of Stride for structured buffers") and of the checks above.

### 2.4 Cache-control bits (§9.1.10.2, Tables 49–50)
"SC[1:0] System Cache level: 0=wave, 1=group, 2=device, 3=system"; "NT Non-Temporal: 0=expect temporal reuse;
1=do not expect temporal reuse". This covers "Flat, Global, Scratch, and Buffer". Cells read CU / L2 / last level.

| Scope | SC1 SC0 | NT=0 load | NT=0 store | NT=1 (load and store) |
|---|---|---|---|---|
| Wave | 0 0 | Hit LRU / Hit LRU / Hit LRU | Miss LRU / Hit LRU / Hit LRU | Miss Evict / Hit Stream / Hit Evict |
| Group | 0 1 | Hit LRU / Hit LRU / Hit Evict | Miss LRU / Hit LRU / Hit LRU | Miss Evict / Hit Stream / Hit Evict |
| Device | 1 0 | Miss Evict / Hit LRU† / Hit LRU | as load | Miss Evict / Hit Stream† / Hit Evict |
| System | 1 1 | Miss Evict / Coherent Cache Bypass / Hit LRU | as load | Miss Evict / Coherent Cache Bypass / Hit Evict |

† "(1 L2 cache)"; with "(>1 L2 cache)" the L2 cell is "Coherent Cache Bypass". The ISA does not say which case
gfx950 is. Loads: "if TG_SPLIT mode is active, SC1==0, SC0==1, NT==0 case becomes: Miss LRU, Hit LRU, Hit LRU."
- Atomics: "SC0 indicates whether or not to return the "pre-op" memory value"; "SC1 : 0 = device scope
  atomic, 1 = system scope atomic"; "NT : 0 = last level cache "allocate" policy; 1 = "no Allocate" policy".
  FLAT/global float atomics "must set SC[0]=0 (no return value)" (ch10 §10.3.1).
- BUFFER_WBL2 (opcode 40, Table 51): `SC1=0` NOP; `SC1=1` writes back dirty L2 data (with `SC0=0`, only with more
  than one L2). BUFFER_INV (41, Table 52): `SC1=1` invalidates the CU cache, and the L2's "non-coherently cached
  lines" when `SC0=1` or with more than one L2; `SC1=0`, `SC0=1` invalidates the CU cache in TG Split mode and is
  a NOP otherwise. Both "Returns ACK to shader" (ch12_13). Scalar memory has "a single memory control bit: GLC"
  (ch09 §9.1.10.1).

### 2.5 FLAT, GLOBAL, SCRATCH (ch10)

| | FLAT | GLOBAL | SCRATCH |
|---|---|---|---|
| Address | VGPR address; apertures decide global / scratch / LDS | "VGPR-address + instruction offset" or "SGPR-address + VGPR-offset + instruction offset" (32-bit VGPR offset; SADDR `0x7F` disables) | `flat_scratch.addr + swizzle(V/SGPR_offset + inst_offset, threadID)` |
| Immediate | "12-bit unsigned offset (MSB is ignored)" | "13-bit signed byte offset" | 13-bit signed |
| Counters | VM_CNT and LGKM_CNT; wait for zero | VM_CNT only | VM_CNT only |
| To LDS | "cannot return data directly to LDS" | GLOBAL_LOAD_LDS_* | SCRATCH_LOAD_LDS_* |
| Bad address | unmapped region → "memory-violation error" | touching LDS "returns MEM_VIOL" | no aperture check; no descriptor, so no `num_records` check on any of the three |

## 3. Loading into LDS directly

| Path | Instructions that write LDS | Source |
|---|---|---|
| MUBUF, `LDS=1` | BUFFER_LOAD_{ubyte, sbyte, ushort, sshort, dword, dwordX3, dwordX4, format_x} | ch09 §9.1.9 |
| GLOBAL | GLOBAL_LOAD_LDS_{UBYTE, SBYTE, USHORT, SSHORT, DWORD, DWORDX3, DWORDX4} | ch10 Table 55; ch13 Table 103 |
| SCRATCH | SCRATCH_LOAD_LDS_{UBYTE, SBYTE, USHORT, SSHORT, DWORD} | ch10 Table 55 |
| FLAT | none | ch10 §10.2 |

- No path has an 8-byte (`dwordx2`) variant. Byte and short variants "zero extend [or sign extend] to 32 bits
  and store the result into a data share" (ch12_15). Only loads are described: no opcode table names a store
  from LDS, though the MUBUF LDS bit reads "transfer data between LDS and memory" (ch13 Table 99).
- **MUBUF** (§9.1.9 figure, `cdna4/images/x377.png`, p.92): `LDS_ADDR = LDSbase(HW-Alloc) + LDS_offset(M0) +
  inst_offset + TIDinWave*4`; `MEM_ADDR = Base(T#) + mem_offset(soffset) + inst_offset + off_vgpr + stride*(idx_vgpr
  + TIDinWave)`. "TIDinWave is only added if the resource (T#) has the ADD_TID_ENABLE field set to 1, whereas LDS
  adds it." "For loads to LDS with data-size of 3 or 4 dwords, the equation is modified to be: (TIDinWAve * 16).
  Note that LOAD_DWORDX3 writes 3 dwords and skips the 4th."
- **GLOBAL/SCRATCH** (ch10 §10.3): `LDS_ADDR = LDSbase(hw alloc) + LDSoffset(M0[17:2] * 4) + INST.OFFSET +
  ThreadID*4` (DWORDX1), `… + ThreadID*16` (DWORDX4). M0: "Implied use of M0 for SCRATCH and GLOBAL only when
  LDS=1. Provides the LDS address-offset." (Table 54). How a negative 13-bit `INST.OFFSET` acts on the LDS side is
  not stated.
- **What the formulas imply.** The LDS side is lane-linear: lane *i* writes at +4·*i* (or +16·*i*) from the
  wave-uniform M0 + immediate, so a dword load fills a contiguous 256 B block and an x4 load a contiguous 1 KiB
  block; an x3 load uses the same 16-byte stride but skips every fourth dword (§9.1.9). Per-lane
  freedom exists only on the memory side, and the immediate moves the LDS and the memory address together.
- **M0 width: the ISA disagrees with itself.** §9.1.9 text "18-bit unsigned byte offset from M0[17:0]"; its
  figure `M0[15:0]`; ch03 §3.7 `{14'h0, lds_offset[17:0]} // in bytes`; §10.3 `M0[17:2] * 4`; Table 54 M0 "16" bits.
- **Clamping.** "the return data must not be written outside the LDS space allocated to this wave"; "Set the
  active-mask to limit buffer reads to those threads that return data to a legal LDS location" (§9.1.9.1).
  "Memory Buffer to LDS does NOT return a memory violation if the LDS address is out of range, but masks off
  EXEC bits of threads that would go out of range." (ch03 §3.11). The memory side clamps like any buffer fetch.
- **Counters.** MUBUF and GLOBAL count in VM_CNT (§1). Not in the ISA: when that count retires for LDS data,
  whether a `ds_read` after `vmcnt` is guaranteed to see the DMA's data, and any M0 alignment beyond the dword
  granularity of `M0[17:2] * 4`. The M0-write wait state is in §6.

## 4. LDS (data share)

### 4.1 Capacity, allocation, clamping
- "There are 160 kB memory per compute unit, segmented into 64 banks of 640 Dwords, each bank being 32bits
  wide." "One work-group can request up to 160 kB memory." (ch11 §11.1). A work-group is up to "16 wavefronts
  (1024 work-items)" (ch04 §4.3).
- **Allocation.** LDS is allocated in "contiguous blocks of 1280 bytes on 1280-byte alignment" (ch03 §3.6.5). The
  `LDS_ALLOC` hwreg reports the size "in units of 64 Dwords" (ch05 Table 22), and ch09 §9.1.9.1 speaks of "units
  of 32 Dwords"; neither restates the allocation granule.
- **M0 as clamp.** "M0[16:0] contains the byte-size of the LDS segment. This is used to clamp the final
  address."; "All LDS operations require that M0 be initialized prior to use. … If no clamping is wanted, set M0
  to 0xFFFFFFFF." (§11.3.1; Table 57's rows are column-shifted in the text, check p.105). Out of range is
  "addr < 0 or >= (MIN(lds_size, m0))": writes are discarded, reads return zero (ch03 §3.6.1). `M0[16:0]` cannot
  express 160 kB, and M0 is also the DMA offset (§3); the ISA reconciles neither.

### 4.2 Banks and conflicts
"Dwords are placed in the banks serially, but all banks can execute a store or load simultaneously." "Reads
across wavefront are dispatched over four cycles in waterfall." "it is possible to concurrently execute 32
write or read instructions, each nominally 32-bits; extended instructions, read2/write2, can be 64-bits each."
Conflicting "indexed and atomic operations" become "serial accesses" (§11.1) and take from "two cycles" to "64
cycles" (§11.3.1). **Not in the ISA:** which lanes are serviced together for `b64`, `b128` or transpose reads.

### 4.3 Reads and writes (§11.3.1 Table 58, ch12_12)

| Op | Widths | Offset |
|---|---|---|
| DS_READ_* | B32, B64, B96, B128, U8, I8, U16, I16; U8/I8/U16 `_D16` and `_D16_HI` | 16-bit unsigned `{offset1, offset0}` |
| DS_WRITE_* | B8, B16, B32, B64, B96, B128; B8/B16 `_D16_HI` | same |
| DS_READ2 / WRITE2_{B32,B64} | two addresses | two 8-bit offsets × ADJ, "ADJ = 4 for 8, 16 and 32-bit data types; and ADJ = 8 for 64-bit" |
| …2ST64_{B32,B64} | two addresses | B32 pseudocode × 256 B; B64 pseudocode × 512 B (its prose says "64 elements (256 bytes)") |
| DS_READ/WRITE_ADDTID_B32 | lane-linear | `{OFFSET1, OFFSET0} + M0[15:0] + laneID * 4` |

- `LDS_Addr = LDS_BASE + VGPR[ADDR] + {InstrOffset1,InstrOffset0}`; "both ADDR and InstrOffset are byte
  addresses". LDS instructions obey EXEC (ch03 §3.3).
- "Any DS op reading or writing 64-bit or larger data must use an even-aligned VGPR" (§11.4). The DS ACC bit
  is described only as "VDST is Accumulation VGPR" (ch13 Table 95); DATA0/DATA1 are not mentioned.
- No address alignment is stated for non-transpose DS ops; ch03 §3.11 lists "LDS alignment error" as a
  memory violation.

### 4.4 Transpose reads (§11.4; opcodes 224–227)
"Prior to executing these instructions the EXEC mask must be set to all 1’s. It is required that the LDS
address be aligned to the data size." "DS_READ_B96_TR_B6 does not require even-VGPR alignment." "The operation
takes two instructions with different LDS addresses and VGPR destinations." Each is "Used for either column
major matrix A or row major matrix B data load".

| Instruction | Element | VGPRs | First instruction loads |
|---|---|---|---|
| DS_READ_B64_TR_B16 | 16 b | 2 | K=0..3 and K=8..11; "Each lane (one VGPR) holds 4 consecutive M or N values" |
| DS_READ_B64_TR_B8 | 8 b | 2 | K 0..7, 16..23, 32..39, 48..55 |
| DS_READ_B64_TR_B4 | 4 b | 2 | K 0..15, 32..47 |
| DS_READ_B96_TR_B6 | 6 b | 3 | K 0..15, 32..47 |

### 4.5 Atomics and counters
- "LDS atomics are performed in the LDS hardware" (§11.2), which "contains 32 integer atomic units" (ch02
  §2.2.1). RTN variants "return the value that was in memory before the operation was performed" (ch12_12).
  Float: DS_ADD_F32, DS_PK_ADD_F16/BF16, DS_ADD_F64, MIN/MAX/CMPST F32/F64; "round to nearest even"; denormals per
  MODE except DS_ADD_F64, which does not flush them (ch09 §9.2 Table 53). Data sources "can only be VGPRs or constant values, not SGPRs".
- LDS ops count 1 each in LGKM_CNT and retire in issue order among themselves (§1). For a barrier to protect
  LDS writes, the ISA asks for the matching `s_waitcnt` before the S_BARRIER (ch12_05; §5 below).

## 5. Barriers and scheduling (ch12_05, ch04 §4.1)

| Instruction | Op | ISA |
|---|---|---|
| S_BARRIER | 10 | "If not all waves of the threadgroup have been created yet, waits for entire group before proceeding. If some waves in the threadgroup have already terminated, this waits on only the surviving waves." "Barrier instructions do not wait for any counters to go to zero before issuing. If the barrier is being used to protect an outstanding memory operation use the appropriate S_WAITCNT instruction before the barrier." |
| S_SETPRIO | 15 | "User settable wave priority is set to SIMM16[1:0]. 0 = lowest, 3 = highest. The overall wave priority is {SPIPrio[1:0], UserPrio[1:0], WaveAge[3:0]}." Not in the ISA: which arbitration the priority controls. |
| S_SLEEP | 14 | "(64*(SIMM16[6:0]-1) .. 64*SIMM16[6:0]) clocks"; 0 means no sleep; ch04 Table 9: "64 - 8128 clock cycles" |
| S_WAKEUP | 3 | wakes sleeping waves of the same threadgroup; "The ping is ignored if the waves are not sleeping." |
| S_NOP | 0 | `s_nop N` gives N+1 wait states: the entry's examples are `s_nop 0` "Wait 1 cycle" and `s_nop 0xf` "Wait 16 cycles"; its summary reads "Insert 0..15 wait states based on SIMM16[3:0]" |
| `sched_barrier`, `sched_group_barrier` | — | **Not in the ISA.** Table 74 lists every SOPP opcode and no chapter mentions them; they are compiler directives, and their masks are documented by the compiler. |

## 6. Software-inserted wait states for memory, M0, EXEC and VCC
"The hardware does not check for the following dependencies; they must be resolved by inserting NOPs or
independent instructions." (ch04 §4.5 Table 11, PDF pp.28–29). The rows that touch memory, M0, EXEC or VCC:

| First instruction | Second instruction | Wait | Note |
|---|---|---|---|
| FLAT_STORE_X3/X4, FLAT_ATOMIC_{F}CMPSWAP_X2 "(and global & scratch stores/atomics)", BUFFER_STORE_DWORD_X3/X4, BUFFER_STORE_FORMAT_XYZ/XYZW, BUFFER_ATOMIC_{F}CMPSWAP_X2 | "Write VGPRs holding writedata from those instructions." | 1 | "BUFFER_STORE_* operations that use an SGPR for "offset" do not require any wait states." |
| same list | "VALU writes VGPRs holding writedata from those instructions." | 2 | same note |
| VALU writes SGPR | VMEM reads that SGPR | 5 | "If the VALU writes the SGPR that is used by a VMEM, the user must add five wait states." |
| SALU writes M0 | LDS "add-TID" instruction, buffer_store_LDS_dword, scratch or global with LDS = 1 | 1 | |
| SALU writes M0 | S_SENDMSG / S_MOVEREL | 1 | |
| VALU that sets VCC or EXEC | VALU that uses EXECZ or VCCZ as a data source | 5 | |
| VALU writes EXEC | VALU DPP op | 5 | "ALU does not forward EXEC to DPP." |
| v_cmpx | VALU reads EXEC as constant / V_readlane, v_readfirstlane, v_writelane / other VALU | 2 / 4 / 0 | |
| V_CMPX (writes exec) | V_PERMLANE* | 4 | |
| VALU* writes vdst | V_PERMLANE* reads vdst | 2 | |
| VALU writes SGPR/VCC (v_readlane, v_readfirstlane, v_cmp, v_add/sub, v_div_scale) | VALU reads it as constant / as carry-in / as readlane-writelane lane select | 2 / 0 / 4 | |
| VALU writes VCC (including v_div_scale) | V_DIV_FMAS | 4 | |
| VCC used both by name and as its SGPR number | "VALU which reads VCC as a constant (not as a carry-in which is 0 wait states)." | 1 | |

- `v_readfirstlane` is a VALU write of an SGPR: a descriptor field or `soffset` made that way needs 5 wait
  states before the VMEM that reads it.
- ch03 §3.9 adds that an SALU write to the SGPR holding VCC, followed at once by a branch on VCCZ, must be
  prevented; the next sentence reads self-contradictory ("cannot detect this, and inserts the one required wait
  state"); check p.22.
- **Absent rows.** None for a VALU-written VGPR used as a VMEM or DS address or data, and none naming MUBUF
  loads with `LDS=1` after an M0 write; the M0 row names "buffer_store_LDS_dword", which has no opcode in ch13
  Table 100. A missing row is a gap, not proof that there is no hazard.

## 7. Where the GEMM cards use this

| Card | What the ISA adds |
|---|---|
| `../../gemm/datapath.md` §2–§4 (swizzle, banks, LDS waits) | §4.2: 64 banks × 4 B, serial dword placement; the card's per-width lane phases are not from the ISA. §1: LDS ops in order by type; FLAT needs both counters at zero; SMEM out of order. |
| `../../gemm/datapath.md` §5–§6 (DMA, M0, offsets) | §3: byte/short/dword/x3/x4 only, no x2; the immediate moves both addresses; the M0 width statements disagree; §6 M0 row. §4.3 DS offset windows. Request sizes and per-line cost are not in the ISA. |
| `../../gemm/datapath.md` §8–§9 (descriptors, transpose) | §2.1–§2.3: `soffset` is outside the raw range check; x2/x3/x4 are checked per dword; §6: 5 wait states after `readfirstlane`. §4.4 transpose rules. |
| `../../gemm/schedule_and_locality.md` §2–§4 (XCD order, hints, waits) | §5: `sched_*` are compiler directives. §1: one VM_CNT for loads and stores. XCD count and L2 size are not in the ISA. |
| `../../gemm/epilogue.md` §3–§5 (stores, permlanes, cache policy) | §6: 1–2 wait states before overwriting x3/x4 store data, and the permlane rows. §1: stores share VM_CNT. §2.4: scope tables; `aux` integers are the compiler's encoding, not ISA fields. |

## Related
`lookup.md` (encodings, operands, limits) · `matrix_core.md` (MFMA and its hazards) · `cdna4/ch04_program_flow_control.md` ·
`cdna4/ch09_vector_memory_operations.md` · `cdna4/ch10_flat_memory_instructions.md` · `cdna4/ch11_data_share_operations.md`
