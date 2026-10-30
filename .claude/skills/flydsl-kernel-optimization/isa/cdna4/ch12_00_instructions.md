<!-- page 108 -->

# Chapter 12. Instructions

This chapter lists, and provides descriptions for, all instructions in the CDNA Generation environment. Instructions are grouped according to their format.

**Instruction suffixes have the following definitions:**

- B32 Bitfield (untyped data) 32-bit

- B64 Bitfield (untyped data) 64-bit

- F32 floating-point 32-bit (IEEE 754 single-precision float)

- F64 floating-point 64-bit (IEEE 754 double-precision float)

- BF16 floating-point 16 bit (Bfloat16 format)

- I8 signed 8-bit integer

- I16 signed 16-bit integer

- I32 signed 32-bit integer

- I64 signed 64-bit integer

- U32 unsigned 32-bit integer

- U64 unsigned 64-bit integer

If an instruction has two suffixes (for example, _I32_F32), the first suffix indicates the destination type, the second the source type.

The following abbreviations are used in instruction definitions:

- D = destination

- U = unsigned integer

- S = source

- SCC = scalar condition code

- I = signed integer

- B = bitfield

Note: .u or .i specifies to interpret the argument as an unsigned or signed float.

Note: Rounding and Denormal modes apply to all floating-point operations unless otherwise specified in the instruction description.
