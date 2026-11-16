<!-- page 48 -->

# Chapter 7. Matrix Arithmetic Instructions

Matrix core is an extension to CDNA architecture shader instruction set supporting the Machine Intelligence SIMD. The matrix core has its own VGPR file: the Accumulation ("Acc") GPRs. This is separate from the normal (Architectural, or "Arch") VGPRs in the original SIMD. Shader I/O can only use both types of VGPRs. Instructions have an ACC bit to indicate if data is transferred to/from architectural or accumulation VGPRs. Data can be moved between the ACC and ARCH VGPRs via the V_ACCVGPR_READ and V_ACCVGPR_WRITE instructions.

The core operation implemented inside the matrix core is the 4 × 1 times 1 × 4 _outer_ matrix product, yielding 16 output values. The outer product can be performed both on dense inputs and on 2,4 sparse ones (where two of each set of four values is zero). The matrix core unit uses combinations of these operations, both in parallel and in series, to implement the dense matrix-fused-multiply-add (_MFMA_) instructions described in Subsection Matrix fused-multiply-add (MFMA) and their 2,4-sparse variants described in Subsection Sparse Matrices.

Because these matrix instructions do not produce their output in a single cycle, and since their partially-written results may be observable, a certain amount of independent instructions must sometimes be present between the issuance of a matrix core instruction and accesses to its results or modification of the registers that hold its inputs, as described in Subsection Dependency Resolution: Required Independent Instructions.

Additional information can be found on the GPUOpen blog: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-README/

This blog post relates to CDNA2 but may be helpful with understanding CDNA3.

The AMD Matrix Instruction Calculator (https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator) contains a helper tool that allows developers to view detailed information about the MFMA instructions in the CDNA architecture. It allows users to query instruction-level information such as computational throughput and register usage. It also allows users to generate mappings between matrix element and hardware registers for each MFMA instruction and their modifiers.

## 7.1. Matrix fused-multiply-add (MFMA)

The matrix fused-multiply-add (MFMA) instructions use the matrix core to perform one or more matrix multiplications. Note that the matrix core unit, which executes these instructions, has the 4 × 1 by 1 × 4 outer product as its fundamental computational primitive, and so the MFMA instructions implement outer-product-like operations.

These instructions all have names of the form `V_MFMA_[output type]_[M]X[N]X[K][_[B]B]_[input type]` where _B_ (which is 1 if not specified) is the number of matrices (or _blocks_) that are multiplied, and _M_, _N_, and _K_, are the multiplication dimensions for each block. For example, the instruction V_MFMA_F32_32x32x1_2B_F32 perform the operations

```
D[0,:,:] = C[0,:,:] + A[0,:,:] * B[0,:,:]
D[1,:,:] = C[1,:,:] + A[1,:,:] * B[1,:,:]
```

where the D[_b_,:,:] and C[_b_,:,:] are 32 × 32 matrices of 32-bit floats, the A[_b_,:,:] are 32 × 1 matrices of floats, and the B[_b_,:,:] are 1 × 32 matrices of floats.

<!-- page 49 -->

The input and output values for an MFMA can be stored either in the standard architectural vector registers (VGPRs) or accumulation VGPRs (AccVGPRs), which are additional registers exclusive to the matrix core unit. The register file that the registers holding matrices A and B are controlled by the low and high bits, respectively, of the ACC field of a MFMA instruction (0 for VGPRs, 1 for AccVGPRs), while the ACC_CD bit determines if the C and D matrices are stored in VGPRs (0) or AccVGPRs (1). Data can be moved to and from AGPRs using the V_ACCVGPR_* instructions.

Note that the registers holding input or output data for a MFMA instruction must be contiguous, and that the first register must be aligned to the number of registers required as the input or output. For instance, if an instruction requires four input registers for matrix A, registers 4 through 7 may be used (by setting SRC0 to 4) but not registers 5 through 8.

### 7.1.1. Notation

When indexing values that have multiple blocks, M[b,i,j] is the value in block _b_, row _i_, and column _j_ of the value _M_, where matrices are zero-indexed.

When describing the inputs and outputs of an MFMA operation, they are written as matrices with each column representing a different lane in a wavefront and each row representing a different register (or logical item) that a lane has.

For example, consider the following pair of matrices A

|A[0,0,0]|A[0,0,1]|
|---|---|
|A[0,1,0]|A[0,1,1]|
|…|…|
|A[0,31,0]|A[0,31,1]|
|----|----|
|A[1,0,0]|A[0,0,1]|
|…|…|
|A[1,31,0]|A[1,31,1]|

When the value is written as:

||**Lane 0**|**Lane 1**|**…**|**Lane 31**|**Lane 32**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0**|A[0,0,0]|A[0,1,0]|…|A[0,31,0]|A[1,0,0]|…|A[1,31,0]|
|**Register 1**|A[0,0,1]|A[0,1,1]|…|A[0,31,1]|A[1,0,1]|…|A[1,31,1]|

this means that each lane of a wavefront holds two values across two contiguous registers, which are the two values of a row of one of the blocks of A, with the first 32 lanes holding a different row from block 0 and the second 32 lanes holding successive rows of block 1.

This specification writes matrices in their storage layout.

When showing register layouts, this spec assumes the first register is 0.

Unless otherwise specified, the division operator rounds down (takes the floor).

<!-- page 50 -->

### 7.1.2. List of Dense MFMA instructions

_Table 28. MFMA VALU Opcodes:_

|**Instruction**|**Variants**|**Blocks**|**Cycles**|**Description**|
|---|---|---|---|---|
|V_MFMA_F32_{*}_F32|32x32x1_2B|2|64|Matrix multiply, using FMA with F32 A & B matrices.|
||16x16x1_4B|4|32||
||4x4x1_16B|16|8||
||32x32x2|1|64||
||16x16x4|1|32||
|V_MFMA_F32_{*}_F16|32x32x4_2B|2|64|Matrix multiply, using FMA with F16 A & B matrices.|
||16x16x4_4B|4|32||
||4x4x4_16B|16|8||
||32x32x8|1|32||
||16x16x16|1|16||
|V_MFMA_F32_{*}_BF16|32x32x4_2B|2|64|Matrix multiply, using FMA with BF16 A & B matrices.|
||16x16x4_4B|4|32||
||4x4x4_16B|16|8||
||32x32x8|1|32||
||16x16x16|1|16||
|V_MFMA_I32_{*}_I8|32x32x4_2B|2|64|Matrix multiply, using FMA with I8 A & B matrices|
||16x16x4_4B|4|32||
||4x4x4_16B|16|8||
||32x32x16|1|32||
||16x16x32|1|16||
|V_MFMA_F32_{*}_XF32|16x16x8|1|16|Matrix Multiply on F32 data with reduced multiplication precision.|
||32x32x4|1|32||
|V_MFMA_F64_{*}_F64|16x16x4|1|32|Matrix Multiply on F64 data.|
||4x4x4_4B|4|16||
|V_MFMA_F32\_{\*}\_BF8_BF8<br>V_MFMA_F32\_{\*}\_BF8_FP8<br>V_MFMA_F32\_{\*}\_FP8_BF8<br>V_MFMA_F32\_{\*}\_FP8_FP8|16x16x32|1|16|Matrix Multiply on FP8 or BF8 data.|
||32x32x16|1|32||

|**Control**|**Behavior**|
|---|---|
|Denorm Control|Ignores Denorm Control from MODE and keep Input/Output Denorms.|
|Clamp|Supported. uses the FP16_OVFL bit from MODE<br><br>If set, F32 Result on overflow is clamped to +/- MAX, otherwise the overflow result is normalized to +/-INF.<br><br>If set, I32 Result is clamped to +/-MAX on overflow/underflow, otherwise the carry out bits are dropped.|
|Round Mode|ignores Round Mode from MODE and forces it to RNE.|
|Exceptions|Not Supported|
|Execution Mask|ignores exec mask from MODE and forces it to 1 for all threads|
|Sources|Src0/1/2/VDST if VGPR need to be even aligned.<br><br>Src0/1 can be only VGPR, SRC2 can be inline/constant|

<!-- page 51 -->

**XF32**

The XF32 instructions take 32-bit floats but round the mantissa to 10 bits in order to perform reduced-precision multiplication

### 7.1.3. Usage examples

#### 7.1.3.1. V_MFMA_F32_32X32X1_2B_F32

The first examples show MFMA usage in order to build an intuition for the general semantics of these instructions.

Suppose the user wants do two matrix multiplications of 32 × 1 matrices A[b,:,:] by 1 × 32 matrices B[b,:,:], accumulating the results into 32 × 32 matrices D[b,:,:].

The input register for A stores columns of A across successive lanes (that is, the i coordinate is the fastest-moving) and has the form

||**Lane 0**|**Lane 1**|**…**|**Lane 31**|**Lane 32**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0**|A[0,0,0]|A[0,1,0]|…|A[0,31,0]|A[1,0,0]|…|A[1,31,0]|

that is, lane l holds the value

```
A[l / 32, l % 32, 0]
```

The layout for B holds rows of B in the same way that the A layout stores its columns. That is, B is stored with lane l holding

```
B[l / 32, 0, l % 32]
```

||**Lane 0**|**Lane 1**|**…**|**Lane 31**|**Lane 32**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0**|B[0,0,0]|B[0,0,1]|…|B[0,0,31]|B[1,0,0]|…|B[1,0,31]|

The core component of the output layout is the 4 × _N_ (where _N_ is 32 here) tile of values. (The use of 4 × _N_ tiles, as opposed to a simpler layout, is a consequence of the matrix core’s internal structure). As many of these tiles as possible (here 2 of them) are packed into the lanes of each group of registers, going by row and then by block.

That is, the layout of D (and the corresponding layout of C) is:

||**Lane 0**|**Lane 1**|**…**|**Lane 31**|**Lane 32**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0**|D[0,0,0]|D[0,0,1]|…|D[0,0,31]|D[0,4,0]|…|D[0,4,31]|
|**Register 1**|D[0,1,0]|D[0,1,1]|…|D[0,1,31]|D[0,5,0]|…|D[0,5,31]|
|**…**|…|…|…|…|…|…|…|
|**Register 3**|D[0,3,0]|D[0,3,1]|…|D[0,3,31]|D[0,7,0]|…|D[0,7,31]|
|**Register 4**|D[0,8,0]|D[0,8,1]|…|D[0,8,31]|D[0,12,0]|…|D[0,12,31]|
|**…**|…|…|…|…|…|…|…|
|**Register 15**|D[0,27,0]|D[0,27,1]|…|D[0,27,31]|D[0,31,0]|…|D[0,31,31]|
|**Register 16**|D[1,0,0]|D[1,0,1]|…|D[1,0,31]|D[1,4,0]|…|D[1,4,31]|
|**…**|…|…|…|…|…|…|…|
|**Register 31**|D[1,27,0]|D[1,27,1]|…|D[1,27,31]|D[1,31,0]|…|D[1,31,31]|

<!-- page 52 -->

In other words, the output value D[b, i, j] is located in lane

```
l = j + 32 * ((i/4) % 2)
```

of output register

```
r = 16b + 4(i / 8) + (i % 4)
```

In order to produce these results, the broadcast fields (CBSZ, ABID, and BLGP) must all be set to 0. The usage of these fields is shown in Subsection Broadcasting values.

#### 7.1.3.2. V_MFMA_F32_32X32X2_F32

As another example, consider the instruction V_MFMA_F32_32X32X2_F32. For this instruction, there is only one block being multiplied, and the matrices A[0,:,:] and B[0,:,:] are 32 × 2 and 2 × 32 respectively.

This instruction takes one input register for each of A and B, which the same format as above, except that lanes 32-63 contain the second column of A (row of B) instead of the second block. The output layout is the same as above, except that there is only one block and so there are only 16 output registers. More concretely, the input and output layouts for V_MFMA_F32_32X32X2_F32 are

||**Lane 0**|**Lane 1**|**…**|**Lane 31**|**Lane 32**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0**|A[0,0,0]|A[0,1,0]|…|A[0,31,0]|A[0,0,1]|…|A[0,31,1]|

||**Lane 0**|**Lane 1**|**…**|**Lane 31**|**Lane 32**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0**|B[0,0,0]|B[0,0,1]|…|B[0,0,31]|B[0,1,0]|…|B[0,1,31]|

||**Lane 0**|**Lane 1**|**…**|**Lane 31**|**Lane 32**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0**|D[0,0,0]|D[0,0,1]|…|D[0,0,31]|D[0,4,0]|…|D[0,4,31]|
|**Register 1**|D[0,1,0]|D[0,1,1]|…|D[0,1,31]|D[0,5,0]|…|D[0,5,31]|
|**…**|…|…|…|…|…|…|…|
|**Register 3**|D[0,3,0]|D[0,3,1]|…|D[0,3,31]|D[0,7,0]|…|D[0,7,31]|
|**Register 4**|D[0,8,0]|D[0,8,1]|…|D[0,8,31]|D[0,12,0]|…|D[0,12,31]|
|**…**|…|…|…|…|…|…|…|
|**Register 15**|D[0,27,0]|D[0,27,1]|…|D[0,27,31]|D[0,31,0]|…|D[0,31,31]|

#### 7.1.3.3. V_MFMA_F32_4X4X4_16B_F16

This example demonstrates how values that are not 32 bits long are packed into registers and how the output format changes in the case where an entire matrix cannot fill all lanes in an output register group.

<!-- page 53 -->

The V_MFMA_F32_4X4X4_16B_F16 instruction performs 16 block multiplications of the form

```
D[b,:,:] = C[b,:,:] + A[b,:,:] * B[b,:,:]
```

where each block of A and B is a 4 × 4 block of half-precision floating point values and each block of C and D holds 4 × 4 single-precision floats.

The instruction uses 2 registers to hold each of A and B, even though, following the input format principles from the previous section, each lane needs to hold four values. This is because each input register holds two half-precision values, with the second of those values in the upper bits (16—31) of the register.

That is, the input layout of A is

||**Lane 0**|**Lane 1**|**…**|**Lane 3**|**Lane 4**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0[15:0]**|A[0,0,0]|A[0,1,0]|…|A[0,3,0]|A[1,0,0]|…|A[15,3,0]|
|**Register 0[31:16]**|A[0,0,1]|A[0,1,1]|…|A[0,3,1]|A[1,0,1]|…|A[15,3,1]|
|**Register 1[15:0]**|A[0,0,2]|A[0,1,2]|…|A[0,3,2]|A[1,0,2]|…|A[15,3,2]|
|**Register 1[31:16]**|A[0,0,3]|A[0,1,3]|…|A[0,3,3]|A[1,0,3]|…|A[15,3,3]|

and for B is

||**Lane 0**|**Lane 1**|**…**|**Lane 3**|**Lane 4**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0[15:0]**|B[0,0,0]|B[0,0,1]|…|B[0,0,3]|B[1,0,0]|…|B[15,0,3]|
|**Register 0[31:16]**|B[0,1,0]|B[0,1,1]|…|B[0,1,3]|B[1,1,0]|…|B[15,1,3]|
|**Register 1[15:0]**|B[0,2,0]|B[0,2,1]|…|B[0,2,3]|B[1,2,0]|…|B[15,2,3]|
|**Register 1[31:16]**|B[0,3,0]|B[0,3,1]|…|B[0,3,3]|B[1,3,0]|…|B[15,3,3]|

The 16 4 × 4 output blocks of this instruction are arranged into four output registers as follows.

||**Lane 0**|**Lane 1**|**…**|**Lane 3**|**Lane 4**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Register 0**|D[0,0,0]|D[0,0,1]|…|D[0,0,3]|D[1,0,0]|…|D[15,0,3]|
|**…**|…|…|…|…|…|…|…|
|**Register 3**|D[0,3,0]|D[0,3,1]|…|D[0,3,3]|D[1,3,0]|…|D[15,3,3]|

That is, because there are not enough groups of 4 rows available in a block to fill 64 lanes of output in each register, successive blocks are used instead. Note that these outputs are 32-bit floats and so are not packed into registers.

#### 7.1.3.4. V_MFMA_F64_16X16X4_F64

This demonstrates how double-precision values are handled using the example of V_MFMA_F64_16X16X4_F64. This instruction follows the same input layout patterns as the previous examples and operates most similarly to V_MFMA_F32_32X32X2_F32. However, each input is spread across multiple registers in order to accommodate the full 64-bit value.

The output of this instruction, and the other double-precision MFMA instructions, does **not** follow the 4 × _N_ block layout of other MFMA instructions. Instead, the output rows are packed contiguously across the lanes of each wavefront, and then packed into pairs (to account for the 64 bits needed to store the output) of registers, as shown below.

<!-- page 54 -->

The input and output formats for V_MFMA_F64_16X16X4_F64 are

||**Lane 0**|**Lane 1**|**…**|**Lane 15**|**Lane 16**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Reg. 0**|A[0,0,0][31:0]|A[0,1,0][31:0]|…|A[0,15,0][31:0]|A[0,0,1][31:0]|…|A[0,15,3][31:0]|
|**Reg. 1**|A[0,0,0][63:32]|A[0,1,0][63:32]|…|A[0,15,0][63:0]|A[0,0,1][63:32]|…|A[0,15,3][63:32]|

||**Lane 0**|**Lane 1**|**…**|**Lane 15**|**Lane 16**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Reg. 0**|B[0,0,0][31:0]|B[0,0,1][31:0]|…|B[0,0,15][31:0]|B[0,1,0][31:0]|…|B[0,3,15][31:0]|
|**Reg. 1**|B[0,0,0][63:32]|B[0,0,1][63:32]|…|B[0,0,15][63:32]|B[0,1,0][63:32]|…|B[0,3,15][63:32]|

||**Lane 0**|**Lane 1**|**…**|**Lane 15**|**Lane 16**|**…**|**Lane 63**|
|---|---|---|---|---|---|---|---|
|**Reg. 0**|D[0,0,0][31:0]|D[0,0,1][31:0]|…|D[0,0,15][31:0]|D[0,1,0][31:0]|…|D[0,3,15][31:0]|
|**Reg. 1**|D[0,0,0][63:32]|D[0,0,1][63:32]|…|D[0,0,15][63:32]|D[0,1,0][63:32]|…|D[0,3,15][63:32]|
|**Reg. 2**|D[0,4,0][31:0]|D[0,4,1][31:0]|…|D[0,4,15][31:0]|D[0,5,0][31:0]|…|D[0,7,15][31:0]|
|**…**|…|…|…|…|…|…|…|
|**Reg. 7**|D[0,12,0][63:32]|D[0,12,1][63:32]|…|D[0,12,15][63:32]|D[0,13,0][63:32]|…|D[0,15,15][63:32]|

### 7.1.4. General input and output layout

In general, an MFMA instruction is parameterized by its input and output datatypes, the sizes _M_, _N_, and _K_ of each matrix block and the number of blocks it operates on _B_.

Semantically, for each 0 <= b < B, 0 <= i < M, and 0 <= j < N, it computes

```
D[b,i,j] = C[b,i,j] + sum_{0 <= k < K} A[b,i,k] * B[b,k,j]
```

where each A[b,:,:] is _M_ × _K_, each B[b,:,:] is _K_ × _N_, and each D[b,:,:] and corresponding C[b,:,:] is _M_ × _N_.

The values of the inputs and outputs are placed into the arguments to the instruction according to a fixed layout. For simplicity, this layout is defined in terms of the lanes of a wavefront and of the sequence of _items_ for each lane: these items are arranged into the 32-bit registers that are the true arguments to an MFMA instruction in little-endian form.

More specifically,

- For 64-bit quantities, each item corresponds to a _pair_ of registers, with the low bits of the quantity in the first of those registers and high bits in the second one

- For 32-bit quantities, each item corresponds to a distinct register

- For 16-bit quantities, an item is half of a register, with the odd-numbered items taking up bits 31-16 and the even ones in bits 0-15

- For 8-bit quantities, four items are packed into a register, analogously to the 16-bit case

#### 7.1.4.1. Input layout

To define the input layout for the matrix A, first define the auxiliary constant

<!-- page 55 -->

```
K_L = K / (64 / (M * B))
```

which is the number of consecutive values of _K_ that each lane holds in its registers.

For example, for the instruction V_MFMA_F32_32X32X1_2B_F32:

```
K_L = 1 / (64 / (32 * 2)) = 1 / 1 = 1
```

and for V_MFMA_F32_32X32X2_F32:

```
K_L = 2 / (64 / (32 * 1)) = 2 / 2 = 1
```

These both show that the one input register holds one value in the K dimension, but for V_MFMA_F32_4X4X4_16B_F16:

```
K_L = 4 / (64 / (4 * 16)) = 4 / 1 = 4
```

representing the fact that, for each lane, the two input registers hold four values in the _K_ dimension. These input values are packed little-endian. For example, the third value in each row (which has k = 2 zero-indexed), is in bits 15:0 of the second input register for both _A_ and _B_ across all lanes. That 16-bit region is "item 2" in the layout computed below for V_MFMA_F32_4X4X4_16B_F16.

Note that, in all MFMA instructions, the products _M_ * _B_ and _N_ * _B_ are less than 64, that is, the values of a single column of A or row of B, considered over all blocks, fit within a single input item.

With this layout defined, a given input value A[b,i,k] is placed in the item

```
k % K_L
```

of lane

```
i + M * (b + B * (k / K_L))
```

The layout for B is the analogous function that places B[b,k,j] in item

```
k % K_L
```

of lane

```
j + N * (b + B * (k / K_L))
```

<!-- page 56 -->

#### 7.1.4.2. Output layout

The output values D[b,i,j] of an MFMA instruction, as well as the corresponding values C[b,i,j] of the matrix to add to the result, are stored in a fixed layout that is a function of the MFMA instruction being used.

To define this layout, first define the following constants:

- H, the _group height_, which indicates how many consecutive rows of output are placed in each _row group_ and which are therefore stored in consecutive items on a single lane. For **f64** instructions, H = 1, but for **all other MFMA instructions**, H = 4.

- B_I = ceil(64 / (_N_ * _M_ / H)), the number of blocks stored in each _output item_ (an item within the storage for _D_ or _C_) across all lanes

- M_I = (64 / B_I) / _N_, the number of rows of D stored in each output item across all lanes

- G = _M_ / (H * M_I), the number of row groups needed to store B_I blocks of output.

For example, using the instruction V_MFMA_F32_32X32X1_2B_F32 gives

```
H = 4
B_I = ceil(64 / (32 * 32 / H)) = ceil(64 / 256) = 1
M_I = (64 / 1) / 32 = 2
G = 32 / (4 * 2) = 4
```

while the instruction V_MFMA_F32_4X4X4_16B_F16 yields the values

```
H = 4
B_I = ceil(64 / (H * 4 / 4)) = ceil(64 / 4) = 16
M_I = (64 / 16) / 4 = 4 / 4 = 1
G = 4 / (H * 1) = 4 / 4 = 1
```

With these constants defined, the value D[b,i,j] of matrix D is located in item

```
(i % H) + H * (i/(H * M_I) + G * (b / B_I))
```

on lane

```
j + N * ((i / H) % M_I + M_I * (b % B_I))
```

### 7.1.5. Broadcasting values

While the operation of multiplying a 32 × 1 matrix of floats A by a 1 × 64 matrix B is not available natively, one can emulate this multiplication using the broadcast controls _Control Broadcast SiZe_ (CBSZ), _A Block ID_ (ABID), and _B Lane-Group Permutation_ (BLGP).

These controls impact the retrieval of values from lanes: after the lane l_a in which a particular element of A would reside is computed, that value is permuted as defined by the CBSZ and ABID fields in order to determine the lane that is accessed during the computation. Similarly, l_b, the lane to be used when retrieving any particular value of B, is permuted in the manner specified by the BLGP field.

<!-- page 57 -->

#### 7.1.5.1. CBSZ and ABID

Together, the CBSZ and ABID fields control the broadcasting of the blocks of matrix A.

When the 3-bit CBSZ field is non-zero, one block of lanes broadcasts the values it holds for matrix A to the other blocks of lanes, superseding the values those other lanes hold for the A matrix. Setting CBSZ such that (1 << CBSZ) exceeds the number of blocks the MFMA instruction processes is **undefined**.

The broadcast block size is

```
S = 64 / (1 << CBSZ)
```

For example, if CBSZ is 1, then one block of 32 lanes provides the inputs to both groups of 32 lanes in the wavefront, while CBSZ being 3 means that a the values from a block of 8 lanes are replicated.

The largest legal value of CBSZ is 4.

The 4-bit ABID field controls which block of S lanes is used as the broadcast source. The possible blocks are numbered in order, with lanes S-1:0 being selected by ABID=0, 2S-1:S corresponding to ABID=1, and so on. For example, if CBSZ=2, then ABID=1 means the values from lanes 16 to 31 are broadcast, to the three other blocks of lanes, while ABID=3 means that lanes 48 to 63 serve as the source of their inputs.

It is not legal to set ABID such that ABID >= (1 << CBSZ), as such values do not refer to a potential source block.

Put differently, the CBSZ and ABID bits cause lane l_a to read thir inputs from the lane given by the permutation

```
p_a(l_a) = (l_a % S) + (S * ABID)
```

As a full example, if CBSZ=1 and ABID=1 when using the instruction V_MFMA_F32_32X32X1_2B_F32, both 1 × 32 blocks of B are multiplied by the values in the second 32 × 1 block of A, which is stored by the first 32 lanes. That is, the operation becomes:

```
D[b,i,j] = C[b,i,j] + A[1,i,0] * B[b,0,j]
```

which is a 32 × 1 by 1 × 64 matrix multiplication if the two blocks of B are treated as one matrix with 64 rows.

#### 7.1.5.2. BLGP

The 3-bit BLGP field selects how the lane from which values in matrix B are read is permuted. Once it is determined that some value B[b,k,j] is in item r on lane l_b, using the defined input layout, the lane to be accessed l_b is permuted depending on the BLGP field as shown in Table Permutations corresponding to BLGP values.

<!-- page 58 -->

_Table 29. Permutations corresponding to BLGP values_

|**Value**|**Description**|**Expression**|
|---|---|---|
|0|No broadcast|l_b|
|1|Broadcast first 32 lanes|l_b % 32|
|2|Broadcast second 32 lanes|l_b % 32 + 32|
|3|Rotate 16 lanes left|(l_b + 16) % 64|
|4|Broadcast first 16 lanes|l_b % 16|
|5|Broadcast second 16 lanes|l_b %16 + 16|
|6|Broadcast third 16 lanes|l_b % 16 + 32|
|7|Broadcast fourth 16 lanes|l_b % 16 + 48|

#### 7.1.5.3. Alternate meaning of broadcast fields for F64 instructions

The MFMA instructions that operate on double-precision floats (f64) do not support the broadcasting methods described above.

These instructions ignore CBSZ and ABID.

The BLGP field is repurposed for signaling the negation of the matrices A, B, and C.

- BLGP[0] causes values from matrix A to be implicitly negated if set

- BLGP[1] causes values from matrix B to be implicitly negated if set

- BLGP[2] causes values from matrix C to be implicitly negated if set

## 7.2. BF8 and FP8 Formats and Conversions

_Table 30. Small Float Data Formats_

|**Fmt**|**Sign-Exp-Mant**|**Bias**|**+0**<br>**-0**|**INF,**<br>**-INF**|**NaN,**<br>**-NaN**|**Max**|**Min**<br>**(norm)**|**Min (denorm)**|
|---|---|---|---|---|---|---|---|---|
|FP16|E5M10|15|0x0000<br>0x8000|0x7C00<br>0xFC00|(normal)|65504|6.10352E-05|5.96046E-08|
|FP8|E4M3|8|+/-: 0x00|N/A|0x80|240|+/-2.0^(-7)|+/-2.0^(-10)|
|BF8|E5M2|16|+/-: 0x00|0x80|0x80|57344|+/-2^(-15)|2.0^(-17)|

_Table 31. Small Float Data Format Conversion ops_

|**Instruction**|**Dst**|**Src0**|**Src1**|**Encoding**|**Control**|**Notes**|
|---|---|---|---|---|---|---|
|CVT_PK_FP8_F32|FP8|FP32|FP32|VOP3|Op_Sel[3]<br>ignores: clamp, omod<br>supports: neg, abs|RNE|
|CVT_PK_BF8_F32|BF8|FP32|FP32|VOP3|Op_Sel[3]<br>ignores: clamp, omod<br>supports: neg, abs|RNE|
|CVT_SR_FP8_F32|FP8|FP32|U32|VOP3|Op_Sel[3:2]<br>ignores: clamp, omod<br>supports: neg, abs|Stochastic Rounding|
|CVT_SR_BF8_F32|BF8|FP32|U32|VOP3|Op_Sel[3:2]<br>ignores: clamp, omod<br>supports: neg, abs|Stochastic Rounding|
|CVT_PK_F32_FP8|F32|FP8|-|VOP1|SDWA, Op_Sel[0], dst,dst+1<br>ignores: abs, neg, sext|dst must be even|
|CVT_PK_F32_BF8|F32|BF8|-|VOP1|SDWA, Op_Sel[0], dst,dst+1<br>ignores: abs, neg, sext|dst must be even|
|CVT_F32_FP8|F32|FP8|-|VOP1|SDWA, Op_Sel<br>ignores: abs, neg, sext|-|
|CVT_F32_BF8|F32|BF8|-|VOP1|SDWA, Op_Sel<br>ignores: abs, neg, sext|-|

<!-- page 59 -->

Note: VOP3 instructions may not use SDWA.

In the above table, the CVT_*_F32 instructions do not support 4-cycle forwarding on these operations. The user must insert a NOP or instruction writing some other destination VREG between the conversions writing the low/high half or bytes of the same destination register.

Convert instructions come in two types:

- Packed – convert two 8-bit values into 32-bit values per instruction

- Stochastic Round – one source has the number to convert and the other has a random number used in rounding

   - These ops add a random value from the specified VGPR and then truncate to the smaller result data type

Converts from 8-bit formats and SDWA in VOP1:

- SRC0 must be set to “SDWA”, and the SRC0 VGPR is specified in the SDWA word as is the SRC0_SELECT which specifies which bytes to be converted. The other SDWA fields are ignored.

**CVT_SR_FP8_F32 and CVT_SR_BF8_F32 OP_SEL usage:**

Op_sel[3:2] == 00: dest_vgpr[31:0] = {prev_dst_vgpr[31:8], result[7:0]}<br>
Op_sel[3:2] == 01: dest_vgpr[31:0] = {prev_dst_vgpr[31:16], result[7:0], prev_dst_vgpr[7:0]}<br>
Op_sel[3:2] == 10: dest_vgpr[31:0] = {prev_dst_vgpr[31:24], result[7:0], prev_dst_vgpr[15:0]}<br>
Op_sel[3:2] == 11: dest_vgpr[31:0] = {result[7:0], prev_dst_vgpr[23:0]}

**CVT_SR_FP8_F32 OP_SEL usage**:

Uses Src0 and Src1 as inputs supplied by the VOP3 encoding, it adds the two operands with attention to not use msbs of src1 mantissa based on opcode and dependent on the F8 data type for the stochastic round before converting to F8 type.  Then OP_SEL bits 3 and 2 are repurposed for this 8b write op and used to steer the resulting 8 bits into the correct byte lane of the 32b output preserving the remaining 24b of the destination.

**CVT_\*FP8_F32 and CVT\*_BF8_F32 FP16_OVFL rule**

The FP16_OVFL flag is applied to data conversions from F32 to FP8/BF8 formats. The overflow behaviour is specified in the table below:

CVT_SR_* and CVT_PK_* support only VGPRs as inputs, not SGPRs, literal or inline constants.

<!-- page 60 -->

<table>
<thead>
<tr><th rowspan="3">Source Value</th><th colspan="4">Destination Value</th></tr>
<tr><td colspan="2">FP8</td><td colspan="2">BF8</td></tr>
<tr><td>FP16_OVFL=1</td><td>FP16_OVFL=0</td><td>FP16_OVFL=1</td><td>FP16_OVFL=0</td></tr>
</thead>
<tbody>
<tr><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td></tr>
<tr><td>±Inf</td><td>±max_E4M3</td><td>NaN</td><td>±max_E5M2</td><td>±Inf</td></tr>
<tr><td>Greater than max FP8 magnitude</td><td>±max_E4M3</td><td>NaN</td><td>±max_E5M2</td><td>±Inf</td></tr>
</tbody>
</table>

The register SH_MEM_CONFIG, bit[8] must be set to 1 to produce the correct results for BF8 and FP8 operations.

## 7.3. Floating-point handling details and formats

The handling of denormal numbers varies depending on the datatypes the instruction takes and, in some cases, the MODE flags.

- V_MFMA_F32_*_F32 instructions, which take 32-bit inputs, honor the denormal-handling flags in MODE

- However, the V_MFMA_F32_*_XF32 instructions, which also take 32-bit inputs but only use 10 bits of mantissa, ignore MODE.denorm and do not flush denormals

- Matrix-C input and result-matrix output ignore MODE.denorm and do not flush denormals

- All instructions that take floats whose size is less than 32-bits (F16, BF16, BF8, FP8) ignore MODE.denorm and do not flush denormals

- The V_MFMA_F64_*_F64 instructions, which take 64-bit inputs and outputs ignores MODE and rounds to nearest even and allows denorms in the input and output

- The V_MFMA_I32_*_I8 perform integer multiply-add and thus do not respect the MODE bits. The 16-bit results of multiplying the I8 input values are sign-extended to 32 bits before multiplication, and the 16-bit result of the multiplication is sign extended to 32 bits prior to being added to the 32-bit result

The matrix core unit does not support arithmetic exceptions, except for DGEMM matrix operation which does support exceptions.

## 7.4. Sparse Matrices

The V_SMFMAC family of instructions perform matrix multiply-accumulate operations on a 4:2 structurally-sparse matrix A and dense matrices B, C, and D: D = C + A × B.

The A matrix is represented using 4:2 structured sparsity which means that two out of every 4 elements along the matrix K-dimension are zero. These zeros are not stored directly but instead are described in a separate VGPR which holds pairs of 2-bit index values. The index values indicate which two values out of each group of 4 are non-zero and are used to reconstruct full A-matrix. Non-zero samples are tightly packed resulting in 2:1 compression. Only the A-matrix may be sparse.

These SMFMAC instructions are all "accumulate" ops, where the C and D matrices are identical, referenced by the instruction’s VDST field (D-matrix). The C operand input is repurposed to hold the index data offset.

_Table 32. SMFMA VALU Opcodes:_

<!-- page 61 -->

|**Instruction**|**Variants**|**Blocks**|**Cycles**|**Description**|
|---|---|---|---|---|
|V_SMFMAC_F32_{*}_F16|16x16x32|1|16|Sparse Matrix multiply of F16 data|
||32x32x16|1|32||
|V_SMFMAC_F32_{*}_BF16|16x16x32|1|16|Sparse Matrix multiply of BF16 data|
||32x32x16|1|32||
|V_SMFMAC_I32_{*}_I8|16x16x64|1|16|Sparse Matrix multiply of I8 data|
||32x32x32|1|32||
|V_SMFMAC_F32\_{\*}\_BF8_BF8<br>V_SMFMAC_F32\_{\*}\_BF8_FP8<br>V_SMFMAC_F32\_{\*}\_FP8_BF8<br>V_SMFMAC_F32\_{\*}\_FP8_FP8|16x16x64|1|16|Sparse Matrix multiply of BF8 or FP8 data|
||32x32x32|1|32||

Matrix A is structurally sparse and occupies two VGPRs per lane at srcA offset. Matrix B is dense and occupies four VGPRs per lane at srcB offset. Matrix C shares VGPR offset with destination argument and occupies 16 VGPRs.

**16-bit source data**

If CBSZ[1:0] =0, ABID[1:0] selects one of four 8-bit sets of sparse-indices within a VGPR starting at srcC containing 8-bits of index information for a lane. If CBSZ[1:0] !=0; the very first is selected (VGPR[srcC][7..0]).

**8-bit source data**

If CBSZ[1:0] =0, ABID[0] selects one of two 16-bit sets of sparse-indices within a VGPR starting at srcC containing 16-bits of index information for a lane. If CBSZ[1:0] !=0; the very first is selected (VGPR[srcC][15..0]).

All SMFMAC instructions must follow these restrictions:

1. The Matrix A is sparse matrix and matrix B is the dense matrix. The ALU loads twice more data from VGPR for matrix B comparing with matrix A.

2. Matrix C is the same as the result Matrix. The ALU uses the VDST VGPR to load matrix C. All instructions are encoded as accumulation opcodes.

3. Src2 has the index encoded (all of the indexes are in one VGPR) and it can only be VGPR. Index Data provides information about which 2 out 4 SRCA are non-zero. For this index pair, index 0 < index 1 & index0 != index1.

4. The VGPR address of Src0, src1 and VDST must be even aligned.

5. CBSZ and ABID controls are ONLY used to pick the index from the VGPR read and don’t affect SRCA matrix broadcast etc. as defined for other MFMA opcodes that use CBSZ and ABID controls.

SMFMAC instructions interpret the ACC_CD differently from other instructions: For SMFMAC the ACC_CD bit control only the DEST vgpr (arch vs accum), not the SRC2 location. The SRC2 argument provides the index data for sparse data supplied by the SRC0 argument which must reside in the ARCH-vgprs along with the A and B matrix data. In other words SRC2 acts as if ACC_CD==0.

### 7.4.1. Details of Sparsity Structure

Every index for the matrix B selection is a 2-bit number which identifies one of K=4 is selected. Depending on the matrix B layout, SRC2 may hold multiple sets of indices.

<!-- page 62 -->

#### 7.4.1.1. 16-bit A/B Matrix

When the A and B matrices consist of 16-bit data (FP16, BF16), the rules below apply.

_Table 33. Matrix B Layout_

|**16x16x32**|**Row0**|**Row1**|**Row2**|**Row3**|
|---|---|---|---|---|
|v0|k=0,1|k=8,9|k=16,17|k=24,25|
|v1|k=2,3|k=10,11|k=18,19|k=26,27|
|v2|k=4,5|k=12,13|k=20,21|k=28,29|
|v3|k=6,7|k=14,15|k=22,23|k=30,31|
|32x32x16|Row0|Row1|Row2|Row3|
|v0|k=0,1|k=0,1|k=8,9|k=8,9|
|v1|k=2,3|k=2,3|k=10,11|k=10,11|
|v2|k=4,5|k=4,5|k=12,13|k=12,13|
|v3|k=6,7|k=6,7|k=14,15|k=14,15|

Each lane has K=8 values which requires 4 indices per lane (8 bits), so each SRC2 VGPR holds 4 sets of indices.

_Table 34. Index Layout_

<table>
<thead>
<tr><th>Lane ID</th><th>0</th><th>1</th><th>…</th><th>3</th><th>4</th><th>…</th><th>31</th><th>32</th><th>…</th><th>63</th></tr>
</thead>
<tbody>
<tr><td>Vn[31:30]</td><td colspan="10">set3, V1[31:16]</td></tr>
<tr><td>Vn[29:28]</td><td colspan="10">set3, V1[15:0]</td></tr>
<tr><td>Vn[27:26]</td><td colspan="10">set3, V0[31:16]</td></tr>
<tr><td>Vn[25:24]</td><td colspan="10">set3, V0[15:0]</td></tr>
<tr><td>…</td><td colspan="10">…</td></tr>
<tr><td>Vn[9:8]</td><td colspan="10">set1, V0[15:0]</td></tr>
<tr><td>Vn[7:6]</td><td colspan="10">set0, V1[31:16]</td></tr>
<tr><td>Vn[5:4]</td><td colspan="10">set0, V1[15:0]</td></tr>
<tr><td>Vn[3:2]</td><td colspan="10">set0, V0[31:16]</td></tr>
<tr><td>Vn[1:0]</td><td colspan="10">set0, V0[15:0]</td></tr>
</tbody>
</table>

"Vn" is the SRC-C VGPR holding the index values.

#### 7.4.1.2. 8-bit A/B Matrix

When the A and B matrices consist of 8-bit data (I8, FP8, BF8), the rules below apply.

_Table 35. Matrix B Layout_

|**16x16x64**|**Row0**|**Row1**|**Row2**|**Row3**|
|---|---|---|---|---|
|v0|k=0,1,2,3|k=16,17,18,19|k=32,33,34,35|k=48,49,50,51|
|v1|k=4,5,6,7|k=20,21,22,23|k=36,37,38,39|k=52,53,54,55|
|v2|k=8,9.10,11|k=24,25,26,27|k=40,41,42,43|k=56,57,58,59|
|v3|k=12,13,14,15|k=28,29,30,31|k=44,45,46,47|k=60,61,62,63|
|32x32x32|Row0|Row1|Row2|Row3|
|v0|k=0,1,2,3|k=0,1,2,3|k=16,17,18,19|k=16,17,18,19|
|v1|k=4,5,6,7|k=4,5,6,7|k=20,21,23,23|k=20,21,23,23|
|v2|k=8,9,10,11|k=8,9,10,11|k=24,25,26,27|k=24,25,26,27|
|v3|k=12,13,14,15|k=12,13,14,15|k=28,29,30,31|k=28,29,30,31|

<!-- page 63 -->

Each lane has K=16 values which requires 8 indices per lane (16 bits), so each SRC2 VGPR holds 2 sets of indices.

_Table 36. Index Layout_

<table>
<thead>
<tr><th>Lane ID</th><th>0</th><th>1</th><th>…</th><th>3</th><th>4</th><th>…</th><th>31</th><th>32</th><th>…</th><th>63</th></tr>
</thead>
<tbody>
<tr><td>Vn[31:30]</td><td colspan="10">set1, V1[31:24]</td></tr>
<tr><td>Vn[29:28]</td><td colspan="10">set1, V1[23:16]</td></tr>
<tr><td>Vn[27:26]</td><td colspan="10">set1, V1[15:8]</td></tr>
<tr><td>Vn[25:24]</td><td colspan="10">set1, V1[7:0]</td></tr>
<tr><td>Vn[23:22]</td><td colspan="10">set1, V0[31:24]</td></tr>
<tr><td>Vn[21:20]</td><td colspan="10">set1, V0[23:16]</td></tr>
<tr><td>Vn[19:19]</td><td colspan="10">set1, V0[15:8]</td></tr>
<tr><td>Vn[17:16]</td><td colspan="10">set1, V0[7:0]</td></tr>
<tr><td>Vn[15:14]</td><td colspan="10">set0, V1[31:24]</td></tr>
<tr><td>…</td><td colspan="10">…</td></tr>
<tr><td>Vn[3:2]</td><td colspan="10">set0, V0[15:8]</td></tr>
<tr><td>Vn[1:0]</td><td colspan="10">set0, V0[7:0]</td></tr>
</tbody>
</table>

## 7.5. Dependency Resolution: Required Independent Instructions

The table below indicates timing conditions which require the user to insert NOPs (or independent VALU instructions).

|||
|---|---|
|**DLop**|Dot products|
|**XDLOP**|Matrix math on {I8, F16, BF16}|
|**DGEMM**|V_MFMA…F64|
|**PASS**|4 clock cycles|

_Table 37. VOP3P-Matrix Opcodes Required NOPs_

<table>
<thead>
<tr><th>First Instruction</th><th>Second Instruction</th><th>Required Waits</th><th>Comments</th></tr>
</thead>
<tbody>
<tr><td>Non-DLops VALU Write VGPR</td><td>V_MFMA* read VGPR OR V_SMFMA* read VGPR</td><td>2</td><td>No internal 4 & 8 cycle forwarding path.</td></tr>
<tr><td rowspan="3">DL ops Write VGPR</td><td>DLops read VGPR as SrcC, and the opcode is exactly the same as 1st DLops</td><td>0</td><td>supports same opcode of DLops back-to-back SrcC forwarding which is used for accumulation.</td></tr>
<tr><td>DLops read VGPR as SrcA/B, and the opcode is exactly the same as 1st DLops</td><td>3</td><td>does not support SrcA/B forwarding in DLops</td></tr>
<tr><td>Any opcode read/write VGPR that is not the same as 1st DLops opcode (RAW + WAW)</td><td>3</td><td>Disable all of the forwarding path from DL ops to normal VALU/VM/LDS/FLAT ops</td></tr>
<tr><td rowspan="4">XDL Write VGPR or V_SMFMA* Write VGPR</td><td rowspan="4">XDL read VGPR as Source C exactly same with 1st vDst<br><br>OR<br><br>V_SMFMA* read VGPR for Matrix C exactly same with 1st vDst</td><td>2 if 1st V_MFMA is 2 passes</td><td rowspan="4">the two V_MFMA must be the same number passes and vDst and vSrc start from the same offset and same VGPR size.<br><br>V_MFMA & V_SMFMA must be the same number passes and both vDst start from the same offset and same VGPR size. Note: V_SMFMA reads vdst for Matrix C.</td></tr>
<tr><td>0 if 1st V_MFMA is 4 passes</td></tr>
<tr><td>0 if 1st V_MFMA is 8 passes</td></tr>
<tr><td>0 if 1st V_MFMA is 16 passes</td></tr>
<tr><td rowspan="4">XDL Write VGPR or V_SMFMA* Write VGPR</td><td rowspan="4">XDL read VGPR as Source C overlapped with 1st vDst<br><br>OR<br><br>V_SMFMA* read VGPR for Matrix C overlapped with 1st vDst</td><td>3 if 1st V_MFMA is 2 passes</td><td rowspan="4">overlapped with XDL<br><br>Note: V_SMFMA reads vdst for Matrix C.</td></tr>
<tr><td>5 if 1st V_MFMA is 4 passes</td></tr>
<tr><td>9 if 1st V_MFMA is 8 passes</td></tr>
<tr><td>17 if 1st V_MFMA is 16 passes</td></tr>
<tr><td rowspan="4">XDL Write VGPR or V_SMFMA* Write VGPR</td><td rowspan="4">S/DGEMM read VGPR as Source C</td><td>3 if 1st V_MFMA is 2 passes</td><td rowspan="4">Overlapped with S/DGEMM</td></tr>
<tr><td>5 if 1st V_MFMA is 4 passes</td></tr>
<tr><td>9 if 1st V_MFMA is 8 passes</td></tr>
<tr><td>17 if 1st V_MFMA is 16 passes</td></tr>
<tr><td rowspan="4">XDL Write VGPR or V_SMFMA* Write VGPR</td><td rowspan="4">V_MFMA read VGPR as SrcA or SrcB<br><br>OR<br><br>V_SMFMA* read VGPR as SrcA or SrcB or Index SrcC</td><td>5 if 1st V_MFMA is 2 passes</td><td rowspan="4">No internal forwarding path waits for previous V_MFMA/V_SMFMA* commit result to VGPR<br><br>V_SMFMA uses srcC address for extra Index C Reads</td></tr>
<tr><td>7 if 1st V_MFMA is 4 passes</td></tr>
<tr><td>11 if 1st V_MFMA is 8 passes</td></tr>
<tr><td>19 if 1st V_MFMA is 16 passes</td></tr>
<tr><td rowspan="4">XDL Write VGPR or V_SMFMA* Write VGPR</td><td rowspan="4">1) VM, L/GDS, FLAT, Export Read VGPR overlapped with 1st vDst<br><br>2) VALU read/write VGPR (RAW + WAW)</td><td>5 if 1st V_MFMA is 2 passes</td><td>V_MFMA_F32_4X4X4F16</td></tr>
<tr><td>7 if 1st V_MFMA is 4 passes</td><td>V_MFMA_F32_16X16X16F16</td></tr>
<tr><td>11 if 1st V_MFMA is 8 passes</td><td>V_MFMA_F32_32X32X8F16</td></tr>
<tr><td>19 if 1st V_MFMA is 16 passes</td><td>V_MFMA_F32_32X32X4F16</td></tr>
<tr><td>SGEMM Write VGPR</td><td>XDL read VGPR as Source C exactly same with 1st vDst<br><br>OR<br><br>V_SMFMA* read VGPR for Matrix C exactly same with 1st vDst</td><td>0</td><td>the two V_MFMA must be the same number passes and vDst and vSrc start from the same offset and same VGPR size.<br><br>V_MFMA & V_SMFMA must be the same number passes and both vDst start from the same offset and same VGPR size. Note: V_SMFMA reads vdst for Matrix C.</td></tr>
</tbody>
</table>

<!-- page 64 -->

<!-- page 65 -->

<table>
<thead>
<tr><th>First Instruction</th><th>Second Instruction</th><th>Required Waits</th><th>Comments</th></tr>
</thead>
<tbody>
<tr><td rowspan="4">SGEMM Write VGPR</td><td rowspan="4">XDL read VGPR as Source C overlapped with 1st vDst<br><br>OR<br><br>V_SMFMA* read VGPR for Matrix C overlapped with 1st vDst</td><td>2 if 1st V_MFMA is 2 passes</td><td rowspan="4">overlapped with XDL<br><br>Note: V_SMFMA reads vdst for Matrix C.</td></tr>
<tr><td>4 if 1st V_MFMA is 4 passes</td></tr>
<tr><td>8 if 1st V_MFMA is 8 passes</td></tr>
<tr><td>16 if 1st V_MFMA is 16 passes</td></tr>
<tr><td rowspan="4">SGEMM Write VGPR</td><td rowspan="4">S/DGEMM read VGPR as Source C</td><td>2 if 1st V_MFMA is 2 passes</td><td rowspan="4">Overlapped with S/DGEMM</td></tr>
<tr><td>4 if 1st V_MFMA is 4 passes</td></tr>
<tr><td>8 if 1st V_MFMA is 8 passes</td></tr>
<tr><td>16 if 1st V_MFMA is 16 passes</td></tr>
<tr><td rowspan="4">SGEMM Write VGPR</td><td rowspan="4">V_MFMA read VGPR as SrcA or SrcB<br><br>OR<br><br>V_SMFMA* read VGPR as SrcA or SrcB or Index SrcC</td><td>4 if 1st V_MFMA is 2 passes</td><td rowspan="4">No internal forwarding path, need to wait previous V_MFMA commit result to VGPR.<br><br>Note: V_SMFMA used SrcC for Index reads.</td></tr>
<tr><td>6 if 1st V_MFMA is 4 passes</td></tr>
<tr><td>10 if 1st V_MFMA is 8 passes</td></tr>
<tr><td>18 if 1st V_MFMA is 16 passes</td></tr>
<tr><td rowspan="4">SGEMM Write VGPR</td><td rowspan="4">1) VM, L/GDS, FLAT, Export Read VGPR overlapped with 1st vDst<br><br>2) VALU read/write VGPR (RAW + WAW)</td><td>4 if 1st V_MFMA is 2 passes</td><td>V_MFMA_F32_4X4X4F16</td></tr>
<tr><td>6 if 1st V_MFMA is 4 passes</td><td>V_MFMA_F32_16X16X16F16</td></tr>
<tr><td>10 if 1st V_MFMA is 8 passes</td><td>V_MFMA_F32_32X32X8F16</td></tr>
<tr><td>18 if 1st V_MFMA is 16 passes</td><td>V_MFMA_F32_32X32X4F16</td></tr>
<tr><td rowspan="9">V_MFMA_16x16x4_F64 Write VGPR</td><td>V_MFMA_16x16x4_F64 read VGPR as Source C exactly same with 1st vDst</td><td>0</td><td>the two V_MFMA must be the same number passes and vDst and vSrc start from the same offset.</td></tr>
<tr><td>S/DGEMM read VGPR as Source C overlapped with 1st vDst</td><td>9</td><td>overlapped, different VGPR access sequence</td></tr>
<tr><td>XDL read VGPR as Source C overlapped with 1st vDst</td><td>0</td><td></td></tr>
<tr><td>V_SMFMA* read VGPR for Matrix C overlapped with 1st vDst</td><td>0</td><td>V_SMFMA reads vdst for Matrix C.</td></tr>
<tr><td>S/DGEMM read VGPR as SrcA or SrcB</td><td>11</td><td>No internal forwarding path, need to wait previous V_MFMA commit result to VGPR</td></tr>
<tr><td>XDL read VGPR as SrcA or SrcB</td><td>11</td><td></td></tr>
<tr><td>V_SMFMA* read VGPR as SrcA or SrcB or Index SrcC</td><td>11</td><td>V_SMFMA uses srcC address for extra Index C Reads</td></tr>
<tr><td>VALU read/write VGPR<br><br>(RAW + WAW)</td><td>11</td><td></td></tr>
<tr><td>VM, L/GDS, FLAT and Export Read VGPR overlapped with 1st vDst</td><td>18</td><td>No internal forwarding path, need to wait previous V_MFMA commit result to VGPR</td></tr>
<tr><td rowspan="9">V_MFMA_4x4x4_F64 Write VGPR</td><td>V_MFMA_4x4x4_F64, read VGPR as Source C exactly same with 1st vDst</td><td>4</td><td>4x4x4 needs to do accumulation, so the write VGPR is later than normal XDL 4x4x4, so needs extra wait</td></tr>
<tr><td>S/DGEMM read VGPR as Source C overlapped with 1st vDst</td><td>4</td><td>overlapped, different VGPR access sequence</td></tr>
<tr><td>XDL read VGPR as Source C overlapped with 1st vDst</td><td>0</td><td></td></tr>
<tr><td>V_SMFMA read VGPR for Matrix C overlapped with 1st vDst</td><td>0</td><td>V_SMFMA reads vdst for Matrix C.</td></tr>
<tr><td>S/DGEMM read VGPR as SrcA or SrcB</td><td>6</td><td>No internal forwarding path, need to wait previous V_MFMA commit result to VGPR</td></tr>
<tr><td>XDL read VGPR as SrcA or SrcB</td><td>6</td><td></td></tr>
<tr><td>V_SMFMA* read VGPR as SrcA or SrcB or Index SrcC</td><td>6</td><td>V_SMFMA uses srcC address for extra Index C Reads</td></tr>
<tr><td>VALU read/write VGPR<br><br>(RAW + WAW)</td><td>6</td><td></td></tr>
<tr><td>VM, L/GDS, FLAT and Export Read VGPR overlapped with 1st vDst</td><td>9</td><td>No internal forwarding path, need to wait previous V_MFMA commit result to VGPR</td></tr>
<tr><td>V_CMPX* write EXEC MASK</td><td>V_MFMA*</td><td>4</td><td>CDNA doesn’t support execution mask forwarding with XDL/DGEMM</td></tr>
<tr><td rowspan="4">XDL/SMFMA Read VGPR SrcC</td><td rowspan="4">VALU write VGPR (WAR), Co-Execution Anti-Dependency for over-lapping with 1st SrcC</td><td>1</td><td>if 1st MFMA is 2 passes</td></tr>
<tr><td>3</td><td>if 1st MFMA is 4 passes</td></tr>
<tr><td>7</td><td>if 1st MFMA is 8 passes</td></tr>
<tr><td>15</td><td>if 1st MFMA is 16 passes</td></tr>
</tbody>
</table>

<!-- page 66 -->
