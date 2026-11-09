"""MXFP4 fused MoE with clamped SwiGLU on gfx950, in FlyDSL.

Routes are sorted expert-major into blocks of tile_m rows (torch, sync-free, fixed upper bound on the row count).
Activations are rounded to MXFP4 per 1 x 32 (packed e2m1 codes plus E8M0 scales laid out in sorted, shuffled order).
Stage 1 is a scaled-fp4 MFMA gate/up GEMM over the 16x16-preshuffled w1 with the clamped SwiGLU epilogue, writing
bf16; its output is rounded to MXFP4 and stage 2 is the scaled-fp4 down GEMM over w2, routing-weighted and
accumulated atomically into the bf16 output. The FlyDSL kernel builders below are aiter's MoE 2-stage builders
(buffer ops, preshuffle pipeline, epilogues, mixed MoE GEMM), each kept in its own namespace.
"""

from __future__ import annotations
from flydsl._mlir.dialects.vector import *
import enum as _enum
import functools as _functools
import types as _types
import torch as _torch
import torch


class _GateMode(str, _enum.Enum):
    SEPARATED = "separated"
    MOCK_GATE_ONLY = "mock_gate_only"
    GATE_ONLY = "gate_only"
    INTERLEAVE = "interleave"


def _get_cu_num():
    return int(_torch.cuda.get_device_properties(_torch.cuda.current_device()).multi_processor_count)


_PRELUDE_END = None

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import builtin
from flydsl._mlir.dialects import gpu as _gpu
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import memref
from flydsl._mlir.dialects import rocdl
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import vector as _vector
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl._mlir.extras import types as T
from flydsl._mlir.extras import types as _mT
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith
from flydsl.expr import arith as _arith
from flydsl.expr import const_expr
from flydsl.expr import gpu
from flydsl.expr import range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.gpu import lds_space as _lds_space
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import ReductionOp
from flydsl.expr.typing import Vector
from flydsl.expr.typing import as_ir_value
from flydsl.expr.typing import empty_like
from flydsl.expr.typing import full
from flydsl.expr.typing import full_like
from flydsl.expr.typing import ones_like
from flydsl.expr.typing import zeros_like
from flydsl.runtime.device import get_rocm_arch
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.runtime.device import is_rdna_arch
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.utils.smem_allocator import SmemPtr
import builtins as _builtins
import flydsl.compiler as flyc
import flydsl.expr as fx
import math as _math
import re

def _mod_buffer_ops():
    # SPDX-License-Identifier: Apache-2.0
    # Copyright (c) 2025 FlyDSL Project Contributors
    # Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

    """AMD buffer load/store operations, vendored into aiter.

    flydsl moved these from ``flydsl.expr.buffer_ops`` to its repo-level
    ``kernels/common/``, which its wheel does not ship, so aiter keeps an
    equivalent copy here. Only published flydsl APIs are used.

    Buffer instructions are an AMD hardware feature (buffer resource descriptor
    plus ROCDL intrinsics) providing out-of-bounds protection and better memory
    throughput; plain memref load/store is not a substitute.

    Upstream: FlyDSL ``kernels/common/buffer_ops.py`` @ ROCm/FlyDSL#880.
    Behavior matches upstream; formatting and type annotations differ only to
    satisfy aiter's lint.

    Example:
        >>> from aiter.ops.flydsl.kernels import buffer_ops
        >>> from flydsl.expr import arith
        >>>
        >>> rsrc = buffer_ops.create_buffer_resource(A)
        >>> offset = row * arith.index(4096) + col
        >>> data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
        >>> buffer_ops.buffer_store(data, rsrc, offset)
    """




    def _get_buffer_flags(arch=None):
        """Get AMD buffer resource descriptor (V#) flags word (bits 127:96).

        Constructs the 32-bit flags field for rocdl.make.buffer.rsrc, following the
        same logic as LLVM's AMDGPUToROCDL makeBufferRsrc():
          https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp

        Bit layout (common to all architectures):
          bits [11:0]  - DST_SEL: ignored by raw buffer intrinsics
          bits [14:12] - DATA_FORMAT: must be nonzero, 7 = float
          bits [18:15] - NUM_FORMAT:  must be nonzero, 4 = 32-bit
          bit  [19]    - In nested heap (0)
          bit  [20]    - Behavior on unmap (0 = return 0 / ignore)
          bits [22:21] - Index stride for swizzles (0)
          bit  [23]    - Add thread ID (0)
          bit  [24]    - Reserved: must be 1 on RDNA, 0 on CDNA
          bits [26:25] - Reserved (0)
          bit  [27]    - Non-volatile (CDNA only, 0)
          bits [29:28] - OOB_SELECT (RDNA only): 0=structured, 2=none, 3=check offset
          bits [31:30] - Type (must be 0)

        CDNA (gfx9xx):    (7 << 12) | (4 << 15)                         = 0x20070
        RDNA (gfx10+):    (7 << 12) | (4 << 15) | (1 << 24) | (2 << 28) = 0x21020070
          - bit 24 set to 1 (required on RDNA)
          - OOB_SELECT=2 (no bounds checking, matching LLVM boundsCheck=false)
        """
        import os

        if arch is None:
            arch = None
        flags = (7 << 12) | (4 << 15)
        if is_rdna_arch(arch):
            flags |= 1 << 24  # reserved bit, must be 1 on RDNA
            flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
        return flags


    __all__ = [
        "BufferResourceDescriptor",
        "buffer_load",
        "buffer_store",
        "create_buffer_resource",
        "create_buffer_resource_from_addr",
        "create_llvm_ptr",
        "extract_base_index",
        "get_element_ptr",
    ]


    def _unwrap_value(value):
        """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.

        Handles:
        - FlyDSL ArithValue (has ._value)
        - flyc DSL Numeric like fx.Int32 (has .ir_value() method)
        - flyc ArithValue (is already ir.Value subclass)
        """
        # DSL Numeric (Int32, Float32, etc.) — use ir_value() to materialize
        if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
            return value.ir_value()
        max_depth = 10  # Safety limit
        depth = 0
        while depth < max_depth and not isinstance(value, ir.Value):
            if hasattr(value, "_value"):
                value = value._value
            elif hasattr(value, "value"):
                value = value.value
            else:
                break
            depth += 1
        return value


    @dsl_loc_tracing
    def _create_i32_constant(value: int) -> ir.Value:
        """Create i32 constant using standard MLIR arith dialect."""
        i32_type = T.i32()
        if value > 0x7FFFFFFF:
            value = int(value - 2**32)
        attr = ir.IntegerAttr.get(i32_type, value)
        op = std_arith.ConstantOp(i32_type, attr)
        return _unwrap_value(op.result)


    def _to_i32_offset(offset: ir.Value) -> ir.Value:
        """Normalize an already-unwrapped offset value to i32.

        Accepts index (index_cast), i32 (as-is), wider ints e.g. i64 (trunc), and
        narrower ints (sign-extend). Lets callers pass fx.Int64 offsets (element or
        byte) without hitting the ``index_cast i64<->i32`` incompatibility.
        """
        ot = offset.type
        if isinstance(ot, ir.IntegerType):
            if ot.width == 32:
                return offset
            if ot.width > 32:
                return _unwrap_value(std_arith.TruncIOp(T.i32(), offset).result)
            return _unwrap_value(std_arith.ExtSIOp(T.i32(), offset).result)
        return _unwrap_value(std_arith.IndexCastOp(T.i32(), offset).result)


    @dsl_loc_tracing
    def _create_i16_constant(value: int) -> ir.Value:
        """Create i16 constant using standard MLIR arith dialect."""
        i16_type = T.i16()
        attr = ir.IntegerAttr.get(i16_type, value)
        op = std_arith.ConstantOp(i16_type, attr)
        return _unwrap_value(op.result)


    @dsl_loc_tracing
    def _create_i64_constant(value: int) -> ir.Value:
        """Create i64 constant using standard MLIR arith dialect."""
        i64_type = T.i64()
        attr = ir.IntegerAttr.get(i64_type, value)
        op = std_arith.ConstantOp(i64_type, attr)
        return _unwrap_value(op.result)


    @dsl_loc_tracing
    def _ptr8_to_v4i32(ptr8_val) -> ir.Value:
        """Reinterpret a buffer resource (!llvm.ptr<8>) as a <4 x i32> vector.

        Required by the scalar ``s.buffer.load`` intrinsic, whose resource operand is
        a v4i32 rather than the opaque buffer pointer used by the vector path.
        """
        i128_ty = ir.IntegerType.get_signless(128)
        v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
        i128_val = llvm.ptrtoint(i128_ty, _unwrap_value(ptr8_val))
        return llvm.bitcast(v4i32_ty, i128_val)


    @dsl_loc_tracing
    def create_llvm_ptr(value, address_space: int = 0) -> ir.Value:
        """Create an LLVM pointer from an integer or index value."""
        value = _unwrap_value(value)
        if isinstance(value.type, ir.IndexType):
            i64_type = T.i64()
            value = _unwrap_value(std_arith.IndexCastOp(i64_type, value).result)
        ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
        return llvm.IntToPtrOp(ptr_type, value).result


    @dsl_loc_tracing
    def extract_base_index(tensor, address_space: int = 1) -> ir.Value:
        """Extract the base address of a fly.memref as an index value.

        Inverse of :func:`create_llvm_ptr` (index -> ptr). Useful when ISA
        requires a raw pointer instead of a buffer resource descriptor
        (e.g. global_atomic_pk_add_bf16 on gfx942).
        """
        from flydsl._mlir.dialects import fly as _fly
        from flydsl._mlir.dialects import memref as _memref

        raw = _unwrap_value(tensor)
        try:
            ir.MemRefType(raw.type)
            return _memref.extract_aligned_pointer_as_index(raw)
        except ValueError:
            pass

        ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
        ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw)
        i64_val = llvm.PtrToIntOp(ir.IntegerType.get_signless(64), ptr).result
        return _unwrap_value(std_arith.IndexCastOp(ir.IndexType.get(), i64_val).result)


    @dsl_loc_tracing
    def get_element_ptr(
        base_ptr,
        byte_offset: int | ir.Value | None = None,
        static_byte_offset: int = 0,
        elem_type: ir.Type | None = None,
        no_wrap_flags=None,
    ) -> ir.Value:
        """Build an LLVM GEP from a base pointer plus byte offsets."""
        _gep_dynamic_index_sentinel = -(2**31)

        base_ptr = _unwrap_value(base_ptr)
        if not isinstance(static_byte_offset, int):
            raise TypeError(
                f"static_byte_offset must be int, got {type(static_byte_offset).__name__}"
            )
        if elem_type is None:
            elem_type = T.i8()
        elif callable(elem_type):
            elem_type = elem_type()

        if byte_offset is None:
            dynamic_indices = []
            raw_constant_indices = [int(static_byte_offset)]
        elif isinstance(byte_offset, int):
            dynamic_indices = []
            raw_constant_indices = [int(byte_offset) + int(static_byte_offset)]
        else:
            offset_val = _unwrap_value(byte_offset)
            if isinstance(offset_val.type, ir.IndexType):
                i64_type = T.i64()
                offset_val = _unwrap_value(
                    std_arith.IndexCastOp(i64_type, offset_val).result
                )
            elif not isinstance(offset_val.type, ir.IntegerType):
                raise TypeError(
                    "byte_offset must be int, index, or integer-typed MLIR value; "
                    f"got {offset_val.type}"
                )

            if static_byte_offset != 0:
                static_type = offset_val.type
                static_attr = ir.IntegerAttr.get(static_type, int(static_byte_offset))
                static_const = _unwrap_value(
                    std_arith.ConstantOp(static_type, static_attr).result
                )
                offset_val = _unwrap_value(
                    std_arith.AddIOp(offset_val, static_const).result
                )

            dynamic_indices = [offset_val]
            raw_constant_indices = [_gep_dynamic_index_sentinel]

        return llvm.GEPOp(
            base_ptr.type,
            base_ptr,
            dynamic_indices,
            raw_constant_indices,
            elem_type,
            no_wrap_flags,
        ).result


    class BufferResourceDescriptor:
        """AMD Buffer Resource Descriptor

        A buffer resource descriptor contains:
        - base_pointer: Scalar base pointer (wave-uniform, stored in SGPRs)
        - stride: Stride for structured buffers (typically 0 for contiguous)
        - num_records: Buffer size in bytes
        - flags: Data format and access flags

        The descriptor is stored in a special LLVM pointer type (!llvm.ptr<8>)
        """

        def __init__(self, rsrc: ir.Value):
            """Initialize with ROCDL resource descriptor value."""
            self.rsrc = rsrc

        @staticmethod
        @dsl_loc_tracing
        def from_memref(
            memref_val: ir.Value,
            stride: int = 0,
            max_size: bool = True,
            data_format: str = "f32",
            num_records_bytes: int | ir.Value | None = None,
            base_byte_offset: int | ir.Value | None = None,
        ) -> BufferResourceDescriptor:
            """Create buffer resource descriptor from memref.

            Args:
                memref_val: Memref value to create descriptor for
                stride: Stride in elements (0 for contiguous)
                max_size: If True, use max buffer size for flexibility
                num_records_bytes: Override buffer size (in BYTES) used by hardware OOB checking.
                                  If provided, this takes precedence over `max_size`.
                base_byte_offset: Optional byte offset added to the descriptor base pointer.
                data_format: Data format ('f32', 'f16', 'i32', etc.)

            Returns:
                BufferResourceDescriptor instance

            Example:
                >>> rsrc = BufferResourceDescriptor.from_memref(A)
            """
            # Extract raw pointer from fly.memref.
            raw_val = _unwrap_value(memref_val)
            from flydsl._mlir.dialects import fly as _fly

            ptr_type = ir.Type.parse("!llvm.ptr")
            base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw_val)
            if base_byte_offset is not None:
                base_ptr = get_element_ptr(base_ptr, byte_offset=base_byte_offset)

            # Create buffer resource descriptor
            flags_val = _get_buffer_flags()
            flags = _create_i32_constant(flags_val)
            stride_val = _create_i16_constant(stride)

            def _num_records_from_memref_type() -> int | None:
                """Best-effort: derive logical buffer size (in bytes) from static memref type."""
                try:
                    mt = ir.MemRefType(_unwrap_value(memref_val).type)
                    shape = list(mt.shape)
                    if any(int(d) < 0 for d in shape):
                        return None
                    # Compute element size in bytes (scalar element type).
                    elem_t = mt.element_type
                    elem_bits = getattr(elem_t, "width", None)
                    if elem_bits is None:
                        return None
                    elem_bytes = int(elem_bits) // 8
                    if elem_bytes <= 0:
                        return None
                    num_elems = 1
                    for d in shape:
                        num_elems *= int(d)
                    return int(num_elems) * int(elem_bytes)
                # best-effort size probe: any failure just means "unknown"
                except Exception:  # noqa: BLE001
                    return None

            if num_records_bytes is not None:
                # Caller-provided size in BYTES (preferred for exact hardware OOB behavior).
                if isinstance(num_records_bytes, int):
                    nbytes = int(num_records_bytes)
                    nbytes = max(0, nbytes)
                    # Descriptor uses i32 bytes; clamp to the max representable.
                    nbytes = min(nbytes, 0xFFFFFFFF)
                    num_records = _create_i64_constant(nbytes)
                else:
                    v = _unwrap_value(num_records_bytes)
                    i64_type = T.i64()
                    if not isinstance(v.type, ir.IntegerType) or v.type.width != 64:
                        if isinstance(v.type, ir.IndexType):
                            op = std_arith.IndexCastOp(i64_type, v)
                        else:
                            op = std_arith.ExtSIOp(i64_type, v)
                        v = _unwrap_value(op.result)
                    num_records = v
            elif max_size:
                # Use max for flexibility (hardware will check actual bounds)
                # Note: FlyDSL's rocdl.make.buffer.rsrc requires i32, not i64
                num_records = _create_i64_constant(0xFFFFFFFF)  # FALLBACK_MAX_SIZE
            else:
                # Use the logical memref size (in bytes) for hardware OOB checking.
                nbytes = _num_records_from_memref_type()
                if nbytes is None:
                    # Fall back to max-size if we can't infer statically.
                    num_records = _create_i64_constant(0xFFFFFFFF)
                else:
                    nbytes = min(nbytes, 0xFFFFFFFF)
                    num_records = _create_i64_constant(int(nbytes))

            # Create resource descriptor (returns !llvm.ptr<8>)
            rsrc_type = ir.Type.parse("!llvm.ptr<8>")
            rsrc = rocdl.MakeBufferRsrcOp(
                rsrc_type, base_ptr, stride_val, num_records, flags
            ).result

            return BufferResourceDescriptor(rsrc)


    @dsl_loc_tracing
    def create_buffer_resource_from_addr(
        addr_i64: ir.Value,
        *,
        num_records_bytes: int | ir.Value | None = None,
    ) -> ir.Value:
        """Create AMD buffer resource descriptor from a raw i64 device address.

        Useful when working with runtime pointer arrays (e.g. IPC-mapped addresses
        or device-side pointer tables) where no fly.memref is available.
        The full address is encoded as the buffer base; callers should pass
        byte offset 0 to buffer_load / buffer_store.

        Args:
            addr_i64: Raw 64-bit device address (i64 MLIR value).
            num_records_bytes: Optional buffer size in bytes for hardware OOB checking.

        Returns:
            ROCDL buffer resource descriptor (!llvm.ptr<8>).

        Example:
            >>> rsrc = create_buffer_resource_from_addr(raw_addr_i64)
            >>> data = buffer_load(rsrc, i32_zero, vec_width=4, dtype=T.i32)
        """
        addr_i64 = _unwrap_value(addr_i64)
        ptr_type = ir.Type.parse("!llvm.ptr")
        base_ptr = llvm.IntToPtrOp(ptr_type, addr_i64).result
        flags = _create_i32_constant(_get_buffer_flags())
        stride = _create_i16_constant(0)
        if num_records_bytes is None:
            num_records = _create_i64_constant(0xFFFFFFFF)
        elif isinstance(num_records_bytes, int):
            nbytes = max(0, min(int(num_records_bytes), 0xFFFFFFFF))
            num_records = _create_i64_constant(nbytes)
        else:
            num_records = _unwrap_value(num_records_bytes)
            i64_type = T.i64()
            if (
                not isinstance(num_records.type, ir.IntegerType)
                or num_records.type.width != 64
            ):
                if isinstance(num_records.type, ir.IndexType):
                    num_records = _unwrap_value(
                        std_arith.IndexCastOp(i64_type, num_records).result
                    )
                else:
                    num_records = _unwrap_value(
                        std_arith.ExtSIOp(i64_type, num_records).result
                    )
        rsrc_type = ir.Type.parse("!llvm.ptr<8>")
        return rocdl.MakeBufferRsrcOp(
            rsrc_type, base_ptr, stride, num_records, flags
        ).result


    @dsl_loc_tracing
    def create_buffer_resource(
        memref_val: ir.Value,
        stride: int = 0,
        max_size: bool = True,
        *,
        num_records_bytes: int | ir.Value | None = None,
        base_byte_offset: int | ir.Value | None = None,
    ) -> ir.Value:
        """Create AMD buffer resource descriptor from memref.

        This is a simplified wrapper around BufferResourceDescriptor.from_memref()
        that returns the raw ROCDL resource value.

        Args:
            memref_val: Memref value
            stride: Buffer stride (0 for contiguous)
            max_size: Use maximum buffer size
            num_records_bytes: Override buffer size in bytes.
            base_byte_offset: Optional byte offset added to the descriptor base pointer.

        Returns:
            ROCDL buffer resource descriptor (!llvm.ptr<8>)

        Example:
            >>> rsrc = create_buffer_resource(A)
            >>> data = buffer_load(rsrc, offset)
        """
        desc = BufferResourceDescriptor.from_memref(
            memref_val,
            stride,
            max_size,
            num_records_bytes=num_records_bytes,
            base_byte_offset=base_byte_offset,
        )
        return desc.rsrc


    @dsl_loc_tracing
    def buffer_load(
        rsrc: ir.Value,
        offset: ir.Value,
        vec_width: int = 4,
        dtype=None,
        mask: ir.Value | None = None,
        cache_modifier: int = 0,
        soffset_bytes: int | ir.Value | None = None,
        is_scalar: bool = False,
    ) -> ir.Value:
        """AMD buffer load operation.

        Load data from global memory using buffer descriptor and offset.
        Uses hardware-level bounds checking and vectorization.

        Args:
            rsrc: Buffer resource descriptor (!llvm.ptr<8>)
            offset: Offset in elements (i32 type)
            vec_width: Vector width (1, 2, or 4)
            dtype: Element data type (None for f32, or ir.F32Type, etc.)
            mask: Optional mask for predicated load (i1 type)
            cache_modifier: Cache control flags (0 for default)
            soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                          Use this to fold small constant deltas into the instruction instead of emitting
                          extra VGPR address arithmetic.
            is_scalar: Emit a uniform/SGPR scalar load (llvm.amdgcn.s.buffer.load) instead of the
                          vector buffer load. Use only for wave-uniform addresses to route through the
                          SMEM cache and land the result directly in SGPRs. Restricted to vec_width 1 or 4;
                          dtype is forced to i32 (the result is raw i32 dwords). mask and soffset_bytes
                          are not supported in this mode and raise ValueError if provided.

        Returns:
            Loaded data (scalar or vector depending on vec_width)

        Example:
            >>> # Load 4xf32
            >>> data = buffer_load(rsrc, offset, vec_width=4)
            >>>
            >>> # Load with mask
            >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
        """
        # Scalar (uniform) loads return raw i32 dwords; force the element type so the
        # element->byte offset math below uses 4 and the result type is i32 / v4i32.
        if is_scalar:
            if vec_width not in (1, 4):
                raise ValueError(
                    f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}"
                )
            if mask is not None or soffset_bytes is not None:
                raise ValueError(
                    "buffer_load(is_scalar=True) does not support mask or soffset_bytes"
                )
            dtype = T.i32()
        # Default dtype to f32
        elif dtype is None:
            dtype = T.f32()
        # Accept DSL Numeric class (e.g. fx.Int32) as dtype: unwrap to ir.Type
        elif hasattr(dtype, "ir_type"):
            dtype = dtype.ir_type

        # Unwrap offset first (accept Python ints and DSL Numeric values).
        if isinstance(offset, int):
            offset = _create_i32_constant(offset)
        elif hasattr(offset, "ir_value"):
            offset = offset.ir_value()
        offset = _unwrap_value(offset)

        # Convert offset to i32 if needed (accepts index/i64/i32)
        offset = _to_i32_offset(offset)

        # IMPORTANT: Buffer load offset is in BYTES, not elements!
        # For vec4xf32, each element is 4 bytes, so multiply offset by 4
        element_bytes = dtype.width // 8
        bytes_const = _create_i32_constant(element_bytes)
        op = std_arith.MulIOp(offset, bytes_const)
        offset = _unwrap_value(op.result)

        # Apply mask by setting invalid offsets to max
        if mask is not None:
            mask = _unwrap_value(mask)
            max_offset = _create_i32_constant(0x7FFFFFFF)
            op = std_arith.SelectOp(mask, offset, max_offset)
            offset = _unwrap_value(op.result)

        # Create vector type
        if vec_width == 1:
            result_type = dtype
        else:
            result_type = ir.VectorType.get([vec_width], dtype)

        # Scalar/uniform load path: emit s.buffer.load with a v4i32 resource and the
        # byte offset computed above. Returns i32 (vec_width 1) or v4i32 (vec_width 4).
        if is_scalar:
            rsrc_v4 = _ptr8_to_v4i32(rsrc)
            cache_policy = _create_i32_constant(cache_modifier)
            suffix = "i32" if vec_width == 1 else "v4i32"
            return llvm.call_intrinsic(
                result_type,
                f"llvm.amdgcn.s.buffer.load.{suffix}",
                [rsrc_v4, offset, cache_policy],
                [],
                [],
            )

        # Create instruction offset and aux flags
        if soffset_bytes is None:
            soffset = _create_i32_constant(0)
        else:
            if isinstance(soffset_bytes, int):
                soffset = _create_i32_constant(soffset_bytes)
            else:
                soffset = _to_i32_offset(_unwrap_value(soffset_bytes))
        aux_attr = (
            ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
            if cache_modifier
            else None
        )

        # Emit buffer load
        load_op = rocdl.RawPtrBufferLoadOp(
            result_type,
            rsrc,
            offset,
            soffset,
            aux=aux_attr,
        )

        return load_op.result


    @dsl_loc_tracing
    def buffer_store(
        data: ir.Value,
        rsrc: ir.Value,
        offset: ir.Value,
        mask: ir.Value | None = None,
        cache_modifier: int = 0,
        *,
        soffset_bytes: int | ir.Value | None = None,
        offset_is_bytes: bool = False,
    ):
        """AMD buffer store operation.

        Store data to global memory using buffer descriptor and offset.

        Args:
            data: Data to store (scalar or vector)
            rsrc: Buffer resource descriptor (!llvm.ptr<8>)
            offset: Offset in elements (i32 type)
            mask: Optional mask for predicated store (i1 type)
            cache_modifier: Cache control flags (0 for default)

        Example:
            >>> buffer_store(data, rsrc, offset)
            >>>
            >>> # Store with mask
            >>> buffer_store(data, rsrc, offset, mask=valid)
        """
        # Unwrap all inputs (accept DSL Numeric values via ir_value())
        if hasattr(data, "ir_value"):
            data = data.ir_value()
        if isinstance(offset, int):
            offset = _create_i32_constant(offset)
        elif hasattr(offset, "ir_value"):
            offset = offset.ir_value()
        data = _unwrap_value(data)
        rsrc = _unwrap_value(rsrc)
        offset = _unwrap_value(offset)

        # Convert offset to i32 if needed (accepts index/i64/i32)
        offset = _to_i32_offset(offset)

        # IMPORTANT: RawPtrBufferStoreOp offset is in BYTES.
        # For backward compat, `buffer_store()` accepts element offsets by default
        # and scales them to bytes. Set `offset_is_bytes=True` to skip scaling.
        if not offset_is_bytes:
            # Get element size from data type
            data_type = data.type
            if hasattr(data_type, "element_type"):  # Vector type
                element_type = data_type.element_type
            else:  # Scalar type
                element_type = data_type
            element_bytes = element_type.width // 8
            bytes_const = _create_i32_constant(element_bytes)
            op = std_arith.MulIOp(offset, bytes_const)
            offset = _unwrap_value(op.result)

        # Apply mask by setting invalid offsets to max
        if mask is not None:
            mask = _unwrap_value(mask)
            max_offset = _create_i32_constant(0x7FFFFFFF)
            op = std_arith.SelectOp(mask, offset, max_offset)
            offset = _unwrap_value(op.result)

        # Create instruction offset (soffset) and aux flags
        if soffset_bytes is None:
            soffset = _create_i32_constant(0)
        else:
            if isinstance(soffset_bytes, int):
                soffset = _create_i32_constant(int(soffset_bytes))
            else:
                soffset = _to_i32_offset(_unwrap_value(soffset_bytes))
        aux_attr = (
            ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
            if cache_modifier
            else None
        )

        # Emit buffer store
        rocdl.RawPtrBufferStoreOp(
            data,
            rsrc,
            offset,
            soffset,
            aux=aux_attr,
        )
    return _types.SimpleNamespace(**dict(locals()))
_M_buffer_ops = _mod_buffer_ops()

def _mod_vector():
    # SPDX-License-Identifier: Apache-2.0
    # Copyright (c) 2025 FlyDSL Project Contributors
    # Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

    """Vector dialect wrappers, vendored into aiter.

    flydsl deleted ``flydsl.expr.vector``; callers are now expected to use the raw
    ``flydsl._mlir.dialects.vector`` and wrap every operand in ``as_ir_value``.
    aiter has 500+ vector call sites, so the auto-unwrapping layer is kept here
    instead — a missing ``as_ir_value`` or ``kDynamic`` sentinel only surfaces at
    trace time.

    Re-exports the whole raw dialect (``broadcast``, ``shuffle``, ``insert``,
    ``extract_strided_slice``, ``reduction``, ...) and rebuilds the five wrappers
    that unwrap DSL values: ``from_elements``, ``store``, ``extract``,
    ``load_op``, ``bitcast``. ``extract`` also supplies the ``kDynamic`` sentinel
    for dynamic positions. Only published flydsl APIs are used.

    Upstream: FlyDSL ``python/flydsl/expr/vector.py``, deleted before
    ROCm/FlyDSL#880; the wrappers use FlyDSL's canonical ``as_ir_value``
    converter.
    """



    # Re-export the raw dialect so vector.broadcast / vector.shuffle work directly

    # Vector and related types, which flydsl.expr.vector also re-exported

    # ═══════════════════════════════════════════════════════════════════════
    # Dialect helper wrappers (legacy, will be deprecated)
    # Prefer using Vector methods or _mlir.dialects.vector directly.
    # ═══════════════════════════════════════════════════════════════════════


    def _as_index_ir_value(value):
        if isinstance(value, int):
            from flydsl.expr import arith as _arith_ext

            return _arith_ext.constant(value, index=True)
        v = as_ir_value(value)
        # vector/memref indices must be index-typed; cast integer offsets
        # (e.g. fx.Int64/i32) so callers can pass explicit-width offsets.
        if isinstance(v.type, ir.IntegerType):
            from flydsl._mlir.dialects import arith as _std_arith

            v = _std_arith.IndexCastOp(ir.IndexType.get(), v).result
        return v


    @dsl_loc_tracing
    def from_elements(*args, **kwargs):
        """Construct a vector from scalar elements, auto-unwrapping ArithValue wrappers."""
        if len(args) >= 2:
            args = list(args)
            elems = args[1]
            if isinstance(elems, (list, tuple)):
                args[1] = [as_ir_value(v) for v in elems]
            return _vector.from_elements(*args, **kwargs)

        return _vector.from_elements(*args, **kwargs)


    @dsl_loc_tracing
    def store(value, memref, indices, **kwargs):
        """Vector store wrapper that accepts ArithValue/wrappers for value/indices."""
        return _vector.store(
            as_ir_value(value),
            as_ir_value(memref),
            [_as_index_ir_value(i) for i in indices],
            **kwargs,
        )


    # -----------------------------------------------------------------------------
    # Thin wrappers for common op classes that otherwise require `.result` access.
    # -----------------------------------------------------------------------------


    @dsl_loc_tracing
    def extract(vector, static_position=None, dynamic_position=None):
        """Wrapper around `vector.ExtractOp(...).result`.

        When only ``dynamic_position`` is supplied (without explicit
        ``static_position``), each dynamic index needs a corresponding
        ``kDynamic`` sentinel in the static attribute so the ODS builder
        pairs them correctly.  This wrapper fills in the sentinels
        automatically.
        """
        if static_position is None:
            static_position = []
        if dynamic_position is None:
            dynamic_position = []
        dynamic_position = [_as_index_ir_value(i) for i in dynamic_position]

        n_static = len(static_position)
        n_dynamic = len(dynamic_position)
        if n_dynamic > 0 and n_static < n_dynamic:
            kDynamic = ir.ShapedType.get_dynamic_size()
            static_position = list(static_position) + [kDynamic] * (n_dynamic - n_static)

        return _vector.ExtractOp(
            as_ir_value(vector),
            static_position=static_position,
            dynamic_position=dynamic_position,
        ).result


    @dsl_loc_tracing
    def load_op(result_type, memref, indices):
        """Wrapper around `vector.LoadOp(...).result`."""
        return _vector.LoadOp(
            result_type,
            as_ir_value(memref),
            [_as_index_ir_value(i) for i in indices],
        ).result


    @dsl_loc_tracing
    def bitcast(result_type, source):
        """Wrapper around `vector.BitCastOp(...).result`."""
        return _vector.BitCastOp(
            result_type,
            as_ir_value(source),
        ).result
    return _types.SimpleNamespace(**dict(locals()))
_M_vector = _mod_vector()

def _mod_kernels_common():
    """Common helpers shared by kernel modules.

    Keep helper naming consistent with other kernel helpers (e.g. `mfma_preshuffle_pipeline.py`),
    but this module is intentionally small and MLIR-dialect facing.
    """

    from flydsl.expr.typing import T

    buffer_ops = _M_buffer_ops


    def format_kernel_name(name: str) -> str:
        """Sanitize a kernel symbol name for the amdhsa assembler.

        Config values interpolated into a kernel name may be negative (e.g. the
        grouped-contiguous sentinel ``topk=-1`` renders as ``tk-1``). A hyphen is
        not a legal symbol character, so the assembler misparses the
        ``.amdhsa_kernel`` directive and the whole module fails to link.
        """
        return name.replace("-", "_")


    def get_warp_size(arch=None):
        """Return the wavefront/warp size for the given GPU architecture.

        CDNA (gfx9xx) uses wave64, RDNA (gfx10xx/gfx11xx/gfx12xx) uses wave32.
        NOTE: we do not defer the gfx12 case to ``flydsl.runtime.device.is_rdna_arch``:
        that helper only matches ``gfx120*`` and so misclassifies gfx1250 (which is
        wave32, RDNA-family) as CDNA, yielding wave64. Building a kernel for wave64
        while gfx1250 dispatches wave32 corrupts every >1-warp-per-block kernel
        (the phantom upper lanes silently drop their work). Classify all gfx10/11/12
        as wave32 directly here so the kernel body matches the wave32 dispatch.
        """
        if arch is None:
            arch = get_rocm_arch()
        arch_l = (arch or "").lower()
        if arch_l.startswith(("gfx10", "gfx11", "gfx12")):
            return 32
        return 32 if is_rdna_arch(arch) else 64


    def default_f8_type() -> ir.Type:
        """Select the E4M3 f8 type compatible with the current GPU arch.

        - gfx95* (MI350): FP8 E4M3FN (OCP)
        - gfx12*: FP8 E4M3FN (OCP)
        - gfx94* (MI300): FP8 E4M3FNUZ

        Raises ``RuntimeError`` on gfx11* (RDNA3/RDNA3.5): these chips have no
        native FP8 instructions, so FP8 compute would surface as a late LLVM
        "cannot select" error. Fail early with a clear message instead.

        Replaces the ``T.f8`` shortcut removed from ``flydsl.expr.typing`` in
        flydsl 0.3.0 (upstream moved arch-specific FP8 selection into the kernel
        layer); mirrors that helper here for the vendored kernels.
        """
        arch = ""
        try:
            arch = str(get_rocm_arch())
        except Exception:  # noqa: BLE001
            arch = ""
        if "gfx95" in arch or "gfx12" in arch:
            return fx.Float8E4M3FN.ir_type
        if arch.startswith("gfx11"):
            raise RuntimeError(
                f"default_f8_type(): no native FP8 support on {arch}; "
                "FP8 instructions are available on gfx94*, gfx95*, and gfx12*."
            )
        return fx.Float8E4M3FNUZ.ir_type


    def dtype_to_elem_type(dtype_str: str):
        """Map a dtype string to its MLIR scalar type.

        Supported: ``'f32'``, ``'f16'``, ``'bf16'``.
        """
        if dtype_str == "f32":
            return T.f32
        if dtype_str == "f16":
            return T.f16
        if dtype_str == "bf16":
            return T.bf16
        raise ValueError(
            f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', or 'bf16')"
        )


    def _create_llvm_ptr(value, address_space: int = 1):
        value = buffer_ops._unwrap_value(value)
        if isinstance(value.type, ir.IndexType):
            i64_type = T.i64
            value = buffer_ops._unwrap_value(_std_arith.IndexCastOp(i64_type, value).result)
        ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
        return _llvm.IntToPtrOp(ptr_type, value).result


    def stream_ptr_to_async_token(stream_ptr_value, loc=None, ip=None):
        stream_llvm_ptr = _create_llvm_ptr(stream_ptr_value)

        async_token_type = _gpu.AsyncTokenType.get()
        cast_op = builtin.UnrealizedConversionCastOp(
            [async_token_type], [stream_llvm_ptr], loc=loc, ip=ip
        )
        return cast_op.results[0]
    return _types.SimpleNamespace(**dict(locals()))
_M_kernels_common = _mod_kernels_common()

def _mod_layout_utils():
    """Layout helpers for GEMM kernels.

    Parses fly layout type strings (e.g. '(4,64):(64,1)') and computes
    idx2crd / crd2idx with plain arith ops for static layouts.
    Falls back to fly dialect ops for dynamic layouts.

    Optimisation: power-of-2 strides/shapes emit ``shrui`` / ``andi`` instead of
    ``divui`` / ``remui``, avoiding 10-15-cycle V_DIV sequences on CDNA GPUs.
    """


    from flydsl.expr.typing import T


    def _wrap(v):
        """Wrap raw ir.Value in ArithValue for operator overloading compatibility."""
        if isinstance(v, ArithValue):
            return v
        if isinstance(v, ir.Value):
            return ArithValue(v)
        return v


    def _is_pow2(n):
        """Return True when *n* is a positive power of two."""
        return n > 0 and (n & (n - 1)) == 0


    def _div_pow2(val, divisor):
        """Unsigned divide index *val* by a **compile-time** power-of-2 *divisor*.

        Emits ``arith.shrui`` (1 VALU cycle) instead of ``arith.divui``
        (10-15 VALU cycles on CDNA).
        """
        shift = _math.log2(divisor)
        assert shift == int(shift), f"{divisor} is not a power of 2"
        return arith.shrui(val, arith.index(int(shift)))


    def _mod_pow2(val, modulus):
        """Unsigned remainder of index *val* by a **compile-time** power-of-2 *modulus*.

        Emits ``arith.andi`` (1 VALU cycle) instead of ``arith.remui``.
        """
        return arith.andi(val, arith.index(modulus - 1))


    def _parse_dim(tok):
        """Parse a single dimension token: '?' -> None, otherwise int."""
        tok = tok.strip()
        return None if tok == "?" else int(tok)


    def _parse_layout(ly):
        """Parse '(s0,s1,...):(d0,d1,...)' -> (shapes, strides) as lists (None for '?')."""
        ly_str = str(ly.type) if hasattr(ly, "type") else str(ly)
        m = re.search(r"\(([^)]+)\):\(([^)]+)\)", ly_str)
        if not m:
            return None
        shapes = [_parse_dim(s) for s in m.group(1).split(",")]
        strides = [_parse_dim(s) for s in m.group(2).split(",")]
        return shapes, strides


    def _has_dynamic_strides(strides):
        """Check if any stride is dynamic (None)."""
        return any(s is None for s in strides)


    def idx2crd(idx, layout):
        """Decompose flat index into a list of coordinate values.

        For static layouts, computes coordinates with plain arith ops.
        Power-of-2 strides/shapes use shift/mask instead of div/rem.
        For dynamic layouts, falls back to fx.idx2crd + fx.get.
        """
        parsed = _parse_layout(layout)

        if hasattr(idx, "ir_value"):
            idx = idx.ir_value()

        if parsed is None or _has_dynamic_strides(parsed[1]):
            result = fx.idx2crd(fx.Int32(idx), layout)
            ndims = len(parsed[1]) if parsed else 1
            return [_wrap(fx.get(result, i)) for i in range(ndims)]

        if isinstance(idx, ir.Value) and not isinstance(idx.type, ir.IndexType):
            idx = arith.index_cast(T.index, idx)
        shapes, strides = parsed
        ndims = len(strides)

        ordered = sorted(
            [
                (i, s, sz)
                for i, s, sz in _builtins.zip(range(ndims), strides, shapes)
                if s != 0
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        coords = [None] * ndims
        remaining = idx
        for i, stride_val, size_val in ordered:
            if stride_val == 1:
                c = remaining
            elif _is_pow2(stride_val):
                c = _div_pow2(remaining, stride_val)
            else:
                c = remaining / arith.index(stride_val)
            if size_val is not None:
                if _is_pow2(size_val):
                    c = _mod_pow2(c, size_val)
                else:
                    c = c % arith.index(size_val)
            coords[i] = c
        for i in range(ndims):
            if coords[i] is None:
                coords[i] = remaining
        return coords


    def crd2idx(crd, layout):
        """Compute flat index from a coordinate tuple/list.

        For static layouts, computes with plain arith ops.
        For dynamic layouts, falls back to fx.crd2idx with fx.make_coord.
        """
        if not isinstance(crd, (list, tuple)):
            crd = [crd]
        parsed = _parse_layout(layout)

        if parsed is None or _has_dynamic_strides(parsed[1]):
            # fly.make_coord requires i32/i64, not index
            crd_i32 = []
            for c in crd:
                cv = c
                if isinstance(cv, ArithValue):
                    cv = cv.ir_value() if hasattr(cv, "ir_value") else cv
                if isinstance(cv, ir.Value) and isinstance(cv.type, ir.IndexType):
                    cv = arith.index_cast(T.i32, cv)
                crd_i32.append(cv)
            coord_val = fx.make_coord(*crd_i32)
            scalar = fx.get_scalar(fx.crd2idx(coord_val, layout)).ir_value()
            if not isinstance(scalar.type, ir.IndexType):
                scalar = arith.index_cast(T.index, scalar)
            return _wrap(scalar)

        _, strides = parsed
        result = None
        for coord_v, stride_v in _builtins.zip(crd, strides):
            if stride_v == 0:
                continue
            term = coord_v if stride_v == 1 else coord_v * arith.index(stride_v)
            result = term if result is None else result + term
        return result if result is not None else arith.index(0)


    def get(int_tuple, mode):
        """Extract element at `mode` from a Python list/tuple."""
        return int_tuple[mode]
    return _types.SimpleNamespace(**dict(locals()))
_M_layout_utils = _mod_layout_utils()

def _mod_mfma_epilogues():
    # SPDX-License-Identifier: Apache-2.0
    # Copyright (c) 2025 FlyDSL Project Contributors

    """Reusable epilogue helpers for MFMA 16x16-based kernels.

    This module provides:

    - `mfma_epilog(...)`
      A single entrypoint that dispatches to either the default row-epilogue or the
      LDS CShuffle epilogue based on input parameters.

    - `default_epilog(...)` (implementation helper)
      A lightweight row-iterator for the common MFMA accumulator-to-output mapping
      (mi in [0,m_repeat), ii in [0,4), row = bx_m + mi*16 + lane_div_16*4 + ii).
      The caller supplies `body_row(...)` that performs the per-row epilogue work
      (e.g. loads scales once, loops over ni, stores).

    - `c_shuffle_epilog(...)` (implementation helper)
      A LDS CShuffle epilogue skeleton:
        1) call `write_row_to_lds(...)` for each MFMA output row to populate `lds_out`
           in row-major [tile_m, tile_n] order
        2) barrier
        3) remap threads into (MLane, NLane) = (8,32) and read half2 from LDS,
           then call `store_pair(...)` to emit the final global store/atomic.

      When ``lds_out_split`` is provided, the epilogue runs in split-LDS mode:
      waves are partitioned into two groups (group A uses ``lds_out``, group B
      uses ``lds_out_split``), each handling half of the N dimension.

    These helpers are intentionally *dialect-agnostic*: callers pass the dialect
    modules (`arith`, `vector`, `gpu`) and the `range_constexpr` iterator.
    """



    from flydsl.expr.typing import T


    @contextmanager
    def _if_then(if_op, scf):
        """Compat helper for SCF IfOp then-region across old/new Python APIs."""
        with ir.InsertionPoint(if_op.then_block):
            try:
                yield if_op.then_block
            finally:
                blk = if_op.then_block
                if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                    scf.YieldOp([])


    def default_epilog(
        *,
        arith,
        range_constexpr,
        m_repeat: int,
        lane_div_16,
        bx_m,
        body_row: Callable,
    ):
        """Iterate the standard MFMA 16x16 row mapping and call `body_row(...)`.

        The mapping matches the common MFMA fragment layout used across kernels in this repo.

        Args:
          arith: flydsl arith ext module.
          range_constexpr: compile-time unrolled range helper.
          m_repeat: tile_m // 16 (python int).
          lane_div_16: index Value (0..3).
          bx_m: base row (index Value). For MoE, this is the base sorted-row for the tile.
          body_row: callback invoked as:
            body_row(mi=<int>, ii=<int>, row_in_tile=<index>, row=<index>)
        """
        bx_m_v = bx_m
        lane_div_16_mul4 = lane_div_16 * 4
        ii_idx_list = [fx.Index(ii) for ii in range(4)]

        for mi in range_constexpr(m_repeat):
            mi_base = arith.constant(mi * 16, index=True)
            for ii in range_constexpr(4):
                row_off = lane_div_16_mul4 + ii_idx_list[ii]
                row_in_tile = mi_base + row_off
                row = bx_m_v + row_in_tile
                body_row(mi=mi, ii=ii, row_in_tile=row_in_tile, row=row)


    def c_shuffle_epilog(
        *,
        arith,
        vector,
        gpu,
        scf=None,
        range_constexpr,
        # Tile params
        tile_m: int,
        tile_n: int,
        e_vec: int = 2,
        cshuffle_nlane: int = 32,
        block_size: int = 256,
        m_repeat: int,
        num_acc_n: int,
        # Thread mapping inputs
        tx,
        lane_div_16,
        lane_mod_16,
        bx_m,
        by_n,
        n_tile_base,
        # LDS buffer (f16 view, row-major [tile_m, tile_n] flattened)
        lds_out,
        # Element type for LDS loads (defaults to f16). Pass bf16 to support bf16 epilogues.
        frag_elem_type: ir.Type | None = None,
        # Callbacks
        write_row_to_lds: Callable,
        precompute_row: Callable | None = None,
        store_pair: Callable,
        # When LDS overflows, split lds_out across two buffers by wave-group.
        # Pass the second buffer here; first buffer is `lds_out`.
        lds_out_split=None,
        # Row offset in lds_out for 8-wave mode (MLIR index value).
        # Shifts both write and read LDS indices by lds_row_offset * tile_n elements.
        lds_row_offset=None,
    ):
        """LDS CShuffle epilogue skeleton.

        Call pattern:
          - `write_row_to_lds(...)` is called once per MFMA row produced by this thread.
            It is responsible for writing all ni columns for that row into `lds_out`.
          - `store_pair(...)` is called for each (row_local, col_pair0) half2 after shuffle.

        `store_pair` can implement either global stores or atomics.
        """
        if int(block_size) <= 0 or (int(block_size) % int(cshuffle_nlane)) != 0:
            raise ValueError(
                f"block_size ({block_size}) must be divisible by cshuffle_nlane ({cshuffle_nlane})"
            )
        cshuffle_mlane = int(block_size) // int(cshuffle_nlane)
        if (int(tile_m) % cshuffle_mlane) != 0:
            raise ValueError(
                f"tile_m must be divisible by CShuffleMLane ({cshuffle_mlane}), got tile_m={tile_m}"
            )
        if int(e_vec) <= 0:
            raise ValueError(f"e_vec must be positive, got {e_vec}")
        if (int(tile_n) % (int(cshuffle_nlane) * int(e_vec))) != 0:
            raise ValueError(
                f"tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_vec}, got tile_n={tile_n}"
            )

        # ===================== Split-LDS mode (early return) =====================
        # When lds_out_split is provided, waves are divided into two groups:
        #   Group A (waves 0..N/2-1) uses lds_out,  columns [0, tile_n/2)
        #   Group B (waves N/2..N-1) uses lds_out_split, columns [tile_n/2, tile_n)
        # Each group writes/reads independently; same barriers synchronise all waves.
        if lds_out_split is not None:
            if scf is None:
                raise ValueError("scf module is required for split-LDS cshuffle")

            _half_n = int(tile_n) // 2
            _half_threads = int(block_size) // 2
            EVec = int(e_vec)

            CShuffleNLane_s = min(int(cshuffle_nlane), _half_n // EVec)
            if _half_threads % CShuffleNLane_s != 0:
                raise ValueError(
                    f"half_threads={_half_threads} not divisible by CShuffleNLane_split={CShuffleNLane_s}"
                )
            CShuffleMLane_s = _half_threads // CShuffleNLane_s
            if int(tile_m) % CShuffleMLane_s != 0:
                raise ValueError(
                    f"tile_m={tile_m} not divisible by CShuffleMLane_split={CShuffleMLane_s}"
                )
            m_reps_s = int(tile_m) // CShuffleMLane_s
            n_reps_s = _half_n // (CShuffleNLane_s * EVec)

            _half_n_idx = arith.constant(_half_n, index=True)
            _half_thr_idx = arith.constant(_half_threads, index=True)
            _zero_idx = arith.constant(0, index=True)

            _is_group_b = arith.cmpi(CmpIPredicate.uge, tx, _half_thr_idx)

            # -- write phase (all waves, each to its group's LDS buffer) --
            n_tile_base_v = n_tile_base
            col_base_local_a = n_tile_base_v + lane_mod_16
            col_base_local_b = col_base_local_a - _half_n_idx

            def _write_row_split(mi: int, ii: int, row_in_tile, row):
                row_base_lds = row_in_tile * _half_n_idx
                _if_g = scf.IfOp(_is_group_b, has_else=True)
                with ir.InsertionPoint(_if_g.then_block):
                    write_row_to_lds(
                        mi=mi,
                        ii=ii,
                        row_in_tile=row_in_tile,
                        row=row,
                        row_base_lds=row_base_lds,
                        col_base_local=col_base_local_b,
                        num_acc_n=num_acc_n,
                        lds_out=lds_out_split,
                    )
                    scf.YieldOp([])
                with ir.InsertionPoint(_if_g.else_block):
                    write_row_to_lds(
                        mi=mi,
                        ii=ii,
                        row_in_tile=row_in_tile,
                        row=row,
                        row_base_lds=row_base_lds,
                        col_base_local=col_base_local_a,
                        num_acc_n=num_acc_n,
                        lds_out=lds_out,
                    )
                    scf.YieldOp([])

            gpu.barrier()
            default_epilog(
                arith=arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=_write_row_split,
            )
            gpu.barrier()

            # -- read phase (each group reads from its own LDS buffer) --
            tx_local = tx - arith.select(_is_group_b, _half_thr_idx, _zero_idx)
            c_nlane_s = arith.constant(CShuffleNLane_s, index=True)
            m_lane_s = tx_local // c_nlane_s
            n_lane_s = tx_local % c_nlane_s
            c_evec = arith.constant(EVec, index=True)

            if frag_elem_type is None:
                frag_elem_type = T.f16
            vec_frag = T.vec(EVec, frag_elem_type)
            bx_m_v = bx_m
            by_n_v = by_n

            _precomputed_rows_s = []
            for mr in range_constexpr(m_reps_s):
                row_base_m = arith.constant(mr * CShuffleMLane_s, index=True)
                row_local = row_base_m + m_lane_s
                row = bx_m_v + row_local
                row_ctx_raw = (
                    precompute_row(row_local=row_local, row=row)
                    if precompute_row is not None
                    else None
                )
                row_ctx = row_ctx_raw
                row_pred = None
                if (
                    scf is not None
                    and row_ctx_raw is not None
                    and isinstance(row_ctx_raw, tuple)
                    and len(row_ctx_raw) == 2
                ):
                    row_ctx, row_pred = row_ctx_raw
                _precomputed_rows_s.append((row_local, row, row_ctx, row_pred))

            for mr in range_constexpr(m_reps_s):
                row_local, row, row_ctx, row_pred = _precomputed_rows_s[mr]

                def _do_store_row_split(row=row, row_ctx=row_ctx, row_local=row_local):
                    row_base_lds = row_local * _half_n_idx
                    for nr in range_constexpr(n_reps_s):
                        col_base_nr = arith.constant(
                            nr * (CShuffleNLane_s * EVec), index=True
                        )
                        col_pair0_local = col_base_nr + (n_lane_s * c_evec)
                        lds_idx = row_base_lds + col_pair0_local

                        _if_ld = scf.IfOp(_is_group_b, [vec_frag], has_else=True)
                        with ir.InsertionPoint(_if_ld.then_block):
                            fb = vector.load_op(vec_frag, lds_out_split, [lds_idx])
                            scf.YieldOp([fb])
                        with ir.InsertionPoint(_if_ld.else_block):
                            fa = vector.load_op(vec_frag, lds_out, [lds_idx])
                            scf.YieldOp([fa])
                        frag = _if_ld.results[0]

                        col_pair0 = col_pair0_local + arith.select(
                            _is_group_b, _half_n_idx, _zero_idx
                        )
                        store_pair(
                            row_local=row_local,
                            row=row,
                            row_ctx=row_ctx,
                            col_pair0=col_pair0,
                            col_g0=by_n_v + col_pair0,
                            frag=frag,
                        )

                if row_pred is not None:
                    _if_row = scf.IfOp(row_pred)
                    with _if_then(_if_row, scf):
                        _do_store_row_split()
                else:
                    _do_store_row_split()

            return  # split path complete

        # ===================== Standard (non-split) path below =====================

        # ---------------- Step 1: write C tile to LDS (row-major, fp16) ----------------
        tile_n_idx = arith.constant(int(tile_n), index=True)
        n_tile_base_v = n_tile_base
        col_base_local = n_tile_base_v + lane_mod_16  # index within [0,tile_n)

        _lds_row_base_offset = (
            lds_row_offset * tile_n_idx if lds_row_offset is not None else None
        )

        def _write_row(mi: int, ii: int, row_in_tile, row):
            row_base_lds = row_in_tile * tile_n_idx
            if _lds_row_base_offset is not None:
                row_base_lds = row_base_lds + _lds_row_base_offset
            write_row_to_lds(
                mi=mi,
                ii=ii,
                row_in_tile=row_in_tile,
                row=row,
                row_base_lds=row_base_lds,
                col_base_local=col_base_local,
                num_acc_n=num_acc_n,
                lds_out=lds_out,
            )

        # Ensure all LDS reads finished before the lds write.
        gpu.barrier()
        default_epilog(
            arith=arith,
            range_constexpr=range_constexpr,
            m_repeat=m_repeat,
            lane_div_16=lane_div_16,
            bx_m=bx_m,
            body_row=_write_row,
        )

        # Ensure all LDS writes are visible before the shuffle-read.
        gpu.barrier()

        # ---------------- Step 2: shuffle mapping + half2 store/atomic ----------------
        CShuffleNLane = int(cshuffle_nlane)
        CShuffleMLane = int(cshuffle_mlane)
        EVec = int(e_vec)

        m_reps_shuffle = int(tile_m) // CShuffleMLane
        n_reps_shuffle = int(tile_n) // (CShuffleNLane * EVec)

        c_nlane = fx.Index(CShuffleNLane)
        m_lane = tx // c_nlane
        n_lane = tx % c_nlane
        c_evec = fx.Index(EVec)

        if frag_elem_type is None:
            frag_elem_type = T.f16
        vec_frag = T.vec(EVec, frag_elem_type)
        bx_m_v = bx_m
        by_n_v = by_n

        # Batch-precompute all row contexts (sorted_idx loads) before the store loop.
        # This issues all buffer_load instructions upfront so the compiler can pipeline
        # them instead of serializing each load with s_waitcnt vmcnt(0).
        _precomputed_rows = []
        for mr in range_constexpr(m_reps_shuffle):
            row_base_m = arith.constant(mr * CShuffleMLane, index=True)
            row_local = row_base_m + m_lane
            row = bx_m_v + row_local

            row_ctx_raw = (
                precompute_row(row_local=row_local, row=row)
                if precompute_row is not None
                else None
            )

            # Optional row-level predicate: if `precompute_row` returns `(ctx, pred_i1)` and `scf`
            # is provided, we can skip the entire N-loop for invalid rows (cheaper than per-store checks).
            row_ctx = row_ctx_raw
            row_pred = None
            if (
                scf is not None
                and row_ctx_raw is not None
                and isinstance(row_ctx_raw, tuple)
                and len(row_ctx_raw) == 2
            ):
                row_ctx, row_pred = row_ctx_raw

            _precomputed_rows.append((row_local, row, row_ctx, row_pred))

        # Now perform LDS reads and stores using the pre-fetched row contexts.
        for mr in range_constexpr(m_reps_shuffle):
            row_local, row, row_ctx, row_pred = _precomputed_rows[mr]

            def _do_store_row(row=row, row_ctx=row_ctx, row_local=row_local):
                row_base_lds = row_local * tile_n_idx
                if _lds_row_base_offset is not None:
                    row_base_lds = row_base_lds + _lds_row_base_offset
                for nr in range_constexpr(n_reps_shuffle):
                    col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
                    col_pair0 = col_base_nr + (n_lane * c_evec)  # even col within tile

                    lds_idx_pair = row_base_lds + col_pair0
                    frag = vector.load_op(vec_frag, lds_out, [lds_idx_pair])

                    store_pair(
                        row_local=row_local,
                        row=row,
                        row_ctx=row_ctx,
                        col_pair0=col_pair0,
                        col_g0=by_n_v + col_pair0,
                        frag=frag,
                    )

            if row_pred is not None:
                _if_row = scf.IfOp(row_pred)
                with _if_then(_if_row, scf):
                    _do_store_row()
            else:
                _do_store_row()


    def mfma_epilog(
        *,
        use_cshuffle: bool,
        # Common (always required)
        arith,
        range_constexpr,
        m_repeat: int,
        lane_div_16,
        bx_m,
        # Default epilog (required when use_cshuffle=False)
        body_row: Callable | None = None,
        # CShuffle epilog (required when use_cshuffle=True)
        vector=None,
        gpu=None,
        scf=None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        e_vec: int = 2,
        cshuffle_nlane: int = 32,
        block_size: int = 256,
        num_acc_n: int | None = None,
        tx=None,
        lane_mod_16=None,
        by_n=None,
        n_tile_base=None,
        lds_out=None,
        write_row_to_lds: Callable | None = None,
        precompute_row: Callable | None = None,
        store_pair: Callable | None = None,
        frag_elem_type: ir.Type | None = None,
    ):
        if not use_cshuffle:
            if body_row is None:
                raise ValueError("mfma_epilog(use_cshuffle=False) requires `body_row`.")
            return default_epilog(
                arith=arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=body_row,
            )

        return c_shuffle_epilog(
            arith=arith,
            vector=vector,
            gpu=gpu,
            scf=scf,
            range_constexpr=range_constexpr,
            tile_m=int(tile_m),
            tile_n=int(tile_n),
            e_vec=int(e_vec),
            cshuffle_nlane=int(cshuffle_nlane),
            block_size=int(block_size),
            m_repeat=m_repeat,
            num_acc_n=int(num_acc_n),
            tx=tx,
            lane_div_16=lane_div_16,
            lane_mod_16=lane_mod_16,
            bx_m=bx_m,
            by_n=by_n,
            n_tile_base=n_tile_base,
            lds_out=lds_out,
            frag_elem_type=frag_elem_type,
            write_row_to_lds=write_row_to_lds,
            precompute_row=precompute_row,
            store_pair=store_pair,
        )
    return _types.SimpleNamespace(**dict(locals()))
_M_mfma_epilogues = _mod_mfma_epilogues()

def _mod_mfma_preshuffle_pipeline():
    # SPDX-License-Identifier: Apache-2.0
    # Copyright (c) 2025 FlyDSL Project Contributors

    """Shared MFMA preshuffle helpers for preshuffle GEMM kernels.

    Key primitives:
    - B preshuffle layout builder (supports byte-packed element types, incl. packed int4)
    - B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)
    """



    from flydsl.expr.typing import T


    def crd2idx(crd, layout):
        """crd2idx returning an index-typed ir.Value (unwraps fly.int_tuple)."""
        scalar = fx.get_scalar(fx.crd2idx(crd, layout)).ir_value()
        if isinstance(scalar.type, ir.IndexType):
            return scalar
        return _arith.IndexCastOp(T.index, scalar).result


    def swizzle_xor16(row, col, k_blocks16):
        """XOR-with-row swizzle on the K dimension at 16B granularity.

        Computes: col XOR ((row & (k_blocks16 - 1)) * 16)

        k_blocks16 is always a power of 2 (tile_k_bytes / 16), so use
        bitwise AND instead of remui to save ~10 VALU cycles on CDNA.
        """
        from flydsl.expr import arith as _swz_arith

        mask = k_blocks16 - _swz_arith.index(1)
        rem = _swz_arith.andi(row, mask)
        return col ^ (rem * 16)


    def lds_row_major_idx(row, col, row_stride, base=None):
        """Linearize a 2D LDS coordinate with explicit index arithmetic."""
        idx = row * row_stride + col
        return idx if base is None else idx + base


    def split_row_major_2d(index, minor_extent):
        """Split a linear row-major index into (major, minor)."""
        return index // minor_extent, index % minor_extent


    def _buffer_load_vec(
        buffer_ops,
        vector,
        rsrc,
        idx,
        *,
        elem_type,
        vec_elems,
        elem_bytes,
        offset_in_bytes,
        cache_modifier=0,
    ):
        """Load vec_elems elements via buffer_load dwordx[1,2,4] + bitcast."""
        from flydsl.expr import arith as _ld_arith

        elem_size = int(elem_bytes)
        load_bytes = int(vec_elems) * elem_size
        vec_width = load_bytes // 4

        if offset_in_bytes:
            idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(2))
        elif elem_bytes == 2:
            idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(1))
        else:
            idx_i32 = idx

        i32_val = buffer_ops.buffer_load(
            rsrc,
            idx_i32,
            vec_width=vec_width,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
        if vec_width == 1:
            i32_vec = vector.from_elements(T.vec(1, T.i32), [i32_val])
        else:
            i32_vec = i32_val
        return vector.bitcast(T.vec(int(vec_elems), elem_type), i32_vec)


    @dataclass(frozen=True)
    class PreshuffleScaleLayout:
        """Container returned by `make_preshuffle_scale_layout`.

        The scale layout is ``(c_mn1, c_k1, 4, 16) : (stride_n0, stride_k0, stride_klane, 1)``.
        Callers compute flat index directly with plain arith::

            idx = mni * stride_n0 + ku * stride_k0 + k_lane * stride_klane + n_lane
        """

        layout_scale: object
        stride_n0: object
        stride_k0: object
        stride_klane: object


    def make_preshuffle_scale_layout(
        arith,
        *,
        c_mn: ir.Value,
        c_k: ir.Value,
        mn_pack: int = 2,
        k_pack: int = 2,
        elem_bytes: int = 4,
        scale_block_size: int = 32,
    ) -> PreshuffleScaleLayout:
        """Build scale layout matching aiter/CK preshuffle for FP4/FP8 microscale.

        Layout shape: ``(c_mn1, c_k1, 4, 16)`` where
        ``c_mn1 = c_mn / 16 / mn_pack`` and ``c_k1 = (c_k / scale_block_size) / 4 / k_pack``.
        """
        c16 = fx.Index(16)
        c4 = fx.Index(4)
        c_k_scale = c_k // fx.Index(scale_block_size)

        c_mn1 = (c_mn // c16) // fx.Index(mn_pack)
        c_k1 = (c_k_scale // c4) // fx.Index(k_pack)
        if elem_bytes != mn_pack * k_pack:
            raise ValueError(
                f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}"
            )

        stride_klane = c16
        stride_k0 = c4 * stride_klane
        stride_n0 = c_k1 * stride_k0

        c_mn1_i32 = arith.index_cast(T.i32, c_mn1)
        c_k1_i32 = arith.index_cast(T.i32, c_k1)
        stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
        stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
        stride_klane_i32 = arith.index_cast(T.i32, stride_klane)

        layout_scale = fx.make_layout(
            (c_mn1_i32, c_k1_i32, 4, 16),
            stride=(stride_n0_i32, stride_k0_i32, stride_klane_i32, 1),
        )

        return PreshuffleScaleLayout(
            layout_scale=layout_scale,
            stride_n0=stride_n0,
            stride_k0=stride_k0,
            stride_klane=stride_klane,
        )


    @dataclass(frozen=True)
    class PreshuffleBLayout:
        """Container returned by `make_preshuffle_b_layout`."""

        layout_b: object
        kpack_bytes: int


    def make_preshuffle_b_layout(
        arith,
        *,
        c_n: ir.Value,
        c_k: ir.Value,
        kpack_bytes: int = 16,
        elem_bytes: int = 1,
        k_major: bool = False,
    ) -> PreshuffleBLayout:
        """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels.

        When *k_major* is True the block-level order is K-major (``k_blk`` outermost),
        matching the ``(0,3,1,4,2,5)`` shuffle permutation.  The default N-major
        order (``k_major=False``) matches the legacy ``(0,1,3,4,2,5)`` permutation.
        """
        if kpack_bytes not in (8, 16):
            raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

        c16 = fx.Index(16)
        c_kpack = fx.Index(kpack_bytes)

        if elem_bytes not in (1, 2):
            raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
        c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
        n0 = c_n // c16

        c_kpack_elems = (
            c_kpack
            if elem_bytes == 1
            else (c_kpack // arith.constant(int(elem_bytes), index=True))
        )

        stride_nlane = c_kpack_elems

        if k_major:
            c32 = fx.Index(32)
            c2 = fx.Index(2)
            c_k0 = c_k_bytes // c32
            klane_dim = 2
            stride_klane = c16 * stride_nlane
            stride_n0 = c2 * stride_klane
            stride_k0 = n0 * stride_n0
        else:
            c64 = fx.Index(64)
            c4 = fx.Index(4)
            c_k0 = c_k_bytes // c64
            klane_dim = 4
            stride_klane = c16 * stride_nlane
            stride_k0 = c4 * stride_klane
            stride_n0 = c_k0 * stride_k0

        kpack_elems_static = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
        n0_i32 = arith.index_cast(T.i32, n0)
        c_k0_i32 = arith.index_cast(T.i32, c_k0)
        stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
        stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
        stride_klane_i32 = arith.index_cast(T.i32, stride_klane)
        stride_nlane_i32 = arith.index_cast(T.i32, stride_nlane)

        stride_b = (stride_n0_i32, stride_k0_i32, stride_klane_i32, stride_nlane_i32, 1)
        layout_b = fx.make_layout(
            (n0_i32, c_k0_i32, klane_dim, 16, kpack_elems_static), stride_b
        )
        return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


    def _unpack_int4_to_int8_pair(packed32):
        """Split packed int4 dword into two int8 dwords (even/odd nibbles).

        7-op bit manipulation shared by all int4 unpack paths (W4A8, W4A16, W4A_FP8).
        """
        c_08 = fx.Int32(0x08080808)
        c_0f = fx.Int32(0x0F0F0F0F)
        c_1e = fx.Int32(0x1E)
        c_4 = fx.Int32(4)
        s0 = (packed32 & c_08) * c_1e
        even = (packed32 & c_0f) | s0
        t = packed32 >> c_4
        s1 = (t & c_08) * c_1e
        odd = (t & c_0f) | s1
        return even, odd


    def _pack_i32_pair_to_i64(lo, hi, vector):
        """Pack two i32 values into one i64 via vector bitcast."""
        v2 = vector.from_elements(T.vec(2, T.i32), [lo, hi])
        v64 = vector.bitcast(T.vec(1, T.i64), v2)
        return vector.extract(v64, static_position=[0], dynamic_position=[])


    def _i8x4_in_i32_to_bf16x4_i64(val_i32, arith, vector, scale_val=None):
        """Convert one i32 (4 signed int8 bytes) to 4 bf16 packed as i64.

        Uses shift-based f32->bf16 truncation (lshr 16) instead of arith.truncf
        which on gfx942 expands to ~5 VALU per element. The shift is exact for
        unscaled int8 values and introduces <0.5 ULP error for scaled values.
        """
        v1 = vector.from_elements(T.vec(1, T.i32), [val_i32])
        i8x4 = vector.bitcast(T.i8x4, v1)

        f32_vals = []
        for i in range(4):
            val_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
            v = arith.sitofp(T.f32, val_i8)
            if scale_val is not None:
                v = v * scale_val
            f32_vals.append(v)

        c16 = fx.Int32(16)
        c_ffff0000 = fx.Int32(0xFFFF0000)
        bits = [arith.bitcast(T.i32, f) for f in f32_vals]
        i32_lo = (bits[0] >> c16) | (bits[1] & c_ffff0000)
        i32_hi = (bits[2] >> c16) | (bits[3] & c_ffff0000)
        return _pack_i32_pair_to_i64(i32_lo, i32_hi, vector)


    def load_b_raw_w4a16(
        buffer_ops,
        arith,
        vector,
        *,
        arg_b,
        b_rsrc,
        layout_b,
        base_k: ir.Value,
        ku: int,
        n_blk: ir.Value,
        n_intra: ir.Value,
        lane_div_16: ir.Value,
        elem_type: ir.Type,
        kpack_bytes: int = 8,
    ):
        """Phase 1 of W4A16 B load: issue buffer_load_dword, return raw packed i32.

        Same address calculation as the int4 unpack path in load_b_pack_k32
        but using ku-based indexing for 2-phase latency hiding.
        """
        if kpack_bytes != 8:
            raise ValueError(f"W4A16 requires kpack_bytes=8, got {kpack_bytes!r}")

        c64 = fx.Index(64)
        half_bytes = kpack_bytes // 2
        c2_idx = fx.Index(2)
        c4_idx = fx.Index(4)

        k0_base = base_k // c64

        k1_layout_offset = ku * 2
        lane_div_32 = lane_div_16 // c2_idx
        total_k1 = fx.Index(k1_layout_offset) + lane_div_32
        k0 = k0_base + (total_k1 // c4_idx)
        k1_local = total_k1 % c4_idx
        lane_odd = lane_div_16 % c2_idx
        k2_base = lane_odd * fx.Index(half_bytes)

        coord_pack = (n_blk, k0, k1_local, n_intra, fx.Index(0))
        idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)
        idx_bytes = idx_pack + k2_base

        b4 = _buffer_load_vec(
            buffer_ops,
            vector,
            b_rsrc,
            idx_bytes,
            elem_type=elem_type,
            vec_elems=4,
            elem_bytes=1,
            offset_in_bytes=True,
        )
        packed32 = vector.extract(
            vector.bitcast(T.vec(1, T.i32), b4),
            static_position=[0],
            dynamic_position=[],
        )
        return packed32


    def _int4_to_bf16x4_i64_gfx950(
        packed32, nibble_offsets, arith, vector, scale_val=None, defer_scale16=False
    ):
        """Convert 4 int4 nibbles to 4 bf16 packed as i64 using gfx950 instructions.

        Uses v_cvt_off_f32_i4_sdwa with byte_sel to avoid per-nibble shifts.
        Even nibbles (0,2,4,6) → SDWA BYTE_0/1/2/3 on original src.
        Odd nibbles (1,3,5,7)  → SDWA BYTE_0/1/2/3 on (src >> 4).
        Only 1 shift total instead of 7.

        When defer_scale16=True, the ×16 correction factor for v_cvt_off_f32_i4 is
        omitted and must be applied later (e.g. in the epilogue).  This saves VALU
        in the hot loop and uses v_cvt_pk_bf16_f32 for proper f32→bf16 conversion.
        """
        from flydsl._mlir.dialects._arith_ops_gen import MulFOp as _MulFOp
        from flydsl.expr import rocdl

        _uw = _arith._to_raw
        _av = _arith.ArithValue

        src_even = packed32
        src_odd = packed32 >> fx.Int32(4)

        f32_vals = []
        for nib in nibble_offsets:
            byte_idx = nib // 2
            src = src_odd if (nib % 2) else src_even
            v = rocdl.cvt_off_f32_i4(src, byte_sel=byte_idx)
            f32_vals.append(v)

        if defer_scale16:
            # Skip ×16; multiply by scale_val only if groupwise.
            if scale_val is not None:
                raw_scale = _uw(scale_val)
                f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
            # Use v_cvt_pk_bf16_f32 for proper f32→bf16 (no bit-shift trick needed).
            i32_lo = rocdl.cvt_pk_bf16_f32(f32_vals[0], f32_vals[1])
            i32_hi = rocdl.cvt_pk_bf16_f32(f32_vals[2], f32_vals[3])
        else:
            c16 = fx.Float32(16.0)
            effective_scale = scale_val * c16 if scale_val is not None else c16
            raw_scale = _uw(effective_scale)
            f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
            # Truncate f32→bf16 via bit-shift (exact for scaled int values).
            c16_shift = fx.Int32(16)
            c_ffff0000 = fx.Int32(0xFFFF0000)
            bf16_vals = [arith.bitcast(T.i32, _av(v)) for v in f32_vals]
            i32_lo = (bf16_vals[0] >> c16_shift) | (bf16_vals[1] & c_ffff0000)
            i32_hi = (bf16_vals[2] >> c16_shift) | (bf16_vals[3] & c_ffff0000)

        return _pack_i32_pair_to_i64(i32_lo, i32_hi, vector)


    def unpack_b_w4a16(
        packed32, arith, vector, scale_val=None, use_gfx950_cvt=False, defer_scale16=False
    ):
        """Phase 2 of W4A16 B load: unpack int4->int8 + convert int8->bf16.

        Takes raw packed32 from load_b_raw_w4a16 and produces (b0, b1) --
        two i64 values each containing 4 bf16 for one MFMA.

        When use_gfx950_cvt=True, uses v_cvt_off_f32_i4 + v_cvt_pk_bf16_f32
        for ~2x fewer VALU instructions.

        When defer_scale16=True (requires use_gfx950_cvt=True), the ×16
        correction for v_cvt_off_f32_i4 is omitted; caller must apply it
        in the epilogue.
        """
        if use_gfx950_cvt:
            b0 = _int4_to_bf16x4_i64_gfx950(
                packed32,
                [0, 2, 4, 6],
                arith,
                vector,
                scale_val,
                defer_scale16=defer_scale16,
            )
            b1 = _int4_to_bf16x4_i64_gfx950(
                packed32,
                [1, 3, 5, 7],
                arith,
                vector,
                scale_val,
                defer_scale16=defer_scale16,
            )
            return (b0, b1)
        even, odd = _unpack_int4_to_int8_pair(packed32)
        b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
        b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)
        return (b0, b1)


    def load_b_pack_k32(
        buffer_ops,
        arith,
        vector,
        *,
        arg_b,
        b_rsrc,
        layout_b,
        base_k: ir.Value,
        ki_step: int,
        n_blk: ir.Value,
        n_intra: ir.Value,
        lane_div_16: ir.Value,
        elem_type: ir.Type,
        kpack_bytes: int = 16,
        elem_bytes: int = 1,
        unpack_int4: bool = False,
    ) -> ir.Value:
        """Load one B pack for one MFMA(x32) micro-step.

        Returns an i64 Value containing 8 bytes consumed by MFMA.
        """
        if kpack_bytes not in (8, 16):
            raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")
        if unpack_int4 and kpack_bytes != 8:
            raise ValueError("unpack_int4 requires kpack_bytes=8 (packed int4 layout)")
        if elem_bytes not in (1, 2):
            raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")

        c64 = fx.Index(64)
        base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
        k0_base = base_k_bytes // c64
        k0 = k0_base + arith.constant(ki_step // 2, index=True)
        k1 = lane_div_16
        half_bytes = kpack_bytes // 2
        k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)

        coord_pack = (n_blk, k0, k1, n_intra, fx.Index(0))
        idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)

        if unpack_int4:
            idx_bytes = idx_pack + k2_base
            b4 = _buffer_load_vec(
                buffer_ops,
                vector,
                b_rsrc,
                idx_bytes,
                elem_type=elem_type,
                vec_elems=4,
                elem_bytes=1,
                offset_in_bytes=True,
            )
            packed32 = vector.extract(
                vector.bitcast(T.vec(1, T.i32), b4),
                static_position=[0],
                dynamic_position=[],
            )
            even, odd = _unpack_int4_to_int8_pair(packed32)
            return _pack_i32_pair_to_i64(even, odd, vector)

        vec_elems = kpack_bytes // int(elem_bytes)
        b16 = _buffer_load_vec(
            buffer_ops,
            vector,
            b_rsrc,
            idx_pack,
            elem_type=elem_type,
            vec_elems=vec_elems,
            elem_bytes=elem_bytes,
            offset_in_bytes=(elem_bytes == 1),
        )

        b_i32x4 = vector.bitcast(T.i32x4, b16)

        base = (ki_step % 2) * 2
        d0 = vector.extract(b_i32x4, static_position=[base], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[base + 1], dynamic_position=[])
        return _pack_i32_pair_to_i64(d0, d1, vector)


    def tile_chunk_coord_i32(
        arith,
        *,
        tx_i32_base: ir.Value,
        i: int,
        total_threads: int,
        layout_tile_div4,
        chunk_i32: int = 4,
    ):
        """Map (thread, chunk_id) -> (row_local, col_local_i32) for X/A loads."""
        if chunk_i32 not in (1, 2, 4):
            raise ValueError(f"chunk_i32 must be one of (1,2,4), got {chunk_i32!r}")
        chunk_off_i32 = arith.constant(i * total_threads * chunk_i32, index=True)
        tile_idx_i32 = tx_i32_base + chunk_off_i32
        coord_local = fx.idx2crd(fx.Int32(tile_idx_i32), layout_tile_div4)
        row_local = fx.get(coord_local, 0)
        col_local_i32 = fx.get(coord_local, 1)
        return row_local, col_local_i32


    def buffer_copy_gmem16_dwordx4(
        buffer_ops,
        vector,
        *,
        elem_type,
        idx_i32: ir.Value,
        rsrc,
        vec_elems: int = 16,
        elem_bytes: int = 1,
    ):
        """Copy 16 bytes from global memory into regs via buffer-load dwordx4 lowering."""
        if int(vec_elems) <= 0:
            raise ValueError(f"vec_elems must be > 0, got {vec_elems!r}")
        return _buffer_load_vec(
            buffer_ops,
            vector,
            rsrc,
            idx_i32,
            elem_type=elem_type,
            vec_elems=vec_elems,
            elem_bytes=elem_bytes,
            offset_in_bytes=False,
        )


    def _lds_store_xor16(
        vector,
        *,
        lds_memref,
        vec_ty,
        layout_lds,
        row_local: ir.Value,
        col_local_i32: ir.Value,
        tx_c4: ir.Value,
        k_blocks16: ir.Value,
        lds_base: ir.Value,
        vec_part: ir.Value,
        elem_bytes: int,
    ):
        """Store one chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
        if elem_bytes not in (1, 2):
            raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
        col_swz_bytes = swizzle_xor16(row_local, col_local_i32 * tx_c4, k_blocks16)
        col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
        idx0 = crd2idx((fx.Int32(row_local), fx.Int32(col_swz)), layout_lds) + lds_base
        vector.store(vector.bitcast(vec_ty, vec_part), lds_memref, [idx0])


    def lds_store_16b_xor16(
        arith,
        vector,
        *,
        lds_memref,
        vec16_ty,
        layout_lds,
        row_local: ir.Value,
        col_local_i32: ir.Value,
        tx_c4: ir.Value,
        k_blocks16: ir.Value,
        lds_base: ir.Value,
        vec_part_i32x4: ir.Value,
        elem_bytes: int = 1,
    ):
        """Store one 16B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
        _lds_store_xor16(
            vector,
            lds_memref=lds_memref,
            vec_ty=vec16_ty,
            layout_lds=layout_lds,
            row_local=row_local,
            col_local_i32=col_local_i32,
            tx_c4=tx_c4,
            k_blocks16=k_blocks16,
            lds_base=lds_base,
            vec_part=vec_part_i32x4,
            elem_bytes=elem_bytes,
        )


    def lds_store_8b_xor16(
        arith,
        vector,
        *,
        lds_memref,
        vec8_ty,
        layout_lds,
        row_local: ir.Value,
        col_local_i32: ir.Value,
        tx_c4: ir.Value,
        k_blocks16: ir.Value,
        lds_base: ir.Value,
        vec_part_i32x2: ir.Value,
        elem_bytes: int = 1,
    ):
        """Store one 8B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
        _lds_store_xor16(
            vector,
            lds_memref=lds_memref,
            vec_ty=vec8_ty,
            layout_lds=layout_lds,
            row_local=row_local,
            col_local_i32=col_local_i32,
            tx_c4=tx_c4,
            k_blocks16=k_blocks16,
            lds_base=lds_base,
            vec_part=vec_part_i32x2,
            elem_bytes=elem_bytes,
        )


    def lds_store_4b_xor16(
        arith,
        vector,
        *,
        lds_memref,
        vec4_ty,
        layout_lds,
        row_local: ir.Value,
        col_local_i32: ir.Value,
        tx_c4: ir.Value,
        k_blocks16: ir.Value,
        lds_base: ir.Value,
        vec_part_i32x1: ir.Value,
        elem_bytes: int = 1,
    ):
        """Store one 4B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
        _lds_store_xor16(
            vector,
            lds_memref=lds_memref,
            vec_ty=vec4_ty,
            layout_lds=layout_lds,
            row_local=row_local,
            col_local_i32=col_local_i32,
            tx_c4=tx_c4,
            k_blocks16=k_blocks16,
            lds_base=lds_base,
            vec_part=vec_part_i32x1,
            elem_bytes=elem_bytes,
        )


    def lds_load_pack_k32(
        arith,
        vector,
        *,
        lds_memref,
        layout_lds,
        k_blocks16: ir.Value,
        curr_row_a_lds: ir.Value,
        col_base: ir.Value,
        half: int,
        lds_base: ir.Value,
        ck_lds128: bool,
        vec16_ty,
        vec8_ty,
        vec2_i64_ty,
        vec1_i64_ty,
    ):
        """Load one i64 A-pack for an MFMA K32 micro-step from LDS."""
        col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
        if ck_lds128:
            coord_a16 = (curr_row_a_lds, col_base_swz)
            idx_a16 = crd2idx(tuple(fx.Int32(c) for c in coord_a16), layout_lds) + lds_base
            loaded_a16 = vector.load_op(vec16_ty, lds_memref, [idx_a16])
            a_vec128 = vector.bitcast(vec2_i64_ty, loaded_a16)
            return vector.extract(a_vec128, static_position=[half], dynamic_position=[])
        else:
            col_swizzled = col_base_swz + (half * 8)
            coord_a = (curr_row_a_lds, col_swizzled)
            idx_a = crd2idx(tuple(fx.Int32(c) for c in coord_a), layout_lds) + lds_base
            loaded_a8 = vector.load_op(vec8_ty, lds_memref, [idx_a])
            a_vec64 = vector.bitcast(vec1_i64_ty, loaded_a8)
            return vector.extract(a_vec64, static_position=[0], dynamic_position=[])


    def xcd_remap_bx_by(
        bx,
        by,
        c_m,
        *,
        tile_m: int,
        tile_n: int,
        N: int,
        xcd_swizzle: int,
        num_xcds: int = 8,
    ):
        """Remap (bx, by) for L2-cache reuse via XCD swizzle.

        No-op when ``xcd_swizzle <= 0``. Otherwise:
          1. Linearize the original (bx, by) grid round-robin across ``num_xcds``
             XCDs so that contiguous workgroup ids stay on the same XCD.
          2. Re-tile that 1-D order with an M-major group of size ``xcd_swizzle``,
             folding the tail group when ``gy`` does not divide evenly.

        Designed to be called inside a ``@flyc.kernel`` immediately after::

            bx = gpu.block_id("x")
            by = gpu.block_id("y")
            bx, by = xcd_remap_bx_by(bx, by, c_m, tile_m=..., tile_n=..., N=...,
                                     xcd_swizzle=xcd_swizzle)

        ``c_m`` is the dynamic ``fx.Index`` for runtime ``M``; ``tile_m``,
        ``tile_n``, ``N`` and ``xcd_swizzle`` are compile-time Python ints.
        """
        if xcd_swizzle <= 0:
            return bx, by

        _c1 = fx.arith.constant(1, index=True)
        _c_tm = fx.arith.constant(tile_m, index=True)
        _gx = fx.arith.constant(N // tile_n, index=True)
        _gy = (c_m + _c_tm - _c1) // _c_tm

        _linear_id = bx * _gx + by
        _num_wgs = _gx * _gy

        _c_xcds = fx.arith.constant(num_xcds, index=True)
        _q = _num_wgs // _c_xcds
        _r = _num_wgs % _c_xcds
        _xcd = _linear_id % _c_xcds
        _in_xcd = _linear_id // _c_xcds
        _xcd_lt_r = fx.arith.cmpi(CmpIPredicate.ult, _xcd, _r)
        _clip = fx.arith.select(_xcd_lt_r, _xcd, _r)
        _wgid = _xcd * _q + _clip + _in_xcd

        _c_wgm = fx.arith.constant(xcd_swizzle, index=True)
        _num_wgid_in_group = _c_wgm * _gx
        _group_id = _wgid // _num_wgid_in_group
        _first_pid_m = _group_id * _c_wgm
        _remaining_m = _gy - _first_pid_m
        _cmp_m = fx.arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
        _group_size_m = fx.arith.select(_cmp_m, _remaining_m, _c_wgm)

        _wgid_in_group = _wgid % _num_wgid_in_group
        new_bx = _first_pid_m + (_wgid_in_group % _group_size_m)
        new_by = _wgid_in_group // _group_size_m
        return new_bx, new_by


    __all__ = [
        "PreshuffleBLayout",
        "PreshuffleScaleLayout",
        "buffer_copy_gmem16_dwordx4",
        "extract_bf16_scale",
        "lds_load_pack_k32",
        "lds_row_major_idx",
        "lds_store_4b_xor16",
        "lds_store_8b_xor16",
        "lds_store_16b_xor16",
        "load_b_pack_k32",
        "load_b_raw_mxfp4_dwordx4",
        "load_b_raw_w4a16",
        "load_b_raw_w4a16_groupwise",
        "make_preshuffle_b_layout",
        "make_preshuffle_scale_layout",
        "split_row_major_2d",
        "swizzle_xor16",
        "tile_chunk_coord_i32",
        "unpack_b_mxfp4_bf16",
        "unpack_b_w4a16",
        "unpack_b_w4a16_groupwise",
        "xcd_remap_bx_by",
    ]


    # ---------------------------------------------------------------------------
    # Groupwise scale load helper (shared by W4A16 and W4A8 groupwise paths)
    # ---------------------------------------------------------------------------


    def _load_groupwise_scale(
        buffer_ops,
        arith,
        *,
        scale_rsrc,
        expert_offset,
        n_blk,
        n_intra,
        k_pos,
        num_groups: int,
        group_size: int,
        n_per_expert: int,
        scale_dtype=None,
    ):
        """Load one per-group scale value from the scale buffer.

        Computes the linear index into the scale tensor from expert offset,
        N position, and group index derived from ``k_pos``.

        For bf16 scales the tensor uses ``(E, G//2, N, 2)`` layout — two
        adjacent groups for the same N position are packed into one dword.
        We load the raw i32 dword (no extraction) so it can be carried as
        loop state without register copies.  Use :func:`extract_bf16_scale`
        in the compute phase to obtain the f32 value.
        """
        c16 = fx.Index(16)
        n_global = n_blk * c16 + n_intra
        c_group_size = fx.Index(group_size)
        c_npe = fx.Index(n_per_expert)
        group_idx = k_pos // c_group_size
        if scale_dtype is None:
            scale_dtype = T.f32

        if scale_dtype == T.bf16:
            # (E, G//2, N, 2) layout: same flat formula but with G//2 pairs of groups.
            pair_idx = group_idx >> fx.Index(1)  # group_idx // 2
            num_pairs = num_groups // 2
            c_npm1 = fx.Index(num_pairs - 1)
            dword_base = expert_offset * c_npm1 + n_global
            dword_elem = dword_base + pair_idx * c_npe
            dword_idx = arith.index_cast(T.i32, dword_elem)
            scale_val = buffer_ops.buffer_load(
                scale_rsrc, dword_idx, vec_width=1, dtype=T.i32
            )
        else:
            # (E, G, N) layout with f32 dtype
            c_gm1 = fx.Index(num_groups - 1)
            base_scale = expert_offset * c_gm1 + n_global
            elem_idx = base_scale + group_idx * c_npe
            scale_idx_i32 = arith.index_cast(T.i32, elem_idx)
            scale_val = buffer_ops.buffer_load(
                scale_rsrc, scale_idx_i32, vec_width=1, dtype=T.f32
            )
        return scale_val


    def extract_bf16_scale(arith, scale_raw_i32, ku: int):
        """Extract f32 scale from raw i32 dword loaded by bf16 groupwise path.

        In the ``(E, G//2, N, 2)`` layout two adjacent groups share one dword.
        ``ku`` determines which half: even ku → low bf16, odd ku → high bf16.
        """
        if ku % 2 == 0:
            # Low bf16: shift left by 16 to place in upper 16 bits → f32
            return arith.bitcast(T.f32, scale_raw_i32 << fx.Int32(16))
        else:
            # High bf16: mask upper 16 bits → f32
            return arith.bitcast(T.f32, scale_raw_i32 & fx.Int32(0xFFFF0000))


    # ---------------------------------------------------------------------------
    # W4A16 groupwise load / unpack helpers
    # ---------------------------------------------------------------------------


    def load_b_raw_w4a16_groupwise(
        buffer_ops,
        arith,
        vector,
        *,
        arg_b,
        b_rsrc,
        layout_b,
        base_k,
        ku: int,
        n_blk,
        n_intra,
        lane_div_16,
        elem_type,
        scale_rsrc,
        expert_offset,
        num_groups: int,
        group_size: int,
        n_per_expert: int,
        kpack_bytes: int = 8,
        scale_dtype=None,
    ):
        """Phase 1 of W4A16 groupwise B load: buffer_loads for weight + scale.

        Reuses :func:`load_b_raw_w4a16` for the weight load, then issues an
        additional ``buffer_load_dword`` for the per-group scale.

        Returns ``(packed32, scale_val)``.
        """
        packed32 = load_b_raw_w4a16(
            buffer_ops,
            arith,
            vector,
            arg_b=arg_b,
            b_rsrc=b_rsrc,
            layout_b=layout_b,
            base_k=base_k,
            ku=ku,
            n_blk=n_blk,
            n_intra=n_intra,
            lane_div_16=lane_div_16,
            elem_type=elem_type,
            kpack_bytes=kpack_bytes,
        )
        k_pos = base_k + fx.Index(ku * 32)
        scale_val = _load_groupwise_scale(
            buffer_ops,
            arith,
            scale_rsrc=scale_rsrc,
            expert_offset=expert_offset,
            n_blk=n_blk,
            n_intra=n_intra,
            k_pos=k_pos,
            num_groups=num_groups,
            group_size=group_size,
            n_per_expert=n_per_expert,
            scale_dtype=scale_dtype,
        )
        return (packed32, scale_val)


    def unpack_b_w4a16_groupwise(packed32, scale_val, arith, vector, use_gfx950_cvt=False):
        """Phase 2 of W4A16 groupwise: unpack + scale + convert to bf16."""
        return unpack_b_w4a16(
            packed32, arith, vector, scale_val=scale_val, use_gfx950_cvt=use_gfx950_cvt
        )


    def _cvt_scalef32_pk_bf16_fp4(packed_i32, scale_f32, byte_idx, arith, vector):
        """GFX950 hardware: v_cvt_scalef32_pk_bf16_fp4.

        Converts 2 FP4 E2M1 nibbles (from *byte_idx* of *packed_i32*) to
        2 bf16 values (already scaled by *scale_f32*), returned as i32
        (2 packed bf16).

        One instruction replaces ~36 VALU of the software path.
        """
        from flydsl._mlir.dialects import llvm

        byte_idx_i32 = arith.constant(byte_idx, type=T.i32)
        result_v2bf16 = llvm.call_intrinsic(
            T.vec(2, T.bf16),
            "llvm.amdgcn.cvt.scalef32.pk.bf16.fp4",
            [packed_i32, scale_f32, byte_idx_i32],
            [],
            [],
        )
        vec1_i32_t = T.vec(1, T.i32)
        return vector.extract(
            vector.bitcast(vec1_i32_t, result_v2bf16),
            static_position=[0],
            dynamic_position=[],
        )


    def _fp4x4_in_i32_to_bf16x4_i64(packed4, arith, vector, scale_f32=None):
        """Convert 4 FP4 E2M1 nibbles (in 4 bytes of i32) to 4 bf16 packed as i64.

        Each byte of *packed4* holds one nibble in bits [3:0]:
          bit[3] = sign, bits[2:1] = exponent (bias=1), bit[0] = mantissa.

        Unsigned value table (3-bit index):
          000->0.0, 001->0.5, 010->1.0, 011->1.5,
          100->2.0, 101->3.0, 110->4.0, 111->6.0

        *scale_f32*, when provided, is an f32 E8M0 block-scale multiplied
        into every element before truncation to bf16.
        """
        vec1_i32_t = T.vec(1, T.i32)
        vec2_i32 = T.i32x2
        vec4_i8 = T.i8x4
        vec1_i64 = T.vec(1, T.i64)

        v1 = vector.from_elements(vec1_i32_t, [packed4])
        i8x4 = vector.bitcast(vec4_i8, v1)

        c1 = arith.constant(1, type=T.i32)
        c3_shift = arith.constant(3, type=T.i32)
        c7 = arith.constant(7, type=T.i32)
        c22 = arith.constant(22, type=T.i32)
        c23 = arith.constant(23, type=T.i32)
        c31 = arith.constant(31, type=T.i32)
        c126 = arith.constant(126, type=T.i32)
        c_zero = arith.constant(0, type=T.i32)
        c_half_bits = arith.constant(0x3F000000, type=T.i32)  # 0.5f

        f32_vals = []
        for i in range(4):
            nibble_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
            n = arith.extui(T.i32, nibble_i8)

            sign_bit = arith.andi(arith.shrui(n, c3_shift), c1)
            unsigned_val = arith.andi(n, c7)
            exp_field = arith.shrui(unsigned_val, c1)
            mant_field = arith.andi(unsigned_val, c1)

            f32_norm = arith.ori(
                arith.shli(arith.addi(exp_field, c126), c23),
                arith.shli(mant_field, c22),
            )

            is_zero = arith.cmpi(arith.CmpIPredicate.eq, unsigned_val, c_zero)
            is_subnorm = arith.cmpi(arith.CmpIPredicate.eq, unsigned_val, c1)

            f32_bits = arith.select(
                is_zero,
                c_zero,
                arith.select(is_subnorm, c_half_bits, f32_norm),
            )
            f32_bits = arith.ori(f32_bits, arith.shli(sign_bit, c31))

            v = arith.bitcast(T.f32, f32_bits)
            if scale_f32 is not None:
                v = v * scale_f32
            f32_vals.append(v)

        c16 = arith.constant(16, type=T.i32)
        c_ffff0000 = arith.constant(0xFFFF0000, type=T.i32)
        bits0 = arith.bitcast(T.i32, f32_vals[0])
        bits1 = arith.bitcast(T.i32, f32_vals[1])
        bits2 = arith.bitcast(T.i32, f32_vals[2])
        bits3 = arith.bitcast(T.i32, f32_vals[3])
        i32_lo = arith.shrui(bits0, c16) | (bits1 & c_ffff0000)
        i32_hi = arith.shrui(bits2, c16) | (bits3 & c_ffff0000)

        v2 = vector.from_elements(vec2_i32, [i32_lo, i32_hi])
        v64 = vector.bitcast(vec1_i64, v2)
        return vector.extract(v64, static_position=[0], dynamic_position=[])


    def load_b_raw_mxfp4(
        buffer_ops,
        arith,
        vector,
        *,
        arg_b,
        b_rsrc,
        layout_b,
        base_k: ir.Value,
        ku: int,
        n_blk: ir.Value,
        n_intra: ir.Value,
        lane_div_16: ir.Value,
        elem_type: ir.Type,
        kpack_bytes: int = 16,
    ):
        """Load 4 bytes of packed FP4 from a kpack=16 preshuffle layout.

        Addressing for kpack=16 (``shuffle_weight_a16w4`` format):
          - Layout shape: ``(n0, k0, klane=4, nlane=16, kpack=16)``
          - The A-side LDS has klane stride = 8 bf16 elements, advancing
            by 32 bf16 per ku step.  B must match: each klane loads 4 bytes
            (8 FP4 = 8 K elements) at K_start = base_k + ku*32 + lane*8.
          - In the preshuffle layout this maps to:
              k0 = base_k//128 + ku//4
              klane_hw = ku % 4          (compile-time)
              kpack_byte = lane_div_16*4  (runtime)

        Returns a single i32 containing 4 packed bytes (8 FP4 nibbles).
        """
        if kpack_bytes != 16:
            raise ValueError(f"MXFP4 requires kpack_bytes=16, got {kpack_bytes!r}")

        c128 = arith.constant(128, index=True)
        c4 = arith.constant(4, index=True)

        k0_base = base_k // c128
        k0 = k0_base + arith.constant(ku // 4, index=True)
        klane_hw = arith.constant(ku % 4, index=True)
        byte_offset = lane_div_16 * c4

        coord_pack = (n_blk, k0, klane_hw, n_intra, arith.constant(0, index=True))
        idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)
        idx_bytes = idx_pack + byte_offset

        b4 = _buffer_load_vec(
            buffer_ops,
            vector,
            b_rsrc,
            idx_bytes,
            elem_type=elem_type,
            vec_elems=4,
            elem_bytes=1,
            offset_in_bytes=True,
        )
        packed32 = vector.extract(
            vector.bitcast(T.vec(1, T.i32), b4),
            static_position=[0],
            dynamic_position=[],
        )
        return packed32


    def load_b_raw_mxfp4_dwordx4(
        buffer_ops,
        arith,
        vector,
        *,
        arg_b,
        b_rsrc,
        layout_b,
        base_k: ir.Value,
        n_blk: ir.Value,
        n_intra: ir.Value,
        lane_div_16: ir.Value,
        elem_type: ir.Type,
        kpack_bytes: int = 16,
        cache_modifier: int = 0,
    ):
        """Load 16 bytes (vec4_i32) of packed FP4 via buffer_load_dwordx4.

        CK-style addressing: klane = lane_div_16, loading the full kpack
        for the thread's sub-lane. Returns vec4_i32 where i32[j] contains
        8 FP4 elements for kIter j.

        Layout: ``(n0, k0, klane=4, nlane=16, kpack=16)``
        """
        if kpack_bytes != 16:
            raise ValueError(f"MXFP4 requires kpack_bytes=16, got {kpack_bytes!r}")

        c128 = arith.constant(128, index=True)
        k0 = base_k // c128

        coord_pack = (n_blk, k0, lane_div_16, n_intra, arith.constant(0, index=True))
        idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)

        b16 = _buffer_load_vec(
            buffer_ops,
            vector,
            b_rsrc,
            idx_pack,
            elem_type=elem_type,
            vec_elems=16,
            elem_bytes=1,
            offset_in_bytes=True,
        )
        return vector.bitcast(T.vec(4, T.i32), b16)


    def _unpack_b_mxfp4_bf16_hw(packed32, arith, vector, scale_f32):
        """Hardware fast-path: 4 x v_cvt_scalef32_pk_bf16_fp4."""
        vec2_i32 = T.i32x2
        vec1_i64 = T.vec(1, T.i64)

        lo0 = _cvt_scalef32_pk_bf16_fp4(packed32, scale_f32, 0, arith, vector)
        lo1 = _cvt_scalef32_pk_bf16_fp4(packed32, scale_f32, 1, arith, vector)
        v2_lo = vector.from_elements(vec2_i32, [lo0, lo1])
        v64_lo = vector.bitcast(vec1_i64, v2_lo)
        b0 = vector.extract(v64_lo, static_position=[0], dynamic_position=[])

        hi0 = _cvt_scalef32_pk_bf16_fp4(packed32, scale_f32, 2, arith, vector)
        hi1 = _cvt_scalef32_pk_bf16_fp4(packed32, scale_f32, 3, arith, vector)
        v2_hi = vector.from_elements(vec2_i32, [hi0, hi1])
        v64_hi = vector.bitcast(vec1_i64, v2_hi)
        b1 = vector.extract(v64_hi, static_position=[0], dynamic_position=[])

        return (b0, b1)


    def _unpack_b_mxfp4_bf16_sw(packed32, arith, vector, scale_f32):
        """Software fallback for non-GFX950 targets."""
        c_0f = arith.constant(0x0F, type=T.i32)
        c4 = arith.constant(4, type=T.i32)
        c8 = arith.constant(8, type=T.i32)
        c12 = arith.constant(12, type=T.i32)
        c16 = arith.constant(16, type=T.i32)
        c20 = arith.constant(20, type=T.i32)
        c24 = arith.constant(24, type=T.i32)
        c28 = arith.constant(28, type=T.i32)

        n0 = packed32 & c_0f
        n1 = arith.shrui(packed32, c4) & c_0f
        n2 = arith.shrui(packed32, c8) & c_0f
        n3 = arith.shrui(packed32, c12) & c_0f
        first = n0 | arith.shli(n1, c8) | arith.shli(n2, c16) | arith.shli(n3, c24)

        n4 = arith.shrui(packed32, c16) & c_0f
        n5 = arith.shrui(packed32, c20) & c_0f
        n6 = arith.shrui(packed32, c24) & c_0f
        n7 = arith.shrui(packed32, c28) & c_0f
        second = n4 | arith.shli(n5, c8) | arith.shli(n6, c16) | arith.shli(n7, c24)

        b0 = _fp4x4_in_i32_to_bf16x4_i64(first, arith, vector, scale_f32=scale_f32)
        b1 = _fp4x4_in_i32_to_bf16x4_i64(second, arith, vector, scale_f32=scale_f32)
        return (b0, b1)


    def unpack_b_mxfp4_bf16(packed32, arith, vector, scale_f32=None, use_hw_cvt=True):
        """Unpack 8 FP4 E2M1 nibbles (packed in i32) to 2 x i64 (8 bf16).

        Each byte of *packed32* holds two FP4 nibbles: low nibble = K_even,
        high nibble = K_even+1.  For ``mfma_f32_16x16x16bf16_1k`` the B
        operand needs 4 consecutive K values per i64.  So we unpack the
        lower 2 bytes (4 consecutive nibbles) into b0 and the upper 2 bytes
        into b1.

        *scale_f32* is the decoded E8M0 block-scale (as f32).

        When *use_hw_cvt* is True (default), uses the GFX950 hardware
        instruction ``v_cvt_scalef32_pk_bf16_fp4`` which converts 2 FP4
        nibbles to 2 bf16 (with scale) in a single VALU cycle.  This
        replaces ~144 VALU of the software fallback with 4 instructions.

        Returns ``(b0, b1)`` -- two i64 values, each containing 4 bf16 for
        one ``mfma_f32_16x16x16bf16_1k`` call.
        """
        if use_hw_cvt and scale_f32 is not None:
            return _unpack_b_mxfp4_bf16_hw(packed32, arith, vector, scale_f32)
        return _unpack_b_mxfp4_bf16_sw(packed32, arith, vector, scale_f32)
    return _types.SimpleNamespace(**dict(locals()))
_M_mfma_preshuffle_pipeline = _mod_mfma_preshuffle_pipeline()

def _mod_mixed_moe_gemm_2stage_common():
    # SPDX-License-Identifier: Apache-2.0
    # Copyright (c) 2025 FlyDSL Project Contributors

    """Shared MXFP4/FP8 MoE and heterogeneous MoE kernel builders."""

    """MoE GEMM stage1/stage2 kernel implementations (FlyDSL MFMA FP8/FP16/FP4).

    This module contains the **kernel builder code** for:
    - `moe_gemm1` (stage1, with silu/swiglu activation)
    - `moe_gemm2` (stage2)

    It is extracted from `tests/kernels/test_moe_gemm.py` so that:
    - `kernels/` holds the implementation
    - `tests/` holds correctness/perf harnesses

    Mixed-precision support (a_dtype x b_dtype):
    - fp8 x fp8, fp8 x fp4 (A8W4 on gfx950), fp4 x fp4,
      fp16 x fp16, int8 x int4, ...

    A8W4 path is selected by `a_dtype='fp8', b_dtype='fp4'` plus
    `gate_mode=GateMode.INTERLEAVE` + `a_scale_one=True` in stage1.
    """



    try:
        from flydsl.runtime.device import supports_bf16_global_atomics
    except ImportError:

        def supports_bf16_global_atomics(arch: str) -> bool:
            return str(arch).startswith(("gfx94", "gfx95", "gfx12"))


    from flydsl.expr import arith, const_expr, gpu, rocdl
    from flydsl.expr.typing import T

    buffer_ops = _M_buffer_ops
    vector = _M_vector
    default_f8_type = _M_kernels_common.default_f8_type
    GateMode = _GateMode

    crd2idx = _M_layout_utils.crd2idx
    idx2crd = _M_layout_utils.idx2crd
    layout_get = _M_layout_utils.get
    c_shuffle_epilog = _M_mfma_epilogues.c_shuffle_epilog
    default_epilog = _M_mfma_epilogues.default_epilog
    _buffer_load_vec = _M_mfma_preshuffle_pipeline._buffer_load_vec
    buffer_copy_gmem16_dwordx4 = _M_mfma_preshuffle_pipeline.buffer_copy_gmem16_dwordx4
    lds_store_4b_xor16 = _M_mfma_preshuffle_pipeline.lds_store_4b_xor16
    lds_store_8b_xor16 = _M_mfma_preshuffle_pipeline.lds_store_8b_xor16
    lds_store_16b_xor16 = _M_mfma_preshuffle_pipeline.lds_store_16b_xor16
    make_preshuffle_b_layout = _M_mfma_preshuffle_pipeline.make_preshuffle_b_layout
    make_preshuffle_scale_layout = _M_mfma_preshuffle_pipeline.make_preshuffle_scale_layout
    swizzle_xor16 = _M_mfma_preshuffle_pipeline.swizzle_xor16
    tile_chunk_coord_i32 = _M_mfma_preshuffle_pipeline.tile_chunk_coord_i32


    @contextmanager
    def _if_then(if_op):
        """Compat helper for SCF IfOp then-region across old/new Python APIs."""
        with ir.InsertionPoint(if_op.then_block):
            try:
                yield if_op.then_block
            finally:
                blk = if_op.then_block
                if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                    scf.YieldOp([])


    _VALID_A_DTYPES = frozenset(("fp8", "fp16", "bf16", "int8", "fp4"))
    _VALID_B_DTYPES = frozenset(("fp8", "fp16", "int8", "int4", "fp4", "mxfp4"))


    def validate_moe_dtypes(a_dtype: str, b_dtype: str) -> None:
        """Validate a_dtype/b_dtype strings for mixed MoE kernels."""
        if a_dtype not in _VALID_A_DTYPES:
            raise ValueError(
                f"a_dtype must be one of {tuple(sorted(_VALID_A_DTYPES))}, got {a_dtype!r}"
            )
        if b_dtype not in _VALID_B_DTYPES:
            raise ValueError(
                f"b_dtype must be one of {tuple(sorted(_VALID_B_DTYPES))}, got {b_dtype!r}"
            )


    def _barrier(vmcnt=63, lgkmcnt=63):
        """Emit s_waitcnt + s_barrier via inline asm.

        Bypasses LLVM SIInsertWaitcnts which would insert a conservative
        s_waitcnt vmcnt(0) lgkmcnt(0) before every S_BARRIER MI.
        """
        parts = []
        needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
        if needs_waitcnt:
            wc = []
            if vmcnt < 63:
                wc.append(f"vmcnt({vmcnt})")
            if lgkmcnt < 63:
                wc.append(f"lgkmcnt({lgkmcnt})")
            parts.append("s_waitcnt " + " ".join(wc))
        parts.append("s_barrier")
        llvm.InlineAsmOp(
            res=None,
            operands_=[],
            asm_string="\n".join(parts),
            constraints="",
            has_side_effects=True,
            is_align_stack=False,
        )


    # HEAD generic path uses `barrier`; a16w4 uses `_barrier`.
    barrier = _barrier


    def compile_mixed_moe_gemm1_common(
        *,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        doweight_stage1: bool,
        a_dtype: str = "fp8",
        b_dtype: str = "fp4",
        out_dtype: str = "f16",
        act: str = "silu",
        use_cshuffle_epilog: bool | None = None,
        enable_bias: bool = False,
        model_dim_pad: int = 0,
        inter_dim_pad: int = 0,
        persist_m: int = 1,
        use_async_copy: bool = False,
        waves_per_eu: int = 4,
        k_batch: int = 1,
        b_nt: int = 0,
        gate_mode: GateMode = GateMode.SEPARATED,
        a_scale_one: bool = False,
        xcd_swizzle: int = 0,
        k_wave: int = 1,
        shared_expert_id: int | None = None,
        v2_output_layout: bool = False,
    ):
        """Compile stage1 kernel: act(X @ W_gate.T, X @ W_up.T) -> [tokens*topk, inter_dim]."""
        heterogeneous_b = shared_expert_id is not None
        if heterogeneous_b and shared_expert_id != experts - 1:
            raise ValueError(
                "FHMoE stage1 requires shared_expert_id == experts - 1; "
                f"got {shared_expert_id=} and {experts=}"
            )
        gpu_arch = get_hip_arch()
        allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
        allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

        if a_dtype not in ("fp8", "fp4"):
            raise ValueError(f"a_dtype must be one of ('fp8','fp4'), got {a_dtype!r}")
        if b_dtype not in ("fp8", "fp4"):
            raise ValueError(f"b_dtype must be one of ('fp8','fp4'), got {b_dtype!r}")
        is_f8_a = a_dtype == "fp8"
        is_f4_a = a_dtype == "fp4"
        is_f4_b = b_dtype == "fp4"
        is_f8_b = b_dtype == "fp8"
        if heterogeneous_b and not is_f4_b:
            raise ValueError("Heterogeneous B requires MXFP4 routed weights")

        sort_block_m = tile_m
        num_waves = min(4, tile_n // 32)
        # accumulators are reduced in LDS before the epilogue. k_wave=1 keeps the
        num_n_waves = num_waves
        num_waves_total = num_n_waves * k_wave
        # threads, so the cooperative-load striding is group-local.
        a_load_threads = num_n_waves * 64
        total_threads = num_waves_total * 64
        pack_M = 1 if tile_m < 32 else 2
        n_per_wave = tile_n // num_waves
        pack_N = min(2, n_per_wave // 16)
        pack_K = 2
        scale_mn_pack = 2
        elem_bytes = 1
        a_elem_bytes = 1
        b_elem_bytes = 1
        tile_k_bytes = int(tile_k) * int(a_elem_bytes)
        a_elem_vec_pack = 2 if is_f4_a else 1
        cbsz = 0 if is_f8_a else 4
        blgp = 0 if is_f8_b else 4
        b_byte_div = 2 if is_f4_b else 1
        b_cells_per_ku = 2 if is_f8_b else 1

        if (tile_k_bytes % 64) != 0:
            raise ValueError(f"tile_k_bytes must be divisible by 64, got {tile_k_bytes}")

        out_s = str(out_dtype).strip().lower()
        out_is_f32 = out_s in ("f32", "fp32", "float")
        out_is_bf16 = out_s in ("bf16", "bfloat16")

        def x_elem_type():
            if is_f4_b:
                return default_f8_type() if is_f8_a else T.i8
            return default_f8_type()

        def w_elem_type():
            if is_f4_b:
                return T.i8
            return default_f8_type()

        def out_elem():
            return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

        def load_bias_scalar(bias_rsrc, offset):
            return buffer_ops.buffer_load(bias_rsrc, offset, vec_width=1, dtype=T.f32)

        mock_gate_only = gate_mode is GateMode.MOCK_GATE_ONLY
        gate_up_interleave = gate_mode is GateMode.INTERLEAVE
        gate_only = gate_mode is GateMode.GATE_ONLY

        is_splitk = k_batch > 1
        if mock_gate_only and not is_splitk:
            raise ValueError("mock_gate_only requires k_batch > 1 (split-K)")
        if is_splitk:
            k_per_batch = model_dim // k_batch
            assert (
                model_dim % k_batch == 0
            ), f"model_dim={model_dim} not divisible by k_batch={k_batch}"
            assert (
                k_per_batch % tile_k == 0
            ), f"K_per_batch={k_per_batch} not divisible by tile_k={tile_k}"

            out_dtype = "bf16"
        else:
            k_per_batch = model_dim
        k_dim = k_per_batch

        if k_wave > 1:
            if k_dim % k_wave != 0:
                raise ValueError(f"model_dim={k_dim} not divisible by k_wave={k_wave}")
            klen = k_dim // k_wave
            if klen % tile_k != 0:
                raise ValueError(f"K per group={klen} not divisible by tile_k={tile_k}")
        else:
            klen = k_dim

        bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
        # For k_wave=1 this equals total_threads (unchanged behaviour).
        if bytes_x_per_tile % a_load_threads != 0:
            raise ValueError(
                f"tile_m*tile_k*elem_bytes must be divisible by {a_load_threads}"
            )
        bytes_per_thread_x = bytes_x_per_tile // a_load_threads

        lds_stride = tile_k

        _use_cshuffle_epilog = True

        need_fp4 = out_dtype == "fp4"
        need_fp8 = out_dtype == "fp8"
        need_quant = need_fp4 or need_fp8
        need_sort = need_quant

        fp4q_tag = "_fp4q" if need_fp4 else ""
        fp8q_tag = "_fp8q" if need_fp8 else ""
        sort_tag = "_sort" if need_sort else ""
        async_tag = "_async" if use_async_copy else ""
        sk_tag = f"_sk{k_batch}" if is_splitk else ""
        kw_tag = f"_kw{k_wave}" if k_wave > 1 else ""
        go_tag = "_go" if mock_gate_only else ""
        gui_tag = "_gui" if gate_up_interleave else ""
        as1_tag = "_as1" if a_scale_one else ""
        xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
        v2out_tag = "_v2out" if v2_output_layout else ""
        # Keep the historical name for silu; swiglu/situv2 get distinct symbols so
        # they cannot alias. SiTUv2 beta values are runtime kernel arguments and
        # therefore must not be part of the on-disk symbol/cache identity.
        act_tag = "" if act == "silu" else f"_{act}"
        heterogeneous_tag = f"_shared_fp8_e{shared_expert_id}" if heterogeneous_b else ""
        # ABI v33 adds four runtime SiTUv2 beta scalars; heterogeneous ABI tracks one
        # version ahead of the ordinary kernel.
        kernel_version = 34 if heterogeneous_b else 33
        module_name = (
            f"mfma_moe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"
            f"_t{tile_m}x{tile_n}x{tile_k}_pm{persist_m}{fp4q_tag}{fp8q_tag}{sort_tag}{async_tag}{sk_tag}{kw_tag}{go_tag}{gui_tag}{as1_tag}{xcd_tag}{act_tag}{v2out_tag}{heterogeneous_tag}_v{kernel_version}"
        ).replace("-", "_")

        cshuffle_elem_bytes = 4 if need_quant else (4 if out_is_f32 else 2)
        single_x_bytes = int(tile_m) * int(lds_stride) * int(a_elem_bytes)
        x_region_bytes = k_wave * single_x_bytes
        lds_out_bytes = (
            cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
        )
        lds_tid_bytes = int(tile_m) * 4
        input_elems = single_x_bytes if a_elem_bytes == 1 else (single_x_bytes // 2)

        GLOBAL_ALIGN = 1024
        std_pong = max(x_region_bytes, lds_out_bytes) + lds_tid_bytes
        std_ping = x_region_bytes
        std_pong_aligned = allocator_pong._align(std_pong, 128)
        std_total = allocator_pong._align(
            std_pong_aligned, GLOBAL_ALIGN
        ) + allocator_pong._align(std_ping, 128)
        lds_limit = {"gfx950": 163840, "gfx942": 65536}.get(gpu_arch, 0)

        split_lds_out = (
            k_wave == 1
            and lds_limit > 0
            and lds_out_bytes > 0
            and std_total > lds_limit
            and num_waves >= 2
        )

        if split_lds_out:
            half_out_bytes = cshuffle_elem_bytes * int(tile_m) * (int(tile_n) // 2)
            pong_buffer_bytes = max(single_x_bytes, half_out_bytes)
            ping_buffer_bytes = max(single_x_bytes, half_out_bytes)
        else:
            pong_buffer_bytes = max(x_region_bytes, lds_out_bytes)
            ping_buffer_bytes = x_region_bytes

        def x_lds_elem():
            return default_f8_type()

        lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = lds_pong_offset + pong_buffer_bytes
        lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
        allocator_pong.ptr = lds_tid_offset_pong + lds_tid_bytes

        lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
        allocator_ping.ptr = lds_ping_offset + ping_buffer_bytes

        if waves_per_eu is not None and waves_per_eu >= 1:
            total_cu_lds = 160 * 1024
            min_lds = total_cu_lds // (waves_per_eu + 1) + 1
            pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
            ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
            cur_lds = pong_sz + ping_sz
            if cur_lds < min_lds:
                allocator_ping.ptr += min_lds - cur_lds

        kpack_bytes = 16
        out_elem_bytes = 4 if out_is_f32 else 2
        w_elem_bytes = 1
        w_elem_pack = 2 if is_f4_b else 1
        w_nbytes = (experts * (2 * inter_dim) * model_dim * w_elem_bytes) // w_elem_pack
        shared_w_nbytes = (2 * inter_dim) * model_dim
        bias_nbytes = experts * (2 * inter_dim) * 4

        e_vec_s1 = min(tile_n // 32, 8)
        if need_quant:
            e_vec_s1 = max(2, e_vec_s1)
        num_threads_per_quant_blk_s1 = 32 // e_vec_s1
        shuffle_dists_s1 = []
        sh_val = 1
        while sh_val < num_threads_per_quant_blk_s1:
            shuffle_dists_s1.append(sh_val)
            sh_val *= 2
        num_shuffle_steps_s1 = len(shuffle_dists_s1)

        pipe_m_repeat = tile_m // 16
        pipe_k_unroll = tile_k_bytes // 128
        pipe_k_unroll_packed = pipe_k_unroll // pack_K
        pipe_num_acc_n = n_per_wave // 16

        pipe_a_groups = []
        for mi in range(pipe_m_repeat):
            grp = []
            for k in range(pipe_k_unroll):
                grp.append((k, mi))
                if len(grp) == 2:
                    pipe_a_groups.append(grp)
                    grp = []
            if grp:
                pipe_a_groups.append(grp)

        pipe_b_loads = []
        for ku in range(pipe_k_unroll):
            for ni in range(pipe_num_acc_n):
                pipe_b_loads.append(("gate", ku, ni))
                if not mock_gate_only and not gate_up_interleave:
                    pipe_b_loads.append(("up", ku, ni))

        pipe_num_acc_n_packed = pipe_num_acc_n // pack_N
        pipe_all_mfma = []
        for ku128 in range(pipe_k_unroll_packed):
            for ni_packed in range(pipe_num_acc_n_packed):
                for ikxdl in range(pack_K):
                    for inxdl in range(pack_N):
                        k_idx = ku128 * pack_K + ikxdl
                        ni_idx = ni_packed * pack_N + inxdl
                        pipe_all_mfma.append((k_idx, ni_idx, ikxdl, inxdl, ku128))

        pipe_mfma_per_phase = max(1, len(pipe_all_mfma) // 4)
        pipe_n_phases = len(pipe_all_mfma) // pipe_mfma_per_phase

        a_groups_per_phase = (len(pipe_a_groups) + pipe_n_phases - 1) // pipe_n_phases
        pipe_phases = []
        mfma_i = 0
        a_i = 0
        for _p in range(pipe_n_phases):
            a_reads = []
            for _ in range(a_groups_per_phase):
                if a_i < len(pipe_a_groups):
                    a_reads.extend(pipe_a_groups[a_i])
                    a_i += 1
            phase = {
                "mfma": pipe_all_mfma[mfma_i : mfma_i + pipe_mfma_per_phase],
                "a_reads": a_reads,
                "b_loads": [],
                "has_scale": (_p == 0),
            }
            mfma_i += pipe_mfma_per_phase
            pipe_phases.append(phase)

        bi = 0
        for _p in range(1, pipe_n_phases):
            rem_b = len(pipe_b_loads) - bi
            rem_p = pipe_n_phases - _p
            n_b = (rem_b + rem_p - 1) // rem_p if rem_p > 0 else 0
            for _ in range(n_b):
                if bi < len(pipe_b_loads):
                    pipe_phases[_p]["b_loads"].append(pipe_b_loads[bi])
                    bi += 1

        pp_mfma = [p["mfma"] for p in pipe_phases]
        pp_a_reads = [p["a_reads"] for p in pipe_phases]
        pp_b_loads = [p["b_loads"] for p in pipe_phases]
        pp_has_scale = [p["has_scale"] for p in pipe_phases]

        fp4_ratio = 2 if a_dtype == "fp4" else 1
        gui_ratio = 1 if gate_up_interleave else 2

        if True:

            def _emit_moe_gemm1(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_shared_w: fx.Pointer,
                arg_shared_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_num_valid_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                arg_out_scale_sorted: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_n_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
                f32_situ_beta: fx.Float32,
                f32_situ_beta_rcp: fx.Float32,
                f32_situ_linear_beta: fx.Float32,
                f32_situ_linear_beta_rcp: fx.Float32,
                f32_swiglu_limit: fx.Float32,
            ):

                tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
                n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
                k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
                size_expert_ids_in = arith.index_cast(
                    ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
                )
                # Runtime clamp bound for the activation.  Host passes the configured
                # swiglu_limit (7.0 default for swiglu) or +inf to disable clamping.
                # ``-lim`` is precomputed once; ``min(x, lim) == -max(-x, -lim)`` so
                # the kernel uses only the wrapped maximumf/negation ops.
                swiglu_neg_limit = -f32_swiglu_limit

                x_elem = default_f8_type()
                f32 = T.f32
                i32 = T.i32
                i64 = T.i64
                vec4_f32 = T.vec(4, f32)
                vec16_elems = 16 if a_elem_bytes == 1 else 8
                vec16_x = T.vec(vec16_elems, x_elem)
                vec2_i64 = T.vec(2, i64)

                def ptr_buffer_resource(ptr, num_records_bytes):
                    addr = fx.ptrtoint(ptr)
                    addr_i64 = arith.index_cast(T.i64, addr)
                    return buffer_ops.create_buffer_resource_from_addr(
                        addr_i64, num_records_bytes=num_records_bytes
                    )

                acc_init = arith.constant_vector(0.0, vec4_f32)

                c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
                b_layout = make_preshuffle_b_layout(
                    arith,
                    c_n=c_n_total,
                    c_k=k_in // b_byte_div,
                    kpack_bytes=kpack_bytes,
                    elem_bytes=b_elem_bytes,
                )
                layout_b = b_layout.layout_b
                shared_c_n_total = arith.constant(2 * inter_dim, index=True)
                shared_b_layout = make_preshuffle_b_layout(
                    arith,
                    c_n=shared_c_n_total,
                    c_k=k_in,
                    kpack_bytes=kpack_bytes,
                    elem_bytes=1,
                )
                shared_layout_b = shared_b_layout.layout_b

                sorted_m = size_expert_ids_in * arith.constant(sort_block_m, index=True)
                layout_a_scale = make_preshuffle_scale_layout(
                    arith, c_mn=sorted_m, c_k=arith.constant(model_dim, index=True)
                )
                layout_b_scale = make_preshuffle_scale_layout(
                    arith, c_mn=c_n_total, c_k=arith.constant(model_dim, index=True)
                )

                eff_lds_stride = lds_stride
                eff_tile_k_bytes = tile_k_bytes
                if const_expr(use_async_copy and a_elem_vec_pack > 1):
                    eff_lds_stride = lds_stride // a_elem_vec_pack
                    eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack

                shape_lds = fx.make_shape(tile_m, eff_lds_stride)
                stride_lds = fx.make_stride(eff_lds_stride, 1)
                layout_lds = fx.make_layout(shape_lds, stride_lds)

                tx = gpu.thread_id("x")
                by = gpu.block_id("x")
                bx_persist = gpu.block_id("y")

                if const_expr(xcd_swizzle > 0):
                    num_xcds = 8
                    one = arith.constant(1, index=True)
                    tile_n_idx = arith.constant(tile_n, index=True)
                    inter_pad_idx = arith.constant(2 * inter_dim_pad, index=True)
                    if const_expr(mock_gate_only or gate_up_interleave):
                        grid_x = (n_in - inter_pad_idx + tile_n_idx - one) // tile_n_idx
                    else:
                        two = arith.constant(2, index=True)
                        grid_x = (
                            (n_in - inter_pad_idx + two * tile_n_idx - one)
                            // tile_n_idx
                            // two
                        )
                    persist_m_idx = arith.constant(persist_m, index=True)
                    grid_y = (size_expert_ids_in + persist_m_idx - one) // persist_m_idx
                    linear_id = bx_persist * grid_x + by
                    num_workgroups = grid_x * grid_y
                    num_xcds_idx = arith.constant(num_xcds, index=True)
                    workgroups_per_xcd = num_workgroups // num_xcds_idx
                    workgroup_id = (linear_id % num_xcds_idx) * workgroups_per_xcd + (
                        linear_id // num_xcds_idx
                    )
                    group_m = arith.constant(xcd_swizzle, index=True)
                    workgroups_per_group = group_m * grid_x
                    group_id = workgroup_id // workgroups_per_group
                    first_m = group_id * group_m
                    remaining_m = grid_y - first_m
                    group_size_m = arith.select(
                        arith.cmpi(CmpIPredicate.ult, remaining_m, group_m),
                        remaining_m,
                        group_m,
                    )
                    workgroup_in_group = workgroup_id % workgroups_per_group
                    bx_persist = first_m + (workgroup_in_group % group_size_m)
                    by = workgroup_in_group // group_size_m

                by_n = by * arith.constant(tile_n, index=True)

                k_base_idx = arith.index(0)
                if const_expr(is_splitk):
                    bz = gpu.block_id("z")
                    k_base_idx = bz * arith.constant(k_dim, index=True)

                k_blocks16 = arith.constant(eff_tile_k_bytes // 16, index=True)
                layout_tx_wave_lane = fx.make_layout((num_waves_total, 64), stride=(64, 1))
                layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

                base_ptr_pong = allocator_pong.get_base()
                base_ptr_ping = allocator_ping.get_base()
                lds_x_pong = SmemPtr(
                    base_ptr_pong, lds_pong_offset, x_lds_elem(), shape=(input_elems,)
                ).get()
                lds_x_ping = SmemPtr(
                    base_ptr_ping, lds_ping_offset, x_lds_elem(), shape=(input_elems,)
                ).get()
                lds_out_elem_type = (
                    T.f32 if need_quant else (T.bf16 if out_is_bf16 else T.f16)
                )
                if const_expr(split_lds_out and _use_cshuffle_epilog):
                    half_out_elems = int(tile_m) * (int(tile_n) // 2)
                    lds_out = SmemPtr(
                        base_ptr_pong,
                        lds_pong_offset,
                        lds_out_elem_type,
                        shape=(half_out_elems,),
                    ).get()
                    lds_out_B = SmemPtr(
                        base_ptr_ping,
                        lds_ping_offset,
                        lds_out_elem_type,
                        shape=(half_out_elems,),
                    ).get()
                else:
                    lds_out = (
                        SmemPtr(
                            base_ptr_pong,
                            lds_pong_offset,
                            lds_out_elem_type,
                            shape=(tile_m * tile_n,),
                        ).get()
                        if _use_cshuffle_epilog
                        else None
                    )
                    lds_out_B = None
                lds_tid = SmemPtr(
                    base_ptr_pong, lds_tid_offset_pong, T.i32, shape=(tile_m,)
                ).get()

                c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
                c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)

                x_nbytes_idx = (tokens_in * k_in * c_elem_bytes) // c_a_pack
                x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
                x_rsrc = ptr_buffer_resource(arg_x, x_nbytes_i32)

                shared_w_rsrc = ptr_buffer_resource(arg_shared_w, shared_w_nbytes)

                numids_rsrc = ptr_buffer_resource(
                    arg_num_valid_ids, arith.constant(4, type=T.i32)
                )
                num_valid_i32 = buffer_ops.buffer_load(
                    numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
                )

                sx_rsrc = 1
                sw_rsrc = 1
                if const_expr(not a_scale_one):
                    c32 = arith.constant(32, index=True)
                    kblk = k_in // c32
                    scale_rows = (
                        (sorted_m + arith.constant(31, index=True))
                        // arith.constant(32, index=True)
                        * arith.constant(32, index=True)
                    )
                    sx_nbytes_idx = scale_rows * kblk
                    sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                    sx_rsrc = ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)

                c32 = arith.constant(32, index=True)
                kblk_w = k_in // c32
                mn_w = arith.constant(experts * (2 * inter_dim), index=True)
                sw_nbytes_idx = mn_w * kblk_w
                sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
                sw_rsrc = ptr_buffer_resource(arg_scale_w, sw_nbytes_i32)
                shared_scale_rows = arith.constant(
                    ((2 * inter_dim + 255) // 256) * 256, index=True
                )
                shared_scale_cols = arith.constant(
                    ((model_dim // 32 + 7) // 8) * 8, index=True
                )
                shared_sw_nbytes_i32 = arith.index_cast(
                    T.i32, shared_scale_rows * shared_scale_cols
                )
                shared_sw_rsrc = ptr_buffer_resource(
                    arg_shared_scale_w, shared_sw_nbytes_i32
                )

                sorted_nbytes_idx = size_expert_ids_in * arith.constant(
                    sort_block_m * 4, index=True
                )
                sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
                sorted_rsrc = ptr_buffer_resource(arg_sorted_token_ids, sorted_nbytes_i32)
                sorted_w_rsrc = ptr_buffer_resource(arg_sorted_weights, sorted_nbytes_i32)

                eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
                eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
                expert_rsrc = ptr_buffer_resource(arg_expert_ids, eid_nbytes_i32)
                bias_rsrc = (
                    ptr_buffer_resource(arg_bias, bias_nbytes) if enable_bias else None
                )

                # #3476: pad group-N (= inter_dim/32) up to a multiple of 8 so it
                sorted_scale_cols = ((inter_dim // 32) + 7) // 8 * 8
                sorted_scale_cols_i32 = arith.constant(sorted_scale_cols, type=T.i32)
                sorted_scale_rsrc = None
                if const_expr(need_sort):
                    sort_rows_idx = size_expert_ids_in * arith.constant(
                        sort_block_m, index=True
                    )
                    sort_padded_rows = (
                        (sort_rows_idx + arith.constant(255, index=True))
                        // arith.constant(256, index=True)
                        * arith.constant(256, index=True)
                    )
                    sort_padded_cols = arith.constant(
                        ((sorted_scale_cols + 7) // 8) * 8, index=True
                    )
                    sort_scale_nbytes = arith.index_cast(
                        T.i32, sort_padded_rows * sort_padded_cols
                    )
                    sorted_scale_rsrc = ptr_buffer_resource(
                        arg_out_scale_sorted, sort_scale_nbytes
                    )

                PERSIST_M = persist_m
                c0_p = arith.constant(0, index=True)
                c1_p = arith.constant(1, index=True)
                c_pm = arith.constant(PERSIST_M, index=True)
                for_persist = scf.ForOp(c0_p, c_pm, c1_p)
                for_ip = ir.InsertionPoint(for_persist.body)
                for_ip.__enter__()
                mi_p = for_persist.induction_variable
                bx = bx_persist * c_pm + mi_p
                bx_m = bx * arith.constant(sort_block_m, index=True)

                bx_m_i32 = arith.index_cast(T.i32, bx_m)
                blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=T.i32
                )
                expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
                exp_valid = arith.cmpi(
                    CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
                )
                if const_expr(heterogeneous_b):
                    is_shared_expert = arith.cmpi(
                        CmpIPredicate.eq,
                        expert_i32,
                        arith.constant(shared_expert_id, type=T.i32),
                    )

                def moe_gemm1_body(shared_b: bool = False):
                    body_k_base_idx = k_base_idx
                    body_lds_x_pong = lds_x_pong
                    body_lds_x_ping = lds_x_ping
                    body_b_has_full_operand = is_f8_b or shared_b
                    body_b_load_mult = 2 if body_b_has_full_operand else 1
                    body_vmcnt_before_barrier = (
                        tile_m // 32 // fp4_ratio
                        + tile_n // 32 * gui_ratio * body_b_load_mult
                    )
                    expert_off_idx = expert_idx * arith.constant(2 * inter_dim, index=True)
                    if const_expr(shared_b):
                        weight_expert_off_idx = arith.index(0)
                        weight_scale_rsrc = shared_sw_rsrc
                    else:
                        weight_expert_off_idx = expert_off_idx
                        weight_scale_rsrc = sw_rsrc

                    def mixed_b_mfma(
                        a128,
                        b128,
                        acc,
                        opsel_a,
                        a_scale_val,
                        opsel_b,
                        b_scale_val,
                    ):
                        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            vec4_f32,
                            [
                                a128,
                                b128,
                                acc,
                                cbsz,
                                0 if shared_b else blgp,
                                opsel_a,
                                a_scale_val,
                                opsel_b,
                                b_scale_val,
                            ],
                        )

                    # per-expert 64-bit re-base: 32-bit buffer voffset overflows when w1 > 4GB
                    per_expert_w_bytes = w_nbytes // experts
                    w_addr_i64 = arith.index_cast(T.i64, fx.ptrtoint(arg_w))
                    expert_byte_off = arith.index_cast(
                        T.i64, expert_idx * arith.constant(per_expert_w_bytes, index=True)
                    )
                    w_rsrc_e = buffer_ops.create_buffer_resource_from_addr(
                        arith.addi(w_addr_i64, expert_byte_off),
                        num_records_bytes=per_expert_w_bytes,
                    )

                    x_load_bytes = 16
                    num_x_loads = bytes_per_thread_x // x_load_bytes
                    chunk_i32 = x_load_bytes // 4

                    c_k_div4 = (
                        (k_in // c_a_pack) * arith.constant(int(a_elem_bytes), index=True)
                    ) // arith.index(4)
                    tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                        4 * int(a_elem_vec_pack)
                    )
                    layout_x_tile_div4 = fx.make_layout(
                        (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                    )
                    c_chunk_i32 = arith.constant(chunk_i32, index=True)
                    if const_expr(k_wave > 1):
                        x_load_tid = tx % arith.constant(a_load_threads, index=True)
                    else:
                        x_load_tid = tx
                    tx_i32_base = x_load_tid * c_chunk_i32

                    topk_i32 = arith.constant(topk)
                    mask24 = arith.constant(0xFFFFFF)
                    tokens_i32 = arith.index_cast(T.i32, tokens_in)

                    def x_tile_chunk_coord_i32(i: int):
                        return tile_chunk_coord_i32(
                            arith,
                            tx_i32_base=tx_i32_base,
                            i=i,
                            total_threads=a_load_threads,
                            layout_tile_div4=layout_x_tile_div4,
                            chunk_i32=chunk_i32,
                        )

                    def load_x(idx_i32):
                        idx_elem = (
                            idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                        )

                    x_row_base_div4 = []
                    x_col_local_i32 = []
                    x_row_local = []

                    for i in range_constexpr(num_x_loads):
                        row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                        x_row_local.append(row_local)
                        x_col_local_i32.append(col_local_i32)

                        sorted_row_i = bx_m + row_local
                        fused_i = buffer_ops.buffer_load(
                            sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                        )
                        t_i32 = arith.andi(fused_i, mask24)
                        s_i32 = arith.shrui(fused_i, arith.constant(24))
                        t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                        s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32)
                        ts_valid = arith.andi(t_valid, s_valid)
                        t_safe = arith.select(ts_valid, t_i32, arith.constant(0))

                        t_idx = arith.index_cast(ir.IndexType.get(), t_safe)
                        x_row_base_div4.append(t_idx * c_k_div4)

                    def load_x_tile(base_k):
                        base_k_div4 = (
                            (base_k // c_a_pack)
                            * arith.constant(int(a_elem_bytes), index=True)
                        ) // arith.index(4)
                        parts = []
                        for i in range_constexpr(num_x_loads):
                            idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                            x_vec = load_x(idx_i32)
                            parts.append(vector.bitcast(T.vec(4, i32), x_vec))
                        return parts

                    coord_wl = idx2crd(fx.Int32(tx), layout_tx_wave_lane)
                    wave_id = layout_get(coord_wl, 0)
                    lane_id = layout_get(coord_wl, 1)
                    coord_l16 = idx2crd(fx.Int32(lane_id), layout_lane16)
                    lane_div_16 = layout_get(coord_l16, 0)
                    lane_mod_16 = layout_get(coord_l16, 1)
                    row_a_lds = lane_mod_16
                    col_offset_base = lane_div_16 * arith.constant(16, index=True)

                    if const_expr(k_wave > 1):
                        wave_k_id = wave_id / arith.constant(num_n_waves, index=True)
                        body_k_base_idx = body_k_base_idx + wave_k_id * arith.constant(
                            klen, index=True
                        )
                        grp_x_bytes = wave_k_id * arith.constant(single_x_bytes, index=True)
                        x_view_ty = _mT.memref(
                            input_elems, x_lds_elem(), memory_space=_lds_space()
                        )
                        pong_off = arith.constant(lds_pong_offset, index=True) + grp_x_bytes
                        ping_off = arith.constant(lds_ping_offset, index=True) + grp_x_bytes
                        body_lds_x_pong = memref.view(
                            x_view_ty, base_ptr_pong, pong_off, sizes=[]
                        )
                        body_lds_x_ping = memref.view(
                            x_view_ty, base_ptr_ping, ping_off, sizes=[]
                        )
                    else:
                        wave_k_id = arith.index(0)

                    num_acc_n = n_per_wave // 16
                    c_n_per_wave = arith.constant(n_per_wave, index=True)
                    wave_n_id = wave_id % arith.constant(num_waves, index=True)
                    n_tile_base = wave_n_id * c_n_per_wave

                    gate_n_intra_list = []
                    gate_n_blk_list = []
                    up_n_intra_list = []
                    up_n_blk_list = []
                    col_g_list = []
                    c_n0_static = experts * (2 * inter_dim) // 16
                    layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                    inter_idx = arith.constant(inter_dim, index=True)

                    for i in range_constexpr(num_acc_n):
                        offset = i * 16
                        c_offset = arith.constant(offset, index=True)
                        if const_expr(not gate_up_interleave):
                            col_g = by_n + n_tile_base + c_offset + lane_mod_16
                            col_g_list.append(col_g)

                        global_n = by_n + n_tile_base + c_offset + lane_mod_16
                        gate_row_w = global_n
                        gate_coord = idx2crd(fx.Int32(gate_row_w), layout_n_blk_intra)
                        gate_n_blk_list.append(layout_get(gate_coord, 0))
                        gate_n_intra_list.append(layout_get(gate_coord, 1))
                        if const_expr(not mock_gate_only and not gate_up_interleave):
                            up_row_w = gate_row_w + inter_idx
                            up_coord = idx2crd(fx.Int32(up_row_w), layout_n_blk_intra)
                            up_n_blk_list.append(layout_get(up_coord, 0))
                            up_n_intra_list.append(layout_get(up_coord, 1))

                    if const_expr(gate_up_interleave):
                        gui_num_acc_n_out = num_acc_n // pack_N
                        for gui_i in range_constexpr(gui_num_acc_n_out):
                            gui_offset = gui_i * 16
                            gui_c_offset = arith.constant(gui_offset, index=True)
                            gui_col_g = (
                                (by_n + n_tile_base) // arith.constant(2, index=True)
                                + gui_c_offset
                                + lane_mod_16
                            )
                            col_g_list.append(gui_col_g)

                    m_repeat = tile_m // 16
                    k_unroll = tile_k_bytes // 128
                    k_unroll_packed = k_unroll // pack_K
                    m_repeat_packed = m_repeat // pack_M
                    num_acc_n_packed = num_acc_n // pack_N

                    K_per_ku = tile_k // k_unroll
                    pad_k_elems = (
                        (model_dim_pad % tile_k)
                        if (not is_splitk and model_dim_pad > 0)
                        else 0
                    )
                    pad_ku_skip = pad_k_elems // K_per_ku
                    tail_ku = k_unroll - pad_ku_skip
                    tail_ku_packed = (
                        (tail_ku + pack_K - 1) // pack_K if pad_ku_skip > 0 else None
                    )

                    def load_b_packs_k64(base_k, ku: int, n_blk, n_intra):
                        c64 = arith.constant(64, index=True)
                        c1 = arith.constant(1, index=True)
                        k1 = lane_div_16
                        vec_elems = kpack_bytes // int(b_elem_bytes)

                        def load_cell(rsrc, b_layout_arg, elem_type, k0):
                            coord_pack = (
                                n_blk,
                                k0,
                                k1,
                                n_intra,
                                arith.constant(0, index=True),
                            )
                            idx_pack = crd2idx(coord_pack, b_layout_arg)
                            b16 = _buffer_load_vec(
                                buffer_ops,
                                vector,
                                rsrc,
                                idx_pack,
                                elem_type=elem_type,
                                vec_elems=vec_elems,
                                elem_bytes=b_elem_bytes,
                                offset_in_bytes=(b_elem_bytes == 1),
                                cache_modifier=b_nt,
                            )
                            b_i64x2 = vector.bitcast(vec2_i64, b16)
                            return (
                                vector.extract(
                                    b_i64x2, static_position=[0], dynamic_position=[]
                                ),
                                vector.extract(
                                    b_i64x2, static_position=[1], dynamic_position=[]
                                ),
                            )

                        routed_k0_base = base_k // c64 + arith.constant(
                            ku * b_cells_per_ku, index=True
                        )
                        if const_expr(shared_b):
                            shared_k0_base = (base_k * arith.index(2)) // c64
                            shared_k0_base += arith.constant(ku * 2, index=True)
                            s0, s1 = load_cell(
                                shared_w_rsrc,
                                shared_layout_b,
                                default_f8_type(),
                                shared_k0_base,
                            )
                            s2, s3 = load_cell(
                                shared_w_rsrc,
                                shared_layout_b,
                                default_f8_type(),
                                shared_k0_base + c1,
                            )
                            return s0, s1, s2, s3

                        b0, b1 = load_cell(
                            w_rsrc_e, layout_b, w_elem_type(), routed_k0_base
                        )
                        if const_expr(is_f8_b):
                            b2, b3 = load_cell(
                                w_rsrc_e,
                                layout_b,
                                w_elem_type(),
                                routed_k0_base + c1,
                            )
                            return b0, b1, b2, b3
                        return b0, b1

                    def load_b_tile(base_k, ku_limit=k_unroll):
                        """Load B tiles -> (gate_b_tile, up_b_tile); up is None for
                        interleave/mock. Tile entry (packs0..3); packs2/3 fp8-B only."""
                        gate_b_tile = []
                        up_b_tile = (
                            [] if (not mock_gate_only and not gate_up_interleave) else None
                        )
                        for ku in range_constexpr(ku_limit):
                            g_packs0, g_packs1, g_packs2, g_packs3 = [], [], [], []
                            u_packs0, u_packs1, u_packs2, u_packs3 = [], [], [], []
                            for ni in range_constexpr(num_acc_n):
                                gb = load_b_packs_k64(
                                    base_k, ku, gate_n_blk_list[ni], gate_n_intra_list[ni]
                                )
                                g_packs0.append(gb[0])
                                g_packs1.append(gb[1])
                                if const_expr(body_b_has_full_operand):
                                    g_packs2.append(gb[2])
                                    g_packs3.append(gb[3])
                                if const_expr(
                                    not mock_gate_only and not gate_up_interleave
                                ):
                                    ub = load_b_packs_k64(
                                        base_k, ku, up_n_blk_list[ni], up_n_intra_list[ni]
                                    )
                                    u_packs0.append(ub[0])
                                    u_packs1.append(ub[1])
                                    if const_expr(body_b_has_full_operand):
                                        u_packs2.append(ub[2])
                                        u_packs3.append(ub[3])
                            gate_b_tile.append((g_packs0, g_packs1, g_packs2, g_packs3))
                            if const_expr(not mock_gate_only and not gate_up_interleave):
                                up_b_tile.append((u_packs0, u_packs1, u_packs2, u_packs3))
                        return gate_b_tile, up_b_tile

                    scale_lane_elem = (
                        lane_div_16 * layout_b_scale.stride_klane + lane_mod_16
                    )

                    gate_scale_bases = []
                    up_scale_bases = []
                    for ni in range_constexpr(num_acc_n_packed):
                        col_base = (
                            by_n
                            + n_tile_base
                            + arith.constant(ni * 16 * pack_N, index=True)
                        )
                        gate_mni = (weight_expert_off_idx + col_base) // arith.constant(
                            32, index=True
                        )
                        gate_scale_bases.append(
                            gate_mni * layout_b_scale.stride_n0 + scale_lane_elem
                        )
                        if const_expr(not mock_gate_only and not gate_up_interleave):
                            up_mni = (
                                weight_expert_off_idx + inter_idx + col_base
                            ) // arith.constant(32, index=True)
                            up_scale_bases.append(
                                up_mni * layout_b_scale.stride_n0 + scale_lane_elem
                            )

                    if const_expr(not a_scale_one):
                        a_scale_bases = []
                        for mi in range_constexpr(m_repeat_packed):
                            a_mni = mi + bx_m // scale_mn_pack // 16
                            a_scale_bases.append(
                                a_mni * layout_a_scale.stride_n0 + scale_lane_elem
                            )

                    c16_idx = arith.constant(16, index=True)
                    c2_idx = arith.constant(2, index=True)
                    scale_mask_lo = arith.constant(0xFF, type=T.i32)

                    m_half_idx = arith.constant(0, type=T.i32)
                    m_half_i32 = arith.constant(0, type=T.i32)
                    scale_shift = arith.constant(0, type=T.i32)
                    scale_shift_hi = arith.constant(0, type=T.i32)
                    n_half_idx = arith.constant(0, type=T.i32)
                    n_half_i32 = arith.constant(0, type=T.i32)
                    bscale_shift = arith.constant(0, type=T.i32)
                    bscale_shift_hi = arith.constant(0, type=T.i32)
                    if const_expr(pack_M < scale_mn_pack):
                        m_half_idx = (bx_m // c16_idx) % c2_idx
                        m_half_i32 = arith.index_cast(T.i32, m_half_idx)
                        scale_shift = m_half_i32 * arith.constant(8, type=T.i32)
                        scale_shift_hi = scale_shift + arith.constant(16, type=T.i32)

                    if const_expr(pack_N < scale_mn_pack):
                        n_half_idx = (n_tile_base // c16_idx) % c2_idx
                        n_half_i32 = arith.index_cast(T.i32, n_half_idx)
                        bscale_shift = n_half_i32 * arith.constant(8, type=T.i32)
                        bscale_shift_hi = bscale_shift + arith.constant(16, type=T.i32)

                    def rearrange_a_scale(raw_i32):
                        """Rearrange scale bytes for pack_M=1: extract m_half's k0,k1 bytes."""
                        if const_expr(pack_M >= scale_mn_pack):
                            return raw_i32
                        b_k0 = arith.andi(arith.shrui(raw_i32, scale_shift), scale_mask_lo)
                        b_k1 = arith.andi(
                            arith.shrui(raw_i32, scale_shift_hi), scale_mask_lo
                        )
                        return arith.ori(
                            b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32))
                        )

                    def rearrange_b_scale(raw_i32):
                        """Rearrange scale bytes for pack_N=1: extract n_half's k0,k1 bytes."""
                        if const_expr(pack_N >= scale_mn_pack):
                            return raw_i32
                        b_k0 = arith.andi(arith.shrui(raw_i32, bscale_shift), scale_mask_lo)
                        b_k1 = arith.andi(
                            arith.shrui(raw_i32, bscale_shift_hi), scale_mask_lo
                        )
                        return arith.ori(
                            b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32))
                        )

                    if const_expr(a_scale_one):
                        as1_const = arith.constant(0x7F7F7F7F, type=T.i32)
                        as1_vec = vector.from_elements(T.vec(1, T.i32), [as1_const])

                    def prefetch_ab_scale_tile(base_k, ku_packed_limit=k_unroll_packed):
                        a_scale_tile = []
                        gate_b_scale = []
                        up_b_scale = (
                            [] if (not mock_gate_only and not gate_up_interleave) else None
                        )
                        for ku in range_constexpr(ku_packed_limit):
                            k_off = (ku + base_k) * layout_b_scale.stride_k0
                            for mi in range_constexpr(m_repeat_packed):
                                if const_expr(a_scale_one):
                                    a_scale_tile.append(as1_vec)
                                else:
                                    s = buffer_ops.buffer_load(
                                        sx_rsrc,
                                        a_scale_bases[mi] + k_off,
                                        vec_width=1,
                                        dtype=T.i32,
                                        cache_modifier=0,
                                    )
                                    s = rearrange_a_scale(s)
                                    a_scale_tile.append(
                                        vector.from_elements(T.vec(1, T.i32), [s])
                                    )
                            for ni in range_constexpr(num_acc_n_packed):
                                gs = buffer_ops.buffer_load(
                                    weight_scale_rsrc,
                                    gate_scale_bases[ni] + k_off,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=0,
                                )
                                gs = rearrange_b_scale(gs)
                                gate_b_scale.append(
                                    vector.from_elements(T.vec(1, T.i32), [gs])
                                )
                                if const_expr(
                                    not mock_gate_only and not gate_up_interleave
                                ):
                                    us = buffer_ops.buffer_load(
                                        weight_scale_rsrc,
                                        up_scale_bases[ni] + k_off,
                                        vec_width=1,
                                        dtype=T.i32,
                                        cache_modifier=0,
                                    )
                                    us = rearrange_b_scale(us)
                                    up_b_scale.append(
                                        vector.from_elements(T.vec(1, T.i32), [us])
                                    )
                        return [a_scale_tile, gate_b_scale, up_b_scale]

                    lds_base_zero = arith.index(0)

                    def store_x_tile_to_lds(vec_x_in_parts, lds_buffer):
                        for i in range_constexpr(num_x_loads):
                            row_local = x_row_local[i]
                            col_local_i32 = x_col_local_i32[i]
                            if const_expr(x_load_bytes == 16):
                                lds_store_16b_xor16(
                                    arith,
                                    vector,
                                    lds_memref=lds_buffer,
                                    vec16_ty=vec16_x,
                                    layout_lds=layout_lds,
                                    row_local=row_local,
                                    col_local_i32=col_local_i32,
                                    tx_c4=arith.index(4),
                                    k_blocks16=k_blocks16,
                                    lds_base=lds_base_zero,
                                    vec_part_i32x4=vec_x_in_parts[i],
                                    elem_bytes=elem_bytes,
                                )

                    if const_expr(use_async_copy):
                        dma_bytes = 16
                        wave_size = 64
                        eff_bytes_per_buffer = (
                            int(tile_m) * int(eff_lds_stride) * int(a_elem_bytes)
                        )
                        # A-LDS buffer with a_load_threads (== total_threads at k_wave=1).
                        num_dma_loads = max(
                            1, eff_bytes_per_buffer // (a_load_threads * dma_bytes)
                        )

                        def dma_x_tile_to_lds(base_k, lds_buffer):
                            c4_idx = arith.index(4)
                            base_k_div4 = (
                                (base_k // c_a_pack)
                                * arith.constant(int(elem_bytes), index=True)
                            ) // arith.index(4)

                            lds_ptr_i64 = None
                            for i in range_constexpr(num_dma_loads):
                                row_local_i = x_row_local[i]
                                col_local_i32_i = x_col_local_i32[i]
                                col_local_sw = swizzle_xor16(
                                    row_local_i, col_local_i32_i * c4_idx, k_blocks16
                                )
                                row_k_dw = x_row_base_div4[i] + base_k_div4
                                global_byte_idx = row_k_dw * c4_idx + col_local_sw
                                global_offset = arith.index_cast(T.i32, global_byte_idx)

                                if const_expr(i == 0):
                                    lds_addr = memref.extract_aligned_pointer_as_index(
                                        lds_buffer
                                    ) + wave_n_id * arith.constant(
                                        wave_size * dma_bytes, index=True
                                    )
                                    lds_ptr_i64 = rocdl.readfirstlane(
                                        T.i64, arith.index_cast(T.i64, lds_addr)
                                    )
                                else:
                                    lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                        a_load_threads * dma_bytes, type=T.i64
                                    )

                                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                                rocdl.raw_ptr_buffer_load_lds(
                                    x_rsrc,
                                    lds_ptr,
                                    arith.constant(dma_bytes, type=T.i32),
                                    global_offset,
                                    arith.constant(0, type=T.i32),
                                    arith.constant(0, type=T.i32),
                                    arith.constant(0, type=T.i32),
                                )

                        def prefetch_x_to_lds(base_k, lds_buffer):
                            dma_x_tile_to_lds(base_k, lds_buffer)

                    def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                        col_base_swz_bytes = swizzle_xor16(
                            curr_row_a_lds, col_base, k_blocks16
                        )
                        col_base_swz = (
                            col_base_swz_bytes
                            if elem_bytes == 1
                            else (col_base_swz_bytes // arith.index(2))
                        )
                        idx_a16 = crd2idx(
                            [fx.Int32(curr_row_a_lds), fx.Int32(col_base_swz)], layout_lds
                        )
                        loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                        a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                        a0 = vector.extract(
                            a_i64x2, static_position=[0], dynamic_position=[]
                        )
                        a1 = vector.extract(
                            a_i64x2, static_position=[1], dynamic_position=[]
                        )
                        return a0, a1

                    def prefetch_full_a_from_lds(lds_buffer, ku_limit=k_unroll):
                        """Load entire A tile from LDS into registers before compute."""
                        a_regs = []
                        for k_idx in range_constexpr(ku_limit):
                            col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack
                            for mi_idx in range_constexpr(m_repeat):
                                mi_val = arith.constant(mi_idx * 16, index=True)
                                curr_row = row_a_lds + mi_val
                                a0, a1 = lds_load_packs_k64(curr_row, col_base, lds_buffer)
                                if const_expr(is_f8_a):
                                    a2, a3 = lds_load_packs_k64(
                                        curr_row, col_base + 64, lds_buffer
                                    )
                                    a_regs.append((a0, a1, a2, a3))
                                else:
                                    a_regs.append((a0, a1))
                        return a_regs

                    def compute_tile(
                        acc_gate_in,
                        acc_up_in,
                        gate_b_tile_in,
                        up_b_tile_in,
                        a_tile_regs,
                        a_scale=None,
                        gate_b_scale=None,
                        up_b_scale=None,
                        *,
                        prefetch_epilogue=False,
                        ku_count=k_unroll,
                    ):
                        gate_list = list(acc_gate_in)
                        single_b = mock_gate_only or gate_up_interleave
                        up_list = None if single_b else list(acc_up_in)
                        epilogue_pf = None
                        bias_pf = None
                        if const_expr(prefetch_epilogue):
                            if const_expr(enable_bias):
                                if const_expr(gate_up_interleave):
                                    bias_pf = []
                                    for ni in range_constexpr(num_acc_n):
                                        logical_col = (
                                            (by_n + n_tile_base)
                                            // arith.constant(2, index=True)
                                            + arith.constant((ni // 2) * 16, index=True)
                                            + lane_mod_16
                                        )
                                        up_off = (
                                            inter_idx
                                            if (ni % 2 == 1)
                                            else arith.constant(0, index=True)
                                        )
                                        bias_offset = expert_off_idx + up_off + logical_col
                                        bias_pf.append(
                                            load_bias_scalar(bias_rsrc, bias_offset)
                                        )
                                else:
                                    gate_bias_pf = []
                                    up_bias_pf = (
                                        [] if const_expr(not mock_gate_only) else None
                                    )
                                    for ni in range_constexpr(num_acc_n):
                                        global_n = (
                                            by_n
                                            + n_tile_base
                                            + arith.constant(ni * 16, index=True)
                                            + lane_mod_16
                                        )
                                        gate_bias_pf.append(
                                            load_bias_scalar(
                                                bias_rsrc, expert_off_idx + global_n
                                            )
                                        )
                                        if const_expr(not mock_gate_only):
                                            up_bias_pf.append(
                                                load_bias_scalar(
                                                    bias_rsrc,
                                                    expert_off_idx + inter_idx + global_n,
                                                )
                                            )
                                    bias_pf = (gate_bias_pf, up_bias_pf)
                            tw_pf = None
                            if const_expr(doweight_stage1):
                                tw_pf = []
                                lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                                ii_idx_list_pf = [
                                    arith.constant(ii, index=True) for ii in range(4)
                                ]
                                for mi in range_constexpr(m_repeat):
                                    mi_base_pf = arith.constant(mi * 16, index=True)
                                    for ii in range_constexpr(4):
                                        row_off_pf = (
                                            lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                        )
                                        sorted_row_pf = bx_m + mi_base_pf + row_off_pf
                                        tw_pf.append(
                                            buffer_ops.buffer_load(
                                                sorted_w_rsrc,
                                                sorted_row_pf,
                                                vec_width=1,
                                                dtype=f32,
                                            )
                                        )
                            epilogue_pf = (None, tw_pf, bias_pf)

                        c0_i64 = arith.constant(0, type=T.i64)
                        vec4_i64 = T.vec(4, T.i64)
                        vec8_i32 = T.vec(8, T.i32)

                        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                            v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                            return vector.bitcast(vec8_i32, v4)

                        eff_packed = (ku_count + pack_K - 1) // pack_K
                        for ku128 in range_constexpr(eff_packed):
                            for ni in range_constexpr(num_acc_n_packed):
                                gate_bs_i32 = gate_b_scale[ku128 * num_acc_n_packed + ni]
                                gate_bs_val = vector.extract(
                                    gate_bs_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                                if const_expr(not single_b):
                                    up_bs_i32 = up_b_scale[ku128 * num_acc_n_packed + ni]
                                    up_bs_val = vector.extract(
                                        up_bs_i32, static_position=[0], dynamic_position=[]
                                    )
                                for ikxdl in range_constexpr(pack_K):
                                    k_idx = ku128 * pack_K + ikxdl
                                    if const_expr(k_idx < ku_count):
                                        gate_bp = gate_b_tile_in[k_idx]
                                        if const_expr(not single_b):
                                            up_bp = up_b_tile_in[k_idx]
                                        for inxdl in range_constexpr(pack_N):
                                            ni_idx = ni * pack_N + inxdl
                                            if const_expr(body_b_has_full_operand):
                                                gb128 = pack_i64x4_to_i32x8(
                                                    gate_bp[0][ni_idx],
                                                    gate_bp[1][ni_idx],
                                                    gate_bp[2][ni_idx],
                                                    gate_bp[3][ni_idx],
                                                )
                                            else:
                                                gb128 = pack_i64x4_to_i32x8(
                                                    gate_bp[0][ni_idx],
                                                    gate_bp[1][ni_idx],
                                                    c0_i64,
                                                    c0_i64,
                                                )
                                            if const_expr(not single_b):
                                                if const_expr(body_b_has_full_operand):
                                                    ub128 = pack_i64x4_to_i32x8(
                                                        up_bp[0][ni_idx],
                                                        up_bp[1][ni_idx],
                                                        up_bp[2][ni_idx],
                                                        up_bp[3][ni_idx],
                                                    )
                                                else:
                                                    ub128 = pack_i64x4_to_i32x8(
                                                        up_bp[0][ni_idx],
                                                        up_bp[1][ni_idx],
                                                        c0_i64,
                                                        c0_i64,
                                                    )
                                            for mi in range_constexpr(m_repeat_packed):
                                                a_scale_i32 = a_scale[
                                                    ku128 * m_repeat_packed + mi
                                                ]
                                                a_scale_val = vector.extract(
                                                    a_scale_i32,
                                                    static_position=[0],
                                                    dynamic_position=[],
                                                )
                                                for imxdl in range_constexpr(pack_M):
                                                    mi_idx = mi * pack_M + imxdl
                                                    a_reg_idx = k_idx * m_repeat + mi_idx
                                                    if const_expr(is_f8_a):
                                                        a0, a1, a2, a3 = a_tile_regs[
                                                            a_reg_idx
                                                        ]
                                                        a128 = pack_i64x4_to_i32x8(
                                                            a0, a1, a2, a3
                                                        )
                                                    else:
                                                        a0, a1 = a_tile_regs[a_reg_idx]
                                                        a128 = pack_i64x4_to_i32x8(
                                                            a0, a1, c0_i64, c0_i64
                                                        )
                                                    acc_idx = mi_idx * num_acc_n + ni_idx
                                                    gate_list[acc_idx] = mixed_b_mfma(
                                                        a128,
                                                        gb128,
                                                        gate_list[acc_idx],
                                                        ikxdl * pack_M + imxdl,
                                                        a_scale_val,
                                                        ikxdl * pack_N + inxdl,
                                                        gate_bs_val,
                                                    )
                                                    if const_expr(not single_b):
                                                        up_list[acc_idx] = mixed_b_mfma(
                                                            a128,
                                                            ub128,
                                                            up_list[acc_idx],
                                                            ikxdl * pack_M + imxdl,
                                                            a_scale_val,
                                                            ikxdl * pack_N + inxdl,
                                                            up_bs_val,
                                                        )
                        return gate_list, up_list, epilogue_pf

                    def load_a_subtile(k_idx, mi_idx, lds_buffer):
                        """Load a single A sub-tile from LDS (one ds_read)."""
                        col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack
                        mi_val = arith.constant(mi_idx * 16, index=True)
                        curr_row = row_a_lds + mi_val
                        a0, a1 = lds_load_packs_k64(curr_row, col_base, lds_buffer)
                        if const_expr(is_f8_a):
                            a2, a3 = lds_load_packs_k64(curr_row, col_base + 64, lds_buffer)
                            return (a0, a1, a2, a3)
                        else:
                            return (a0, a1)

                    single_b_pipe = mock_gate_only or gate_up_interleave

                    def compute_bmajor_mfma_phase(
                        all_a_tiles,
                        gate_b_single,
                        up_b_single,
                        a_scale_vals,
                        gate_bs_val,
                        up_bs_val,
                        gate_list,
                        up_list,
                        k_idx,
                        ni_idx,
                        ikxdl,
                        inxdl,
                    ):
                        """B-major MFMA: fix one B (ni), cycle all A tiles (mi).

                        Packs B once and reuses across all mi iterations.
                        A tiles come from LDS (already available, no VMEM wait).

                        all_a_tiles: flat list indexed by [k*m_repeat + mi].
                        gate_b_single/up_b_single: (b0,b1) fp4 or (b0,b1,b2,b3) fp8 for
                          one ni; up is None under _single_b_pipe.
                        a_scale_vals: list of A scale scalars indexed by mi_packed.
                        """
                        c0_i64 = arith.constant(0, type=T.i64)
                        vec4_i64 = T.vec(4, T.i64)
                        vec8_i32 = T.vec(8, T.i32)

                        def pack(x0, x1, x2, x3):
                            v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                            return vector.bitcast(vec8_i32, v4)

                        def pack_b(b_single):
                            if const_expr(body_b_has_full_operand):
                                return pack(
                                    b_single[0], b_single[1], b_single[2], b_single[3]
                                )
                            return pack(b_single[0], b_single[1], c0_i64, c0_i64)

                        gb128 = pack_b(gate_b_single)
                        if const_expr(not single_b_pipe):
                            ub128 = pack_b(up_b_single)

                        for mi_p in range_constexpr(m_repeat_packed):
                            a_scale_val = a_scale_vals[mi_p]
                            for imxdl in range_constexpr(pack_M):
                                mi_idx = mi_p * pack_M + imxdl
                                a_reg = all_a_tiles[k_idx * m_repeat + mi_idx]

                                if const_expr(is_f8_a):
                                    a128 = pack(a_reg[0], a_reg[1], a_reg[2], a_reg[3])
                                else:
                                    a128 = pack(a_reg[0], a_reg[1], c0_i64, c0_i64)

                                acc_idx = mi_idx * num_acc_n + ni_idx
                                gate_list[acc_idx] = mixed_b_mfma(
                                    a128,
                                    gb128,
                                    gate_list[acc_idx],
                                    ikxdl * pack_M + imxdl,
                                    a_scale_val,
                                    ikxdl * pack_N + inxdl,
                                    gate_bs_val,
                                )
                                if const_expr(not single_b_pipe):
                                    up_list[acc_idx] = mixed_b_mfma(
                                        a128,
                                        ub128,
                                        up_list[acc_idx],
                                        ikxdl * pack_M + imxdl,
                                        a_scale_val,
                                        ikxdl * pack_N + inxdl,
                                        up_bs_val,
                                    )

                    def interleaved_half(
                        lds_read,
                        lds_write,
                        next_k_dma_py,
                        next_k_load,
                        prev_a_tile,
                        prev_gate_w,
                        prev_up_w,
                        prev_a_scale,
                        prev_gate_bs,
                        prev_up_bs,
                        acc_gate,
                        acc_up,
                    ):
                        """One flatmm-style interleaved half-iteration (deep pipeline).

                        Generalized for arbitrary m_repeat (block_m=32, 64, ...).
                        DMA targets lds_write (OTHER buffer) while ds_read uses
                        lds_read (already DMA'd in previous half).

                        Interleaving schedule (per half):
                          Phase 0: scale VMEM + 2 ds_read(A) -> 4 MFMA(prev)
                          Phase 1..N: B VMEM(distributed) + 2 ds_read(A, if avail) -> 4 MFMA(prev)
                          Phase N+1..: remaining B VMEM -> 4 MFMA(prev)
                        """
                        abs_k = body_k_base_idx + arith.constant(next_k_load, index=True)
                        bk = abs_k // arith.constant(b_byte_div, index=True)
                        sk = abs_k // arith.constant(pack_K * 128, index=True)
                        k_off = sk * layout_b_scale.stride_k0

                        rocdl.sched_barrier(0)
                        if const_expr(heterogeneous_b and use_async_copy):
                            barrier(vmcnt=0)
                        else:
                            rocdl.s_waitcnt(body_vmcnt_before_barrier)
                            barrier()
                        rocdl.sched_barrier(0)

                        abs_k_dma = body_k_base_idx + arith.constant(
                            next_k_dma_py, index=True
                        )
                        if const_expr(use_async_copy and next_k_dma_py < int(k_dim)):
                            prefetch_x_to_lds(abs_k_dma, lds_write)
                        if const_expr(not use_async_copy):
                            x_regs = load_x_tile(abs_k_dma)

                        prev_asvs = []
                        for i_as in range_constexpr(len(prev_a_scale)):
                            prev_asvs.append(
                                vector.extract(
                                    prev_a_scale[i_as],
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                            )
                        prev_gsv_list = []
                        for i_gs in range_constexpr(len(prev_gate_bs)):
                            prev_gsv_list.append(
                                vector.extract(
                                    prev_gate_bs[i_gs],
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                            )
                        if const_expr(not single_b_pipe):
                            prev_usv_list = []
                            for i_us in range_constexpr(len(prev_up_bs)):
                                prev_usv_list.append(
                                    vector.extract(
                                        prev_up_bs[i_us],
                                        static_position=[0],
                                        dynamic_position=[],
                                    )
                                )

                        a_all = {}
                        b_gate_all = {}
                        b_up_all = {}

                        for _p in range_constexpr(pipe_n_phases):
                            if const_expr(pp_has_scale[_p]):
                                new_as_list = []
                                new_gs_list = []
                                new_us_list = [] if const_expr(not single_b_pipe) else None
                                for ku_s in range_constexpr(k_unroll_packed):
                                    ku_off = (
                                        k_off
                                        + arith.constant(ku_s, index=True)
                                        * layout_b_scale.stride_k0
                                    )
                                    for mi_p in range_constexpr(m_repeat_packed):
                                        if const_expr(a_scale_one):
                                            new_as_list.append(as1_const)
                                        else:
                                            raw_as = buffer_ops.buffer_load(
                                                sx_rsrc,
                                                a_scale_bases[mi_p] + ku_off,
                                                vec_width=1,
                                                dtype=T.i32,
                                                cache_modifier=0,
                                            )
                                            new_as_list.append(rearrange_a_scale(raw_as))
                                    for gs_ni in range_constexpr(num_acc_n_packed):
                                        gs_raw = buffer_ops.buffer_load(
                                            weight_scale_rsrc,
                                            gate_scale_bases[gs_ni] + ku_off,
                                            vec_width=1,
                                            dtype=T.i32,
                                            cache_modifier=0,
                                        )
                                        new_gs_list.append(rearrange_b_scale(gs_raw))
                                    if const_expr(not single_b_pipe):
                                        for us_ni in range_constexpr(num_acc_n_packed):
                                            us_raw = buffer_ops.buffer_load(
                                                weight_scale_rsrc,
                                                up_scale_bases[us_ni] + ku_off,
                                                vec_width=1,
                                                dtype=T.i32,
                                                cache_modifier=0,
                                            )
                                            new_us_list.append(rearrange_b_scale(us_raw))

                            for b_j in range_constexpr(len(pp_b_loads[_p])):
                                b_type, b_ku, b_ni = pp_b_loads[_p][b_j]
                                if const_expr(b_type == "gate"):
                                    b_gate_all[(b_ku, b_ni)] = load_b_packs_k64(
                                        bk,
                                        b_ku,
                                        gate_n_blk_list[b_ni],
                                        gate_n_intra_list[b_ni],
                                    )
                                else:
                                    b_up_all[(b_ku, b_ni)] = load_b_packs_k64(
                                        bk,
                                        b_ku,
                                        up_n_blk_list[b_ni],
                                        up_n_intra_list[b_ni],
                                    )

                            rocdl.sched_barrier(0)
                            for a_j in range_constexpr(len(pp_a_reads[_p])):
                                ak, ami = pp_a_reads[_p][a_j]
                                a_all[(ak, ami)] = load_a_subtile(
                                    ak,
                                    ami,
                                    lds_read,
                                )
                            rocdl.sched_barrier(0)

                            rocdl.s_setprio(1)
                            for m_j in range_constexpr(len(pp_mfma[_p])):
                                k_idx, ni_idx, ikxdl, inxdl, ku128 = pp_mfma[_p][m_j]
                                ni_packed_idx = ni_idx // pack_N

                                def mk_single(tile_entry, ni):
                                    if const_expr(body_b_has_full_operand):
                                        return (
                                            tile_entry[0][ni],
                                            tile_entry[1][ni],
                                            tile_entry[2][ni],
                                            tile_entry[3][ni],
                                        )
                                    return (tile_entry[0][ni], tile_entry[1][ni])

                                up_b_single = (
                                    mk_single(prev_up_w[k_idx], ni_idx)
                                    if not single_b_pipe
                                    else None
                                )
                                as_off = ku128 * m_repeat_packed
                                bs_idx = ku128 * num_acc_n_packed + ni_packed_idx
                                compute_bmajor_mfma_phase(
                                    prev_a_tile,
                                    mk_single(prev_gate_w[k_idx], ni_idx),
                                    up_b_single,
                                    prev_asvs[as_off : as_off + m_repeat_packed],
                                    prev_gsv_list[bs_idx],
                                    (prev_usv_list[bs_idx] if not single_b_pipe else None),
                                    acc_gate,
                                    acc_up,
                                    k_idx,
                                    ni_idx,
                                    ikxdl,
                                    inxdl,
                                )
                            rocdl.s_setprio(0)
                            rocdl.sched_barrier(0)

                        cur_a_tile = []
                        for k in range_constexpr(k_unroll):
                            for mi in range_constexpr(m_repeat):
                                cur_a_tile.append(a_all[(k, mi)])

                        cur_gate_w = []
                        cur_up_w = None if single_b_pipe else []
                        for ku in range_constexpr(k_unroll):
                            g_packs0, g_packs1, g_packs2, g_packs3 = [], [], [], []
                            u_packs0, u_packs1, u_packs2, u_packs3 = [], [], [], []
                            for ni in range_constexpr(num_acc_n):
                                g = b_gate_all[(ku, ni)]
                                g_packs0.append(g[0])
                                g_packs1.append(g[1])
                                if const_expr(body_b_has_full_operand):
                                    g_packs2.append(g[2])
                                    g_packs3.append(g[3])
                                if const_expr(not single_b_pipe):
                                    u = b_up_all[(ku, ni)]
                                    u_packs0.append(u[0])
                                    u_packs1.append(u[1])
                                    if const_expr(body_b_has_full_operand):
                                        u_packs2.append(u[2])
                                        u_packs3.append(u[3])
                            cur_gate_w.append((g_packs0, g_packs1, g_packs2, g_packs3))
                            if const_expr(not single_b_pipe):
                                cur_up_w.append((u_packs0, u_packs1, u_packs2, u_packs3))

                        cur_a_scale = []
                        for i_as in range_constexpr(len(new_as_list)):
                            cur_a_scale.append(
                                vector.from_elements(
                                    T.vec(1, T.i32),
                                    [new_as_list[i_as]],
                                )
                            )
                        cur_gate_bs = []
                        for i_gs in range_constexpr(len(new_gs_list)):
                            cur_gate_bs.append(
                                vector.from_elements(T.vec(1, T.i32), [new_gs_list[i_gs]])
                            )
                        if const_expr(not single_b_pipe):
                            cur_up_bs = []
                            for i_us in range_constexpr(len(new_us_list)):
                                cur_up_bs.append(
                                    vector.from_elements(
                                        T.vec(1, T.i32), [new_us_list[i_us]]
                                    )
                                )
                        else:
                            cur_up_bs = None

                        if const_expr(not use_async_copy):
                            store_x_tile_to_lds(x_regs, lds_write)

                        return (
                            cur_a_tile,
                            cur_gate_w,
                            cur_up_w,
                            cur_a_scale,
                            cur_gate_bs,
                            cur_up_bs,
                            acc_gate,
                            acc_up,
                        )

                    rocdl.sched_barrier(0)

                    k0 = body_k_base_idx
                    if const_expr(use_async_copy):
                        prefetch_x_to_lds(k0, body_lds_x_pong)
                    else:
                        x_regs0 = load_x_tile(k0)
                        store_x_tile_to_lds(x_regs0, body_lds_x_pong)
                    rocdl.sched_barrier(0)
                    k0_scale = body_k_base_idx // arith.constant(pack_K * 128, index=True)
                    a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(
                        k0_scale
                    )
                    c_tile_m_idx = arith.constant(tile_m, index=True)
                    tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, c_tile_m_idx)
                    if_tid = scf.IfOp(tid_in_range)
                    with ir.InsertionPoint(if_tid.then_block):
                        tid_row = bx_m + tx
                        tid_val = buffer_ops.buffer_load(
                            sorted_rsrc, tid_row, vec_width=1, dtype=T.i32
                        )
                        tid_vec1 = vector.from_elements(T.vec(1, T.i32), [tid_val])
                        vector.store(tid_vec1, lds_tid, [tx])
                        scf.YieldOp([])

                    acc_gate = [acc_init] * num_acc_n * m_repeat
                    acc_up = (
                        [acc_init] * num_acc_n * m_repeat if not single_b_pipe else None
                    )

                    k1 = body_k_base_idx + arith.constant(tile_k, index=True)
                    rocdl.sched_barrier(0)
                    if const_expr(use_async_copy):
                        prefetch_x_to_lds(k1, body_lds_x_ping)
                    else:
                        x_regs_prime = load_x_tile(k1)
                        store_x_tile_to_lds(x_regs_prime, body_lds_x_ping)

                    k0_b = body_k_base_idx // arith.constant(b_byte_div, index=True)
                    gate_w0, up_w0 = load_b_tile(k0_b)
                    if const_expr(use_async_copy):
                        rocdl.s_waitcnt(0)
                    gpu.barrier()
                    rocdl.sched_barrier(0)
                    a_tile_pong = prefetch_full_a_from_lds(body_lds_x_pong)

                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(6)

                    num_k_tiles_py = int(klen) // int(tile_k)
                    odd_k_tiles = (num_k_tiles_py % 2) == 1
                    tail_tiles = 1 if odd_k_tiles else 2
                    k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                    if const_expr(k_main2_py < 0):
                        k_main2_py = 0

                    gate_w_pong = gate_w0
                    up_w_pong = up_w0

                    rocdl.sched_barrier(0)

                    if const_expr(k_main2_py > 0):
                        for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                            next_k_load_1 = k_iv_py + tile_k
                            next_k_load_2 = k_iv_py + tile_k * 2
                            next_k_dma_1 = k_iv_py + tile_k * 2
                            next_k_dma_2 = k_iv_py + tile_k * 3

                            (
                                a_tile_ping,
                                gate_w_ping,
                                up_w_ping,
                                a_scale_ping,
                                gate_bs_ping,
                                up_bs_ping,
                                acc_gate,
                                acc_up,
                            ) = interleaved_half(
                                body_lds_x_ping,
                                body_lds_x_pong,
                                next_k_dma_1,
                                next_k_load_1,
                                a_tile_pong,
                                gate_w_pong,
                                up_w_pong,
                                a_scale_pong,
                                gate_bs_pong,
                                up_bs_pong,
                                acc_gate,
                                acc_up,
                            )

                            (
                                a_tile_pong,
                                gate_w_pong,
                                up_w_pong,
                                a_scale_pong,
                                gate_bs_pong,
                                up_bs_pong,
                                acc_gate,
                                acc_up,
                            ) = interleaved_half(
                                body_lds_x_pong,
                                body_lds_x_ping,
                                next_k_dma_2,
                                next_k_load_2,
                                a_tile_ping,
                                gate_w_ping,
                                up_w_ping,
                                a_scale_ping,
                                gate_bs_ping,
                                up_bs_ping,
                                acc_gate,
                                acc_up,
                            )

                    if const_expr(odd_k_tiles):
                        acc_gate, acc_up, epilogue_pf = compute_tile(
                            acc_gate,
                            acc_up,
                            gate_w_pong,
                            up_w_pong,
                            a_tile_pong,
                            a_scale_pong,
                            gate_bs_pong,
                            up_bs_pong,
                            prefetch_epilogue=True,
                            ku_count=tail_ku if pad_ku_skip > 0 else k_unroll,
                        )
                    else:
                        k_tail_rel = arith.constant(klen - tile_k, index=True)
                        k_tail1 = body_k_base_idx + k_tail_rel
                        x_regs_ping = []
                        if const_expr(use_async_copy):
                            prefetch_x_to_lds(k_tail1, body_lds_x_ping)
                        else:
                            x_regs_ping = load_x_tile(k_tail1)
                        if const_expr(pad_ku_skip > 0):
                            gate_w_ping, up_w_ping = load_b_tile(
                                k_tail1 // arith.constant(b_byte_div, index=True),
                                ku_limit=tail_ku,
                            )
                            (
                                a_scale_ping,
                                gate_bs_ping,
                                up_bs_ping,
                            ) = prefetch_ab_scale_tile(
                                k_tail1 // arith.constant(pack_K * 128, index=True),
                                ku_packed_limit=tail_ku_packed,
                            )
                        else:
                            gate_w_ping, up_w_ping = load_b_tile(
                                k_tail1 // arith.constant(b_byte_div, index=True)
                            )
                            (
                                a_scale_ping,
                                gate_bs_ping,
                                up_bs_ping,
                            ) = prefetch_ab_scale_tile(
                                k_tail1 // arith.constant(pack_K * 128, index=True)
                            )
                        acc_gate, acc_up, _ = compute_tile(
                            acc_gate,
                            acc_up,
                            gate_w_pong,
                            up_w_pong,
                            a_tile_pong,
                            a_scale_pong,
                            gate_bs_pong,
                            up_bs_pong,
                        )
                        if const_expr(not use_async_copy):
                            store_x_tile_to_lds(x_regs_ping, body_lds_x_ping)
                        rocdl.s_waitcnt(0)
                        barrier()
                        if const_expr(pad_ku_skip > 0):
                            a_tile_ping = prefetch_full_a_from_lds(
                                body_lds_x_ping, ku_limit=tail_ku
                            )
                        else:
                            a_tile_ping = prefetch_full_a_from_lds(body_lds_x_ping)
                        acc_gate, acc_up, epilogue_pf = compute_tile(
                            acc_gate,
                            acc_up,
                            gate_w_ping,
                            up_w_ping,
                            a_tile_ping,
                            a_scale_ping,
                            gate_bs_ping,
                            up_bs_ping,
                            prefetch_epilogue=True,
                            ku_count=tail_ku if pad_ku_skip > 0 else k_unroll,
                        )

                    bias_pf = None
                    if const_expr(epilogue_pf is not None):
                        _, _, bias_pf = epilogue_pf

                    def sigmoid_elem(g):
                        neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                        t = g * neg_log2e
                        emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                        one = arith.constant(1.0, type=f32)
                        den = one + emu
                        return llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [den], [], []
                        )

                    def silu_elem(g):
                        """silu(x) = x * sigmoid(x); HW fast path: exp2, rcp"""
                        return g * sigmoid_elem(g)

                    def tanh_elem(x):
                        """tanh(x), expanded via exp2 like preshuffle_gemm's GeLU path."""
                        zero = arith.constant(0.0, type=f32)
                        one = arith.constant(1.0, type=f32)
                        neg_two_log2e = arith.constant(-2.8853900817779268, type=f32)
                        abs_x = x.maximumf(-x)
                        e = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.exp2.f32", [abs_x * neg_two_log2e], [], []
                        )
                        den = one + e
                        recip = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [den], [], []
                        )
                        tanh_abs = (one - e) * recip
                        is_pos = x > zero
                        return is_pos.select(tanh_abs, -tanh_abs)

                    def situ_elem(g):
                        """situ(x) = beta * tanh(x / beta) * sigmoid(x)"""
                        return (
                            f32_situ_beta
                            * tanh_elem(g * f32_situ_beta_rcp)
                            * sigmoid_elem(g)
                        )

                    def situ_up_elem(u):
                        """linear_beta * tanh(up / linear_beta)."""
                        return f32_situ_linear_beta * tanh_elem(
                            u * f32_situ_linear_beta_rcp
                        )

                    def _clamp_gate(x):
                        # min(x, lim) == -max(-x, -lim); upper bound only.
                        return -((-x).maximumf(swiglu_neg_limit))

                    def _clamp_lin(x):
                        # clamp to [-lim, lim].
                        return (-((-x).maximumf(swiglu_neg_limit))).maximumf(
                            swiglu_neg_limit
                        )

                    def silu_mul_vec4(gate_v4, up_v4):
                        """Element-wise silu(gate) * up on vec4_f32.
                        Clamp gate <= limit and -limit <= up <= limit (runtime limit;
                        +inf disables the clamp) before applying silu(gate) * up.
                        """
                        result_elems = []
                        for ei in range_constexpr(4):
                            g = vector.extract(
                                gate_v4, static_position=[ei], dynamic_position=[]
                            )
                            u = vector.extract(
                                up_v4, static_position=[ei], dynamic_position=[]
                            )
                            g = _clamp_gate(g)
                            u = _clamp_lin(u)
                            result_elems.append(silu_elem(g) * u)
                        return vector.from_elements(vec4_f32, result_elems)

                    def swiglu_mul_vec4(gate_v4, up_v4):
                        """Element-wise swiglu(gate, up) on vec4_f32.
                        swiglu(g, u) = g * sigmoid(alpha * g) * (u + 1)
                        Clamp gate <= limit and -limit <= up <= limit (runtime limit,
                        7.0 default) before the activation.
                        """
                        result_elems = []
                        alpha = arith.constant(1.702, type=f32)
                        one = arith.constant(1.0, type=f32)
                        neg_log2e = arith.constant(-1.4426950408889634, type=f32)

                        for ei in range_constexpr(4):
                            g = vector.extract(
                                gate_v4, static_position=[ei], dynamic_position=[]
                            )
                            u = vector.extract(
                                up_v4, static_position=[ei], dynamic_position=[]
                            )
                            g = _clamp_gate(g)
                            u = _clamp_lin(u)
                            t = g * alpha * neg_log2e
                            emu = llvm.call_intrinsic(
                                f32, "llvm.amdgcn.exp2.f32", [t], [], []
                            )
                            den = one + emu
                            sig = llvm.call_intrinsic(
                                f32, "llvm.amdgcn.rcp.f32", [den], [], []
                            )
                            result_elems.append(g * sig * (u + one))
                        return vector.from_elements(vec4_f32, result_elems)

                    def situ_mul_vec4(gate_v4, up_v4):
                        """Element-wise situv2(gate, up) on vec4_f32."""
                        result_elems = []
                        for ei in range_constexpr(4):
                            g = vector.extract(
                                gate_v4, static_position=[ei], dynamic_position=[]
                            )
                            u = vector.extract(
                                up_v4, static_position=[ei], dynamic_position=[]
                            )
                            g = _clamp_gate(g)
                            u = _clamp_lin(u)
                            result_elems.append(situ_elem(g) * situ_up_elem(u))
                        return vector.from_elements(vec4_f32, result_elems)

                    def act_vec4(gate_v4, up_v4):
                        """Dispatch activation based on `act` parameter."""
                        if const_expr(shared_b and need_fp8):
                            vec4_bf16 = T.vec(4, T.bf16)
                            gate_v4 = arith.extf(
                                vec4_f32, arith.trunc_f(vec4_bf16, gate_v4)
                            )
                            up_v4 = arith.extf(vec4_f32, arith.trunc_f(vec4_bf16, up_v4))
                        if const_expr(act == "swiglu"):
                            result = swiglu_mul_vec4(gate_v4, up_v4)
                        elif const_expr(act == "situv2"):
                            result = situ_mul_vec4(gate_v4, up_v4)
                        else:
                            result = silu_mul_vec4(gate_v4, up_v4)
                        if const_expr(shared_b and need_fp8):
                            result = arith.extf(
                                vec4_f32, arith.trunc_f(T.vec(4, T.bf16), result)
                            )
                        return result

                    def act_elem(g, u):
                        """Scalar activation, byte-identical per-element to _act_vec4.
                        Used by the fused k-split epilogue (operates on summed f32
                        gate/up scalars in the CShuffle read phase)."""
                        if const_expr(shared_b and need_fp8):
                            g = arith.extf(f32, arith.trunc_f(T.bf16, g))
                            u = arith.extf(f32, arith.trunc_f(T.bf16, u))
                        if const_expr(act == "swiglu"):
                            alpha = arith.constant(1.702, type=f32)
                            one = arith.constant(1.0, type=f32)
                            neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                            g = _clamp_gate(g)
                            u = _clamp_lin(u)
                            t = g * alpha * neg_log2e
                            emu = llvm.call_intrinsic(
                                f32, "llvm.amdgcn.exp2.f32", [t], [], []
                            )
                            den = one + emu
                            sig = llvm.call_intrinsic(
                                f32, "llvm.amdgcn.rcp.f32", [den], [], []
                            )
                            result = g * sig * (u + one)
                        elif const_expr(act == "situv2"):
                            g = _clamp_gate(g)
                            u = _clamp_lin(u)
                            result = situ_elem(g) * situ_up_elem(u)
                        else:
                            g = _clamp_gate(g)
                            u = _clamp_lin(u)
                            result = silu_elem(g) * u
                        if const_expr(shared_b and need_fp8):
                            result = arith.extf(f32, arith.trunc_f(T.bf16, result))
                        return result

                    kwave_fused = const_expr(
                        k_wave > 1
                        and not enable_bias
                        and not is_splitk
                        and not gate_up_interleave
                        and need_quant
                    )

                    if const_expr(k_wave > 1 and not kwave_fused):
                        has_up = const_expr(acc_up is not None)
                        nm = num_acc_n * m_repeat
                        grp_stride = 64 * nm
                        scr_ty = _mT.memref(
                            num_waves_total * grp_stride * 4, f32, memory_space=_lds_space()
                        )
                        scr_g = memref.view(
                            scr_ty,
                            base_ptr_pong,
                            arith.constant(lds_pong_offset, index=True),
                            sizes=[],
                        )
                        if const_expr(has_up):
                            scr_u = memref.view(
                                scr_ty,
                                base_ptr_ping,
                                arith.constant(lds_ping_offset, index=True),
                                sizes=[],
                            )
                        c_gs = arith.constant(grp_stride, index=True)
                        c4 = arith.constant(4, index=True)
                        c64 = arith.constant(64, index=True)
                        my_base = wave_id * c_gs + lane_id
                        gpu.barrier()
                        for ai in range_constexpr(nm):
                            sidx = (my_base + arith.constant(ai, index=True) * c64) * c4
                            vector.store(acc_gate[ai], scr_g, [sidx], alignment=16)
                            if const_expr(has_up):
                                vector.store(acc_up[ai], scr_u, [sidx], alignment=16)
                        gpu.barrier()
                        for ai in range_constexpr(nm):
                            ai_off = arith.constant(ai, index=True) * c64 + lane_id
                            gvs = []
                            uvs = []
                            for g in range_constexpr(k_wave):
                                peer = (
                                    arith.constant(g * num_n_waves, index=True) + wave_n_id
                                )
                                pidx = (peer * c_gs + ai_off) * c4
                                gvs.append(vector.load_op(vec4_f32, scr_g, [pidx]))
                                if const_expr(has_up):
                                    uvs.append(vector.load_op(vec4_f32, scr_u, [pidx]))

                            sg = gvs[0]
                            for g in range_constexpr(1, k_wave):
                                sg = arith.addf(sg, gvs[g])
                            acc_gate[ai] = sg
                            if const_expr(has_up):
                                su = uvs[0]
                                for g in range_constexpr(1, k_wave):
                                    su = arith.addf(su, uvs[g])
                                acc_up[ai] = su
                        # No trailing barrier: CShuffle's leading barrier already gates

                    if const_expr(enable_bias and not is_splitk):
                        bias_up_vals = None
                        if const_expr(bias_pf is not None):
                            if const_expr(gate_up_interleave):
                                bias_gate_vals = bias_pf
                            else:
                                bias_gate_vals, bias_up_vals = bias_pf
                        else:
                            bias_gate_vals = []
                            for ni in range_constexpr(num_acc_n):
                                if const_expr(gate_up_interleave):
                                    logical_col = (
                                        (by_n + n_tile_base)
                                        // arith.constant(2, index=True)
                                        + arith.constant((ni // 2) * 16, index=True)
                                        + lane_mod_16
                                    )
                                    up_off = (
                                        inter_idx
                                        if (ni % 2 == 1)
                                        else arith.constant(0, index=True)
                                    )
                                    bias_off = expert_off_idx + up_off + logical_col
                                else:
                                    bn = (
                                        by_n
                                        + n_tile_base
                                        + arith.constant(ni * 16, index=True)
                                        + lane_mod_16
                                    )
                                    bias_off = expert_off_idx + bn
                                bias_gate_vals.append(load_bias_scalar(bias_rsrc, bias_off))
                            if const_expr(not (mock_gate_only or gate_up_interleave)):
                                bias_up_vals = []
                                for ni in range_constexpr(num_acc_n):
                                    bn = (
                                        by_n
                                        + n_tile_base
                                        + arith.constant(ni * 16, index=True)
                                        + lane_mod_16
                                    )
                                    bias_up_vals.append(
                                        load_bias_scalar(
                                            bias_rsrc, expert_off_idx + inter_idx + bn
                                        )
                                    )
                        for mi in range_constexpr(m_repeat):
                            for ni in range_constexpr(num_acc_n):
                                aidx = mi * num_acc_n + ni
                                bsplat = vector.from_elements(
                                    vec4_f32, [bias_gate_vals[ni]] * 4
                                )
                                acc_gate[aidx] = arith.addf(acc_gate[aidx], bsplat)

                        if const_expr(not (mock_gate_only or gate_up_interleave)):
                            for mi in range_constexpr(m_repeat):
                                for ni in range_constexpr(num_acc_n):
                                    aidx = mi * num_acc_n + ni
                                    bsplat = vector.from_elements(
                                        vec4_f32, [bias_up_vals[ni]] * 4
                                    )
                                    acc_up[aidx] = arith.addf(acc_up[aidx], bsplat)

                    if const_expr(gate_up_interleave and not is_splitk):
                        gui_out_n = num_acc_n // pack_N
                        acc = [None] * (gui_out_n * m_repeat)
                        for mi in range_constexpr(m_repeat):
                            for ni in range_constexpr(gui_out_n):
                                g_idx = mi * num_acc_n + ni * pack_N
                                u_idx = g_idx + 1
                                out_idx = mi * gui_out_n + ni
                                acc[out_idx] = act_vec4(acc_gate[g_idx], acc_gate[u_idx])
                    elif const_expr(not is_splitk and not kwave_fused):
                        acc = [None] * (int(num_acc_n) * int(m_repeat))
                        for mi in range_constexpr(m_repeat):
                            for ni in range_constexpr(num_acc_n):
                                aidx = mi * num_acc_n + ni
                                acc[aidx] = act_vec4(acc_gate[aidx], acc_up[aidx])

                    tw_pf = None
                    bias_pf = None
                    if const_expr(epilogue_pf is not None):
                        _, tw_pf, bias_pf = epilogue_pf

                    mask24_i32 = arith.constant(0xFFFFFF)
                    topk_i32_v = topk_i32
                    tokens_i32_v = tokens_i32

                    out_base_i64 = arith.index_cast(T.i64, fx.ptrtoint(arg_out))
                    out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

                    if const_expr(lds_out is None):
                        raise RuntimeError("CShuffle epilogue requires lds_out")

                    apply_weight = doweight_stage1 and not is_splitk

                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        if const_expr(apply_weight):
                            tw_idx = (mi * 4) + ii
                            if const_expr(tw_pf is not None):
                                tw = tw_pf[tw_idx]
                            else:
                                tw = buffer_ops.buffer_load(
                                    sorted_w_rsrc, row, vec_width=1, dtype=f32
                                )
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            if const_expr(apply_weight):
                                v = v * tw
                            if const_expr(need_quant):
                                lds_idx = row_base_lds + col_local
                                vec1_f32 = T.vec(1, f32)
                                v1 = vector.from_elements(vec1_f32, [v])
                                vector.store(v1, lds_out, [lds_idx], alignment=4)
                            else:
                                v_out = arith.trunc_f(out_elem(), v)
                                lds_idx = row_base_lds + col_local
                                vec1_out = T.vec(1, out_elem())
                                v1 = vector.from_elements(vec1_out, [v_out])
                                vector.store(v1, lds_out, [lds_idx], alignment=2)

                    out_row_stride = (
                        inter_dim * 2 * out_elem_bytes
                        if is_splitk
                        else (
                            inter_dim // 2
                            if need_fp4
                            else (inter_dim if need_fp8 else inter_dim * out_elem_bytes)
                        )
                    )

                    def precompute_row(*, row_local, row):
                        fused2 = memref.load(lds_tid, [row_local])
                        row_i32 = arith.index_cast(T.i32, row)
                        row_valid0 = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32_v)
                        s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                        row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                        t_idx = arith.index_cast(ir.IndexType.get(), t)
                        s_idx = arith.index_cast(ir.IndexType.get(), s)
                        ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                        if const_expr(v2_output_layout):
                            payload_row_idx = row
                        else:
                            payload_row_idx = ts_idx
                        row_byte_base = out_base_idx + payload_row_idx * arith.constant(
                            out_row_stride, index=True
                        )
                        return ((fused2, row_byte_base), row_valid)

                    def idx_to_llvm_ptr(idx_val, addr_space=1):
                        idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
                        i64_v = arith.index_cast(T.i64, idx_v)
                        i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
                        ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                        return llvm.inttoptr(ptr_ty, i64_raw)

                    e_vec = e_vec_s1
                    e_vec_sk = 2
                    cshuffle_nlane = min(32, tile_n // e_vec)
                    cshuffle_nlane_sk = min(32, tile_n // e_vec_sk)

                    c0_i32 = arith.constant(0, type=T.i32)
                    c1_i32 = arith.constant(1, type=T.i32)
                    c2_i32 = arith.constant(2, type=T.i32)
                    c3_i32 = arith.constant(3, type=T.i32)
                    c4_i32 = arith.constant(4, type=T.i32)
                    c5_i32 = arith.constant(5, type=T.i32)
                    c15_i32 = arith.constant(15, type=T.i32)
                    c22_i32 = arith.constant(22, type=T.i32)
                    c23_i32 = arith.constant(23, type=T.i32)
                    c28_i32 = arith.constant(28, type=T.i32)
                    c31_i32 = arith.constant(31, type=T.i32)
                    c32_i32 = arith.constant(32, type=T.i32)
                    c64_i32 = arith.constant(64, type=T.i32)
                    c254_i32 = arith.constant(254, type=T.i32)
                    c256_i32 = arith.constant(256, type=T.i32)
                    c0xFF800000_i32 = arith.constant(0xFF800000, type=T.i32)
                    c0x400000_i32 = arith.constant(0x400000, type=T.i32)
                    c0x7FFFFFFF_i32 = arith.constant(0x7FFFFFFF, type=T.i32)
                    c0x80000000_i32 = arith.constant(0x80000000, type=T.i32)
                    c0x3F800000_i32 = arith.constant(0x3F800000, type=T.i32)
                    c0x40C00000_i32 = arith.constant(0x40C00000, type=T.i32)
                    c0x4A800000_i32 = arith.constant(0x4A800000, type=T.i32)
                    c0xC11FFFFF_i32 = arith.constant(0xC11FFFFF, type=T.i32)
                    c0x7_i32 = arith.constant(0x7, type=T.i32)
                    c0_f32 = arith.constant(0.0, type=T.f32)

                    fp_headroom = 2 if need_fp4 else (8 if need_fp8 else 0)
                    c_headroom_i32 = arith.constant(fp_headroom, type=T.i32)

                    def f32_to_e2m1(qx_f32):
                        """Convert a scaled f32 value to fp4 (e2m1) 4-bit integer."""
                        qx = qx_f32.bitcast(T.i32)
                        s = qx & c0x80000000_i32
                        qx_abs = qx & c0x7FFFFFFF_i32
                        denormal_mask = arith.cmpi(
                            CmpIPredicate.ult, qx_abs, c0x3F800000_i32
                        )
                        normal_mask = arith.andi(
                            arith.cmpi(CmpIPredicate.ult, qx_abs, c0x40C00000_i32),
                            arith.cmpi(CmpIPredicate.uge, qx_abs, c0x3F800000_i32),
                        )

                        denorm_f32 = qx_abs.bitcast(T.f32) + c0x4A800000_i32.bitcast(T.f32)
                        denormal_x = denorm_f32.bitcast(T.i32) - c0x4A800000_i32

                        mant_odd = (qx_abs >> c22_i32) & c1_i32
                        normal_x = qx_abs + c0xC11FFFFF_i32 + mant_odd
                        normal_x = normal_x >> c22_i32

                        e2m1 = arith.select(normal_mask, normal_x, c0x7_i32)
                        e2m1 = arith.select(denormal_mask, denormal_x, e2m1)
                        return (s >> c28_i32) | e2m1

                    if const_expr(need_sort):
                        n32_sort = sorted_scale_cols_i32 * c32_i32

                    sk_n_offset = [0]

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        _fused, row_byte_base = row_ctx
                        if const_expr(need_quant and not is_splitk):
                            frag_vals = []
                            for i in range_constexpr(e_vec):
                                frag_vals.append(
                                    vector.extract(
                                        frag, static_position=[i], dynamic_position=[]
                                    )
                                )

                            local_max = c0_f32
                            for i in range_constexpr(e_vec):
                                abs_v = llvm.call_intrinsic(
                                    f32, "llvm.fabs.f32", [frag_vals[i]], [], []
                                )
                                local_max = arith.maximumf(local_max, abs_v)

                            for si in range_constexpr(num_shuffle_steps_s1):
                                off = arith.constant(shuffle_dists_s1[si], type=T.i32)
                                peer = local_max.shuffle_xor(off, c64_i32)
                                local_max = arith.maximumf(local_max, peer)

                            max_i32 = local_max.bitcast(T.i32)
                            max_rounded = (max_i32 + c0x400000_i32) & c0xFF800000_i32
                            exp_field = max_rounded >> c23_i32
                            e8m0_biased = arith.maxsi(exp_field - c_headroom_i32, c0_i32)

                            quant_exp = c254_i32 - e8m0_biased
                            quant_scale = (quant_exp << c23_i32).bitcast(T.f32)

                            if const_expr(need_fp4):
                                packed_i32 = c0_i32
                                native_scale = (e8m0_biased << c23_i32).bitcast(T.f32)
                                native_scale_raw = (
                                    native_scale._value
                                    if hasattr(native_scale, "_value")
                                    else native_scale
                                )
                                for k in range_constexpr(e_vec // 2):
                                    old_raw = (
                                        packed_i32._value
                                        if hasattr(packed_i32, "_value")
                                        else packed_i32
                                    )
                                    src0 = frag_vals[2 * k]
                                    src1 = frag_vals[2 * k + 1]
                                    packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                                        T.i32,
                                        old_raw,
                                        (src0._value if hasattr(src0, "_value") else src0),
                                        (src1._value if hasattr(src1, "_value") else src1),
                                        native_scale_raw,
                                        k,
                                    )

                                ptr_addr_idx = row_byte_base + col_g0 // arith.constant(
                                    2, index=True
                                )
                                out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                                pack_bytes = e_vec // 2
                                if const_expr(pack_bytes == 1):
                                    store_val = arith.TruncIOp(T.i8, packed_i32)
                                    store_raw = (
                                        store_val._value
                                        if hasattr(store_val, "_value")
                                        else store_val
                                    )
                                    llvm.StoreOp(
                                        store_raw, out_ptr_v, alignment=1, nontemporal=True
                                    )
                                elif const_expr(pack_bytes == 2):
                                    store_val = arith.TruncIOp(T.i16, packed_i32)
                                    store_raw = (
                                        store_val._value
                                        if hasattr(store_val, "_value")
                                        else store_val
                                    )
                                    llvm.StoreOp(
                                        store_raw, out_ptr_v, alignment=2, nontemporal=True
                                    )
                                else:
                                    packed_raw = (
                                        packed_i32._value
                                        if hasattr(packed_i32, "_value")
                                        else packed_i32
                                    )
                                    llvm.StoreOp(
                                        packed_raw, out_ptr_v, alignment=4, nontemporal=True
                                    )

                            elif const_expr(need_fp8):
                                scaled_vals = []
                                for i in range_constexpr(e_vec):
                                    scaled_vals.append(frag_vals[i] * quant_scale)

                                ptr_addr_idx = row_byte_base + col_g0
                                if const_expr(e_vec <= 4):
                                    packed_i32 = c0_i32
                                    for w in range_constexpr(e_vec // 2):
                                        packed_i32 = rocdl.cvt_pk_fp8_f32(
                                            T.i32,
                                            scaled_vals[2 * w],
                                            scaled_vals[2 * w + 1],
                                            packed_i32,
                                            w,
                                        )
                                    out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                                    if const_expr(e_vec == 2):
                                        store_val = arith.TruncIOp(T.i16, packed_i32)
                                        store_raw = (
                                            store_val._value
                                            if hasattr(store_val, "_value")
                                            else store_val
                                        )
                                        llvm.StoreOp(
                                            store_raw,
                                            out_ptr_v,
                                            alignment=2,
                                            nontemporal=True,
                                        )
                                    else:
                                        packed_raw = (
                                            packed_i32._value
                                            if hasattr(packed_i32, "_value")
                                            else packed_i32
                                        )
                                        llvm.StoreOp(
                                            packed_raw,
                                            out_ptr_v,
                                            alignment=4,
                                            nontemporal=True,
                                        )
                                else:
                                    for wg in range_constexpr(e_vec // 4):
                                        b = wg * 4
                                        packed_w = c0_i32
                                        packed_w = rocdl.cvt_pk_fp8_f32(
                                            T.i32,
                                            scaled_vals[b],
                                            scaled_vals[b + 1],
                                            packed_w,
                                            0,
                                        )
                                        packed_w = rocdl.cvt_pk_fp8_f32(
                                            T.i32,
                                            scaled_vals[b + 2],
                                            scaled_vals[b + 3],
                                            packed_w,
                                            1,
                                        )
                                        word_ptr = ptr_addr_idx + arith.constant(
                                            wg * 4, index=True
                                        )
                                        out_ptr_v = idx_to_llvm_ptr(word_ptr)
                                        packed_raw = (
                                            packed_w._value
                                            if hasattr(packed_w, "_value")
                                            else packed_w
                                        )
                                        llvm.StoreOp(
                                            packed_raw,
                                            out_ptr_v,
                                            alignment=4,
                                            nontemporal=True,
                                        )

                            if const_expr(need_sort):
                                col_g0_i32 = arith.index_cast(T.i32, col_g0)
                                is_scale_writer = arith.cmpi(
                                    CmpIPredicate.eq, col_g0_i32 & c31_i32, c0_i32
                                )
                                if_scale = scf.IfOp(is_scale_writer)
                                with ir.InsertionPoint(if_scale.then_block):
                                    row_i32_s = arith.index_cast(T.i32, row)
                                    col_s_i32 = col_g0_i32 >> c5_i32
                                    d0 = row_i32_s >> c5_i32
                                    d1 = (row_i32_s >> c4_i32) & c1_i32
                                    d2 = row_i32_s & c15_i32
                                    d3 = col_s_i32 >> c3_i32
                                    d4 = (col_s_i32 >> c2_i32) & c1_i32
                                    d5 = col_s_i32 & c3_i32
                                    byte_off = (
                                        d0 * n32_sort
                                        + d3 * c256_i32
                                        + d5 * c64_i32
                                        + d2 * c4_i32
                                        + d4 * c2_i32
                                        + d1
                                    )
                                    e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased)
                                    buffer_ops.buffer_store(
                                        e8m0_i8,
                                        sorted_scale_rsrc,
                                        byte_off,
                                        offset_is_bytes=True,
                                    )
                                    scf.YieldOp([])
                        elif const_expr(is_splitk):
                            col_idx = col_g0 + arith.constant(sk_n_offset[0], index=True)
                            byte_off_col = col_idx * arith.constant(
                                out_elem_bytes, index=True
                            )
                            ptr_addr_idx = row_byte_base + byte_off_col
                            out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                            frag_v = frag._value if hasattr(frag, "_value") else frag
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                out_ptr_v,
                                frag_v,
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent",
                                alignment=e_vec_sk * out_elem_bytes,
                            )
                        else:
                            col_idx = col_g0
                            byte_off_col = col_idx * arith.constant(
                                out_elem_bytes, index=True
                            )
                            ptr_addr_idx = row_byte_base + byte_off_col
                            out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                            frag_v = frag._value if hasattr(frag, "_value") else frag
                            llvm.StoreOp(
                                frag_v,
                                out_ptr_v,
                                alignment=e_vec * out_elem_bytes,
                                nontemporal=True,
                            )

                    frag_elem = (
                        ir.F32Type.get()
                        if need_quant
                        else (ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get())
                    )

                    if const_expr(kwave_fused):
                        slab_n = tile_m * tile_n
                        slab_ty = _mT.memref(
                            k_wave * slab_n, f32, memory_space=_lds_space()
                        )
                        gate_slab = memref.view(
                            slab_ty,
                            base_ptr_pong,
                            arith.constant(lds_pong_offset, index=True),
                            sizes=[],
                        )
                        up_slab = memref.view(
                            slab_ty,
                            base_ptr_ping,
                            arith.constant(lds_ping_offset, index=True),
                            sizes=[],
                        )
                        c_tn = arith.constant(tile_n, index=True)
                        c_slabn = arith.constant(slab_n, index=True)
                        kg_base = wave_k_id * c_slabn
                        vec1_f32 = T.vec(1, f32)
                        vecev_f32 = T.vec(e_vec, f32)

                        gpu.barrier()

                        def fused_write(mi, ii, row_in_tile, row):
                            rb = row_in_tile * c_tn
                            for ni in range_constexpr(num_acc_n):
                                col = (
                                    n_tile_base
                                    + lane_mod_16
                                    + arith.constant(ni * 16, index=True)
                                )
                                aidx = mi * num_acc_n + ni
                                gv = vector.extract(
                                    acc_gate[aidx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                uv = vector.extract(
                                    acc_up[aidx], static_position=[ii], dynamic_position=[]
                                )
                                idx = kg_base + rb + col
                                vector.store(
                                    vector.from_elements(vec1_f32, [gv]),
                                    gate_slab,
                                    [idx],
                                    alignment=4,
                                )
                                vector.store(
                                    vector.from_elements(vec1_f32, [uv]),
                                    up_slab,
                                    [idx],
                                    alignment=4,
                                )

                        default_epilog(
                            arith=arith,
                            range_constexpr=range_constexpr,
                            m_repeat=m_repeat,
                            lane_div_16=lane_div_16,
                            bx_m=bx_m,
                            body_row=fused_write,
                        )
                        gpu.barrier()

                        cn = int(cshuffle_nlane)
                        cm = int(total_threads) // cn
                        mreps = int(tile_m) // cm
                        nreps = int(tile_n) // (cn * int(e_vec))
                        c_cn = arith.constant(cn, index=True)
                        c_ev = arith.constant(e_vec, index=True)
                        m_lane = tx / c_cn
                        n_lane = tx % c_cn
                        for mr in range_constexpr(mreps):
                            _row_local = arith.constant(mr * cm, index=True) + m_lane
                            row = bx_m + _row_local
                            # Unpack unconditionally (a Python `if` becomes scf.if and loses the binding).
                            rc, rp = precompute_row(row_local=_row_local, row=row)

                            def fused_read(_row_local=_row_local, row=row, rc=rc):
                                rb = _row_local * c_tn
                                for nr in range_constexpr(nreps):
                                    cp0 = (
                                        arith.constant(nr * (cn * int(e_vec)), index=True)
                                        + n_lane * c_ev
                                    )
                                    base = rb + cp0
                                    gsum = [None] * int(e_vec)
                                    usum = [None] * int(e_vec)
                                    for kg in range_constexpr(k_wave):
                                        ko = arith.constant(kg, index=True) * c_slabn + base
                                        gvv = vector.load_op(vecev_f32, gate_slab, [ko])
                                        uvv = vector.load_op(vecev_f32, up_slab, [ko])
                                        for e in range_constexpr(int(e_vec)):
                                            ge = vector.extract(
                                                gvv,
                                                static_position=[e],
                                                dynamic_position=[],
                                            )
                                            ue = vector.extract(
                                                uvv,
                                                static_position=[e],
                                                dynamic_position=[],
                                            )
                                            if kg == 0:
                                                gsum[e] = ge
                                                usum[e] = ue
                                            else:
                                                gsum[e] = arith.addf(gsum[e], ge)
                                                usum[e] = arith.addf(usum[e], ue)
                                    fe = [
                                        act_elem(gsum[e], usum[e])
                                        for e in range_constexpr(int(e_vec))
                                    ]
                                    frag = vector.from_elements(vecev_f32, fe)
                                    store_pair(
                                        row_local=_row_local,
                                        row=row,
                                        row_ctx=rc,
                                        col_pair0=cp0,
                                        col_g0=by_n + cp0,
                                        frag=frag,
                                    )

                            ifr = scf.IfOp(rp)
                            with ir.InsertionPoint(ifr.then_block):
                                fused_read()
                                scf.YieldOp([])
                    elif const_expr(gate_up_interleave and not is_splitk):
                        gui_eff_n = gui_out_n
                        gui_tile_n = tile_n // 2
                        gui_cshuffle_nlane = min(32, gui_tile_n // e_vec)
                        gui_by_n = by_n // arith.constant(2, index=True)
                        gui_n_tile_base = n_tile_base // arith.constant(2, index=True)
                        c_shuffle_epilog(
                            arith=arith,
                            vector=vector,
                            gpu=gpu,
                            scf=scf,
                            range_constexpr=range_constexpr,
                            tile_m=tile_m,
                            tile_n=gui_tile_n,
                            e_vec=e_vec,
                            cshuffle_nlane=gui_cshuffle_nlane,
                            block_size=total_threads,
                            m_repeat=m_repeat,
                            num_acc_n=gui_eff_n,
                            tx=tx,
                            lane_div_16=lane_div_16,
                            lane_mod_16=lane_mod_16,
                            bx_m=bx_m,
                            by_n=gui_by_n,
                            n_tile_base=gui_n_tile_base,
                            lds_out=lds_out,
                            frag_elem_type=frag_elem,
                            write_row_to_lds=write_row_to_lds,
                            precompute_row=precompute_row,
                            store_pair=store_pair,
                        )
                    elif const_expr(mock_gate_only or (gate_up_interleave and is_splitk)):
                        eff_e_vec = e_vec_sk
                        acc = acc_gate
                        c_shuffle_epilog(
                            arith=arith,
                            vector=vector,
                            gpu=gpu,
                            scf=scf,
                            range_constexpr=range_constexpr,
                            tile_m=tile_m,
                            tile_n=tile_n,
                            e_vec=eff_e_vec,
                            cshuffle_nlane=cshuffle_nlane_sk,
                            block_size=total_threads,
                            m_repeat=m_repeat,
                            num_acc_n=num_acc_n,
                            tx=tx,
                            lane_div_16=lane_div_16,
                            lane_mod_16=lane_mod_16,
                            bx_m=bx_m,
                            by_n=by_n,
                            n_tile_base=n_tile_base,
                            lds_out=lds_out,
                            frag_elem_type=frag_elem,
                            write_row_to_lds=write_row_to_lds,
                            precompute_row=precompute_row,
                            store_pair=store_pair,
                            lds_out_split=lds_out_B,
                        )
                    elif const_expr(is_splitk):
                        eff_e_vec = e_vec_sk

                        acc = acc_gate
                        sk_n_offset[0] = 0
                        c_shuffle_epilog(
                            arith=arith,
                            vector=vector,
                            gpu=gpu,
                            scf=scf,
                            range_constexpr=range_constexpr,
                            tile_m=tile_m,
                            tile_n=tile_n,
                            e_vec=eff_e_vec,
                            cshuffle_nlane=cshuffle_nlane_sk,
                            block_size=total_threads,
                            m_repeat=m_repeat,
                            num_acc_n=num_acc_n,
                            tx=tx,
                            lane_div_16=lane_div_16,
                            lane_mod_16=lane_mod_16,
                            bx_m=bx_m,
                            by_n=by_n,
                            n_tile_base=n_tile_base,
                            lds_out=lds_out,
                            frag_elem_type=frag_elem,
                            write_row_to_lds=write_row_to_lds,
                            precompute_row=precompute_row,
                            store_pair=store_pair,
                            lds_out_split=lds_out_B,
                        )

                        gpu.barrier()

                        acc = acc_up
                        sk_n_offset[0] = inter_dim
                        c_shuffle_epilog(
                            arith=arith,
                            vector=vector,
                            gpu=gpu,
                            scf=scf,
                            range_constexpr=range_constexpr,
                            tile_m=tile_m,
                            tile_n=tile_n,
                            e_vec=eff_e_vec,
                            cshuffle_nlane=cshuffle_nlane_sk,
                            block_size=total_threads,
                            m_repeat=m_repeat,
                            num_acc_n=num_acc_n,
                            tx=tx,
                            lane_div_16=lane_div_16,
                            lane_mod_16=lane_mod_16,
                            bx_m=bx_m,
                            by_n=by_n,
                            n_tile_base=n_tile_base,
                            lds_out=lds_out,
                            frag_elem_type=frag_elem,
                            write_row_to_lds=write_row_to_lds,
                            precompute_row=precompute_row,
                            store_pair=store_pair,
                            lds_out_split=lds_out_B,
                        )
                    else:
                        c_shuffle_epilog(
                            arith=arith,
                            vector=vector,
                            gpu=gpu,
                            scf=scf,
                            range_constexpr=range_constexpr,
                            tile_m=tile_m,
                            tile_n=tile_n,
                            e_vec=e_vec,
                            cshuffle_nlane=cshuffle_nlane,
                            block_size=total_threads,
                            m_repeat=m_repeat,
                            num_acc_n=num_acc_n,
                            tx=tx,
                            lane_div_16=lane_div_16,
                            lane_mod_16=lane_mod_16,
                            bx_m=bx_m,
                            by_n=by_n,
                            n_tile_base=n_tile_base,
                            lds_out=lds_out,
                            frag_elem_type=frag_elem,
                            write_row_to_lds=write_row_to_lds,
                            precompute_row=precompute_row,
                            store_pair=store_pair,
                            lds_out_split=lds_out_B,
                        )

                if_blk = scf.IfOp(blk_valid)
                with ir.InsertionPoint(if_blk.then_block):
                    ifexpert_of = scf.IfOp(exp_valid)
                    with ir.InsertionPoint(ifexpert_of.then_block):
                        if const_expr(heterogeneous_b):
                            format_if = scf.IfOp(is_shared_expert, has_else=True)
                            with ir.InsertionPoint(format_if.then_block):
                                moe_gemm1_body(shared_b=True)
                                scf.YieldOp([])
                            with ir.InsertionPoint(format_if.else_block):
                                moe_gemm1_body(shared_b=False)
                                scf.YieldOp([])
                        else:
                            moe_gemm1_body()
                        scf.YieldOp([])
                    scf.YieldOp([])

                gpu.barrier()
                scf.YieldOp([])
                for_ip.__exit__(None, None, None)

        if heterogeneous_b:

            @flyc.kernel(name=module_name, known_block_size=[total_threads, 1, 1])
            def moe_gemm1(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_shared_w: fx.Pointer,
                arg_shared_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_num_valid_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                arg_out_scale_sorted: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_n_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
                f32_situ_beta: fx.Float32,
                f32_situ_beta_rcp: fx.Float32,
                f32_situ_linear_beta: fx.Float32,
                f32_situ_linear_beta_rcp: fx.Float32,
                f32_swiglu_limit: fx.Float32,
            ):
                _emit_moe_gemm1(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_shared_w,
                    arg_shared_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_num_valid_ids,
                    arg_bias,
                    arg_out_scale_sorted,
                    i32_tokens_in,
                    i32_n_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                    f32_situ_beta,
                    f32_situ_beta_rcp,
                    f32_situ_linear_beta,
                    f32_situ_linear_beta_rcp,
                    f32_swiglu_limit,
                )

        else:

            @flyc.kernel(name=module_name, known_block_size=[total_threads, 1, 1])
            def moe_gemm1(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_num_valid_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                arg_out_scale_sorted: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_n_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
                f32_situ_beta: fx.Float32,
                f32_situ_beta_rcp: fx.Float32,
                f32_situ_linear_beta: fx.Float32,
                f32_situ_linear_beta_rcp: fx.Float32,
                f32_swiglu_limit: fx.Float32,
            ):
                _emit_moe_gemm1(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_w,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_num_valid_ids,
                    arg_bias,
                    arg_out_scale_sorted,
                    i32_tokens_in,
                    i32_n_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                    f32_situ_beta,
                    f32_situ_beta_rcp,
                    f32_situ_linear_beta,
                    f32_situ_linear_beta_rcp,
                    f32_swiglu_limit,
                )

        cache_tag = (
            module_name,
            a_dtype,
            b_dtype,
            out_dtype,
            tile_m,
            tile_n,
            tile_k,
            doweight_stage1,
            act,
            enable_bias,
            model_dim_pad,
            inter_dim_pad,
            use_cshuffle_epilog,
            persist_m,
            use_async_copy,
            waves_per_eu,
            k_batch,
            gate_mode,
            a_scale_one,
            xcd_swizzle,
            v2_output_layout,
        )
        if heterogeneous_b:
            cache_tag += (shared_expert_id,)

        def _launch_mixed_moe_gemm1(
            arg_out: fx.Pointer,
            arg_x: fx.Pointer,
            arg_w: fx.Pointer,
            arg_scale_x: fx.Pointer,
            arg_scale_w: fx.Pointer,
            arg_shared_w: fx.Pointer,
            arg_shared_scale_w: fx.Pointer,
            arg_sorted_token_ids: fx.Pointer,
            arg_expert_ids: fx.Pointer,
            arg_sorted_weights: fx.Pointer,
            arg_max_token_ids: fx.Pointer,
            arg_bias: fx.Pointer,
            arg_out_scale_sorted: fx.Pointer,
            i32_tokens_in: fx.Int32,
            i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
            f32_situ_beta: fx.Float32,
            f32_situ_beta_rcp: fx.Float32,
            f32_situ_linear_beta: fx.Float32,
            f32_situ_linear_beta_rcp: fx.Float32,
            f32_swiglu_limit: fx.Float32,
            stream: fx.Stream,
        ):
            _ = cache_tag
            allocator_pong.finalized = False
            allocator_ping.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator_pong.finalize()
                allocator_ping.finalize()

            inter_dim_pad_total = arith.constant(2 * inter_dim_pad, index=True)
            tile2_pad = 0
            if const_expr(not gate_only):
                tile_k_stage2 = tile_k // 2
                tile2_pad = (
                    tile_k_stage2 - (inter_dim - inter_dim_pad) % tile_k_stage2
                ) % tile_k_stage2

            inter_in = arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value())
            tile_n_index = arith.constant(tile_n, index=True)
            if const_expr(mock_gate_only or gate_up_interleave):
                gx = (
                    inter_in - inter_dim_pad_total + tile2_pad + tile_n_index - 1
                ) // tile_n_index
            else:
                gx = (
                    (inter_in - inter_dim_pad_total + tile2_pad + 2 * tile_n_index - 1)
                    // tile_n_index
                    // arith.constant(2, index=True)
                )

            c_pm_l = arith.constant(persist_m, index=True)
            gy = (
                arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
                + c_pm_l
                - arith.constant(1, index=True)
            ) // c_pm_l

            if const_expr(heterogeneous_b):
                launcher = moe_gemm1(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_shared_w,
                    arg_shared_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_max_token_ids,
                    arg_bias,
                    arg_out_scale_sorted,
                    i32_tokens_in,
                    i32_inter_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                    f32_situ_beta,
                    f32_situ_beta_rcp,
                    f32_situ_linear_beta,
                    f32_situ_linear_beta_rcp,
                    f32_swiglu_limit,
                )
            else:
                launcher = moe_gemm1(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_max_token_ids,
                    arg_bias,
                    arg_out_scale_sorted,
                    i32_tokens_in,
                    i32_inter_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                    f32_situ_beta,
                    f32_situ_beta_rcp,
                    f32_situ_linear_beta,
                    f32_situ_linear_beta_rcp,
                    f32_swiglu_limit,
                )
            if const_expr(heterogeneous_b and waves_per_eu is not None):
                wpe = int(waves_per_eu)
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, wpe)
            launcher.launch(
                grid=(gx, gy, k_batch), block=(total_threads, 1, 1), stream=stream
            )

        if heterogeneous_b:

            @flyc.jit
            def launch_mixed_moe_gemm1(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_shared_w: fx.Pointer,
                arg_shared_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_max_token_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                arg_out_scale_sorted: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_inter_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
                f32_situ_beta: fx.Float32,
                f32_situ_beta_rcp: fx.Float32,
                f32_situ_linear_beta: fx.Float32,
                f32_situ_linear_beta_rcp: fx.Float32,
                f32_swiglu_limit: fx.Float32,
                stream: fx.Stream,
            ):
                _launch_mixed_moe_gemm1(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_shared_w,
                    arg_shared_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_max_token_ids,
                    arg_bias,
                    arg_out_scale_sorted,
                    i32_tokens_in,
                    i32_inter_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                    f32_situ_beta,
                    f32_situ_beta_rcp,
                    f32_situ_linear_beta,
                    f32_situ_linear_beta_rcp,
                    f32_swiglu_limit,
                    stream,
                )

        else:

            @flyc.jit
            def launch_mixed_moe_gemm1(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_max_token_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                arg_out_scale_sorted: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_inter_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
                f32_situ_beta: fx.Float32,
                f32_situ_beta_rcp: fx.Float32,
                f32_situ_linear_beta: fx.Float32,
                f32_situ_linear_beta_rcp: fx.Float32,
                f32_swiglu_limit: fx.Float32,
                stream: fx.Stream,
            ):
                _launch_mixed_moe_gemm1(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_w,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_max_token_ids,
                    arg_bias,
                    arg_out_scale_sorted,
                    i32_tokens_in,
                    i32_inter_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                    f32_situ_beta,
                    f32_situ_beta_rcp,
                    f32_situ_linear_beta,
                    f32_situ_linear_beta_rcp,
                    f32_swiglu_limit,
                    stream,
                )

        return launch_mixed_moe_gemm1


    def compile_mixed_moe_gemm2_common(
        *,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        doweight_stage2: bool,
        a_dtype: str = "fp8",
        b_dtype: str = "fp4",
        out_dtype: str = "f16",
        use_cshuffle_epilog: bool | None = None,
        accumulate: bool = True,
        enable_bias: bool = False,
        model_dim_pad: int = 0,
        inter_dim_pad: int = 0,
        persist_m: int = 4,
        sort_block_m: int = 0,
        waves_per_eu: int | None = None,
        use_async_copy: bool = False,
        cu_num_mul: int = 1,
        b_nt: int = 0,
        xcd_swizzle: int = 0,
        shared_expert_id: int | None = None,
    ):
        """Compile stage2 kernel (moe_gemm2): A2 @ W2.T -> [tokens, model_dim], atomic-add."""
        heterogeneous_b = shared_expert_id is not None
        if heterogeneous_b and shared_expert_id != experts - 1:
            raise ValueError(
                "FHMoE stage2 requires shared_expert_id == experts - 1; "
                f"got {shared_expert_id=} and {experts=}"
            )
        del b_nt
        _sort_block_m = tile_m if sort_block_m <= 0 else sort_block_m
        if const_expr(_sort_block_m != tile_m and _sort_block_m % tile_m != 0):
            raise ValueError(
                f"sort_block_m ({_sort_block_m}) must be a multiple of tile_m ({tile_m})"
            )

        r139_xdma_first = bool(
            use_async_copy
            and tile_m == 64
            and tile_n == 128
            and tile_k == 256
            and a_dtype == "fp4"
            and b_dtype == "fp4"
            and bool(accumulate)
        )

        gpu_arch = get_hip_arch()
        allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")

        if const_expr(a_dtype not in ("fp8", "fp4")):
            raise ValueError(f"a_dtype must be one of ('fp8','fp4'), got {a_dtype!r}")
        if const_expr(b_dtype not in ("fp8", "fp4")):
            raise ValueError(f"b_dtype must be one of ('fp8','fp4'), got {b_dtype!r}")

        is_f8_a = a_dtype == "fp8"
        is_f4_a = a_dtype == "fp4"
        is_f4_b = b_dtype == "fp4"
        is_f8_b = b_dtype == "fp8"
        if heterogeneous_b and not is_f4_b:
            raise ValueError("Heterogeneous B requires MXFP4 routed weights")
        serial_shared_n = heterogeneous_b and tile_n == 256

        scale_pack_m = 2
        scale_pack_n = 2
        scale_pack_k = 2
        pack_M = min(scale_pack_m, tile_m // 16)
        pack_N = min(scale_pack_n, tile_n // 64)
        k_unroll_raw = int(tile_k) // 128
        pack_K = min(scale_pack_k, k_unroll_raw)

        elem_bytes = 1

        a_elem_bytes = 1
        b_elem_bytes = 1
        tile_k_bytes = int(tile_k) * int(a_elem_bytes)

        a_elem_vec_pack = 2 if is_f4_a else 1
        cbsz = 0 if is_f8_a else 4
        blgp = 0 if is_f8_b else 4
        b_byte_div = 2 if is_f4_b else 1
        b_cells_per_ku = 2 if is_f8_b else 1

        b_kpack_bytes_s = 16
        b_kpack_elems_s = b_kpack_bytes_s // b_elem_bytes
        b_c_k_s = inter_dim // b_byte_div
        b_c_k0_s = (b_c_k_s * b_elem_bytes) // 64
        b_stride_nlane = b_kpack_elems_s
        b_stride_klane = 16 * b_stride_nlane
        b_stride_k0 = 4 * b_stride_klane
        b_stride_n0 = b_c_k0_s * b_stride_k0
        shared_b_c_k0_s = inter_dim // 64
        shared_b_stride_n0 = shared_b_c_k0_s * b_stride_k0
        assert model_dim % 16 == 0, "model_dim must be divisible by 16"
        expert_b_stride = (model_dim // 16) * b_stride_n0

        if const_expr((tile_k_bytes % 64) != 0):
            raise ValueError(
                f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
                f"(tile_k={tile_k}, elem_bytes={a_elem_bytes})"
            )

        out_s = str(out_dtype).strip().lower()
        if const_expr(
            out_s
            not in (
                "f16",
                "fp16",
                "half",
                "bf16",
                "bfloat16",
                "f32",
                "fp32",
                "float",
                "fp8",
            )
        ):
            raise ValueError(
                f"out_dtype must be 'f16', 'bf16', 'f32', or 'fp8', got {out_dtype!r}"
            )
        out_is_f32 = out_s in ("f32", "fp32", "float")
        need_fp8_out = out_s == "fp8"
        out_is_bf16 = out_s in ("bf16", "bfloat16") or need_fp8_out
        if const_expr(need_fp8_out and bool(accumulate)):
            raise ValueError(
                "compile_mixed_moe_gemm2 fp8 output requires accumulate=False (reduce path)"
            )
        if const_expr((not bool(accumulate)) and out_is_f32):
            raise ValueError(
                "compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16','fp8'}"
            )
        w_elem_bytes = 1
        w_elem_pack = 2 if is_f4_b else 1
        w_nbytes = (experts * model_dim * inter_dim * w_elem_bytes) // w_elem_pack
        shared_w_nbytes = model_dim * inter_dim
        # #3476: host e8m0_shuffle pads scale group-N up to a multiple of 8, i.e.
        # 128- but not 256-aligned (e.g. 384) read OOB scales -> garbage e8m0 -> NaN.
        scale_k_padded = (inter_dim + 255) // 256 * 256
        scale_kblk_padded = scale_k_padded // 32
        bias_nbytes = experts * model_dim * 4

        def x_elem_type():
            if const_expr(is_f4_b):
                return default_f8_type() if is_f8_a else T.i8
            return default_f8_type()

        def w_elem_type():
            if const_expr(is_f4_b):
                return T.i8
            return default_f8_type()

        def scale_elem_type():
            return T.i32

        total_threads = 256
        bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
        if const_expr(bytes_x_per_tile % total_threads != 0):
            raise ValueError(
                "tile_m*tile_k*elem_bytes must be divisible by "
                f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}"
            )
        bytes_per_thread_x = bytes_x_per_tile // total_threads

        lds_stride = tile_k

        if const_expr(out_is_f32):
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
            if const_expr(_use_cshuffle_epilog):
                raise ValueError(
                    "out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False)."
                )
        else:
            _use_cshuffle_epilog = True

        def out_elem():
            return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

        def load_bias_scalar(bias_rsrc, offset):
            return buffer_ops.buffer_load(bias_rsrc, offset, vec_width=1, dtype=T.f32)

        epilog_tag = "cshuffle"
        persistent = persist_m <= 0
        if const_expr(not isinstance(cu_num_mul, int) or cu_num_mul < 1):
            raise ValueError(f"cu_num_mul must be int >= 1, got {cu_num_mul}")
        if const_expr(persistent):
            get_cu_num = _get_cu_num

            cu_num = get_cu_num() * int(cu_num_mul)
        else:
            cu_num = 0
        sbm_tag = "" if _sort_block_m == tile_m else f"_sbm{_sort_block_m}"
        pm_tag = f"_persist_cu{cu_num}" if persistent else f"_pm{persist_m}"
        wpe_tag = f"_w{waves_per_eu}" if waves_per_eu is not None else ""
        if const_expr(waves_per_eu is not None and not (1 <= int(waves_per_eu) <= 10)):
            raise ValueError(f"waves_per_eu must be in [1, 10] or None, got {waves_per_eu}")
        num_k_tiles_per_batch = int(inter_dim) // int(tile_k)
        async_tag = "_async" if use_async_copy else ""
        cumul_tag = f"_cumul{int(cu_num_mul)}" if int(cu_num_mul) != 1 else ""
        acc_tag = "" if accumulate else "_acc0"
        xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
        heterogeneous_tag = f"_shared_fp8_e{shared_expert_id}" if heterogeneous_b else ""
        serial_n_tag = "_serialn128" if serial_shared_n else ""
        if heterogeneous_b:
            variant_tags = (
                f"_vscale_fix3_fp4opt_v2{pm_tag}{sbm_tag}{wpe_tag}{async_tag}"
                f"{cumul_tag}{acc_tag}{xcd_tag}{heterogeneous_tag}{serial_n_tag}"
            )
        else:
            variant_tags = (
                f"_vscale_fix3_fp4opt_v1{pm_tag}{sbm_tag}{wpe_tag}{async_tag}"
                f"{cumul_tag}{xcd_tag}{acc_tag}"
            )
        module_name = (
            f"mfma_moe2_a{a_dtype}_w{b_dtype}_{out_s}_{epilog_tag}"
            f"_t{tile_m}x{tile_n}x{tile_k}{variant_tags}"
        ).replace("-", "_")
        lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
        lds_out_bytes = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
        lds_tid_bytes = int(tile_m) * 4
        lds_tw_bytes = (int(tile_m) * 4) if bool(doweight_stage2) else 0
        lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes + lds_tw_bytes
        lds_total_elems = lds_total_bytes if a_elem_bytes == 1 else (lds_total_bytes // 2)

        def x_lds_elem():
            return default_f8_type()

        lds_alloc_bytes = int(lds_total_elems) * int(a_elem_bytes)
        lds_alloc_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = lds_alloc_offset + lds_alloc_bytes

        if const_expr(True):

            def _emit_moe_gemm2(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_shared_w: fx.Pointer,
                arg_shared_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_num_valid_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_n_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
            ):

                tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
                n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
                k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
                size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
                x_elem = default_f8_type()
                f32 = T.f32
                i32 = T.i32
                i64 = T.i64
                vec4_f32 = T.vec(4, f32)
                vec16_elems = 16 if a_elem_bytes == 1 else 8
                vec8_elems = 8 if a_elem_bytes == 1 else 4
                vec4_elems = 4 if a_elem_bytes == 1 else 2
                vec16_x = T.vec(vec16_elems, x_elem)
                vec2_i64 = T.vec(2, i64)

                def ptr_buffer_resource(ptr, num_records_bytes):
                    addr = fx.ptrtoint(ptr)
                    addr_i64 = arith.index_cast(T.i64, addr)
                    return buffer_ops.create_buffer_resource_from_addr(
                        addr_i64, num_records_bytes=num_records_bytes
                    )

                acc_init = arith.constant_vector(0.0, vec4_f32)

                topk_idx = arith.constant(topk, index=True)
                m_in = tokens_in * topk_idx

                c_n_total = arith.constant(experts * model_dim, index=True)
                kpack_bytes = 16
                _div_pow2 = _M_layout_utils._div_pow2
                _mod_pow2 = _M_layout_utils._mod_pow2

                def check_c_n_valid_gate(base_n):
                    return arith.cmpi(CmpIPredicate.ult, base_n, model_dim - model_dim_pad)

                def check_c_k_valid_gate(base_k):
                    return arith.cmpi(CmpIPredicate.ult, base_k, inter_dim - inter_dim_pad)

                # A&B's scale preshuffle layout.  #3476: host e8m0_shuffle pads the
                c_k_orig = arith.constant(scale_k_padded, index=True)
                layout_a_scale = make_preshuffle_scale_layout(
                    arith, c_mn=m_in, c_k=c_k_orig
                )
                layout_b_scale = make_preshuffle_scale_layout(
                    arith, c_mn=c_n_total, c_k=c_k_orig
                )

                if const_expr(use_async_copy and a_elem_vec_pack > 1):
                    eff_lds_stride = lds_stride // a_elem_vec_pack
                    eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack
                else:
                    eff_lds_stride = lds_stride
                    eff_tile_k_bytes = tile_k_bytes

                shape_lds = fx.make_shape(tile_m, eff_lds_stride)
                stride_lds = fx.make_stride(eff_lds_stride, 1)
                layout_lds = fx.make_layout(shape_lds, stride_lds)

                tx = gpu.thread_id("x")
                by_outer = gpu.block_id("x")
                bx_persist = gpu.block_id("y")

                if const_expr(xcd_swizzle > 0):
                    num_xcds = 8
                    one = arith.constant(1, index=True)
                    tile_n_idx = arith.constant(tile_n, index=True)
                    model_pad_idx = arith.constant(model_dim_pad, index=True)
                    grid_x = (n_in - model_pad_idx + tile_n_idx - one) // tile_n_idx
                    if const_expr(persistent):
                        grid_y = arith.constant(cu_num, index=True)
                    else:
                        persist_m_idx = arith.constant(persist_m, index=True)
                        grid_y = (size_expert_ids_in + persist_m_idx - one) // persist_m_idx
                    linear_id = bx_persist * grid_x + by_outer
                    num_workgroups = grid_x * grid_y
                    num_xcds_idx = arith.constant(num_xcds, index=True)
                    workgroups_per_xcd = num_workgroups // num_xcds_idx
                    workgroup_id = (linear_id % num_xcds_idx) * workgroups_per_xcd + (
                        linear_id // num_xcds_idx
                    )
                    group_m = arith.constant(xcd_swizzle, index=True)
                    workgroups_per_group = group_m * grid_x
                    group_id = workgroup_id // workgroups_per_group
                    first_m = group_id * group_m
                    remaining_m = grid_y - first_m
                    group_size_m = arith.select(
                        arith.cmpi(CmpIPredicate.ult, remaining_m, group_m),
                        remaining_m,
                        group_m,
                    )
                    workgroup_in_group = workgroup_id % workgroups_per_group
                    bx_persist = first_m + (workgroup_in_group % group_size_m)
                    by_outer = workgroup_in_group // group_size_m

                by = by_outer

                k_blocks16 = arith.constant(eff_tile_k_bytes // 16, index=True)
                layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
                layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

                base_ptr = allocator.get_base()
                lds_x_ptr = SmemPtr(
                    base_ptr,
                    lds_alloc_offset,
                    x_lds_elem(),
                    shape=(lds_total_elems,),
                )
                lds_x = lds_x_ptr.get()
                lds_out = (
                    SmemPtr(
                        base_ptr,
                        lds_x_ptr.byte_offset,
                        (T.bf16 if out_is_bf16 else T.f16),
                        shape=(tile_m * tile_n,),
                    ).get()
                    if _use_cshuffle_epilog
                    else None
                )

                lds_x_b = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
                lds_out_b = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
                lds_tid_off = max(lds_x_b, lds_out_b)
                lds_tid = SmemPtr(
                    base_ptr, lds_x_ptr.byte_offset + lds_tid_off, T.i32, shape=(tile_m,)
                ).get()
                # lds_tw aliases the LDS slot immediately after
                lds_tw_off = lds_tid_off + int(tile_m) * 4
                lds_tw = (
                    SmemPtr(
                        base_ptr,
                        lds_x_ptr.byte_offset + lds_tw_off,
                        T.f32,
                        shape=(tile_m,),
                    ).get()
                    if doweight_stage2
                    else None
                )

                c_topk = arith.constant(topk, index=True)

                c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)
                c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
                x_nbytes_idx = _div_pow2(
                    (tokens_in * c_topk) * k_in * c_elem_bytes, int(a_elem_vec_pack)
                )
                x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
                x_rsrc = ptr_buffer_resource(arg_x, x_nbytes_i32)

                w_rsrc = ptr_buffer_resource(arg_w, w_nbytes)
                shared_w_rsrc = ptr_buffer_resource(arg_shared_w, shared_w_nbytes)

                out_elem_bytes = 1 if need_fp8_out else (4 if out_is_f32 else 2)
                # fp8 route-out row = [N fp8 value bytes | N/8 e8m0 scale bytes].
                scale_bytes_per_row = (int(model_dim) // 8) if need_fp8_out else 0
                out_row_bytes_const = int(model_dim) * int(out_elem_bytes) + int(
                    scale_bytes_per_row
                )
                out_nbytes_idx = (
                    tokens_in * n_in * arith.constant(out_elem_bytes, index=True)
                )
                if const_expr(not bool(accumulate)):
                    out_nbytes_idx = (
                        tokens_in
                        * arith.index(topk)
                        * n_in
                        * arith.constant(out_elem_bytes, index=True)
                    )
                if const_expr(need_fp8_out):
                    out_nbytes_idx = (
                        tokens_in
                        * arith.index(topk)
                        * arith.constant(out_row_bytes_const, index=True)
                    )
                out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes_idx)
                out_rsrc = ptr_buffer_resource(arg_out, out_nbytes_i32)

                numids_rsrc = ptr_buffer_resource(
                    arg_num_valid_ids, arith.constant(4, type=T.i32)
                )
                num_valid_i32 = buffer_ops.buffer_load(
                    numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
                )
                num_valid_i32 = rocdl.ReadfirstlaneOp(T.i32, num_valid_i32).res
                num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)

                if const_expr(is_f4_a or is_f8_a):
                    # #3476: use 256-padded K/32 to match host scale padding.
                    kblk = arith.constant(scale_kblk_padded, index=True)
                    scale_rows = (
                        (num_valid_idx + arith.constant(31, index=True))
                        // arith.constant(32, index=True)
                        * arith.constant(32, index=True)
                    )
                    sx_nbytes_idx = scale_rows * kblk
                    sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                    sx_rsrc = ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)
                else:
                    sx_nbytes_idx = (tokens_in * c_topk) * arith.constant(4, index=True)
                    sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                    sx_rsrc = ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)

                kblk_w = arith.constant(scale_kblk_padded, index=True)
                mn_w = arith.constant(experts * model_dim, index=True)
                sw_nbytes_idx = mn_w * kblk_w
                sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
                sw_rsrc = ptr_buffer_resource(arg_scale_w, sw_nbytes_i32)
                shared_scale_rows = arith.constant(
                    ((model_dim + 255) // 256) * 256, index=True
                )
                shared_scale_cols = arith.constant(
                    ((scale_kblk_padded + 7) // 8) * 8, index=True
                )
                shared_sw_nbytes_i32 = arith.index_cast(
                    T.i32, shared_scale_rows * shared_scale_cols
                )
                shared_sw_rsrc = ptr_buffer_resource(
                    arg_shared_scale_w, shared_sw_nbytes_i32
                )

                sorted_nbytes_idx = (
                    size_expert_ids_in
                    * arith.constant(tile_m, index=True)
                    * arith.constant(4, index=True)
                )
                sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
                sorted_rsrc = ptr_buffer_resource(arg_sorted_token_ids, sorted_nbytes_i32)
                sorted_w_rsrc = ptr_buffer_resource(arg_sorted_weights, sorted_nbytes_i32)

                c_sbm = arith.constant(_sort_block_m, index=True)
                c_tm = arith.constant(tile_m, index=True)
                c1 = arith.constant(1, index=True)
                sort_blocks_ub = _div_pow2(
                    size_expert_ids_in * c_tm + c_sbm - c1, _sort_block_m
                )
                eid_nbytes_idx = sort_blocks_ub * arith.constant(4, index=True)
                eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
                expert_rsrc = ptr_buffer_resource(arg_expert_ids, eid_nbytes_i32)
                bias_rsrc = (
                    ptr_buffer_resource(arg_bias, bias_nbytes) if enable_bias else None
                )

                c0_p = arith.constant(0, index=True)
                c1_p = arith.constant(1, index=True)

                if const_expr(persistent):
                    c_cu = arith.constant(cu_num, index=True)
                    c_tm_p = arith.constant(tile_m, index=True)
                    _num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)
                    total_m_tiles = (_num_valid_idx + c_tm_p - c1_p) // c_tm_p
                    tiles_per_block_base = total_m_tiles // c_cu
                    tiles_remainder = total_m_tiles - (tiles_per_block_base * c_cu)
                    has_extra_tile = arith.cmpi(
                        CmpIPredicate.ult, bx_persist, tiles_remainder
                    )
                    extra_tile = arith.select(has_extra_tile, c1_p, c0_p)
                    tiles_per_block = tiles_per_block_base + extra_tile
                    start_tail = arith.select(has_extra_tile, bx_persist, tiles_remainder)
                    persist_start_tile = bx_persist * tiles_per_block_base + start_tail
                    i1 = ir.IntegerType.get_signless(1)
                    init_active = arith.constant(1, type=i1)
                    for_persist = scf.ForOp(c0_p, tiles_per_block, c1_p, [init_active])
                else:
                    c_pm = arith.constant(persist_m, index=True)
                    init_prev_expert = arith.constant(0, type=T.i32)
                    init_prev_b_base = arith.constant(0, index=True)
                    for_persist = scf.ForOp(
                        c0_p,
                        c_pm,
                        c1_p,
                        [init_prev_expert, init_prev_b_base],
                    )

                for_ip = ir.InsertionPoint(for_persist.body)
                for_ip.__enter__()
                mi_p = for_persist.induction_variable

                if const_expr(persistent):
                    still_active = for_persist.inner_iter_args[0]
                    bx = persist_start_tile + mi_p
                else:
                    prev_expert_i32 = for_persist.inner_iter_args[0]
                    prev_expert_b_base = for_persist.inner_iter_args[1]
                    bx = bx_persist * arith.constant(persist_m, index=True) + mi_p

                bx_m = bx * arith.constant(tile_m, index=True)

                bx_m_i32 = arith.index_cast(T.i32, bx_m)
                blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)

                sort_blk = _div_pow2(bx_m, _sort_block_m)
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, sort_blk, vec_width=1, dtype=T.i32
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                exp_valid = arith.cmpi(
                    CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
                )
                if const_expr(heterogeneous_b):
                    is_shared_expert = arith.cmpi(
                        CmpIPredicate.eq,
                        expert_i32,
                        arith.constant(shared_expert_id, type=T.i32),
                    )

                if const_expr(persistent):
                    expert_b_base = expert_idx * arith.constant(expert_b_stride, index=True)
                else:
                    delta_expert = arith.subi(expert_i32, prev_expert_i32)
                    delta_expert_idx = arith.index_cast(ir.IndexType.get(), delta_expert)
                    delta_b = delta_expert_idx * arith.constant(expert_b_stride, index=True)
                    expert_b_base = prev_expert_b_base + delta_b

                first_tok = buffer_ops.buffer_load(
                    sorted_rsrc, bx_m, vec_width=1, dtype=T.i32
                )
                first_tid = arith.andi(first_tok, arith.constant(0xFFFFFF, type=T.i32))
                tokens_i32_guard = arith.index_cast(T.i32, tokens_in)
                tile_has_tokens = arith.cmpi(CmpIPredicate.ult, first_tid, tokens_i32_guard)

                if const_expr(pack_M < scale_pack_m):
                    m_off = _mod_pow2(_div_pow2(bx_m, 16), scale_pack_m)
                    m_scale_shift_i32 = arith.index_cast(
                        T.i32, m_off * arith.constant(8, index=True)
                    )
                else:
                    m_scale_shift_i32 = None

                def moe_gemm2_then_body(
                    shared_b: bool = False,
                    shared_n_half: int | None = None,
                ):
                    body_b_has_full_operand = is_f8_b or shared_b
                    body_tile_n = tile_n // 2 if shared_n_half is not None else tile_n
                    body_n_offset = (
                        shared_n_half * body_tile_n if shared_n_half is not None else 0
                    )
                    n_idx = arith.constant(model_dim, index=True)
                    expert_off_idx = expert_idx * n_idx
                    if const_expr(shared_b):
                        weight_expert_off_idx = arith.index(0)
                        weight_scale_rsrc = shared_sw_rsrc
                    else:
                        weight_expert_off_idx = expert_off_idx
                        weight_scale_rsrc = sw_rsrc

                    def mixed_b_mfma(
                        a128,
                        b128,
                        acc,
                        opsel_a,
                        a_scale_val,
                        opsel_b,
                        b_scale_val,
                    ):
                        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            vec4_f32,
                            [
                                a128,
                                b128,
                                acc,
                                cbsz,
                                0 if shared_b else blgp,
                                opsel_a,
                                a_scale_val,
                                opsel_b,
                                b_scale_val,
                            ],
                        )

                    if const_expr(bytes_per_thread_x % 16 == 0):
                        x_load_bytes = 16
                    elif const_expr(bytes_per_thread_x % 8 == 0):
                        x_load_bytes = 8
                    elif const_expr(bytes_per_thread_x % 4 == 0):
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                        )
                    num_x_loads = bytes_per_thread_x // x_load_bytes
                    chunk_i32 = x_load_bytes // 4
                    vec4_i32 = T.vec(4, i32)

                    c_k_div4 = _div_pow2(
                        _div_pow2(k_in, int(a_elem_vec_pack))
                        * arith.constant(int(a_elem_bytes), index=True),
                        4,
                    )
                    tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                        4 * int(a_elem_vec_pack)
                    )
                    layout_x_tile_div4 = fx.make_layout(
                        (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                    )
                    c_chunk_i32 = arith.constant(chunk_i32, index=True)
                    tx_i32_base = tx * c_chunk_i32

                    topk_i32 = arith.constant(topk)
                    mask24 = arith.constant(0xFFFFFF)
                    tokens_i32 = arith.index_cast(T.i32, tokens_in)

                    def x_tile_chunk_coord_i32(i: int):
                        return tile_chunk_coord_i32(
                            arith,
                            tx_i32_base=tx_i32_base,
                            i=i,
                            total_threads=total_threads,
                            layout_tile_div4=layout_x_tile_div4,
                            chunk_i32=chunk_i32,
                        )

                    vec1_i32 = T.vec(1, i32)
                    vec2_i32 = T.vec(2, i32)
                    x_load_vec_elems = (
                        x_load_bytes if a_elem_bytes == 1 else x_load_bytes // a_elem_bytes
                    )

                    def load_x(idx_i32):
                        """Load `x_load_bytes` bytes from X (gmem) into regs.

                        For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                        """
                        if const_expr(x_load_bytes == 16):
                            idx_elem = (
                                idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
                            )
                            return buffer_copy_gmem16_dwordx4(
                                buffer_ops,
                                vector,
                                elem_type=x_elem,
                                idx_i32=idx_elem,
                                rsrc=x_rsrc,
                                vec_elems=vec16_elems,
                            )
                        idx_bytes = idx_i32 * arith.index(4)
                        return _buffer_load_vec(
                            buffer_ops,
                            vector,
                            x_rsrc,
                            idx_bytes,
                            elem_type=x_elem,
                            vec_elems=x_load_vec_elems,
                            elem_bytes=a_elem_bytes,
                            offset_in_bytes=True,
                        )

                    if const_expr(use_async_copy and a_elem_vec_pack > 1):
                        dma_bytes_pre = 16
                        eff_bytes_pre = (
                            int(tile_m) * int(eff_lds_stride) * int(a_elem_bytes)
                        )
                        num_x_addr_loads = max(
                            1, eff_bytes_pre // (total_threads * dma_bytes_pre)
                        )
                    else:
                        num_x_addr_loads = num_x_loads
                    x_row_base_div4 = []
                    x_col_local_i32 = []
                    x_row_local = []
                    for i in range_constexpr(num_x_loads):
                        row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                        x_row_local.append(row_local)
                        x_col_local_i32.append(col_local_i32)

                        if const_expr(i < num_x_addr_loads):
                            sorted_row_i = bx_m + row_local
                            fused_i = buffer_ops.buffer_load(
                                sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                            )
                            t_i32 = arith.andi(fused_i, mask24)
                            s_i32 = arith.shrui(fused_i, arith.constant(24))

                            t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                            s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32)
                            ts_valid = arith.andi(t_valid, s_valid)
                            t_safe = arith.select(ts_valid, t_i32, arith.constant(0))
                            s_safe = arith.select(ts_valid, s_i32, arith.constant(0))
                            row_ts_i32 = t_safe * topk_i32 + s_safe
                            row_ts_idx = arith.index_cast(T.index, row_ts_i32)

                            x_row_base_div4.append(row_ts_idx * c_k_div4)
                        else:
                            x_row_base_div4.append(arith.index(0))

                    def load_x_tile(base_k):
                        base_k_div4 = _div_pow2(
                            _div_pow2(base_k, int(a_elem_vec_pack))
                            * arith.constant(int(a_elem_bytes), index=True),
                            4,
                        )
                        parts = []
                        for i in range_constexpr(num_x_loads):
                            idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                            x_vec = load_x(idx_i32)

                            if const_expr(x_load_bytes == 16):
                                parts.append(vector.bitcast(vec4_i32, x_vec))
                            elif const_expr(x_load_bytes == 8):
                                parts.append(vector.bitcast(vec2_i32, x_vec))
                            else:
                                parts.append(vector.bitcast(vec1_i32, x_vec))
                        return parts

                    coord_wl = idx2crd(fx.Int32(tx), layout_tx_wave_lane)
                    wave_id = layout_get(coord_wl, 0)
                    lane_id = layout_get(coord_wl, 1)
                    coord_l16 = idx2crd(fx.Int32(lane_id), layout_lane16)
                    lane_div_16 = layout_get(coord_l16, 0)
                    lane_mod_16 = layout_get(coord_l16, 1)

                    row_a_lds = lane_mod_16

                    col_offset_base = lane_div_16 * arith.constant(16, index=True)

                    num_waves = 4
                    body_n_per_wave = body_tile_n // num_waves
                    body_num_acc_n = body_n_per_wave // 16
                    c_n_per_wave = arith.constant(body_n_per_wave, index=True)
                    wave_mod_4 = _mod_pow2(wave_id, 4)
                    n_tile_base = wave_mod_4 * c_n_per_wave

                    body_by_n = by * arith.constant(tile_n, index=True) + arith.constant(
                        body_n_offset, index=True
                    )

                    if const_expr(pack_N < scale_pack_n):
                        global_n_base = expert_off_idx + body_by_n + n_tile_base
                        n_off = _mod_pow2(_div_pow2(global_n_base, 16), scale_pack_n)
                        n_scale_shift_i32 = arith.index_cast(
                            T.i32, n_off * arith.constant(8, index=True)
                        )
                    else:
                        n_scale_shift_i32 = None
                    n_intra_list = [None] * body_num_acc_n
                    n_blk_list = [None] * body_num_acc_n
                    for i in range_constexpr(body_num_acc_n):
                        offset = i * 16
                        c_offset = arith.constant(offset, index=True)
                        global_n = body_by_n + n_tile_base + c_offset + lane_mod_16
                        n_blk_list[i] = _div_pow2(global_n, 16)
                        n_intra_list[i] = _mod_pow2(global_n, 16)

                    m_repeat = tile_m // 16
                    k_unroll = tile_k_bytes // 128

                    k_unroll_packed = k_unroll // pack_K
                    m_repeat_packed = m_repeat // pack_M
                    num_acc_n_packed = body_num_acc_n // pack_N

                    def load_b_packs_k64(
                        base_k, ku: int, ni: int, *, n_blk_p=None, n_intra_p=None
                    ):
                        """Load one B MFMA k-step.

                        fp4: single 16B load -> 2x i64 (lower MFMA half).
                        fp8: two adjacent k0 cells (32B) -> 4x i64 (full operand).

                        `n_blk_p` / `n_intra_p` allow callers to override the per-N
                        list. When ``None`` we fall back to the body-level defaults
                        (``n_blk_list`` / ``n_intra_list``).
                        """
                        blk = n_blk_p if n_blk_p is not None else n_blk_list
                        intra = n_intra_p if n_intra_p is not None else n_intra_list
                        routed_k0_base = _div_pow2(base_k, 64) + arith.constant(
                            ku * b_cells_per_ku, index=True
                        )
                        k1 = lane_div_16
                        vec_elems = kpack_bytes // int(b_elem_bytes)

                        def load_cell(rsrc, expert_base, stride_n0, elem_type, k0):
                            idx_pack = (
                                expert_base
                                + blk[ni] * arith.constant(stride_n0, index=True)
                                + k0 * arith.constant(b_stride_k0, index=True)
                                + k1 * arith.constant(b_stride_klane, index=True)
                                + intra[ni] * arith.constant(b_stride_nlane, index=True)
                            )
                            b16 = _buffer_load_vec(
                                buffer_ops,
                                vector,
                                rsrc,
                                idx_pack,
                                elem_type=elem_type,
                                vec_elems=vec_elems,
                                elem_bytes=b_elem_bytes,
                                offset_in_bytes=(b_elem_bytes == 1),
                            )
                            b_i64x2 = vector.bitcast(vec2_i64, b16)
                            return (
                                vector.extract(
                                    b_i64x2, static_position=[0], dynamic_position=[]
                                ),
                                vector.extract(
                                    b_i64x2, static_position=[1], dynamic_position=[]
                                ),
                            )

                        if const_expr(shared_b):
                            shared_k0_base = _div_pow2(base_k * arith.index(2), 64)
                            shared_k0_base += arith.constant(ku * 2, index=True)
                            s0, s1 = load_cell(
                                shared_w_rsrc,
                                arith.index(0),
                                shared_b_stride_n0,
                                default_f8_type(),
                                shared_k0_base,
                            )
                            s2, s3 = load_cell(
                                shared_w_rsrc,
                                arith.index(0),
                                shared_b_stride_n0,
                                default_f8_type(),
                                shared_k0_base + arith.index(1),
                            )
                            return s0, s1, s2, s3

                        b0, b1 = load_cell(
                            w_rsrc,
                            expert_b_base,
                            b_stride_n0,
                            w_elem_type(),
                            routed_k0_base,
                        )
                        if const_expr(is_f8_b):
                            b2, b3 = load_cell(
                                w_rsrc,
                                expert_b_base,
                                b_stride_n0,
                                w_elem_type(),
                                routed_k0_base + arith.index(1),
                            )
                            return b0, b1, b2, b3
                        return b0, b1

                    def accum_b_ku(b_tile, base_k, ku, n_blk_p, n_intra_p):
                        """Append one ku's (p0..p3) N-packs to b_tile (p2/p3 fp8-B only)."""
                        packs0, packs1, packs2, packs3 = [], [], [], []
                        for ni in range_constexpr(body_num_acc_n):
                            b = load_b_packs_k64(
                                base_k,
                                ku,
                                ni,
                                n_blk_p=n_blk_p,
                                n_intra_p=n_intra_p,
                            )
                            packs0.append(b[0])
                            packs1.append(b[1])
                            if const_expr(body_b_has_full_operand):
                                packs2.append(b[2])
                                packs3.append(b[3])
                        b_tile.append((packs0, packs1, packs2, packs3))

                    def load_b_tile(base_k, *, n_blk_p=None, n_intra_p=None):
                        b_tile = []
                        for ku in range_constexpr(k_unroll):
                            accum_b_ku(b_tile, base_k, ku, n_blk_p, n_intra_p)
                        return b_tile

                    b_split_enabled = k_unroll >= 2
                    is_prefill_shape = (
                        int(tile_m) == 64
                        and int(body_tile_n) == 128
                        and int(tile_k) == 256
                        and bool(use_async_copy)
                        and a_elem_vec_pack > 1
                    )
                    if const_expr(b_split_enabled and is_prefill_shape and k_unroll >= 4):
                        b_split_ku = k_unroll - 1
                    else:
                        b_split_ku = k_unroll // 2 if b_split_enabled else k_unroll

                    def load_b_tile_lo(base_k, *, n_blk_p=None, n_intra_p=None):
                        """Load first half of B tile (ku < _b_split_ku)."""
                        b_tile = []
                        for ku in range_constexpr(b_split_ku):
                            accum_b_ku(b_tile, base_k, ku, n_blk_p, n_intra_p)
                        return b_tile

                    def load_b_tile_hi(base_k, *, n_blk_p=None, n_intra_p=None):
                        """Load second half of B tile (ku >= _b_split_ku)."""
                        b_tile = []
                        for ku in range_constexpr(b_split_ku, k_unroll):
                            accum_b_ku(b_tile, base_k, ku, n_blk_p, n_intra_p)
                        return b_tile

                    def load_scale(arg_scale, rsrc, scale_info, ku, mni):
                        k_lane = lane_div_16
                        n_lane = lane_mod_16
                        idx_pack = (
                            mni * scale_info.stride_n0
                            + ku * scale_info.stride_k0
                            + k_lane * scale_info.stride_klane
                            + n_lane
                        )
                        s = buffer_ops.buffer_load(rsrc, idx_pack, vec_width=1, dtype=T.i32)
                        return vector.from_elements(T.vec(1, T.i32), [s])

                    def apply_k_shift(scale_vec, k_shift_bits):
                        if const_expr(k_shift_bits > 0):
                            val = vector.extract(
                                scale_vec, static_position=[0], dynamic_position=[]
                            )
                            val = arith.shrui(val, arith.constant(k_shift_bits, type=T.i32))
                            return vector.from_elements(T.vec(1, T.i32), [val])
                        return scale_vec

                    def load_b_scale_tile(base_k, k_shift_bits=0, *, by_n_p=None):
                        """Load B-scale tile.

                        `by_n_p` overrides the body-default `by_n`.
                        """
                        byn = by_n_p if by_n_p is not None else body_by_n
                        b_scale_tile = []
                        for ku in range_constexpr(k_unroll_packed):
                            for ni in range_constexpr(num_acc_n_packed):
                                scale = load_scale(
                                    arg_scale_w,
                                    weight_scale_rsrc,
                                    layout_b_scale,
                                    ku + base_k,
                                    ni
                                    + _div_pow2(
                                        _div_pow2(
                                            weight_expert_off_idx + byn + n_tile_base,
                                            scale_pack_n,
                                        ),
                                        16,
                                    ),
                                )
                                scale = apply_k_shift(scale, k_shift_bits)
                                b_scale_tile.append(scale)
                        return b_scale_tile

                    def load_a_scale_tile(base_k, k_shift_bits=0):
                        a_scale_tile = []
                        for ku in range_constexpr(k_unroll_packed):
                            for mi in range_constexpr(m_repeat_packed):
                                scale = load_scale(
                                    arg_scale_x,
                                    sx_rsrc,
                                    layout_a_scale,
                                    ku + base_k,
                                    mi + _div_pow2(_div_pow2(bx_m, scale_pack_m), 16),
                                )
                                scale = apply_k_shift(scale, k_shift_bits)
                                a_scale_tile.append(scale)
                        return a_scale_tile

                    def prefetch_ab_scale_tile(base_k, k_shift_bits=0, *, by_n_p=None):
                        return [
                            load_a_scale_tile(base_k, k_shift_bits),
                            load_b_scale_tile(base_k, k_shift_bits, by_n_p=by_n_p),
                        ]

                    vec8_x = T.vec(vec8_elems, x_elem)
                    vec4_x_lds = T.vec(vec4_elems, x_elem)

                    def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                        for i in range_constexpr(num_x_loads):
                            row_local = x_row_local[i]
                            col_local_i32 = x_col_local_i32[i]
                            if const_expr(x_load_bytes == 16):
                                lds_store_16b_xor16(
                                    arith,
                                    vector,
                                    lds_memref=lds_x,
                                    vec16_ty=vec16_x,
                                    layout_lds=layout_lds,
                                    row_local=row_local,
                                    col_local_i32=col_local_i32,
                                    tx_c4=arith.index(4),
                                    k_blocks16=k_blocks16,
                                    lds_base=lds_base,
                                    vec_part_i32x4=vec_x_in_parts[i],
                                    elem_bytes=elem_bytes,
                                )
                            elif const_expr(x_load_bytes == 8):
                                lds_store_8b_xor16(
                                    arith,
                                    vector,
                                    lds_memref=lds_x,
                                    vec8_ty=vec8_x,
                                    layout_lds=layout_lds,
                                    row_local=row_local,
                                    col_local_i32=col_local_i32,
                                    tx_c4=arith.index(4),
                                    k_blocks16=k_blocks16,
                                    lds_base=lds_base,
                                    vec_part_i32x2=vec_x_in_parts[i],
                                    elem_bytes=elem_bytes,
                                )
                            else:
                                lds_store_4b_xor16(
                                    arith,
                                    vector,
                                    lds_memref=lds_x,
                                    vec4_ty=vec4_x_lds,
                                    layout_lds=layout_lds,
                                    row_local=row_local,
                                    col_local_i32=col_local_i32,
                                    tx_c4=arith.index(4),
                                    k_blocks16=k_blocks16,
                                    lds_base=lds_base,
                                    vec_part_i32x1=vec_x_in_parts[i],
                                    elem_bytes=elem_bytes,
                                )

                    if const_expr(use_async_copy):
                        dma_bytes = 16
                        wave_size = 64
                        eff_bytes_per_buffer = (
                            int(tile_m) * int(eff_lds_stride) * int(a_elem_bytes)
                        )
                        num_dma_loads = max(
                            1, eff_bytes_per_buffer // (total_threads * dma_bytes)
                        )
                        c_a_elem_bytes_dma = arith.constant(int(a_elem_bytes), index=True)
                        c_wave_dma_bytes = arith.constant(wave_size * dma_bytes, index=True)

                        def dma_x_tile_to_lds(base_k, lds_base):
                            c4_idx = arith.index(4)
                            base_k_div4 = (
                                (base_k // c_a_pack)
                                * arith.constant(int(elem_bytes), index=True)
                            ) // arith.index(4)

                            lds_ptr_i64 = None
                            for i in range_constexpr(num_dma_loads):
                                row_local_i = x_row_local[i]
                                col_local_i32_i = x_col_local_i32[i]
                                col_local_sw = swizzle_xor16(
                                    row_local_i, col_local_i32_i * c4_idx, k_blocks16
                                )
                                row_k_dw = x_row_base_div4[i] + base_k_div4
                                global_byte_idx = row_k_dw * c4_idx + col_local_sw
                                global_offset = arith.index_cast(T.i32, global_byte_idx)

                                if const_expr(i == 0):
                                    lds_addr = (
                                        memref.extract_aligned_pointer_as_index(lds_x)
                                        + lds_base * c_a_elem_bytes_dma
                                        + wave_id * c_wave_dma_bytes
                                    )
                                    lds_ptr_i64 = rocdl.readfirstlane(
                                        T.i64, arith.index_cast(T.i64, lds_addr)
                                    )
                                else:
                                    lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                        total_threads * dma_bytes, type=T.i64
                                    )

                                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                                rocdl.raw_ptr_buffer_load_lds(
                                    x_rsrc,
                                    lds_ptr,
                                    arith.constant(dma_bytes, type=T.i32),
                                    global_offset,
                                    arith.constant(0, type=T.i32),
                                    arith.constant(0, type=T.i32),
                                    arith.constant(0, type=T.i32),
                                )

                        def prefetch_x_to_lds(base_k, lds_base):
                            dma_x_tile_to_lds(base_k, lds_base)

                    def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                        col_base_swz_bytes = swizzle_xor16(
                            curr_row_a_lds, col_base, k_blocks16
                        )
                        col_base_swz = (
                            col_base_swz_bytes
                            if elem_bytes == 1
                            else (col_base_swz_bytes // arith.index(2))
                        )
                        idx_a16 = crd2idx(
                            [fx.Int32(curr_row_a_lds), fx.Int32(col_base_swz)], layout_lds
                        )
                        idx_a16 = idx_a16 + lds_base
                        loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                        a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                        a0 = vector.extract(
                            a_i64x2, static_position=[0], dynamic_position=[]
                        )
                        a1 = vector.extract(
                            a_i64x2, static_position=[1], dynamic_position=[]
                        )
                        return a0, a1

                    def compute_tile(
                        acc_in,
                        b_tile_in,
                        lds_base,
                        a_scale=None,
                        b_scale=None,
                        *,
                        prefetch_epilogue: bool = False,
                        a0_prefetch=None,
                        a1_prefetch=None,
                        b_hi_loader=None,
                        n_scale_shift_p=None,
                        ku_count=None,
                    ):
                        nss = (
                            n_scale_shift_p
                            if n_scale_shift_p is not None
                            else n_scale_shift_i32
                        )
                        # PR3117 stage2 NaN fix: restrict the MFMA K-loop to the valid
                        # to mfma_scale, eliminating 0*NaN propagation into the output.
                        ku_loop = k_unroll if ku_count is None else ku_count
                        if const_expr(b_hi_loader is not None):
                            b_tile_full = [None] * k_unroll
                            for i in range_constexpr(b_split_ku):
                                b_tile_full[i] = b_tile_in[i]
                        else:
                            b_tile_full = b_tile_in
                        acc_list = list(acc_in)
                        epilogue_pf = None
                        bias = None
                        if const_expr(prefetch_epilogue):
                            if const_expr(enable_bias):
                                bias = []
                                for ni in range_constexpr(body_num_acc_n):
                                    global_n = (
                                        body_by_n + n_tile_base + ni * 16 + lane_mod_16
                                    )
                                    bias_offset = expert_off_idx + global_n
                                    bias.append(load_bias_scalar(bias_rsrc, bias_offset))
                            tw_pf = None
                            if const_expr(doweight_stage2):
                                tw_pf = []
                                lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                                vec4_f32_pf = T.vec(4, f32)
                                if const_expr(lds_tw is not None):
                                    for mi in range_constexpr(m_repeat):
                                        mi_base_pf = arith.constant(mi * 16, index=True)
                                        lds_row_pf = mi_base_pf + lane_div_16_mul4_pf
                                        tw_v4 = vector.load_op(
                                            vec4_f32_pf, lds_tw, [lds_row_pf]
                                        )
                                        for ii in range_constexpr(4):
                                            tw_pf.append(
                                                vector.extract(
                                                    tw_v4,
                                                    static_position=[ii],
                                                    dynamic_position=[],
                                                )
                                            )
                                else:
                                    for mi in range_constexpr(m_repeat):
                                        mi_base_pf = arith.constant(mi * 16, index=True)
                                        base_row_pf = (
                                            bx_m + mi_base_pf + lane_div_16_mul4_pf
                                        )
                                        tw_v4 = buffer_ops.buffer_load(
                                            sorted_w_rsrc,
                                            base_row_pf,
                                            vec_width=4,
                                            dtype=f32,
                                        )
                                        for ii in range_constexpr(4):
                                            tw_pf.append(
                                                vector.extract(
                                                    tw_v4,
                                                    static_position=[ii],
                                                    dynamic_position=[],
                                                )
                                            )
                            epilogue_pf = (None, tw_pf, bias)

                        c0_i64 = arith.constant(0, type=T.i64)
                        vec4_i64 = T.vec(4, T.i64)
                        vec8_i32 = T.vec(8, T.i32)

                        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                            v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                            return vector.bitcast(vec8_i32, v4)

                        pack_K_shift = (pack_K - 1).bit_length()
                        pack_K_mask = pack_K - 1

                        xdl_arb_hint = (
                            int(tile_m) == 64
                            and int(body_tile_n) == 128
                            and int(tile_k) == 256
                            and bool(use_async_copy)
                            and a_elem_vec_pack > 1
                        )
                        if const_expr(xdl_arb_hint):
                            rocdl.disable_xdl_arb_stall()

                        if const_expr(b_hi_loader is not None):
                            b_hi = b_hi_loader()
                            for bhi_i in range_constexpr(len(b_hi)):
                                b_tile_full[b_split_ku + bhi_i] = b_hi[bhi_i]

                        rocdl.s_setprio(1)

                        for k_idx in range_constexpr(ku_loop):
                            ku128 = k_idx >> pack_K_shift
                            ikxdl = k_idx & pack_K_mask

                            b_packs = b_tile_full[k_idx]
                            b_packs0 = b_packs[0]
                            b_packs1 = b_packs[1]
                            if const_expr(body_b_has_full_operand):
                                b_packs2 = b_packs[2]
                                b_packs3 = b_packs[3]

                            col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack

                            for mi in range_constexpr(m_repeat_packed):
                                a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                                a_scale_val = vector.extract(
                                    a_scale_i32, static_position=[0], dynamic_position=[]
                                )
                                if const_expr(m_scale_shift_i32 is not None):
                                    a_scale_val = arith.shrui(
                                        a_scale_val, m_scale_shift_i32
                                    )
                                a128_list = [None] * pack_M
                                for imxdl in range_constexpr(pack_M):
                                    col_base0 = col_base
                                    mi_idx = mi * pack_M + imxdl
                                    mi_val = arith.constant(mi_idx * 16, index=True)
                                    curr_row_a_lds = row_a_lds + mi_val

                                    if const_expr(
                                        (a0_prefetch is not None)
                                        and (k_idx == 0)
                                        and (mi_idx == 0)
                                    ):
                                        a0, a1 = a0_prefetch
                                    elif const_expr(
                                        (a1_prefetch is not None)
                                        and (k_idx == 1)
                                        and (mi_idx == 0)
                                    ):
                                        a0, a1 = a1_prefetch
                                    else:
                                        a0, a1 = lds_load_packs_k64(
                                            curr_row_a_lds, col_base0, lds_base
                                        )

                                    if const_expr(is_f8_a):
                                        col_base1 = col_base + 64
                                        a2, a3 = lds_load_packs_k64(
                                            curr_row_a_lds, col_base1, lds_base
                                        )
                                        a128_list[imxdl] = pack_i64x4_to_i32x8(
                                            a0, a1, a2, a3
                                        )
                                    else:
                                        a128_list[imxdl] = pack_i64x4_to_i32x8(
                                            a0, a1, c0_i64, c0_i64
                                        )

                                for ni in range_constexpr(num_acc_n_packed):
                                    b_scale_i32 = b_scale[ku128 * num_acc_n_packed + ni]
                                    b_scale_val = vector.extract(
                                        b_scale_i32,
                                        static_position=[0],
                                        dynamic_position=[],
                                    )
                                    if const_expr(nss is not None):
                                        b_scale_val = arith.shrui(b_scale_val, nss)

                                    b128_list = [None] * pack_N
                                    for inxdl in range_constexpr(pack_N):
                                        ni_idx = ni * pack_N + inxdl
                                        if const_expr(body_b_has_full_operand):
                                            b128_list[inxdl] = pack_i64x4_to_i32x8(
                                                b_packs0[ni_idx],
                                                b_packs1[ni_idx],
                                                b_packs2[ni_idx],
                                                b_packs3[ni_idx],
                                            )
                                        else:
                                            b128_list[inxdl] = pack_i64x4_to_i32x8(
                                                b_packs0[ni_idx],
                                                b_packs1[ni_idx],
                                                c0_i64,
                                                c0_i64,
                                            )

                                    for imxdl in range_constexpr(pack_M):
                                        mi_idx = mi * pack_M + imxdl
                                        a128 = a128_list[imxdl]

                                        for inxdl in range_constexpr(pack_N):
                                            ni_idx = ni * pack_N + inxdl

                                            b128 = b128_list[inxdl]

                                            acc_idx = mi_idx * body_num_acc_n + ni_idx
                                            acc_list[acc_idx] = mixed_b_mfma(
                                                a128,
                                                b128,
                                                acc_list[acc_idx],
                                                ikxdl * scale_pack_m + imxdl,
                                                a_scale_val,
                                                ikxdl * scale_pack_n + inxdl,
                                                b_scale_val,
                                            )

                        return acc_list, epilogue_pf

                    lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                    lds_base_cur = arith.index(0)
                    lds_base_nxt = lds_tile_elems

                    rocdl.sched_barrier(0)

                    def hot_loop_scheduler():
                        mfma_group = body_num_acc_n
                        mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                        mfma_per_iter = 2 * mfma_group
                        sche_iters = (
                            0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                        )

                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16):
                            rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16):
                            rocdl.sched_vmem(1)
                        if const_expr(body_num_acc_n < 4):
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(1)
                            if const_expr(tile_m == 16):
                                rocdl.sched_vmem(1)
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(1)
                            if const_expr(tile_m == 16):
                                rocdl.sched_vmem(1)
                            rocdl.sched_mfma(1)

                        if const_expr(use_async_copy):
                            dswr_tail = 0
                        else:
                            dswr_tail = num_x_loads
                            if const_expr(dswr_tail > sche_iters):
                                dswr_tail = sche_iters
                        dswr_start = sche_iters - dswr_tail

                        for sche_i in range_constexpr(sche_iters):
                            rocdl.sched_vmem(1)
                            rocdl.sched_mfma(mfma_group)
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(mfma_group)
                            if const_expr(dswr_tail > 0 and sche_i >= dswr_start - 1):
                                rocdl.sched_dswr(1)

                        rocdl.sched_barrier(0)

                    def k_shift_bits(k_py):
                        if const_expr(pack_K >= scale_pack_k):
                            return 0
                        return ((k_py // 128) % scale_pack_k) * scale_pack_m * 8

                    def k_base(k_py):
                        return k_py // scale_pack_k // 128

                    row_stride_bytes_pre = int(model_dim) * int(out_elem_bytes)
                    use_buf_atomic_pre = bool(accumulate) and (
                        row_stride_bytes_pre <= 16384
                    )
                    c_tile_m_idx = arith.constant(tile_m, index=True)
                    tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, c_tile_m_idx)
                    r216_defer_tid = bool(
                        r139_xdma_first
                        and use_async_copy
                        and int(tile_m) == 64
                        and int(body_tile_n) == 128
                        and int(tile_k) == 256
                    )

                    def emit_tid_lds_prologue():
                        if_tid = scf.IfOp(tid_in_range)
                        with ir.InsertionPoint(if_tid.then_block):
                            tid_row = bx_m + tx
                            tid_val = buffer_ops.buffer_load(
                                sorted_rsrc, tid_row, vec_width=1, dtype=T.i32
                            )
                            if const_expr(doweight_stage2):
                                tw_val_m = buffer_ops.buffer_load(
                                    sorted_w_rsrc, tid_row, vec_width=1, dtype=f32
                                )
                            if const_expr(use_buf_atomic_pre):
                                t_pre = tid_val & arith.constant(0xFFFFFF, type=T.i32)
                                s_pre = arith.shrui(tid_val, arith.constant(24, type=T.i32))
                                row_byte_off = t_pre * arith.constant(
                                    row_stride_bytes_pre, type=T.i32
                                )
                                global_row_i32 = arith.index_cast(T.i32, tid_row)
                                valid = arith.andi(
                                    arith.andi(
                                        arith.cmpi(
                                            CmpIPredicate.ult,
                                            global_row_i32,
                                            num_valid_i32,
                                        ),
                                        arith.cmpi(
                                            CmpIPredicate.ult, t_pre, tokens_i32_guard
                                        ),
                                    ),
                                    arith.cmpi(
                                        CmpIPredicate.ult,
                                        s_pre,
                                        arith.constant(topk, type=T.i32),
                                    ),
                                )
                                stored_val = arith.select(
                                    valid,
                                    row_byte_off,
                                    arith.constant(0x7FFF0000, type=T.i32),
                                )
                            else:
                                stored_val = tid_val
                            tid_vec1 = vector.from_elements(T.vec(1, T.i32), [stored_val])
                            vector.store(tid_vec1, lds_tid, [tx])
                            if const_expr(doweight_stage2):
                                tw_vec1 = vector.from_elements(T.vec(1, T.f32), [tw_val_m])
                                vector.store(tw_vec1, lds_tw, [tx], alignment=4)
                            scf.YieldOp([])

                    if const_expr(not r216_defer_tid):
                        emit_tid_lds_prologue()

                    k0 = arith.index(0)
                    k0_bk = k0
                    if const_expr(r139_xdma_first):
                        prefetch_x_to_lds(k0, lds_base_cur)
                        rocdl.sched_barrier(0)
                    if const_expr(b_split_enabled):
                        b_cur = load_b_tile_lo(k0_bk)
                    else:
                        b_cur = load_b_tile(k0_bk)
                    a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                        k_base(0), k_shift_bits(0)
                    )
                    rocdl.sched_barrier(0)
                    if const_expr(not r139_xdma_first):
                        if const_expr(use_async_copy):
                            prefetch_x_to_lds(k0, lds_base_cur)
                        else:
                            x_regs0 = load_x_tile(k0)
                            store_x_tile_to_lds(x_regs0, lds_base_cur)
                    elif const_expr(not use_async_copy):
                        x_regs0 = load_x_tile(k0)
                        store_x_tile_to_lds(x_regs0, lds_base_cur)
                    if const_expr(r216_defer_tid):
                        emit_tid_lds_prologue()
                        rocdl.sched_barrier(0)
                    gpu.barrier()

                    acc = [acc_init] * body_num_acc_n * m_repeat
                    lds_base_pong = lds_base_cur
                    lds_base_ping = lds_base_nxt

                    a0_prefetch_pong = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_base_pong
                    )
                    a1_col_base = col_offset_base + 128 // a_elem_vec_pack
                    a1_prefetch_pong = (
                        lds_load_packs_k64(row_a_lds, a1_col_base, lds_base_pong)
                        if pack_K >= 2
                        else None
                    )

                    num_k_tiles_py = num_k_tiles_per_batch
                    odd_k_tiles = (num_k_tiles_py % 2) == 1
                    tail_tiles = 1 if odd_k_tiles else 2
                    k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                    if const_expr(k_main2_py < 0):
                        k_main2_py = 0

                    # uninitialized -> feeding them to mfma_scale yields NaN.  Skip
                    K_per_ku_s2 = int(tile_k) // int(k_unroll)
                    pad_ku_skip_s2 = (
                        min(int(k_unroll), int(inter_dim_pad) // K_per_ku_s2)
                        if inter_dim_pad > 0
                        else 0
                    )
                    tail_ku_s2 = int(k_unroll) - pad_ku_skip_s2

                    c2_tile_k = arith.constant(tile_k * 2, index=True)
                    b_pong = b_cur
                    k0_pong_bk = k0_bk

                    # would create a region whose internal SSA values cannot be used
                    def make_b_hi_loader(base_k):
                        """Create a b_hi_loader callable for a given base_k."""
                        return lambda bk=base_k: load_b_tile_hi(bk)

                    if const_expr(k_main2_py > 0):
                        for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                            k_iv = arith.index(k_iv_py)
                            next_k1 = k_iv + tile_k
                            next_k1_py = k_iv_py + tile_k
                            next_k1_bk = next_k1 // b_byte_div
                            if const_expr(use_async_copy):
                                prefetch_x_to_lds(next_k1, lds_base_ping)
                            else:
                                x_regs_ping = load_x_tile(next_k1)
                            a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                                k_base(next_k1_py), k_shift_bits(next_k1_py)
                            )
                            b_ping_lo = (
                                load_b_tile_lo(next_k1_bk)
                                if b_split_enabled
                                else load_b_tile(next_k1_bk)
                            )

                            acc, _ = compute_tile(
                                acc,
                                b_pong,
                                lds_base_pong,
                                a_scale_pong,
                                b_scale_pong,
                                a0_prefetch=a0_prefetch_pong,
                                a1_prefetch=a1_prefetch_pong,
                                b_hi_loader=(
                                    make_b_hi_loader(k0_pong_bk)
                                    if b_split_enabled
                                    else None
                                ),
                            )
                            if const_expr(not use_async_copy):
                                store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                            gpu.barrier()

                            a0_prefetch_ping = lds_load_packs_k64(
                                row_a_lds, col_offset_base, lds_base_ping
                            )
                            a1_prefetch_ping = (
                                lds_load_packs_k64(row_a_lds, a1_col_base, lds_base_ping)
                                if pack_K >= 2
                                else None
                            )

                            next_k2 = k_iv + c2_tile_k
                            next_k2_py = k_iv_py + tile_k * 2
                            next_k2_bk = next_k2 // b_byte_div
                            if const_expr(use_async_copy):
                                prefetch_x_to_lds(next_k2, lds_base_pong)
                            else:
                                x_regs_pong = load_x_tile(next_k2)
                            a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                                k_base(next_k2_py), k_shift_bits(next_k2_py)
                            )
                            b_pong = (
                                load_b_tile_lo(next_k2_bk)
                                if b_split_enabled
                                else load_b_tile(next_k2_bk)
                            )

                            acc, _ = compute_tile(
                                acc,
                                b_ping_lo,
                                lds_base_ping,
                                a_scale_ping,
                                b_scale_ping,
                                a0_prefetch=a0_prefetch_ping,
                                a1_prefetch=a1_prefetch_ping,
                                b_hi_loader=(
                                    make_b_hi_loader(next_k1_bk)
                                    if b_split_enabled
                                    else None
                                ),
                            )
                            k0_pong_bk = next_k2_bk
                            if const_expr(not use_async_copy):
                                store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                            gpu.barrier()

                            a0_prefetch_pong = lds_load_packs_k64(
                                row_a_lds, col_offset_base, lds_base_pong
                            )
                            a1_prefetch_pong = (
                                lds_load_packs_k64(row_a_lds, a1_col_base, lds_base_pong)
                                if pack_K >= 2
                                else None
                            )

                    if const_expr(odd_k_tiles):
                        acc, epilogue_pf = compute_tile(
                            acc,
                            b_pong,
                            lds_base_pong,
                            a_scale_pong,
                            b_scale_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a1_prefetch=a1_prefetch_pong,
                            prefetch_epilogue=True,
                            ku_count=tail_ku_s2,
                            b_hi_loader=(
                                make_b_hi_loader(k0_pong_bk) if b_split_enabled else None
                            ),
                        )

                    else:
                        k_tail1 = (k_in + tile_k - 1) // tile_k * tile_k - tile_k
                        k_tail1_py = (
                            int(inter_dim) + tile_k - 1
                        ) // tile_k * tile_k - tile_k
                        k_tail1_bk = k_tail1 // b_byte_div
                        if const_expr(use_async_copy):
                            prefetch_x_to_lds(k_tail1, lds_base_ping)
                        else:
                            x_regs_ping = load_x_tile(k_tail1)
                        b_ping_lo = (
                            load_b_tile_lo(k_tail1_bk)
                            if b_split_enabled
                            else load_b_tile(k_tail1_bk)
                        )
                        a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                            k_base(k_tail1_py), k_shift_bits(k_tail1_py)
                        )

                        acc, _ = compute_tile(
                            acc,
                            b_pong,
                            lds_base_pong,
                            a_scale_pong,
                            b_scale_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a1_prefetch=a1_prefetch_pong,
                            b_hi_loader=(
                                make_b_hi_loader(k0_pong_bk) if b_split_enabled else None
                            ),
                        )

                        if const_expr(not use_async_copy):
                            store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        gpu.barrier()

                        a0_prefetch_ping = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_ping
                        )
                        a1_prefetch_ping = (
                            lds_load_packs_k64(row_a_lds, a1_col_base, lds_base_ping)
                            if pack_K >= 2
                            else None
                        )
                        acc, epilogue_pf = compute_tile(
                            acc,
                            b_ping_lo,
                            lds_base_ping,
                            a_scale_ping,
                            b_scale_ping,
                            a0_prefetch=a0_prefetch_ping,
                            a1_prefetch=a1_prefetch_ping,
                            prefetch_epilogue=True,
                            ku_count=tail_ku_s2,
                            b_hi_loader=(
                                make_b_hi_loader(k_tail1_bk) if b_split_enabled else None
                            ),
                        )

                    tw_pf = None
                    bias_pf = None
                    if const_expr(epilogue_pf is not None):
                        _, tw_pf, bias_pf = epilogue_pf

                    mask24_i32 = arith.constant(0xFFFFFF)
                    topk_i32_v = topk_i32

                    zero_i32 = arith.constant(0)

                    def atomic_add_f16x2(val_f16x2, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f16x2,
                            out_rsrc,
                            byte_off_i32,
                            zero_i32,
                            zero_i32,
                        )

                    if const_expr(lds_out is None):
                        raise RuntimeError(
                            "FLIR_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                        )

                    out_base_i64 = arith.index_cast(T.i64, fx.ptrtoint(arg_out))
                    out_base_idx = arith.index_cast(T.index, out_base_i64)

                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        if const_expr(doweight_stage2):
                            tw_idx = (mi * 4) + ii
                            if const_expr(tw_pf is not None):
                                tw = tw_pf[tw_idx]
                            else:
                                tw = buffer_ops.buffer_load(
                                    sorted_w_rsrc, row, vec_width=1, dtype=f32
                                )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            if const_expr(enable_bias):
                                v = v + bias_pf[ni]

                            if const_expr(doweight_stage2):
                                v = v * tw
                            v_out = arith.trunc_f(out_elem(), v)

                            lds_idx = row_base_lds + col_local
                            vec1_out = T.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])

                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    row_stride_bytes_py = int(model_dim) * int(out_elem_bytes)
                    use_buf_atomic = bool(accumulate) and (row_stride_bytes_py <= 16384)
                    out_elem_bytes_i32 = (
                        arith.constant(int(out_elem_bytes), type=T.i32)
                        if use_buf_atomic
                        else None
                    )

                    def precompute_row(*, row_local, row):
                        if const_expr(use_buf_atomic):
                            row_byte_off_i32 = memref.load(lds_tid, [row_local])
                            return (
                                (None, None, row_byte_off_i32),
                                None,
                            )
                        fused2 = memref.load(lds_tid, [row_local])
                        row_i32 = arith.index_cast(T.i32, row)
                        row_valid0 = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32)
                        s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                        row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                        t_idx = arith.index_cast(ir.IndexType.get(), t)
                        s_idx = arith.index_cast(ir.IndexType.get(), s)
                        ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                        if const_expr(accumulate):
                            row_byte_base = out_base_idx + t_idx * arith.constant(
                                model_dim * out_elem_bytes, index=True
                            )
                        elif const_expr(need_fp8_out):
                            row_byte_base = out_base_idx + ts_idx * arith.constant(
                                out_row_bytes_const, index=True
                            )
                        else:
                            row_byte_base = out_base_idx + ts_idx * arith.constant(
                                model_dim * out_elem_bytes, index=True
                            )
                        row_byte_off_i32 = None
                        return ((fused2, row_byte_base, row_byte_off_i32), row_valid)

                    def idx_to_llvm_ptr(idx_val, addr_space=1):
                        """Convert an index-typed byte address to !llvm.ptr<addr_space>."""
                        idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
                        i64_v = arith.index_cast(T.i64, idx_v)
                        i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
                        ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                        return llvm.inttoptr(ptr_ty, i64_raw)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        _fused, row_byte_base, row_byte_off_i32 = row_ctx
                        if const_expr(need_fp8_out):
                            # frag is vector<e_vec x f32>; e_vec==8 == one opus
                            # 8-col MXFP8 group. Quantize to e4m3 + one e8m0 byte.
                            c0_i32_q = fx.Int32(0)
                            c1_i32_q = fx.Int32(1)
                            c7_i32_q = fx.Int32(7)
                            c23_i32_q = fx.Int32(23)
                            c254_i32_q = fx.Int32(254)
                            c255_i32_q = fx.Int32(0xFF)
                            c0_f32_q = fx.Float32(0.0)
                            frag_vec = fx.Vector(frag)
                            frag_vals = []
                            for i in range_constexpr(e_vec):
                                frag_vals.append(frag_vec[i].to(fx.Float32))
                            local_max = c0_f32_q
                            for i in range_constexpr(e_vec):
                                abs_v = fx.Float32(
                                    llvm.call_intrinsic(
                                        f32,
                                        "llvm.fabs.f32",
                                        [frag_vals[i].ir_value()],
                                        [],
                                        [],
                                    )
                                )
                                local_max = local_max.maximumf(abs_v)
                            # opus: E = bf16/f32 biased exp(amax) - 7, clamp E>=1,
                            # E=0 when amax==0. scale byte = E; dequant = fp8*2^(E-127).
                            amax_bits = local_max.bitcast(fx.Int32)
                            ax_e = (amax_bits >> c23_i32_q) & c255_i32_q
                            E = ax_e - c7_i32_q
                            E = (E > c1_i32_q).select(E, c1_i32_q)
                            E = (amax_bits == c0_i32_q).select(c0_i32_q, E)
                            quant_scale = ((c254_i32_q - E) << c23_i32_q).bitcast(
                                fx.Float32
                            )
                            scaled_vals = []
                            for i in range_constexpr(e_vec):
                                scaled_vals.append(frag_vals[i] * quant_scale)
                            ptr_addr_idx = row_byte_base + col_g0
                            for wg in range_constexpr(e_vec // 4):
                                b = wg * 4
                                packed_w = c0_i32_q
                                packed_w = rocdl.cvt_pk_fp8_f32(
                                    T.i32, scaled_vals[b], scaled_vals[b + 1], packed_w, 0
                                )
                                packed_w = rocdl.cvt_pk_fp8_f32(
                                    T.i32,
                                    scaled_vals[b + 2],
                                    scaled_vals[b + 3],
                                    packed_w,
                                    1,
                                )
                                word_ptr = ptr_addr_idx + arith.constant(wg * 4, index=True)
                                out_ptr_v = idx_to_llvm_ptr(word_ptr)
                                packed_raw = (
                                    packed_w._value
                                    if hasattr(packed_w, "_value")
                                    else packed_w
                                )
                                llvm.StoreOp(
                                    packed_raw, out_ptr_v, alignment=4, nontemporal=True
                                )
                            # e8m0 scale byte at [model_dim + col_g0/8].
                            scale_byte_idx = (
                                row_byte_base
                                + arith.constant(model_dim, index=True)
                                + (col_g0 // arith.constant(8, index=True))
                            )
                            scale_ptr_v = idx_to_llvm_ptr(scale_byte_idx)
                            e8m0_raw = fx.Int8(E).ir_value()
                            llvm.StoreOp(
                                e8m0_raw, scale_ptr_v, alignment=1, nontemporal=True
                            )
                        elif const_expr(not bool(accumulate)):
                            col_idx = col_g0
                            byte_off_col = col_idx * arith.constant(
                                out_elem_bytes, index=True
                            )
                            ptr_addr_idx = row_byte_base + byte_off_col
                            out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                            frag_v = frag._value if hasattr(frag, "_value") else frag
                            llvm.StoreOp(
                                frag_v,
                                out_ptr_v,
                                alignment=e_vec * out_elem_bytes,
                                nontemporal=True,
                            )
                        elif const_expr(use_buf_atomic):
                            col_i32 = arith.index_cast(T.i32, col_g0)
                            col_byte_off_i32 = col_i32 * out_elem_bytes_i32
                            byte_off_i32 = row_byte_off_i32 + col_byte_off_i32
                            rocdl.raw_ptr_buffer_atomic_fadd(
                                frag,
                                out_rsrc,
                                byte_off_i32,
                                zero_i32,
                                zero_i32,
                            )
                        else:
                            col_idx = col_g0
                            byte_off_col = col_idx * arith.constant(
                                out_elem_bytes, index=True
                            )
                            ptr_addr_idx = row_byte_base + byte_off_col
                            out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                            frag_v = frag._value if hasattr(frag, "_value") else frag
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                out_ptr_v,
                                frag_v,
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent",
                                alignment=e_vec * out_elem_bytes,
                            )

                    e_vec = 2 if accumulate else min(body_tile_n // 32, 8)
                    rocdl.s_setprio(3)
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=body_tile_n,
                        e_vec=e_vec,
                        m_repeat=m_repeat,
                        num_acc_n=body_num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=body_by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=(
                            ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get()
                        ),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
                    rocdl.s_setprio(0)

                all_valid = arith.andi(blk_valid, arith.andi(exp_valid, tile_has_tokens))

                def emit_moe_gemm2_body():
                    if const_expr(heterogeneous_b):
                        format_if = scf.IfOp(is_shared_expert, has_else=True)
                        with ir.InsertionPoint(format_if.then_block):
                            if const_expr(serial_shared_n):
                                moe_gemm2_then_body(shared_b=True, shared_n_half=0)
                                rocdl.s_waitcnt(0)
                                barrier(vmcnt=0, lgkmcnt=0)
                                moe_gemm2_then_body(shared_b=True, shared_n_half=1)
                            else:
                                moe_gemm2_then_body(shared_b=True)
                            scf.YieldOp([])
                        with ir.InsertionPoint(format_if.else_block):
                            moe_gemm2_then_body(shared_b=False)
                            scf.YieldOp([])
                    else:
                        moe_gemm2_then_body()

                if const_expr(persistent):
                    cur_active = arith.andi(still_active, blk_valid)
                    do_gemm = arith.andi(cur_active, arith.andi(exp_valid, tile_has_tokens))
                    if_valid = scf.IfOp(do_gemm)
                    with ir.InsertionPoint(if_valid.then_block):
                        emit_moe_gemm2_body()
                        scf.YieldOp([])

                    gpu.barrier()
                    scf.YieldOp([cur_active])
                else:
                    if_valid = scf.IfOp(all_valid)
                    with ir.InsertionPoint(if_valid.then_block):
                        emit_moe_gemm2_body()
                        scf.YieldOp([])

                    gpu.barrier()
                    scf.YieldOp([expert_i32, expert_b_base])
                for_ip.__exit__(None, None, None)

        if heterogeneous_b:

            @flyc.kernel(name=module_name)
            def moe_gemm2(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_shared_w: fx.Pointer,
                arg_shared_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_num_valid_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_n_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
            ):
                _emit_moe_gemm2(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_shared_w,
                    arg_shared_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_num_valid_ids,
                    arg_bias,
                    i32_tokens_in,
                    i32_n_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                )

        else:

            @flyc.kernel(name=module_name)
            def moe_gemm2(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_num_valid_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_n_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
            ):
                _emit_moe_gemm2(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_w,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_num_valid_ids,
                    arg_bias,
                    i32_tokens_in,
                    i32_n_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                )

        cache_tag = (
            module_name,
            a_dtype,
            b_dtype,
            out_dtype,
            tile_m,
            tile_n,
            tile_k,
            doweight_stage2,
            accumulate,
            enable_bias,
            model_dim_pad,
            inter_dim_pad,
            use_cshuffle_epilog,
            persist_m,
            _sort_block_m,
            cu_num if persistent else 0,
            waves_per_eu,
            use_async_copy,
            xcd_swizzle,
        )
        if heterogeneous_b:
            cache_tag += (shared_expert_id,)

        def _launch_mixed_moe_gemm2(
            arg_out: fx.Pointer,
            arg_x: fx.Pointer,
            arg_w: fx.Pointer,
            arg_scale_x: fx.Pointer,
            arg_scale_w: fx.Pointer,
            arg_shared_w: fx.Pointer,
            arg_shared_scale_w: fx.Pointer,
            arg_sorted_token_ids: fx.Pointer,
            arg_expert_ids: fx.Pointer,
            arg_sorted_weights: fx.Pointer,
            arg_num_valid_ids: fx.Pointer,
            arg_bias: fx.Pointer,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
            stream: fx.Stream,
        ):
            _ = cache_tag
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()

            n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            tile_n_idx = arith.constant(tile_n, index=True)
            model_dim_pad_idx = arith.constant(model_dim_pad, index=True)
            gx = (
                n_in - model_dim_pad_idx + tile_n_idx - arith.constant(1, index=True)
            ) // tile_n_idx
            if const_expr(persistent):
                gy = arith.constant(cu_num, index=True)
            else:
                c_pm_l = arith.constant(persist_m, index=True)
                gy = (
                    arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
                    + c_pm_l
                    - arith.constant(1, index=True)
                ) // c_pm_l

            if const_expr(heterogeneous_b):
                launcher = moe_gemm2(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_shared_w,
                    arg_shared_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_num_valid_ids,
                    arg_bias,
                    i32_tokens_in,
                    i32_n_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                )
            else:
                launcher = moe_gemm2(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_num_valid_ids,
                    arg_bias,
                    i32_tokens_in,
                    i32_n_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                )
            if const_expr(waves_per_eu is not None):
                wpe = int(waves_per_eu)
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, wpe)
            launcher.launch(
                grid=(gx, gy, 1),
                block=(256, 1, 1),
                stream=stream,
            )

        if heterogeneous_b:

            @flyc.jit
            def launch_mixed_moe_gemm2(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_shared_w: fx.Pointer,
                arg_shared_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_num_valid_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_n_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
                stream: fx.Stream,
            ):
                _launch_mixed_moe_gemm2(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_shared_w,
                    arg_shared_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_num_valid_ids,
                    arg_bias,
                    i32_tokens_in,
                    i32_n_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                    stream,
                )

        else:

            @flyc.jit
            def launch_mixed_moe_gemm2(
                arg_out: fx.Pointer,
                arg_x: fx.Pointer,
                arg_w: fx.Pointer,
                arg_scale_x: fx.Pointer,
                arg_scale_w: fx.Pointer,
                arg_sorted_token_ids: fx.Pointer,
                arg_expert_ids: fx.Pointer,
                arg_sorted_weights: fx.Pointer,
                arg_num_valid_ids: fx.Pointer,
                arg_bias: fx.Pointer,
                i32_tokens_in: fx.Int32,
                i32_n_in: fx.Int32,
                i32_k_in: fx.Int32,
                i32_size_expert_ids_in: fx.Int32,
                stream: fx.Stream,
            ):
                _launch_mixed_moe_gemm2(
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_w,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_num_valid_ids,
                    arg_bias,
                    i32_tokens_in,
                    i32_n_in,
                    i32_k_in,
                    i32_size_expert_ids_in,
                    stream,
                )

        return launch_mixed_moe_gemm2
    return _types.SimpleNamespace(**dict(locals()))
_M_mixed_moe_gemm_2stage_common = _mod_mixed_moe_gemm_2stage_common()



# ----------------------------------------------------------------------------------------------------------------
# Host side: routing sort, MXFP4 activation quantization and the two launches.
# ----------------------------------------------------------------------------------------------------------------

_MODEL_DIM = 6144
_INTER_DIM = 768


def _config(n_tokens):
    # (tile_m, stage-1 tile_n, stage-1 waves_per_eu, stage-2 tile_n); one sort block serves both stages.
    if n_tokens <= 1024:
        return 32, 128, 3, 128
    if n_tokens <= 8192:
        return 64, 128, 3, 128
    return 64, 128, 3, 128


@_functools.cache
def _compiled(n_tokens, experts, topk):
    tile_m, s1_tn, s1_wpe, s2_tn = _config(n_tokens)
    common = _M_mixed_moe_gemm_2stage_common
    exe1 = common.compile_mixed_moe_gemm1_common(
        model_dim=_MODEL_DIM, inter_dim=_INTER_DIM, experts=experts, topk=topk, tile_m=tile_m, tile_n=s1_tn,
        tile_k=256, doweight_stage1=False, a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", act="swiglu",
        use_cshuffle_epilog=None, enable_bias=False, model_dim_pad=0, inter_dim_pad=0, persist_m=1,
        use_async_copy=False, waves_per_eu=s1_wpe, k_batch=1, b_nt=2, gate_mode=_GateMode.SEPARATED,
        a_scale_one=False, xcd_swizzle=0, k_wave=1, v2_output_layout=False)
    max_sorted = n_tokens * topk + experts * (tile_m - 1)
    m_blocks = (max_sorted + tile_m - 1) // tile_m
    s2_blocks = min(m_blocks, n_tokens * topk)
    exe2 = common.compile_mixed_moe_gemm2_common(
        model_dim=_MODEL_DIM, inter_dim=_INTER_DIM, experts=experts, topk=topk, tile_m=tile_m, tile_n=s2_tn,
        tile_k=256, doweight_stage2=True, a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        use_cshuffle_epilog=None, accumulate=True, enable_bias=False, model_dim_pad=0, inter_dim_pad=0,
        persist_m=(-1 if s2_blocks > 256 else 1), sort_block_m=0, waves_per_eu=None, use_async_copy=False,
        cu_num_mul=1, b_nt=0, xcd_swizzle=0)
    return tile_m, m_blocks, s2_blocks, exe1, exe2


def _launch(exe, args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


def _ptr(t):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _quant(x, grid_t):
    # MXFP4 per 1 x 32 as the reference rounds: E8M0 exponent from the block amax (rounded at 1.5x), round to
    # nearest e2m1 with ties to even code. Returns packed codes (rows, K/2) and E8M0 bytes (rows, K/32).
    rows, K = x.shape
    blocks = x.float().view(rows, K // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    mantissa, exponent = _torch.frexp(amax.clamp_min(1e-10))
    exponent = (exponent - 3 + (mantissa > 0.75).int()).clamp(-127, 127)
    scaled = blocks * _torch.exp2(-exponent.float())
    m = scaled.abs()
    lo = _torch.bucketize(m, grid_t, right=True) - 1
    hi = (lo + 1).clamp_max(7)
    glo, ghi = grid_t[lo], grid_t[hi]
    below, above = m - glo, ghi - m
    up = (above < below) | ((above == below) & (hi % 2 == 0))
    code = _torch.where(up, hi, lo)
    code = code | ((scaled < 0).long() << 3)
    code = code.view(rows, K // 2, 2)
    packed = (code[..., 0] | (code[..., 1] << 4)).to(_torch.uint8)
    e8m0 = (exponent + 127).to(_torch.uint8).view(rows, K // 32)
    return packed, e8m0


def _sorted_scales(e8m0, pos, n_rows):
    # E8M0 rows placed at their sorted positions, then shuffled [rows/32][cols/8][4][16][2 col half][2 row half].
    cols = e8m0.shape[1]
    rows_pad = (n_rows + 255) // 256 * 256
    cols_pad = (cols + 7) // 8 * 8
    s = _torch.zeros((rows_pad, cols_pad), dtype=_torch.uint8, device=e8m0.device)
    s[pos, :cols] = e8m0
    s = s.view(rows_pad // 32, 2, 16, cols_pad // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous()
    return s.view(-1)


def run(*, hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale, activation, quant_type,
        doweight_stage1, gate_mode, act_quant, expert_mask, beta, linear_beta, swiglu_limit, out_dtype, out):
    M, D = hidden_states.shape
    E = w1.shape[0]
    top = topk_ids.shape[1]
    R = M * top
    dev = hidden_states.device
    tile_m, m_blocks, s2_blocks, exe1, exe2 = _compiled(M, E, top)
    grid_t = _torch.arange(8, device=dev, dtype=_torch.float32)
    grid_t = _torch.where(grid_t < 4, 0.5 * grid_t, _torch.where(grid_t < 7, grid_t - 2.0, grid_t * 0 + 6.0))

    # Routing sort: expert-major, each expert padded to tile_m rows, routes in input order within an expert.
    ids = topk_ids.reshape(-1).long()
    counts = _torch.zeros(E, dtype=_torch.int64, device=dev).scatter_add(0, ids, _torch.ones_like(ids))
    padded = (counts + tile_m - 1) // tile_m * tile_m
    ends = _torch.cumsum(padded, 0)
    order = _torch.argsort(ids, stable=True)
    cstart = _torch.cumsum(counts, 0) - counts
    rank = _torch.empty_like(ids)
    rank[order] = _torch.arange(R, device=dev) - cstart[ids[order]]
    pos = (ends - padded)[ids] + rank
    n_sorted = m_blocks * tile_m
    r = _torch.arange(R, device=dev)
    sorted_ids = _torch.full((n_sorted,), (top << 24) | M, dtype=_torch.int32, device=dev)
    sorted_ids[pos] = (((r % top) << 24) | (r // top)).to(_torch.int32)
    sorted_w = _torch.zeros((n_sorted,), dtype=_torch.float32, device=dev)
    sorted_w[pos] = topk_weights.reshape(-1).float()
    block_start = _torch.arange(m_blocks, device=dev) * tile_m
    sorted_eids = _torch.searchsorted(ends, block_start, right=True).clamp_max(E - 1).to(_torch.int32)
    num_valid = _torch.stack([ends[-1], _torch.full_like(ends[-1], M)]).to(_torch.int32)

    # Stage 1: x quantized in token order, its scales in sorted order; gate/up GEMM with clamped SwiGLU, bf16 out.
    xq, xs = _quant(hidden_states, grid_t)
    xs_sorted = _sorted_scales(xs[r // top], pos, n_sorted)
    y = _torch.empty((M, top, _INTER_DIM), dtype=_torch.bfloat16, device=dev)
    empty_f32 = _torch.empty(1, dtype=_torch.float32, device=dev)
    empty_u8 = _torch.empty(1, dtype=_torch.uint8, device=dev)
    lim = float(swiglu_limit) if swiglu_limit else 7.0
    s1_blocks = min(m_blocks, R)
    _launch(exe1, (
        _ptr(y), _ptr(xq), _ptr(w1), _ptr(xs_sorted), _ptr(w1_scale), _ptr(sorted_ids), _ptr(sorted_eids),
        _ptr(sorted_w), _ptr(num_valid), _ptr(empty_f32), _ptr(empty_u8), M, 2 * _INTER_DIM, _MODEL_DIM,
        s1_blocks, 1.0, 1.0, 1.0, 1.0, lim, _torch.cuda.current_stream()))

    # Stage 2: y quantized per route, scales in sorted order; down GEMM weighted and accumulated atomically.
    yq, ys = _quant(y.view(R, _INTER_DIM), grid_t)
    ys_sorted = _sorted_scales(ys, pos, n_sorted)
    out.zero_()
    _launch(exe2, (
        _ptr(out), _ptr(yq), _ptr(w2), _ptr(ys_sorted), _ptr(w2_scale), _ptr(sorted_ids), _ptr(sorted_eids),
        _ptr(sorted_w), _ptr(num_valid), _ptr(empty_f32), M, _MODEL_DIM, _INTER_DIM, s2_blocks,
        _torch.cuda.current_stream()))
