# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import torch
    from torch import Tensor

_Dims = tuple[int | str, ...]  # a shape whose dims are ints or expression texts


class DType(str, Enum):
    """A logical element type."""

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT4_E2M1 = "float4_e2m1"
    FLOAT8_E8M0 = "float8_e8m0"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"


class DTypeTraits(NamedTuple):
    unit_roundoff: float  # 2**-p, p significand bits incl. the hidden bit; 0: integral
    largest_finite_value: float
    is_integral: bool
    storage_dtypes: tuple[str, ...]  # torch dtypes that may hold it; [0]: primary
    bits: int  # 4: two values per stored element of the last dim
    decoding: Mapping[str, Any]  # "kind" and the parameters that decoding reads


_CAST_DECODING = {"kind": "cast"}


def _float_traits(significand_bits, largest, bits, *storage, decoding=_CAST_DECODING):
    return DTypeTraits(2.0**-significand_bits, largest, False, storage, bits, decoding)


def _int_traits(bits, largest, name):
    return DTypeTraits(0.0, float(largest), True, (name,), bits, _CAST_DECODING)


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_DECODING = {
    "kind": "nibble_table",
    "decoded_values": _E2M1_MAGNITUDES + tuple(-v for v in _E2M1_MAGNITUDES),
}
_E8M0_DECODING = {"kind": "power_of_two", "exponent_bias": 127, "nan_code": 255}


# Keyed by dtype name, in ``DType`` order.
DTYPE_TRAITS: Mapping[str, DTypeTraits] = {
    "float64": _float_traits(53, 1.7976931348623157e308, 64, "float64"),
    "float32": _float_traits(24, 3.4028234663852886e38, 32, "float32"),
    "float16": _float_traits(11, 65504.0, 16, "float16"),
    "bfloat16": _float_traits(8, 3.3895313892515355e38, 16, "bfloat16"),
    "float8_e4m3fn": _float_traits(4, 448.0, 8, "float8_e4m3fn"),
    "float8_e5m2": _float_traits(3, 57344.0, 8, "float8_e5m2"),
    "float4_e2m1": _float_traits(2, 6.0, 4, "float4_e2m1fn_x2", "uint8", decoding=_E2M1_DECODING),
    "float8_e8m0": _float_traits(1, 2.0**127, 8, "float8_e8m0fnu", "uint8", decoding=_E8M0_DECODING),
    "int64": _int_traits(64, 2**63 - 1, "int64"),
    "int32": _int_traits(32, 2**31 - 1, "int32"),
    "int8": _int_traits(8, 127, "int8"),
    "uint8": _int_traits(8, 255, "uint8"),
    "bool": _int_traits(1, 1, "bool"),
}


def torch_dtypes(dtype: str) -> tuple[torch.dtype, ...]:
    """The storage dtypes of ``dtype`` this torch has; ``[0]`` is the primary one."""
    import torch

    names = DTYPE_TRAITS[dtype].storage_dtypes
    return tuple(getattr(torch, n) for n in names if hasattr(torch, n))


def to_float64(tensor: Tensor, dtype: str) -> Tensor:
    """The values ``tensor`` stores as ``dtype``, in float64; a 4-bit dtype doubles the
    last dim."""
    import torch

    match DTYPE_TRAITS[DType(dtype)].decoding:
        case {"kind": "cast"}:
            return tensor.double()
        case {"kind": "nibble_table", "decoded_values": decoded_values}:
            # two 4-bit codes per byte, low nibble first
            codes = tensor.view(torch.uint8).long()
            table = torch.tensor(decoded_values, dtype=torch.float64, device=tensor.device)
            return table[torch.stack((codes & 0xF, codes >> 4), dim=-1).flatten(-2)]
        case {
            "kind": "power_of_two",
            "exponent_bias": exponent_bias,
            "nan_code": nan_code,
        }:
            codes = tensor.view(torch.uint8)
            decoded = torch.exp2(codes.double() - exponent_bias)
            return decoded.masked_fill(codes == nan_code, math.nan)
        case _:
            raise ValueError(f"no decoding rule for {dtype}")


def to_float32(tensor: Tensor, dtype: str) -> Tensor:
    """``to_float64``'s values in float32, which holds every dtype's exactly but an int's
    past 2**24 and a float64's."""
    if DTYPE_TRAITS[DType(dtype)].decoding["kind"] == "cast":
        return tensor.float()
    return to_float64(tensor, dtype).float()  # 1-byte codes: their float64 is small


def from_float64(float64_tensor: Tensor, dtype: str) -> Tensor:
    """The inverse of ``to_float64`` for a logical-shape tensor, in the primary storage
    dtype on its device (-0 may become +0); a value ``dtype`` cannot hold exactly (NaN
    included) raises ``ValueError``."""
    import torch

    decoding = DTYPE_TRAITS[DType(dtype)].decoding  # an unknown name: ValueError
    shape = dtype_storage_shape(dtype, float64_tensor.shape)
    stored = torch.empty(shape, dtype=torch_dtypes(dtype)[0], device=float64_tensor.device)
    match decoding:
        case {"kind": "cast"}:
            stored.copy_(float64_tensor)
        case {"kind": "nibble_table", "decoded_values": decoded_values}:
            table = torch.tensor(decoded_values, dtype=torch.float64, device=float64_tensor.device)
            nibbles = (float64_tensor.unsqueeze(-1) == table).byte().argmax(-1).byte()
            stored.view(torch.uint8).copy_(nibbles[..., 0::2] | nibbles[..., 1::2] << 4)
        case {"kind": "power_of_two", "exponent_bias": exponent_bias}:
            codes = float64_tensor.log2() + exponent_bias
            stored.view(torch.uint8).copy_(codes.clamp(0, 255))
        case _:
            raise ValueError(f"no decoding rule for {dtype}")
    if not torch.equal(to_float64(stored, dtype), float64_tensor):
        raise ValueError(f"a {dtype} tensor cannot hold the values")
    return stored


def absolute_value(tensor: Tensor, dtype: str) -> Tensor:
    """``|x|`` in ``tensor``'s storage (int8 widens to int16): the sign bits cleared."""
    import torch

    DTYPE_TRAITS[dtype]  # an unknown name raises KeyError
    if dtype == "float4_e2m1":
        return (tensor.view(torch.uint8) & 0x77).view(tensor.dtype)  # both nibbles
    if dtype in ("float8_e4m3fn", "float8_e5m2"):
        return (tensor.view(torch.uint8) & 0x7F).view(tensor.dtype)
    if dtype in ("float8_e8m0", "uint8", "bool"):
        return tensor  # no sign
    if dtype == "int8":
        return tensor.to(torch.int16).abs()  # abs(-128) overflows int8
    return tensor.abs()


def _operand(dim: int | str) -> str:
    return str(dim) if isinstance(dim, int) or dim.isidentifier() else f"({dim})"


def dtype_storage_shape(dtype: str, shape: Sequence[int | str]) -> _Dims:
    """The row-major stored shape of logical ``shape``: a 4-bit dtype halves the last
    dim."""
    if DTYPE_TRAITS[dtype].bits != 4:
        return tuple(shape)
    *lead, last = shape
    return (*lead, last // 2 if isinstance(last, int) else f"{_operand(last)} // 2")
