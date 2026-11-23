# Created by OpenAI

"""Shared tensor specifications and rotary dimensions."""

from aklr.core.contract.semantics import DType, FiniteDistribution
from aklr.core.contract.spec import DTypeSpec, TensorSpec

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_INT64 = DTypeSpec(any_of=(DType.INT64,))
_FINITE = FiniteDistribution()


def _tensor(description, shape, dtype, *, mutable=False, bound=None, distribution=_FINITE):
    return TensorSpec(
        description=description,
        shape=shape,
        dtype=dtype,
        distribution=distribution,
        mutable=mutable,
        error_bounds=() if bound is None else bound,
    )


ROTARY_DIM = 64
MROPE_SECTION = (11, 11, 10)
