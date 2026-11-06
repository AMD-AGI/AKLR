# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Created by OpenAI
# Applies to BoundedCuSeqlensDistribution additions.

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import Field, TypeAdapter

from aklr.core.contract import expr

from ._dtypes import DTYPE_TRAITS, DType, to_float64, torch_dtypes
from ._frozen_model import FrozenModel

if TYPE_CHECKING:
    import torch
    from torch import Tensor


class Distribution(FrozenModel):
    """One item of a tensor input's ``distribution``: the values a kernel may assume."""

    kind: str
    # expression fields in declaration order (op.py unpacks ``lo, hi`` by position)
    expr_fields: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def accepts_dtype(self, dtype: str) -> bool:
        """Whether this kind is defined for ``dtype``, its decided parameters in range."""

    @abstractmethod
    def sample(
        self,
        shape: Sequence[int],
        dtype: str,
        generator: torch.Generator | None,
    ) -> Tensor:
        """A ``dtype`` tensor of ``shape`` (dtype storage shape) drawn with
        ``generator``."""

    @abstractmethod
    def contains(self, tensor: Tensor, dtype: str) -> bool:
        """Whether every value of ``tensor`` (row-major layout) lies in these values."""

    def constant_value(self, field_name: str) -> int | float | None:
        """The value of the expr field ``field_name`` when it reads no symbol, else
        None."""
        value = getattr(self, field_name)
        if not isinstance(value, str):
            return value
        closure, symbols = expr.compile(value)
        return None if symbols else closure({})


class FiniteDistribution(Distribution):
    """Any finite value of the dtype: never NaN or Inf, but maybe +-max, -0, a subnormal
    or int8's -128. A cast float is drawn as float32 N(0, 1), then converted; an int8,
    uint8, bool or fp4, every code finite, as uniform storage words."""

    kind: Literal["finite"] = "finite"

    def accepts_dtype(self, dtype):
        traits = DTYPE_TRAITS[dtype]
        if traits.is_integral or traits.decoding["kind"] == "nibble_table":
            return traits.bits <= 8
        return traits.decoding["kind"] == "cast"

    def sample(self, shape, dtype, generator):
        import torch

        traits, storage_dtype = DTYPE_TRAITS[dtype], torch_dtypes(dtype)[0]
        if dtype == "bool":  # torch.iinfo rejects bool; its only codes are 0 and 1
            return torch.randint(0, 2, shape, generator=generator).bool()
        if traits.is_integral:
            limits = torch.iinfo(storage_dtype)
            return torch.randint(
                limits.min,
                limits.max + 1,
                shape,
                generator=generator,
                dtype=storage_dtype,
            )
        if traits.decoding["kind"] == "nibble_table":
            words = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
            return words.view(storage_dtype)
        return torch.randn(shape, generator=generator).to(storage_dtype)

    def contains(self, tensor, dtype):
        return bool(to_float64(tensor, dtype).isfinite().all())


class LogUniformDistribution(Distribution):
    """Any finite positive value; drawn log-uniform, as 2**U[log2_lo, log2_hi]."""

    kind: Literal["log_uniform"] = "log_uniform"
    log2_lo: str | int | float
    log2_hi: str | int | float
    expr_fields = ("log2_lo", "log2_hi")

    def accepts_dtype(self, dtype):
        """2**log2_lo..2**log2_hi within the dtype's positive finite values
        (subnormals included), or float32's, where they are drawn, if it is wider."""
        import torch

        traits = DTYPE_TRAITS[dtype]
        if traits.is_integral or traits.decoding["kind"] not in (
            "cast",
            "power_of_two",
        ):
            return False
        limits = torch.finfo(torch_dtypes(dtype)[0])
        if limits.bits > 32:
            limits = torch.finfo(torch.float32)
        lo = self.constant_value("log2_lo")
        hi = self.constant_value("log2_hi")
        smallest = limits.tiny * limits.eps  # a subnormal where the dtype has them
        return (lo is None or lo >= math.log2(smallest)) and (hi is None or hi <= math.log2(limits.max))

    def sample(self, shape, dtype, generator):
        import torch

        decoding, storage_dtype = DTYPE_TRAITS[dtype].decoding, torch_dtypes(dtype)[0]
        lo, hi = self.log2_lo, self.log2_hi
        if decoding["kind"] == "power_of_two":
            if lo != int(lo) or hi != int(hi):
                raise ValueError(f"e8m0 exponents [{lo}, {hi}] are not integers")
            bias = decoding["exponent_bias"]
            codes = torch.randint(int(lo) + bias, int(hi) + bias + 1, shape, generator=generator)
            return codes.to(torch.uint8).view(storage_dtype)
        exponents = lo + (hi - lo) * torch.rand(shape, generator=generator)
        return torch.exp2(exponents).to(storage_dtype)

    def contains(self, tensor, dtype):
        decoded = to_float64(tensor, dtype)
        return bool((decoded.isfinite() & (decoded > 0)).all())


class IntRangeDistribution(Distribution):
    """Integers in [lo, hi]; with ``distinct``, distinct along the last dim.
    Drawn uniformly; distinct ones as the top indices of a row of N(0, 1) draws."""

    kind: Literal["int_range"] = "int_range"
    lo: str | int
    hi: str | int
    distinct: bool = False
    expr_fields = ("lo", "hi")

    def accepts_dtype(self, dtype):
        import torch

        if not DTYPE_TRAITS[dtype].is_integral:
            return False
        storage_dtype = torch_dtypes(dtype)[0]
        if storage_dtype == torch.bool:  # torch.iinfo rejects bool
            smallest, largest = 0, 1
        else:
            limits = torch.iinfo(storage_dtype)
            smallest, largest = limits.min, limits.max
        lo, hi = self.constant_value("lo"), self.constant_value("hi")
        return (lo is None or lo >= smallest) and (hi is None or hi <= largest)

    def sample(self, shape, dtype, generator):
        import torch

        storage_dtype = torch_dtypes(dtype)[0]
        span = self.hi - self.lo + 1
        if self.distinct:
            rows, n = math.prod(shape[:-1]), shape[-1]
            draws = torch.randn((rows, span), generator=generator)
            index = (draws.topk(n, dim=-1).indices + self.lo).reshape(shape)
        elif storage_dtype.itemsize == 1 and self.lo == 0 and 256 % span == 0:
            # a multi-GB byte pool: masked random bytes, not an int64 draw per value
            n = math.prod(shape)
            limits = torch.iinfo(torch.int64)
            words = torch.randint(limits.min, limits.max, (-(-n // 8),), generator=generator)
            codes = words.view(torch.uint8)[:n].bitwise_and_(span - 1)
            return codes.reshape(shape).view(storage_dtype)
        else:
            index = torch.randint(self.lo, self.hi + 1, shape, generator=generator)
        return index.to(storage_dtype)

    def contains(self, tensor, dtype):
        if not bool(((tensor >= self.lo) & (tensor <= self.hi)).all()):
            return False
        if not self.distinct:
            return True
        ordered = tensor.sort(dim=-1).values
        return bool((ordered[..., 1:] != ordered[..., :-1]).all())


# rows a step of CausalTopkDistribution.sample: a step scores [rows, positions]
_CAUSAL_TOPK_ROWS = 256


class CausalTopkDistribution(Distribution):
    """Sparse attention's top-k KV slots of a causal sequence's tokens, -1 for none: any
    distinct slots in [0, hi] and any number of -1s per row (a batch of sequences mixes
    rows). Drawn as one sequence of ``prefix`` cached tokens and the rows' tokens, its
    slots a run of the pool at a random offset: a row whose token sees no more positions
    than a row holds names them all, in order, then -1; any other row the top of its
    positions under a random permutation of them, in a random order. A seed's draws agree
    bit for bit: the keys are distinct and the shuffle sorts stably, where a GPU sort may
    break ties differently from run to run (and a kernel sums a row in its order)."""

    kind: Literal["causal_topk"] = "causal_topk"
    hi: str | int
    prefix: int = 0
    expr_fields = ("hi",)

    def accepts_dtype(self, dtype):
        import torch

        if not DTYPE_TRAITS[dtype].is_integral or torch_dtypes(dtype)[0] == torch.bool:
            return False
        limits = torch.iinfo(torch_dtypes(dtype)[0])
        hi = self.constant_value("hi")
        return limits.min <= -1 and (hi is None or hi <= limits.max)

    def sample(self, shape, dtype, generator):
        import torch

        *batch, rows, k = shape
        positions = self.prefix + rows
        if positions > self.hi + 1:
            raise ValueError(f"{positions} positions hold no run of slots in [0, {self.hi}]")
        base = int(torch.randint(self.hi + 2 - positions, (), generator=generator))
        keys = torch.randperm(positions, generator=generator)
        out = torch.full((math.prod(batch), rows, k), -1, dtype=torch.int64)
        for start in range(0, rows, _CAUSAL_TOPK_ROWS):
            seen = torch.arange(start, min(rows, start + _CAUSAL_TOPK_ROWS)) + self.prefix + 1
            width = min(int(seen[-1]), k)
            column = torch.arange(int(seen[-1]))
            scores = torch.where(column < seen[:, None], keys[: int(seen[-1])], -1)
            picked = scores.topk(width, dim=-1).indices
            order = torch.rand(picked.shape, generator=generator).argsort(dim=-1, stable=True)
            picked = picked.gather(-1, order)
            # a row that sees at most k positions: all of them, in order
            in_order = torch.where(column[:width] < seen[:, None], column[:width], -1)
            picked = torch.where((seen <= k)[:, None], in_order, picked)
            out[:, start : start + seen.numel(), :width] = torch.where(picked >= 0, picked + base, -1)
        return out.reshape(shape).to(torch_dtypes(dtype)[0])

    def contains(self, tensor, dtype):
        if not bool(((tensor >= -1) & (tensor <= self.hi)).all()):
            return False
        ordered = tensor.sort(dim=-1).values
        return bool(((ordered[..., 1:] != ordered[..., :-1]) | (ordered[..., 1:] < 0)).all())


class ExpertMaskDistribution(Distribution):
    """One expert-parallel rank's mask: 1 for ``local_experts`` experts, 0 elsewhere;
    drawn as one block at a random multiple of ``local_experts``, as sglang assigns."""

    kind: Literal["expert_mask"] = "expert_mask"
    local_experts: str | int
    expr_fields = ("local_experts",)

    def accepts_dtype(self, dtype):
        return DTYPE_TRAITS[dtype].is_integral

    def sample(self, shape, dtype, generator):
        import torch

        n = self.local_experts
        rank = int(torch.randint(shape[-1] // n, (), generator=generator))
        mask = torch.zeros(shape, dtype=torch_dtypes(dtype)[0])
        mask[..., rank * n : (rank + 1) * n] = 1
        return mask

    def contains(self, tensor, dtype):
        """Exactly ``local_experts`` ones, anywhere: more would index past them."""
        is_binary = bool(((tensor == 0) | (tensor == 1)).all())
        return is_binary and int(tensor.sum()) == self.local_experts


class CuSeqlensDistribution(Distribution):
    """Varlen cumulative sequence lengths: 0..``total``, each sequence at least
    ``min_seqlen`` rows; the spare rows are split by weights exp(N(0, 1) clipped to +-2).
    """

    kind: Literal["cu_seqlens"] = "cu_seqlens"
    min_seqlen: str | int
    total: str | int
    # (lo, hi) order for op.py's check: min_seqlen <= total
    expr_fields = ("min_seqlen", "total")

    def accepts_dtype(self, dtype):
        return dtype == DType.INT32  # as the offsets kernels read

    def sample(self, shape, dtype, generator):
        import torch

        parts = shape[-1] - 1
        spare = self.total - parts * self.min_seqlen
        if spare < 0:
            raise ValueError(f"{self.total} rows hold no {parts} parts of {self.min_seqlen}")
        # float64 keeps the split exact past 2**24 rows; the last offset is pinned
        weights = torch.randn(parts, generator=generator, dtype=torch.float64)
        shares = weights.clamp(-2, 2).exp().cumsum(0)
        ends = (shares / shares[-1:] * spare).floor().long()
        ends[-1:] = spare
        ends += self.min_seqlen * torch.arange(1, parts + 1)
        offsets = torch.cat((ends.new_zeros(1), ends))
        return offsets.expand(shape).to(torch_dtypes(dtype)[0])

    def contains(self, tensor, dtype):
        steps = tensor.diff(dim=-1)
        return bool(
            (tensor[..., 0] == 0).all()
            & (tensor[..., -1] == self.total).all()
            & (steps >= self.min_seqlen).all()
        )


class BoundedCuSeqlensDistribution(Distribution):
    """Positive sequence partitions whose actual maximum is ``max_seqlen``, so that dispatch
    reads no device metadata for it."""

    kind: Literal["bounded_cu_seqlens"] = "bounded_cu_seqlens"
    total: str | int
    max_seqlen: str | int
    expr_fields = ("total", "max_seqlen")

    def accepts_dtype(self, dtype):
        return dtype == DType.INT32

    def sample(self, shape, dtype, generator):
        import torch

        parts = shape[-1] - 1
        total, maximum = self.total, self.max_seqlen
        if parts < 1 or maximum < 1:
            raise ValueError("invalid bounded sequence partition")
        lengths = [maximum]
        remaining = total - maximum
        count = parts - 1
        if not count <= remaining <= count * maximum:
            raise ValueError("sequence bounds cannot partition the token count")
        for left in range(count, 0, -1):
            lo = max(1, remaining - (left - 1) * maximum)
            hi = min(maximum, remaining - left + 1)
            length = int(torch.randint(lo, hi + 1, (), generator=generator))
            lengths.append(length)
            remaining -= length
        order = torch.randperm(parts, generator=generator).tolist()
        lengths = [lengths[i] for i in order]
        offsets = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0)
        return offsets.expand(shape).to(torch_dtypes(dtype)[0])

    def contains(self, tensor, dtype):
        steps = tensor.diff(dim=-1)
        return bool(
            (tensor[..., 0] == 0).all()
            & (tensor[..., -1] == self.total).all()
            & (steps >= 1).all()
            & (steps <= self.max_seqlen).all()
            & (steps.max(dim=-1).values == self.max_seqlen).all()
        )


class LecunNormalDistribution(Distribution):
    """Any finite value; drawn as float32 LeCun normal N(0, 1 / fan_in), untruncated
    (unlike JAX's), then converted."""

    kind: Literal["lecun_normal"] = "lecun_normal"
    fan_in: str | int
    expr_fields = ("fan_in",)

    def accepts_dtype(self, dtype):
        traits = DTYPE_TRAITS[dtype]
        return not traits.is_integral and traits.decoding["kind"] == "cast"

    def sample(self, shape, dtype, generator):
        import torch

        draws = torch.randn(shape, generator=generator) / math.sqrt(self.fan_in)
        return draws.to(torch_dtypes(dtype)[0])

    # every decoded value is finite; fan_in does not bound it
    contains = FiniteDistribution.contains


class PermutationDistribution(Distribution):
    """Any finite value; drawn as a permutation of 0..n-1 divided by n per last-dim row
    (no ties; exact in float32 up to n = 2**24)."""

    kind: Literal["permutation"] = "permutation"

    def accepts_dtype(self, dtype):
        return dtype == DType.FLOAT32

    def sample(self, shape, dtype, generator):
        import torch

        n = shape[-1]
        draws = torch.rand((math.prod(shape[:-1]), n), generator=generator)
        permutations = draws.argsort(dim=-1).reshape(shape)
        return (permutations.double() / n).to(torch_dtypes(dtype)[0])

    # every decoded value is finite; rows need not be permutations
    contains = FiniteDistribution.contains


class TopkStaticPlanDistribution(Distribution):
    """sglang ``plan_topk_v2`` at threshold 2**31 - 1, for any lengths: zeros but the
    first element, ``2**31 - 1``."""

    kind: Literal["topk_static_plan"] = "topk_static_plan"

    def accepts_dtype(self, dtype):
        return dtype == DType.INT32

    def sample(self, shape, dtype, generator):
        import torch

        plan = torch.zeros(shape, dtype=torch_dtypes(dtype)[0])
        plan.view(-1)[:1] = 2**31 - 1
        return plan

    def contains(self, tensor, dtype):
        plan = self.sample(tensor.shape, dtype, None)
        return bool(tensor.equal(plan.to(tensor.device)))


# A DeepSeek-V4 fp8 kv cache token, in bytes: payload, codes, scale slot, exponents
_PAYLOAD_BYTES, _CODES, _SCALE_BYTES, _EXPONENTS = 576, 448, 8, 7


def _kv_cache_pages(shape: Sequence[int]) -> tuple[int, int]:
    if tuple(shape[-2:]) != (1, _PAYLOAD_BYTES + _SCALE_BYTES):
        raise ValueError(f"a kv cache of shape {tuple(shape)} is not (..., 1, 584)")
    return math.prod(shape[:-3]), shape[-3]


# Payload pages drawn per pool, then tiled: a prime, so no power-of-two page offset
# lands on a copy, and every slot's scale bytes are drawn afresh anyway.
_DISTINCT_PAGES = 251


class Deepseekv4Fp8KvCacheDistribution(Distribution):
    """Pages of a DeepSeek-V4 fp8 kv cache whose decoded values are all normal. The
    codes and rope values of ``_DISTINCT_PAGES`` pages repeat across the pool; each
    slot's exponent bytes are drawn on their own."""

    kind: Literal["deepseekv4_fp8_kv_cache"] = "deepseekv4_fp8_kv_cache"

    def accepts_dtype(self, dtype):
        return dtype in (DType.UINT8, DType.FLOAT8_E4M3FN)

    def sample(self, shape, dtype, generator):
        import torch

        pages, page_size = _kv_cache_pages(shape)
        distinct = min(pages, _DISTINCT_PAGES)
        values = torch.randn((distinct, page_size, 512), generator=generator)
        codes = values[..., :_CODES].clamp(-448, 448).to(torch.float8_e4m3fn)
        rope = values[..., _CODES:].to(torch.bfloat16)
        payloads = torch.cat((codes.view(torch.uint8), rope.view(torch.uint8)), dim=-1)
        page_bytes = page_size * (_PAYLOAD_BYTES + _SCALE_BYTES)
        payload_bytes = page_size * _PAYLOAD_BYTES
        raw = torch.empty((pages, page_bytes), dtype=torch.uint8)
        tile = payloads.flatten(1)
        for start in range(0, pages, distinct):
            stop = min(start + distinct, pages)
            raw[start:stop, :payload_bytes] = tile[: stop - start]
        slots = raw[:, payload_bytes:].view(pages, page_size, _SCALE_BYTES)
        slots[..., :_EXPONENTS] = torch.randint(
            124,
            128,
            (pages, page_size, _EXPONENTS),
            generator=generator,
            dtype=torch.uint8,
        )
        slots[..., _EXPONENTS:] = 0
        return raw.reshape(shape).view(torch_dtypes(dtype)[0])

    def contains(self, tensor, dtype):
        """Exponent bytes in [8, 246] (any code then assembles a normal bfloat16) and
        finite rope values."""
        import torch

        pages, page_size = _kv_cache_pages(tensor.shape)
        raw = tensor.view(torch.uint8).reshape(pages, -1)
        payloads = raw[:, : page_size * _PAYLOAD_BYTES].reshape(pages, page_size, -1)
        slots = raw[:, page_size * _PAYLOAD_BYTES :].reshape(pages, page_size, -1)
        exponents = slots[..., :_EXPONENTS]
        rope = payloads[..., _CODES:].contiguous().view(torch.bfloat16)
        is_normal = ((exponents >= 8) & (exponents <= 246)).all()
        return bool(is_normal & rope.isfinite().all())


# sglang's DSA index-K cache on aiter's preshuffled kernels: a page of 64 keys, their 128 fp8
# values in 16 x 16 tiles (16 keys by 16 dims), then their fp32 scales
_INDEX_PAGE, _INDEX_DIM = 64, 128


def _index_k_pages(shape: Sequence[int]) -> int:
    if len(shape) != 2 or shape[1] != _INDEX_PAGE * (_INDEX_DIM + 4):
        raise ValueError(f"an index-K cache of shape {tuple(shape)} is not (pages, 8448)")
    return shape[0]


class DsaIndexKCacheDistribution(Distribution):
    """Pages of sglang's DSA index-K cache as aiter's preshuffled kernels hold them, values
    finite and each key's scale a power of two, as the indexer's ue8m0 quant makes them.
    """

    kind: Literal["dsa_index_k_cache"] = "dsa_index_k_cache"

    def accepts_dtype(self, dtype):
        return dtype in (DType.UINT8, DType.FLOAT8_E4M3FN)

    def sample(self, shape, dtype, generator):
        import torch

        pages = _index_k_pages(shape)
        values = torch.randn((pages, _INDEX_PAGE, _INDEX_DIM), generator=generator) * 64
        codes = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        # [page, key // 16, dim // 16, key % 16, dim % 16]
        tiles = codes.reshape(pages, 4, 16, 8, 16).permute(0, 1, 3, 2, 4)
        exponents = torch.randint(117, 128, (pages, _INDEX_PAGE), generator=generator, dtype=torch.int32)
        scales = (exponents << 23).view(torch.uint8)
        raw = torch.cat((tiles.reshape(pages, -1), scales), dim=-1)
        return raw.view(torch_dtypes(dtype)[0])

    def contains(self, tensor, dtype):
        """Scales positive powers of two, values fp8 codes other than NaN."""
        import torch

        _index_k_pages(tensor.shape)
        raw = tensor.view(torch.uint8)
        codes = raw[:, : _INDEX_PAGE * _INDEX_DIM]
        bits = raw[:, _INDEX_PAGE * _INDEX_DIM :].contiguous().view(torch.int32)
        exponents = bits >> 23
        is_power = (bits & 0x7FFFFF == 0) & (exponents > 0) & (exponents < 255)
        return bool(is_power.all() & ((codes & 0x7F) != 0x7F).all())


class Mxfp8NopeRowsDistribution(Distribution):
    """448 E4M3FN codes, fourteen E8M0 bytes, then padding to 512 bytes.

    A row is the NoPE operand of DeepSeek-V4's split-precision CSR prefill.
    Payload patterns repeat over large pools; each row has independent scales.
    """

    kind: Literal["mxfp8_nope_rows"] = "mxfp8_nope_rows"

    def accepts_dtype(self, dtype):
        return dtype == DType.UINT8

    def sample(self, shape, dtype, generator):
        import torch

        if shape[-1] != 512:
            raise ValueError("an MXFP8 NoPE row occupies 512 bytes")
        rows = math.prod(shape[:-1])
        distinct = min(rows, _DISTINCT_PAGES)
        codes = torch.randn((distinct, 448), generator=generator).to(torch.float8_e4m3fn)
        raw = torch.zeros((rows, 512), dtype=torch.uint8)
        for start in range(0, rows, distinct):
            count = min(distinct, rows - start)
            raw[start : start + count, :448] = codes[:count].view(torch.uint8)
        raw[:, 448:462] = torch.randint(124, 128, (rows, 14), generator=generator, dtype=torch.uint8)
        return raw.reshape(shape)

    def contains(self, tensor, dtype):
        if tensor.shape[-1] != 512:
            return False
        raw = tensor.reshape(-1, 512)
        codes, exponents = raw[:, :448], raw[:, 448:462]
        return bool(((codes & 0x7F) != 0x7F).all() & ((exponents >= 8) & (exponents <= 246)).all())


DistributionUnion = Annotated[
    FiniteDistribution
    | LogUniformDistribution
    | IntRangeDistribution
    | CausalTopkDistribution
    | ExpertMaskDistribution
    | CuSeqlensDistribution
    | BoundedCuSeqlensDistribution
    | LecunNormalDistribution
    | PermutationDistribution
    | TopkStaticPlanDistribution
    | Deepseekv4Fp8KvCacheDistribution
    | Mxfp8NopeRowsDistribution
    | DsaIndexKCacheDistribution,
    Field(discriminator="kind"),
]
_DISTRIBUTION_ADAPTER = TypeAdapter(DistributionUnion)


def contains(distribution: Mapping[str, Any], tensor: Tensor, dtype: str) -> bool:
    """Whether every value of ``tensor`` (row-major layout) lies in ``distribution``."""
    return _DISTRIBUTION_ADAPTER.validate_python(distribution).contains(tensor, dtype)
